"""Index building, the four retrieval variants, and the evaluation run.

Variants (same chunks, same gold set):

* ``bm25``           BM25 only
* ``dense``          dense vectors only (cosine)
* ``hybrid``         RRF over the BM25 and dense lists
* ``hybrid_rerank``  RRF, then the top-N graded by the LLM reranker

The evaluation computes every retriever once per question and derives the
variants from the same lists, so the differences between variants are not
noise from separate runs. All LLM work (reranking, then generation) uses one
model, so only one model is in memory at a time.
"""

from __future__ import annotations

import json
import os
import platform
import statistics
import time
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

import requests

from .bm25 import BM25
from .chunking import Chunk, chunk_corpus, gold_chunk_ids
from .dense import LSAEmbedder, VectorIndex, build_embedder
from .fusion import rrf
from .generate import generate_answer
from .metrics import (bootstrap_ci, fact_present, mean, paired_bootstrap, retrieval_row, unsupported_figures)
from .rerank import LLMReranker
from .synth import write_dataset
from .text import is_idk, strip_citations
from .verify import verify

VARIANTS = ("bm25", "dense", "hybrid", "hybrid_rerank")
RETRIEVAL_METRICS = ("recall@1", "recall@3", "recall@5", "mrr@10", "ndcg@10", "contam@1", "contam@5")
COMPARISONS = (("bm25", "dense"), ("bm25", "hybrid"), ("dense", "hybrid"), ("hybrid", "hybrid_rerank"),
               ("bm25", "hybrid_rerank"))


# ------------------------------------------------------------------ io


def load_config(path: str | Path) -> dict:
    with open(path, "rb") as fh:
        return tomllib.load(fh)


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_jsonl(path: str | Path, rows) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_json(path: str | Path, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2)
        fh.write("\n")


def r4(x):
    return None if x is None else round(float(x), 4)


# ------------------------------------------------------------------ synth / index


def step_synth(cfg: dict, log=print) -> dict:
    info = write_dataset(cfg["run"]["data_dir"], int(cfg["run"]["seed"]))
    log(f"synth: {info['documents']} documents, {info['questions']} questions {info['by_type']} -> {cfg['run']['data_dir']}/")
    return info


def step_index(cfg: dict, log=print) -> dict:
    docs = read_jsonl(Path(cfg["run"]["data_dir"]) / "corpus.jsonl")
    chunks = chunk_corpus(docs, int(cfg["chunking"]["max_words"]), bool(cfg["chunking"].get("title_header", True)))
    emb = build_embedder(cfg["embedder"])
    texts = [c.text for c in chunks]
    t0 = time.perf_counter()
    emb.fit(texts)
    vecs = emb.embed(texts, kind="passage")
    embed_s = time.perf_counter() - t0
    out = Path(cfg["run"]["index_dir"])
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "chunks.jsonl", (c.to_dict() for c in chunks))
    VectorIndex(vecs).save(out / "vectors.npy")
    if isinstance(emb, LSAEmbedder):
        emb.save(out / "embedder")
    meta = {"embedder": emb.name, "embedder_cfg": cfg["embedder"], "chunking": cfg["chunking"],
            "documents": len(docs), "chunks": len(chunks),
            "dims": int(vecs.shape[1]), "embed_s": round(embed_s, 3),
            "words_per_chunk_median": statistics.median(len(c.text.split()) for c in chunks)}
    write_json(out / "meta.json", meta)
    log(f"index: {len(docs)} docs -> {len(chunks)} chunks, {meta['embedder']} vectors {vecs.shape} "
        f"in {embed_s:.2f}s -> {out}/")
    return meta


# ------------------------------------------------------------------ retrieval


@dataclass
class Retrieved:
    ranked: dict[str, list[str]]                 # variant -> ranked chunk ids
    timings_ms: dict[str, float]                 # stage -> milliseconds
    rerank: list[dict] = field(default_factory=list)


class Retriever:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        idx = Path(cfg["run"]["index_dir"])
        if not (idx / "chunks.jsonl").exists():
            raise FileNotFoundError(f"no index in {idx}/; run `python -m hybridrag index` first")
        self.meta = json.loads((idx / "meta.json").read_text())
        self.chunks = [Chunk(**c) for c in read_jsonl(idx / "chunks.jsonl")]
        self.by_id = {c.chunk_id: c for c in self.chunks}
        self.ids = [c.chunk_id for c in self.chunks]
        self.bm25 = BM25().fit([c.text for c in self.chunks])
        ecfg = self.meta["embedder_cfg"]
        if ecfg.get("kind", "lsa") == "lsa":
            self.embedder = LSAEmbedder.load(idx / "embedder", int(ecfg.get("dims", 128)))
        else:
            self.embedder = build_embedder(ecfg)
        self.vindex = VectorIndex.load(idx / "vectors.npy")
        r = cfg["retrieval"]
        self.n_cand, self.rrf_k = int(r["candidates"]), int(r["rrf_k"])
        self.top_n, self.final_k = int(r["rerank_top_n"]), int(r["final_k"])

    def chunks_by_doc(self) -> dict[str, list[Chunk]]:
        out: dict[str, list[Chunk]] = {}
        for c in self.chunks:
            out.setdefault(c.doc_id, []).append(c)
        return out

    def first_stage(self, question: str) -> Retrieved:
        t = {}
        t0 = time.perf_counter()
        bm = [self.ids[i] for i, _ in self.bm25.search(question, self.n_cand)]
        t["bm25"] = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        qv = self.embedder.embed([question], kind="query")[0]
        t["dense_embed"] = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        de = [self.ids[i] for i, _ in self.vindex.search(qv, self.n_cand)]
        t["dense_search"] = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        fused = [cid for cid, _ in rrf([bm, de], k=self.rrf_k)]
        t["rrf"] = (time.perf_counter() - t0) * 1000
        return Retrieved({"bm25": bm, "dense": de, "hybrid": fused}, t)

    def add_rerank(self, question: str, res: Retrieved, reranker: LLMReranker) -> Retrieved:
        fused = res.ranked["hybrid"]
        head = fused[:self.top_n]
        t0 = time.perf_counter()
        graded, llm_s = reranker.rerank(question, [(cid, self.by_id[cid].text) for cid in head])
        res.timings_ms["rerank_wall"] = (time.perf_counter() - t0) * 1000
        res.timings_ms["rerank_llm"] = llm_s * 1000
        res.ranked["hybrid_rerank"] = [g.chunk_id for g in graded] + fused[self.top_n:]
        res.rerank = [{"chunk_id": g.chunk_id, "grade": g.grade, "fused_rank": g.fused_rank, "cached": g.cached}
                      for g in graded]
        return res

    def retrieve(self, question: str, variant: str, reranker: LLMReranker | None = None) -> Retrieved:
        res = self.first_stage(question)
        if variant == "hybrid_rerank":
            if reranker is None:
                raise ValueError("hybrid_rerank needs an LLM reranker")
            self.add_rerank(question, res, reranker)
        return res

    def passages(self, ranked: list[str]) -> list[tuple[str, str]]:
        return [(cid, self.by_id[cid].text) for cid in ranked[:self.final_k]]


# ------------------------------------------------------------------ evaluation


def _gold(questions: list[dict], retr: Retriever) -> dict[str, set[str]]:
    by_doc = retr.chunks_by_doc()
    return {q["qid"]: set(gold_chunk_ids(q, by_doc)) for q in questions}


def retrieval_metrics(per_q: list[dict], questions: list[dict], gold: dict[str, set[str]]) -> dict:
    qmap = {q["qid"]: q for q in questions}
    rows: dict[str, list[dict]] = {v: [] for v in VARIANTS}
    for rec in per_q:
        q = qmap[rec["qid"]]
        if q["qtype"] == "unanswerable":
            continue
        for v in VARIANTS:
            if v in rec["ranked"]:
                row = retrieval_row(rec["ranked"][v], gold[q["qid"]], set(q["distractor_doc_ids"]))
                row["qid"], row["qtype"] = q["qid"], q["qtype"]
                rows[v].append(row)
    summary: dict = {"variants": {}, "comparisons": {}, "by_qtype": {}}
    for v, rs in rows.items():
        if not rs:
            continue
        s = {"n": len(rs), "n_with_distractors": sum(1 for r in rs if "contam@1" in r)}
        for m in RETRIEVAL_METRICS:
            vals = [r[m] for r in rs if m in r]
            s[m] = r4(mean(vals))
            s[m + "_ci95"] = [r4(x) for x in bootstrap_ci(vals)]
        summary["variants"][v] = s
        by_t: dict[str, dict] = {}
        for t in ("exact_code", "lexical", "paraphrase"):
            sub = [r for r in rs if r["qtype"] == t]
            if sub:
                by_t[t] = {"n": len(sub), "recall@5": r4(mean([r["recall@5"] for r in sub])),
                           "mrr@10": r4(mean([r["mrr@10"] for r in sub])),
                           "contam@5": r4(mean([r["contam@5"] for r in sub if "contam@5" in r]))}
        summary["by_qtype"][v] = by_t
    for a, b in COMPARISONS:
        if a in summary["variants"] and b in summary["variants"]:
            ra = {r["qid"]: r for r in rows[a]}
            rb = {r["qid"]: r for r in rows[b]}
            comp = {}
            for m in RETRIEVAL_METRICS:
                qids = [q for q in ra if m in ra[q]]
                comp[m] = {k: (r4(v) if isinstance(v, float) else ([r4(x) for x in v] if isinstance(v, list) else v))
                           for k, v in paired_bootstrap([ra[q][m] for q in qids], [rb[q][m] for q in qids]).items()}
            summary["comparisons"][f"{b}_vs_{a}"] = comp
    return summary


def answer_record(q: dict, variant: str, passages: list[tuple[str, str]], ans, vcfg: dict,
                  doc_text: dict[str, str], gold: set[str] | None = None) -> dict:
    t0 = time.perf_counter()
    ver = verify(ans.text, dict(passages), remove=True, min_claim_words=int(vcfg["min_claim_words"]),
                 min_overlap=float(vcfg["min_overlap"]), question=q["question"])
    verify_ms = (time.perf_counter() - t0) * 1000
    allowed = q["question"] + "\n" + (doc_text.get(q["gold_doc_id"], "") if q["gold_doc_id"] else "")
    raw, final = ans.text, ver.final_text
    rec = {
        "qid": q["qid"], "qtype": q["qtype"], "variant": variant, "question": q["question"],
        "gold_doc_id": q["gold_doc_id"], "gold_fact": q["gold_fact"], "context_ids": ans.context_ids,
        "answer_raw": raw, "answer_verified": final, "caveat": ver.caveat,
        "idk_raw": is_idk(raw), "idk_verified": is_idk(final),
        "claims": ver.claims, "flagged": ver.flagged,
        "flags": [{"sentence": s.text, "reasons": s.reasons, "content_in_context": s.in_context}
                  for s in ver.sentences if not s.supported],
        "unsupported_figures_raw": unsupported_figures(raw, allowed),
        "unsupported_figures_verified": unsupported_figures(final, allowed),
        "generate_s": round(ans.latency_s, 3), "generate_cached": ans.cached, "verify_ms": round(verify_ms, 3),
    }
    if q["gold_fact"] is not None:
        rec["fact_raw"] = fact_present(q["gold_fact"], raw)
        rec["fact_verified"] = fact_present(q["gold_fact"], final)
        rec["gold_in_context"] = bool(set(ans.context_ids) & (gold or set()))
    return rec


def answer_metrics(recs: list[dict]) -> dict:
    ans = [r for r in recs if r["qtype"] != "unanswerable"]
    una = [r for r in recs if r["qtype"] == "unanswerable"]
    claims = sum(r["claims"] for r in recs)
    flagged = sum(r["flagged"] for r in recs)

    def rate(rows, key):
        return r4(mean([1.0 if r[key] else 0.0 for r in rows]))

    def ci(rows, key):
        return [r4(x) for x in bootstrap_ci([1.0 if r[key] else 0.0 for r in rows])]

    out = {
        "n_answerable": len(ans), "n_unanswerable": len(una),
        "fact_accuracy_raw": rate(ans, "fact_raw"), "fact_accuracy_raw_ci95": ci(ans, "fact_raw"),
        "fact_accuracy_verified": rate(ans, "fact_verified"), "fact_accuracy_verified_ci95": ci(ans, "fact_verified"),
        "false_idk_raw": rate(ans, "idk_raw"), "false_idk_verified": rate(ans, "idk_verified"),
        "idk_correct_raw": rate(una, "idk_raw"), "idk_correct_raw_ci95": ci(una, "idk_raw"),
        "idk_correct_verified": rate(una, "idk_verified"), "idk_correct_verified_ci95": ci(una, "idk_verified"),
        "claim_sentences": claims, "flagged_sentences": flagged,
        "verifier_flag_rate": r4(flagged / claims) if claims else None,
        # flagged sentences whose content is nowhere in the 5 passages vs found in a passage other than the cited one
        "flagged_content_not_in_context": sum(1 for r in recs for f in r["flags"] if f.get("content_in_context") is False),
        "flagged_citation_only": sum(1 for r in recs for f in r["flags"] if f.get("content_in_context")),
        "answers_with_flag": r4(mean([1.0 if r["flagged"] else 0.0 for r in recs])),
        "gold_unsupported_raw": r4(mean([1.0 if r["unsupported_figures_raw"] else 0.0 for r in recs])),
        "gold_unsupported_verified": r4(mean([1.0 if r["unsupported_figures_verified"] else 0.0 for r in recs])),
        # answerable questions answered with something other than "I don't know" and without the gold fact
        "confidently_wrong_verified": r4(mean([1.0 if (not r["idk_verified"] and not r["fact_verified"]) else 0.0
                                               for r in ans])),
        "gold_in_context": r4(mean([1.0 if r.get("gold_in_context") else 0.0 for r in ans])),
        "fact_accuracy_when_gold_in_context": r4(mean([1.0 if r["fact_verified"] else 0.0 for r in ans
                                                       if r.get("gold_in_context")])),
        "generate_s_median": r4(statistics.median(r["generate_s"] for r in recs)),
        "verify_ms_median": r4(statistics.median(r["verify_ms"] for r in recs)),
    }
    out["by_qtype"] = {t: {"n": len(sub), "fact_accuracy_verified": rate(sub, "fact_verified")}
                       for t in ("exact_code", "lexical", "paraphrase")
                       if (sub := [r for r in ans if r["qtype"] == t])}
    return out


def ollama_state(base_url: str) -> dict:
    try:
        v = requests.get(f"{base_url}/api/version", timeout=5).json().get("version")
        ps = [m["name"] for m in requests.get(f"{base_url}/api/ps", timeout=5).json().get("models", [])]
        return {"version": v, "loaded": ps}
    except (requests.RequestException, ValueError) as exc:
        return {"error": str(exc)}


def probe_ollama_embed(base_url: str, model: str) -> dict:
    """Ask Ollama for one embedding with ``model`` and record what happened."""
    from .dense import EmbedderError, OllamaEmbedder

    try:
        v = OllamaEmbedder(model, base_url).embed(["probe"])
        return {"model": model, "ok": True, "dims": int(v.shape[1])}
    except EmbedderError as exc:
        return {"model": model, "ok": False, "error": str(exc)}


def step_eval(cfg: dict, provider, log=print, answer_variants: list[str] | None = None,
              limit: int | None = None, rerank: bool = True) -> dict:
    """``rerank=False`` evaluates the first-stage variants only (no LLM call at all)."""
    res_dir = Path(cfg["run"]["results_dir"])
    res_dir.mkdir(parents=True, exist_ok=True)
    questions = read_jsonl(Path(cfg["run"]["data_dir"]) / "questions.jsonl")
    if limit:
        questions = questions[:limit]
    docs = read_jsonl(Path(cfg["run"]["data_dir"]) / "corpus.jsonl")
    doc_text = {d["doc_id"]: d["title"] + "\n" + d["text"] for d in docs}
    retr = Retriever(cfg)
    gold = _gold(questions, retr)
    model = cfg["llm"]["model"]
    reranker = LLMReranker(provider, model) if rerank else None
    variants_eval = VARIANTS if rerank else VARIANTS[:3]
    timing = {}

    # 1. retrieval, every variant from the same first-stage lists
    t0 = time.perf_counter()
    per_q, results = [], {}
    for i, q in enumerate(questions, 1):
        r = retr.first_stage(q["question"])
        if reranker is not None:
            retr.add_rerank(q["question"], r, reranker)
        results[q["qid"]] = r
        per_q.append({"qid": q["qid"], "qtype": q["qtype"], "gold_chunk_ids": sorted(gold[q["qid"]]),
                      "ranked": {v: r.ranked[v][:10] for v in variants_eval}, "rerank": r.rerank,
                      "gold_rank_fused": next((i for i, c in enumerate(r.ranked["hybrid"], 1) if c in gold[q["qid"]]), None),
                      "timings_ms": {k: round(v, 3) for k, v in r.timings_ms.items()}})
        if i % 20 == 0 or i == len(questions):
            log(f"  retrieval{' + rerank' if reranker else ''}: {i}/{len(questions)} ({time.perf_counter() - t0:.0f}s)")
    timing["retrieval_and_rerank_s"] = round(time.perf_counter() - t0, 1)
    write_jsonl(res_dir / "retrieval_per_question.jsonl", per_q)
    rmetrics = retrieval_metrics(per_q, questions, gold)
    rmetrics["reranker_unparseable"] = reranker.unparseable if reranker else None
    # the reranker only reorders the fused top N: a gold chunk below N is out of its reach
    ceiling = {}
    for t in ("all", "exact_code", "lexical", "paraphrase"):
        sub = [p for p in per_q if p["qtype"] != "unanswerable" and (t == "all" or p["qtype"] == t)]
        if sub:
            ceiling[t] = r4(mean([1.0 if p["gold_rank_fused"] and p["gold_rank_fused"] <= retr.top_n else 0.0
                                  for p in sub]))
    rmetrics[f"gold_in_fused_top{retr.top_n}"] = ceiling
    rmetrics["index"] = retr.meta
    write_json(res_dir / "retrieval_metrics.json", rmetrics)
    log("retrieval: " + "; ".join(f"{v} R@5={s['recall@5']} MRR={s['mrr@10']} contam@5={s['contam@5']}"
                                  for v, s in rmetrics["variants"].items()))

    # 2. answers + verifier for the configured variants
    t0 = time.perf_counter()
    variants = answer_variants if answer_variants is not None else list(cfg["answer"]["variants"])
    variants = [v for v in variants if v in variants_eval] if provider is not None else []
    ametrics, all_recs = {}, {}
    for v in variants:
        recs = []
        t1 = time.perf_counter()
        for i, q in enumerate(questions, 1):
            passages = retr.passages(results[q["qid"]].ranked[v])
            ans = generate_answer(provider, model, q["question"], passages, int(cfg["answer"]["max_tokens"]))
            recs.append(answer_record(q, v, passages, ans, cfg["verifier"], doc_text, gold[q["qid"]]))
            if i % 45 == 0 or i == len(questions):
                log(f"  answers {v}: {i}/{len(questions)} ({time.perf_counter() - t1:.0f}s)")
        write_jsonl(res_dir / f"answers_{v}.jsonl", recs)
        ametrics[v] = answer_metrics(recs)
        all_recs[v] = recs
        m = ametrics[v]
        log(f"answers {v}: fact acc raw={m['fact_accuracy_raw']} verified={m['fact_accuracy_verified']} "
            f"idk={m['idk_correct_verified']} gold-unsupported {m['gold_unsupported_raw']} -> {m['gold_unsupported_verified']}")
    timing["answers_s"] = round(time.perf_counter() - t0, 1)
    comps = {}
    for a, b in (("bm25", "hybrid_rerank"), ("hybrid", "hybrid_rerank"), ("dense", "hybrid_rerank"),
                 ("bm25", "hybrid")):
        if a in all_recs and b in all_recs:
            ans_a = [r for r in all_recs[a] if r["qtype"] != "unanswerable"]
            ans_b = [r for r in all_recs[b] if r["qtype"] != "unanswerable"]
            d = paired_bootstrap([float(r["fact_verified"]) for r in ans_a], [float(r["fact_verified"]) for r in ans_b])
            comps[f"{b}_vs_{a}"] = {"fact_accuracy_verified": {"delta": r4(d["delta"]), "ci95": [r4(x) for x in d["ci95"]],
                                                               "excludes_zero": d["excludes_zero"]}}
    if ametrics:
        write_json(res_dir / "answer_metrics.json", {"variants": ametrics, "comparisons": comps})

    # 3. latency per stage (medians over questions; LLM latencies are the uncached measurements)
    def med(key):
        return r4(statistics.median(p["timings_ms"][key] for p in per_q))

    lat = {"bm25_ms": med("bm25"), "dense_query_embed_ms": med("dense_embed"), "dense_search_ms": med("dense_search"),
           "rrf_ms": med("rrf")}
    if reranker is not None:
        lat.update({"rerank_llm_ms_per_question": med("rerank_llm"), "rerank_pairs_per_question": retr.top_n,
                    "rerank_llm_ms_per_pair": r4(statistics.median(p["timings_ms"]["rerank_llm"] / max(len(p["rerank"]), 1)
                                                                  for p in per_q))})
    for v, m in ametrics.items():
        lat[f"generate_ms_{v}"] = r4(m["generate_s_median"] * 1000)
        lat[f"verify_ms_{v}"] = m["verify_ms_median"]
    write_json(res_dir / "latency.json", lat)
    return {"retrieval": rmetrics, "answers": ametrics, "latency": lat, "timing": timing}


# ------------------------------------------------------------------ run-all


def run_all(cfg: dict, provider, log=print, skip_synth: bool = False) -> dict:
    res_dir = Path(cfg["run"]["results_dir"])
    meta = {"started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "python": platform.python_version(),
            "platform": platform.platform(), "machine": platform.machine(), "config": cfg}
    base = cfg["llm"].get("base_url", "http://localhost:11434")
    t_all = time.perf_counter()
    steps = {}
    if cfg["llm"].get("provider", "ollama") == "ollama":
        meta["ollama_before"] = ollama_state(base)
        meta["ollama_embed_probe"] = probe_ollama_embed(base, cfg["llm"]["model"])
        log(f"ollama: {meta['ollama_before']}; embed probe: {meta['ollama_embed_probe']}")
    t0 = time.perf_counter()
    if not skip_synth:
        steps["synth"] = step_synth(cfg, log)
    steps["synth_s"] = round(time.perf_counter() - t0, 2)
    t0 = time.perf_counter()
    steps["index"] = step_index(cfg, log)
    steps["index_s"] = round(time.perf_counter() - t0, 2)
    t0 = time.perf_counter()
    out = step_eval(cfg, provider, log)
    steps["eval_s"] = round(time.perf_counter() - t0, 1)
    steps["eval_timing"] = out["timing"]
    evalgen_path = cfg.get("evalgen", {}).get("path") or os.environ.get("HYBRIDRAG_EVALGEN_PATH", "")
    if evalgen_path:
        from .evalgen_bridge import run_evalgen

        t0 = time.perf_counter()
        steps["evalgen"] = run_evalgen(cfg, evalgen_path, log)
        steps["evalgen_s"] = round(time.perf_counter() - t0, 1)
    if hasattr(provider, "hits"):
        meta["llm_cache"] = {"hits": provider.hits, "misses": provider.misses}
    provider.release(cfg["llm"]["model"])
    if cfg["llm"].get("provider", "ollama") == "ollama":
        meta["ollama_after"] = ollama_state(base)
    meta["steps"] = steps
    meta["wall_s"] = round(time.perf_counter() - t_all, 1)
    write_json(res_dir / "run_meta.json", meta)
    from .report import write_report

    write_report(cfg)
    log(f"run-all: done in {meta['wall_s']}s; see {res_dir}/report.md")
    return meta
