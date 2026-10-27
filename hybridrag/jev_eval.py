"""The Jev (TypeSafe) reranking experiment: can Jev do (part of) the LLM reranker's job?

Same questions, same first-stage lists, same LLM grades as the reference run.
The LLM reranker is replayed from the sqlite3 cache (a miss is an error, so the
``hybrid_rerank`` row is the reference row, not a new run); Jev scores the same
fused top N once per question. Variants:

* ``hybrid_jev``      the fused top N sorted by Jev's noul (ties: fused rank)
* ``hybrid_jev_rrf``  RRF (same k) over the fused list and the Jev-sorted top N
* ``hybrid_jev_llm``  cascade: Jev keeps its top ``cascade_n``, the LLM grades
                      only those; the rest follow in Jev order

``cascade_n`` was fixed before any Jev score existed (see config.toml); the
sweep over other values is reported as sensitivity, not used to pick it.
Every cascade value replays LLM grades that the reference run already
measured, so its LLM latency is the sum of the per-pair latencies stored with
those calls, not a new wall-clock measurement.
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

from .fusion import rrf
from .generate import answer_messages, generate_answer
from .jev import BudgetExceeded, JevReranker, cost_usd, estimate_tokens
from .metrics import mean, paired_bootstrap, retrieval_row
from .pipeline import (Retriever, _gold, answer_metrics, answer_record, r4, read_jsonl, retrieval_metrics, write_json,
                       write_jsonl)
from .rerank import LLMReranker

JEV_VARIANTS = ("hybrid_jev", "hybrid_jev_rrf", "hybrid_jev_llm")
ALL_VARIANTS = ("bm25", "dense", "hybrid", "hybrid_rerank") + JEV_VARIANTS
COMPARISONS = (("hybrid", "hybrid_jev"), ("hybrid_rerank", "hybrid_jev"), ("hybrid", "hybrid_jev_rrf"),
               ("hybrid_jev", "hybrid_jev_rrf"), ("hybrid_rerank", "hybrid_jev_rrf"),
               ("hybrid", "hybrid_jev_llm"), ("hybrid_rerank", "hybrid_jev_llm"))


def cascade(question: str, jev_sorted: list[str], fused: list[str], top_n: int, n: int, texts: dict[str, str],
            reranker: LLMReranker) -> tuple[list[str], float, int]:
    """LLM-grade Jev's top n; returns (ranking, LLM seconds, pairs graded).

    The kept chunks go to the LLM in fused order, so grade ties break by fused rank exactly as in
    ``hybrid_rerank``: with n = top_n the cascade is the reference ranking, and for smaller n the
    only difference is which chunks the LLM got to see.
    """
    keep = set(jev_sorted[:n])
    head = [c for c in fused[:top_n] if c in keep]
    graded, llm_s = reranker.rerank(question, [(cid, texts[cid]) for cid in head])
    return [g.chunk_id for g in graded] + jev_sorted[n:] + fused[top_n:], llm_s, len(head)


def gold_rank(ranked: list[str], gold: set[str]) -> int | None:
    return next((i for i, c in enumerate(ranked, 1) if c in gold), None)


def step_jev_eval(cfg: dict, provider, jev: JevReranker, log=print, limit: int | None = None,
                  answers: bool = True, max_answer_minutes: float = 20.0) -> dict:
    jcfg = cfg["jev"]
    out_dir = Path(cfg["run"]["results_dir"]) / "jev"
    out_dir.mkdir(parents=True, exist_ok=True)
    questions = read_jsonl(Path(cfg["run"]["data_dir"]) / "questions.jsonl")
    if limit:
        questions = questions[:limit]
    retr = Retriever(cfg)
    gold = _gold(questions, retr)
    texts = {c.chunk_id: c.text for c in retr.chunks}
    model = cfg["llm"]["model"]
    reranker = LLMReranker(provider, model)
    n_casc = int(jcfg["cascade_n"])
    sweep = sorted({int(x) for x in jcfg.get("cascade_sweep", [])} | {n_casc})
    k = retr.rrf_k

    # 0. first stage for every question, and the budget check before any Jev call
    first = {q["qid"]: retr.first_stage(q["question"]) for q in questions}
    bodies = [jev.request(q["question"], [(c, texts[c]) for c in first[q["qid"]].ranked["hybrid"][:retr.top_n]])
              for q in questions]
    todo = [b for b in bodies if jev.client.cached(b) is None]
    est_tokens = sum(estimate_tokens(b) for b in todo)
    est_usd = cost_usd(est_tokens, jev.price)
    log(f"jev: {len(bodies)} requests ({len(bodies) - len(todo)} cached), estimated {est_tokens:,} input tokens "
        f"= US${est_usd:.4f} (cap US${jev.max_usd:.2f})")
    if jev.max_usd is not None and est_usd > jev.max_usd:
        raise BudgetExceeded(f"jev: estimated US${est_usd:.4f} is over the US${jev.max_usd:.2f} cap; nothing was sent")

    # 1. retrieval: LLM rerank (cache replay), Jev scores, the three Jev variants, the cascade sweep
    allow = getattr(provider, "allow_miss", None)
    if allow is not None:
        provider.allow_miss = False
    per_q, results, sweep_rows = [], {}, {n: [] for n in sweep}
    t0 = time.perf_counter()
    try:
        for i, q in enumerate(questions, 1):
            r = first[q["qid"]]
            retr.add_rerank(q["question"], r, reranker)
            fused = r.ranked["hybrid"]
            head = fused[:retr.top_n]
            scored, jres = jev.score(q["question"], [(c, texts[c]) for c in head])
            jev_sorted = [s.chunk_id for s in scored]
            r.ranked["hybrid_jev"] = jev_sorted + fused[retr.top_n:]
            r.ranked["hybrid_jev_rrf"] = [c for c, _ in rrf([fused, jev_sorted], k=k)]
            casc = {}
            for n in sweep:
                casc[n] = cascade(q["question"], jev_sorted, fused, retr.top_n, n, texts, reranker)
            r.ranked["hybrid_jev_llm"] = casc[n_casc][0]
            results[q["qid"]] = r
            g = gold[q["qid"]]
            for n in sweep:
                ranked, llm_s, pairs = casc[n]
                sweep_rows[n].append({"qid": q["qid"], "qtype": q["qtype"], "ranked": ranked[:10], "llm_ms": llm_s * 1000,
                                      "pairs": pairs, "gold_in_jev_topn": bool(set(jev_sorted[:n]) & g),
                                      "gold_in_fused_topn": bool(set(fused[:n]) & g)})
            per_q.append({
                "qid": q["qid"], "qtype": q["qtype"], "gold_chunk_ids": sorted(g),
                "ranked": {v: r.ranked[v][:10] for v in ALL_VARIANTS},
                "jev": [{"chunk_id": s.chunk_id, "noul": round(s.noul, 4), "fused_rank": s.fused_rank} for s in scored],
                "jev_model": jres.model, "jev_input_tokens": jres.input_tokens, "jev_output_tokens": jres.output_tokens,
                "jev_cached": jres.cached,
                "gold_rank_fused": gold_rank(fused, g), "gold_rank_jev": gold_rank(jev_sorted, g),
                "timings_ms": {"jev_request": round(jres.latency_s * 1000, 3),
                               "rerank_llm_20": round(r.timings_ms["rerank_llm"], 3),
                               "cascade_llm": round(casc[n_casc][1] * 1000, 3),
                               "cascade_total": round(jres.latency_s * 1000 + casc[n_casc][1] * 1000, 3)}})
            if i % 20 == 0 or i == len(questions):
                log(f"  jev retrieval: {i}/{len(questions)} ({time.perf_counter() - t0:.0f}s, "
                    f"US${cost_usd(jev.billed_input_tokens, jev.price):.4f} sent)")
    finally:
        if allow is not None:
            provider.allow_miss = allow
    write_jsonl(out_dir / "per_question.jsonl", per_q)
    rm = retrieval_metrics(per_q, questions, gold, variants=ALL_VARIANTS, comparisons=COMPARISONS)
    rm["jev_model"] = sorted(jev.models_seen)
    rm["cascade_n"] = n_casc
    write_json(out_dir / "retrieval_metrics.json", rm)
    log("jev retrieval: " + "; ".join(f"{v} R@1={rm['variants'][v]['recall@1']} nDCG={rm['variants'][v]['ndcg@10']}"
                                      for v in ("hybrid", "hybrid_rerank") + JEV_VARIANTS))

    # cascade sensitivity (answerable questions)
    qmap = {q["qid"]: q for q in questions}
    sw = []
    for n in sweep:
        rows = [x for x in sweep_rows[n] if x["qtype"] != "unanswerable"]
        mets = [retrieval_row(x["ranked"], gold[x["qid"]], set(qmap[x["qid"]]["distractor_doc_ids"])) for x in rows]
        sw.append({"n": n, "recall@1": r4(mean([m["recall@1"] for m in mets])),
                   "mrr@10": r4(mean([m["mrr@10"] for m in mets])), "ndcg@10": r4(mean([m["ndcg@10"] for m in mets])),
                   "contam@5": r4(mean([m["contam@5"] for m in mets if "contam@5" in m])),
                   "gold_in_jev_topn": r4(mean([1.0 if x["gold_in_jev_topn"] else 0.0 for x in rows])),
                   "gold_in_fused_topn": r4(mean([1.0 if x["gold_in_fused_topn"] else 0.0 for x in rows])),
                   "llm_ms_median": r4(statistics.median(x["llm_ms"] for x in sweep_rows[n]))})
    write_json(out_dir / "cascade_sweep.json", sw)

    # latency and cost
    def med(key):
        return r4(statistics.median(p["timings_ms"][key] for p in per_q))

    lat = {"jev_request_ms": med("jev_request"), "jev_request_ms_p90": r4(sorted(
        p["timings_ms"]["jev_request"] for p in per_q)[int(0.9 * (len(per_q) - 1))]),
        "rerank_llm_ms_20_pairs": med("rerank_llm_20"), "cascade_llm_ms": med("cascade_llm"),
        "cascade_total_ms": med("cascade_total"), "cascade_n": n_casc, "jev_candidates": retr.top_n,
        "jev_requests_uncached_this_run": sum(1 for p in per_q if not p["jev_cached"])}
    write_json(out_dir / "latency.json", lat)
    tok_in = sum(p["jev_input_tokens"] for p in per_q)
    tok_out = sum(p["jev_output_tokens"] for p in per_q)
    cost = {"price_usd_per_mtok_input": jev.price, "output_billed": False, "requests": len(per_q),
            "pairs_scored": len(per_q) * retr.top_n, "input_tokens": tok_in, "output_tokens": tok_out,
            "token_source": "usage.input_tokens / usage.output_tokens as reported by the API",
            "cost_usd": round(cost_usd(tok_in, jev.price), 6),
            "cost_usd_per_question": round(cost_usd(tok_in, jev.price) / max(len(per_q), 1), 8),
            "cost_usd_per_1000_questions": round(cost_usd(tok_in, jev.price) / max(len(per_q), 1) * 1000, 4),
            "estimated_input_tokens_before_run": est_tokens, "billed_this_process_usd": round(
                cost_usd(jev.billed_input_tokens, jev.price), 6), "cache_hits": jev.client.hits,
            "cache_misses": jev.client.misses, "retries": jev.client.retries, "models": sorted(jev.models_seen)}
    write_json(out_dir / "cost.json", cost)
    log(f"jev cost: {tok_in:,} input tokens = US${cost['cost_usd']:.4f}; latency median {lat['jev_request_ms']:.0f} ms "
        f"per request ({retr.top_n} nouls)")

    # 2. answers, only where the generation calls it needs are cheap enough
    am, skipped = {}, {}
    if answers and provider is not None:
        docs = read_jsonl(Path(cfg["run"]["data_dir"]) / "corpus.jsonl")
        doc_text = {d["doc_id"]: d["title"] + "\n" + d["text"] for d in docs}
        mt = int(cfg["answer"]["max_tokens"])
        ref_lat = Path(cfg["run"]["results_dir"]) / "latency.json"
        gen_s = (json.loads(ref_lat.read_text()).get("generate_ms_hybrid_rerank", 3400.0) / 1000) if ref_lat.exists() else 3.4
        all_recs = {}
        for v in ("hybrid", "hybrid_rerank") + JEV_VARIANTS:
            misses = 0
            if hasattr(provider, "has"):
                misses = sum(1 for q in questions if not provider.has(
                    model, answer_messages(q["question"], retr.passages(results[q["qid"]].ranked[v])), max_tokens=mt))
            est_min = misses * gen_s / 60
            if est_min > max_answer_minutes:
                skipped[v] = {"uncached_generations": misses, "estimated_minutes": round(est_min, 1)}
                log(f"answers {v}: skipped, {misses} uncached generations = about {est_min:.0f} min")
                continue
            log(f"answers {v}: {misses} uncached generations (about {est_min:.1f} min)")
            recs = []
            for q in questions:
                passages = retr.passages(results[q["qid"]].ranked[v])
                ans = generate_answer(provider, model, q["question"], passages, mt)
                recs.append(answer_record(q, v, passages, ans, cfg["verifier"], doc_text, gold[q["qid"]]))
            write_jsonl(out_dir / f"answers_{v}.jsonl", recs)
            am[v] = answer_metrics(recs)
            all_recs[v] = recs
            log(f"answers {v}: fact acc verified={am[v]['fact_accuracy_verified']}")
        comps = {}
        for a, b in (("hybrid", "hybrid_jev"), ("hybrid_rerank", "hybrid_jev"), ("hybrid_rerank", "hybrid_jev_rrf"),
                     ("hybrid_rerank", "hybrid_jev_llm"), ("hybrid", "hybrid_jev_llm")):
            if a in all_recs and b in all_recs:
                xa = [float(r["fact_verified"]) for r in all_recs[a] if r["qtype"] != "unanswerable"]
                xb = [float(r["fact_verified"]) for r in all_recs[b] if r["qtype"] != "unanswerable"]
                d = paired_bootstrap(xa, xb)
                comps[f"{b}_vs_{a}"] = {"fact_accuracy_verified": {"delta": r4(d["delta"]), "ci95": [r4(x) for x in d["ci95"]],
                                                                   "excludes_zero": d["excludes_zero"]}}
        write_json(out_dir / "answer_metrics.json", {"variants": am, "comparisons": comps, "skipped": skipped})
    return {"retrieval": rm, "sweep": sw, "latency": lat, "cost": cost, "answers": am, "answers_skipped": skipped}
