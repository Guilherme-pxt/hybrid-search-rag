"""results/report.md, rendered from the JSON artifacts of a run."""

from __future__ import annotations

import json
from pathlib import Path

LABEL = {"bm25": "BM25 only", "dense": "Dense only", "hybrid": "Hybrid (RRF)", "hybrid_rerank": "Hybrid (RRF) + LLM rerank",
         "hybrid_jev": "Hybrid (RRF) + Jev rerank", "hybrid_jev_rrf": "Hybrid (RRF) + Jev, fused by RRF",
         "hybrid_jev_llm": "Hybrid (RRF) + Jev prune + LLM rerank"}


def _f(x, pct=False):
    if x is None:
        return "n/a"
    return f"{x * 100:.1f}%" if pct else f"{x:.3f}"


def _ci(c, pct=False):
    return f"[{_f(c[0], pct)}, {_f(c[1], pct)}]" if c else ""


def retrieval_table(rm: dict) -> str:
    head = ("| variant | Recall@1 | Recall@3 | Recall@5 | MRR@10 | nDCG@10 | Contam@1 | Contam@5 |\n"
            "|---|---|---|---|---|---|---|---|\n")
    rows = []
    for v, s in rm["variants"].items():
        rows.append(f"| {LABEL[v]} | " + " | ".join(_f(s[m]) for m in
                    ("recall@1", "recall@3", "recall@5", "mrr@10", "ndcg@10", "contam@1", "contam@5")) + " |")
    return head + "\n".join(rows)


def ci_table(rm: dict) -> str:
    head = "| variant | Recall@5 (95% CI) | MRR@10 (95% CI) | Contam@5 (95% CI) |\n|---|---|---|---|\n"
    rows = [f"| {LABEL[v]} | {_f(s['recall@5'])} {_ci(s['recall@5_ci95'])} | {_f(s['mrr@10'])} {_ci(s['mrr@10_ci95'])} | "
            f"{_f(s['contam@5'])} {_ci(s['contam@5_ci95'])} |" for v, s in rm["variants"].items()]
    return head + "\n".join(rows)


def comparison_table(rm: dict, metrics=("recall@1", "recall@5", "mrr@10", "contam@5")) -> str:
    head = "| comparison | metric | delta | paired bootstrap 95% CI |\n|---|---|---|---|\n"
    rows = []
    for name, comp in rm["comparisons"].items():
        b, a = name.split("_vs_")
        for m in metrics:
            c = comp[m]
            mark = "" if c["excludes_zero"] else " (includes 0)"
            rows.append(f"| {LABEL[b]} vs {LABEL[a]} | {m} | {c['delta']:+.3f} | {_ci(c['ci95'])}{mark} |")
    return head + "\n".join(rows)


def qtype_table(rm: dict) -> str:
    types = ("exact_code", "lexical", "paraphrase")
    head = "| variant | " + " | ".join(f"{t} R@5 / MRR" for t in types) + " |\n|---|---|---|---|\n"
    rows = []
    for v, by in rm["by_qtype"].items():
        cells = [f"{_f(by[t]['recall@5'])} / {_f(by[t]['mrr@10'])}" if t in by else "n/a" for t in types]
        rows.append(f"| {LABEL[v]} | " + " | ".join(cells) + " |")
    n = {t: rm["by_qtype"][next(iter(rm["by_qtype"]))][t]["n"] for t in types if t in rm["by_qtype"][next(iter(rm["by_qtype"]))]}
    return head + "\n".join(rows) + "\n\nQuestions per type: " + ", ".join(f"{t} {c}" for t, c in n.items())


def answer_table(am: dict) -> str:
    head = ("| variant | exact-fact acc. (raw) | exact-fact acc. (verified) | correct \"I don't know\" (verified) | "
            "false \"I don't know\" | verifier-flagged claim sentences | answers with a figure not in the gold doc: "
            "before -> after verifier |\n|---|---|---|---|---|---|---|\n")
    rows = []
    for v, m in am["variants"].items():
        rows.append(f"| {LABEL[v]} | {_f(m['fact_accuracy_raw'], True)} | {_f(m['fact_accuracy_verified'], True)} "
                    f"{_ci(m['fact_accuracy_verified_ci95'], True)} | {_f(m['idk_correct_verified'], True)} "
                    f"{_ci(m['idk_correct_verified_ci95'], True)} | {_f(m['false_idk_verified'], True)} | "
                    f"{m['flagged_sentences']}/{m['claim_sentences']} ({_f(m['verifier_flag_rate'], True)}) | "
                    f"{_f(m['gold_unsupported_raw'], True)} -> {_f(m['gold_unsupported_verified'], True)} |")
    return head + "\n".join(rows)


def answer_error_table(am: dict) -> str:
    head = ("| variant | gold chunk in the 5 passages | fact acc. when it is | confidently wrong (answered, fact missing) | "
            "flagged sentences: figures/words found nowhere in the 5 passages / found in another passage |\n|---|---|---|---|---|\n")
    rows = [f"| {LABEL[v]} | {_f(m.get('gold_in_context'), True)} | {_f(m.get('fact_accuracy_when_gold_in_context'), True)} | "
            f"{_f(m.get('confidently_wrong_verified'), True)} | {m.get('flagged_content_not_in_context', 'n/a')} / "
            f"{m.get('flagged_citation_only', 'n/a')} |" for v, m in am["variants"].items()]
    return head + "\n".join(rows)


def latency_table(lat: dict) -> str:
    rows = [("BM25 search", lat["bm25_ms"], "ms"), ("Dense: embed query", lat["dense_query_embed_ms"], "ms"),
            ("Dense: cosine search", lat["dense_search_ms"], "ms"), ("RRF fusion", lat["rrf_ms"], "ms")]
    if "rerank_llm_ms_per_question" in lat:
        rows += [(f"LLM rerank ({lat['rerank_pairs_per_question']} pairs per question)", lat["rerank_llm_ms_per_question"], "ms"),
                 ("LLM rerank, per pair", lat["rerank_llm_ms_per_pair"], "ms")]
    for k, v in lat.items():
        if k.startswith("generate_ms_"):
            rows.append((f"Generation ({LABEL[k[len('generate_ms_'):]]} context)", v, "ms"))
    for k, v in lat.items():
        if k.startswith("verify_ms_hybrid_rerank"):
            rows.append(("Citation verifier", v, "ms"))
    return "| stage | median |\n|---|---|\n" + "\n".join(f"| {n} | {x:,.2f} {u} |" for n, x, u in rows)


def jev_section(jdir: Path) -> list[str]:
    rm = json.loads((jdir / "retrieval_metrics.json").read_text())
    n = rm.get("cascade_n")
    parts = ["## Jev (TypeSafe) reranking", "",
             f"Generated by `python -m hybridrag jev-eval` from `{jdir.name}/`. Jev model: "
             f"{', '.join(rm.get('jev_model') or ['n/a'])}. The cascade keeps Jev's top {n} for the LLM.", "",
             retrieval_table(rm), "", ci_table(rm), "", "### Paired comparisons", "",
             comparison_table(rm, ("recall@1", "recall@3", "recall@5", "mrr@10", "ndcg@10", "contam@5")), ""]
    if (jdir / "cascade_sweep.json").exists():
        parts += ["### Cascade: Jev keeps the top N, the LLM grades only those", "",
                  "| N | Recall@1 | MRR@10 | nDCG@10 | Contam@5 | gold in Jev top N | gold in fused top N | "
                  "LLM ms per question (median) |", "|---|---|---|---|---|---|---|---|"]
        for r in json.loads((jdir / "cascade_sweep.json").read_text()):
            parts.append(f"| {r['n']} | {_f(r['recall@1'])} | {_f(r['mrr@10'])} | {_f(r['ndcg@10'])} | {_f(r['contam@5'])} | "
                         f"{_f(r['gold_in_jev_topn'])} | {_f(r['gold_in_fused_topn'])} | {r['llm_ms_median']:,.0f} |")
        parts.append("")
    if (jdir / "latency.json").exists():
        lat = json.loads((jdir / "latency.json").read_text())
        parts += ["### Latency (medians per question)", "", "| stage | median |", "|---|---|",
                  f"| Jev, one request with {lat['jev_candidates']} nouls | {lat['jev_request_ms']:,.0f} ms "
                  f"(p90 {lat['jev_request_ms_p90']:,.0f} ms) |",
                  f"| LLM rerank, {lat['jev_candidates']} pairs (reference) | {lat['rerank_llm_ms_20_pairs']:,.0f} ms |",
                  f"| Cascade: LLM on Jev's top {lat['cascade_n']} | {lat['cascade_llm_ms']:,.0f} ms |",
                  f"| Cascade: Jev + LLM | {lat['cascade_total_ms']:,.0f} ms |", ""]
    if (jdir / "cost.json").exists():
        c = json.loads((jdir / "cost.json").read_text())
        parts += [f"Jev cost: {c['requests']} requests, {c['pairs_scored']:,} pairs, {c['input_tokens']:,} input tokens "
                  f"(as reported by the API) at US${c['price_usd_per_mtok_input']} per million = US${c['cost_usd']:.4f} "
                  f"(US${c['cost_usd_per_1000_questions']:.2f} per 1,000 questions); {c['output_tokens']:,} output tokens, "
                  "not billed.", ""]
    if (jdir / "answer_metrics.json").exists():
        am = json.loads((jdir / "answer_metrics.json").read_text())
        if am["variants"]:
            parts += ["### Answers", "", answer_table(am), ""]
        for name, cmp in am.get("comparisons", {}).items():
            b, a = name.split("_vs_")
            d = cmp["fact_accuracy_verified"]
            parts.append(f"- {LABEL[b]} vs {LABEL[a]}: exact-fact accuracy {d['delta'] * 100:+.1f} pts "
                         f"[{d['ci95'][0] * 100:+.1f}, {d['ci95'][1] * 100:+.1f}]")
        for v, sk in am.get("skipped", {}).items():
            parts.append(f"- {LABEL[v]}: answers skipped, {sk['uncached_generations']} uncached generations "
                         f"(about {sk['estimated_minutes']} min)")
        parts.append("")
    return parts


def write_report(cfg: dict) -> Path:
    res = Path(cfg["run"]["results_dir"])
    rm = json.loads((res / "retrieval_metrics.json").read_text())
    parts = ["# Reference run report", "", "Generated by `python -m hybridrag eval` from the JSON files in this directory.", "",
             "## Retrieval (answerable questions)", "", retrieval_table(rm), "", ci_table(rm), "",
             "### Paired comparisons", "", comparison_table(rm), "", "### By question type", "", qtype_table(rm), ""]
    ceil = next((v for k, v in rm.items() if k.startswith("gold_in_fused_top")), None)
    if ceil:
        key = next(k for k in rm if k.startswith("gold_in_fused_top"))
        parts += [f"Reranker ceiling (gold chunk within the fused top {key[len('gold_in_fused_top'):]}): "
                  + ", ".join(f"{t} {_f(x)}" for t, x in ceil.items()), ""]
    if (res / "answer_metrics.json").exists():
        am = json.loads((res / "answer_metrics.json").read_text())
        parts += ["## Answers", "", answer_table(am), "", answer_error_table(am), ""]
        if am.get("comparisons"):
            parts += ["| comparison | exact-fact acc. delta (verified) | paired bootstrap 95% CI |", "|---|---|---|"]
            for name, c in am["comparisons"].items():
                b, a = name.split("_vs_")
                d = c["fact_accuracy_verified"]
                parts.append(f"| {LABEL[b]} vs {LABEL[a]} | {d['delta'] * 100:+.1f} pts | "
                             f"[{d['ci95'][0] * 100:+.1f}, {d['ci95'][1] * 100:+.1f}] |")
            parts.append("")
    if (res / "latency.json").exists():
        parts += ["## Latency", "", latency_table(json.loads((res / "latency.json").read_text())), ""]
    if (res / "evalgen" / "summary.json").exists():
        eg = json.loads((res / "evalgen" / "summary.json").read_text())
        parts += ["## eval-dataset-generator", "", "| version | det_score | det pass rate | action accuracy |",
                  "|---|---|---|---|"]
        for v, s in eg["summaries"].items():
            parts.append(f"| {v} | {_f(s['det_score'])} | {_f(s['det_pass_rate'])} | {_f(s['action_accuracy'])} |")
        parts.append("")
        for g in eg["gates"]:
            parts.append(f"- gate `{g['candidate']}` vs `{g['baseline']}`: **{g['verdict']}** (exit {g['exit_code']}), "
                         f"det_score delta {g['delta']:+.3f}, 95% CI {g['delta_ci95']}" +
                         ("; " + "; ".join(g["reasons"]) if g["reasons"] else ""))
        parts.append("")
    if (res / "jev" / "retrieval_metrics.json").exists():
        parts += jev_section(res / "jev")
    out = res / "report.md"
    out.write_text("\n".join(parts), encoding="utf-8")
    return out
