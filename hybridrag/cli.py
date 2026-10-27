"""Command line entry point: ``python -m hybridrag <command>``."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import pipeline as P
from .dense import EmbedderError
from .providers import ProviderError, build_provider


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="hybridrag", description=(
        "Hybrid search RAG over internal docs (BM25 + dense + RRF + LLM rerank + citation verifier), "
        "with a measured ablation."))
    p.add_argument("--config", default="config.toml", help="TOML config (default: config.toml)")
    p.add_argument("--results-dir", help="override [run].results_dir")
    p.add_argument("--index-dir", help="override [run].index_dir")
    p.add_argument("--no-cache", action="store_true", help="do not read or write the LLM response cache")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("synth", help="generate the synthetic corpus and gold questions (seeded)")
    s.add_argument("--seed", type=int)
    s.add_argument("--data-dir")

    s = sub.add_parser("index", help="chunk the corpus, embed the chunks, save vectors to disk")
    s.add_argument("--embedder", choices=["lsa", "ollama", "openai_compatible"])
    s.add_argument("--embed-model", help="model name for the ollama / openai_compatible embedder")
    s.add_argument("--no-title-header", action="store_true", help="do not prefix chunks with the document title")

    s = sub.add_parser("retrieve", help="show the ranked chunks for one question")
    s.add_argument("question")
    s.add_argument("--variant", choices=list(P.VARIANTS), default="hybrid_rerank")
    s.add_argument("-k", type=int, default=5)

    s = sub.add_parser("answer", help="answer one question with citations, then verify them")
    s.add_argument("question")
    s.add_argument("--variant", choices=list(P.VARIANTS), default="hybrid_rerank")
    s.add_argument("--keep-unsupported", action="store_true", help="flag unsupported sentences but do not remove them")

    s = sub.add_parser("eval", help="ablation over the gold set: retrieval metrics, answers, latency, report")
    s.add_argument("--variants", nargs="*", choices=list(P.VARIANTS),
                   help="variants to generate answers for (default: [answer].variants; none = retrieval only)")
    s.add_argument("--limit", type=int, help="first N questions only (smoke test)")
    s.add_argument("--no-rerank", action="store_true", help="first-stage variants only, no LLM call (no answers)")
    s.add_argument("--evalgen", help="path to an eval-dataset-generator checkout (score + gate the answers)")

    s = sub.add_parser("jev-eval", help="Jev (TypeSafe) reranking experiment over the reference run ([jev] in config)")
    s.add_argument("--limit", type=int, help="first N questions only (smoke test)")
    s.add_argument("--no-answers", action="store_true", help="retrieval, latency and cost only; no generation")
    s.add_argument("--max-usd", type=float, help="override [jev].max_usd")
    s.add_argument("--max-answer-minutes", type=float, default=20.0,
                   help="skip answers for a variant whose uncached generations would take longer (default 20)")

    s = sub.add_parser("run-all", help="synth, index, eval (and evalgen if configured), report")
    s.add_argument("--skip-synth", action="store_true")
    s.add_argument("--evalgen", help="path to an eval-dataset-generator checkout")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = P.load_config(args.config)
    if args.results_dir:
        cfg["run"]["results_dir"] = args.results_dir
    if args.index_dir:
        cfg["run"]["index_dir"] = args.index_dir

    def provider():
        return build_provider(cfg["llm"], cache_path=None if args.no_cache else cfg["run"].get("cache"))

    try:
        if args.command == "synth":
            if args.seed is not None:
                cfg["run"]["seed"] = args.seed
            if args.data_dir:
                cfg["run"]["data_dir"] = args.data_dir
            P.step_synth(cfg)
        elif args.command == "index":
            if args.embedder:
                cfg["embedder"]["kind"] = args.embedder
            if args.embed_model:
                cfg["embedder"]["model"] = args.embed_model
            if args.no_title_header:
                cfg["chunking"]["title_header"] = False
            P.step_index(cfg)
        elif args.command in ("retrieve", "answer"):
            retr = P.Retriever(cfg)
            prov = provider() if args.variant == "hybrid_rerank" or args.command == "answer" else None
            reranker = P.LLMReranker(prov, cfg["llm"]["model"]) if args.variant == "hybrid_rerank" else None
            res = retr.retrieve(args.question, args.variant, reranker)
            ranked = res.ranked[args.variant]
            if args.command == "retrieve":
                for i, cid in enumerate(ranked[:args.k], 1):
                    print(f"{i:>2}. [{cid}] {retr.by_id[cid].text.splitlines()[0]}")
                    print("    " + " ".join(retr.by_id[cid].text.splitlines()[1:])[:300])
                print("timings (ms): " + json.dumps({k: round(v, 2) for k, v in res.timings_ms.items()}))
            else:
                from .generate import generate_answer
                from .verify import verify

                passages = retr.passages(ranked)
                ans = generate_answer(prov, cfg["llm"]["model"], args.question, passages, int(cfg["answer"]["max_tokens"]))
                ver = verify(ans.text, dict(passages), remove=not args.keep_unsupported,
                             min_claim_words=int(cfg["verifier"]["min_claim_words"]),
                             min_overlap=float(cfg["verifier"]["min_overlap"]), question=args.question)
                print("context: " + ", ".join(cid for cid, _ in passages))
                print(f"\nraw answer:\n{ans.text}\n")
                for s in ver.sentences:
                    status = "ok" if s.supported else "UNSUPPORTED: " + "; ".join(s.reasons)
                    print(f"  - [{status}] {s.text}")
                print(f"\nverified answer:\n{ver.shown_text}")
        elif args.command == "eval":
            if args.evalgen:
                cfg.setdefault("evalgen", {})["path"] = args.evalgen
            P.step_eval(cfg, None if args.no_rerank else provider(), answer_variants=args.variants, limit=args.limit,
                        rerank=not args.no_rerank)
            path = cfg.get("evalgen", {}).get("path") or os.environ.get("HYBRIDRAG_EVALGEN_PATH", "")
            if path and not args.limit and not args.no_rerank:
                from .evalgen_bridge import run_evalgen

                run_evalgen(cfg, path)
            from .report import write_report

            print(f"report: {write_report(cfg)}")
        elif args.command == "jev-eval":
            from .jev import JevClient, JevError, JevReranker
            from .jev_eval import step_jev_eval
            from .report import write_report

            jcfg = cfg["jev"]
            client = JevClient(None if args.no_cache else cfg["run"].get("cache"), key_root=Path(args.config).resolve().parent,
                               timeout=float(jcfg.get("timeout_s", 60)))
            jev = JevReranker(client, jcfg["model"], float(jcfg["price_usd_per_mtok"]),
                              args.max_usd if args.max_usd is not None else float(jcfg["max_usd"]))
            try:
                step_jev_eval(cfg, provider(), jev, limit=args.limit, answers=not args.no_answers,
                              max_answer_minutes=args.max_answer_minutes)
            except JevError as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 2
            finally:
                client.close()
            if (Path(cfg["run"]["results_dir"]) / "retrieval_metrics.json").exists():
                print(f"report: {write_report(cfg)}")
        elif args.command == "run-all":
            if args.evalgen:
                cfg.setdefault("evalgen", {})["path"] = args.evalgen
            P.run_all(cfg, provider(), skip_synth=args.skip_synth)
    except (ProviderError, EmbedderError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
