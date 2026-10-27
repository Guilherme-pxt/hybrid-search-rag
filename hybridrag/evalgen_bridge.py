"""Score the RAG answers with eval-dataset-generator (``evalgen``) and run its gate.

evalgen grades a system's replies against a reviewed eval set with
deterministic checks (action, required-fact recall, numbers not supported by
the documentation) and gates a candidate against a baseline. This module:

1. writes our gold questions as an evalgen eval set (answerable -> action
   ``answer`` with the gold fact as the required fact; unanswerable -> action
   ``decline``), our corpus as its ``kb.json``, and each variant's verified
   answers as evalgen predictions (an "I don't know" reply is a ``decline``);
2. scores each variant with evalgen's own ``score_version`` (deterministic
   checks only: its LLM judge rubric is written for its Ledgerfox support
   assistant and does not apply here);
3. runs evalgen's CLI ``gate`` on det_score and records verdict and exit code.

evalgen is an optional, separate checkout; nothing here is imported unless a
path is configured.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from .generate import IDK
from .pipeline import read_jsonl, write_json, write_jsonl
from .text import is_idk, strip_citations

CLUSTERS = {"exact_code": 0, "lexical": 1, "paraphrase": 2, "unanswerable": 3}


def to_eval_items(questions: list[dict]) -> list[dict]:
    items = []
    for q in questions:
        if q["qtype"] == "unanswerable":
            expected = {"action": "decline", "tool": None, "required_facts": [], "reference_answer": IDK}
        else:
            expected = {"action": "answer", "tool": None, "required_facts": [q["gold_fact"]],
                        "reference_answer": q["evidence"]}
        items.append({"item_id": q["qid"], "log_id": q["qid"], "user_message": q["question"],
                      "context_doc_ids": [q["gold_doc_id"]] if q["gold_doc_id"] else [],
                      "cluster_id": CLUSTERS[q["qtype"]], "cluster_label": q["qtype"],
                      "selection_reasons": ["gold"], "failure_signals": [], "group_size": 1,
                      "expected": expected, "review": {"decision": "accept", "reviewer": "synthetic-gold"}})
    return items


def to_predictions(recs: list[dict], variant: str, model: str) -> list[dict]:
    out = []
    for r in recs:
        text = r["answer_verified"]
        action = "decline" if is_idk(text) else "answer"
        resp = {"action": action, "message": strip_citations(text), "tool": None}
        out.append({"item_id": r["qid"], "version": variant, "model": model, "prompt": f"rag-{variant}",
                    "raw": json.dumps(resp), "response": resp, "parse_error": None,
                    "latency_s": r["generate_s"], "cached": r["generate_cached"]})
    return out


def run_evalgen(cfg: dict, evalgen_path: str, log=print) -> dict:
    root = Path(evalgen_path).resolve()
    if not (root / "evalgen" / "score.py").exists():
        raise FileNotFoundError(f"{root} is not an eval-dataset-generator checkout")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from evalgen.schema import EvalItem  # noqa: E402
    from evalgen.score import score_version, summarize  # noqa: E402

    res = Path(cfg["run"]["results_dir"])
    run_dir = res / "evalgen"
    data_dir = run_dir / "data"
    questions = read_jsonl(Path(cfg["run"]["data_dir"]) / "questions.jsonl")
    docs = read_jsonl(Path(cfg["run"]["data_dir"]) / "corpus.jsonl")
    items = to_eval_items(questions)
    write_jsonl(run_dir / "03_eval_set.jsonl", items)
    write_json(data_dir / "kb.json", {d["doc_id"]: {"title": d["title"], "text": d["text"]} for d in docs})
    kb = json.loads((data_dir / "kb.json").read_text())
    eval_items = [EvalItem.from_dict(i) for i in items]
    ecfg = cfg.get("evalgen", {})
    variants = [ecfg.get("baseline", "hybrid")] + list(ecfg.get("candidates", ["hybrid_rerank"]))
    summaries = {}
    for v in variants:
        recs = read_jsonl(res / f"answers_{v}.jsonl")
        preds = to_predictions(recs, v, cfg["llm"]["model"])
        write_jsonl(run_dir / f"04_predictions_{v}.jsonl", preds)
        scores = score_version(eval_items, preds, kb, judge=None)
        write_jsonl(run_dir / f"04_scores_{v}.jsonl", (s.to_dict() for s in scores))
        s = summarize(scores)
        s.pop("per_cluster", None)
        summaries[v] = s
        log(f"evalgen score {v}: det_score={s['det_score']} det_pass={s['det_pass_rate']} action_acc={s['action_accuracy']}")
    (run_dir / "config.toml").write_text(
        f'[run]\nseed = {int(cfg["run"]["seed"])}\ndata_dir = "data"\n'
        '\n[gate]\nmetric = "det_score"\nmax_overall_drop = 0.05\nmax_cluster_drop = 0.15\n'
        'min_cluster_items = 3\nmax_new_failures = 3\n', encoding="utf-8")
    gates = []
    for cand in variants[1:]:
        cmd = [sys.executable, "-m", "evalgen", "--config", str((run_dir / "config.toml").resolve()),
               "--run-dir", str(run_dir.resolve()), "gate", "--baseline", variants[0], "--candidate", cand,
               "--metric", "det_score"]
        p = subprocess.run(cmd, cwd=str(root), capture_output=True, text=True)
        log(p.stdout.strip() or p.stderr.strip())
        g = json.loads((run_dir / f"05_gate_{cand}.json").read_text())
        gates.append({"baseline": variants[0], "candidate": cand, "verdict": g["verdict"], "exit_code": p.returncode,
                      "delta": g["delta"], "delta_ci95": g["delta_ci95"], "reasons": g["reasons"],
                      "command": (f"python -m evalgen --config {run_dir}/config.toml --run-dir {run_dir} gate "
                                  f"--baseline {variants[0]} --candidate {cand} --metric det_score")})
    out = {"evalgen_checkout": root.name, "summaries": summaries, "gates": gates}
    write_json(run_dir / "summary.json", out)
    return out
