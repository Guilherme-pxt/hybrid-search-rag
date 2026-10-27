"""Retrieval and answer metrics, and paired bootstrap confidence intervals.

Retrieval is scored at chunk level: the relevant chunks of a question are the
chunks of its gold document that contain the evidence sentence (usually one).

* Recall@k        share of relevant chunks in the top k
* MRR@10          1 / rank of the first relevant chunk (0 if not in the top 10)
* nDCG@10         binary relevance, log2 discount
* Contamination@k the top k holds a chunk from a sibling document of the same
                  template family (a near-duplicate distractor) ranked above
                  every relevant chunk, or while no relevant chunk is in the top k.
                  Only defined for questions whose gold document has siblings.
"""

from __future__ import annotations

import math

import numpy as np

from .text import content_words, extract_figures, is_idk, strip_citations


def recall_at(ranked: list[str], gold: set[str], k: int) -> float:
    return len(set(ranked[:k]) & gold) / len(gold) if gold else 0.0


def mrr_at(ranked: list[str], gold: set[str], k: int = 10) -> float:
    for i, c in enumerate(ranked[:k], start=1):
        if c in gold:
            return 1.0 / i
    return 0.0


def ndcg_at(ranked: list[str], gold: set[str], k: int = 10) -> float:
    dcg = sum(1.0 / math.log2(i + 1) for i, c in enumerate(ranked[:k], start=1) if c in gold)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(gold), k) + 1))
    return dcg / ideal if ideal else 0.0


def contaminated_at(ranked: list[str], gold: set[str], distractor_docs: set[str], k: int) -> bool:
    for c in ranked[:k]:
        if c in gold:
            return False
        if c.split("#", 1)[0] in distractor_docs:
            return True
    return False


def retrieval_row(ranked: list[str], gold: set[str], distractor_docs: set[str]) -> dict:
    row = {f"recall@{k}": recall_at(ranked, gold, k) for k in (1, 3, 5)}
    row["mrr@10"] = mrr_at(ranked, gold, 10)
    row["ndcg@10"] = ndcg_at(ranked, gold, 10)
    if distractor_docs:
        row["contam@1"] = float(contaminated_at(ranked, gold, distractor_docs, 1))
        row["contam@5"] = float(contaminated_at(ranked, gold, distractor_docs, 5))
    return row


# ------------------------------------------------------------------ answers


def fact_present(fact: str, answer: str) -> bool:
    """Every figure of the gold fact and every content word of it appear in the answer.

    "EUR 90" needs 90; "31 March" needs 31 and "march"; "qvctl drain ingest-gateway"
    needs each of its words. Order and surrounding wording are free.
    """
    answer = strip_citations(answer)
    figs = extract_figures(fact)
    if not figs <= extract_figures(answer):
        return False
    words = {w for w in content_words(fact) if not any(ch.isdigit() for ch in w)}
    return words <= content_words(answer)


def unsupported_figures(answer: str, allowed_text: str) -> list[str]:
    """Figures stated in the answer that appear neither in the gold document nor in the question."""
    answer = strip_citations(answer)
    if is_idk(answer):
        return []
    return sorted(extract_figures(answer) - extract_figures(allowed_text))


# ------------------------------------------------------------------ bootstrap


def bootstrap_ci(values: list[float], n: int = 10_000, seed: int = 42) -> tuple[float, float]:
    x = np.asarray(values, dtype=float)
    if len(x) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n, len(x)))
    lo, hi = np.percentile(x[idx].mean(axis=1), [2.5, 97.5])
    return float(lo), float(hi)


def paired_bootstrap(a: list[float], b: list[float], n: int = 10_000, seed: int = 42) -> dict:
    """Delta mean(b) - mean(a) over the same questions, with a 95% percentile interval."""
    xa, xb = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if xa.shape != xb.shape or len(xa) == 0:
        raise ValueError("paired bootstrap needs two equal-length, non-empty samples")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(xa), size=(n, len(xa)))
    d = (xb[idx] - xa[idx]).mean(axis=1)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"delta": float(xb.mean() - xa.mean()), "ci95": [float(lo), float(hi)], "n": int(len(xa)),
            "excludes_zero": bool(lo > 0 or hi < 0)}


def mean(xs) -> float | None:
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None
