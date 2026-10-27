"""Reciprocal Rank Fusion.

Cormack, Clarke and Buettcher, "Reciprocal Rank Fusion outperforms Condorcet
and individual Rank Learning Methods", SIGIR 2009:

    RRF(d) = sum over rankings r of 1 / (k + rank_r(d)),   k = 60, ranks start at 1

RRF uses ranks only, so BM25 scores (unbounded) and cosine similarities
(-1..1) never need to be calibrated against each other.
"""

from __future__ import annotations


def rrf(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    first_seen: dict[str, tuple[int, int]] = {}
    for r_idx, ranking in enumerate(rankings):
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
            first_seen.setdefault(item, (rank, r_idx))
    # ties broken by best rank in any list, then by list order: deterministic
    return sorted(scores.items(), key=lambda kv: (-kv[1], first_seen[kv[0]]))
