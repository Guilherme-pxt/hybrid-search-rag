"""Okapi BM25 in numpy.

score(q, d) = sum over unique query terms t of
    idf(t) * tf(t, d) * (k1 + 1) / (tf(t, d) + k1 * (1 - b + b * |d| / avgdl))
idf(t) = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))   (the non-negative Lucene variant)

The per-term weights are precomputed into a dense (chunks x vocabulary)
matrix: at a few hundred chunks and a few thousand terms that is a few MB,
and a query is a column gather plus a row sum.
"""

from __future__ import annotations

import numpy as np

from .text import tokenize


class BM25:
    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.vocab: dict[str, int] = {}
        self.weights: np.ndarray | None = None
        self.idf: np.ndarray | None = None

    def fit(self, texts: list[str]) -> "BM25":
        docs = [tokenize(t) for t in texts]
        for toks in docs:
            for t in toks:
                self.vocab.setdefault(t, len(self.vocab))
        n, v = len(docs), len(self.vocab)
        tf = np.zeros((n, v), dtype=np.float32)
        for i, toks in enumerate(docs):
            for t in toks:
                tf[i, self.vocab[t]] += 1
        dl = tf.sum(axis=1)
        avgdl = float(dl.mean()) if n else 0.0
        df = (tf > 0).sum(axis=0)
        self.idf = np.log1p((n - df + 0.5) / (df + 0.5)).astype(np.float32)
        norm = self.k1 * (1 - self.b + self.b * dl / max(avgdl, 1e-9))
        self.weights = (self.idf * tf * (self.k1 + 1) / (tf + norm[:, None])).astype(np.float32)
        return self

    def scores(self, query: str) -> np.ndarray:
        ids = sorted({self.vocab[t] for t in tokenize(query) if t in self.vocab})
        if not ids:
            return np.zeros(self.weights.shape[0], dtype=np.float32)
        return self.weights[:, ids].sum(axis=1)

    def search(self, query: str, top_n: int) -> list[tuple[int, float]]:
        s = self.scores(query)
        order = np.lexsort((np.arange(len(s)), -s))[:top_n]  # score desc, index asc on ties
        return [(int(i), float(s[i])) for i in order if s[i] > 0]
