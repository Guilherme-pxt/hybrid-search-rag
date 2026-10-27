"""Dense retrieval: an embedder interface, three embedders and a numpy vector index.

``Embedder.embed(texts, kind)`` returns one L2-normalised row per text;
``kind`` is "query" or "passage" (some embedding models want a prefix).

* ``LSAEmbedder``              TF-IDF + truncated SVD (latent semantic analysis),
                               fitted on the corpus, numpy only. Used for the
                               reference run because the local Ollama refuses
                               /api/embed for every installed model (see README).
* ``OllamaEmbedder``           POST /api/embed, for a server with an embedding model
                               (nomic-embed-text, bge-m3, ...).
* ``OpenAICompatibleEmbedder`` POST {base}/embeddings; env-configured; tested with
                               mocks only, never called against a paid API.

The index is brute-force cosine similarity over a float32 matrix stored with
``np.save``: exact, and at a few hundred chunks it takes well under a millisecond.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Protocol

import numpy as np
import requests

from .text import tokenize


class EmbedderError(RuntimeError):
    pass


class Embedder(Protocol):
    name: str

    def fit(self, texts: list[str]) -> None: ...

    def embed(self, texts: list[str], kind: str = "passage") -> np.ndarray: ...


def l2_normalize(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32)
    n = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.maximum(n, 1e-12)


# ------------------------------------------------------------------ LSA


class LSAEmbedder:
    """TF-IDF (sublinear tf, smooth idf, L2) projected on the top-k right singular vectors."""

    def __init__(self, dims: int = 128):
        self.dims = dims
        self.vocab: dict[str, int] = {}
        self.idf: np.ndarray | None = None
        self.proj: np.ndarray | None = None  # vocab x k

    @property
    def name(self) -> str:
        return f"lsa-{self.dims}"

    def _tfidf(self, texts: list[str]) -> np.ndarray:
        m = np.zeros((len(texts), len(self.vocab)), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in tokenize(t):
                j = self.vocab.get(tok)
                if j is not None:
                    m[i, j] += 1
        m = np.where(m > 0, 1 + np.log(np.maximum(m, 1)), 0) * self.idf
        return l2_normalize(m)

    def fit(self, texts: list[str]) -> None:
        df: dict[str, int] = {}
        for t in texts:
            for tok in set(tokenize(t)):
                df[tok] = df.get(tok, 0) + 1
        self.vocab = {tok: i for i, tok in enumerate(sorted(df))}
        n = len(texts)
        self.idf = np.array([np.log((1 + n) / (1 + df[t])) + 1 for t in sorted(df)], dtype=np.float32)
        x = self._tfidf(texts)
        k = min(self.dims, min(x.shape) - 1)
        _, _, vt = np.linalg.svd(x, full_matrices=False)
        vt = vt[:k]
        # SVD signs are arbitrary; fix them so the saved index is byte-stable across runs
        signs = np.sign(vt[np.arange(k), np.abs(vt).argmax(axis=1)])
        self.proj = (vt * signs[:, None]).T.astype(np.float32)

    def embed(self, texts: list[str], kind: str = "passage") -> np.ndarray:
        if self.proj is None:
            raise EmbedderError("LSAEmbedder.embed called before fit")
        return l2_normalize(self._tfidf(texts) @ self.proj)

    def save(self, path: Path) -> None:
        np.savez(path, idf=self.idf, proj=self.proj)
        Path(str(path) + ".vocab.json").write_text(json.dumps(sorted(self.vocab, key=self.vocab.get)))

    @classmethod
    def load(cls, path: Path, dims: int) -> "LSAEmbedder":
        e = cls(dims)
        z = np.load(str(path) if str(path).endswith(".npz") else str(path) + ".npz")
        e.idf, e.proj = z["idf"], z["proj"]
        vocab = json.loads(Path(str(path) + ".vocab.json").read_text())
        e.vocab = {t: i for i, t in enumerate(vocab)}
        return e


# ------------------------------------------------------------------ HTTP embedders


class OllamaEmbedder:
    def __init__(self, model: str, base_url: str = "http://localhost:11434", batch_size: int = 32,
                 timeout: float = 300.0, query_prefix: str = "", passage_prefix: str = ""):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.batch_size = batch_size
        self.timeout = timeout
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix

    @property
    def name(self) -> str:
        return f"ollama:{self.model}"

    def fit(self, texts: list[str]) -> None:
        return None

    def embed(self, texts: list[str], kind: str = "passage") -> np.ndarray:
        prefix = self.query_prefix if kind == "query" else self.passage_prefix
        rows: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = [prefix + t for t in texts[i:i + self.batch_size]]
            try:
                r = requests.post(f"{self.base_url}/api/embed", json={"model": self.model, "input": batch},
                                  timeout=self.timeout)
            except requests.RequestException as exc:
                raise EmbedderError(f"ollama embed request failed: {exc}") from exc
            data = r.json() if r.content else {}
            if r.status_code != 200 or "embeddings" not in data:
                raise EmbedderError(f"ollama /api/embed ({self.model}) returned {r.status_code}: "
                                    f"{data.get('error', r.text[:200])}")
            rows.extend(data["embeddings"])
        return l2_normalize(np.array(rows, dtype=np.float32))


class OpenAICompatibleEmbedder:
    """Any server implementing POST /v1/embeddings. Reads HYBRIDRAG_OPENAI_BASE_URL and HYBRIDRAG_OPENAI_API_KEY."""

    def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None, batch_size: int = 64,
                 timeout: float = 120.0):
        self.model = model
        self.base_url = (base_url or os.environ.get("HYBRIDRAG_OPENAI_BASE_URL", "")).rstrip("/")
        self.api_key = api_key if api_key is not None else os.environ.get("HYBRIDRAG_OPENAI_API_KEY", "")
        self.batch_size = batch_size
        self.timeout = timeout
        if not self.base_url:
            raise EmbedderError("set HYBRIDRAG_OPENAI_BASE_URL (e.g. http://localhost:8000/v1)")

    @property
    def name(self) -> str:
        return f"openai-compatible:{self.model}"

    def fit(self, texts: list[str]) -> None:
        return None

    def embed(self, texts: list[str], kind: str = "passage") -> np.ndarray:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        rows: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            try:
                r = requests.post(f"{self.base_url}/embeddings", headers=headers,
                                  json={"model": self.model, "input": batch}, timeout=self.timeout)
            except requests.RequestException as exc:
                raise EmbedderError(f"embeddings request failed: {exc}") from exc
            if r.status_code != 200:
                raise EmbedderError(f"/embeddings returned {r.status_code}: {r.text[:200]}")
            data = sorted(r.json()["data"], key=lambda d: d["index"])
            rows.extend(d["embedding"] for d in data)
        return l2_normalize(np.array(rows, dtype=np.float32))


def build_embedder(cfg: dict) -> Embedder:
    kind = cfg.get("kind", "lsa")
    if kind == "lsa":
        return LSAEmbedder(int(cfg.get("dims", 128)))
    if kind == "ollama":
        return OllamaEmbedder(cfg["model"], cfg.get("base_url", "http://localhost:11434"),
                              query_prefix=cfg.get("query_prefix", ""), passage_prefix=cfg.get("passage_prefix", ""))
    if kind == "openai_compatible":
        return OpenAICompatibleEmbedder(cfg["model"])
    raise EmbedderError(f"unknown embedder kind '{kind}' (lsa | ollama | openai_compatible)")


# ------------------------------------------------------------------ index


class VectorIndex:
    def __init__(self, vectors: np.ndarray):
        self.vectors = l2_normalize(vectors)

    def search(self, qvec: np.ndarray, top_n: int) -> list[tuple[int, float]]:
        s = self.vectors @ np.asarray(qvec, dtype=np.float32).reshape(-1)
        order = np.lexsort((np.arange(len(s)), -s))[:top_n]
        return [(int(i), float(s[i])) for i in order]

    def save(self, path: Path) -> None:
        np.save(path, self.vectors)

    @classmethod
    def load(cls, path: Path) -> "VectorIndex":
        return cls(np.load(path))
