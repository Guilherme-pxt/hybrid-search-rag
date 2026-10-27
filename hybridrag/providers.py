"""Chat LLM providers behind one small interface.

``chat(model, messages, max_tokens, top_logprobs=0) -> ChatResult`` is all the
reranker and the generator need.

* ``OllamaProvider``           local server, POST /api/chat (every real run)
* ``OpenAICompatibleProvider`` any /v1/chat/completions endpoint, configured from
                               HYBRIDRAG_OPENAI_BASE_URL / HYBRIDRAG_OPENAI_API_KEY;
                               tested with mocks only, never called against a paid API
* ``ScriptedProvider``         deterministic fake for the offline tests
* ``CachedProvider``           sqlite3 cache around any of the above

Decoding is pinned (temperature 0, fixed seed). The cache stores the original
latency, so a cached rerun still reports what the stage cost when it ran.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol

import requests


@dataclass
class ChatResult:
    text: str
    latency_s: float
    cached: bool = False
    top_logprobs: list[dict[str, float]] = field(default_factory=list)  # per generated token: token -> logprob
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class ProviderError(RuntimeError):
    pass


class Provider(Protocol):
    name: str

    def chat(self, model: str, messages: list[dict], max_tokens: int = 256, top_logprobs: int = 0) -> ChatResult: ...

    def release(self, model: str) -> None: ...


class OllamaProvider:
    name = "ollama"

    def __init__(self, base_url: str = "http://localhost:11434", temperature: float = 0.0, seed: int = 42,
                 num_ctx: int = 4096, timeout: float = 300.0, keep_alive: str = "10m"):
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.seed = seed
        self.num_ctx = num_ctx
        self.timeout = timeout
        self.keep_alive = keep_alive

    def payload(self, model: str, messages: list[dict], max_tokens: int, top_logprobs: int) -> dict:
        body = {"model": model, "messages": messages, "stream": False, "think": False,
                "keep_alive": self.keep_alive,
                "options": {"temperature": self.temperature, "seed": self.seed, "num_ctx": self.num_ctx,
                            "num_predict": max_tokens}}
        if top_logprobs:
            body["logprobs"] = True
            body["top_logprobs"] = top_logprobs
        return body

    def chat(self, model: str, messages: list[dict], max_tokens: int = 256, top_logprobs: int = 0) -> ChatResult:
        body = self.payload(model, messages, max_tokens, top_logprobs)
        t0 = time.perf_counter()
        try:
            r = requests.post(f"{self.base_url}/api/chat", json=body, timeout=self.timeout)
        except requests.RequestException as exc:
            raise ProviderError(f"ollama request failed: {exc}") from exc
        if r.status_code != 200:
            raise ProviderError(f"ollama /api/chat returned {r.status_code}: {r.text[:300]}")
        data = r.json()
        lps = [{t["token"]: t["logprob"] for t in step.get("top_logprobs", [])} for step in data.get("logprobs") or []]
        return ChatResult(text=data.get("message", {}).get("content", ""), latency_s=time.perf_counter() - t0,
                          top_logprobs=lps, prompt_tokens=data.get("prompt_eval_count"),
                          completion_tokens=data.get("eval_count"))

    def release(self, model: str) -> None:
        try:
            requests.post(f"{self.base_url}/api/generate", json={"model": model, "keep_alive": 0}, timeout=60)
        except requests.RequestException:
            pass  # unloading is best effort; the server frees the model after keep_alive anyway


class OpenAICompatibleProvider:
    name = "openai_compatible"

    def __init__(self, base_url: str | None = None, api_key: str | None = None, temperature: float = 0.0,
                 seed: int = 42, timeout: float = 120.0):
        self.base_url = (base_url or os.environ.get("HYBRIDRAG_OPENAI_BASE_URL", "")).rstrip("/")
        self.api_key = api_key if api_key is not None else os.environ.get("HYBRIDRAG_OPENAI_API_KEY", "")
        self.temperature = temperature
        self.seed = seed
        self.timeout = timeout
        if not self.base_url:
            raise ProviderError("set HYBRIDRAG_OPENAI_BASE_URL (e.g. http://localhost:8000/v1)")

    def payload(self, model: str, messages: list[dict], max_tokens: int, top_logprobs: int) -> dict:
        body = {"model": model, "messages": messages, "temperature": self.temperature, "seed": self.seed,
                "max_tokens": max_tokens}
        if top_logprobs:
            body["logprobs"] = True
            body["top_logprobs"] = top_logprobs
        return body

    def chat(self, model: str, messages: list[dict], max_tokens: int = 256, top_logprobs: int = 0) -> ChatResult:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        t0 = time.perf_counter()
        try:
            r = requests.post(f"{self.base_url}/chat/completions", headers=headers,
                              json=self.payload(model, messages, max_tokens, top_logprobs), timeout=self.timeout)
        except requests.RequestException as exc:
            raise ProviderError(f"chat request failed: {exc}") from exc
        if r.status_code != 200:
            raise ProviderError(f"/chat/completions returned {r.status_code}: {r.text[:300]}")
        data = r.json()
        choice = data["choices"][0]
        lps = []
        for step in ((choice.get("logprobs") or {}).get("content") or []):
            lps.append({t["token"]: t["logprob"] for t in step.get("top_logprobs", [])})
        usage = data.get("usage") or {}
        return ChatResult(text=choice["message"].get("content") or "", latency_s=time.perf_counter() - t0,
                          top_logprobs=lps, prompt_tokens=usage.get("prompt_tokens"),
                          completion_tokens=usage.get("completion_tokens"))

    def release(self, model: str) -> None:
        return None


class ScriptedProvider:
    """Calls ``fn(model, messages) -> str | ChatResult``; records every call. For tests."""

    name = "scripted"

    def __init__(self, fn: Callable[[str, list[dict]], "str | ChatResult"]):
        self.fn = fn
        self.calls: list[dict] = []
        self.released: list[str] = []

    def chat(self, model: str, messages: list[dict], max_tokens: int = 256, top_logprobs: int = 0) -> ChatResult:
        self.calls.append({"model": model, "messages": messages, "max_tokens": max_tokens, "top_logprobs": top_logprobs})
        out = self.fn(model, messages)
        return out if isinstance(out, ChatResult) else ChatResult(text=out, latency_s=0.0)

    def release(self, model: str) -> None:
        self.released.append(model)


class CachedProvider:
    """sqlite3 cache keyed on the full request (provider, model, messages, decoding)."""

    def __init__(self, inner, path: str | Path):
        self.inner = inner
        self.name = f"cached:{inner.name}"
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path))
        self.db.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        self.hits = 0
        self.misses = 0
        self.allow_miss = True  # False: a miss raises instead of calling the model (cache-replay phases)

    def _key(self, model: str, messages: list[dict], max_tokens: int, top_logprobs: int) -> str:
        extra = {k: getattr(self.inner, k, None) for k in ("temperature", "seed", "num_ctx")}
        raw = json.dumps([self.inner.name, model, messages, max_tokens, top_logprobs, extra], sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()

    def has(self, model: str, messages: list[dict], max_tokens: int = 256, top_logprobs: int = 0) -> bool:
        key = self._key(model, messages, max_tokens, top_logprobs)
        return self.db.execute("SELECT 1 FROM cache WHERE key = ?", (key,)).fetchone() is not None

    def chat(self, model: str, messages: list[dict], max_tokens: int = 256, top_logprobs: int = 0) -> ChatResult:
        key = self._key(model, messages, max_tokens, top_logprobs)
        row = self.db.execute("SELECT value FROM cache WHERE key = ?", (key,)).fetchone()
        if row:
            self.hits += 1
            d = json.loads(row[0])
            return ChatResult(text=d["text"], latency_s=d["latency_s"], cached=True,
                              top_logprobs=d.get("top_logprobs", []), prompt_tokens=d.get("prompt_tokens"),
                              completion_tokens=d.get("completion_tokens"))
        if not self.allow_miss:
            raise ProviderError(f"cache miss with allow_miss=False ({model}); the reference LLM calls are not all cached")
        self.misses += 1
        res = self.inner.chat(model, messages, max_tokens=max_tokens, top_logprobs=top_logprobs)
        self.db.execute("INSERT OR REPLACE INTO cache VALUES (?, ?)", (key, json.dumps({
            "text": res.text, "latency_s": res.latency_s, "top_logprobs": res.top_logprobs,
            "prompt_tokens": res.prompt_tokens, "completion_tokens": res.completion_tokens})))
        self.db.commit()
        return res

    def release(self, model: str) -> None:
        self.inner.release(model)

    def close(self) -> None:
        self.db.close()


def build_provider(cfg: dict, cache_path: str | Path | None = None):
    kind = cfg.get("provider", "ollama")
    if kind == "ollama":
        p = OllamaProvider(cfg.get("base_url", "http://localhost:11434"), float(cfg.get("temperature", 0.0)),
                           int(cfg.get("seed", 42)), int(cfg.get("num_ctx", 4096)), float(cfg.get("timeout_s", 300)))
    elif kind == "openai_compatible":
        p = OpenAICompatibleProvider(temperature=float(cfg.get("temperature", 0.0)), seed=int(cfg.get("seed", 42)))
    else:
        raise ProviderError(f"unknown provider '{kind}' (ollama | openai_compatible)")
    return CachedProvider(p, cache_path) if cache_path else p
