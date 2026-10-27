"""Pointwise LLM reranker.

Each (question, passage) pair is graded on its own against a fixed 0-3
rubric, at temperature 0 with a fixed seed, and the model is asked for a
single digit. When the provider returns token log-probabilities, the grade is
the expected value over the digits 0-3 (renormalised), which breaks most ties
between passages that would all get a "3"; otherwise the digit is parsed from
the text. Candidates are then sorted by grade, ties broken by their fused rank.

The rubric puts the entity first (same region, plan, tier, SKU, service,
version) because the failure this stage exists to fix is the near-duplicate
sibling document. The question comes before the passage so consecutive calls
share a prompt prefix that the server can reuse.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

RUBRIC = """You grade whether a passage from a company's internal documents answers a question.
Reply with a single digit and nothing else:
3 = the passage states the answer, for exactly the entity the question asks about (same region or country, plan, product tier or SKU, service, version or office).
2 = the passage is about the right entity and topic but only partly answers.
1 = the passage is on the same topic but about a different entity (another region, plan, tier, SKU, service, version or office), or it is only background.
0 = the passage is unrelated to the question."""


@dataclass
class Graded:
    chunk_id: str
    grade: float
    raw: str
    latency_s: float
    cached: bool
    fused_rank: int


def grade_messages(question: str, passage: str) -> list[dict]:
    return [{"role": "system", "content": RUBRIC},
            {"role": "user", "content": f"Question: {question}\n\nPassage:\n{passage}\n\nGrade (0-3):"}]


def grade_from(text: str, top_logprobs: list[dict[str, float]]) -> float | None:
    if top_logprobs:
        first = top_logprobs[0]
        probs = {int(t.strip()): math.exp(lp) for t, lp in first.items() if t.strip() in {"0", "1", "2", "3"}}
        z = sum(probs.values())
        if z > 0:
            return sum(d * p for d, p in probs.items()) / z
    m = re.search(r"[0-3]", text or "")
    return float(m.group(0)) if m else None


class LLMReranker:
    def __init__(self, provider, model: str, use_logprobs: bool = True):
        self.provider = provider
        self.model = model
        self.use_logprobs = use_logprobs
        self.unparseable = 0

    def grade(self, question: str, passage: str) -> tuple[float, str, float, bool]:
        res = self.provider.chat(self.model, grade_messages(question, passage), max_tokens=1,
                                 top_logprobs=10 if self.use_logprobs else 0)
        g = grade_from(res.text, res.top_logprobs if self.use_logprobs else [])
        if g is None:
            self.unparseable += 1
            g = 0.0
        return g, res.text, res.latency_s, res.cached

    def rerank(self, question: str, candidates: list[tuple[str, str]]) -> tuple[list[Graded], float]:
        """candidates: (chunk_id, text) in fused order. Returns graded list (best first) and LLM seconds spent."""
        graded = []
        llm_s = 0.0
        for rank, (cid, text) in enumerate(candidates, start=1):
            g, raw, lat, cached = self.grade(question, text)
            llm_s += lat
            graded.append(Graded(cid, round(g, 4), raw, round(lat, 4), cached, rank))
        graded.sort(key=lambda x: (-x.grade, x.fused_rank))
        return graded, llm_s
