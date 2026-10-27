"""Answer generation with inline citations to chunk ids."""

from __future__ import annotations

from dataclasses import dataclass

IDK = "I don't know based on the provided documents."

SYSTEM = f"""You answer questions about Quorvia Instruments' internal documents, using only the passages provided.
Rules:
1. Every sentence that states a fact ends with the id of the passage that supports it, in square brackets, for example [policy-travel-eu#1].
2. Use only facts stated in the passages. Many passages look alike but describe a different region, plan, product tier, SKU, service, version or office: use only a passage about exactly the entity in the question.
3. If no passage answers the question for that exact entity, reply exactly: {IDK}
4. Answer in at most three short sentences. Do not add advice or background."""


@dataclass
class Answer:
    text: str
    latency_s: float
    cached: bool
    context_ids: list[str]


def answer_messages(question: str, passages: list[tuple[str, str]]) -> list[dict]:
    ctx = "\n\n".join(f"[{cid}]\n{text}" for cid, text in passages)
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Passages:\n\n{ctx}\n\nQuestion: {question}"}]


def generate_answer(provider, model: str, question: str, passages: list[tuple[str, str]],
                    max_tokens: int = 200) -> Answer:
    res = provider.chat(model, answer_messages(question, passages), max_tokens=max_tokens)
    return Answer(res.text.strip(), res.latency_s, res.cached, [cid for cid, _ in passages])
