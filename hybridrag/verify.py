"""Deterministic citation verifier.

Every sentence of a generated answer is checked against the chunks it cites.
No model is involved, so every verdict can be reproduced by hand:

1. A sentence needs support when it states a figure (a number or an
   identifier such as QV-PU200, ING-5031, IP67) or makes a specific claim
   (at least ``min_claim_words`` content words). The "I don't know" sentence
   and short connective sentences are exempt.
2. A sentence that needs support is supported when
   - it cites at least one chunk id, and every cited id was in the context
     given to the generator (an invented id is a failure), and
   - every figure in the sentence appears in the text of the cited chunks, and
   - for sentences without figures, at least ``min_overlap`` of its content
     words appear in the cited chunks.
3. Unsupported sentences are flagged. With ``remove=True`` they are dropped
   and a caveat is appended; if nothing supported is left, the answer becomes
   the "I don't know" reply.

What it cannot catch: a claim copied faithfully from the wrong chunk. If the
model cites the UK policy for a question about the EU, the citation is valid
and the verifier passes it; that error belongs to retrieval and reranking.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .generate import IDK
from .text import content_words, extract_figures, find_citations, is_idk, split_sentences, strip_citations


@dataclass
class SentenceCheck:
    text: str
    citations: list[str]
    needs_support: bool
    supported: bool
    reasons: list[str] = field(default_factory=list)


@dataclass
class Verification:
    sentences: list[SentenceCheck]
    flagged: int
    claims: int
    final_text: str      # the answer after removal, without the caveat
    removed: int
    caveat: str = ""

    @property
    def shown_text(self) -> str:
        """What a user would see: the kept sentences plus the caveat."""
        return f"{self.final_text} {self.caveat}".strip()

    def to_dict(self) -> dict:
        return asdict(self)


def check_sentence(sentence: str, chunk_text: dict[str, str], min_claim_words: int = 4,
                   min_overlap: float = 0.5) -> SentenceCheck:
    cites = find_citations(sentence)
    body = strip_citations(sentence)
    figures = extract_figures(body)
    words = content_words(body)
    if is_idk(body) or (not figures and len(words) < min_claim_words):
        return SentenceCheck(sentence, cites, needs_support=False, supported=True)
    reasons = []
    if not cites:
        reasons.append("no citation")
    unknown = [c for c in cites if c not in chunk_text]
    if unknown:
        reasons.append("cites a chunk that was not in the context: " + ", ".join(unknown))
    known = [chunk_text[c] for c in cites if c in chunk_text]
    if known:
        support = " ".join(known)
        missing = sorted(figures - extract_figures(support))
        if missing:
            reasons.append("figures not in the cited chunks: " + ", ".join(missing))
        if not figures and words:
            overlap = len(words & content_words(support)) / len(words)
            if overlap < min_overlap:
                reasons.append(f"low word overlap with the cited chunks ({overlap:.2f})")
    return SentenceCheck(sentence, cites, needs_support=True, supported=not reasons, reasons=reasons)


def verify(answer: str, chunk_text: dict[str, str], remove: bool = True, min_claim_words: int = 4,
           min_overlap: float = 0.5) -> Verification:
    """``chunk_text`` maps every chunk id given to the generator to its text."""
    checks = [check_sentence(s, chunk_text, min_claim_words, min_overlap) for s in split_sentences(answer)]
    flagged = [c for c in checks if not c.supported]
    claims = sum(1 for c in checks if c.needs_support)
    caveat = ""
    if not remove or not flagged:
        final = answer.strip()
    else:
        kept = [c.text for c in checks if c.supported]
        if not any(c.needs_support for c in checks if c.supported):
            final = IDK
        else:
            final = " ".join(kept)
        n = len(flagged)
        caveat = (f"(Note: {n} statement{'s' if n > 1 else ''} removed because the cited documents do not "
                  f"support {'them' if n > 1 else 'it'}.)")
    return Verification(checks, len(flagged), claims, final, len(flagged) if remove else 0, caveat)
