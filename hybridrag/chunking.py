"""Split documents into retrievable chunks.

Sections ("## heading") are packed greedily into chunks of at most
``max_words`` words; a section longer than that is split at sentence
boundaries. Every chunk starts with the document title, so a chunk that says
"The meal per diem is EUR 90" still carries "Travel and Expense Policy (EU)".
Without that header, sibling documents produce chunks that differ only in a
number, and no retriever can tell which entity they belong to.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from .text import split_sentences


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    family: str
    text: str

    def to_dict(self) -> dict:
        return asdict(self)


def _sections(text: str) -> list[tuple[str, str]]:
    out = []
    for block in re.split(r"\n(?=## )", text.strip()):
        lines = block.strip().split("\n", 1)
        if lines[0].startswith("## "):
            out.append((lines[0][3:].strip(), lines[1].strip() if len(lines) > 1 else ""))
        else:
            out.append(("", block.strip()))
    return out


def _pieces(heading: str, body: str, max_words: int) -> list[str]:
    """One section as one or more pieces, each at most max_words (unless a single sentence is longer)."""
    label = f"{heading}: " if heading else ""
    if len(body.split()) <= max_words:
        return [label + body]
    pieces, cur = [], []
    for sent in split_sentences(body):
        if cur and len(" ".join(cur + [sent]).split()) > max_words:
            pieces.append(label + " ".join(cur))
            cur = []
        cur.append(sent)
    if cur:
        pieces.append(label + " ".join(cur))
    return pieces


def chunk_document(doc: dict, max_words: int = 50, title_header: bool = True) -> list[Chunk]:
    pieces: list[str] = []
    for heading, body in _sections(doc["text"]):
        pieces.extend(_pieces(heading, body, max_words))
    chunks, cur = [], []
    for p in pieces:
        if cur and len(" ".join(cur + [p]).split()) > max_words:
            chunks.append(cur)
            cur = []
        cur.append(p)
    if cur:
        chunks.append(cur)
    head = f"{doc['title']}\n" if title_header else ""
    return [Chunk(f"{doc['doc_id']}#{i}", doc["doc_id"], doc["title"], doc.get("family", ""), head + "\n".join(c))
            for i, c in enumerate(chunks)]


def chunk_corpus(docs: list[dict], max_words: int = 50, title_header: bool = True) -> list[Chunk]:
    out: list[Chunk] = []
    for d in docs:
        out.extend(chunk_document(d, max_words, title_header))
    return out


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def gold_chunk_ids(question: dict, chunks_by_doc: dict[str, list[Chunk]]) -> list[str]:
    """Chunks of the gold document that contain the evidence sentence (empty for unanswerable questions)."""
    if not question.get("gold_doc_id"):
        return []
    ev = _norm(question["evidence"])
    hits = [c.chunk_id for c in chunks_by_doc.get(question["gold_doc_id"], []) if ev in _norm(c.text)]
    if not hits:
        raise ValueError(f"{question['qid']}: evidence not found in any chunk of {question['gold_doc_id']}")
    return hits
