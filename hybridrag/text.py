"""Text utilities shared by the retrievers, the verifier and the metrics.

Everything here is deterministic and dependency free, so a reader can
predict every tokenization, figure match and sentence split by hand.
"""

from __future__ import annotations

import re

STOPWORDS = frozenset("""
a about above after again all also am an and any are as at be because been before being below between both but by
can could did do does doing down during each few for from further had has have having he her here hers him his how
i if in into is it its itself just me more most my no nor not now of off on once only or other our ours out over own
same she should so some such than that the their theirs them then there these they this those through to too under
until up very was we were what when where which while who whom why will with would you your yours
get gets got per much many s t
""".split())

# A token is a run of letters/digits, optionally joined by - . / (so "qv-pu200",
# "ing-5031", "7.3" and "0.35" stay whole). Compound tokens also emit their parts.
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-./][a-z0-9]+)*")


def stem(word: str) -> str:
    """Tiny suffix stripper: enough to match 'days'/'day', 'policies'/'policy'."""
    if any(c.isdigit() for c in word):
        return word
    if len(word) > 5 and word.endswith("ing"):
        return word[:-3]
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith("es") and word[-3] in "sxz":
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def tokenize(text: str) -> list[str]:
    """Lowercase, keep compound codes whole and also emit their parts, drop stopwords, stem."""
    out: list[str] = []
    for m in _TOKEN_RE.finditer(text.lower()):
        tok = m.group(0)
        parts = re.split(r"[-./]", tok)
        if len(parts) > 1:
            out.append(tok)
            out.extend(p for p in parts if p and p not in STOPWORDS)
        elif tok not in STOPWORDS:
            out.append(stem(tok))
    return out


def content_words(text: str) -> set[str]:
    """Stemmed words of length >= 3 that are not stopwords (used for overlap checks)."""
    return {t for t in tokenize(text) if len(t) >= 3 and not t.isdigit()}


# ------------------------------------------------------------------ figures

_IDENT_SHAPE = re.compile(r"^[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)+$|^[A-Za-z]+\d[A-Za-z0-9]*$")
_UPPER_CODE = re.compile(r"^[A-Z]{2,}(?:-[A-Z0-9]+)+$")
_NUM_RE = re.compile(r"(?<![\w.])(\d[\d,]*(?:\.\d+)?)")
_STRIP = "()[]{}<>\"'`*,;:!?#"


def normalize_number(tok: str) -> str:
    tok = tok.replace(",", "")
    if "." in tok:
        tok = tok.rstrip("0").rstrip(".") or "0"
    if tok.startswith("0") and len(tok) > 1 and not tok.startswith("0."):
        tok = tok.lstrip("0") or "0"
    return tok


def _is_identifier(tok: str) -> bool:
    has_alpha = any(c.isalpha() for c in tok)
    has_digit = any(c.isdigit() for c in tok)
    if has_alpha and has_digit and (_IDENT_SHAPE.match(tok) or (len(tok) >= 4 and tok.isalnum())):
        return True
    return bool(_UPPER_CODE.match(tok))


def extract_figures(text: str) -> set[str]:
    """Numbers and identifiers stated in ``text``.

    Identifiers (tokens mixing letters and digits such as QV-PU200, ING-5031,
    IP67, or upper-case codes such as RMA-EU) are kept whole and upper-cased;
    numbers are normalized ("1,500.00" -> "1500"). Currency words and units are
    ignored on purpose: "EUR 65" and "65 euros" state the same figure.
    """
    figures: set[str] = set()
    rest = []
    for raw in text.split():
        tok = raw.strip(_STRIP).rstrip(".")
        if tok and _is_identifier(tok):
            figures.add(tok.upper())
            rest.append(" ")
        else:
            rest.append(raw)
    remaining = " ".join(rest)
    for m in _NUM_RE.finditer(remaining):
        token = m.group(1).rstrip(".,")
        if token:
            figures.add(normalize_number(token))
    return figures


# ------------------------------------------------------------------ sentences

_CITE_RE = re.compile(r"\[([^\[\]]+)\]")
_CHUNK_ID_RE = re.compile(r"[a-z0-9][a-z0-9_\-]*#\d+")


def find_citations(text: str) -> list[str]:
    """Chunk ids cited as [doc-id#n] (several ids per bracket allowed)."""
    out = []
    for m in _CITE_RE.finditer(text):
        for cid in _CHUNK_ID_RE.findall(m.group(1).lower()):
            if cid not in out:
                out.append(cid)
    return out


def strip_citations(text: str) -> str:
    return re.sub(r"\s*\[[^\[\]]*#\d+[^\[\]]*\]", "", text).strip()


_LEADING_CITES_RE = re.compile(r"^(?:\s*\[[^\[\]]*#\d+[^\[\]]*\]\.?)+")
_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(\[])|\n+")


def split_sentences(text: str) -> list[str]:
    """Split an answer into sentences without breaking decimals ("2.5") or ids.

    A fragment that is only a citation ("[doc#1]" written after the period) is
    attached to the sentence before it.
    """
    parts = [p.strip(" -*\t") for p in _SPLIT_RE.split(text.strip()) if p and p.strip(" -*\t")]
    out: list[str] = []
    for p in parts:
        lead = _LEADING_CITES_RE.match(p)
        if out and lead:  # "[doc#1] Next sentence": the citation belongs to the sentence before
            out[-1] = out[-1] + " " + lead.group(0).strip()
            p = p[lead.end():].strip()
            if not p:
                continue
        out.append(p)
    return out


_IDK_RE = re.compile(
    r"\bi (?:do not|don't|dont) know\b|\b(?:not|no information) (?:\w+ ){0,3}(?:in|from) the (?:provided |given )?"
    r"(?:documents|passages|context)\b|\bdocuments do not (?:contain|mention|say|specify)\b|"
    r"\bpassages do not (?:contain|mention|say|specify)\b",
    re.IGNORECASE,
)


def is_idk(text: str) -> bool:
    return bool(_IDK_RE.search(text or ""))
