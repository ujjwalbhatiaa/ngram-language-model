"""Minimal, dependency-free sentence + word tokenizer.

This is intentionally simple (regex-based) rather than a full NLP tokenizer
library — the project's point is the n-gram modeling, not tokenization, and
a transparent tokenizer makes every downstream number easy to audit.
"""

import re

# Split on whitespace-normalized text after a sentence-ending punctuation
# mark followed by whitespace. Good enough for 19th-century prose; it will
# occasionally over/under-split on abbreviations ("Mr.") — a known, documented
# limitation (see README "Limitations").
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

# Words: letters plus internal apostrophes (don't, it's) or hyphens (well-known).
_WORD_RE = re.compile(r"[A-Za-z]+(?:['-][A-Za-z]+)*")


def split_sentences(text):
    """Split raw text into a list of sentence strings."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    sentences = _SENT_SPLIT_RE.split(text)
    return [s.strip() for s in sentences if s.strip()]


def tokenize(sentence):
    """Lowercase word tokens for a single sentence string."""
    return [w.lower() for w in _WORD_RE.findall(sentence)]


def tokenize_corpus(text):
    """Text -> list of token lists, one per sentence."""
    return [tokenize(s) for s in split_sentences(text) if tokenize(s)]
