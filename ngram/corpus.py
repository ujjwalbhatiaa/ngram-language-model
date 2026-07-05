"""Vocabulary building and OOV / sentence-boundary handling."""

from collections import Counter

UNK = "<unk>"
BOS = "<s>"
EOS = "</s>"


def build_vocab(train_sentences, min_count=2):
    """Build a vocabulary from training sentences.

    Words appearing fewer than `min_count` times in training are treated as
    unknown at train time too (not just at test time) — this is what lets the
    smoothing math ever assign realistic probability mass to genuinely novel
    words at test time, instead of a vocabulary that has "seen everything".
    """
    counts = Counter(w for sent in train_sentences for w in sent)
    vocab = {w for w, c in counts.items() if c >= min_count}
    vocab.add(UNK)
    # EOS is a legitimate prediction target (the model must be able to say
    # "the sentence ends here"); BOS is deliberately NOT added — it is only
    # ever context, never something the model predicts.
    vocab.add(EOS)
    return vocab


def replace_oov(sentence, vocab):
    return [w if w in vocab else UNK for w in sentence]


def add_boundaries(sentence, n):
    """Pad a token list with (n-1) BOS tokens and one EOS token."""
    pad = max(n - 1, 1)
    return [BOS] * pad + sentence + [EOS]
