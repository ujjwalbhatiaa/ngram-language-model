"""Unit + integration tests for the n-gram language model.

Run with:  python3 -m pytest test_ngram.py -v
       or:  python3 test_ngram.py           (falls back to a plain runner)
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ngram.corpus import add_boundaries, build_vocab, replace_oov, BOS, EOS, UNK
from ngram.model import NGramModel
from ngram.tokenizer import split_sentences, tokenize, tokenize_corpus

TOY_TEXT = (
    "the cat sat on the mat. the dog sat on the rug. "
    "the cat chased the dog. the dog chased the cat."
)


def toy_sentences(n, min_count=1):
    raw = tokenize_corpus(TOY_TEXT)
    vocab = build_vocab(raw, min_count=min_count)
    prepped = [add_boundaries(replace_oov(s, vocab), n) for s in raw]
    return prepped, vocab, raw


# ---------------------------------------------------------------- tokenizer

def test_split_sentences_basic():
    sents = split_sentences("Hello there. How are you? Fine!")
    assert sents == ["Hello there.", "How are you?", "Fine!"]


def test_tokenize_lowercases_and_strips_punctuation():
    toks = tokenize("Alice said, \"Curiouser and curiouser!\"")
    assert toks == ["alice", "said", "curiouser", "and", "curiouser"]


def test_tokenize_keeps_apostrophes_and_hyphens():
    toks = tokenize("It's a well-known fact, don't you think?")
    assert "it's" in toks
    assert "well-known" in toks
    assert "don't" in toks


def test_tokenize_corpus_empty_sentences_dropped():
    result = tokenize_corpus("Hello world. !!! Goodbye.")
    # the middle "sentence" has no word characters and should be dropped
    assert all(len(s) > 0 for s in result)
    assert len(result) == 2


# -------------------------------------------------------------------- corpus

def test_build_vocab_respects_min_count():
    sentences = [["a", "a", "a", "b", "b", "c"]]
    vocab = build_vocab(sentences, min_count=2)
    assert "a" in vocab and "b" in vocab
    assert "c" not in vocab  # count 1, below threshold
    assert UNK in vocab


def test_replace_oov():
    vocab = {"a", "b", UNK}
    out = replace_oov(["a", "b", "z"], vocab)
    assert out == ["a", "b", UNK]


def test_add_boundaries_padding_length():
    sent = ["a", "b"]
    padded = add_boundaries(sent, n=3)
    assert padded[:2] == [BOS, BOS]
    assert padded[-1] == EOS
    assert padded == [BOS, BOS, "a", "b", EOS]


def test_add_boundaries_unigram_still_pads_one():
    # n=1 has no real "context", but we still pad with 1 BOS for consistency
    padded = add_boundaries(["a"], n=1)
    assert padded[0] == BOS
    assert padded[-1] == EOS


# --------------------------------------------------------------------- model

def test_unigram_probabilities_sum_to_one_over_vocab():
    prepped, vocab, _ = toy_sentences(n=1)
    model = NGramModel(1, vocab, k=1.0)
    model.fit(prepped)
    total = sum(model.prob((), w) for w in vocab)
    assert math.isclose(total, 1.0, rel_tol=1e-9)


def test_bigram_interpolated_distribution_sums_to_one():
    prepped, vocab, _ = toy_sentences(n=2)
    model = NGramModel(2, vocab, k=1.0, lam=0.5)
    model.fit(prepped)
    words, probs = model.next_word_distribution(["the"])
    assert math.isclose(sum(probs), 1.0, rel_tol=1e-6)
    assert all(p >= 0 for p in probs)


def test_trigram_distribution_sums_to_one():
    prepped, vocab, _ = toy_sentences(n=3)
    model = NGramModel(3, vocab, k=1.0, lam=0.6)
    model.fit(prepped)
    words, probs = model.next_word_distribution([BOS, "the"])
    assert math.isclose(sum(probs), 1.0, rel_tol=1e-6)


def test_add_k_smoothing_gives_unseen_bigram_nonzero_probability():
    prepped, vocab, _ = toy_sentences(n=2, min_count=1)
    model = NGramModel(2, vocab, k=1.0, lam=0.5)
    model.fit(prepped)
    # "mat rug" never appears together in the toy corpus
    p = model.prob(["mat"], "rug")
    assert p > 0.0


def test_higher_frequency_bigram_gets_higher_probability():
    # "the cat" / "the dog" appear multiple times; "the mat" never as a bigram
    prepped, vocab, _ = toy_sentences(n=2, min_count=1)
    model = NGramModel(2, vocab, k=1.0, lam=0.7)
    model.fit(prepped)
    p_common = model.prob(["the"], "cat")
    p_rare = model.prob(["the"], "mat")
    assert p_common > p_rare


def test_perplexity_is_finite_and_positive_on_seen_data():
    prepped, vocab, _ = toy_sentences(n=2)
    model = NGramModel(2, vocab, k=1.0, lam=0.5)
    model.fit(prepped)
    ppl = model.perplexity(prepped)
    assert ppl > 1.0
    assert ppl != float("inf")


def test_model_trained_on_corpus_is_less_perplexed_by_itself_than_by_noise():
    """Sanity check on the actual math: a model should assign much lower
    perplexity to sentences drawn from its own training distribution than to
    sentences built from words it barely knows."""
    prepped, vocab, raw = toy_sentences(n=2, min_count=1)
    model = NGramModel(2, vocab, k=1.0, lam=0.6)
    model.fit(prepped)
    own_ppl = model.perplexity(prepped)

    noise_vocab = {UNK}
    noise = [add_boundaries(replace_oov(["zzz", "qqq", "xxx"], noise_vocab), 2)]
    noise_ppl = model.perplexity(noise)

    assert own_ppl < noise_ppl


def test_generation_is_deterministic_for_fixed_seed():
    prepped, vocab, _ = toy_sentences(n=2, min_count=1)
    model = NGramModel(2, vocab, k=1.0, lam=0.5)
    model.fit(prepped)
    out1 = model.generate(max_tokens=15, seed=7)
    out2 = model.generate(max_tokens=15, seed=7)
    assert out1 == out2


def test_generation_stops_at_eos_or_max_tokens():
    prepped, vocab, _ = toy_sentences(n=2, min_count=1)
    model = NGramModel(2, vocab, k=1.0, lam=0.5)
    model.fit(prepped)
    out = model.generate(max_tokens=5, seed=1)
    assert len(out) <= 5
    assert EOS not in out  # EOS is a stop signal, never emitted into the output


def test_generation_only_produces_vocab_tokens():
    prepped, vocab, _ = toy_sentences(n=3, min_count=1)
    model = NGramModel(3, vocab, k=1.0, lam=0.5)
    model.fit(prepped)
    out = model.generate(max_tokens=20, seed=3)
    assert all(w in vocab for w in out)


def test_unknown_word_at_query_time_falls_back_to_unk():
    prepped, vocab, _ = toy_sentences(n=1, min_count=1)
    model = NGramModel(1, vocab, k=1.0)
    model.fit(prepped)
    p_known = model.prob((), "cat")
    p_totally_unseen_word = model.prob((), "gigantosaurus")  # not in training vocab at all
    p_unk_direct = model.prob((), UNK)
    assert math.isclose(p_totally_unseen_word, p_unk_direct, rel_tol=1e-9)
    assert p_known != p_unk_direct  # they shouldn't collide by coincidence


if __name__ == "__main__":
    # Minimal runner so `python3 test_ngram.py` works without pytest installed.
    tests = [obj for name, obj in list(globals().items())
             if name.startswith("test_") and callable(obj)]
    passed, failed = 0, 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL  {t.__name__}: {e}")
            failed += 1
    print(f"\n{passed}/{len(tests)} tests passed")
    if failed:
        sys.exit(1)
