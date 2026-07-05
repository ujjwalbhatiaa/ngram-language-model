#!/usr/bin/env python3
"""Train unigram..4-gram interpolated language models on Alice's Adventures in
Wonderland, tune the interpolation weight on a held-out validation split,
report train/val/test/out-of-domain perplexity, and print sample generated
text at each order. Writes a full report to reports/RESULTS.md.

Usage:
    python3 train.py
"""

import argparse
import os
import random
import sys

from ngram.corpus import add_boundaries, build_vocab, replace_oov
from ngram.model import NGramModel
from ngram.tokenizer import tokenize_corpus

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
REPORTS_DIR = os.path.join(HERE, "reports")

LAMBDA_GRID = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
ORDERS = [1, 2, 3, 4]
SPLIT_SEED = 42
GEN_SEEDS = [1, 2, 3]


def load_sentences(path):
    with open(path, encoding="utf-8") as f:
        text = f.read()
    return tokenize_corpus(text)


def split_train_val_test(sentences, seed=SPLIT_SEED, train_frac=0.8, val_frac=0.1):
    sents = sentences[:]
    random.Random(seed).shuffle(sents)
    n = len(sents)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train = sents[:n_train]
    val = sents[n_train:n_train + n_val]
    test = sents[n_train + n_val:]
    return train, val, test


def prep(sentences, vocab, n):
    return [add_boundaries(replace_oov(s, vocab), n) for s in sentences]


def tune_lambda(order, train_sents_by_order, val_sents_by_order, vocab):
    model = NGramModel(order, vocab, k=1.0, lam=0.4)
    model.fit(train_sents_by_order[order])
    best_lam, best_ppl = None, float("inf")
    for lam in LAMBDA_GRID:
        model.lam = lam
        ppl = model.perplexity(val_sents_by_order[order])
        if ppl < best_ppl:
            best_ppl, best_lam = ppl, lam
    model.lam = best_lam
    return model, best_lam, best_ppl


def top_k_next(model, context, k=5):
    words, probs = model.next_word_distribution(context)
    ranked = sorted(zip(words, probs), key=lambda x: -x[1])
    return ranked[:k]


def unk_rate(sentences_prepped):
    total = sum(len(s) for s in sentences_prepped)
    unk = sum(1 for s in sentences_prepped for w in s if w == "<unk>")
    return unk, total, (unk / total * 100 if total else 0.0)


def unk_vs_known_perplexity(model, sentences_prepped, order):
    """Split perplexity into positions where the target is <unk> vs a real
    vocabulary word, to check whether OOV bucketing is masking a domain-shift
    effect rather than genuinely modeling it."""
    import math as _math
    pad = max(order - 1, 1)
    unk_logs, known_logs = [], []
    for sent in sentences_prepped:
        for i in range(pad, len(sent)):
            ctx = sent[max(0, i - order + 1):i]
            w = sent[i]
            p = model.prob(ctx, w)
            lp = _math.log2(max(p, 1e-12))
            (unk_logs if w == "<unk>" else known_logs).append(lp)

    def _ppl(logs):
        return 2 ** (-sum(logs) / len(logs)) if logs else None

    return {
        "unk_count": len(unk_logs), "unk_ppl": _ppl(unk_logs),
        "known_count": len(known_logs), "known_ppl": _ppl(known_logs),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default=os.path.join(DATA_DIR, "alice_wonderland.txt"))
    parser.add_argument("--ood-corpus", default=os.path.join(DATA_DIR, "sherlock_holmes.txt"),
                         help="Out-of-domain corpus for a generalization sanity check.")
    parser.add_argument("--min-count", type=int, default=2)
    args = parser.parse_args()

    os.makedirs(REPORTS_DIR, exist_ok=True)

    print(f"Loading corpus: {args.corpus}")
    all_sentences = load_sentences(args.corpus)
    print(f"  {len(all_sentences)} sentences, "
          f"{sum(len(s) for s in all_sentences)} tokens")

    train_raw, val_raw, test_raw = split_train_val_test(all_sentences)
    print(f"Split: {len(train_raw)} train / {len(val_raw)} val / {len(test_raw)} test sentences")

    vocab = build_vocab(train_raw, min_count=args.min_count)
    print(f"Vocabulary size (min_count={args.min_count}): {len(vocab)} tokens "
          f"(incl. <unk>)")

    print(f"Loading out-of-domain corpus: {args.ood_corpus}")
    ood_raw = load_sentences(args.ood_corpus)
    print(f"  {len(ood_raw)} sentences (used entirely as an out-of-domain test set)")

    results = []
    models = {}
    for n in ORDERS:
        train_p = prep(train_raw, vocab, n)
        val_p = prep(val_raw, vocab, n)
        test_p = prep(test_raw, vocab, n)
        ood_p = prep(ood_raw, vocab, n)

        if n == 1:
            model = NGramModel(1, vocab, k=1.0)
            model.fit(train_p)
            best_lam = None
        else:
            model, best_lam, _ = tune_lambda(
                n, {n: train_p}, {n: val_p}, vocab
            )

        train_ppl = model.perplexity(train_p)
        val_ppl = model.perplexity(val_p)
        test_ppl = model.perplexity(test_p)
        ood_ppl = model.perplexity(ood_p)

        models[n] = model
        results.append({
            "n": n, "lambda": best_lam,
            "train_ppl": train_ppl, "val_ppl": val_ppl,
            "test_ppl": test_ppl, "ood_ppl": ood_ppl,
        })
        print(f"n={n}  lambda={best_lam}  train_ppl={train_ppl:.2f}  "
              f"val_ppl={val_ppl:.2f}  test_ppl={test_ppl:.2f}  "
              f"ood_ppl(sherlock)={ood_ppl:.2f}")

    # Sample generations at each order
    samples = {}
    for n in ORDERS:
        model = models[n]
        gens = []
        for seed in GEN_SEEDS:
            tokens = model.generate(max_tokens=25, seed=seed, temperature=0.9)
            gens.append(" ".join(tokens))
        samples[n] = gens

    # A small explainability panel: top-5 next-word predictions for a fixed context
    trigram_model = models[3]
    demo_context = ["the", "queen"]
    top5 = top_k_next(trigram_model, demo_context, k=5)

    # OOV/UNK diagnostic: is out-of-domain perplexity actually reflecting
    # domain shift, or is it being masked by heavy <unk> bucketing?
    bigram_model = models[2]
    test_2 = prep(test_raw, vocab, 2)
    ood_2 = prep(ood_raw, vocab, 2)
    test_unk = unk_rate(test_2)
    ood_unk = unk_rate(ood_2)
    test_split = unk_vs_known_perplexity(bigram_model, test_2, 2)
    ood_split = unk_vs_known_perplexity(bigram_model, ood_2, 2)

    # Write report
    report_path = os.path.join(REPORTS_DIR, "RESULTS.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# N-gram Language Model — Results\n\n")
        f.write(f"Trained on *Alice's Adventures in Wonderland* "
                f"({len(all_sentences)} sentences, "
                f"{sum(len(s) for s in all_sentences)} tokens). "
                f"Vocabulary: {len(vocab)} tokens (min_count={args.min_count}).\n\n")
        f.write(f"Split: {len(train_raw)} train / {len(val_raw)} validation / "
                f"{len(test_raw)} test sentences (seed={SPLIT_SEED}).\n\n")
        f.write("## Perplexity by model order\n\n")
        f.write("| n | interpolation λ | train PPL | val PPL | test PPL | "
                "out-of-domain PPL (Sherlock Holmes) |\n")
        f.write("|---|---|---|---|---|---|\n")
        for r in results:
            lam_str = f"{r['lambda']:.1f}" if r["lambda"] is not None else "n/a (pure unigram)"
            f.write(f"| {r['n']} | {lam_str} | {r['train_ppl']:.2f} | "
                    f"{r['val_ppl']:.2f} | {r['test_ppl']:.2f} | {r['ood_ppl']:.2f} |\n")
        f.write("\n")

        f.write("## Sample generated text (temperature=0.9)\n\n")
        for n in ORDERS:
            f.write(f"**n={n}:**\n\n")
            for g in samples[n]:
                f.write(f"> {g}\n\n")

        f.write("## Explainability: next-word distribution\n\n")
        f.write(f"Top-5 next-word predictions from the trigram model given "
                f"context `{' '.join(demo_context)}`:\n\n")
        f.write("| word | probability |\n|---|---|\n")
        for w, p in top5:
            f.write(f"| {w} | {p:.4f} |\n")

        f.write("\n## OOV diagnostic: does out-of-domain perplexity reflect "
                "real domain shift, or is it masked by <unk> bucketing?\n\n")
        f.write(f"Using the bigram model (n=2, λ={results[1]['lambda']}):\n\n")
        f.write("| set | tokens | <unk> rate | known-word PPL | <unk>-position PPL |\n")
        f.write("|---|---|---|---|---|\n")
        f.write(f"| Alice test (in-domain) | {test_unk[1]} | {test_unk[2]:.1f}% | "
                f"{test_split['known_ppl']:.2f} | {test_split['unk_ppl']:.2f} |\n")
        f.write(f"| Sherlock Holmes (out-of-domain) | {ood_unk[1]} | {ood_unk[2]:.1f}% | "
                f"{ood_split['known_ppl']:.2f} | {ood_split['unk_ppl']:.2f} |\n")
        f.write("\nSee README \"Findings\" section for interpretation.\n")

    print(f"\nWrote report to {report_path}")
    print(f"OOV diagnostic: test unk-rate={test_unk[2]:.1f}% "
          f"(known PPL {test_split['known_ppl']:.2f}); "
          f"sherlock unk-rate={ood_unk[2]:.1f}% "
          f"(known PPL {ood_split['known_ppl']:.2f})")
    return results


if __name__ == "__main__":
    main()
