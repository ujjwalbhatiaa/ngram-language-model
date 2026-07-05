#!/usr/bin/env python3
"""Standalone text-generation CLI: train an n-gram model on a corpus and
sample sentences from it.

Usage:
    python3 generate.py --order 3 --num 5
    python3 generate.py --corpus data/sherlock_holmes.txt --order 4 --temperature 1.3
"""

import argparse
import os
import random

from ngram.corpus import add_boundaries, build_vocab, replace_oov
from ngram.model import NGramModel
from ngram.tokenizer import tokenize_corpus

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CORPUS = os.path.join(HERE, "data", "alice_wonderland.txt")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default=DEFAULT_CORPUS)
    parser.add_argument("--order", type=int, default=3, choices=[1, 2, 3, 4])
    parser.add_argument("--lam", type=float, default=0.6,
                         help="Interpolation weight (ignored for order=1). "
                              "train.py reports the validation-tuned optimum per order.")
    parser.add_argument("--min-count", type=int, default=2)
    parser.add_argument("--num", type=int, default=5, help="How many sentences to generate.")
    parser.add_argument("--max-tokens", type=int, default=30)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=None,
                         help="Base RNG seed; each sentence uses seed+i. Omit for random output.")
    args = parser.parse_args()

    with open(args.corpus, encoding="utf-8") as f:
        text = f.read()
    sentences = tokenize_corpus(text)
    vocab = build_vocab(sentences, min_count=args.min_count)
    prepped = [add_boundaries(replace_oov(s, vocab), args.order) for s in sentences]

    model = NGramModel(args.order, vocab, k=1.0, lam=args.lam)
    model.fit(prepped)

    base_seed = args.seed if args.seed is not None else random.randint(0, 10_000_000)
    print(f"Trained order-{args.order} model on {len(sentences)} sentences "
          f"({len(vocab)}-word vocab). Generating {args.num} sentence(s) "
          f"(base seed={base_seed}):\n")
    for i in range(args.num):
        tokens = model.generate(max_tokens=args.max_tokens, seed=base_seed + i,
                                 temperature=args.temperature)
        print(f"  {i + 1}. {' '.join(tokens)}")


if __name__ == "__main__":
    main()
