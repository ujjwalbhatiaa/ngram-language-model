"""An n-gram language model with add-k smoothing and recursive
linear interpolation down to the unigram level.

Methodology (documented honestly, see README for the full writeup):

  P_1(w)              = (count(w) + k) / (N + k*|V|)                    [add-k unigram]
  P_n(w | ctx)         = lam * P_hat_n(w | ctx) + (1 - lam) * P_{n-1}(w | ctx[1:])
  P_hat_n(w | ctx)     = (count(ctx, w) + k) / (count(ctx) + k*|V|)      [add-k n-gram MLE]

`lam` is a single scalar re-used at every level of the recursion for a given
top-level order (this is a simplification of full Jelinek-Mercer
interpolation, which tunes a separate lambda per order — documented as a
limitation, not hidden). `lam` is tuned per order by grid search against
held-out validation perplexity in train.py, not hand-picked.
"""

import math
import random
from collections import Counter, defaultdict

from ngram.corpus import BOS, EOS, UNK


class NGramModel:
    def __init__(self, n, vocab, k=1.0, lam=0.4):
        if n < 1:
            raise ValueError("n must be >= 1")
        self.n = n
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.k = k
        self.lam = lam
        # counts[order] maps a context tuple (length order-1) -> Counter over next word
        # counts[order][ctx] is a Counter; context_totals[order][ctx] = sum of that counter
        self.ngram_counts = {order: defaultdict(Counter) for order in range(1, n + 1)}
        self.context_totals = {order: defaultdict(int) for order in range(1, n + 1)}
        self.unigram_counts = Counter()
        self.total_unigrams = 0
        self._fitted = False

    def fit(self, sentences):
        """sentences: list of token lists, ALREADY boundary-padded and OOV-mapped.

        BOS is used as context but is never counted as a *predicted* word at
        any order — it isn't a real vocabulary item, just a fixed start
        marker, and letting it leak into the predicted-word counts would
        break the "probabilities sum to 1 over vocab" property (BOS isn't in
        `vocab`, so any probability mass assigned to it would vanish instead
        of landing on a real word)."""
        for sent in sentences:
            for order in range(1, self.n + 1):
                for i in range(order - 1, len(sent)):
                    w = sent[i]
                    if w == BOS:
                        continue
                    ctx = tuple(sent[i - order + 1:i]) if order > 1 else ()
                    self.ngram_counts[order][ctx][w] += 1
                    self.context_totals[order][ctx] += 1
            for w in sent:
                if w == BOS:
                    continue
                self.unigram_counts[w] += 1
                self.total_unigrams += 1
        self._fitted = True
        return self

    def _p_hat(self, order, ctx, w):
        """Add-k MLE estimate at a single order, no backoff."""
        if order == 1:
            count_w = self.unigram_counts.get(w, 0)
            return (count_w + self.k) / (self.total_unigrams + self.k * self.vocab_size)
        counts = self.ngram_counts[order].get(ctx)
        count_ctx_w = counts[w] if counts else 0
        count_ctx = self.context_totals[order].get(ctx, 0)
        return (count_ctx_w + self.k) / (count_ctx + self.k * self.vocab_size)

    def prob(self, context, w):
        """Interpolated probability of word w given a context tuple, recursing
        down to the unigram base case. context should have length >= n-1;
        only the last (order-1) tokens are used at each level."""
        if w not in self.vocab:
            w = UNK
        return self._prob_at_order(self.n, context, w)

    def _prob_at_order(self, order, context, w):
        if order == 1:
            return self._p_hat(1, (), w)
        ctx = tuple(context[-(order - 1):]) if order > 1 else ()
        p_hat = self._p_hat(order, ctx, w)
        p_lower = self._prob_at_order(order - 1, context, w)
        return self.lam * p_hat + (1 - self.lam) * p_lower

    def sentence_logprob(self, sentence):
        """Sum of log2 P(w_i | context) for every predicted token (everything
        after the (n-1) leading BOS pads, through and including EOS)."""
        pad = max(self.n - 1, 1)
        total = 0.0
        count = 0
        for i in range(pad, len(sentence)):
            ctx = sentence[max(0, i - self.n + 1):i]
            w = sentence[i]
            p = self.prob(ctx, w)
            if p <= 0:
                p = 1e-12  # numerical floor; add-k smoothing should prevent this in practice
            total += math.log2(p)
            count += 1
        return total, count

    def perplexity(self, sentences):
        """Corpus-level perplexity: 2 ** (-1/N * sum(log2 P))."""
        total_logprob = 0.0
        total_tokens = 0
        for sent in sentences:
            lp, c = self.sentence_logprob(sent)
            total_logprob += lp
            total_tokens += c
        if total_tokens == 0:
            return float("inf")
        avg_neg_logprob = -total_logprob / total_tokens
        return 2 ** avg_neg_logprob

    def next_word_distribution(self, context):
        """Full probability distribution over the vocabulary given a context.
        Returns (words, probs) as parallel lists, probs normalized to sum to 1
        (interpolated probabilities already should, but we renormalize to
        absorb floating point drift)."""
        words = sorted(self.vocab)
        probs = [self.prob(context, w) for w in words]
        s = sum(probs)
        probs = [p / s for p in probs]
        return words, probs

    def generate(self, max_tokens=40, seed=None, temperature=1.0):
        """Sample a sentence by repeatedly drawing from the model's own
        next-token distribution, starting from (n-1) BOS tokens and stopping
        at EOS or max_tokens. Deterministic for a fixed seed."""
        rng = random.Random(seed)
        pad = max(self.n - 1, 1)
        context = [BOS] * pad
        output = []
        for _ in range(max_tokens):
            words, probs = self.next_word_distribution(context)
            if temperature != 1.0:
                # temperature-scale in log space, then renormalize
                logp = [math.log(max(p, 1e-12)) / temperature for p in probs]
                m = max(logp)
                exps = [math.exp(x - m) for x in logp]
                s = sum(exps)
                probs = [e / s for e in exps]
            w = rng.choices(words, weights=probs, k=1)[0]
            if w == EOS:
                break
            output.append(w)
            context = (context + [w])[-pad:] if pad > 0 else []
        return output
