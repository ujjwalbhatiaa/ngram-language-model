# N-gram Language Model — Results

Trained on *Alice's Adventures in Wonderland* (968 sentences, 26500 tokens). Vocabulary: 1303 tokens (min_count=2).

Split: 774 train / 96 validation / 98 test sentences (seed=42).

## Perplexity by model order

| n | interpolation λ | train PPL | val PPL | test PPL | out-of-domain PPL (Sherlock Holmes) |
|---|---|---|---|---|---|
| 1 | n/a (pure unigram) | 265.49 | 228.43 | 217.54 | 139.70 |
| 2 | 0.3 | 213.81 | 211.94 | 198.03 | 144.30 |
| 3 | 0.1 | 232.42 | 222.99 | 210.69 | 146.35 |
| 4 | 0.1 | 234.35 | 233.09 | 220.05 | 154.91 |

## Sample generated text (temperature=0.9)

**n=1:**

> about this the but large indeed said the <unk>

> which what <unk> <unk> they than see down other otherwise of alice i've her stay you what myself in cat

> be myself hardly or puzzling <unk>

**n=2:**

> all thing that cut likely it said the <unk>

> who what <unk> <unk> there stick said falling out out of and in her size you when never is directions <unk>

> caucus-race names have our pool <unk>

**n=3:**

> alice this the clock let's is running the <unk>

> who what <unk> <unk> there stretching salt executed out out of and in here so you what mystery into curious <unk>

> but natural hatter other presents <unk>

**n=4:**

> alice's think that cried like it round the a

> who what <unk> <unk> there speech said fan out out off and in herself sneezing you what name is didn't <unk>

> carried name have our poor <unk>

## Explainability: next-word distribution

Top-5 next-word predictions from the trigram model given context `the queen`:

| word | probability |
|---|---|
| the | 0.0463 |
| <unk> | 0.0401 |
| </s> | 0.0282 |
| and | 0.0240 |
| to | 0.0208 |

## OOV diagnostic: does out-of-domain perplexity reflect real domain shift, or is it masked by <unk> bucketing?

Using the bigram model (n=2, λ=0.3):

| set | tokens | <unk> rate | known-word PPL | <unk>-position PPL |
|---|---|---|---|---|
| Alice test (in-domain) | 2615 | 9.7% | 248.88 | 25.61 |
| Sherlock Holmes (out-of-domain) | 114724 | 21.2% | 237.59 | 25.05 |

See README "Findings" section for interpretation.
