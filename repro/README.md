# Independent reproduction

This folder re-runs every result in the repository from scratch on a second machine, using only the
scripts in [`scripts/`](../scripts/) and the authors' public data. The goal is to confirm the
original numbers before changing anything.

**Machine:** Apple M4, 16 GB, MPS backend. Python 3.11, PyTorch 2.14, Transformers 5.20.
The original runs used an Apple M2 with older library versions.

## Scripts

| Script | What it re-runs | Time on M4 |
|---|---|---|
| [`run_retrieval.sh`](run_retrieval.sh) | Paper's BM25+MonoT5 pipeline on dev and test, then trains our reranker and scores it on dev and test | about 1.5 hours |
| [`run_classification.sh`](run_classification.sh) | Rebuilds the 8 dev classifier inputs and diffs them against the committed ones, then runs the released checkpoint on each and on the authors' own released input | about 1 hour |
| [`check_authors_input.py`](check_authors_input.py) | Inspects the authors' released classifier input from their Drive folder | seconds |

## Status

| Result | Original | Re-run | Match |
|---|---|---|---|
| Test, BM25 only | Recall@20 0.444, MRR 0.257 | Recall@20 0.444, MRR 0.257 | ✅ exact on all 26 metrics |
| Test, paper's pipeline | Recall@20 0.529, MRR 0.323 | Recall@20 0.529, MRR 0.323 | ✅ exact on all 26 metrics |
| Dev, paper's pipeline | Recall@20 0.534, MRR 0.364 | Recall@20 0.534, MRR 0.364 | ✅ exact on all 26 metrics |
| Dev, our reranker (retrained) | Recall@20 0.643, MRR 0.405 | Recall@20 0.643, MRR 0.405 | ✅ within noise; AllGold@20 0.525 vs 0.510 |
| Test, our reranker (retrained) | Recall@20 0.638, MRR 0.365 | Recall@20 0.627, MRR 0.379 | ✅ within noise |
| Test, our reranker minus paper's pipeline | Recall@20 +0.109 [+0.084, +0.131], 17/20 articles | Recall@20 +0.098 [+0.071, +0.121], 17/20 articles | ✅ same conclusion |
| Dev classification, paper's retrieval, top 20 and top 10 | see below | see below | ✅ exact |
| Dev classification, the other 6 settings | see below | see below | ✅ same conclusions |
| Released classifier input from the authors | not run originally | see below | new |

"Exact" means the largest difference across every metric in the file is below 1e-15.

"Within noise" applies to the retrained reranker. Training on a GPU is not bit-for-bit repeatable,
and this machine uses newer library versions, so a second training run lands slightly differently.
Every gap is inside the original 95% confidence interval, and the reranker still beats the paper's
pipeline on every metric with every interval excluding zero:

| Test, rerun reranker minus paper's pipeline | Delta | 95% CI | Articles won |
|---|---|---|---|
| Recall@20 | +0.098 | +0.071 to +0.121 | 17 of 20 |
| Recall@10 | +0.087 | +0.046 to +0.123 | 15 of 20 |
| AllGold@20 | +0.121 | +0.081 to +0.166 | 16 of 20 |
| MRR | +0.070 | +0.022 to +0.115 | 15 of 20 |

The two training runs differ by about 0.01 Recall@20 on test. Ablation effects of that size or
smaller, such as several in the findings report, are within run-to-run variance.

## Classification, dev split (316 citations)

The released MultiVerS checkpoint was run on all 8 evidence settings. The paper's retrieval
inputs were rebuilt byte-for-byte identical to the committed ones, and their predictions match the
original exactly, per class. The other six settings are built from the retrained reranker's
rankings, so they differ slightly from the committed inputs.

| Setting | Evidence recall | Original accuracy | Rerun accuracy | Original macro-F1 | Rerun macro-F1 |
|---|---|---|---|---|---|
| Random, top 20 | 0.131 | 0.487 | 0.484 | 0.473 | 0.476 |
| **Paper's retrieval, top 20** | 0.534 | 0.677 | **0.677** | 0.684 | **0.684** |
| Our retrieval, top 20 | 0.643 | 0.674 | 0.658 | 0.674 | 0.651 |
| Perfect evidence, top 20 | 1.000 | 0.655 | 0.668 | 0.656 | 0.662 |
| Random, top 10 | 0.059 | 0.386 | 0.361 | 0.369 | 0.343 |
| **Paper's retrieval, top 10** | 0.425 | 0.633 | **0.633** | 0.615 | **0.615** |
| Our retrieval, top 10 | 0.551 | 0.620 | 0.614 | 0.593 | 0.593 |
| Perfect evidence, top 10 | 1.000 | 0.601 | 0.617 | 0.582 | 0.590 |

Paired against the paper's retrieval, with McNemar's exact test:

| Comparison | Rerun macro-F1 delta | Only this setting right | Only paper's right | p |
|---|---|---|---|---|
| Ours, top 20 | -0.033 | 25 | 31 | 0.504 |
| Perfect, top 20 | -0.022 | 25 | 28 | 0.784 |
| Random, top 20 | -0.208 | 21 | 82 | < 0.001 |
| Ours, top 10 | -0.022 | 24 | 30 | 0.497 |
| Perfect, top 10 | -0.025 | 24 | 29 | 0.583 |
| Random, top 10 | -0.272 | 23 | 109 | < 0.001 |

**Both original conclusions hold.** Random evidence is much worse, so the classifier reads its
evidence. Better or even perfect evidence does not beat the paper's retrieval under the frozen
checkpoint, and no such difference is significant.

## The authors' released classifier input

The authors' Google Drive folder holds, next to the checkpoint, a `scifact_10` folder with their own
classifier input for train, dev and test. The original work did not use it.
[`check_authors_input.py`](check_authors_input.py) and the comparisons below describe it.

- **It confirms the 3-way label mapping exactly.** Its gold labels match the mapping derived from
  the raw annotations on dev and test, including `INDIRECT_NOT_REVIEW` as ACCURATE. Train has 3
  fewer citations than the official file.
- **It is a top-5 setting, not top 20.** Each citation gets one document of 5 sentences. About 75% of
  those sentences are in the paper pipeline's own top 5 in our rerun, and 90% are in its top 20.
  Its citation texts are the full citing sentence, while the official files hold a trimmed version,
  which likely explains the rest.
- **Its evidence field only carries the label.** Every labelled citation marks exactly one sentence.

Running the released checkpoint on it gives the first test-split classification numbers measured
here:

| Test, 606 citations | Paper, top 20 | Authors' released input, top 5 |
|---|---|---|
| Accuracy (micro-F1) | 0.59 | 0.558 |
| Macro-F1 | 0.52 | 0.444 |
| ACCURATE F1 | 0.69 | 0.693 |
| NOT ACCURATE F1 | 0.43 | 0.312 |
| NOT ENOUGH INFO F1 | 0.42 | 0.326 |

On dev the same input gives accuracy 0.680 and macro-F1 0.641. Because this is a top-5 setting,
it does not reproduce or refute the paper's top-20 test figures. No released file matches that
setting, so those figures remain unreproduced.

## Metrics files

Each `*.metrics.json` here is the harness output of one re-run. Rankings and predictions are
regenerable and not committed.
