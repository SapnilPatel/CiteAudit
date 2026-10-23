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
| Dev classification, 8 evidence settings | see main README | queued | ⏳ |

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

## Metrics files

Each `*.metrics.json` here is the harness output of one re-run. Rankings and predictions are
regenerable and not committed.
