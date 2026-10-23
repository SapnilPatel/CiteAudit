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
| Dev and test, our reranker | Test Recall@20 0.638 | running | ⏳ |
| Dev classification, 8 evidence settings | see main README | queued | ⏳ |

"Exact" means the largest difference across every metric in the file is below 1e-15.

## Metrics files

Each `*.metrics.json` here is the harness output of one re-run. Rankings and predictions are
regenerable and not committed.
