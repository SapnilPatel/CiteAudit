#!/usr/bin/env bash
# Re-runs the retrieval half of the original results on this machine.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
R=results/retrieval
echo "== $(date) paper pipeline, test"
python scripts/retrieval_harness.py --split test --ranker bm25-monot5 \
  --rankings-out $R/repro_test_monot5.jsonl --metrics-out repro/test_monot5.metrics.json
echo "== $(date) paper pipeline, dev (all claims, for the classifier export)"
python scripts/retrieval_harness.py --split dev --ranker bm25-monot5 --rank-all \
  --rankings-out $R/repro_dev_monot5.jsonl --metrics-out repro/dev_monot5.metrics.json
echo "== $(date) train cross-encoder (README recipe), predict dev"
python scripts/train_reranker.py --epochs 1 --negatives 4 --groups-per-batch 4 \
  --out checkpoints/ce_repro --predict dev --rankings-out $R/repro_dev_ce.jsonl
python scripts/retrieval_harness.py --split dev --ranker file \
  --rankings-in $R/repro_dev_ce.jsonl --metrics-out repro/dev_ce.metrics.json
echo "== $(date) cross-encoder, test"
python scripts/train_reranker.py --no-train --init-from checkpoints/ce_repro \
  --predict test --rankings-out $R/repro_test_ce.jsonl
python scripts/retrieval_harness.py --split test --ranker file \
  --rankings-in $R/repro_test_ce.jsonl --metrics-out repro/test_ce.metrics.json
echo "== $(date) done"
