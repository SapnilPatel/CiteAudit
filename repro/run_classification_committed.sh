#!/usr/bin/env bash
# Runs the released checkpoint on the original committed exports (data_train/target) for the arms
# whose rebuilt export differs, i.e. those built from the retrained reranker. This checks the
# original classification numbers directly, separate from reranker retraining variance.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
C=results/classification/repro_committed; mkdir -p $C
for arm in ours_20 oracle_20 random_20 ours_10 oracle_10 random_10; do
  echo "== $(date) committed $arm"
  python scripts/predict_multivers.py --checkpoint checkpoints/drive/bestModel.ckpt \
    --data-dir data_train/target/$arm --split dev --out $C/$arm.jsonl --resume
done
echo "== $(date) done"
