#!/usr/bin/env bash
# Re-runs the classification half: rebuild the original 8 dev exports from our own rankings,
# diff them against the committed ones, run the released MultiVerS checkpoint on each, and also on
# the authors' own released classifier input (Drive folder scifact_10, dev and test).
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
R=results/retrieval; E=repro/exports; C=results/classification/repro
mkdir -p $E $C
CK=checkpoints/drive/bestModel.ckpt
for k in 10 20; do
  python scripts/export_for_multivers.py --rankings $R/repro_dev_monot5.jsonl --split dev --topk $k --out $E/paper_$k
  python scripts/export_for_multivers.py --rankings $R/repro_dev_ce.jsonl --split dev --topk $k --out $E/ours_$k
  python scripts/export_for_multivers.py --rankings $R/repro_dev_ce.jsonl --split dev --topk $k --oracle --out $E/oracle_$k
  python scripts/export_for_multivers.py --rankings $R/repro_dev_ce.jsonl --split dev --topk $k --random --out $E/random_$k
done
echo "== export diff against the committed exports"
for arm in paper_20 ours_20 oracle_20 random_20 paper_10 ours_10 oracle_10 random_10; do
  for f in claims_dev.jsonl corpus.jsonl; do
    if cmp -s $E/$arm/$f data_train/target/$arm/$f; then echo "$arm/$f identical"; else echo "$arm/$f DIFFERS"; fi
  done
done
echo "== $(date) authors' released input"
for s in dev test; do
  python scripts/predict_multivers.py --checkpoint $CK --data-dir checkpoints/drive/scifact_10 --split $s \
    --out $C/authors_input_$s.jsonl --resume
done
for arm in paper_20 ours_20 oracle_20 random_20 paper_10 ours_10 oracle_10 random_10; do
  echo "== $(date) $arm"
  python scripts/predict_multivers.py --checkpoint $CK --data-dir $E/$arm --split dev --out $C/$arm.jsonl --resume
done
echo "== $(date) done"
