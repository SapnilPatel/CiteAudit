#!/usr/bin/env bash
# Re-verify the attribution on 70 reference articles instead of dev's 10.
#
# The bootstrap resamples ARTICLES, so the width of every CI reported so far is set by dev having
# only 10 of them. Cross-validating over the 70 training articles widens the evaluation base 7x
# (CIs ~2.6x tighter) and leaves the test split untouched.
#
# Folds are cut by article, never by citance: every citance of one reference ranks sentences of
# that same reference, so splitting within an article would let train and eval share the text
# being ranked. train_reranker.py --folds does the cut; it refuses a --predict that names a
# different split than --train-split, which would silently un-hold-out the eval set.
#
# Each arm's 5 fold outputs are concatenated into one rankings file covering all 2141 train
# citances, which the harness then scores exactly like any other run.
set -u
PY=${PY:-python3}
K=${K:-5}
BASE="--epochs 1 --negatives 4 --groups-per-batch 4 --seed 0 --log-every 200"
EXTRA=${EXTRA:-}
OUT=results/retrieval/cv
mkdir -p "$OUT"

arm () {
  name=$1; shift
  for k in $(seq 0 $((K - 1))); do
    f="$OUT/${name}_f${k}.jsonl"
    if [ -f "$f" ]; then echo "== $name fold $k done, skipping"; continue; fi
    echo "== $name fold $k/$K  $*  $(date +%H:%M)"
    $PY scripts/train_reranker.py $BASE $EXTRA "$@" \
        --folds "$K" --fold "$k" --train-split train \
        --out "checkpoints/cv_${name}_f${k}" \
        --predict train --rankings-out "$f" || return
    rm -rf "checkpoints/cv_${name}_f${k}"   # ~440 MB each; the rankings are what we keep
  done
  all="results/retrieval/train_cv_${name}.jsonl"
  cat "$OUT/${name}_f"*.jsonl > "$all"
  $PY scripts/retrieval_harness.py --split train --ranker file --rankings-in "$all" \
      --metrics-out "results/retrieval/train_cv_${name}.metrics.json"
}

arm pilot                                                  # the recipe, as control
arm noctx    --no-context                                  # the +0.050 context claim
arm bertbase --model bert-base-uncased                     # the +0.039 PubMedBERT claim

echo; echo "=== paired comparisons over 70 articles (delta = arm - pilot) ==="
for v in noctx bertbase; do
  f="results/retrieval/train_cv_${v}.jsonl"
  [ -f "$f" ] || { echo "  ${v}: not finished, skipped"; continue; }
  for m in Recall@20 AllGold@20 MRR; do
    $PY scripts/compare_runs.py "$f" "results/retrieval/train_cv_pilot.jsonl" --metric $m \
        | grep -E "delta" | sed "s/^/  ${v} - pilot  ${m}:  /"
  done
done
