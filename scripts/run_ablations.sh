#!/usr/bin/env bash
# One variable changed per run, everything else identical to the baseline recipe, so each
# delta is attributable. Compare with scripts/compare_runs.py (paired over articles), never
# by eyeballing two independent confidence intervals.
#
# On an M2 Air each run is ~40 min; on a CUDA GPU with --amp roughly 5 min. Add
# EXTRA="--amp --groups-per-batch 8 --lr 3e-5" there.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
SPLIT=${SPLIT:-dev}
EXTRA=${EXTRA:-}
BASE="--epochs 1 --negatives 4 --groups-per-batch 4 --seed 0 --log-every 100"
export TOKENIZERS_PARALLELISM=false

run () {                     # run <name> <extra flags...>
  local name=$1; shift
  local rk=results/retrieval/${SPLIT}_ce_${name}.jsonl
  if [ -f "results/retrieval/${SPLIT}_ce_${name}.metrics.json" ]; then
    echo "== $name already scored, skipping"; return
  fi
  echo "== $name  $* $(date +%H:%M)"
  $PY scripts/train_reranker.py $BASE $EXTRA "$@" \
      --out "checkpoints/ce_${name}" --predict "$SPLIT" --rankings-out "$rk" \
      2>&1 | grep --line-buffered -viE "^warn|warning|UNEXPECTED|MISSING|^[[:space:]]*$"
  $PY scripts/retrieval_harness.py --split "$SPLIT" --ranker file --rankings-in "$rk" \
      --metrics-out "results/retrieval/${SPLIT}_ce_${name}.metrics.json" | head -3
}

run pilot                                   # the recipe itself
run marked    --center-markers              # measured: no effect (+0.003 R@20 on dev)
run noctx     --no-context                  # does a trained model still need neighbours?
run neg0      --neg-from 0                  # are the top BM25 non-golds false negatives?
run bce       --loss bce                    # is in-group competition what helps?
run bertbase  --model bert-base-uncased     # how much comes from biomedical pretraining?

echo; echo "=== paired comparisons against the pilot ==="
for v in nomark noctx neg0 bce bertbase; do
  for m in Recall@20 MRR; do
    $PY scripts/compare_runs.py "results/retrieval/${SPLIT}_ce_pilot.jsonl" \
        "results/retrieval/${SPLIT}_ce_${v}.jsonl" --metric $m \
        | grep -E "delta" | sed "s/^/  pilot - ${v}  ${m}:  /"
  done
done
