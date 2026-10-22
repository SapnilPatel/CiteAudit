#!/usr/bin/env bash
# How much neighbour context, measured at a --max-len that does not truncate it away.
#
# At --max-len 192 (the default) 7% of query+window pairs are already cut, and 12-15% once the
# window is widened, so a sweep run at 192 measures the truncation rate rather than the window.
# Every arm here runs at 512, where 97.6-99.9% of pairs survive intact, and the control arm is
# re-run at 512 too: max_len must not differ between the thing being compared.
#
# Pilot scale (1 epoch, 4 negatives) like the ablations in run_ablations.sh — only the relative
# ordering of the arms is wanted.
set -u
PY=${PY:-python3}
SPLIT=${SPLIT:-dev}
BASE="--epochs 1 --negatives 4 --groups-per-batch 4 --seed 0 --log-every 100 --max-len 512"
EXTRA=${EXTRA:-}

run () {
  name=$1; shift
  m="results/retrieval/${SPLIT}_ce_${name}.metrics.json"
  if [ -f "$m" ]; then echo "== $name already scored, skipping"; return; fi
  echo "== $name  $*  $(date +%H:%M)"
  $PY scripts/train_reranker.py $BASE $EXTRA "$@" \
      --out "checkpoints/ce_${name}" \
      --predict "$SPLIT" --rankings-out "results/retrieval/${SPLIT}_ce_${name}.jsonl" || return
  $PY scripts/retrieval_harness.py --split "$SPLIT" --ranker file \
      --rankings-in "results/retrieval/${SPLIT}_ce_${name}.jsonl" --metrics-out "$m"
}

run w200ctl --neighbour-chars 200    --neighbours 1   # control: the current recipe at max_len 512
run wfull   --neighbour-chars 100000 --neighbours 1   # does lengthening each neighbour help?
run n2      --neighbour-chars 100000 --neighbours 2   # does taking more neighbours help?

echo; echo "=== paired comparisons against w200ctl (delta = arm - control) ==="
for v in wfull n2; do
  f="results/retrieval/${SPLIT}_ce_${v}.jsonl"
  [ -f "$f" ] || { echo "  ${v}: no rankings file, skipped"; continue; }
  for m in Recall@20 AllGold@20 MRR; do
    $PY scripts/compare_runs.py "$f" "results/retrieval/${SPLIT}_ce_w200ctl.jsonl" --metric $m \
        | grep -E "delta" | sed "s/^/  ${v} - w200ctl  ${m}:  /"
  done
done
