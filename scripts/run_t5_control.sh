#!/usr/bin/env bash
# The control the attribution is missing: MonoT5 fine-tuned with OUR recipe.
#
# Everything measured so far compares a fine-tuned cross-encoder against a zero-shot MonoT5, which
# confounds "better architecture" with "trained at all". Fine-tuning MonoT5 on the same data, with
# the same groups, loss, negatives and window, separates them:
#
#   zero-shot MonoT5 + ctx @192  ->  fine-tuned MonoT5 + ctx @192   isolates TRAINING
#   fine-tuned MonoT5 + ctx      ->  fine-tuned bert-base + ctx     isolates ARCHITECTURE
#   fine-tuned bert-base + ctx   ->  fine-tuned PubMedBERT + ctx    isolates DOMAIN PRETRAINING
#
# Every arm runs at --max-len 192, including a fresh zero-shot MonoT5 run: the existing zero-shot
# number was produced at 512, and comparing across lengths would add a second variable. The 512
# run stays as the reference point against the published pipeline.
#
# train_reranker.py scores MonoT5 through the same Scorer at training and inference time, and that
# scorer was verified to reproduce retrieval_harness.py's zero-shot scores bit for bit, so
# fine-tuning moves exactly the quantity the published pipeline ranks by.
set -u
PY=${PY:-python3}
SPLIT=${SPLIT:-dev}
EXTRA=${EXTRA:-}
T5=castorini/monot5-base-med-msmarco

# 1. zero-shot MonoT5 at 192, the matched baseline (inference only)
m="results/retrieval/${SPLIT}_monot5_ctx192.metrics.json"
if [ -f "$m" ]; then
  echo "== zero-shot MonoT5 @192 already scored, skipping"
else
  echo "== zero-shot MonoT5 + ctx @192  $(date +%H:%M)"
  $PY scripts/retrieval_harness.py --split "$SPLIT" --ranker bm25-monot5 --context \
      --monot5-max-len 192 --rerank-depth 60 \
      --rankings-out "results/retrieval/${SPLIT}_monot5_ctx192.jsonl" --metrics-out "$m"
fi

# 2. the same model fine-tuned with the recipe
m="results/retrieval/${SPLIT}_ce_t5ft.metrics.json"
if [ -f "$m" ]; then
  echo "== fine-tuned MonoT5 already scored, skipping"
else
  echo "== fine-tuned MonoT5 + ctx @192  $(date +%H:%M)"
  $PY scripts/train_reranker.py --model "$T5" \
      --epochs 1 --negatives 4 --groups-per-batch 4 --seed 0 --log-every 100 --max-len 192 \
      $EXTRA --out checkpoints/ce_t5ft \
      --predict "$SPLIT" --rankings-out "results/retrieval/${SPLIT}_ce_t5ft.jsonl" \
    && $PY scripts/retrieval_harness.py --split "$SPLIT" --ranker file \
         --rankings-in "results/retrieval/${SPLIT}_ce_t5ft.jsonl" --metrics-out "$m"
fi

echo; echo "=== the decomposition, paired over articles (delta = B-arm - A-arm) ==="
pair () {  # pair <later> <earlier> <label>
  [ -f "results/retrieval/${SPLIT}_$1.jsonl" ] && [ -f "results/retrieval/${SPLIT}_$2.jsonl" ] || {
    echo "  $3: missing a run, skipped"; return; }
  for m in Recall@20 AllGold@20 MRR; do
    $PY scripts/compare_runs.py "results/retrieval/${SPLIT}_$1.jsonl" \
        "results/retrieval/${SPLIT}_$2.jsonl" --metric $m \
        | grep -E "delta" | sed "s/^/  $3  ${m}:  /"
  done
  echo
}
pair ce_t5ft    monot5_ctx192 "training      (MonoT5 ft - MonoT5 zero-shot)"
pair ce_bertbase ce_t5ft      "architecture  (bert-base - MonoT5, both ft)"
pair ce_pilot    ce_bertbase  "domain pretr. (PubMedBERT - bert-base)"
pair ce_pilot    monot5_ctx192 "all three together"
