# CSCI 544 Group 23 — Week 1 Baseline Reproduction

This starter repo is for **Week 1 only**: reproduce the baseline from Sarol et al. (Bioinformatics, 2024) before changing the model.

Published pipeline:

`citation sentence (citance) -> BM25 -> MonoT5 reranker -> top 20 evidence sentences -> fine-tuned MultiVerS -> 3-way label`

The official authors' repository is:
- https://github.com/ScienceNLP-Lab/Citation-Integrity

The original MultiVerS repository is:
- https://github.com/dwadden/multivers

## Paper targets for Week 1

For the **citance** retrieval query:
- Recall@1: 0.09
- Recall@5: 0.28
- Recall@10: 0.40
- Recall@20: 0.53
- MRR: 0.32

For citation classification:
- Top-20 retrieved evidence + MultiVerS: **micro-F1 0.59, macro-F1 0.52**
- Gold evidence oracle: **micro-F1 0.73, macro-F1 0.75**
- Gold citation context + gold evidence: **micro-F1 0.75, macro-F1 0.78**

Per-class top-20 F1:
- ACCURATE: 0.69
- NOT_ACCURATE: 0.43
- IRRELEVANT: 0.42

## 3-way label mapping used by the paper

The original 8 labels are collapsed as follows:

- `ACCURATE`, `INDIRECT` -> `ACCURATE` (MultiVerS Support)
- `CONTRADICT`, `NOT_SUBSTANTIATE`, `OVERSIMPLIFY`, `MISQUOTE`, `ETIQUETTE` -> `NOT_ACCURATE` (MultiVerS Refute)
- `IRRELEVANT` -> `IRRELEVANT` (MultiVerS Not Enough Information)

## Fast start

### 1. Create environment

Python 3.8 is safest for the original MultiVerS-era dependencies.

```bash
conda create -n citation-integrity python=3.8 -y
conda activate citation-integrity
pip install -r requirements-local.txt
```

### 2. Download the official repositories

```bash
bash bootstrap.sh
```

This clones both official repos into `upstream/`. Large checkpoints are intentionally **not** bundled in this ZIP.

### 3. Locate the important upstream files

```bash
python scripts/locate_upstream.py
```

You should see paths for things like `bm25t5.ipynb`, `train_target.py`, `claims_train.jsonl`, `claims_dev.jsonl`, `claims_test.jsonl`, and corpus JSONL files if they exist in the upstream checkout.

### 4. Inspect the released dataset

Example:

```bash
python scripts/inspect_dataset.py upstream/Citation-Integrity/Data
```

This recursively finds JSON/JSONL files, counts examples, shows keys, and extracts label distributions where possible.

### 5. Reproduce the authors' retrieval stage

The authors explicitly point to `bm25t5.ipynb` in their repository. Run that notebook first and record Recall@1/5/10/20 and MRR in `results/ours.csv`.

The paper's citance-query targets are listed in `results/paper_targets.csv`.

### 6. Reproduce MultiVerS classification

The official repo README says the training data should live under:

```text
./data_train/target/scifact_10/
```

with files such as:

```text
claims_train.jsonl
claims_dev.jsonl
claims_test.jsonl
```

and trains with:

```bash
CUDA_LAUNCH_BLOCKING=1 TOKENIZERS_PARALLELISM=false \
python train_target.py --dataset scifact_10 --gpus=1
```

The authors also release their best checkpoint. Follow their README/checkpoint link rather than retraining first if the immediate goal is pure reproduction.

### 7. Evaluate any prediction file

If you have a simple CSV containing gold and predicted labels:

```bash
python scripts/evaluate_classification.py \
  --input predictions.csv \
  --gold-column gold \
  --pred-column prediction
```

The evaluator recognizes both citation labels (`ACCURATE`, `NOT_ACCURATE`, `IRRELEVANT`) and MultiVerS names (`SUPPORT`, `REFUTE`, `NOT ENOUGH INFO`).

### 8. Generate a Week-1 comparison table

Edit `results/ours.csv`, then run:

```bash
python scripts/make_results_table.py
```

This writes `results/week1_comparison.md`.

## Definition of done for Week 1

Do **not** require exact floating-point equality. A successful reproduction means:

1. Official dataset loads and split sizes are understood.
2. BM25+MonoT5 retrieval runs and Recall@20 is in the neighborhood of the published value.
3. The released/fine-tuned MultiVerS classifier runs on the test setup.
4. You record top-20 micro/macro F1.
5. You run the gold-evidence oracle or at least prepare it if checkpoint/runtime issues block completion.
6. Every discrepancy is documented (package version, checkpoint, split, preprocessing, GPU, query type).

See `TEAM_TASKS.md` for the six-person Week-1 split.
