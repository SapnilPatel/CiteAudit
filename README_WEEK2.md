# CSCI 544 Group 23 — Week 1–2 continuation

## Reproducibility status (important)

**Retrieval stage: reproduced.** On the held-out test split the paper's pipeline (BM25 top-60 +
MonoT5) scores R@20 **0.529** / MRR **0.323** against its published 0.53 / 0.32 (§1, §11). Our
cross-encoder reranker reaches R@20 **0.638** on the same split, +0.109 [+0.084,+0.131] over the
paper's pipeline, winning 17 of 20 articles (§11).

**Classification stage: the released checkpoint runs, but the published figures are not
reproduced.** The authors' `predict.py` cannot run as shipped and their training-time input
construction cannot be reconstructed from the released data (§9), so absolute F1 here is only
loosely comparable. Swapping in the better retriever under the frozen checkpoint changes nothing at
any evidence-quality level up to perfect evidence (§10); whether the retrieval gain is worth
anything to the task needs a retraining run that has not been done.

Do not submit the paper's numbers as your own experimental results. Every figure in
`results/week2_retrieval_findings.md` is measured here, and the sections flag their own
corrections where a later measurement overturned an earlier reading.

**Paper's actual pipeline:** citance → BM25 → MonoT5 → top 20 → *fine-tuned* MultiVerS. Pure BM25 is an ablation and **is not** the paper's final retrieval baseline.

## Week 1: retrieve the official data and checkpoint (networked machine)

```bash
bash bootstrap.sh
python scripts/locate_upstream.py
python scripts/inspect_dataset.py upstream/Citation-Integrity
```

The official [author README](https://github.com/ScienceNLP-Lab/Citation-Integrity) says the splits contain 70 training, 10 development and 20 test reference articles and that data use the SciFact JSONL schema. **Inspect files before choosing inputs**—some paper assets or checkpoints may need downloading separately. The released checkpoint is linked from the author README, via Google Drive. Follow the repository setup/inference instructions directly to preserve paper preprocessing and model settings.

## Week 2: retrieval sanity check on authors' SciFact-style exported inputs

```bash
python scripts/week2_pipeline.py --claims path/to/claims_test.jsonl --corpus path/to/corpus.jsonl --out results/test_bm25.jsonl --mode bm25
python scripts/evaluate_retrieval.py results/test_bm25.jsonl
```

The retriever uses each claim's single cited document: its `doc_id` or, as a fallback, the sole evidence document ID. Rows lacking an unambiguous cited document are excluded with counts. **Check missing document count**; citations with no gold evidence are excluded from retrieval recall but are still exported.

Optional experimental MonoT5 stage (model download + substantial compute required):

```bash
pip install transformers sentencepiece torch
python scripts/week2_pipeline.py --claims path/to/claims_test.jsonl --corpus path/to/corpus.jsonl --out results/test_monot5.jsonl --mode bm25-monot5
python scripts/evaluate_retrieval.py results/test_monot5.jsonl
```

The MonoT5 model default is a general reranker; it is **not verified as the exact authors' notebook weights or preprocessing**. For a strict paper reproduction use `bm25t5.ipynb` from the authors' repo. This helper provides an independent sanity-check/ablation.

## Retrieval harness for the official multivers-format data

`week2_pipeline.py` assumes one abstract per claim and does not work on `Data/multivers-format/` (claims list `cited_doc_ids`, a contiguous range of chunks forming the reference article). Use `scripts/retrieval_harness.py` instead. It ranks every sentence of the cited article and reports sentence-level Recall@k (the paper's definition), Hit@k, AllGold@k, and MRR, with 95% CIs from an article-level bootstrap.

```bash
python scripts/retrieval_harness.py --split test --rankings-out results/retrieval/test_bm25.jsonl
python scripts/retrieval_harness.py --split test --ranker bm25-monot5      # needs torch transformers sentencepiece protobuf; ~30 min on MPS
python scripts/retrieval_harness.py --split test --ranker file --rankings-in my_reranker.jsonl   # {id, ranked:["doc_id:sent_idx", ...]}
```

## MultiVerS classification

The authors' `multivers/predict.py` **cannot run as shipped** — it opens `../train_configs.json`,
which is not in the repository, and it builds its dataloader from a hardcoded directory rather than
from `--input_file`/`--corpus_file`. `results/week2_retrieval_findings.md` §9 has the details.
`scripts/predict_multivers.py` reimplements the label path against the released checkpoint instead:

The exporter needs the **aligned annotations** first (see above): the 3-way gold label cannot be
derived from `claims-*.jsonl`, which stores a label only inside `evidence[...]["label"]` and so
carries none at all for citances with no gold evidence sentences. Those are not one class — dev's
61 are 37 `ETIQUETTE` (→ NOT_ACCURATE) and 24 `IRRELEVANT` (→ NEI) — so treating "no evidence" as
NEI mislabels 13.6% of dev. §9 has the full mapping and the check that confirms it.

```bash
python scripts/align_annotations.py --split dev --out results/annotations/dev.jsonl   # prerequisite

# one export per arm; --rank-all is required for the rankings file, see below
python scripts/export_for_multivers.py --rankings results/retrieval/dev_ce_pilot.jsonl \
  --split dev --topk 20 --out data_train/target/ours_20
python scripts/export_for_multivers.py --rankings results/retrieval/dev_ce_pilot.jsonl \
  --split dev --topk 20 --oracle --out data_train/target/oracle_20   # perfect evidence, same length
python scripts/export_for_multivers.py --rankings results/retrieval/dev_ce_pilot.jsonl \
  --split dev --topk 20 --random --out data_train/target/random_20   # lower anchor

python scripts/predict_multivers.py --data-dir data_train/target/ours_20 --split dev \
  --out results/classification/ours_20.jsonl          # ~10 min on M2 MPS, resumable
python scripts/score_multivers.py results/classification/ours_20.jsonl \
  results/classification/paper_20.jsonl --gold-from data_train/target/ours_20
```

Pass `--gold-from` so gold comes from the export rather than from the labels frozen into the
prediction files — predictions depend only on the document, so a label-mapping fix needs no
re-running of the model, but it must be applied through one path for both arms.

Each claim becomes one synthetic document holding its top-k retrieved sentences, so there is one
instance per claim. Rewriting `evidence` over the original corpus instead gives **24797 instances
for dev's 316 claims** — a claim's `cited_doc_ids` is the reference article in ~79 paragraph chunks
— with no defined way to aggregate ~78 predictions per claim.

**The rankings file must cover every claim.** The harness skips the 61 dev citances with no gold
evidence because they cannot be scored: exporting from a 255-row file gives them an empty document
in one arm and a full one in the other, making the label partly readable off the input length. Pass
`--rank-all` (with `--resume`, only the missing rows are computed and no metric changes).

The oracle and random arms are what make the result interpretable rather than just null — §10. Keep
§9's caveat in view: the authors' training-time input construction cannot be reconstructed from the
released data, so absolute F1 is only loosely comparable to the paper's and the A/B between arms is
what carries weight.

For reference, the older gold-vs-retrieved comparison path:

```bash
python scripts/week2_compare.py --retrieved results/top20_predictions.csv --oracle results/oracle_predictions.csv --out results/week2_verified_comparison.json
python scripts/evaluate_classification.py --input results/top20_predictions.csv
```

**Do not conflate retrieval gold evidence with citation label ground truth.** No gold oracle metrics exist until the released MultiVerS checkpoint has actually predicted on gold passages. Preserve the train/dev/test splits and do not fit/tune on test.

## Completion criteria

- [x] Official train/dev/test citation datasets and document corpus verified, including label distribution (§6: nine fine-grained labels, not eight; 3-way collapse is undocumented, §9)
- [x] Authors' BM25+MonoT5 pipeline reproduced and Recall@1/5/10/20, MRR recorded — test R@20 0.529 against the paper's 0.53 (§1)
- [x] Released MultiVerS checkpoint loads and generates dev predictions (`scripts/predict_multivers.py`; their `predict.py` cannot run, §9)
- [x] Three-way micro/macro/per-class F1 measured on dev, citation IDs saved in `results/classification/*.jsonl`
- [x] Gold-evidence oracle run on the **same** citation IDs, plus a random-evidence lower anchor (§10)
- [x] Comparisons and discrepancies documented against paper targets (§1, §8, §9, §10, §11)
- [x] Single held-out test measurement of the retrieval stage: R@20 0.638 vs the paper pipeline's 0.529 (§11)
- [ ] MultiVerS **retrained** on each arm's retrieval output — the open experiment §10 turns on; needs a CUDA GPU
- [x] Working independent BM25 retrieval/evaluation script and output exports
- [x] Optional MonoT5 reranker runner for experimentation
- [x] Fail-closed, ID-aligned comparison tool for gold versus retrieved runs
- [x] Offline end-to-end synthetic test of retrieval and comparison utilities

Paper: Sarol et al., *Bioinformatics* 40(7), btae420 (2024), https://doi.org/10.1093/bioinformatics/btae420 .

## Aligning the annotations to claim ids

`Data/annotations.zip` holds the per-citation records the multivers-format files drop: the citing
paragraph, citation-context spans, gold evidence spans, `marker_span`, and the original label. They
are keyed by reference article and filename, not by claim id, so they need an explicit join.

```bash
unzip -q upstream/Citation-Integrity/Data/annotations.zip -d upstream/annotations_extracted -x "__MACOSX/*"
for s in train dev test; do
  python scripts/align_annotations.py --split $s --out results/annotations/$s.jsonl
done
```

The join is structural first (`corpus doc_id = reference * 1000 + paragraph index`, so a claim's
`cited_doc_ids` fix its reference article), then text keys within each reference group. It aligns
all 2141/316/606 records 1:1 and prints which key carried each match; rows that could have been
permuted with another row are flagged `ambiguous: true`. **Do not assume file order matches claim
order** — and do not weaken the key tiers: a gold evidence sentence is shared by several citances of
the same reference, so matching on evidence before citance text produces wrong pairs.

## Training the cross-encoder reranker

`scripts/train_reranker.py` fine-tunes a biomedical cross-encoder to replace the MonoT5 stage. BM25
stays as the first stage. It imports the candidate pool and context window from
`retrieval_harness.py` so training and evaluation rank exactly the same candidates, and it writes
`{id, ranked}` JSONL that the harness scores, so model numbers and baseline numbers come out of the
same metric code.

**The recipe is 1 epoch and 4 negatives** — not a reduced pilot, the final setting. Two epochs with
7 negatives halves the training loss (1.104 → 0.437) and makes dev slightly *worse*
(R@20 -0.007, AllGold@20 -0.009 on 10/10 articles), so the model overfits 4138 training groups
after a single pass. Train longer only if the training set grows.

```bash
python scripts/train_reranker.py --epochs 1 --negatives 4 --groups-per-batch 4 \
  --out checkpoints/ce_pilot \
  --predict dev --rankings-out results/retrieval/dev_ce_pilot.jsonl
python scripts/retrieval_harness.py --split dev --ranker file \
  --rankings-in results/retrieval/dev_ce_pilot.jsonl

# same recipe on a CUDA GPU: AMP and a larger batch both pay off there
python scripts/train_reranker.py --epochs 1 --negatives 4 --groups-per-batch 8 --amp \
  --out checkpoints/ce_pilot --predict dev --rankings-out results/retrieval/dev_ce_pilot.jsonl

# score an existing checkpoint on another split without retraining
python scripts/train_reranker.py --no-train --init-from checkpoints/ce_pilot \
  --predict test --rankings-out results/retrieval/test_ce.jsonl
```

What the recipe reaches, in the paper's pooled metrics:

| split | R@20 | AllGold@20 | MRR | vs the paper's pipeline |
|---|---|---|---|---|
| dev (10 articles) | 0.643 | 0.525 | 0.405 | +0.109 |
| train, 5-fold article CV (70 articles) | 0.560 | 0.429 | 0.369 | +0.073 [+0.055,+0.091], 62/70 |
| **test (20 articles, single measurement)** | **0.638** | **0.493** | **0.365** | **+0.109 [+0.084,+0.131], 17/20** |

The CV figure is lower because each fold trains on 56 articles rather than 70; test reproduces dev
exactly (§8, §11). Reference points on test: BM25 alone 0.444, the paper's pipeline 0.529,
zero-shot MonoT5 with the context window 0.575.

Widening the context window needs `--max-len` raised with it. At the default 192 7% of
query+window pairs are already truncated and 12-15% once the window grows, so a sweep run at 192
measures the truncation rate instead of the window; `scripts/run_window_sweep.sh` runs every arm at
512, where 97.6-99.9% of pairs survive, including a re-run of the current window as the control.

Throughput measured on an M2 Air (MPS): 13.2 pairs/s at `--max-len 192`, so the recipe is about
40 min plus 15 min to rerank dev's top-60; `--max-len 512` is ~2.7x that, since `padding` is
`max_length` (dynamic padding compiles one MPS graph per shape and blows up memory). A desktop CUDA
GPU is roughly 10-15x faster with `--amp`, which also makes full-pool inference (`--depth 1000`)
practical. `--amp` is not worth passing on MPS.

### Attribution of the retrieval gain (70 articles, 5-fold CV, paired bootstrap)

Eight candidate causes have been ablated one variable at a time, and the surviving ones re-measured
on the 70 training articles by article-level 5-fold CV against the paper's own pipeline run over the
same 70. **Three ingredients produce the gain**; `results/week2_retrieval_findings.md` §8 has the
full tables and §7 the dev-only ablations.

| step | R@20 | delta | 95% CI | articles |
|---|---|---|---|---|
| paper pipeline, BM25 + MonoT5 top-60 | 0.487 | — | — | baseline |
| **our pipeline (all three ingredients)** | **0.560** | **+0.073** | **[+0.055,+0.091]** | **62/70** |
| └ contribution of the neighbour-sentence context window | | +0.033 | | |
| └ contribution of fine-tuning the reranker at all | | +0.049 (dev) | [+0.020,+0.070] | 7/10 |
| └ contribution of PubMedBERT over generic BERT | | +0.031 | | |
| swapping the MonoT5 *architecture* for BERT | | -0.007 | [-0.025,+0.010] | **nil** |

| knob that turned out not to matter | delta in R@20 | 95% CI |
|---|---|---|
| `[C]`/`[/C]` markers around the centre sentence | +0.003 | [-0.010,+0.023] |
| listwise in-group softmax vs pointwise BCE | +0.000 | [-0.017,+0.018] |
| sampling negatives from BM25 rank 5 vs rank 0 | -0.007 | [-0.020,+0.005] |
| 2 epochs and 7 negatives vs 1 and 4 | -0.007 | [-0.029,+0.009] |
| the encoder architecture (MonoT5 vs BERT, both fine-tuned) | -0.007 | [-0.025,+0.010] |

The architecture row is a **withdrawn earlier claim**: this table used to credit +0.020 to "a
fine-tuned cross-encoder replacing MonoT5", because the `bertbase` ablation changed the architecture
and the training regime together — its baseline MonoT5 had never been fine-tuned. Fine-tuning MonoT5
itself with the same recipe (`scripts/run_t5_control.sh`) isolates the two: training is worth +0.049
and the architecture swap nothing.

Two findings worth carrying into the write-up: removing the context window from the *trained* model
costs more than the +0.050 it buys zero-shot MonoT5, so fine-tuning amplifies the window rather than
absorbing it; and the entire MRR gain over the paper is the biomedical encoder — a fine-tuned
generic BERT sits at MRR +0.001 [-0.023,+0.027], i.e. exactly the paper's level, while PubMedBERT
reaches +0.062 [+0.037,+0.087]. The context window still costs a little MRR even after fine-tuning
(+0.062 with it vs +0.069 without), so §3's recall/top-rank trade-off is shrunk, not reversed.

Dev's 10 articles inflated every effect by 1.3-1.5x and got the two MRR findings backwards; quote
the 70-article column.

```bash
bash scripts/run_ablations.sh                                      # ~40 min per run on M2 MPS
EXTRA="--amp --groups-per-batch 8 --lr 3e-5" bash scripts/run_ablations.sh    # ~5 min per run on CUDA
bash scripts/run_t5_control.sh    # fine-tunes MonoT5 on the same recipe: isolates architecture
bash scripts/run_cv.sh            # 5-fold article-level CV over the 70 train articles, 3 arms, ~9 h
bash scripts/run_paper_cv.sh      # the paper's own pipeline over the same 70 articles, ~50 min
```

`run_cv.sh` passes `--folds 5 --fold I`; folds are cut **by reference article**, so `--folds` also
requires `--predict` to name `--train-split` — otherwise the held-out articles are computed from one
split while the model trains on another and the eval set is not held out at all. Each fold's
checkpoint is deleted after predicting (440 MB apiece) and the five fold rankings are concatenated
into one file the harness scores.

It skips runs that already have a metrics file, and ends with paired comparisons against the
baseline recipe. **Compare runs with `scripts/compare_runs.py`, not by eyeballing two independent
confidence intervals**: each interval carries the variance of article difficulty, which both systems
share, so on 10-20 articles they overlap even when one system wins on 9 of 10 articles. The script
prints `delta = A - B`, so pass the variant first for the delta to read *variant − baseline*; it
accepts `MRR`, `Recall@k`, `Hit@k` and `AllGold@k` and rejects anything else rather than quietly
answering for a different metric.
