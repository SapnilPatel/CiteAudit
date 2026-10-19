# CSCI 544 Group 23 — Week 1–2 continuation

## Reproducibility status (important)

This package contains **working evaluation code**, a pure-Python BM25 retrieval baseline, optional MonoT5 reranking, and a gold-vs-retrieved result comparator. It **does not claim to have reproduced** Sarol et al.'s published experiment. The official dataset and trained checkpoint were not bundled and the creation environment has no outbound network access. Do not submit the paper's numbers as your own experimental results.

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

## MultiVerS classification: strict reproduction

In the cloned authors' repository, use its precise `train_target.py`, `multivers/predict.py`, and published fine-tuned checkpoint per their README. The repository's example inference invocation is skeletal, so inspect `python multivers/predict.py --help` and the upstream implementation to obtain valid CLI flag names (do not treat the placeholder braces as flags). Generate one prediction file using BM25+MonoT5 top-20 evidence and a second using annotated gold evidence, with identical IDs and gold labels. Export both as `id,gold,prediction` CSV or JSONL.

```bash
python scripts/week2_compare.py --retrieved results/top20_predictions.csv --oracle results/oracle_predictions.csv --out results/week2_verified_comparison.json
python scripts/evaluate_classification.py --input results/top20_predictions.csv
```

**Do not conflate retrieval gold evidence with citation label ground truth.** No gold oracle metrics exist until the released MultiVerS checkpoint has actually predicted on gold passages. Preserve the train/dev/test splits and do not fit/tune on test.

## Completion criteria

- [ ] Official train/dev/test citation datasets and document corpus verified, including label distribution
- [ ] Authors' BM25+MonoT5 notebook run and Recall@1/5/10/20, MRR recorded
- [ ] Released MultiVerS checkpoint loads and generates top-20 test predictions
- [ ] Three-way micro/macro/per-class F1 measured, with evaluation citation IDs saved
- [ ] Gold-evidence oracle run on **same** test citation IDs
- [ ] Comparisons and discrepancies documented against paper targets
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

```bash
# pilot: 1 epoch, fewer negatives, then rerank dev
python scripts/train_reranker.py --epochs 1 --negatives 4 \
  --out checkpoints/reranker_pilot \
  --predict dev --rankings-out results/retrieval/dev_ce_pilot.jsonl
python scripts/retrieval_harness.py --split dev --ranker file \
  --rankings-in results/retrieval/dev_ce_pilot.jsonl

# full run on a CUDA GPU: AMP and a larger batch both pay off there
python scripts/train_reranker.py --epochs 2 --negatives 7 --groups-per-batch 8 --amp \
  --out checkpoints/reranker \
  --predict dev --rankings-out results/retrieval/dev_ce.jsonl
```

Targets to beat (zero-shot MonoT5 at the same depth with context, paper's metrics):
dev R@20 0.585 / AllGold@20 0.447; test R@20 0.575 (the paper's own pipeline scores 0.529).

Throughput measured on an M2 Air (MPS): 8.8 pairs/s, so 2 epochs over the train split is about
2.2 h and reranking dev's top-60 about 15 min. A desktop CUDA GPU is roughly 10-15x faster with
`--amp`, which also makes full-pool inference (`--depth 1000`) practical.

### Attribution of the retrieval gain so far (dev, paired bootstrap over articles)

| step | delta in pooled R@20 | status |
|---|---|---|
| paper pipeline 0.534 → MonoT5 + context 0.585 | +0.050 [+0.026,+0.073] | isolated: one variable |
| MonoT5 + context → fine-tuned cross-encoder 0.647 | +0.062 [+0.037,+0.090] | still a bundle |
| of which: the `[C]`/`[/C]` centre markers | +0.003 [-0.010,+0.023] | **no effect, dropped** |

The bundle still mixes the model swap (MonoT5-base trained on MS MARCO vs PubMedBERT), in-domain
fine-tuning, the listwise objective, and the negative sampling range. `scripts/run_ablations.sh`
separates them one variable at a time:

```bash
bash scripts/run_ablations.sh                                      # ~40 min per run on M2 MPS
EXTRA="--amp --groups-per-batch 8 --lr 3e-5" bash scripts/run_ablations.sh    # ~5 min per run on CUDA
```

It skips runs that already have a metrics file, and ends with paired comparisons against the
baseline recipe. **Compare runs with `scripts/compare_runs.py`, not by eyeballing two independent
confidence intervals**: each interval carries the variance of article difficulty, which both systems
share, so on 10-20 articles they overlap even when one system wins on 9 of 10 articles.
