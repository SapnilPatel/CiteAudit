# Trained classifier, v2: inverse class weights, 3 seeds, our reranker's evidence

`notebooks/classifier_v2_colab.ipynb` on a Colab T4. Same model as v1 (`results/classifier_v1.md`)
with `--class-weight inverse`, 3 seeds per group:

- **A**: trained on the paper's BM25+MonoT5 top-20 evidence (`paper_20_train`).
- **B**: trained on our reranker's top-20, with the training split ranked by 3-fold
  cross-validated rerankers cut by article. Those CV rankings score Recall@20 0.631 on train,
  in line with the full reranker on test (0.627 to 0.638), so the training evidence is as good
  as what the classifier sees at test time, not better.

Predictions, training logs and CV metrics are in `classification/v2/`.

## Test split, 606 citations, mean ± sd over 3 seeds

| Trained on | Test evidence | Accuracy | Macro-F1 | ACCURATE F1 | NOT ACCURATE F1 | NEI F1 |
|---|---|---|---|---|---|---|
| *Paper, published* | *paper's* | *0.59* | *0.52* | *0.69* | *0.43* | *0.42* |
| *Released MultiVerS* | *paper's* | *0.584* | *0.524* | *0.692* | *0.426* | *0.455* |
| A, paper's evidence | paper's | 0.630 ± 0.036 | 0.466 ± 0.026 | 0.758 | 0.449 | 0.190 |
| A, paper's evidence | ours | 0.623 ± 0.024 | 0.467 ± 0.025 | 0.750 | 0.423 | 0.226 |
| B, our evidence | paper's | 0.638 ± 0.007 | 0.478 ± 0.018 | 0.769 | 0.430 | 0.236 |
| **B, our evidence** | **ours** | **0.656 ± 0.017** | **0.499 ± 0.037** | **0.780** | **0.450** | **0.268** |

B on our evidence against the released MultiVerS on the paper's evidence, per seed:

| Seed | Macro-F1 | Only B right / only MultiVerS right | McNemar p |
|---|---|---|---|
| 0 | 0.464 | 133 / 88 | 0.003 |
| 1 | **0.538** | 132 / 79 | < 0.001 |
| 2 | 0.495 | 121 / 88 | 0.027 |

## What this shows

- **Accuracy beats the paper reliably.** Every seed of B is significantly more accurate than
  MultiVerS (0.656 mean against 0.584), driven by ACCURATE F1 0.78 against 0.69.
- **Macro-F1 does not, on average.** 0.499 ± 0.037 against 0.524; one seed of three exceeds it.
  NOT ACCURATE F1 now matches the paper (0.450 vs 0.43), so the whole gap is NOT ENOUGH INFO:
  F1 0.27 against MultiVerS's 0.455.
- **Inverse class weights did not fix NOT ENOUGH INFO** and made results noisier: group A fell
  from v1's 0.487 (sqrt weights, one seed) to 0.466 ± 0.026.
- **Training on our evidence is the best setting** (0.499 against 0.466 for A on the paper's
  evidence), the first sign that the better retriever helps once the classifier is trained on
  its output. The difference is about one seed standard deviation, so it is suggestive, not
  established.
- **Seed variance is large**: macro-F1 sd up to 0.037, and NEI F1 sd up to 0.076, because test
  has only 50 NEI citations. Single-seed results, including v1's, should not be over-read.
- Dev macro-F1 was still rising at the last epoch in 5 of 6 runs, so 4 epochs may be too few.

The likely reason NOT ENOUGH INFO stays low: MultiVerS was pretrained on fact verification, where
"not enough info" is a core class, while BiomedBERT starts from plain language modelling and sees
only 143 NEI training citations. That points to fine-tuning MultiVerS itself on our evidence.
