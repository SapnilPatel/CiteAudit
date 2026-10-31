# Trained classifier, first run (BiomedBERT, top-20 paper evidence)

`scripts/train_classifier.py` with default settings, trained on Colab (T4, `--amp`) via
`notebooks/train_classifier_colab.ipynb`: BiomedBERT-base, citing sentence plus the top-20
BM25+MonoT5 sentences packed into 512-token windows, [CLS] vectors mean+max pooled,
sqrt class weights, 4 epochs, lr 2e-5, seed 0. Best epoch chosen on dev macro-F1 (epoch 4).
About 2.7 minutes an epoch on a T4; the training log is in `classification/classifier/config.json`.

| Epoch | Train loss | Dev macro-F1 | Dev accuracy |
|---|---|---|---|
| 1 | 0.979 | 0.414 | 0.649 |
| 2 | 0.775 | 0.465 | 0.633 |
| 3 | 0.479 | 0.550 | 0.677 |
| 4 | 0.224 | 0.554 | 0.684 |

## Test split, 606 citations

| Test, top 20 | Accuracy | Macro-F1 | ACCURATE F1 | NOT ACCURATE F1 | NEI F1 |
|---|---|---|---|---|---|
| Paper, published | 0.59 | 0.52 | 0.69 | 0.43 | 0.42 |
| Released MultiVerS, paper's evidence | 0.584 | 0.524 | 0.692 | 0.426 | 0.455 |
| **This classifier, paper's evidence** | **0.649** | 0.487 | **0.779** | 0.384 | 0.299 |
| This classifier, our reranker's evidence | 0.650 | 0.483 | 0.776 | 0.401 | 0.273 |

- **Accuracy beats the paper**: 0.649 against MultiVerS's 0.584 on the same evidence, McNemar
  p = 0.009 (only this classifier right on 125 citations, only MultiVerS on 86).
- **Macro-F1 does not**: 0.487 against 0.524. The gain is all in the majority ACCURATE class;
  NOT ENOUGH INFO recall is 0.20 (10 of 50) and NOT ACCURATE F1 drops from 0.426 to 0.384.
- **Our reranker's evidence changes nothing** under this classifier either (macro-F1 0.483 vs
  0.487, McNemar p = 1.000). It was trained on the paper's evidence, so this is not yet the
  test of training on our evidence.
- **Dev-tuned decision thresholds do not transfer**: per-class log-probability offsets chosen
  on dev raise dev macro-F1 from 0.554 to 0.558 but lower test to 0.464. Dev has only 24 NEI
  citations, too few to tune on.

Train loss falls to 0.22 while dev macro-F1 flattens, so the model is starting to overfit the
majority class. The next runs target the minority classes rather than accuracy.
