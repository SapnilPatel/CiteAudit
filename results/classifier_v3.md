# Classifier v3: nine labels, and the perfect-evidence ceiling

`notebooks/classifier_v3_colab.ipynb` on a Colab T4. BiomedBERT (`scripts/train_classifier.py`),
`--class-weight sqrt`, 4 epochs. Predictions and logs in `classification/v3/`.

## E. Nine annotation labels against three (our reranker's evidence, 3 seeds each)

| | Dev macro-F1 | Test accuracy | Test macro-F1 | ACCURATE F1 | NOT ACCURATE F1 | NEI F1 |
|---|---|---|---|---|---|---|
| E3, `--labels 3` | 0.472 ± 0.018 | 0.662 ± 0.018 | 0.489 ± 0.013 | 0.789 | 0.438 | 0.239 |
| E9, `--labels 9` | 0.491 ± 0.055 | 0.644 ± 0.023 | 0.487 ± 0.027 | 0.767 | 0.423 | 0.272 |

**Training on the nine labels does not help on its own**: test macro-F1 0.487 against 0.489, well
inside seed noise. It nudges NOT ENOUGH INFO up (0.272 against 0.239) and ACCURATE down.

## C. Perfect-evidence ceiling (1 seed each)

Trained and tested on top-20 documents that contain every gold evidence sentence (`oracle_20_*`):

| | Test, perfect evidence: accuracy | macro-F1 | Test, our evidence: macro-F1 |
|---|---|---|---|
| C3 | 0.630 | 0.436 | 0.422 |
| C9 | 0.531 | 0.359 | 0.363 |

**Perfect evidence does not raise this classifier's ceiling**: with every gold sentence in its
input it scores below the same model on our reranker's evidence (0.436 against 0.489 for E3),
and C9 never predicts NOT ENOUGH INFO. These are single seeds, so the exact gap is uncertain, but
it agrees with the frozen MultiVerS on dev (perfect evidence 0.656 against 0.684 for retrieved)
and with `classifier_v2.md`. Across three models the conclusion is the same: **retrieval is no
longer the bottleneck; the classifier's reading of the evidence is.** Further retrieval gains are
unlikely to move classification.

## Ensembles (weight chosen on dev)

| MultiVerS + | Dev macro-F1 |
|---|---|
| B, 3-label inverse (`ensemble_v1.md`) | 0.698 |
| E3 | 0.712 |
| **E9** | **0.726** |
| B + E3 + E9 together | 0.706 |

MultiVerS + E9, chosen on dev (w = 0.5) and run once on test:

| Test | Accuracy | Macro-F1 | ACCURATE | NOT ACCURATE | NEI |
|---|---|---|---|---|---|
| Reproduced paper system | 0.584 | 0.524 | 0.692 | 0.426 | 0.455 |
| Previous best (MultiVerS + B) | 0.627 | 0.536 | 0.749 | 0.435 | 0.426 |
| **MultiVerS + E9** | **0.642** | **0.538** | 0.764 | 0.426 | 0.424 |

Accuracy is significantly better than the paper system (only the ensemble right 90, only the
paper system 55, McNemar p = 0.005); macro-F1 +0.014, 95% paired bootstrap CI [−0.032, +0.058].

**Most of the dev gain did not transfer**: +0.028 on dev became +0.002 macro-F1 on test over the
previous ensemble. Dev has 24 NOT ENOUGH INFO and 8 OVERSIMPLIFY citations, so choosing between
models on dev differences of this size mostly picks noise. Future model selection should use
cross-validation over the 2141 training citations instead.
