# Citation markers in the hypothesis (v9): higher accuracy, same macro-F1, gate not passed

**Question.** The classifier has always read the citing sentence with its citation markers deleted.
298 of 304 ETIQUETTE citations in train cite the paper inside a group of references. Does keeping the
markers as `[CITED]`, `[CITED, with others]` and `[other]` (`train_nli.py --keep-markers`) help?
Both external reviews (`docs/EXTERNAL_REVIEWS.md`) ranked this first, and so did `docs/HANDOFF.md`.

**Setup.** Identical to the current classifier (`nli_classifier_v1.md`): same model, data, 3 folds by
article, seed 0, settings. Only the hypothesis differs. Notebook: `notebooks/nli_markers_colab.ipynb`.
Gate fixed in advance: CV macro-F1 >= 0.580 and NOT ACCURATE F1 not lower, before any test run.
Predictions: [`classification/v9/`](classification/v9/).

## Result (cross-validated, 2141 citations)

| | Accuracy | Macro-F1 | ACC F1 | NOT ACC F1 | NEI F1 |
|---|---|---|---|---|---|
| Current (markers deleted) | 0.676 | 0.570 | 0.783 | 0.513 | 0.415 |
| Markers kept | **0.695** | 0.569 | **0.801** | **0.543** | 0.362 |
| Current, single citations (1020) | 0.715 | 0.551 | 0.832 | 0.374 | 0.446 |
| Markers, single citations | 0.750 | 0.543 | 0.860 | 0.390 | 0.380 |
| Current, grouped citations (1121) | 0.640 | 0.564 | 0.721 | 0.579 | 0.392 |
| Markers, grouped citations | 0.645 | 0.557 | 0.721 | 0.601 | 0.348 |

- Right only with markers 156, right only without 115: **+41 citations, McNemar p = 0.015.**
- Macro-F1 difference −0.001, paired bootstrap 95% CI −0.026 to +0.021.
- ETIQUETTE recalled as NOT ACCURATE: 0.651 → 0.655, **unchanged.** Substantive errors: 0.439 → 0.466.
- Accurate grouped citations wrongly called NOT ACCURATE: 159 → 164.

**Gate: not passed** (macro-F1 0.569 < 0.580), so no test run was made.

## After recalibration the two are the same

Per-class decision biases tuned on the out-of-fold predictions (`tune_bias.py`, same procedure for both):

| | Bias (N_ACC, NEI) | Accuracy | Macro-F1 |
|---|---|---|---|
| Current | −0.3, −0.3 | 0.687 | 0.574 |
| Markers | +0.5, +1.0 | 0.683 | 0.575 |

The accuracy gain is mostly a shift in where the model draws its decision boundaries (more
ACCURATE, fewer NOT ENOUGH INFO), not better separation of the classes. Once both are calibrated
they are indistinguishable.

## What this says

- **The hypothesis behind it was wrong.** The markers were meant to expose ETIQUETTE, and ETIQUETTE
  recall did not move. The model already picks up whatever distinguishes those sentences: 65% of
  ETIQUETTE was flagged before.
- Markers trade NOT ENOUGH INFO for NOT ACCURATE and ACCURATE. Macro-F1 does not change.
- **This is the third independent input change with no gain** (evidence quality, the evidence
  oracle, now markers), after the model and data changes in `docs/HANDOFF.md`. Every
  variant lands at CV macro-F1 0.57 ± 0.01 once calibrated. That pattern points at a ceiling set by
  the labels or by how the task is framed for one classifier, not at missing input.
