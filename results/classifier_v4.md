# Classifier v4: citing-paragraph context

`notebooks/classifier_v4_colab.ipynb` on a Colab T4. BiomedBERT, 3 labels, sqrt class weights,
4 epochs, our reranker's evidence; the first segment is the citing sentence alone (none), with its
neighbouring sentences (window), or with its paragraph (paragraph), from `build_context.py`.
Variant chosen by 3-fold cross-validation over the 2141 training citations, cut by article.
Predictions and logs in `classification/v4/`.

## Phase 1: cross-validation over train (2141 out-of-fold predictions per variant)

| Context | Accuracy | Macro-F1 | Per fold | ACCURATE F1 | NOT ACCURATE F1 | NEI F1 |
|---|---|---|---|---|---|---|
| **none** | **0.643** | **0.428** | 0.403, 0.423, 0.465 | 0.758 | 0.454 | 0.072 |
| window | 0.597 | 0.378 | 0.341, 0.415, 0.367 | 0.728 | 0.311 | 0.094 |
| paragraph | 0.557 | 0.358 | 0.371, 0.344, 0.353 | 0.691 | 0.299 | 0.085 |

## Phase 2: window context on all of train, 3 seeds, test

| Test | Accuracy | Macro-F1 | ACCURATE F1 | NOT ACCURATE F1 | NEI F1 |
|---|---|---|---|---|---|
| No context (`classifier_v3.md`, E3) | 0.662 ± 0.018 | 0.489 ± 0.013 | 0.789 | 0.438 | 0.239 |
| Window context | 0.645 ± 0.030 | 0.454 ± 0.032 | 0.771 | 0.365 | 0.226 |

**More context makes this classifier worse, monotonically**, on cross-validation and on test, and
the loss is in NOT ACCURATE (0.454 to 0.299 F1 on CV). The likely mechanism is architectural:
the context takes 112 to 250 of each 512-token window, splits the evidence over more windows
(2.4 to 4.2 per citation), and leaves the model to work out which sentence of the context is the
claim, from 2141 examples. The paper's gain from context used the annotators' exact context
span, which is not available to a deployed tool. This does not show the paragraph is useless to
a model that can read it, only that this way of feeding it does not help.

Two other things the cross-validation shows that dev could not:

- **NEI is barely learned**: out-of-fold NEI F1 is 0.07 to 0.09 for every variant, against 0.24
  on test. With about 95 IRRELEVANT citations per training fold, the model mostly cannot
  recognise the class; the better test figure likely depends on the test articles.
- Fold-to-fold spread is up to 0.06 macro-F1, larger than most of the differences that dev-based
  choices were made on earlier.
