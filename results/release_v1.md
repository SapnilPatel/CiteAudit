# Release classifier v1: which checkpoint ships, and how it scores

`notebooks/release_model_colab.ipynb` retrained the NLI classifier (the v5 recipe in
[`nli_classifier_v1.md`](nli_classifier_v1.md)) on all of train with seeds 0, 1 and 2, averaged
the three into a model soup, and picked one checkpoint to publish by a rule fixed before any test
number was seen: **the soup if its dev macro-F1 is at least the seeds' mean, else the seed with the
best dev macro-F1.** Configs, dev and test predictions: [`classification/release/`](classification/release/).

## Dev (316 citations): the selection

| Checkpoint | Dev macro-F1 |
|---|---|
| seed 0 | 0.586 |
| **seed 1** | **0.616** |
| seed 2 | 0.615 |
| soup of the three | 0.600 (seed mean 0.606) |

The soup fell 0.006 short of the seed mean, so the rule picked **seed 1**.

## Test (606 citations): reported, not used for selection

| Checkpoint | Accuracy | Macro-F1 | ACCURATE F1 | NOT ACCURATE F1 | NEI F1 |
|---|---|---|---|---|---|
| Paper system, reproduced | 0.584 | 0.524 | 0.692 | 0.426 | 0.455 |
| seed 0 | 0.680 | 0.549 | 0.802 | 0.415 | 0.430 |
| **seed 1 (released)** | **0.662** | **0.546** | **0.785** | **0.394** | **0.458** |
| seed 2 | 0.690 | 0.582 | 0.801 | 0.450 | 0.494 |
| soup | 0.678 | 0.559 | 0.800 | 0.406 | 0.472 |
| mean ± sd of 3 seeds | 0.677 ± 0.014 | 0.559 ± 0.020 | 0.796 | 0.420 | 0.461 |
| probability average of 3 seeds | 0.691 | 0.577 | 0.804 | 0.442 | 0.484 |

Against the reproduced paper system on the same 606 citations (McNemar exact test on accuracy,
paired bootstrap with 2000 resamples on macro-F1):

| Checkpoint | Right only here | Right only paper | McNemar p | Δ macro-F1 [95% CI] |
|---|---|---|---|---|
| **seed 1 (released)** | 131 | 84 | 0.002 | +0.021 [−0.038, +0.082] |
| seed 0 | 137 | 79 | 0.0001 | +0.025 [−0.038, +0.086] |
| seed 2 | 140 | 76 | 0.00002 | +0.058 [−0.011, +0.123] |
| soup | 138 | 81 | 0.0001 | +0.035 [−0.028, +0.096] |

## What this says

- **The released model beats the paper's system on accuracy, significantly** (+7.8 points,
  p = 0.002), and on macro-F1 by a margin that is not conclusive.
- **Dev selection picked the weakest test checkpoint.** Seeds 1 and 2 differ by 0.001 on dev and
  by 0.036 macro-F1 on test. With 316 dev citations, dev differences below about 0.03 are noise;
  the rule was set in advance and is kept, because swapping now would select on test.
- **Seed variance is as large as every method gain we tested** (sd 0.020 macro-F1 on test). Single
  numbers from one seed should not be compared; the 3-seed mean and the averaged ensemble are the
  stable figures. The averaged ensemble (0.691 / 0.577) matches v5's (0.680 / 0.575), so the recipe
  reproduced.
- **For users who can afford 3× the inference cost,** averaging the three seeds is the stronger,
  steadier option. It is not the default because the CLI loads one model.
