# Ensemble: released MultiVerS + BiomedBERT trained on our evidence

`scripts/ensemble_classifiers.py` averages class probabilities:
`w · p_MultiVerS + (1 − w) · mean of the 3 BiomedBERT seeds` (group B of `classifier_v2.md`),
both reading top-20 evidence from our reranker. `w` is chosen on dev from a 0.1 grid
(dev macro-F1 0.674 for MultiVerS alone, 0.505 for BiomedBERT alone, **0.698 at w = 0.7**) and
then applied once to test.

Why it should work: the two fail differently. MultiVerS is the only model that handles NOT
ENOUGH INFO, BiomedBERT is much better on ACCURATE, and fine-tuning either toward the other's
strength only traded one class for another (`multivers_finetune_v1.md`).

## Test split, 606 citations

| System | Evidence | Accuracy | Macro-F1 | ACCURATE F1 | NOT ACCURATE F1 | NEI F1 |
|---|---|---|---|---|---|---|
| *Paper, published* | paper's | *0.59* | *0.52* | *0.69* | *0.43* | *0.42* |
| Reproduced paper system (released MultiVerS) | paper's | 0.584 | 0.524 | 0.692 | 0.426 | 0.455 |
| Released MultiVerS | ours | 0.574 | 0.505 | 0.691 | 0.408 | 0.414 |
| BiomedBERT, 3 seeds (mean) | ours | 0.656 | 0.499 | 0.780 | 0.450 | 0.268 |
| **Ensemble, w = 0.7** | **ours** | **0.627** | **0.536** | **0.749** | **0.435** | 0.426 |

Against the reproduced paper system on the same citations:

- **Accuracy +0.043, significant**: only the ensemble right on 71 citations, only the paper
  system on 45, McNemar exact p = 0.020.
- **Macro-F1 +0.012, not significant**: 95% paired bootstrap CI [−0.029, +0.054]. With 50 NEI
  citations in test, macro-F1 differences of this size cannot be resolved.

This is the first system here that is at or above the paper on both metrics and every class
except NEI (0.426 against 0.455, inside the paper's own 0.42).

**Per seed** (MultiVerS plus one BiomedBERT seed, `w` tuned on dev for each): test macro-F1
0.523, 0.549, 0.523, so 0.532 ± 0.015 — none below the paper's 0.52. An equal-weight
combination (w = 0.5), which was looked at after the fact and is reported only for transparency,
gives 0.543 macro-F1 and 0.662 accuracy; the dev-chosen 0.7 is the result to cite.

Reproduce:

```bash
R=results/classification
python scripts/ensemble_classifiers.py --mv-dev $R/repro_committed/ours_20.jsonl \
  --mv-test $R/test/ours_20.jsonl \
  --other-dev $R/v2/B_ours_s{0,1,2}/dev.jsonl --other-test $R/v2/B_ours_s{0,1,2}/ours_20_test.jsonl \
  --base $R/test/paper_20.jsonl --out $R/ensemble/test.jsonl
```

## Where the gain comes from

Full tables in `error_analysis.md` (`scripts/error_analysis.py`).

- **Almost all of it is correctly accepted ACCURATE citations.** On the 370 citations annotated
  ACCURATE, accuracy rises from 0.635 to 0.719 (48 fixed, 17 broken), because the ensemble
  wrongly flags far fewer of them as NOT ACCURATE (120 down to 90). For a tool that flags
  suspect citations, that is fewer false alarms.
- **Better retrieval is where it pays off.** Our reranker puts 228 citations in the "all gold
  evidence retrieved" bucket against the paper pipeline's 175, and on those the ensemble is
  right 0.702 of the time against 0.634. Where no gold evidence is retrieved, 0.525 against 0.418.
- **The gain is spread across articles**: more accurate on 13 of the 20 test articles, less on
  4, so it is not carried by one or two.
- **Subtle misrepresentation got worse.** CONTRADICT falls from 0.583 to 0.250 (n = 12) and
  OVERSIMPLIFY from 0.340 to 0.255 (n = 47); the ensemble's four most confident errors are all
  OVERSIMPLIFY citations it calls ACCURATE at 0.98 or above. These are citations whose source
  does say something close to the claim, which is what makes them hard, and the ACCURATE-leaning
  BiomedBERT half pushes them the wrong way. IRRELEVANT is about unchanged (0.500 to 0.460).

The next improvement target is therefore not retrieval or NOT ENOUGH INFO but the partial
misrepresentations, OVERSIMPLIFY and CONTRADICT, which together are 59 of the 170 NOT ACCURATE
test citations.
