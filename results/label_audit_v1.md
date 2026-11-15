# Label audit of confident errors (v1)

**Question.** Every model and input change lands at CV macro-F1 0.57 ± 0.01 once calibrated
(`markers_v1.md`, `docs/HANDOFF.md`). Is the remaining error the model's, or the labels'?

**Method.** Out-of-fold predictions on train (2141 citations) from two independently trained
classifiers: the current one (v5) and the markers one (v9). Of 694 errors by the current
classifier, 97 are **confident disagreements**: both models predict the same wrong label with
probability > 0.95. 30 of the 97 were sampled at random (seed 1) and each was read against its
gold evidence sentences, or the top retrieved sentences when there is no gold evidence. One rater
(Claude), not an expert annotator: treat the split as indicative.

Composition of the 97: NOT_SUBSTANTIATE called ACCURATE 28, ACCURATE called NOT ACCURATE 17,
ETIQUETTE called ACCURATE 12, OVERSIMPLIFY called ACCURATE 11, CONTRADICT called ACCURATE 7,
MISQUOTE called ACCURATE 6, other 16.

## Verdicts (30)

| Verdict | Count | Citation ids |
|---|---|---|
| Gold label right, model wrong | 13 (43%) | 236, 252, 344, 368, 399, 423, 855, 1133, 1272, 1430, 1783, 1819, 1875 |
| Debatable: a reasonable annotator could choose either label | 13 (43%) | 458, 701, 716, 746, 975, 1002, 1195, 1856, 1877, 1898, 1932, 1957, 2027 |
| Gold label looks wrong | 4 (13%) | 719, 937, 1880, 2009 |

Examples of each:

- **Gold wrong.** 2009: "primary mitochondrial diseases affect nearly 5% of patients with ASD"
  is labelled NOT_SUBSTANTIATE; the evidence says "the prevalence of MD in ... ASD was 5.0%". 1880:
  "Type I and III IFNs are typically suppressed in COVID-19" is labelled CONTRADICT; the evidence
  says the infected cells "show no significant IFN-I or IFN-III expression".
- **Debatable.** 1898: the claim adds saliva, urine and an 8th-day detail the evidence lacks, but is
  labelled ACCURATE. 746 (ETIQUETTE): the cited paper does itself report better survival under ART.
- **Model wrong.** These cluster in a few kinds of reasoning:
  - **Numbers and complements:** 344 "35.6% female" against "64.4% were male" (correct, called
    NOT ACCURATE); 1133 "472 million with diabetes" when 472 million is the impaired-glucose-tolerance
    figure; 855 "7.5–8%" against "7.6–8.5%".
  - **Swapped entity or condition:** 1272 claims for BRCA1/2 what holds only for BRCA1; 368 day 29
    against day 15.
  - **Negation and framing:** 1783 "Unlike other reports [cited], ... was not a predictor": it cites
    a paper for the opposite finding, which is correct, and the model calls it NOT ACCURATE.

## What this says

- **About half of the confident errors are not clear model errors** (13 debatable + 4 likely
  mislabelled out of 30). The Citation-Integrity paper also reports annotation inconsistency. A
  meaningful part of the plateau is label ambiguity, which no model change removes. This is the
  most likely explanation for why seven different interventions converge on the same score.
- **The clear model errors are specific and checkable:** numeric mismatches, swapped entities or
  conditions, and negated citations ("unlike [X]"). These are narrow failure types, which a targeted
  component could address. More data or a bigger model is unlikely to.
- **Caveat:** one rater, 30 citations, and only the confident errors. A proper estimate of the label
  ceiling would need a second rater on a random sample.
