# Rule-based detail checks (numbers, entities, negation): real signal, too rare to help

**Question.** The label audit (`label_audit_v1.md`) found that the classifier's clear errors are
narrow: numbers the evidence does not contain, swapped genes or conditions, and negated citations
("Unlike other reports [X]"). Can exact, rule-based checks of these fix them?

**Method.** `scripts/detail_checks.py` computes 18 features from the citing sentence and the 20
retrieved evidence sentences, using only what the deployed tool also has (no model, no LLM):
- numbers missing from the evidence, counting a percentage's complement as present, so "35.6%
  female" matches "64.4% male";
- gene, variant and drug-code names missing from the evidence;
- contrast cues ("unlike", "in contrast to"), negation and hedging counts;
- word overlap with the evidence, and the marker type.

`scripts/stack_checks.py` combines them with the NLI classifier's out-of-fold probabilities (v5, 3
folds by article) and cross-validates over the same folds.

## Result (cross-validated, 2141 citations)

| | Accuracy | Macro-F1 | NOT ACC F1 | NEI F1 |
|---|---|---|---|---|
| **NLI classifier, its own prediction** | 0.676 | **0.570** | 0.513 | 0.415 |
| Logistic stacker, NLI probabilities only | 0.695 | 0.528 | 0.513 | 0.267 |
| Logistic stacker, NLI + checks | 0.704 | 0.544 | 0.519 | 0.305 |
| Same, balanced class weights, NLI only | 0.591 | 0.488 | 0.423 | 0.298 |
| Same, balanced, NLI + checks | 0.618 | 0.520 | 0.511 | 0.308 |
| Anchored model (NLI logits + bias + weighted checks), bias only | 0.623 | 0.504 | 0.447 | 0.300 |
| Anchored model, bias + checks | 0.647 | 0.530 | 0.492 | 0.325 |

- **The checks carry signal:** in all three stacker forms they add 0.016–0.032 macro-F1 over the
  same stacker without them (balanced: +0.032, 95% CI +0.012 to +0.052, McNemar p = 0.002).
- **But no combination beats the classifier's own prediction.** Re-fitting the decision on the
  out-of-fold probabilities costs 0.04–0.08 macro-F1 (it sacrifices NOT ENOUGH INFO), more than the
  checks give back. Best variant against the classifier: −0.026 macro-F1.

**A mistake, corrected.** The first gate compared "stacker + checks" only with "stacker without
checks". It passed, and the test run was worse than the released model (0.639 / 0.537 against
0.662 / 0.546). The gate now also requires beating the classifier's own prediction, and fails.

## Why it cannot help: the errors it targets are rare

How often each check fires, by fine label (train):

| Fine label | n | Number missing | Entity missing | Contrast cue |
|---|---|---|---|---|
| ACCURATE | 1308 | 0.07 | 0.18 | 0.00 |
| ETIQUETTE | 304 | 0.08 | 0.38 | 0.01 |
| NOT_SUBSTANTIATE | 189 | 0.11 | 0.31 | 0.00 |
| IRRELEVANT | 143 | 0.12 | 0.36 | 0.00 |
| CONTRADICT | 58 | 0.12 | 0.14 | 0.00 |
| OVERSIMPLIFY | 56 | 0.04 | 0.21 | 0.02 |
| **MISQUOTE** | **23** | **0.48** | 0.26 | 0.00 |

- The number check does what it should: it fires on 48% of misquotations against 7% of accurate
  citations. But misquotations are 23 of 2141 citations (1%).
- A missing entity mostly marks irrelevant and ETIQUETTE citations, which the classifier already
  separates.
- Contrast cues occur in almost no citations.

The failure types found in the audit are real, but together they are a few percent of the data.
Fixing all of them would move macro-F1 by about a point. This also explains why the synthetic
"wrong detail" rewrites (`synthetic_v1.md`, `synthetic_v2.md`) did not help.

## What this adds to the picture

This is the eighth intervention to land at or below the classifier's CV macro-F1 of 0.57: model,
data, pretraining, length, ensembles, evidence oracle, markers, and now targeted checks. Together
with the label audit, where about half of the confident errors are debatable labels, the evidence
says the remaining gap on this benchmark is mostly label ambiguity plus a long tail of rare error
types, not a missing component.

The checks may still be useful in the tool as **explanations** ("this number does not appear in the
cited paper"), even though they do not change the label.
