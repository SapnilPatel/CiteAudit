#!/usr/bin/env python3
"""Combine MultiVerS with trained classifiers by weighted probability averaging, weight set on dev.

Why. The two model families fail differently (results/classifier_v2.md,
results/multivers_finetune_v1.md): the released MultiVerS is the only model that handles NOT
ENOUGH INFO (test F1 0.455), while the BiomedBERT classifier trained on our reranker's evidence
is much better on ACCURATE (0.78 against 0.69). Fine-tuning either toward the other's strength
traded one class for another; averaging their probabilities keeps both.

    p(label) = w * p_MultiVerS(label) + (1 - w) * mean over --other runs of p_run(label)

`w` is chosen on dev from a 0.1 grid by macro-F1, and only then applied to test. The result is
compared with --base (the reproduced paper system: released MultiVerS on the paper's evidence)
on the same test citations: macro-F1 with a paired bootstrap CI over citations, and accuracy with
McNemar's exact test. Every input is a predictions file in `predict_multivers.py`'s format.

    python scripts/ensemble_classifiers.py \\
        --mv-dev results/classification/repro_committed/ours_20.jsonl \\
        --mv-test results/classification/test/ours_20.jsonl \\
        --other-dev results/classification/v2/B_ours_s{0,1,2}/dev.jsonl \\
        --other-test results/classification/v2/B_ours_s{0,1,2}/ours_20_test.jsonl \\
        --base results/classification/test/paper_20.jsonl \\
        --out results/classification/ensemble/test.jsonl
"""
import argparse
import json
import random
from math import comb
from pathlib import Path

LABELS = ['ACCURATE', 'NOT_ACCURATE', 'NOT ENOUGH INFO']


def read(path):
    with open(path, encoding='utf8') as f:
        return {r['id']: r for r in map(json.loads, filter(str.strip, f))}


def combine(mv, others, w):
    ids = set(mv)
    for o in others:
        if set(o) != ids:
            raise SystemExit('prediction files cover different citations; they must match')
    out = {}
    for i in ids:
        p = {c: w * mv[i]['probs'][c] + (1 - w) * sum(o[i]['probs'][c] for o in others) / len(others)
             for c in LABELS}
        out[i] = {'id': i, 'doc_id': mv[i]['doc_id'], 'gold': mv[i]['gold'],
                  'prediction': max(LABELS, key=p.get),
                  'probs': {c: round(v, 5) for c, v in p.items()}}
    return out


def f1s(pred, gold, ids):
    out = []
    for c in LABELS:
        tp = sum(gold[i] == c and pred[i] == c for i in ids)
        fp = sum(gold[i] != c and pred[i] == c for i in ids)
        fn = sum(gold[i] == c and pred[i] != c for i in ids)
        out.append(2 * tp / (2 * tp + fp + fn) if tp else 0.)
    return out


def macro(pred, gold, ids):
    return sum(f1s(pred, gold, ids)) / 3


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--mv-dev', type=Path, required=True)
    p.add_argument('--mv-test', type=Path, required=True)
    p.add_argument('--other-dev', type=Path, nargs='+', required=True)
    p.add_argument('--other-test', type=Path, nargs='+', required=True)
    p.add_argument('--base', type=Path, required=True,
                   help='test predictions of the system to beat, on the same citations')
    p.add_argument('--weight', type=float, help='use this w instead of choosing it on dev')
    p.add_argument('--out', type=Path, help='write the combined test predictions here')
    p.add_argument('--bootstrap', type=int, default=2000)
    a = p.parse_args()
    if len(a.other_dev) != len(a.other_test):
        raise SystemExit('--other-dev and --other-test must list the same runs in the same order')

    mv_dev, mv_test = read(a.mv_dev), read(a.mv_test)
    od, ot = [read(x) for x in a.other_dev], [read(x) for x in a.other_test]
    gold_dev = {i: r['gold'] for i, r in mv_dev.items()}
    gold = {i: r['gold'] for i, r in mv_test.items()}
    ids = sorted(gold)

    def preds(rows):
        return {i: r['prediction'] for i, r in rows.items()}

    grid = [x / 10 for x in range(11)]
    for w in grid:
        print(f'  w={w:.1f}  dev macro-F1 {macro(preds(combine(mv_dev, od, w)), gold_dev, sorted(gold_dev)):.3f}')
    w = a.weight if a.weight is not None else max(
        grid, key=lambda w: macro(preds(combine(mv_dev, od, w)), gold_dev, sorted(gold_dev)))
    print(f'weight on MultiVerS: {w:.1f}' + (' (given)' if a.weight is not None else ' (best on dev)'))

    test = combine(mv_test, ot, w)
    pe, pb = preds(test), preds(read(a.base))
    if set(pb) != set(pe) or any(read(a.base)[i]['gold'] != gold[i] for i in ids):
        raise SystemExit('--base must predict the same citations with the same gold labels')
    for name, pr in (('ensemble', pe), ('base', pb)):
        f = f1s(pr, gold, ids)
        acc = sum(pr[i] == gold[i] for i in ids) / len(ids)
        print(f'{name:9s} accuracy {acc:.3f}  macro-F1 {sum(f) / 3:.3f}  '
              + '  '.join(f'{c} {x:.3f}' for c, x in zip(LABELS, f)))

    rng = random.Random(0)
    deltas = sorted(macro(pe, gold, s) - macro(pb, gold, s)
                    for s in ([rng.choice(ids) for _ in ids] for _ in range(a.bootstrap)))
    lo, hi = deltas[int(0.025 * a.bootstrap)], deltas[int(0.975 * a.bootstrap) - 1]
    print(f'macro-F1 delta {macro(pe, gold, ids) - macro(pb, gold, ids):+.3f}, '
          f'95% paired bootstrap CI [{lo:+.3f}, {hi:+.3f}]')
    only_e = sum(pe[i] == gold[i] != pb[i] for i in ids)
    only_b = sum(pb[i] == gold[i] != pe[i] for i in ids)
    n = only_e + only_b
    pv = min(1., 2 * sum(comb(n, k) for k in range(min(only_e, only_b) + 1)) / 2 ** n) if n else 1.
    print(f'accuracy: only ensemble right {only_e}, only base right {only_b}, McNemar exact p = {pv:.3f}')

    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(''.join(json.dumps(test[i]) + '\n' for i in ids), encoding='utf8')
        print(f'wrote {a.out}')


if __name__ == '__main__':
    main()
