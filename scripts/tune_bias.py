#!/usr/bin/env python3
"""Choose per-class decision biases on out-of-fold predictions, then apply them to other predictions.

Macro-F1 weighs the two minority classes as much as ACCURATE, and a classifier trained with class
weights does not put its decision boundaries where macro-F1 is best. Adding a constant b_c to each
class's log-probability moves them. b is chosen on the 2141 cross-validated out-of-fold
predictions only (ACCURATE fixed at 0, the other two on a 0.1 grid in [-2, 2]), so applying it to
test is a single, pre-registered change. On the NLI classifier this moved CV macro-F1 0.570 to 0.574
and test 0.575 to 0.585 (results/nli_classifier_v1.md).

    python scripts/tune_bias.py --oof results/classification/v5/cv_nli_f{0,1,2}/oof.jsonl \\
        --apply results/classification/v5/final_nli_s{0,1,2}/ours_20r_test.jsonl --out bias.json
"""
import argparse
import json
import math
from pathlib import Path

LABELS = ['ACCURATE', 'NOT_ACCURATE', 'NOT ENOUGH INFO']


def read(path):
    return [json.loads(x) for x in Path(path).read_text(encoding='utf8').splitlines() if x.strip()]


def predict(rows, bias):
    return [max(LABELS, key=lambda c: math.log(r['probs'][c] + 1e-9) + bias[c]) for r in rows]


def scores(rows, pred):
    f1 = []
    for c in LABELS:
        tp = sum(r['gold'] == c and p == c for r, p in zip(rows, pred))
        fp = sum(r['gold'] != c and p == c for r, p in zip(rows, pred))
        fn = sum(r['gold'] == c and p != c for r, p in zip(rows, pred))
        f1.append(2 * tp / (2 * tp + fp + fn) if tp else 0.)
    return sum(r['gold'] == p for r, p in zip(rows, pred)) / len(rows), sum(f1) / 3


def tune(rows):
    grid = [x / 10 for x in range(-20, 21)]
    cands = ({'ACCURATE': 0., 'NOT_ACCURATE': a, 'NOT ENOUGH INFO': n} for a in grid for n in grid)
    # ties go to the smallest change
    return max(cands, key=lambda b: (round(scores(rows, predict(rows, b))[1], 6),
                                     -abs(b['NOT_ACCURATE']) - abs(b['NOT ENOUGH INFO'])))


def average(files):
    runs = [{r['id']: r for r in read(f)} for f in files]
    return [{'id': i, 'gold': runs[0][i]['gold'],
             'probs': {c: sum(r[i]['probs'][c] for r in runs) / len(runs) for c in LABELS}}
            for i in runs[0]]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--oof', nargs='+', required=True, help='out-of-fold prediction files (concatenated)')
    p.add_argument('--apply', nargs='*', default=[], help='prediction files, each scored and also averaged')
    p.add_argument('--out', type=Path)
    a = p.parse_args()
    oof = sum((read(f) for f in a.oof), [])
    zero = {c: 0. for c in LABELS}
    bias = tune(oof)
    res = {'bias': bias, 'cv_before': scores(oof, predict(oof, zero)), 'cv_after': scores(oof, predict(oof, bias))}
    print(f'bias {bias}\nCV ({len(oof)}): accuracy / macro-F1 {res["cv_before"][0]:.3f} / {res["cv_before"][1]:.3f}'
          f' -> {res["cv_after"][0]:.3f} / {res["cv_after"][1]:.3f}')
    sets = [(f, read(f)) for f in a.apply] + ([('average of the above', average(a.apply))] if len(a.apply) > 1 else [])
    for name, rows in sets:
        b0, b1 = scores(rows, predict(rows, zero)), scores(rows, predict(rows, bias))
        res[name] = {'before': b0, 'after': b1}
        print(f'{name}: {b0[0]:.3f} / {b0[1]:.3f} -> {b1[0]:.3f} / {b1[1]:.3f}')
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(res, indent=2))


if __name__ == '__main__':
    main()
