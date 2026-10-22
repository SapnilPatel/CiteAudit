#!/usr/bin/env python3
"""Score MultiVerS label predictions, and compare two retrieval arms on the same claims.

Reports 3-way micro/macro/per-class F1, the confusion matrix, and — when two prediction files
are given — a **paired** comparison. Paired is the only valid comparison here: both arms predict
the same 316 claims with the same gold labels and differ only in the retrieved evidence, so the
variance of claim difficulty is shared and cancels in the difference. McNemar's exact test on the
claims where exactly one arm is correct is the standard test for that design.

    python scripts/score_multivers.py results/classification/ours_20.jsonl
    python scripts/score_multivers.py results/classification/ours_20.jsonl \
        results/classification/paper_20.jsonl
"""
import argparse
import json
from collections import Counter
from math import comb
from pathlib import Path

CLASSES = ['ACCURATE', 'NOT_ACCURATE', 'NOT ENOUGH INFO']


def read_jsonl(path):
    with open(path, encoding='utf8') as f:
        return [json.loads(x) for x in f if x.strip()]


def gold_of(claim):
    """The 3-way gold label as the export encodes it: NEI is an empty evidence dict."""
    ev = claim.get('evidence')
    return next(iter(ev.values()))[0]['label'] if ev else 'NOT ENOUGH INFO'


def load(path, gold_from=None, split='dev'):
    """Predictions by claim id.

    `gold_from` re-reads the gold labels from an export directory's claims file instead of
    trusting the ones stored in the prediction file. Predictions depend only on the document, so
    a corrected label mapping can be applied without re-running inference — and it must be
    applied through one path for every arm, or two arms would be scored against different gold.
    """
    rows = read_jsonl(path)
    by_id = {}
    for r in rows:
        if r['id'] in by_id:
            raise SystemExit(f'{path}: claim {r["id"]} predicted more than once; this scorer '
                             f'assumes one document per claim')
        by_id[r['id']] = r
    if gold_from:
        gold = {c['id']: gold_of(c) for c in read_jsonl(gold_from / f'claims_{split}.jsonl')}
        missing = [i for i in by_id if i not in gold]
        if missing:
            raise SystemExit(f'{gold_from} has no label for {len(missing)} predicted claims '
                             f'(e.g. {missing[:5]})')
        for i, r in by_id.items():
            r['gold'] = gold[i]
    return by_id


def prf(gold, pred, cls):
    tp = sum(1 for g, p in zip(gold, pred) if g == cls and p == cls)
    fp = sum(1 for g, p in zip(gold, pred) if g != cls and p == cls)
    fn = sum(1 for g, p in zip(gold, pred) if g == cls and p != cls)
    prec = tp / (tp + fp) if tp + fp else 0.
    rec = tp / (tp + fn) if tp + fn else 0.
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.
    return prec, rec, f1, tp + fn


def report(name, by_id, ids, classes):
    gold = [by_id[i]['gold'] for i in ids]
    pred = [by_id[i]['prediction'] for i in ids]
    acc = sum(g == p for g, p in zip(gold, pred)) / len(ids)
    print(f'\n{name}   n={len(ids)}   accuracy {acc:.3f}  (= micro-F1 for single-label 3-way)')
    print(f'  {"class":<18}{"P":>7}{"R":>7}{"F1":>7}{"support":>9}')
    f1s = []
    for c in classes:
        prec, rec, f1, sup = prf(gold, pred, c)
        f1s.append(f1)
        print(f'  {c:<18}{prec:>7.3f}{rec:>7.3f}{f1:>7.3f}{sup:>9}')
    print(f'  {"macro-F1":<18}{"":>7}{"":>7}{sum(f1s) / len(f1s):>7.3f}')
    conf = Counter(zip(gold, pred))
    print('  confusion (gold -> predicted):')
    for g in classes:
        row = '  '.join(f'{conf[(g, p)]:>4}' for p in classes)
        print(f'    {g:<18}{row}')
    print(f'    {"predicted totals":<18}' +
          '  '.join(f'{sum(1 for p in pred if p == c):>4}' for c in classes))
    return acc, sum(f1s) / len(f1s)


def mcnemar_exact(n01, n10):
    """Two-sided exact binomial p for the discordant pairs. n01/n10: only-B / only-A correct."""
    n = n01 + n10
    if n == 0:
        return 1.0
    k = min(n01, n10)
    tail = sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('a', type=Path, help='predictions to score (arm A)')
    p.add_argument('b', type=Path, nargs='?', help='optional second arm to compare against')
    p.add_argument('--gold-from', type=Path,
                   help='export directory to re-read gold labels from (e.g. '
                        'data_train/target/ours_20), instead of the labels stored in the '
                        'prediction files. Use it after correcting the label mapping: '
                        'predictions depend only on the document, so they need no re-running.')
    p.add_argument('--split', default='dev', choices=['train', 'dev', 'test'])
    p.add_argument('--evidence-only', action='store_true',
                   help='restrict to claims that have gold evidence, i.e. drop the NOT ENOUGH '
                        'INFO class. Worth reporting alongside the 3-way numbers: the released '
                        'checkpoint was trained with NEI meaning "this chunk of the reference is '
                        'not evidence", while an export built from retrieved sentences always '
                        'looks evidence-like, so NEI recall is not a fair test of it.')
    a = p.parse_args()

    A = load(a.a, a.gold_from, a.split)
    ids = sorted(A)
    classes = list(CLASSES)
    if a.evidence_only:
        ids = [i for i in ids if A[i]['gold'] != 'NOT ENOUGH INFO']
        classes = [c for c in CLASSES if c != 'NOT ENOUGH INFO']

    if a.b is None:
        report(a.a.stem, A, ids, classes)
        return

    B = load(a.b, a.gold_from, a.split)
    shared = sorted(set(ids) & set(B))
    if len(shared) != len(ids):
        print(f'comparing the {len(shared)} claims present in both files '
              f'(A has {len(A)}, B has {len(B)})')
    for i in shared:
        if A[i]['gold'] != B[i]['gold']:
            raise SystemExit(f'claim {i} has different gold labels in the two files '
                             f'({A[i]["gold"]!r} vs {B[i]["gold"]!r}); the arms are not '
                             f'comparable')
    accA, macroA = report(f'A  {a.a.stem}', A, shared, classes)
    accB, macroB = report(f'B  {a.b.stem}', B, shared, classes)

    n10 = sum(1 for i in shared
              if A[i]['prediction'] == A[i]['gold'] and B[i]['prediction'] != B[i]['gold'])
    n01 = sum(1 for i in shared
              if A[i]['prediction'] != A[i]['gold'] and B[i]['prediction'] == B[i]['gold'])
    disagree = sum(1 for i in shared if A[i]['prediction'] != B[i]['prediction'])
    print(f'\npaired comparison over {len(shared)} claims')
    print(f'  accuracy  A {accA:.3f}  B {accB:.3f}   delta {accA - accB:+.3f}')
    print(f'  macro-F1  A {macroA:.3f}  B {macroB:.3f}   delta {macroA - macroB:+.3f}')
    print(f'  the two arms predict differently on {disagree} claims '
          f'({disagree / len(shared):.1%})')
    print(f'  only A correct: {n10}   only B correct: {n01}   '
          f"McNemar exact p = {mcnemar_exact(n01, n10):.3f}")
    if disagree == 0:
        print('  the arms are identical on every claim: the evidence difference changed nothing')


if __name__ == '__main__':
    main()
