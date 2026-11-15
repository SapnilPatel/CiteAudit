#!/usr/bin/env python3
"""Combine the NLI classifier's probabilities with detail_checks.py's features, and test the gain.

Protocol. The NLI classifier's out-of-fold probabilities on train (3 folds by article,
results/classification/v5/cv_nli_f*) are the input. A multinomial logistic regression is
cross-validated over the same article folds in two variants that differ only in their inputs:

  nli      the three NLI log-probabilities (a learned recalibration, the fair baseline)
  checks   the same plus the detail-check features

Gate: `checks` must beat both `nli` and the NLI classifier's own argmax by at least 0.01
cross-validated macro-F1. (The first version of this script compared only with `nli`; a logistic
stacker on the NLI probabilities is itself 0.04-0.08 below the argmax, so that gate passed while
the result was worse than the classifier: results/checks_v1.md.) Only then is it applied to test: refit on all 2141 citations and scored on the test predictions of
the released classifier and of the 3-seed v5 average.

    python scripts/stack_checks.py --out results/classification/checks
"""
import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from detail_checks import features  # noqa: E402
from train_classifier import fold_articles  # noqa: E402

LABELS = ['ACCURATE', 'NOT_ACCURATE', 'NOT ENOUGH INFO']
read = lambda p: [json.loads(x) for x in Path(p).read_text(encoding='utf8').splitlines() if x.strip()]


def scores(gold, pred):
    f1 = []
    for c in LABELS:
        tp = sum(g == c == p for g, p in zip(gold, pred))
        fp = sum(p == c != g for g, p in zip(gold, pred))
        fn = sum(g == c != p for g, p in zip(gold, pred))
        f1.append(2 * tp / (2 * tp + fp + fn) if tp else 0.)
    return [sum(g == p for g, p in zip(gold, pred)) / len(gold), sum(f1) / 3] + f1


def mcnemar(gold, a, b):
    x = sum(p == g != q for g, p, q in zip(gold, a, b))
    y = sum(q == g != p for g, p, q in zip(gold, a, b))
    n = x + y
    return x, y, min(1., 2 * sum(math.comb(n, j) for j in range(min(x, y) + 1)) / 2 ** n) if n else 1.


def bootstrap(gold, a, b, n=2000):
    rng, ids, d = random.Random(0), range(len(gold)), []
    for _ in range(n):
        s = [rng.choice(ids) for _ in ids]
        g = [gold[i] for i in s]
        d.append(scores(g, [a[i] for i in s])[1] - scores(g, [b[i] for i in s])[1])
    d.sort()
    return d[int(0.025 * n)], d[int(0.975 * n)]


def matrix(preds, export, split, variant):
    claims = {c['id']: c for c in read(Path(export) / f'claims_{split}.jsonl')}
    corpus = {d['doc_id']: d['abstract'] for d in read(Path(export) / 'corpus.jsonl')}
    rows, names = [], None
    for p in preds:
        x = [math.log(p['probs'][c] + 1e-6) for c in LABELS]
        if variant == 'checks':
            c = claims[p['id']]
            f = features(c['claim'], corpus[c['cited_doc_ids'][0]])
            names = list(f)
            x += list(f.values())
        rows.append(x)
    return np.array(rows), names


def fit(X, y, weight):
    sc = StandardScaler().fit(X)
    m = LogisticRegression(C=1.0, max_iter=2000, class_weight=weight).fit(sc.transform(X), y)
    return sc, m


def average(runs):
    out = []
    for rs in zip(*[sorted(r, key=lambda x: x['id']) for r in runs]):
        p = {c: sum(r['probs'][c] for r in rs) / len(rs) for c in LABELS}
        out.append({**rs[0], 'probs': p, 'prediction': max(p, key=p.get)})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--oof', default='results/classification/v5/cv_nli_f{k}/oof.jsonl')
    ap.add_argument('--train-export', default='data_train/target/ours_20r_train_cv')
    ap.add_argument('--test-export', default='data_train/target/ours_20r_test')
    ap.add_argument('--context', default='data_train/context/train.json')
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    oof = sorted(sum((read(a.oof.format(k=k)) for k in range(3)), []), key=lambda r: r['id'])
    gold = [r['gold'] for r in oof]
    y = np.array([LABELS.index(g) for g in gold])
    meta = json.loads(Path(a.context).read_text(encoding='utf8'))
    art = np.array([meta[str(r['id'])]['article'] for r in oof])
    folds = fold_articles(art.tolist(), 3)

    report = {'cv': {}, 'gate': None, 'test': {}}
    cv_pred = {}
    print(f'{"cross-validated, 2141 citations":30s} accuracy  macro-F1    ACC F1  N_ACC F1    NEI F1')
    print(f'{"NLI argmax (no stacker)":30s} ' + ' '.join(f'{x:9.3f}' for x in scores(gold, [r["prediction"] for r in oof])))
    for weight in (None, 'balanced'):
        for variant in ('nli', 'checks'):
            X, names = matrix(oof, a.train_export, 'train', variant)
            pred = np.zeros(len(oof), dtype=int)
            for held in folds:
                te = np.isin(art, list(held))
                sc, m = fit(X[~te], y[~te], weight)
                pred[te] = m.predict(sc.transform(X[te]))
            p = [LABELS[i] for i in pred]
            key = f'{variant}/{weight or "none"}'
            cv_pred[key], report['cv'][key] = p, scores(gold, p)
            print(f'{key:30s} ' + ' '.join(f'{x:9.3f}' for x in report['cv'][key]))
    gains = {}
    for weight in ('none', 'balanced'):
        n, c = cv_pred[f'nli/{weight}'], cv_pred[f'checks/{weight}']
        x, z, pv = mcnemar(gold, c, n)
        lo, hi = bootstrap(gold, c, n)
        gains[weight] = report['cv'][f'checks/{weight}'][1] - report['cv'][f'nli/{weight}'][1]
        print(f'checks vs nli ({weight}): macro-F1 {gains[weight]:+.3f} [95% CI {lo:+.3f}, {hi:+.3f}], '
              f'right only with checks {x}, only without {z}, McNemar p = {pv:.4f}')
    best_w = max(gains, key=gains.get)
    argmax_f1 = scores(gold, [r['prediction'] for r in oof])[1]
    vs_argmax = report['cv'][f'checks/{best_w}'][1] - argmax_f1
    report['gate'] = {'weight': best_w, 'gain_vs_nli_stacker': gains[best_w], 'gain_vs_argmax': vs_argmax,
                      'passed': gains[best_w] >= 0.01 and vs_argmax >= 0.01}

    # Coefficients of the full-data fit, to see what the checks learned.
    X, names = matrix(oof, a.train_export, 'train', 'checks')
    sc, m = fit(X, y, None if best_w == 'none' else best_w)
    coef = {n: dict(zip(LABELS, m.coef_[:, j + 3].round(2).tolist())) for j, n in enumerate(names)}
    report['coefficients'] = coef
    print('\nstandardised coefficients (ACCURATE / NOT_ACCURATE / NEI):')
    for n, c in coef.items():
        print(f'  {n:18s} ' + ' '.join(f'{v:+.2f}' for v in c.values()))

    if not report['gate']['passed']:
        print(f'\nGate not passed (needs +0.010 macro-F1 over the stacker and the argmax; got '
              f'{gains[best_w]:+.3f} and {vs_argmax:+.3f}): test not scored.')
    else:
        print(f'\nGate passed with class_weight={best_w}: scoring test.')
        w = None if best_w == 'none' else best_w
        tests = {'released (seed 1)': read('results/classification/release/preds_s1/ours_20r_test.jsonl'),
                 'v5, 3 seeds averaged': average([read(f'results/classification/v5/final_nli_s{s}/ours_20r_test.jsonl')
                                                  for s in (0, 1, 2)])}
        for name, preds in tests.items():
            preds = sorted(preds, key=lambda r: r['id'])
            tg = [r['gold'] for r in preds]
            res = {}
            for variant in ('nli', 'checks'):
                Xtr, _ = matrix(oof, a.train_export, 'train', variant)
                sc, m = fit(Xtr, y, w)
                Xte, _ = matrix(preds, a.test_export, 'test', variant)
                res[variant] = [LABELS[i] for i in m.predict(sc.transform(Xte))]
            base = [r['prediction'] for r in preds]
            report['test'][name] = {k: scores(tg, v) for k, v in [('argmax', base), *res.items()]}
            x, z, pv = mcnemar(tg, res['checks'], base)
            lo, hi = bootstrap(tg, res['checks'], base)
            print(f'\ntest, {name}')
            for k, v in report['test'][name].items():
                print(f'  {k:8s} ' + ' '.join(f'{s:9.3f}' for s in v))
            print(f'  checks vs argmax: right only with checks {x}, only without {z}, McNemar p = {pv:.4f}; '
                  f'macro-F1 95% CI [{lo:+.3f}, {hi:+.3f}]')
            with open(a.out / f'test_{name.split()[0]}.jsonl', 'w') as f:
                for r, p in zip(preds, res['checks']):
                    f.write(json.dumps({'id': r['id'], 'gold': r['gold'], 'prediction': p}) + '\n')
    (a.out / 'report.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
