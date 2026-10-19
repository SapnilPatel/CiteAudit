#!/usr/bin/env python3
"""Paired article bootstrap between two rankings files.

Two independent confidence intervals are the wrong comparison on this data: each carries the
variance of article difficulty, which both systems share, so the intervals overlap even when
one system wins on nearly every article. Resampling articles and taking the *difference*
inside each resample cancels that shared term.

    python scripts/compare_runs.py A.jsonl B.jsonl --metric Recall@20
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def load(path, keep_ids=None):
    """article -> rows. `keep_ids` restricts to a shared set of query ids.

    The two files need not cover the same queries: the harness writes rankings only for
    citances that have gold evidence, while the reranker writes every citance. Pooled recall
    is unaffected (a row with no gold adds 0 to numerator and denominator alike) but MRR
    divides by the number of rows, so a comparison must run on the intersection.
    """
    out = defaultdict(list)
    for line in open(path, encoding='utf8'):
        r = json.loads(line)
        if keep_ids is None or r['id'] in keep_ids:
            out[r['article']].append(r)
    return out


def ids_in(path):
    return {json.loads(line)['id'] for line in open(path, encoding='utf8')}


def pooled(rows, k, metric):
    """(numerator, denominator) for the paper's pooled recall, or MRR truncated at 20."""
    if metric == 'MRR':
        s = 0.
        for r in rows:
            pos = [i for i, x in enumerate(r['ranked'], 1) if x in set(r['gold'])]
            s += 1 / pos[0] if pos and pos[0] <= 20 else 0
        return s, len(rows)
    gold = sum(len(r['gold']) for r in rows)
    hit = sum(sum(1 for x in r['gold'] if x in r['ranked'][:k]) for r in rows)
    return hit, gold


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('a', type=Path)
    p.add_argument('b', type=Path)
    p.add_argument('--metric', default='Recall@20',
                   help='Recall@k for any k, or MRR (truncated at 20, over the rows present)')
    p.add_argument('--bootstrap', type=int, default=5000)
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()
    k = int(a.metric.split('@')[1]) if '@' in a.metric else 20

    shared = ids_in(a.a) & ids_in(a.b)
    A, B = load(a.a, shared), load(a.b, shared)
    arts = sorted(set(A) & set(B))
    dropped = (len(ids_in(a.a)) - len(shared), len(ids_in(a.b)) - len(shared))
    if any(dropped):
        print(f'comparing the {len(shared)} queries present in both files '
              f'(dropped {dropped[0]} from A, {dropped[1]} from B)')
    if not arts:
        raise SystemExit('the two files share no articles')
    if set(A) != set(B):
        print(f'warning: comparing on {len(arts)} shared articles of '
              f'{len(set(A))}/{len(set(B))}')

    def point(run, pick):
        num = den = 0
        for art in pick:
            n, d = pooled(run[art], k, a.metric)
            num += n
            den += d
        return num / den

    pt = point(A, arts) - point(B, arts)
    rng = random.Random(a.seed)
    deltas = sorted(point(A, pick) - point(B, pick)
                    for pick in (rng.choices(arts, k=len(arts)) for _ in range(a.bootstrap)))
    lo = deltas[int(.025 * a.bootstrap)]
    hi = deltas[min(int(.975 * a.bootstrap), a.bootstrap - 1)]
    share = sum(x > 0 for x in deltas) / a.bootstrap
    per = [(art, point(A, [art]) - point(B, [art])) for art in arts]
    wins = sum(1 for _, d in per if d > 0)

    print(f'{a.metric} pooled over {len(arts)} articles '
          f'({sum(len(v) for k2, v in A.items() if k2 in set(arts))} queries)')
    print(f'  A {a.a.name}: {point(A, arts):.3f}')
    print(f'  B {a.b.name}: {point(B, arts):.3f}')
    print(f'  delta {pt:+.3f}   95% CI [{lo:+.3f},{hi:+.3f}]   '
          f'P(delta>0)={share:.3f}   A better on {wins}/{len(arts)} articles')
    print('  per article: ' + '  '.join(f'{d:+.2f}' for _, d in per))


if __name__ == '__main__':
    main()
