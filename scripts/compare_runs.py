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
    """(numerator, denominator) for one metric, as a ratio of sums so the article bootstrap
    can resample articles and recombine. Every supported name is handled explicitly: an
    unrecognised one must raise, not fall through to recall — two different metrics printing
    the same number is otherwise indistinguishable from a real tie."""
    if metric == 'MRR':
        s = 0.
        for r in rows:
            # The paper truncates the rank at 20, so only the top 20 can contribute and
            # scanning deeper is wasted work. `gold` becomes a set once per row, not once
            # per candidate.
            gold = set(r['gold'])
            s += next((1 / i for i, x in enumerate(r['ranked'][:20], 1) if x in gold), 0.)
        return s, len(rows)
    name = metric.split('@')[0]
    if name == 'Recall':  # pooled over evidence sentences, the paper's definition
        return (sum(sum(1 for x in r['gold'] if x in r['ranked'][:k]) for r in rows),
                sum(len(r['gold']) for r in rows))
    if name == 'Hit':  # per citance: at least one gold in the top k
        return sum(any(x in r['ranked'][:k] for x in r['gold']) for r in rows), len(rows)
    if name == 'AllGold':  # per citance: every gold in the top k
        return sum(all(x in r['ranked'][:k] for x in r['gold']) for r in rows), len(rows)
    raise SystemExit(f'unknown metric {metric!r}: use MRR, Recall@k, Hit@k or AllGold@k')


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('a', type=Path)
    p.add_argument('b', type=Path)
    p.add_argument('--metric', default='Recall@20',
                   help='Recall@k, Hit@k, AllGold@k, or MRR (truncated at 20, over the rows present)')
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

    # Every metric here is a ratio of sums over articles, so one article's (numerator,
    # denominator) does not depend on which other articles a resample drew. Computing it
    # once per article and summing the cached pairs is exactly equivalent to recomputing
    # it inside each of the 5000 resamples, and turns an MRR comparison from minutes into
    # well under a second.
    statA = {art: pooled(A[art], k, a.metric) for art in arts}
    statB = {art: pooled(B[art], k, a.metric) for art in arts}

    def point(stat, pick):
        num = den = 0
        for art in pick:
            n, d = stat[art]
            num += n
            den += d
        return num / den

    pt = point(statA, arts) - point(statB, arts)
    rng = random.Random(a.seed)
    deltas = sorted(point(statA, pick) - point(statB, pick)
                    for pick in (rng.choices(arts, k=len(arts)) for _ in range(a.bootstrap)))
    lo = deltas[int(.025 * a.bootstrap)]
    hi = deltas[min(int(.975 * a.bootstrap), a.bootstrap - 1)]
    share = sum(x > 0 for x in deltas) / a.bootstrap
    per = [(art, point(statA, [art]) - point(statB, [art])) for art in arts]
    wins = sum(1 for _, d in per if d > 0)

    print(f'{a.metric} pooled over {len(arts)} articles '
          f'({sum(len(A[art]) for art in arts)} queries)')
    print(f'  A {a.a.name}: {point(statA, arts):.3f}')
    print(f'  B {a.b.name}: {point(statB, arts):.3f}')
    print(f'  delta {pt:+.3f}   95% CI [{lo:+.3f},{hi:+.3f}]   '
          f'P(delta>0)={share:.3f}   A better on {wins}/{len(arts)} articles')
    print('  per article: ' + '  '.join(f'{d:+.2f}' for _, d in per))


if __name__ == '__main__':
    main()
