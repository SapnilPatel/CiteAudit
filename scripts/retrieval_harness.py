#!/usr/bin/env python3
"""Sentence-retrieval harness for the Citation-Integrity multivers-format data.

Unlike week2_pipeline.py (which expects one abstract per claim), this handles the
real layout: each claim lists `cited_doc_ids`, a contiguous range of corpus chunks
that together form the cited reference article. The candidate pool for a claim is
every sentence of every chunk in that range, keyed "doc_id:sent_idx".

Rankers:
  bm25          pure-python BM25 over the whole pool (default k1=0.9, b=0.4 = Lucene/pyserini)
  bm25-monot5   BM25 top-N (default 60, as in the authors' bm25t5.ipynb) reranked by MonoT5
  file          evaluate an existing rankings JSONL (--rankings-in), e.g. a cross-encoder run

Metrics (per query, then averaged over queries):
  Recall@k   fraction of the query's gold sentences in the top k (paper's definition)
  Hit@k      at least one gold sentence in the top k
  AllGold@k  every gold sentence in the top k
  MRR        reciprocal rank of the first gold sentence
95% CIs come from a bootstrap that resamples reference articles, not queries,
because test has only 20 articles and queries within an article are correlated.
"""
import argparse
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

DEFAULT_DATA = Path(__file__).resolve().parents[1] / 'upstream/Citation-Integrity/Data/multivers-format'
KS = (1, 5, 10, 20, 50, 100)
MARKERS = re.compile(r'<\|(?:multi_cit|cit)\|>')


def read_jsonl(path):
    with open(path, encoding='utf8') as f:
        return [json.loads(x) for x in f if x.strip()]


def clean_query(text):
    return re.sub(r'\s+', ' ', MARKERS.sub(' ', text)).strip()


def key(did, si):
    return f'{int(did)}:{int(si)}'


def load_split(data_dir, split):
    """Return a list of queries: id, article, query, keys, texts, gold (set of keys)."""
    corpus = {int(d['doc_id']): d['abstract'] for d in read_jsonl(Path(data_dir) / 'corpus.jsonl')}
    queries = []
    for c in read_jsonl(Path(data_dir) / f'claims-{split}.jsonl'):
        dids = sorted(int(d) for d in c['cited_doc_ids'])
        keys, texts = [], []
        for did in dids:
            for si, s in enumerate(corpus.get(did, [])):
                keys.append(key(did, si))
                texts.append(s)
        gold = {key(d, si) for d, items in c.get('evidence', {}).items()
                for it in items for si in it['sentences']}
        queries.append({'id': c['id'], 'article': dids[0], 'query': clean_query(c['claim']),
                        'keys': keys, 'texts': texts, 'gold': gold})
    return queries


def with_context(texts, i):
    """Sentence i with its neighbours in the article (pool order crosses chunk boundaries)."""
    prev = texts[i - 1] if i > 0 else ''
    nxt = texts[i + 1] if i + 1 < len(texts) else ''
    return ' '.join(x for x in (prev, texts[i], nxt) if x)


def tok(s):
    return re.findall(r'[a-z0-9]+', s.lower())


def bm25_scores(query, texts, k1, b):
    docs = [tok(t) for t in texts]
    n = len(docs)
    avg = max(sum(map(len, docs)) / max(n, 1), 1)
    df = Counter(w for d in docs for w in set(d))
    qs = set(tok(query))
    idf = {w: math.log(1 + (n - df[w] + .5) / (df[w] + .5)) for w in qs if w in df}
    out = []
    for d in docs:
        f = Counter(d)
        out.append(sum(idf[w] * f[w] * (k1 + 1) / (f[w] + k1 * (1 - b + b * len(d) / avg))
                       for w in idf if f[w]))
    return out


def rank_by(scores):
    return sorted(range(len(scores)), key=lambda i: (-scores[i], i))


def make_monot5(model_name, batch_size):
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    dev = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    tokzr = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(dev).eval()
    true_id = tokzr.encode('true', add_special_tokens=False)[0]
    false_id = tokzr.encode('false', add_special_tokens=False)[0]
    start = model.config.decoder_start_token_id

    def score(query, passages):
        out = []
        for lo in range(0, len(passages), batch_size):
            chunk = passages[lo:lo + batch_size]
            enc = tokzr([f'Query: {query} Document: {p} Relevant:' for p in chunk],
                        return_tensors='pt', padding=True, truncation=True, max_length=512).to(dev)
            dec = torch.full((len(chunk), 1), start, dtype=torch.long, device=dev)
            with torch.inference_mode():
                logits = model(**enc, decoder_input_ids=dec).logits[:, -1, [true_id, false_id]]
            out.extend(torch.log_softmax(logits.float(), dim=-1)[:, 0].tolist())
        return out
    return score, dev


def query_metrics(ranked, gold):
    """Per-query metrics plus the raw counts the paper's pooled definitions need.

    Two averagings are reported downstream and they are NOT interchangeable:
      macro  mean over queries of (gold in top k) / (gold for that query)
      micro  Σ(gold in top k) / Σ(gold), the paper's "proportion of evidence sentences
             within the top k retrieved sentences" — it downweights nothing, so queries
             with many evidence sentences (which are harder) carry their full weight
    On this data macro runs ~0.07 above micro at k=20, which is the whole of what once
    looked like beating the published numbers.
    """
    pos = {k: i for i, k in enumerate(ranked)}
    gp = sorted(pos.get(g, math.inf) for g in gold)
    first = gp[0] + 1 if gp and gp[0] != math.inf else None
    m = {'MRR': 1 / first if first else 0.}
    counts = {'gold_n': len(gold),
              # the paper evaluates MRR "on top 20 sentences": no gold inside 20 scores 0
              'rr20': 1 / first if first and first <= 20 else 0.}
    for k in KS:
        inside = sum(p < k for p in gp)
        m[f'Recall@{k}'] = inside / len(gold)
        m[f'Hit@{k}'] = float(inside > 0)
        m[f'AllGold@{k}'] = float(inside == len(gold))
        counts[f'hits@{k}'] = inside
    return m, counts


def pooled(rows, n_all):
    """The paper's definitions: recall pooled over evidence sentences, MRR over every
    citance — including the ones with no evidence, which contribute 0."""
    gold = sum(r['c']['gold_n'] for r in rows)
    out = {f'Recall@{k}_micro': sum(r['c'][f'hits@{k}'] for r in rows) / gold for k in KS}
    out['MRR_paper'] = sum(r['c']['rr20'] for r in rows) / n_all
    return out


def aggregate(rows, n_boot, seed, n_all):
    names = list(rows[0]['m'])
    point = {n: sum(r['m'][n] for r in rows) / len(rows) for n in names}
    point.update(pooled(rows, n_all))
    by_art = defaultdict(list)
    for r in rows:
        by_art[r['article']].append(r)
    arts = list(by_art)
    rng = random.Random(seed)
    samples = defaultdict(list)
    for _ in range(n_boot):
        pick = [r for a in rng.choices(arts, k=len(arts)) for r in by_art[a]]
        for n in names:
            samples[n].append(sum(r['m'][n] for r in pick) / len(pick))
        # scale the all-citance denominator with the resample so MRR_paper stays comparable
        for n, v in pooled(pick, len(pick) * n_all / len(rows)).items():
            samples[n].append(v)
    ci = {}
    for n in point:
        s = sorted(samples[n])
        ci[n] = [s[int(.025 * n_boot)], s[min(int(.975 * n_boot), n_boot - 1)]] if n_boot else None
    return point, ci, len(arts)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data-dir', type=Path, default=DEFAULT_DATA)
    p.add_argument('--split', default='test', choices=['train', 'dev', 'test'])
    p.add_argument('--ranker', default='bm25', choices=['bm25', 'bm25-monot5', 'file'])
    p.add_argument('--k1', type=float, default=0.9)
    p.add_argument('--b', type=float, default=0.4)
    p.add_argument('--rerank-depth', type=int, default=60, help='BM25 candidates passed to MonoT5')
    p.add_argument('--monot5-model', default='castorini/monot5-base-med-msmarco')
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--context', action='store_true', help='rerank sentence with prev/next neighbour')
    p.add_argument('--rankings-in', type=Path, help='JSONL of {id, ranked:[keys]} for --ranker file')
    p.add_argument('--rankings-out', type=Path, help='write {id, article, gold, ranked} JSONL')
    p.add_argument('--metrics-out', type=Path)
    p.add_argument('--resume', action='store_true',
                   help='reuse rankings already in --rankings-out and only score the remaining queries')
    p.add_argument('--bootstrap', type=int, default=2000)
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()

    queries = load_split(a.data_dir, a.split)
    no_gold = [q for q in queries if not q['gold']]
    gold_outside_pool = sum(bool(q['gold'] - set(q['keys'])) for q in queries if q['gold'])
    queries = [q for q in queries if q['gold']]

    external = None
    if a.ranker == 'file':
        if not a.rankings_in:
            p.error('--ranker file needs --rankings-in')
        external = {r['id']: r['ranked'] for r in read_jsonl(a.rankings_in)}
        missing = [q['id'] for q in queries if q['id'] not in external]
        if missing:
            raise SystemExit(f'{len(missing)} evaluable queries missing from {a.rankings_in}, e.g. {missing[:5]}')
    monot5 = None
    if a.ranker == 'bm25-monot5':
        monot5, device = make_monot5(a.monot5_model, a.batch_size)
        print(f'MonoT5 {a.monot5_model} on {device}', flush=True)

    done = {}
    if a.resume:
        if not a.rankings_out:
            p.error('--resume needs --rankings-out')
        if a.rankings_out.exists():
            done = {r['id']: r['ranked'] for r in read_jsonl(a.rankings_out)}
            print(f'resuming: {len(done)} of {len(queries)} queries already ranked', flush=True)

    rows = []
    if a.rankings_out:
        a.rankings_out.parent.mkdir(parents=True, exist_ok=True)
    out = a.rankings_out.open('a' if done else 'w', encoding='utf8') if a.rankings_out else None
    for n, q in enumerate(queries, 1):
        if q['id'] in done:
            rows.append({'id': q['id'], 'article': q['article'],
                         **dict(zip(('m', 'c'), query_metrics(done[q['id']], q['gold'])))})
            continue
        if external is not None:
            ranked = external[q['id']]
        else:
            order = rank_by(bm25_scores(q['query'], q['texts'], a.k1, a.b))
            if monot5:
                head = order[:a.rerank_depth]
                passages = [with_context(q['texts'], i) if a.context else q['texts'][i] for i in head]
                s = monot5(q['query'], passages)
                order = [head[i] for i in rank_by(s)] + order[a.rerank_depth:]
                if n % 50 == 0:
                    print(f'  {n}/{len(queries)}', flush=True)
            ranked = [q['keys'][i] for i in order]
        m, c = query_metrics(ranked, q['gold'])
        rows.append({'id': q['id'], 'article': q['article'], 'm': m, 'c': c})
        if out:
            out.write(json.dumps({'id': q['id'], 'article': q['article'], 'gold': sorted(q['gold']),
                                  'ranked': ranked}) + '\n')
            out.flush()  # so an interrupted run stays resumable
    if out:
        out.close()

    point, ci, n_art = aggregate(rows, a.bootstrap, a.seed, len(rows) + len(no_gold))
    settings = {k: v for k, v in vars(a).items() if k in ('split', 'ranker', 'k1', 'b', 'rerank_depth',
                                                          'monot5_model', 'context', 'rankings_in')}
    report = {'settings': {k: str(v) if isinstance(v, Path) else v for k, v in settings.items()},
              'queries_scored': len(rows), 'articles': n_art, 'skipped_no_gold': len(no_gold),
              'gold_outside_pool': gold_outside_pool,
              'mean_pool_size': sum(len(q['keys']) for q in queries) / len(queries),
              'metrics': point, 'ci95_article_bootstrap': ci}
    if a.metrics_out:
        a.metrics_out.parent.mkdir(parents=True, exist_ok=True)
        a.metrics_out.write_text(json.dumps(report, indent=2) + '\n')

    print(f"split={a.split} ranker={a.ranker} queries={len(rows)} articles={n_art} "
          f"skipped_no_gold={len(no_gold)} gold_outside_pool={gold_outside_pool} "
          f"pool_mean={report['mean_pool_size']:.0f}")
    fmt = lambda n: f"{point[n]:.3f}" + (f" [{ci[n][0]:.3f},{ci[n][1]:.3f}]" if ci[n] else '')
    print(f"MRR {fmt('MRR')} (per evaluable query)   "
          f"MRR_paper {fmt('MRR_paper')} (all {len(rows) + len(no_gold)} citances, truncated at 20)")
    print(f"{'k':>4}  {'Recall@k micro (paper)':<24}{'Recall@k macro':<24}{'Hit@k':<22}AllGold@k")
    for k in KS:
        print(f"{k:>4}  {fmt(f'Recall@{k}_micro'):<24}{fmt(f'Recall@{k}'):<24}"
              f"{fmt(f'Hit@{k}'):<22}{fmt(f'AllGold@{k}')}")


if __name__ == '__main__':
    main()
