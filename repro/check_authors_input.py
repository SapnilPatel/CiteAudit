#!/usr/bin/env python3
"""Compare the authors' released classifier input (Drive folder scifact_10) with the official data.

Joins each released claim to the official claims-<split>.jsonl by claim text, then reports how many
gold evidence sentences the released 5-sentence documents contain, i.e. their Recall@5.
"""
import json, re, sys, collections
def rd(p): return [json.loads(l) for l in open(p) if l.strip()]
D = sys.argv[1] if len(sys.argv) > 1 else 'checkpoints/drive/scifact_10'
M = 'upstream/Citation-Integrity/Data/multivers-format'
norm = lambda s: re.sub(r'\W+', '', s.lower())
corp = {d['doc_id']: d['abstract'] for d in rd(f'{M}/corpus.jsonl')}
A = {d['doc_id']: d['abstract'] for d in rd(f'{D}/corpus.jsonl')}
for s in ['train', 'dev', 'test']:
    by_text = collections.defaultdict(list)
    for o in rd(f'{M}/claims-{s}.jsonl'):
        by_text[o['claim']].append(o)
    ac = rd(f'{D}/claims_{s}.jsonl')
    matched = ambiguous = golds = goldin = 0
    for c in ac:
        cands = by_text.get(c['claim'], [])
        if not cands:
            continue
        matched += 1
        doc = {norm(x) for x in A[c['cited_doc_ids'][0]]}
        # several official rows can share a citance text; take the best-covered one
        best = None
        for o in cands:
            g = [norm(corp[int(d)][i]) for d, evs in o.get('evidence', {}).items() for e in evs for i in e['sentences']]
            hit = sum(x in doc for x in g)
            if best is None or hit > best[0]:
                best = (hit, len(g))
        ambiguous += len(cands) > 1
        goldin += best[0]; golds += best[1]
    print(f'{s}: released {len(ac)}, matched by text {matched}, ambiguous {ambiguous}, '
          f'gold sentences inside the 5-sentence docs {goldin}/{golds} = {goldin/max(golds,1):.3f}')
