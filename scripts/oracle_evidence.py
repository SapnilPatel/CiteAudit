#!/usr/bin/env python3
"""Oracle-evidence diagnostic: score one trained classifier on dev and test with retrieved, gold-first
and gold-only evidence (same claims, same model, 20-sentence budget).

    python scripts/oracle_evidence.py --model Sapnil/citeaudit-nli --out results/classification/oracle_evidence
"""
import json, sys, argparse
from pathlib import Path
sys.path.insert(0, 'scripts')
from train_nli import read_jsonl, statement, encode, predict, macro_f1, label_head, LABELS
p = argparse.ArgumentParser(); p.add_argument('--model'); p.add_argument('--out')
a = p.parse_args()
SETS = {'dev': ('data_train/target/ours_20r', 'data_train/target/oracle_20'),
        'test': ('data_train/target/ours_20r_test', 'data_train/target/oracle_20_test')}
def rows(split):
    ours, orc = SETS[split]
    oc = {c['id']: c for c in read_jsonl(Path(orc) / f'claims_{split}.jsonl')}
    ocorp = {d['doc_id']: d['abstract'] for d in read_jsonl(Path(orc) / 'corpus.jsonl')}
    rcorp = {d['doc_id']: d['abstract'] for d in read_jsonl(Path(ours) / 'corpus.jsonl')}
    out = {'retrieved': [], 'gold_first': [], 'gold_only': []}
    for c in read_jsonl(Path(ours) / f'claims_{split}.jsonl'):
        doc = c['cited_doc_ids'][0]
        gold = next(iter(c['evidence'].values()))[0]['label'] if c.get('evidence') else 'NOT ENOUGH INFO'
        o = oc[c['id']]; assert o['claim'] == c['claim']
        gi = sorted({i for v in o.get('evidence', {}).values() for e in v for i in e['sentences']})
        gs = [ocorp[o['cited_doc_ids'][0]][i] for i in gi]
        ret = rcorp[doc]
        base = {'id': c['id'], 'doc_id': doc, 'hypothesis': statement(c['claim']), 'gold': gold}
        out['retrieved'].append({**base, 'premise': ' '.join(ret)})
        out['gold_first'].append({**base, 'premise': ' '.join((gs + [s for s in ret if s not in gs])[:20])})
        out['gold_only'].append({**base, 'premise': ' '.join(gs) if gs else ' '.join(ret)})
    return out
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
dev = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForSequenceClassification.from_pretrained(a.model, dtype=torch.float32).to(dev).eval()
head = label_head(model)
class A: max_len = 512; eval_batch = 8; amp = False
res = {}
Path(a.out).mkdir(parents=True, exist_ok=True)
for split in SETS:
    for arm, rs in rows(split).items():
        preds = predict(model, encode(tok, rs, A), A, tok, torch, dev, head)
        f1, acc = macro_f1(preds)
        with open(f'{a.out}/{split}_{arm}.jsonl', 'w') as f:
            for r in preds: f.write(json.dumps({k: v for k, v in r.items() if k != 'enc'}) + '\n')
        res[f'{split}_{arm}'] = (round(acc, 3), round(f1, 3)); print(split, arm, res[f'{split}_{arm}'], flush=True)
json.dump(res, open(f'{a.out}/summary.json', 'w'), indent=1)
