#!/usr/bin/env python3
"""Week 1-2 evidence retrieval and oracle-input preparation.

Input: official SciFact-style claims_*.jsonl and corpus_*.jsonl.
Each claim: id, claim, evidence:{doc_id:[{sentences:[...],label:...}]}.
Each document: doc_id, title, abstract (or sentences).
Output is NOT a MultiVerS prediction or a reproduced paper baseline.
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path
import numpy as np


def read_jsonl(path):
    with open(path, encoding='utf8') as f:
        return [json.loads(x) for x in f if x.strip()]


def sent_list(doc):
    for key in ('sentences','abstract'):
        val=doc.get(key)
        if isinstance(val,list):return [str(x) for x in val]
    raise ValueError('Corpus document lacks sentences/abstract list: '+str(doc.get('doc_id')))


def tok(s):return re.findall(r"[a-z0-9]+", s.lower())


def bm25(query, texts, k1=1.5, b=0.75):
    docs=[tok(x) for x in texts]
    n=len(docs)
    if not n:return []
    lens=[len(d) for d in docs];avg=max(sum(lens)/n,1)
    df=Counter(w for d in docs for w in set(d))
    scores=[]
    for d,ln in zip(docs,lens):
        freq=Counter(d);score=0.
        for w in set(tok(query)):
            if w not in df:continue
            idf=np.log(1+(n-df[w]+.5)/(df[w]+.5))
            tf=freq[w];score+=idf*tf*(k1+1)/(tf+k1*(1-b+b*ln/avg)) if tf else 0
        scores.append(float(score))
    return sorted(range(n), key=lambda i:(-scores[i],i))


def evidence_for(claim, docid):
    e=claim.get('evidence',{})
    items=e.get(str(docid), e.get(docid,[])) if isinstance(e,dict) else []
    ids=[]
    for item in items:
        if isinstance(item,dict):ids.extend(item.get('sentences',[]))
    return sorted({int(x) for x in ids})


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--claims',type=Path,required=True)
    p.add_argument('--corpus',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--top-k',type=int,default=20)
    p.add_argument('--mode',choices=['bm25','bm25-monot5'],default='bm25')
    p.add_argument('--monot5-model',default='castorini/monot5-base-msmarco')
    p.add_argument('--rerank-candidates',type=int,default=100)
    a=p.parse_args()
    if a.mode=='bm25-monot5':
        try:
            import torch
            from transformers import AutoTokenizer,AutoModelForSeq2SeqLM
        except ImportError as exc:raise SystemExit('MonoT5 requires pip install transformers sentencepiece torch') from exc
        tokenizer=AutoTokenizer.from_pretrained(a.monot5_model)
        model=AutoModelForSeq2SeqLM.from_pretrained(a.monot5_model).eval()
        true_id=tokenizer.encode('true',add_special_tokens=False)[0]
        false_id=tokenizer.encode('false',add_special_tokens=False)[0]
        def rerank(query,texts,indices):
            scored=[]
            for ix in indices:
                encoded=tokenizer('Query: '+query+' Document: '+texts[ix]+' Relevant:',return_tensors='pt',max_length=512,truncation=True)
                with torch.inference_mode():
                    dec=torch.tensor([[model.config.decoder_start_token_id or tokenizer.pad_token_id]])
                    logits=model(**encoded,decoder_input_ids=dec).logits[0,-1,[true_id,false_id]]
                    score=torch.softmax(logits,dim=0)[0].item()
                scored.append((ix,score))
            return [i for i,_ in sorted(scored,key=lambda x:-x[1])]
    else:rerank=None
    corpus={str(x['doc_id']):x for x in read_jsonl(a.corpus)}
    claims=read_jsonl(a.claims)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    counts={k:0 for k in (1,5,10,20)};mrr=0;eligible=0;missing=0
    with a.out.open('w',encoding='utf8') as f:
        for claim in claims:
            did=claim.get('doc_id',claim.get('cited_doc_id'))
            if did is None:
                ev=claim.get('evidence',{})
                if isinstance(ev,dict) and len(ev)==1:did=next(iter(ev))
            if did is None or str(did) not in corpus:
                missing+=1;continue
            doc=corpus[str(did)];sents=sent_list(doc)
            query=str(claim.get('claim',claim.get('citance',claim.get('text',''))))
            if not query:missing+=1;continue
            gold=evidence_for(claim,str(did))
            rank=bm25(query,sents)
            if rerank:rank=rerank(query,sents,rank[:a.rerank_candidates])
            retrieved=rank[:a.top_k]
            row={'id':claim.get('id'),'doc_id':did,'query':query,'gold':gold,'retrieved':retrieved,
                 'retrieved_sentences':[sents[i] for i in retrieved],
                 'gold_sentences':[sents[i] for i in gold if 0<=i<len(sents)],
                 'evidence_annotations':claim.get('evidence',{}),'mode':a.mode}
            f.write(json.dumps(row,ensure_ascii=False)+'\n')
            if gold:
                eligible+=1;g=set(gold)
                for k in counts:
                    counts[k]+=bool(g.intersection(retrieved[:k]))
                first=next((i for i,x in enumerate(retrieved,1) if x in g),None)
                if first:mrr+=1/first
    metrics={'mode':a.mode,'total_claims':len(claims),'evaluated_with_gold':eligible,'skipped_missing_doc_or_query':missing,
            **{f'Recall@{k}':counts[k]/eligible if eligible else None for k in counts},
            'MRR':mrr/eligible if eligible else None}
    a.out.with_suffix('.metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    print(json.dumps(metrics,indent=2))

if __name__=='__main__':main()
