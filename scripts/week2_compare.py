#!/usr/bin/env python3
"""Compare actual, externally generated MultiVerS predictions (not placeholder scores)."""
import argparse,csv,json
from pathlib import Path
from sklearn.metrics import f1_score, classification_report
from evaluate_classification import norm,ORDER

def load(path):
    if path.suffix=='.jsonl':
        return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    return list(csv.DictReader(path.open(newline='')))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--retrieved',required=True,type=Path,help='CSV/JSONL with id,gold,prediction')
    p.add_argument('--oracle',required=True,type=Path,help='CSV/JSONL with id,gold,prediction')
    p.add_argument('--out',default='results/week2_verified_comparison.json',type=Path)
    a=p.parse_args()
    def parse(path):
        rows=load(path);m={}
        for row in rows:
            if not {'id','gold','prediction'}<=row.keys():raise ValueError('Each prediction needs id,gold,prediction')
            key=str(row['id'])
            if key in m:raise ValueError('Duplicate id '+key)
            m[key]=(norm(row['gold']),norm(row['prediction']))
            if not set(m[key])<=set(ORDER):raise ValueError('Unknown label '+str(m[key]))
        return m
    retr=parse(a.retrieved);ora=parse(a.oracle)
    if not retr or set(retr)!=set(ora):raise ValueError('Nonempty predictions on IDENTICAL citation IDs required')
    if any(retr[k][0]!=ora[k][0] for k in retr):raise ValueError('Gold labels differ between runs')
    res={'n':len(retr),'sources':{'retrieved':str(a.retrieved),'oracle':str(a.oracle)}}
    for name,data in [('retrieved',retr),('oracle',ora)]:
        y=[v[0] for v in data.values()];pred=[v[1] for v in data.values()]
        res[name]={'micro_f1':f1_score(y,pred,labels=ORDER,average='micro',zero_division=0),
                   'macro_f1':f1_score(y,pred,labels=ORDER,average='macro',zero_division=0),
                   'per_class':classification_report(y,pred,labels=ORDER,output_dict=True,zero_division=0)}
    res['oracle_minus_retrieved_micro_f1']=res['oracle']['micro_f1']-res['retrieved']['micro_f1']
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(res,indent=2)+'\n');print(json.dumps(res,indent=2))
if __name__=='__main__':main()
