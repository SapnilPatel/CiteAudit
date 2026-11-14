#!/usr/bin/env python3
"""Build the intermediate-training set: human-labelled biomedical claim verification, in NLI form.

Why this exists. Moving from BiomedBERT to an NLI-pretrained model was the largest gain in this
project (CV macro-F1 0.428 to 0.570, results/nli_classifier_v1.md): what the classifier needs is a
prior for "does this text support this statement". The NLI model learned that prior from general-
domain text (MNLI, FEVER, ANLI, LingNLI, WANLI). This adds a step between that model and our 2141
citations: the same three-way judgement, in the biomedical domain, labelled by people. The
synthetic data of results/synthetic_v1.md and v2 failed because it carried LLM artefacts; these
datasets were written and labelled by annotators.

  HealthVer (Sarrouti et al. 2021, CC BY 4.0): 14,330 claim-evidence pairs on COVID-19, with
      evidence from scientific abstracts, labelled Supports / Refutes / Neutral. All three
      splits are used: nothing here is evaluated on HealthVer.
  SciFact (Wadden et al. 2020, CC BY 4.0): scientific claims, each paired with the abstracts it
      cites. The premise is the whole abstract, not only the rationale sentences, so that, as in
      our task, the model must find the evidence among unrelated sentences. Each cited abstract
      is labelled SUPPORT or CONTRADICT if it holds a rationale, otherwise NOT ENOUGH INFO. The
      train and dev claims are used; test has no labels.

Labels are written in our names (Supports -> ACCURATE, Refutes -> NOT_ACCURATE, Neutral ->
NOT ENOUGH INFO). Downloads the data, so it needs network access. Nothing here touches
Citation-Integrity.

    python scripts/build_pretrain_data.py --out data_train/pretrain/biomed_nli.jsonl
"""
import argparse
import csv
import io
import json
import random
import tarfile
import urllib.request
from collections import Counter
from pathlib import Path

HEALTHVER = 'https://raw.githubusercontent.com/sarrouti/HealthVer/master/data/healthver_{}.csv'
SCIFACT = 'https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz'
HV_LABEL = {'Supports': 'ACCURATE', 'Refutes': 'NOT_ACCURATE', 'Neutral': 'NOT ENOUGH INFO'}
SF_LABEL = {'SUPPORT': 'ACCURATE', 'CONTRADICT': 'NOT_ACCURATE'}


def fetch(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'citeaudit'}),
                                timeout=120) as r:
        return r.read()


def healthver():
    rows = []
    for split in ('train', 'dev', 'test'):
        for r in csv.DictReader(io.StringIO(fetch(HEALTHVER.format(split)).decode('utf8'))):
            if r['label'] in HV_LABEL and r['evidence'].strip() and r['claim'].strip():
                rows.append({'premise': r['evidence'].strip(), 'hypothesis': r['claim'].strip(),
                             'label': HV_LABEL[r['label']], 'source': 'healthver'})
    return rows


def scifact():
    tar = tarfile.open(fileobj=io.BytesIO(fetch(SCIFACT)), mode='r:gz')
    read = lambda name: [json.loads(x) for x in tar.extractfile(name).read().decode('utf8').splitlines() if x.strip()]
    corpus = {d['doc_id']: d for d in read('data/corpus.jsonl')}
    rows = []
    for split in ('train', 'dev'):
        for c in read(f'data/claims_{split}.jsonl'):
            for doc in c['cited_doc_ids']:
                d = corpus.get(int(doc))
                if not d:
                    continue
                ev = c['evidence'].get(str(doc))
                label = SF_LABEL[ev[0]['label']] if ev else 'NOT ENOUGH INFO'
                rows.append({'premise': ' '.join(s.strip() for s in d['abstract']),
                             'hypothesis': c['claim'].strip(), 'label': label, 'source': 'scifact'})
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()
    rows = healthver() + scifact()
    random.Random(a.seed).shuffle(rows)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf8')
    print(f'wrote {a.out}: {len(rows)} pairs')
    for src in ('healthver', 'scifact'):
        print(f'  {src}: {dict(Counter(r["label"] for r in rows if r["source"] == src))}')


if __name__ == '__main__':
    main()
