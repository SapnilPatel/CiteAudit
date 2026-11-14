#!/usr/bin/env python3
"""Generate synthetic misrepresenting citations from accurate ones, as extra NOT ACCURATE training data.

Why this exists. The NLI classifier (results/nli_classifier_v1.md) is the best system so far, but
its NOT ACCURATE F1 on test is 0.44, barely above the paper's 0.43: it has learned what support
looks like and very little about how a citation goes wrong. Train has about 330 real examples of
content misrepresentation and 1308 accurate citations. Each accurate citation is a free template:
an open instruction-tuned LLM rewrites it, minimally, so that it misrepresents the same evidence in
one specific way, and the rewrite is paired with the original's premise. The model then sees the
same evidence with a faithful and an unfaithful statement, which is the contrast the task needs.

The rewrite types follow the paper's NOT ACCURATE sub-labels (OVERSIMPLIFY -> overstate, causal,
generalise; MISQUOTE -> detail; CONTRADICT -> contradict; NOT_SUBSTANTIATE -> unsupported, which is
the commonest real one and gets a double share). ETIQUETTE (citing a paper for what it only
reports from others) is not synthesised: no rewrite of the sentence can produce it.

Only accurate citations whose gold evidence our retrieval kept are used, so the evidence the rewrite
contradicts is in the premise the classifier will read. The LLM sees the citing sentence (markers
removed, as the classifier sees it) and those gold sentences. Every row keeps `source_id`, so
`train_nli.py --synthetic` can drop rows whose source citation is in a held-out fold. Train only:
dev and test are never touched.

    python scripts/generate_misrep.py --data-dir data_train/target/ours_20r_train_cv \\
        --model Qwen/Qwen2.5-7B-Instruct --load-4bit --out data_train/synthetic/misrep.jsonl
"""
import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm_classify import read_jsonl  # noqa: E402
from train_nli import statement  # noqa: E402

TYPES = {
    'overstate': 'OVERSTATE: remove the hedges and qualifiers (may, suggest, in some cases, '
                 'partially, preliminary, in this cohort) or turn a tentative finding into a '
                 'definitive, proven or universal one.',
    'causal': 'CAUSATION: turn an association, correlation or co-occurrence into a causal claim '
              '(X causes, drives or prevents Y). If the sentence is already causal, overstate how '
              'strong or direct the effect is instead.',
    'generalise': 'OVERGENERALISE: extend the finding beyond the population, model or setting the '
                  'paper studied, for example from cells or mice to humans, from one subgroup or '
                  'disease to all patients, or from one drug to its whole class.',
    'detail': 'WRONG DETAIL: change one specific detail so it no longer matches the paper: a '
              'number, percentage or dose, the direction or size of a change, the gene, drug, cell '
              'type or condition involved, or what was compared with what.',
    'contradict': 'CONTRADICT: reverse the finding so the sentence says the opposite of what the '
                  'paper reports (increases becomes decreases, effective becomes ineffective, '
                  'associated becomes not associated).',
    'unsupported': 'UNSUPPORTED CLAIM: keep the topic but make the sentence claim something the '
                   'paper does not report, such as an additional outcome, a mechanism, a clinical '
                   'recommendation or a broader conclusion that the excerpts do not support.',
}
SCHEDULE = ['unsupported', 'overstate', 'causal', 'unsupported', 'generalise', 'detail', 'contradict']

SYSTEM = ('You are an expert biomedical editor who builds training data for a system that detects '
          'citations that misrepresent the papers they cite.')

PROMPT = """Below is a sentence from a biomedical paper that ACCURATELY cites another paper, followed by excerpts from the cited paper. Rewrite the sentence so that it MISREPRESENTS the cited paper in this specific way:

{instruction}

Rules:
- Change as little as possible. Keep the wording, style and length of the original, so the rewrite still reads like a normal sentence from a paper and the error is subtle, the way real citation errors are.
- Given the excerpts, the rewrite must be false or unsupported, while staying on the same topic.
- Output only the rewritten sentence, with no explanation, quotes or label.

Original sentence:
{claim}

Excerpts from the cited paper:
{evidence}"""

PREFIX_RE = re.compile(r'^\s*(rewritten( sentence)?|rewrite|misrepresent\w*( sentence)?|output|answer)\s*:\s*',
                       re.I)


# Outputs that talk about the task instead of doing it ("Misrepresents citation: ...").
META_RE = re.compile(r'\b(misrepresent\w*|rewrit\w*|original sentence|cited paper|excerpts?|'
                     r'the following sentence)\b', re.I)


def norm(s):
    return re.sub(r'[^a-z0-9]+', ' ', s.lower()).strip()


def clean(text, original):
    """The rewritten sentence, or None if the output is unusable."""
    lines = [x.strip() for x in text.strip().splitlines() if x.strip()]
    if not lines:
        return None
    s = PREFIX_RE.sub('', lines[0]).strip().strip('"“”').strip()
    if norm(s) == norm(original) or not (0.5 * len(original) <= len(s) <= 2 * len(original) + 40):
        return None
    if META_RE.search(s) and not META_RE.search(original):
        return None
    # A minimal edit keeps most of the original's words; a new sentence is a different citation.
    a, b = set(norm(s).split()), set(norm(original).split())
    if len(a & b) / len(a | b) < 0.3:
        return None
    return s


def sources(data_dir, fine_labels):
    """Accurate citations whose gold evidence the export kept: (id, statement, gold sentences)."""
    fine = json.loads(Path(fine_labels).read_text(encoding='utf8'))
    corpus = {d['doc_id']: d['abstract'] for d in read_jsonl(Path(data_dir) / 'corpus.jsonl')}
    out = []
    for c in read_jsonl(Path(data_dir) / 'claims_train.jsonl'):
        if fine[str(c['id'])] != 'ACCURATE':
            continue
        doc = c['cited_doc_ids'][0]
        idx = sorted({i for items in c.get('evidence', {}).values() for e in items for i in e['sentences']})
        if idx:
            out.append((c['id'], statement(c['claim']), [corpus[doc][i] for i in idx[:6]]))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data-dir', type=Path, default=Path('data_train/target/ours_20r_train_cv'))
    p.add_argument('--fine-labels', type=Path, default=Path('data_train/fine_labels/train.json'))
    p.add_argument('--model', default='Qwen/Qwen2.5-7B-Instruct')
    p.add_argument('--load-4bit', action='store_true', help='bitsandbytes NF4; needs CUDA')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--max-new-tokens', type=int, default=160)
    p.add_argument('--temperature', type=float, default=0.7)
    p.add_argument('--retries', type=int, default=1, help='extra attempts for unusable outputs')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--limit', type=int)
    p.add_argument('--device', default='auto')
    a = p.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rng = random.Random(a.seed)
    todo = sources(a.data_dir, a.fine_labels)[:a.limit]
    # One type per citation, spread evenly over SCHEDULE but not aligned with citation order,
    # which runs article by article.
    types = [SCHEDULE[i % len(SCHEDULE)] for i in range(len(todo))]
    rng.shuffle(types)
    jobs = {cid: (claim, ev, t) for (cid, claim, ev), t in zip(todo, types)}
    done = {r['source_id'] for r in read_jsonl(a.out)} if a.out.exists() else set()
    tried = {}
    print(f'{len(jobs)} accurate citations with retrieved gold evidence; {len(done)} already done',
          flush=True)

    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = 'left'
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kw = {}
    if a.load_4bit:
        from transformers import BitsAndBytesConfig
        kw['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.float16)
        kw['device_map'] = 'auto'
    else:
        kw['torch_dtype'] = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(a.model, **kw)
    if not a.load_4bit:
        model.to(a.device if a.device != 'auto' else 'cuda' if torch.cuda.is_available()
                 else 'mps' if torch.backends.mps.is_available() else 'cpu')
    model.eval()
    device = next(model.parameters()).device
    stop = model.generation_config.eos_token_id
    stop = sorted({*(stop if isinstance(stop, list) else [stop]), tok.eos_token_id, tok.pad_token_id})

    def prompt(cid):
        claim, ev, t = jobs[cid]
        msgs = [{'role': 'system', 'content': SYSTEM},
                {'role': 'user', 'content': PROMPT.format(
                    instruction=TYPES[t], claim=claim,
                    evidence='\n'.join(f'{i}. {s}' for i, s in enumerate(ev, 1)))}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    out = a.out.open('a', encoding='utf8')
    t0, n_new, rejected = time.time(), 0, 0
    while True:
        queue = [c for c in jobs if c not in done and tried.get(c, 0) <= a.retries]
        if not queue:
            break
        for i in range(0, len(queue), a.batch):
            ids = queue[i:i + a.batch]
            enc = tok([prompt(c) for c in ids], return_tensors='pt', padding=True,
                      add_special_tokens=False).to(device)
            torch.manual_seed(a.seed + sum(tried.get(c, 0) for c in ids) + ids[0])
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=a.max_new_tokens, do_sample=True,
                                     temperature=a.temperature, top_p=0.95, pad_token_id=tok.pad_token_id)
            new = gen[:, enc['input_ids'].shape[1]:]
            texts = tok.batch_decode(new, skip_special_tokens=True)
            # An output that never reached an end token was cut at --max-new-tokens: a truncated
            # sentence, not a misrepresentation.
            finished = torch.isin(new, torch.tensor(stop, device=new.device)).any(1).tolist()
            for cid, text, fin in zip(ids, texts, finished):
                tried[cid] = tried.get(cid, 0) + 1
                claim, _, t = jobs[cid]
                s = clean(text, claim) if fin else None
                if s is None:
                    rejected += 1
                    continue
                out.write(json.dumps({'source_id': cid, 'type': t, 'statement': s,
                                      'original': claim, 'model': a.model}) + '\n')
                out.flush()
                done.add(cid)
                n_new += 1
            rate = n_new / (time.time() - t0)
            left = sum(c not in done for c in jobs)
            print(f'  {len(done)}/{len(jobs)} done, {rejected} outputs rejected, '
                  f'{rate * 60:.0f} per min' + (f', eta {left / rate / 60:.0f} min' if rate else ''),
                  flush=True)
    out.close()
    rows = read_jsonl(a.out)
    print(f'wrote {a.out}: {len(rows)} synthetic citations, '
          f'{sum(c not in done for c in jobs)} sources gave no usable rewrite')
    for r in rows[:6]:
        print(f'\n[{r["type"]}]\n  original: {r["original"]}\n  rewrite:  {r["statement"]}')


if __name__ == '__main__':
    main()
