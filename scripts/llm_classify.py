#!/usr/bin/env python3
"""Classify citations with an open instruction-tuned LLM, reading off its probability per label.

Why this exists. Every trained classifier here fails the same citations (results/error_analysis.md):
OVERSIMPLIFY and CONTRADICT, where the source says something close to the claim and only a careful
reading shows the claim stretches it. Those need reasoning about what the source does and does not
say, which a 7B instruction-tuned LLM brings with no task training at all, while BiomedBERT has
2141 examples and MultiVerS a fact-verification prior. The LLM reads the citing sentence and the
top-k retrieved sentences (an export from `export_for_multivers.py`) and answers A, B or C.

No text is generated. The prompt ends where the answer letter goes, one forward pass gives the
next-token logits, and the probabilities of A, B and C (renormalised over the three) are the
prediction. That is one pass per citation, deterministic, and it yields calibrated-enough
probabilities for `ensemble_classifiers.py` to combine with the other models. Output is in
`predict_multivers.py`'s format, so `score_multivers.py` scores it directly.

The label descriptions follow the paper's 9 -> 3 collapse (ACCURATE, INDIRECT -> ACCURATE;
CONTRADICT, NOT_SUBSTANTIATE, OVERSIMPLIFY, MISQUOTE, ETIQUETTE -> NOT ACCURATE; IRRELEVANT -> NOT
ENOUGH INFO), worded from annotated train examples since the release carries no guideline text.
Change the prompt only by adding a new --prompt version, judged on dev, never on test.

    python scripts/llm_classify.py --data-dir data_train/target/ours_20 --split dev \\
        --model Qwen/Qwen2.5-7B-Instruct --load-4bit --out results/classification/llm/dev.jsonl
"""
import argparse
import json
import re
import time
from pathlib import Path

LETTERS = {'A': 'ACCURATE', 'B': 'NOT_ACCURATE', 'C': 'NOT ENOUGH INFO'}

SYSTEM = ('You are an expert biomedical reviewer who checks whether citations in scientific papers '
          'accurately represent the papers they cite.')

PROMPTS = {
    'v1': """A sentence from a biomedical paper cites another paper. Below are the sentence and excerpts from the cited paper. Decide how the citation represents the cited paper.

In the sentence, [CITED] marks the citation you are checking, [CITED, with others] marks it when it is one of several references at that position, and [other] marks citations of different papers, which you should ignore.

Options:
A) ACCURATE: the cited paper supports the statement, either directly or by reporting it while citing other work for it.
B) NOT ACCURATE: the statement misrepresents the cited paper. This includes: it contradicts the paper; the paper is on the topic but does not support the statement; the statement oversimplifies or overstates the paper (drops conditions or qualifiers, generalises from a specific population, model or setting, or turns an association into a cause); it gets numbers or details wrong; or the paper is cited for a point it does not actually make.
C) IRRELEVANT: the cited paper has nothing to do with what the statement says, so it can neither support nor contradict it.

The excerpts are the sentences of the cited paper most related to the statement, not the whole paper.

Citing sentence:
{claim}

Excerpts from the cited paper:
{evidence}

Answer with a single letter: A, B or C.""",
}

MARKERS = {'multi_cit': '[CITED, with others]', 'cit': '[CITED]', 'other_cit': '[other]'}
# A marker, with any brackets or parentheses the citing paper put around it: "[<|cit|>]",
# "(<|cit|>)" and a bare "<|cit|>" all become "[CITED]", never "[[CITED]]".
MARKER_RE = re.compile(r'[\[(]?\s*<\|(multi_cit|other_cit|cit)\|>\s*[\])]?')


def read_jsonl(path):
    with open(path, encoding='utf8') as f:
        return [json.loads(x) for x in f if x.strip()]


def readable(claim):
    claim = MARKER_RE.sub(lambda m: ' ' + MARKERS[m.group(1)] + ' ', claim)
    return re.sub(r'\s+([.,;:])', r'\1', re.sub(r'\s+', ' ', claim)).strip()


def build_prompt(tok, version, claim, sentences):
    evidence = '\n'.join(f'{i}. {s}' for i, s in enumerate(sentences, 1))
    msgs = [{'role': 'system', 'content': SYSTEM},
            {'role': 'user', 'content': PROMPTS[version].format(claim=readable(claim),
                                                                evidence=evidence)}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def letter_ids(tok):
    """Token ids that can start each answer, with and without a leading space."""
    ids = {}
    for letter in LETTERS:
        cands = {tok.encode(v, add_special_tokens=False)[0] for v in (letter, ' ' + letter)}
        ids[letter] = sorted(cands)
    return ids


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data-dir', type=Path, required=True,
                   help='an export directory: claims_<split>.jsonl + corpus.jsonl')
    p.add_argument('--split', default='dev', choices=['train', 'dev', 'test'])
    p.add_argument('--model', default='Qwen/Qwen2.5-7B-Instruct')
    p.add_argument('--prompt', default='v1', choices=sorted(PROMPTS))
    p.add_argument('--load-4bit', action='store_true', help='bitsandbytes NF4; needs CUDA')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--limit', type=int)
    p.add_argument('--device', default='auto')
    a = p.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    claims = read_jsonl(a.data_dir / f'claims_{a.split}.jsonl')[:a.limit]
    corpus = {d['doc_id']: d['abstract'] for d in read_jsonl(a.data_dir / 'corpus.jsonl')}
    done = {}
    if a.resume and a.out.exists():
        done = {r['id']: r for r in read_jsonl(a.out)}
        print(f'resuming: {len(done)} of {len(claims)} already classified', flush=True)

    tok = AutoTokenizer.from_pretrained(a.model)
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
        dev = a.device if a.device != 'auto' else (
            'cuda' if torch.cuda.is_available() else
            'mps' if torch.backends.mps.is_available() else 'cpu')
        model.to(dev)
    model.eval()
    device = next(model.parameters()).device
    ids = letter_ids(tok)
    print(f'{a.model} on {device}; prompt {a.prompt}; answer tokens {ids}', flush=True)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    out = a.out.open('a' if done else 'w', encoding='utf8')
    t0, n_new, lengths = time.time(), 0, []
    for n, c in enumerate(claims, 1):
        if c['id'] in done:
            continue
        doc_id = c['cited_doc_ids'][0]
        gold = next(iter(c['evidence'].values()))[0]['label'] if c.get('evidence') else 'NOT ENOUGH INFO'
        enc = tok(build_prompt(tok, a.prompt, c['claim'], corpus[doc_id]),
                  return_tensors='pt', add_special_tokens=False).to(device)
        lengths.append(enc['input_ids'].shape[1])
        with torch.no_grad():
            logits = model(**enc).logits[0, -1].float()
        # Log-sum-exp over each letter's variants, then renormalise over the three letters.
        score = {L: torch.logsumexp(logits[v], 0) for L, v in ids.items()}
        probs = torch.softmax(torch.stack([score[L] for L in LETTERS]), 0).tolist()
        mass = torch.softmax(logits, 0)[[i for v in ids.values() for i in v]].sum().item()
        out.write(json.dumps({'id': c['id'], 'doc_id': doc_id, 'gold': gold,
                              'prediction': LETTERS[max(LETTERS, key=lambda L: probs['ABC'.index(L)])],
                              'probs': {LETTERS[L]: round(probs[i], 5) for i, L in enumerate(LETTERS)},
                              'answer_mass': round(mass, 4),
                              'n_tokens': lengths[-1]}) + '\n')
        out.flush()
        n_new += 1
        if n_new % 25 == 0:
            rate = n_new / (time.time() - t0)
            print(f'  {n}/{len(claims)}  {rate:.2f} citations/s  '
                  f'eta {(len(claims) - n) / rate / 60:.0f} min', flush=True)
    out.close()
    if lengths:
        print(f'prompt tokens: mean {sum(lengths) / len(lengths):.0f}, max {max(lengths)}')
    print(f'wrote {a.out}  ({len(read_jsonl(a.out))} predictions)')


if __name__ == '__main__':
    main()
