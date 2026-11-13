#!/usr/bin/env python3
"""Fine-tune an NLI-pretrained cross-encoder as the citation classifier, keeping its NLI head.

Why this exists. 76% of the best system's test errors are one judgement: given evidence that
exists, is the citation faithful to it (results/diagnosis_v1.md)? That is natural-language
inference, and every model so far learned it from about 330 misrepresenting citations. A model
already trained on about a million NLI and fact-verification pairs separates faithful from
misrepresented dev citations as well as MultiVerS with no training on this data at all (AUC
0.689 against 0.683), and BiomedBERT, the only model here without such a prior, does worst.

So this keeps everything the NLI model knows. The input is its own format, premise = evidence and
hypothesis = the citing sentence (citation markers removed), and its existing 3-way head is
reused rather than replaced, with the labels aligned to their NLI meaning:

  ACCURATE         <- entailment     the evidence supports the statement
  NOT_ACCURATE     <- contradiction  the statement misrepresents it
  NOT ENOUGH INFO  <- neutral        the source does not bear on it

`--epochs 0` therefore scores the NLI model zero-shot on exactly the same footing.

The evidence is an export in rank order (`export_for_multivers.py --order rank`), so when the
premise is cut at --max-len it loses the lowest-ranked sentences. Cross-validation (--folds),
epoch choice on --dev (including epoch 0), and the output format are `train_classifier.py`'s, so
the two models are compared under one protocol and score_multivers.py / ensemble_classifiers.py
read this one unchanged.

`--synthetic` adds rewritten accurate citations (`generate_misrep.py`) as extra NOT ACCURATE
training rows, each with its source citation's premise. In cross-validation a synthetic row is
used only if its source citation is in the training folds, so nothing about a held-out citation
reaches training. Synthetic rows are never evaluated on.

    python scripts/train_nli.py --train data_train/target/ours_20r_train_cv \\
        --dev data_train/target/ours_20r --out checkpoints/nli --pred-dir results/classification/nli \\
        --predict test=data_train/target/ours_20r_test --amp
"""
import argparse
import json
import math
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm_classify import MARKER_RE, read_jsonl  # noqa: E402
from train_classifier import fold_articles  # noqa: E402

LABELS = ['ACCURATE', 'NOT_ACCURATE', 'NOT ENOUGH INFO']
NLI_NAME = {'ACCURATE': 'entailment', 'NOT_ACCURATE': 'contradiction', 'NOT ENOUGH INFO': 'neutral'}
DEFAULT_MODEL = 'MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli'


def statement(claim):
    """The citing sentence as an NLI hypothesis: citation markers removed, whitespace tidied."""
    s = MARKER_RE.sub(' ', claim)
    return re.sub(r'\s+([.,;:])', r'\1', re.sub(r'\s+', ' ', s)).strip()


def load_export(d, split, limit=None):
    claims = read_jsonl(Path(d) / f'claims_{split}.jsonl')[:limit]
    corpus = {x['doc_id']: x['abstract'] for x in read_jsonl(Path(d) / 'corpus.jsonl')}
    rows = []
    for c in claims:
        doc = c['cited_doc_ids'][0]
        gold = next(iter(c['evidence'].values()))[0]['label'] if c.get('evidence') else 'NOT ENOUGH INFO'
        rows.append({'id': c['id'], 'doc_id': doc, 'premise': ' '.join(corpus[doc]),
                     'hypothesis': statement(c['claim']), 'gold': gold})
    return rows


def encode(tok, rows, a):
    for r in rows:
        # longest_first cuts the premise (rank order, so its weakest sentences go first) and only
        # touches the statement in the rare case it alone nears the limit; only_first would fail.
        r['enc'] = tok(r['premise'], r['hypothesis'], truncation='longest_first', max_length=a.max_len)
    return rows


def batches(rows, size, tok, torch, device):
    for i in range(0, len(rows), size):
        b = rows[i:i + size]
        yield b, tok.pad([r['enc'] for r in b], return_tensors='pt').to(device)


def predict(model, rows, a, tok, torch, device, head):
    model.eval()
    out = []
    with torch.no_grad():
        for b, enc in batches(rows, a.eval_batch, tok, torch, device):
            with torch.autocast('cuda', dtype=torch.float16, enabled=a.amp and device.type == 'cuda'):
                logits = model(**enc).logits.float()
            for r, p in zip(b, torch.softmax(logits[:, head], 1).tolist()):
                out.append({'id': r['id'], 'doc_id': r['doc_id'], 'gold': r['gold'],
                            'prediction': LABELS[max(range(3), key=p.__getitem__)],
                            'probs': {c: round(x, 5) for c, x in zip(LABELS, p)},
                            'n_tokens': len(r['enc']['input_ids'])})
    model.train()
    return out


def macro_f1(rows):
    f1s = []
    for c in LABELS:
        tp = sum(r['gold'] == c and r['prediction'] == c for r in rows)
        fp = sum(r['gold'] != c and r['prediction'] == c for r in rows)
        fn = sum(r['gold'] == c and r['prediction'] != c for r in rows)
        f1s.append(2 * tp / (2 * tp + fp + fn) if tp else 0.)
    return sum(f1s) / 3, sum(r['gold'] == r['prediction'] for r in rows) / len(rows)


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf8')


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--train', type=Path, required=True, help='rank-ordered export with claims_train')
    p.add_argument('--dev', type=Path, required=True, help='rank-ordered export with claims_dev')
    p.add_argument('--out', type=Path, required=True, help='checkpoint directory')
    p.add_argument('--pred-dir', type=Path, required=True)
    p.add_argument('--predict', action='append', default=[], metavar='SPLIT=DIR')
    p.add_argument('--model', default=DEFAULT_MODEL)
    p.add_argument('--epochs', type=int, default=3, help='0 scores the NLI model zero-shot')
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--batch', type=int, default=4, help='citations per forward pass')
    p.add_argument('--accum', type=int, default=2, help='forward passes per optimizer step')
    p.add_argument('--eval-batch', type=int, default=16)
    p.add_argument('--max-len', type=int, default=512)
    p.add_argument('--class-weight', default='sqrt', choices=['none', 'sqrt'])
    p.add_argument('--folds', type=int, default=0,
                   help='k-fold CV over training articles; held-out predictions go to <pred-dir>/oof.jsonl')
    p.add_argument('--fold', type=int, default=0)
    p.add_argument('--context-dir', type=Path, default=Path('data_train/context'),
                   help='only for the citation -> article map used to cut folds')
    p.add_argument('--synthetic', type=Path,
                   help='jsonl from generate_misrep.py: extra NOT ACCURATE rows for training only')
    p.add_argument('--synth-frac', type=float, default=1.0,
                   help='share of the usable synthetic rows to add (sampled with --seed)')
    p.add_argument('--amp', action='store_true', help='fp16 mixed precision on CUDA')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='auto')
    p.add_argument('--limit', type=int, help='only the first N citations of each split, for a smoke test')
    a = p.parse_args()

    import torch
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              get_linear_schedule_with_warmup)
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    device = torch.device(a.device if a.device != 'auto' else 'cuda' if torch.cuda.is_available()
                          else 'mps' if torch.backends.mps.is_available() else 'cpu')
    tok = AutoTokenizer.from_pretrained(a.model)
    # Weights in fp32 whatever the checkpoint was saved in: recent transformers load the saved
    # dtype, and this model is published in fp16, which GradScaler cannot unscale. --amp still
    # runs the forward pass in fp16.
    model = AutoModelForSequenceClassification.from_pretrained(a.model).float().to(device)
    by_name = {v.lower(): int(k) for k, v in model.config.id2label.items()}
    head = [by_name[NLI_NAME[c]] for c in LABELS]  # logit index of each of our labels

    train = encode(tok, load_export(a.train, 'train', a.limit), a)
    dev = encode(tok, load_export(a.dev, 'dev', a.limit), a)
    heldout = []
    if a.folds:
        meta = json.loads((a.context_dir / 'train.json').read_text(encoding='utf8'))
        article = {int(k): v['article'] for k, v in meta.items()}
        held = fold_articles(article.values(), a.folds)[a.fold]
        heldout = [r for r in train if article[r['id']] in held]
        train = [r for r in train if article[r['id']] not in held]
        print(f'fold {a.fold}/{a.folds}: training on {len(train)}, holding out {len(heldout)}', flush=True)
    if a.synthetic:
        # Only rows whose source citation is a training row here: in CV that excludes the
        # held-out fold, with --limit it excludes citations not loaded.
        by_id = {r['id']: r for r in train}
        syn = [s for s in read_jsonl(a.synthetic) if s['source_id'] in by_id]
        syn = random.Random(a.seed).sample(syn, round(a.synth_frac * len(syn)))
        train += encode(tok, [{'id': f'syn-{s["source_id"]}', 'doc_id': by_id[s['source_id']]['doc_id'],
                               'premise': by_id[s['source_id']]['premise'],
                               'hypothesis': s['statement'], 'gold': 'NOT_ACCURATE'} for s in syn], a)
        print(f'added {len(syn)} synthetic NOT ACCURATE rows from {a.synthetic}', flush=True)
    counts = Counter(r['gold'] for r in train)
    trunc = sum(len(r['enc']['input_ids']) >= a.max_len for r in train) / len(train)
    print(f'{a.model} on {device}; head {dict(zip(LABELS, head))}; train {len(train)} {dict(counts)}; '
          f'dev {len(dev)}; premise cut at {a.max_len} tokens for {trunc:.0%}', flush=True)

    w = [1.] * 3 if a.class_weight == 'none' else \
        [math.sqrt(len(train) / (3 * max(counts[c], 1))) for c in LABELS]
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(w, device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    steps = max(1, a.epochs * math.ceil(len(train) / (a.batch * a.accum)))
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    scaler = torch.amp.GradScaler('cuda', enabled=a.amp and device.type == 'cuda')
    y = {c: i for i, c in enumerate(LABELS)}

    def evaluate(ep, t0=None, loss=None):
        preds = predict(model, dev, a, tok, torch, device, head)
        f1, acc = macro_f1(preds)
        entry = {'epoch': ep, 'dev_macro_f1': f1, 'dev_accuracy': acc}
        if loss is not None:
            entry.update(train_loss=loss, minutes=(time.time() - t0) / 60)
        log.append(entry)
        print(f'epoch {ep}{" (zero-shot)" if ep == 0 else ""}: '
              + (f'loss {loss:.4f}  ' if loss is not None else '')
              + f'dev macro-F1 {f1:.3f}  accuracy {acc:.3f}', flush=True)
        return f1, preds

    log = []
    f1, preds = evaluate(0)
    best = f1
    a.out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), a.out / 'model.pt')
    write(a.pred_dir / 'dev.jsonl', preds)
    for ep in range(1, a.epochs + 1):
        random.shuffle(train)
        t0, total, n = time.time(), 0., 0
        for i, (b, enc) in enumerate(batches(train, a.batch, tok, torch, device), 1):
            gold = torch.tensor([y[r['gold']] for r in b], device=device)
            with torch.autocast('cuda', dtype=torch.float16, enabled=a.amp and device.type == 'cuda'):
                logits = model(**enc).logits.float()[:, head]
            loss = loss_fn(logits, gold)
            if not torch.isfinite(loss):
                raise SystemExit(f'non-finite loss at epoch {ep}, batch {i}: fp16 overflow is the '
                                 f'usual cause with DeBERTa; rerun without --amp')
            scaler.scale(loss / a.accum).backward()
            total, n = total + loss.item() * len(b), n + len(b)
            if i % a.accum == 0 or i * a.batch >= len(train):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                opt.zero_grad(set_to_none=True)
        f1, preds = evaluate(ep, t0, total / n)
        if f1 > best:
            best = f1
            torch.save(model.state_dict(), a.out / 'model.pt')
            write(a.pred_dir / 'dev.jsonl', preds)

    best_epoch = max((x for x in log), key=lambda x: x['dev_macro_f1'])['epoch']
    print(f'best epoch {best_epoch} (dev macro-F1 {best:.3f})', flush=True)
    a.pred_dir.mkdir(parents=True, exist_ok=True)
    (a.pred_dir / 'config.json').write_text(json.dumps(
        {'model': a.model, 'best_epoch': best_epoch, 'best_dev_macro_f1': best,
         'args': {k: str(v) for k, v in vars(a).items()}, 'log': log}, indent=2))
    model.load_state_dict(torch.load(a.out / 'model.pt', map_location=device))
    if heldout:
        preds = predict(model, heldout, a, tok, torch, device, head)
        write(a.pred_dir / 'oof.jsonl', preds)
        f1, acc = macro_f1(preds)
        print(f'held-out fold {a.fold}: macro-F1 {f1:.3f}  accuracy {acc:.3f}', flush=True)
    for spec in a.predict:
        split, d = spec.split('=', 1)
        preds = predict(model, encode(tok, load_export(d, split, a.limit), a), a, tok, torch, device, head)
        dest = a.pred_dir / f'{Path(d).name}.jsonl'
        write(dest, preds)
        f1, acc = macro_f1(preds)
        print(f'{d} ({split}): macro-F1 {f1:.3f}  accuracy {acc:.3f}  -> {dest}', flush=True)


if __name__ == '__main__':
    main()
