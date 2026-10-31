#!/usr/bin/env python3
"""Fine-tune the released MultiVerS checkpoint on new evidence, starting from its own weights.

Why this exists. The BiomedBERT classifier (`train_classifier.py`, results/classifier_v2.md)
trained on our reranker's evidence beats the paper on accuracy but not on macro-F1, and the whole
gap is NOT ENOUGH INFO (F1 0.27 against MultiVerS's 0.455). MultiVerS was pretrained on fact
verification, where "not enough info" is a core class; BiomedBERT has only 143 NEI training
citations to learn it from. Continuing MultiVerS's own training on the evidence our retriever
produces keeps what it knows about NEI and adapts it to the better evidence.

The model, tokenisation and global attention are `predict_multivers.py`'s, imported rather than
copied, so a fine-tuned model reads its input exactly as the released one does. The label head
starts from the checkpoint too: nothing is re-initialised.

Memory. Longformer-large is 435M parameters; with AdamW that is about 7 GB before activations.
`--grad-checkpointing` and `--amp` (fp16 autocast, CUDA only) bring a top-20 citation, about
1000 tokens, within a 16 GB T4 at one citation per forward; `--accum` sets the effective batch.

The epoch kept is the one with the best dev macro-F1, held in CPU memory, and test is predicted
with it at the end. Predictions use `predict_multivers.py`'s format, so `score_multivers.py`
scores and pairs them with the released checkpoint's.

    python scripts/finetune_multivers.py --checkpoint checkpoints/drive/bestModel.ckpt \\
        --train data_train/target/ours_20_train_cv --dev data_train/target/ours_20 \\
        --predict test=data_train/target/ours_20_test --pred-dir results/classification/mv_ft \\
        --grad-checkpointing --amp
"""
import argparse
import contextlib
import json
import math
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from predict_multivers import (LABELS, MAX_TOKENS, build_model, encode, get_tokenizer,  # noqa: E402
                               pick_device, read_jsonl)

LABEL_ID = {v: k for k, v in LABELS.items()}
NEI = 'NOT ENOUGH INFO'


def load_export(d, split, tok, limit=None):
    """Encoded instances of an export, one per citation; over-long ones are dropped and reported."""
    claims = read_jsonl(Path(d) / f'claims_{split}.jsonl')[:limit]
    corpus = {x['doc_id']: x for x in read_jsonl(Path(d) / 'corpus.jsonl')}
    rows, skipped = [], 0
    for c in claims:
        doc = corpus[c['cited_doc_ids'][0]]
        gold = next(iter(c['evidence'].values()))[0]['label'] if c.get('evidence') else NEI
        enc = encode(tok, c['claim'], doc['abstract'], doc.get('title'))
        if len(enc['input_ids']) > MAX_TOKENS:
            skipped += 1
            continue
        rows.append({'id': c['id'], 'doc_id': doc['doc_id'], 'gold': gold, 'enc': enc})
    if skipped:
        print(f'{d} {split}: skipped {skipped} instances over {MAX_TOKENS} tokens', flush=True)
    return rows


def amp_ctx(a, device):
    if a.amp and device.type == 'cuda':
        return torch.autocast('cuda', dtype=torch.float16)
    return contextlib.nullcontext()


def logits_of(encoder, head, enc, device, a):
    batch = {k: v.unsqueeze(0).to(device) for k, v in enc.items()}
    with amp_ctx(a, device):
        pooled = encoder(**batch).pooler_output
        return head(pooled).float()


def predict(encoder, head, rows, device, a):
    encoder.eval()
    head.eval()
    out = []
    with torch.no_grad():
        for r in rows:
            probs = torch.softmax(logits_of(encoder, head, r['enc'], device, a), dim=1)[0].tolist()
            out.append({'id': r['id'], 'doc_id': r['doc_id'], 'gold': r['gold'],
                        'prediction': LABELS[max(range(3), key=probs.__getitem__)],
                        'probs': {LABELS[i]: round(probs[i], 5) for i in range(3)},
                        'n_tokens': len(r['enc']['input_ids'])})
    encoder.train()
    head.train()
    return out


def macro_f1(rows):
    f1s = []
    for c in LABELS.values():
        tp = sum(r['gold'] == c and r['prediction'] == c for r in rows)
        fp = sum(r['gold'] != c and r['prediction'] == c for r in rows)
        fn = sum(r['gold'] == c and r['prediction'] != c for r in rows)
        f1s.append(2 * tp / (2 * tp + fp + fn) if tp else 0.)
    return sum(f1s) / 3, sum(r['gold'] == r['prediction'] for r in rows) / len(rows)


def snapshot(encoder, head):
    """A CPU copy of both modules' weights, so the best epoch survives later ones."""
    return {'encoder': {k: v.detach().to('cpu', copy=True) for k, v in encoder.state_dict().items()},
            'head': {k: v.detach().to('cpu', copy=True) for k, v in head.state_dict().items()}}


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf8')


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', type=Path, default=Path('checkpoints/drive/bestModel.ckpt'))
    p.add_argument('--train', type=Path, required=True, help='export directory with claims_train')
    p.add_argument('--dev', type=Path, required=True, help='export directory with claims_dev')
    p.add_argument('--predict', action='append', default=[], metavar='SPLIT=DIR',
                   help='after training, predict this export with the best epoch; repeatable')
    p.add_argument('--pred-dir', type=Path, required=True)
    p.add_argument('--save', type=Path, help='also save the best weights here (about 1.7 GB)')
    p.add_argument('--epochs', type=int, default=2)
    p.add_argument('--lr', type=float, default=1e-5,
                   help='lower than training from scratch: this continues a converged model')
    p.add_argument('--accum', type=int, default=8, help='citations per optimizer step')
    p.add_argument('--class-weight', default='none', choices=['none', 'sqrt'])
    p.add_argument('--no-initial-eval', action='store_true',
                   help='skip scoring the unchanged checkpoint on dev before training')
    p.add_argument('--grad-checkpointing', action='store_true')
    p.add_argument('--amp', action='store_true', help='fp16 autocast on CUDA; ignored elsewhere')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='auto')
    p.add_argument('--limit', type=int, help='only the first N citations of each split, for a smoke test')
    a = p.parse_args()

    random.seed(a.seed)
    torch.manual_seed(a.seed)
    device = pick_device(a.device)
    tok = get_tokenizer()
    train = load_export(a.train, 'train', tok, a.limit)
    dev = load_export(a.dev, 'dev', tok, a.limit)
    counts = {c: sum(r['gold'] == c for r in train) for c in LABELS.values()}
    print(f'train {len(train)} {counts}; dev {len(dev)}; {device}', flush=True)

    encoder, head, meta = build_model(a.checkpoint, device)
    print(f'from {a.checkpoint} (epoch {meta["epoch"]}, step {meta["global_step"]})', flush=True)
    if a.grad_checkpointing:
        encoder.gradient_checkpointing_enable()
    encoder.train()
    head.train()

    w = [1.] * 3
    if a.class_weight == 'sqrt':
        w = [math.sqrt(len(train) / (3 * max(counts[LABELS[i]], 1))) for i in range(3)]
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(w, device=device))
    params = list(encoder.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.01)
    steps = a.epochs * math.ceil(len(train) / a.accum)
    from transformers import get_linear_schedule_with_warmup
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    scaler = torch.amp.GradScaler('cuda', enabled=a.amp and device.type == 'cuda')

    log, best, best_state, best_epoch = [], -1., None, None
    if not a.no_initial_eval:
        # The released checkpoint on this dev set: the bar fine-tuning has to clear.
        # It is also a candidate: if no epoch beats it on dev, the released weights are kept.
        preds = predict(encoder, head, dev, device, a)
        f1, acc = macro_f1(preds)
        log.append({'epoch': 0, 'dev_macro_f1': f1, 'dev_accuracy': acc})
        print(f'epoch 0 (released checkpoint): dev macro-F1 {f1:.3f}  accuracy {acc:.3f}',
              flush=True)
        best, best_state, best_epoch = f1, snapshot(encoder, head), 0
        write(a.pred_dir / 'dev.jsonl', preds)

    for ep in range(1, a.epochs + 1):
        random.shuffle(train)
        t0, total = time.time(), 0.
        for i, r in enumerate(train, 1):
            gold = torch.tensor([LABEL_ID[r['gold']]], device=device)
            loss = loss_fn(logits_of(encoder, head, r['enc'], device, a), gold) / a.accum
            scaler.scale(loss).backward()
            total += loss.item() * a.accum
            if i % a.accum == 0 or i == len(train):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                opt.zero_grad(set_to_none=True)
            if i % 200 == 0:
                rate = i / (time.time() - t0)
                print(f'  epoch {ep}: {i}/{len(train)}  loss {total / i:.4f}  '
                      f'eta {(len(train) - i) / rate / 60:.0f} min', flush=True)
        preds = predict(encoder, head, dev, device, a)
        f1, acc = macro_f1(preds)
        log.append({'epoch': ep, 'train_loss': total / len(train), 'dev_macro_f1': f1,
                    'dev_accuracy': acc, 'minutes': (time.time() - t0) / 60})
        print(f'epoch {ep}: loss {total / len(train):.4f}  dev macro-F1 {f1:.3f}  '
              f'accuracy {acc:.3f}  ({(time.time() - t0) / 60:.1f} min)', flush=True)
        if f1 > best:
            best, best_state, best_epoch = f1, snapshot(encoder, head), ep
            write(a.pred_dir / 'dev.jsonl', preds)

    a.pred_dir.mkdir(parents=True, exist_ok=True)
    (a.pred_dir / 'config.json').write_text(json.dumps(
        {'checkpoint': str(a.checkpoint), 'best_epoch': best_epoch, 'best_dev_macro_f1': best,
         'args': {k: str(v) for k, v in vars(a).items()}, 'log': log}, indent=2))

    print(f'best epoch {best_epoch} (dev macro-F1 {best:.3f})', flush=True)
    encoder.load_state_dict(best_state['encoder'])
    head.load_state_dict(best_state['head'])
    if a.save:
        a.save.parent.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, a.save)
    for spec in a.predict:
        split, d = spec.split('=', 1)
        preds = predict(encoder, head, load_export(d, split, tok, a.limit), device, a)
        dest = a.pred_dir / f'{Path(d).name}.jsonl'
        write(dest, preds)
        f1, acc = macro_f1(preds)
        print(f'{d} ({split}): macro-F1 {f1:.3f}  accuracy {acc:.3f}  -> {dest}', flush=True)


if __name__ == '__main__':
    main()
