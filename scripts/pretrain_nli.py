#!/usr/bin/env python3
"""Intermediate training: continue the NLI model on biomedical claim verification, then save it.

The data is build_pretrain_data.py's (HealthVer + SciFact, no Citation-Integrity), so the result is
the same starting point for every cross-validation fold and seed: `train_nli.py --model <out>`
fine-tunes it on citations exactly as it fine-tunes the original NLI model. The NLI head is kept,
with entailment / contradiction / neutral trained as ACCURATE / NOT_ACCURATE / NOT ENOUGH INFO, and
the saved model carries our label names, which train_nli.py and the citeaudit package read.

A random 3% is held out and scored before and after, as a check that the model learned the task
and not that it trained.

    python scripts/pretrain_nli.py --data data_train/pretrain/biomed_nli.jsonl \\
        --out checkpoints/nli_biomed --amp
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_nli import DEFAULT_MODEL, LABELS, label_head  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--model', default=DEFAULT_MODEL)
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--accum', type=int, default=1)
    p.add_argument('--max-len', type=int, default=512)
    p.add_argument('--heldout', type=float, default=0.03)
    p.add_argument('--amp', action='store_true')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--limit', type=int)
    a = p.parse_args()

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else
                          'mps' if torch.backends.mps.is_available() else 'cpu')
    use_amp = a.amp and device.type == 'cuda'
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForSequenceClassification.from_pretrained(a.model).float().to(device)
    head = label_head(model)
    y = {c: i for i, c in enumerate(LABELS)}

    rows = [json.loads(x) for x in a.data.read_text(encoding='utf8').splitlines() if x.strip()][:a.limit]
    random.Random(a.seed).shuffle(rows)
    n_held = max(1, int(a.heldout * len(rows)))
    held, train = rows[:n_held], rows[n_held:]
    for r in rows:
        r['enc'] = tok(r['premise'], r['hypothesis'], truncation='longest_first', max_length=a.max_len)

    def batches(rs, size):
        # Length-bucketed: sort within chunks of 50 batches, so padding stays small but order random.
        order = []
        for i in range(0, len(rs), size * 50):
            chunk = sorted(rs[i:i + size * 50], key=lambda r: len(r['enc']['input_ids']))
            order += [chunk[j:j + size] for j in range(0, len(chunk), size)]
        random.shuffle(order)
        for b in order:
            yield b, tok.pad([r['enc'] for r in b], return_tensors='pt').to(device)

    def accuracy():
        model.eval()
        ok = 0
        with torch.no_grad():
            for b, enc in batches(held, 32):
                with torch.autocast('cuda', dtype=torch.float16, enabled=use_amp):
                    pred = model(**enc).logits.float()[:, head].argmax(1).tolist()
                ok += sum(LABELS[q] == r['label'] for q, r in zip(pred, b))
        model.train()
        return ok / len(held)

    print(f'{a.model} on {device}: {len(train)} training pairs, {len(held)} held out; '
          f'held-out accuracy before: {accuracy():.3f}', flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    steps = max(1, a.epochs * math.ceil(len(train) / (a.batch * a.accum)))
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    loss_fn = torch.nn.CrossEntropyLoss()
    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        random.shuffle(train)
        total, n = 0., 0
        for i, (b, enc) in enumerate(batches(train, a.batch), 1):
            with torch.autocast('cuda', dtype=torch.float16, enabled=use_amp):
                logits = model(**enc).logits.float()[:, head]
            loss = loss_fn(logits, torch.tensor([y[r['label']] for r in b], device=device))
            if not torch.isfinite(loss):
                raise SystemExit(f'non-finite loss at batch {i}; rerun without --amp')
            scaler.scale(loss / a.accum).backward()
            total, n = total + loss.item() * len(b), n + len(b)
            if i % a.accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                opt.zero_grad(set_to_none=True)
            if i % 200 == 0:
                print(f'  epoch {ep} batch {i}: loss {total / n:.4f}, '
                      f'{(time.time() - t0) / 60:.0f} min', flush=True)
        print(f'epoch {ep}: loss {total / n:.4f}; held-out accuracy {accuracy():.3f}', flush=True)

    model.config.id2label = {h: c for c, h in zip(LABELS, head)}
    model.config.label2id = {c: h for c, h in zip(LABELS, head)}
    a.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(a.out, safe_serialization=True)
    tok.save_pretrained(a.out)
    print(f'saved {a.out}', flush=True)


if __name__ == '__main__':
    main()
