#!/usr/bin/env python3
"""Fine-tune an instruction-tuned LLM with (Q)LoRA to answer the citation-label prompt.

Why this exists. Zero-shot, Qwen2.5-7B ranks citations worse than the task-trained models
(results/llm_zero_shot_v1.md), and perfect evidence does not lift the trained classifiers either
(results/classifier_v3.md): what limits accuracy now is how well the model reads the evidence.
Fine-tuning gives that reading 7B parameters of language understanding instead of BiomedBERT's
110M, trained on this task's 2141 citations rather than prompted.

The prompt and answer read-out are `llm_classify.py`'s: the model sees the same instructions,
citing sentence and top-k evidence, and the prediction is its next-token distribution over the
letters A/B/C, renormalised over the three. Training minimises cross-entropy on exactly those
three logits (class-weighted), so it never learns to generate anything else and inference stays
one forward pass. Only the last position's logits are computed.

Memory and time. `--load-4bit` (bitsandbytes NF4, CUDA only) plus LoRA adapters on every linear
layer and gradient checkpointing fit a 7B model and a ~1400-token prompt in a 16 GB T4.

Interruptions. Every --save-every optimizer steps the adapter weights, optimizer, scheduler and
position in the (seeded, so reproducible) data order are written to --state-dir; rerunning the
same command resumes from there. Colab disconnects lose at most that many steps.

    python scripts/finetune_llm.py --train data_train/target/ours_20_train_cv \\
        --dev data_train/target/ours_20 --predict test=data_train/target/ours_20_test \\
        --model Qwen/Qwen2.5-7B-Instruct --load-4bit --state-dir /content/drive/.../state \\
        --pred-dir results/classification/llm_ft
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm_classify import LETTERS, PROMPTS, build_prompt, letter_ids, read_jsonl  # noqa: E402

LABEL_TO_LETTER = {v: k for k, v in LETTERS.items()}


def load_export(d, split, limit=None):
    claims = read_jsonl(Path(d) / f'claims_{split}.jsonl')[:limit]
    corpus = {x['doc_id']: x['abstract'] for x in read_jsonl(Path(d) / 'corpus.jsonl')}
    return [{'id': c['id'], 'doc_id': c['cited_doc_ids'][0], 'claim': c['claim'],
             'sents': corpus[c['cited_doc_ids'][0]],
             'gold': next(iter(c['evidence'].values()))[0]['label'] if c.get('evidence')
             else 'NOT ENOUGH INFO'} for c in claims]


def letter_logits(model, tok, enc, ids, torch):
    """The three answer logits at the prompt's last position (log-sum-exp over token variants)."""
    try:
        out = model(**enc, logits_to_keep=1)
    except TypeError:  # older transformers
        out = model(**enc, num_logits_to_keep=1)
    last = out.logits[0, -1].float()
    return torch.stack([torch.logsumexp(last[ids[L]], 0) for L in LETTERS])


def predict(model, tok, rows, a, ids, torch, device):
    model.eval()
    out = []
    with torch.no_grad():
        for r in rows:
            enc = tok(build_prompt(tok, a.prompt, r['claim'], r['sents']), return_tensors='pt',
                      add_special_tokens=False).to(device)
            p = torch.softmax(letter_logits(model, tok, enc, ids, torch), 0).tolist()
            out.append({'id': r['id'], 'doc_id': r['doc_id'], 'gold': r['gold'],
                        'prediction': LETTERS['ABC'[max(range(3), key=p.__getitem__)]],
                        'probs': {LETTERS[L]: round(p[i], 5) for i, L in enumerate(LETTERS)},
                        'n_tokens': enc['input_ids'].shape[1]})
    model.train()
    return out


def macro_f1(rows):
    f1s = []
    for c in LETTERS.values():
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
    p.add_argument('--train', type=Path, required=True, help='export directory with claims_train')
    p.add_argument('--dev', type=Path, required=True, help='export directory with claims_dev')
    p.add_argument('--predict', action='append', default=[], metavar='SPLIT=DIR',
                   help='after training, predict this export; repeatable')
    p.add_argument('--pred-dir', type=Path, required=True)
    p.add_argument('--state-dir', type=Path, required=True,
                   help='resumable training state; rerunning the same command continues from it')
    p.add_argument('--model', default='Qwen/Qwen2.5-7B-Instruct')
    p.add_argument('--prompt', default='v1', choices=sorted(PROMPTS))
    p.add_argument('--load-4bit', action='store_true', help='bitsandbytes NF4; needs CUDA')
    p.add_argument('--lora-r', type=int, default=16)
    p.add_argument('--lora-alpha', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--accum', type=int, default=8, help='citations per optimizer step')
    p.add_argument('--class-weight', default='sqrt', choices=['none', 'sqrt'])
    p.add_argument('--save-every', type=int, default=25, help='optimizer steps between saves')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--limit', type=int, help='only the first N citations of each split, for a smoke test')
    a = p.parse_args()

    import torch
    from peft import (LoraConfig, get_peft_model, get_peft_model_state_dict,
                      prepare_model_for_kbit_training, set_peft_model_state_dict)
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

    torch.manual_seed(a.seed)
    cuda = torch.cuda.is_available()
    tok = AutoTokenizer.from_pretrained(a.model)
    train = load_export(a.train, 'train', a.limit)
    dev = load_export(a.dev, 'dev', a.limit)

    if a.load_4bit:
        from transformers import BitsAndBytesConfig
        model = AutoModelForCausalLM.from_pretrained(a.model, device_map='auto',
            quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                                                   bnb_4bit_compute_dtype=torch.float16))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            a.model, torch_dtype=torch.float16 if cuda else torch.float32)
        model.to('cuda' if cuda else 'mps' if torch.backends.mps.is_available() else 'cpu')
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=0.05, task_type='CAUSAL_LM',
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']))
    device = next(model.parameters()).device
    n_train = sum(x.numel() for x in model.parameters() if x.requires_grad)
    n_all = sum(x.numel() for x in model.parameters())
    ids = letter_ids(tok)
    counts = {c: sum(r['gold'] == c for r in train) for c in LETTERS.values()}
    print(f'{a.model} on {device}: training {n_train / 1e6:.1f}M of {n_all / 1e6:.0f}M parameters; '
          f'train {len(train)} {counts}; dev {len(dev)}', flush=True)

    w = [1.] * 3
    if a.class_weight == 'sqrt':
        w = [math.sqrt(len(train) / (3 * max(counts[LETTERS[L]], 1))) for L in LETTERS]
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(w, device=device))
    params = [x for x in model.parameters() if x.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    steps_per_epoch = math.ceil(len(train) / a.accum)
    total = a.epochs * steps_per_epoch
    sched = get_linear_schedule_with_warmup(opt, int(0.05 * total), total)
    scaler = torch.amp.GradScaler('cuda', enabled=cuda)

    a.state_dir.mkdir(parents=True, exist_ok=True)
    state_path = a.state_dir / 'state.pt'
    step, log, best = 0, [], (-1., None)
    if state_path.exists():
        st = torch.load(state_path, map_location='cpu', weights_only=False)
        set_peft_model_state_dict(model, st['lora'])
        opt.load_state_dict(st['opt'])
        sched.load_state_dict(st['sched'])
        scaler.load_state_dict(st['scaler'])
        step, log, best = st['step'], st['log'], st['best']
        print(f'resumed at optimizer step {step} of {total}', flush=True)

    def save():
        torch.save({'lora': get_peft_model_state_dict(model), 'opt': opt.state_dict(),
                    'sched': sched.state_dict(), 'scaler': scaler.state_dict(),
                    'step': step, 'log': log, 'best': best}, str(state_path) + '.tmp')
        Path(str(state_path) + '.tmp').replace(state_path)  # never leave a half-written state

    model.train()
    t0 = time.time()
    done_at_start = step
    while step < total:
        ep, k = divmod(step, steps_per_epoch)
        order = list(range(len(train)))
        random.Random(a.seed * 1000 + ep).shuffle(order)  # same order on every resume
        batch = order[k * a.accum:(k + 1) * a.accum]
        loss_sum = 0.
        for i in batch:
            r = train[i]
            enc = tok(build_prompt(tok, a.prompt, r['claim'], r['sents']), return_tensors='pt',
                      add_special_tokens=False).to(device)
            with torch.autocast('cuda', dtype=torch.float16, enabled=cuda):
                logits = letter_logits(model, tok, enc, ids, torch)
            gold = torch.tensor(['ABC'.index(LABEL_TO_LETTER[r['gold']])], device=device)
            loss = loss_fn(logits.unsqueeze(0), gold) / len(batch)
            scaler.scale(loss).backward()
            loss_sum += loss.item()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        log.append({'step': step, 'loss': loss_sum})
        if step % 10 == 0:
            rate = (step - done_at_start) / (time.time() - t0)
            recent = [x['loss'] for x in log[-50:] if 'loss' in x]
            print(f'  step {step}/{total}  loss (last 50) {sum(recent) / len(recent):.4f}  '
                  f'eta {(total - step) / rate / 60:.0f} min', flush=True)
        if step % steps_per_epoch == 0:
            preds = predict(model, tok, dev, a, ids, torch, device)
            f1, acc = macro_f1(preds)
            log.append({'epoch': step // steps_per_epoch, 'dev_macro_f1': f1, 'dev_accuracy': acc})
            print(f'epoch {step // steps_per_epoch}: dev macro-F1 {f1:.3f}  accuracy {acc:.3f}',
                  flush=True)
            if f1 > best[0]:
                best = (f1, step // steps_per_epoch)
                torch.save(get_peft_model_state_dict(model), a.state_dir / 'best_lora.pt')
                write(a.pred_dir / 'dev.jsonl', preds)
            save()
        elif step % a.save_every == 0:
            save()

    print(f'best epoch {best[1]} (dev macro-F1 {best[0]:.3f})', flush=True)
    set_peft_model_state_dict(model, torch.load(a.state_dir / 'best_lora.pt', map_location='cpu'))
    a.pred_dir.mkdir(parents=True, exist_ok=True)
    (a.pred_dir / 'config.json').write_text(json.dumps(
        {'model': a.model, 'best_epoch': best[1], 'best_dev_macro_f1': best[0],
         'args': {k: str(v) for k, v in vars(a).items()},
         'log': [x for x in log if 'epoch' in x or x['step'] % 10 == 0]}, indent=2))
    for spec in a.predict:
        split, d = spec.split('=', 1)
        dest = a.pred_dir / f'{Path(d).name}.jsonl'
        if dest.exists():
            print(f'{dest} exists, skipping')
            continue
        preds = predict(model, tok, load_export(d, split, a.limit), a, ids, torch, device)
        write(dest, preds)
        f1, acc = macro_f1(preds)
        print(f'{d} ({split}): macro-F1 {f1:.3f}  accuracy {acc:.3f}  -> {dest}', flush=True)


if __name__ == '__main__':
    main()
