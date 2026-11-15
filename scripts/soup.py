#!/usr/bin/env python3
"""Average the weights of several fine-tuned classifiers into one model (a uniform "model soup").

Averaging the predictions of 3 seeds of the NLI classifier beats a single seed on test (accuracy
0.680 against 0.668 ± 0.015, macro-F1 0.575 against 0.556 ± 0.015, results/nli_classifier_v1.md),
but shipping three DeBERTa-large models triples the download and the run time. Models fine-tuned
from the same pretrained weights at a small learning rate usually stay in one basin of the loss,
and the average of their weights then behaves like their ensemble (Wortsman et al. 2022, "Model
soups"). The soup is one model and costs one model's time.

Weights are averaged in fp32 and saved in fp16, with config and tokenizer from the first model.
All the models must share architecture and label order.

    python scripts/soup.py --models m0 m1 m2 --out soup
"""
import argparse
import shutil
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--models', nargs='+', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()

    import json
    import torch
    from safetensors.torch import load_file, save_file
    labels = [json.loads((m / 'config.json').read_text())['id2label'] for m in a.models]
    if any(x != labels[0] for x in labels):
        raise SystemExit(f'the models disagree on label order: {labels}')
    total = None
    for m in a.models:
        sd = load_file(str(m / 'model.safetensors'))
        if total is None:
            total = {k: v.float().clone() for k, v in sd.items()}
        else:
            if sd.keys() != total.keys():
                raise SystemExit(f'{m} has different weights from {a.models[0]}')
            for k, v in sd.items():
                total[k] += v.float()
    soup = {k: (v / len(a.models)).half().contiguous() for k, v in total.items()}
    a.out.mkdir(parents=True, exist_ok=True)
    for f in a.models[0].iterdir():
        if f.name != 'model.safetensors' and f.is_file():
            shutil.copy(f, a.out / f.name)
    save_file(soup, str(a.out / 'model.safetensors'), metadata={'format': 'pt'})
    print(f'averaged {len(a.models)} models into {a.out}')


if __name__ == '__main__':
    main()
