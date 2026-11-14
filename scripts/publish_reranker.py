#!/usr/bin/env python3
"""Upload the trained reranker (checkpoints/ce_repro) to the Hugging Face Hub as <user>/citeaudit-reranker.

Needs a write token: run `huggingface-cli login` (or `hf auth login`) first, or set HF_TOKEN.
The classifier is published from Colab by notebooks/release_model_colab.ipynb.

    python scripts/publish_reranker.py
"""
import argparse
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', type=Path, default=ROOT / 'checkpoints/ce_repro')
    p.add_argument('--name', default='citeaudit-reranker')
    p.add_argument('--classifier-name', default='citeaudit-nli')
    a = p.parse_args()

    from huggingface_hub import HfApi
    api = HfApi()
    user = api.whoami()['name']
    repo = f'{user}/{a.name}'
    with tempfile.TemporaryDirectory() as tmp:
        for f in ('config.json', 'model.safetensors', 'tokenizer.json', 'tokenizer_config.json'):
            shutil.copy(a.checkpoint / f, tmp)
        card = (ROOT / 'release/reranker_model_card.md').read_text(encoding='utf8')
        (Path(tmp) / 'README.md').write_text(card.replace('CLASSIFIER_ID', f'{user}/{a.classifier_name}'),
                                             encoding='utf8')
        api.create_repo(repo, exist_ok=True)
        api.upload_folder(folder_path=tmp, repo_id=repo, commit_message='CiteAudit evidence reranker')
    print(f'published https://huggingface.co/{repo}')


if __name__ == '__main__':
    main()
