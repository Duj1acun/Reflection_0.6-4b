"""Relocate this REFLECT bundle without importing torch or starting training."""
import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--python', default=sys.executable, help='Absolute Python path in the target training environment')
    p.add_argument('--gpus', default='0')
    p.add_argument('--config', default=str(root / 'config.json'))
    p.add_argument('--output', default=str(root / 'workspace'))
    args = p.parse_args()
    ids = args.gpus.split(',')
    if not 1 <= len(ids) <= 2 or len(set(ids)) != len(ids) or any(not x.isdigit() for x in ids):
        p.error('Supply one or two distinct GPU indices')
    python = Path(args.python).absolute()
    if not python.is_file():
        p.error('Target Python executable does not exist')
    out = Path(args.output).resolve()
    if (out / 'runs').exists() and any((out / 'runs').iterdir()):
        p.error('Refusing to reconfigure an output directory that contains training runs')
    c = json.loads((root / 'config.source.json').read_text())
    c.update(python=str(python), reflection_repo=str(root / 'code/reflection_grpo_clean'),
             pag_repo=str(root / 'code/Policy-As-GenVerifier'), output=str(out),
             base_model=str(root / 'models/Qwen3-0.6B'), sft_adapter=str(root / 'models/adapter'),
             train_source=str(root / 'assets/sources/math7500.parquet'),
             math500_source=str(root / 'assets/sources/math500.parquet'),
             amc23_source=str(root / 'assets/sources/amc23.dataset.jsonl'), gpus=args.gpus,
             ray_tmp_dir='/tmp/rf-' + hashlib.sha256(str(out).encode()).hexdigest()[:10],
             lock_dir=str(root / 'runtime/locks'), hf_home=str(root / 'runtime/huggingface'))
    out.mkdir(parents=True, exist_ok=True)
    if not (out / 'data').exists():
        shutil.copytree(root / 'assets/prepared_data', out / 'data')
    manifest_path = out / 'data/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    for splits in manifest['methods'].values():
        for entry in splits.values():
            path = out / 'data' / Path(entry['path']).name
            if hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
                raise ValueError(f'Prepared data checksum mismatch: {path}')
            entry['path'] = str(path)
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    (out / 'initialization').mkdir(exist_ok=True)
    merged = out / 'initialization/reflect_merged'
    target = root / 'models/reflect_merged'
    if merged.is_symlink():
        merged.unlink()
    elif merged.exists():
        raise ValueError('Unexpected existing initialization directory')
    merged.symlink_to(os.path.relpath(target, merged.parent), target_is_directory=True)
    adapter_path = root / 'models/adapter/adapter_config.json'
    adapter = json.loads(adapter_path.read_text())
    adapter['base_model_name_or_path'] = c['base_model']
    adapter_path.write_text(json.dumps(adapter, indent=2) + '\n')
    for key in ['ray_tmp_dir', 'lock_dir', 'hf_home']:
        Path(c[key]).mkdir(parents=True, exist_ok=True)
    destination = Path(args.config).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(c, indent=2) + '\n')
    print('Configured:', destination)
    print('No training started. Next: python3 run_reflect.py preflight')


if __name__ == '__main__':
    main()
