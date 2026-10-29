"""Use the configured target Python and the bundled REFLECT code."""
import argparse
import json
import os
import subprocess
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['preflight', 'dry-run', 'train', 'evaluate'])
    p.add_argument('--config', default=str(root / 'config.json'))
    p.add_argument('--gpu', action='store_true')
    p.add_argument('--checkpoint')
    args = p.parse_args()
    path = Path(args.config).resolve()
    if not path.exists():
        p.error('Run configure.py with the target training Python first')
    c = json.loads(path.read_text())
    env = os.environ.copy()
    env['PYTHONPATH'] = c['reflection_repo']
    cmd = [c['python'], '-u', '-m', 'fourway.cli', args.command,
           '--config', str(path), '--method', 'reflect']
    if args.gpu:
        cmd.append('--gpu')
    if args.checkpoint:
        cmd += ['--checkpoint', args.checkpoint]
    raise SystemExit(subprocess.call(cmd, cwd=c['reflection_repo'], env=env))


if __name__ == '__main__':
    main()
