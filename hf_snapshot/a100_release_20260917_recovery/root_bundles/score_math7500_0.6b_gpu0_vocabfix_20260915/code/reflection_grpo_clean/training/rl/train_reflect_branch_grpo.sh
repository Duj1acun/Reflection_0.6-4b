#!/usr/bin/env bash
# Four-way reflection GRPO. Shared defaults live in train_reflect_plus.sh.
set -euo pipefail

case "${1:-}" in
    -h|--help)
        echo "Usage: bash $0 [--dry-run]"
        echo "Uses train_reflect_plus.sh hyperparameters, model, data and reward formula."
        echo "Defaults: 8 initial samples/question; K=4 at each reflection; MAX_ITERS=3."
        echo "Environment: REFLECTION_BRANCHES, BRANCH_EXP_NAME, plus all REFLECT+ overrides."
        echo "--dry-run prints the command without GPU checks, merging, or training."
        exit 0 ;;
    ''|--dry-run) ;;
    *) echo "Unknown argument: $1 (use --help)" >&2; exit 2 ;;
esac
if (( $# > 1 )); then echo "Only --dry-run or --help is supported." >&2; exit 2; fi

if [[ -n "${FOURWAY_CONFIG:-}" ]]; then
    if [[ ! -f "$FOURWAY_CONFIG" ]]; then echo "FOURWAY_CONFIG does not exist" >&2; exit 1; fi
    export PYTHONPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)${PYTHONPATH:+:$PYTHONPATH}"
    exec python3 - "$FOURWAY_CONFIG" "${1:-}" <<'PYENTRY'
import json, os, subprocess, sys
from pathlib import Path
path = Path(sys.argv[1]).resolve()
if not path.is_file():
    raise SystemExit("FOURWAY_CONFIG does not exist")
c = json.loads(path.read_text())
if not c.get("reflection_branching", {}).get("enable", False):
    raise SystemExit("FOURWAY_CONFIG must explicitly enable reflection_branching")
command = 'dry-run' if sys.argv[2] == '--dry-run' else 'train'
raise SystemExit(subprocess.call([c['python'], '-u', '-m', 'fourway.cli', command,
                                '--method', 'reflect', '--config', str(path)], cwd=c['reflection_repo']))
PYENTRY
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/train_reflect_plus.sh" --reflection-branch-grpo "$@"
