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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/train_reflect_plus.sh" --reflection-branch-grpo "$@"
