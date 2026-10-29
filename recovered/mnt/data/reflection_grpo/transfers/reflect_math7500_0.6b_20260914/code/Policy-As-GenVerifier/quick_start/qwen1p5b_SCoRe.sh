#!/usr/bin/env bash

# FOURWAY_CONFIG opts into the reviewed shared experiment. No legacy .env defaults.
if [[ -n "${FOURWAY_CONFIG:-}" ]]; then
    export PYTHONPATH="/root/reflection_grpo_clean${PYTHONPATH:+:$PYTHONPATH}"
    exec /home/vipuser/miniconda3/envs/wrx_env/bin/python -m fourway.cli train --method score --config "$FOURWAY_CONFIG" "$@"
fi
# Aligned with GRPO/REFLECT+; method-specific settings live in aligned_train.sh.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/aligned_train.sh" score "$@"
