#!/usr/bin/env bash
# Aligned with GRPO/REFLECT+; method-specific settings live in aligned_train.sh.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/aligned_train.sh" pag "$@"
