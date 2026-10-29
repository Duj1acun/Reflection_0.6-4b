#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${HERE}/run.sh" "$@" --method reflect --config "${HERE}/config.reflect_math7500.json"
