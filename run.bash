#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
"${PYTHON:-python3}" main.py --dataname Tweet --lambda 0.6 "$@"

