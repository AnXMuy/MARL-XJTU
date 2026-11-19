#!/usr/bin/env bash
set -euo pipefail
TRAFFIC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="$TRAFFIC_ROOT/../.venv/bin:$PATH"
exec "$@"
