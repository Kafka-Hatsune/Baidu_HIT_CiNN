#!/usr/bin/env bash
set -euo pipefail
export MODE=dynamic
exec bash "$(dirname "$0")/run_model_common.sh"
