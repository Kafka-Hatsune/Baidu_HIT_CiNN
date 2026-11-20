#!/usr/bin/env bash
set -euo pipefail
export MODE=cinn
exec bash "$(dirname "$0")/run_model_common.sh"
