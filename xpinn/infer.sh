#!/usr/bin/env bash
set -euo pipefail
export TASK=infer
export MODE=${MODE:-dynamic}
exec bash "$(dirname "$0")/run_model_common.sh"
