#!/usr/bin/env bash
set -euo pipefail
export TASK=train
export MODE=${MODE:-dynamic}
export WARMUP=${WARMUP:-5}
export ITERS=${ITERS:-500}
exec bash "$(dirname "$0")/run_model_common.sh"
