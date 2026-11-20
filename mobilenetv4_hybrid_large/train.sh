#!/usr/bin/env bash
# Train one PaddleClas model (StarNet-S4) on MetaX in dynamic OR CINN mode and
# log per-step loss. Both modes share train_config.yaml (resolved PaddleClas
# config produced by PaddleX) and the same seed, so the two loss curves are
# directly comparable. `plot_loss.py` overlays them.
#
#   MODE=dynamic ./train.sh    # plain dynamic graph
#   MODE=cinn    ./train.sh     # Global.use_cinn=True -> to_static(backend=CINN)
#   python plot_loss.py --dynamic results/dynamic-train/train.log \
#       --cinn results/cinn-train/train.log --out results/train --title StarNet-S4
#
# Prereq: source /data/env.sh ; PaddleClas installed (see README.md).
set -euo pipefail

MODE=${MODE:-dynamic}
PADDLEX=${PADDLEX:-/data/PaddleX-release-3.7}
CLAS="$PADDLEX/paddlex/repo_manager/repos/PaddleClas"
HERE=$(cd "$(dirname "$0")" && pwd)
CFG=${TRAIN_CONFIG:-$HERE/train_config.yaml}
EPOCHS=${EPOCHS:-32}          # flowers 1020/bs64 -> 16 steps/epoch, 32 -> ~512 steps
SEED=${SEED:-42}
DEVICE=${PADDLE_DEVICE:-metax_gpu}
OUT="$HERE/results/${MODE}-train"

case "$MODE" in
  dynamic) USE_CINN=False ;;
  cinn)    USE_CINN=True ;;
  *) echo "MODE must be dynamic|cinn" >&2; exit 2 ;;
esac

rm -rf "$OUT"; mkdir -p "$OUT"
export PADDLE_PDX_DISABLE_DEV_MODEL_WL=true
cd "$CLAS"
PYTHONUNBUFFERED=1 python tools/train.py -c "$CFG" \
  -o Global.use_cinn=$USE_CINN \
  -o Global.epochs=$EPOCHS \
  -o Global.seed=$SEED \
  -o Global.eval_during_train=False \
  -o Global.save_interval=100000 \
  -o Global.print_batch_step=1 \
  -o Global.output_dir="$OUT" \
  -o Global.device=$DEVICE 2>&1 | tee "$OUT/train.log"
echo "done MODE=$MODE -> $OUT/train.log"
