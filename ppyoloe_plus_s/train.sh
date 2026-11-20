#!/usr/bin/env bash
# Train PP-YOLOE_plus-S (PaddleDetection) on MetaX in dynamic OR CINN mode via
# PaddleX, logging per-step loss. CINN is enabled with FLAGS_use_cinn + dy2st
# (PaddleDetection's apply_to_static has no CINN backend arg, so the flag path
# is used). `plot_loss.py` overlays the two loss curves.
#
#   MODE=dynamic ./train.sh
#   MODE=cinn    ./train.sh
#   python plot_loss.py --dynamic results/dynamic-train/stdout.log \
#       --cinn results/cinn-train/stdout.log --out results/train --title PP-YOLOE_plus-S
#
# Prereq: source /data/env.sh ; PaddleDetection installed (see README.md).
set -euo pipefail

MODE=${MODE:-dynamic}
PADDLEX=${PADDLEX:-/data/PaddleX-release-3.7}
HERE=$(cd "$(dirname "$0")" && pwd)
DATASET=${DATASET_DIR:-/data/paddlex_train/datasets/det_coco_examples}
DEVICE=${PADDLE_DEVICE:-metax_gpu:0}
EPOCHS=${EPOCHS:-3}            # det_coco_examples ~175 iters/epoch @bs4 -> ~525 steps
BATCH=${BATCH:-4}
OUT="$HERE/results/${MODE}-train"

rm -rf "$OUT"; mkdir -p "$OUT"
export PADDLE_PDX_DISABLE_DEV_MODEL_WL=true

EXTRA=()
if [ "$MODE" = cinn ]; then
  export FLAGS_use_cinn=1 FLAGS_prim_all=true
  EXTRA=(-o Train.dy2st=True)
elif [ "$MODE" != dynamic ]; then
  echo "MODE must be dynamic|cinn" >&2; exit 2
fi

cd "$PADDLEX"
PYTHONUNBUFFERED=1 python main.py \
  -c paddlex/configs/modules/object_detection/PP-YOLOE_plus-S.yaml \
  -o Global.mode=train \
  -o Global.dataset_dir="$DATASET" \
  -o Global.device="$DEVICE" \
  -o Global.output="$OUT" \
  -o Train.epochs_iters="$EPOCHS" \
  -o Train.batch_size="$BATCH" "${EXTRA[@]}" 2>&1 | tee "$OUT/stdout.log"
echo "done MODE=$MODE -> $OUT/stdout.log"
