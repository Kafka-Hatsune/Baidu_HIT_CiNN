#!/usr/bin/env bash
# Train PP-OCRv4_server_det (PaddleOCR) on MetaX in dynamic OR CINN mode via
# PaddleX, logging per-step loss. CINN uses FLAGS_use_cinn + dy2st (PaddleOCR's
# apply_to_static has no CINN backend arg). `plot_loss.py` overlays the curves.
#
#   MODE=dynamic ./train.sh
#   MODE=cinn    ./train.sh
#   python plot_loss.py --dynamic results/dynamic-train/stdout.log \
#       --cinn results/cinn-train/stdout.log --out results/train --title PP-OCRv4_server_det
#
# Prereq: source /data/env.sh ; PaddleOCR installed (see README.md). Note: this
# repo's PaddleOCR had a metax device-string bug (tools/program.py used
# "metax:" instead of "metax_gpu:"), patched locally.
set -euo pipefail

MODE=${MODE:-dynamic}
PADDLEX=${PADDLEX:-/data/PaddleX-release-3.7}
HERE=$(cd "$(dirname "$0")" && pwd)
DATASET=${DATASET_DIR:-/data/paddlex_train/datasets/ocr_det_dataset_examples}
DEVICE=${PADDLE_DEVICE:-metax_gpu:0}
EPOCHS=${EPOCHS:-10}          # 200 imgs/bs4 -> 50 steps/epoch, 10 -> ~500 steps
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
  -c paddlex/configs/modules/text_detection/PP-OCRv4_server_det.yaml \
  -o Global.mode=train \
  -o Global.dataset_dir="$DATASET" \
  -o Global.device="$DEVICE" \
  -o Global.output="$OUT" \
  -o Train.epochs_iters="$EPOCHS" \
  -o Train.batch_size="$BATCH" "${EXTRA[@]}" 2>&1 | tee "$OUT/stdout.log"
echo "done MODE=$MODE -> $OUT/stdout.log"
