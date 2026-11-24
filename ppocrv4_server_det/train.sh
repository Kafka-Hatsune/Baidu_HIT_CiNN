#!/usr/bin/env bash
# Train PP-OCRv4_server_det (PaddleOCR) in dynamic OR CINN mode via PaddleX,
# logging per-step loss. CINN uses FLAGS_use_cinn + dy2st (PaddleOCR's
# apply_to_static has no CINN backend arg). `plot_loss.py` overlays the curves.
# Runs on CUDA (A100) and MetaX: the device is auto-detected unless
# PADDLE_DEVICE is set.
#
#   MODE=dynamic ./train.sh
#   MODE=cinn    ./train.sh
#   python plot_loss.py --dynamic results/dynamic-train/stdout.log \
#       --cinn results/cinn-train/stdout.log --out results/train --title PP-OCRv4_server_det
#
# Full output goes to results/<mode>-train (local only); train.log, losses.npz
# and train.json are archived to results/<platform>/<mode>-train for commit.
# Platform is auto-detected (a100, metax-c500, ...); CINN_PLATFORM overrides.
#
# Prereq: PaddleOCR installed (see README.md); on MetaX the device-string bug
# in tools/program.py is patched locally.
set -euo pipefail

MODE=${MODE:-dynamic}
PADDLEX=${PADDLEX:-/data/PaddleX-release-3.7}
HERE=$(cd "$(dirname "$0")" && pwd)
PYTHON_BIN=${PYTHON_BIN:-python}
DATASET=${DATASET_DIR:-${DATASET_ROOT:-/data/paddlex_train/datasets}/ocr_det_dataset_examples}
EPOCHS=${EPOCHS:-10}          # 200 imgs/bs4 -> 50 steps/epoch, 10 -> ~500 steps
BATCH=${BATCH:-4}
export CUDA_VISIBLE_DEVICES=${GPU_ID:-${CUDA_VISIBLE_DEVICES:-0}}

EXTRA=()
if [ "$MODE" = cinn ]; then
  export FLAGS_use_cinn=1 FLAGS_prim_all=true
  EXTRA=(-o Train.dy2st=True)
elif [ "$MODE" != dynamic ]; then
  echo "MODE must be dynamic|cinn" >&2; exit 2
fi
# Reproducible training (seed patches from patch.sh read CINN_TRAIN_SEED);
# DETERMINISTIC=0 keeps upstream behavior. Loss is logged every step so curves
# can be compared pointwise.
if [ "${DETERMINISTIC:-1}" = 1 ]; then
  export CINN_TRAIN_SEED=${SEED:-1024} FLAGS_cudnn_deterministic=true FLAGS_embedding_deterministic=1
  EXTRA+=(-o Train.log_interval=1)
else
  unset CINN_TRAIN_SEED
fi
[[ -f "$PADDLEX/main.py" ]] || { echo "PaddleX not found: PADDLEX=$PADDLEX" >&2; exit 2; }
PADDLEX="$PADDLEX" bash "$HERE/patch.sh" apply
[[ -d "$DATASET" ]] || { echo "Dataset not found: $DATASET (set DATASET_DIR)" >&2; exit 2; }

DEVICE=${PADDLE_DEVICE:-$("$PYTHON_BIN" "$HERE/archive.py" device | tail -n1):0}
PLATFORM=$("$PYTHON_BIN" "$HERE/archive.py" platform | tail -n1)
RESULTS=${RESULTS_DIR:-$HERE/results}
OUT="$RESULTS/${MODE}-train"
[[ ! -e "$OUT" ]] || { echo "Refusing to overwrite $OUT" >&2; exit 3; }
mkdir -p "$OUT"
export PADDLE_PDX_DISABLE_DEV_MODEL_WL=true

cd "$PADDLEX"
set +e
PYTHONUNBUFFERED=1 "$PYTHON_BIN" main.py \
  -c paddlex/configs/modules/text_detection/PP-OCRv4_server_det.yaml \
  -o Global.mode=train \
  -o Global.dataset_dir="$DATASET" \
  -o Global.device="$DEVICE" \
  -o Global.output="$OUT" \
  -o Train.epochs_iters="$EPOCHS" \
  -o Train.batch_size="$BATCH" "${EXTRA[@]}" 2>&1 | tee "$OUT/stdout.log"
rc=${PIPESTATUS[0]}
set -e
"$PYTHON_BIN" "$HERE/archive.py" train --parser ocr --mode "$MODE" --device "$DEVICE" --rc "$rc" \
  --log "$OUT/stdout.log" --dest "$RESULTS/$PLATFORM/${MODE}-train"
echo "done MODE=$MODE rc=$rc -> $OUT/stdout.log"
exit "$rc"
