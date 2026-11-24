#!/usr/bin/env bash
# Train one PaddleClas model (StarNet-S4) in dynamic OR CINN mode and log
# per-step loss. Both modes share train_config.yaml (resolved PaddleClas config
# produced by PaddleX) and the same seed, so the two loss curves are directly
# comparable. `plot_loss.py` overlays them. Runs on CUDA (A100) and MetaX: the
# device is auto-detected unless PADDLE_DEVICE is set.
#
#   MODE=dynamic ./train.sh    # plain dynamic graph
#   MODE=cinn    ./train.sh     # Global.use_cinn=True -> to_static(backend=CINN)
#   python plot_loss.py --dynamic results/dynamic-train/train.log \
#       --cinn results/cinn-train/train.log --out results/train --title StarNet-S4
#
# Full output goes to results/<mode>-train (local only); train.log, losses.npz
# and train.json are archived to results/<platform>/<mode>-train for commit.
# Platform is auto-detected (a100, metax-c500, ...); CINN_PLATFORM overrides.
#
# Prereq: PaddleClas installed via PaddleX (see README.md).
set -euo pipefail

MODE=${MODE:-dynamic}
PADDLEX=${PADDLEX:-/data/PaddleX-release-3.7}
CLAS="$PADDLEX/paddlex/repo_manager/repos/PaddleClas"
HERE=$(cd "$(dirname "$0")" && pwd)
PYTHON_BIN=${PYTHON_BIN:-python}
CFG=${TRAIN_CONFIG:-$HERE/train_config.yaml}
DATASET=${DATASET_DIR:-${DATASET_ROOT:-/data/paddlex_train/datasets}/cls_flowers_examples}
EPOCHS=${EPOCHS:-32}          # flowers 1020/bs64 -> 16 steps/epoch, 32 -> ~512 steps
SEED=${SEED:-42}
export CUDA_VISIBLE_DEVICES=${GPU_ID:-${CUDA_VISIBLE_DEVICES:-0}}

case "$MODE" in
  dynamic) USE_CINN=False ;;
  cinn)    USE_CINN=True ;;
  *) echo "MODE must be dynamic|cinn" >&2; exit 2 ;;
esac
# Reproducible training; DETERMINISTIC=0 keeps upstream behavior. Without
# FLAGS_cudnn_deterministic the conv backward picks non-deterministic algorithms
# and two identical runs drift apart from step 2 (last-50 mean 1.79-2.99).
if [ "${DETERMINISTIC:-1}" = 1 ]; then
  export FLAGS_cudnn_deterministic=true
fi
[[ -d "$CLAS" ]] || { echo "PaddleClas not found under PADDLEX=$PADDLEX" >&2; exit 2; }
[[ -f "$DATASET/train.txt" ]] || { echo "Dataset not found: $DATASET (set DATASET_DIR)" >&2; exit 2; }

DEVICE=${PADDLE_DEVICE:-$("$PYTHON_BIN" "$HERE/archive.py" device | tail -n1)}
PLATFORM=$("$PYTHON_BIN" "$HERE/archive.py" platform | tail -n1)
RESULTS=${RESULTS_DIR:-$HERE/results}
OUT="$RESULTS/${MODE}-train"
[[ ! -e "$OUT" ]] || { echo "Refusing to overwrite $OUT" >&2; exit 3; }
mkdir -p "$OUT"

export PADDLE_PDX_DISABLE_DEV_MODEL_WL=true
cd "$CLAS"
set +e
PYTHONUNBUFFERED=1 "$PYTHON_BIN" tools/train.py -c "$CFG" \
  -o Global.use_cinn=$USE_CINN \
  -o Global.epochs=$EPOCHS \
  -o Global.seed=$SEED \
  -o Global.eval_during_train=False \
  -o Global.save_interval=100000 \
  -o Global.print_batch_step=1 \
  -o Global.output_dir="$OUT" \
  -o Global.device=$DEVICE \
  -o DataLoader.Train.dataset.image_root="$DATASET" \
  -o DataLoader.Train.dataset.cls_label_path="$DATASET/train.txt" \
  -o DataLoader.Eval.dataset.image_root="$DATASET" \
  -o DataLoader.Eval.dataset.cls_label_path="$DATASET/val.txt" \
  -o Infer.PostProcess.class_id_map_file="$DATASET/label.txt" 2>&1 | tee "$OUT/train.log"
rc=${PIPESTATUS[0]}
set -e
"$PYTHON_BIN" "$HERE/archive.py" train --parser clas --mode "$MODE" --device "$DEVICE" --rc "$rc" \
  --log "$OUT/train.log" --dest "$RESULTS/$PLATFORM/${MODE}-train"
echo "done MODE=$MODE rc=$rc -> $OUT/train.log"
exit "$rc"
