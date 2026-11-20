#!/usr/bin/env bash
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
: "${PADDLE_PSCI_DIR:?Set PADDLE_PSCI_DIR to the PaddleScience checkout}"
export PADDLE_PSCI_DIR
export CUDA_VISIBLE_DEVICES=${GPU_ID:-0}
MODE=${MODE:-dynamic}
TASK=${TASK:-infer}
PYTHON_BIN=${PYTHON_BIN:-python}
case "$MODE" in dynamic|cinn) ;; *) echo "MODE must be dynamic or cinn"; exit 2 ;; esac
case "$TASK" in infer|train) ;; *) echo "TASK must be infer or train"; exit 2 ;; esac
[[ $(git -C "$PADDLE_PSCI_DIR" rev-parse HEAD) = $(tr -d '[:space:]' < "$here/paddlescience.commit") ]] || {
    echo "Wrong PaddleScience commit: expected $(cat "$here/paddlescience.commit")"; exit 2; }

output_root=${RESULTS_DIR:-$here/results}
mkdir -p "$output_root"
output="$output_root/$MODE-$TASK"
[[ ! -e "$output" ]] || { echo "Refusing to overwrite $output"; exit 3; }

# CINN reuses the dynamic run's weights and batch so both modes see bit-identical input.
args=()
if [[ -f "${CHECKPOINT_DIR:-}/init.pdparams" ]]; then
    args+=(--checkpoint "$CHECKPOINT_DIR/init.pdparams")
elif [[ "$MODE" = cinn && -f "$output_root/dynamic-$TASK/init.pdparams" ]]; then
    args+=(--checkpoint "$output_root/dynamic-$TASK/init.pdparams")
fi
if [[ "$TASK" = train && "$MODE" = cinn && -z "${BATCH_NPZ:-}" && -f "$output_root/dynamic-train/batch.npz" ]]; then
    export BATCH_NPZ="$output_root/dynamic-train/batch.npz"
fi
if [[ "$TASK" = infer && "$MODE" = cinn && -z "${INPUTS_NPZ:-}" && -f "$output_root/dynamic-infer/inputs.npz" ]]; then
    export INPUTS_NPZ="$output_root/dynamic-infer/inputs.npz"
fi

"$PYTHON_BIN" "$here/run_model.py" --mode "$MODE" --task "$TASK" --output "$output" "${args[@]}" 2>&1 | tee "$output.log"
if [[ "$MODE" = cinn ]]; then
    grep -q 'Compiling subgraph with CINN backend' "$output.log" || { echo 'CINN did not compile any subgraph'; exit 4; }
fi

other=$([[ "$MODE" = cinn ]] && echo dynamic || echo cinn)
report="$output_root/comparison.json"
[[ "$TASK" = train ]] && report="$output_root/comparison-train.json"
if [[ -d "$output_root/$other-$TASK" ]]; then
    "$PYTHON_BIN" "$here/compare.py" \
        --dynamic "$output_root/dynamic-$TASK" \
        --cinn "$output_root/cinn-$TASK" \
        --report "$report" 2>&1 | tee "$output_root/comparison-$TASK.log"
fi
