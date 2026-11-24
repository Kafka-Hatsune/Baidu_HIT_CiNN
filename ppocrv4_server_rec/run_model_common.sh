#!/usr/bin/env bash
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
spec_model=$("${PYTHON_BIN:-python}" -c "import json,sys;print(json.load(open(sys.argv[1]))['model_name'])" "$here/model_spec.json")
export CHECKPOINT_DIR=${CHECKPOINT_DIR:-${PADDLEX_MODELS_DIR:-$HOME/.paddlex/official_models}/$spec_model}
export CUDA_VISIBLE_DEVICES=${GPU_ID:-${CUDA_VISIBLE_DEVICES:-0}}
MODE=${MODE:-dynamic}
PYTHON_BIN=${PYTHON_BIN:-python}
case "$MODE" in dynamic|cinn) ;; *) echo "MODE must be dynamic or cinn"; exit 2 ;; esac
[[ -f "$CHECKPOINT_DIR/inference.json" ]] || { echo "Missing inference.json under $CHECKPOINT_DIR"; exit 2; }
"$PYTHON_BIN" "$here/verify_assets.py" "$CHECKPOINT_DIR"

output_root=${RESULTS_DIR:-$here/results}
mkdir -p "$output_root"
output="$output_root/$MODE-infer"
[[ ! -e "$output" ]] || { echo "Refusing to overwrite $output"; exit 3; }

"$PYTHON_BIN" "$here/run_model.py" --mode "$MODE" --task infer --output "$output" --checkpoint "$CHECKPOINT_DIR" 2>&1 | tee "$output.log"
if [[ "$MODE" = cinn ]]; then
    grep -q 'Compiling subgraph with CINN backend' "$output.log" || { echo 'CINN did not compile any subgraph'; exit 4; }
fi

# Small results for cross-platform alignment go to results/<platform> (committed).
platform=$("$PYTHON_BIN" "$here/archive.py" platform | tail -n1)
archive="$output_root/$platform"
"$PYTHON_BIN" "$here/archive.py" infer --src "$output" --log "$output.log" --dest "$archive/$MODE-infer"

other=$([[ "$MODE" = cinn ]] && echo dynamic || echo cinn)
if [[ -d "$output_root/$other-infer" ]]; then
    rc=0
    "$PYTHON_BIN" "$here/compare.py" \
        --dynamic "$output_root/dynamic-infer" \
        --cinn "$output_root/cinn-infer" \
        --report "$output_root/comparison.json" 2>&1 | tee "$output_root/comparison.log" || rc=$?
    cp "$output_root/comparison.json" "$output_root/comparison.log" "$archive/"
    exit "$rc"
fi
