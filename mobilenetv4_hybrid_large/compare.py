"""Compare the dynamic and CINN output tensors of one model run."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

root = Path(__file__).parent

p = argparse.ArgumentParser()
p.add_argument("--dynamic", type=Path, required=True)
p.add_argument("--cinn", type=Path, required=True)
p.add_argument("--report", type=Path, required=True)
args = p.parse_args()

spec = json.loads((root / "model_spec.json").read_text())
compare = spec.get("compare", {"kind": "tensor", "atol": 2e-3})
atol = compare.get("atol", 2e-3)

dyn = np.load(args.dynamic / "outputs.npz")
cinn = np.load(args.cinn / "outputs.npz")
if sorted(dyn.files) != sorted(cinn.files):
    raise SystemExit(f"output names differ: {dyn.files} vs {cinn.files}")

report = dict(model=spec["model_name"], kind=compare["kind"], atol=atol, outputs={})
ok = True
for name in dyn.files:
    a, b = dyn[name], cinn[name]
    if a.shape != b.shape:
        report["outputs"][name] = dict(shapes=[list(a.shape), list(b.shape)], ok=False)
        ok = False
        continue
    entry = dict(shape=list(a.shape))
    if compare["kind"] == "detection" and name == compare.get("detection_output"):
        # NMS row order on this hardware is not run-to-run stable and low-score
        # box membership flips on near ties, so compare the detection count and
        # the score distribution of the top-K highest-scoring boxes. The full
        # distribution and raw row diffs are recorded for reference only.
        score_axis = compare.get("score_index", 1)
        topk = compare.get("topk", 50)
        count_name = compare.get("count_output")
        scores_a = np.sort(a.reshape(-1, a.shape[-1])[:, score_axis].astype("float64"))[::-1]
        scores_b = np.sort(b.reshape(-1, b.shape[-1])[:, score_axis].astype("float64"))[::-1]
        entry["topk"] = topk
        entry["topk_sorted_score_max_abs_diff"] = float(
            np.max(np.abs(scores_a[:topk] - scores_b[:topk]))
        )
        entry["full_sorted_score_max_abs_diff"] = float(
            np.max(np.abs(scores_a - scores_b))
        )
        entry["raw_row_max_abs_diff"] = float(np.max(np.abs(a.astype("float64") - b)))
        entry["ok"] = entry["topk_sorted_score_max_abs_diff"] <= atol
        if count_name in dyn.files:
            entry["count"] = [int(dyn[count_name].item()), int(cinn[count_name].item())]
            entry["ok"] = entry["ok"] and entry["count"][0] == entry["count"][1]
    else:
        diff = float(np.max(np.abs(a.astype("float64") - b.astype("float64"))))
        entry["max_abs_diff"] = diff
        entry["ok"] = diff <= atol
    report["outputs"][name] = entry
    ok = ok and entry["ok"]

report["ok"] = ok
args.report.write_text(json.dumps(report, indent=2))
print(json.dumps(report))
sys.exit(0 if ok else 1)
