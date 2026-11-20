"""Compare the dynamic and CINN output tensors of one PaddleScience case."""
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
atol = spec.get("atol", 1e-2)

dyn = np.load(args.dynamic / "outputs.npz")
cinn = np.load(args.cinn / "outputs.npz")
if sorted(dyn.files) != sorted(cinn.files):
    raise SystemExit(f"output names differ: {dyn.files} vs {cinn.files}")

report = dict(case=spec["case"], atol=atol, outputs={})
ok = True
for name in dyn.files:
    a, b = dyn[name], cinn[name]
    if a.shape != b.shape:
        report["outputs"][name] = dict(shapes=[list(a.shape), list(b.shape)], ok=False)
        ok = False
        continue
    diff = float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))
    scale = float(np.max(np.abs(a)).astype(np.float64)) if a.size else 0.0
    entry = dict(
        shape=list(a.shape),
        max_abs_diff=diff,
        max_abs_value=scale,
        ok=diff <= atol,
    )
    report["outputs"][name] = entry
    ok = ok and entry["ok"]

report["ok"] = ok
args.report.write_text(json.dumps(report, indent=2))
print(json.dumps(report), flush=True)
sys.exit(0 if ok else 1)
