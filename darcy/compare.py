"""Compare the dynamic and CINN results of one PaddleScience case.

Handles both stages:
- infer: output tensors in outputs.npz, element-wise max abs diff <= spec atol
- train: per-step losses in losses.npz, compared over the acceptance window
  with the same rule as PaddleScience's harness (relative diff above 1,
  absolute diff otherwise; at most `max_outliers` steps above `tol`)
"""
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
train_spec = spec.get("train", {})
window = tuple(train_spec.get("window", [400, 500]))
tol = train_spec.get("tol", 1e-2)
max_outliers = train_spec.get("max_outliers", 10)

report = dict(case=spec["case"], atol=atol)
ok = True
compared = False


def compare_outputs(dyn_dir, cinn_dir):
    dyn = np.load(dyn_dir / "outputs.npz")
    cinn = np.load(cinn_dir / "outputs.npz")
    if sorted(dyn.files) != sorted(cinn.files):
        raise SystemExit(f"output names differ: {dyn.files} vs {cinn.files}")
    results = {}
    passed = True
    for name in dyn.files:
        a, b = dyn[name], cinn[name]
        if a.shape != b.shape:
            results[name] = dict(shapes=[list(a.shape), list(b.shape)], ok=False)
            passed = False
            continue
        diff = float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))
        entry = dict(
            shape=list(a.shape),
            max_abs_diff=diff,
            max_abs_value=float(np.max(np.abs(a)).astype(np.float64)) if a.size else 0.0,
            ok=diff <= atol,
        )
        results[name] = entry
        passed = passed and entry["ok"]
    return results, passed


def compare_losses(dyn_dir, cinn_dir):
    dyn = np.load(dyn_dir / "losses.npz")["loss"].astype(np.float64)
    cinn = np.load(cinn_dir / "losses.npz")["loss"].astype(np.float64)
    if dyn.shape != cinn.shape:
        return dict(shapes=[list(dyn.shape), list(cinn.shape)], ok=False), False
    relative = np.abs(dyn) > 1.0
    diff = np.where(relative, np.abs(dyn - cinn) / np.maximum(np.abs(dyn), 1e-12), np.abs(dyn - cinn))
    lo, hi = window
    idx = np.arange(len(dyn)) + 1
    mask = (idx >= lo) & (idx <= hi)
    if mask.any():
        window_idx, window_diff = idx[mask], diff[mask]
    else:
        # round of steps shorter than the acceptance window: compare them all
        window_idx, window_diff = idx, diff
    outlier_steps = [int(i) for i in window_idx[window_diff > tol]]
    passed = len(outlier_steps) <= max_outliers
    return (
        dict(
            window=[lo, hi],
            tol=tol,
            max_outliers=max_outliers,
            total_steps=int(len(dyn)),
            first_loss_diff=float(diff[0]),
            final_loss_diff=float(diff[-1]),
            window_max_diff=float(window_diff.max()) if window_diff.size else 0.0,
            window_mean_diff=float(window_diff.mean()) if window_diff.size else 0.0,
            outliers=len(outlier_steps),
            outlier_steps=outlier_steps[:20],
            ok=passed,
        ),
        passed,
    )


if (args.dynamic / "outputs.npz").is_file() and (args.cinn / "outputs.npz").is_file():
    report["outputs"], passed = compare_outputs(args.dynamic, args.cinn)
    ok = ok and passed
    compared = True
if (args.dynamic / "losses.npz").is_file() and (args.cinn / "losses.npz").is_file():
    report["train"], passed = compare_losses(args.dynamic, args.cinn)
    ok = ok and passed
    compared = True
if not compared:
    raise SystemExit("no comparable artifacts (outputs.npz / losses.npz) in both directories")

report["ok"] = ok
args.report.write_text(json.dumps(report, indent=2))
print(json.dumps(report), flush=True)
sys.exit(0 if ok else 1)
