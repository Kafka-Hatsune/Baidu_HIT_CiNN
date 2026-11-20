#!/usr/bin/env python
"""Overlay dynamic vs CINN training loss on one figure.

Reads the per-step training-loss lines from two logs (one dynamic run, one with
CINN enabled) and writes `loss_curve.csv` (step, dynamic_loss, cinn_loss) plus
`loss_curve.png` with both curves overlaid. The x-axis is the optimizer step;
no learning-rate axis. Handles the three PaddleX backend log formats:
  - PaddleClas: `[Train][Epoch E/T][Iter: i/n]... loss: X`
  - PaddleDetection: `Epoch: [E] [ i/N] ... loss: X ...`
  - PaddleOCR: `epoch: [E/T], ... iter: i, ... loss: X`

Usage:
    python plot_loss.py --dynamic results/dynamic-train/train.log \
                        --cinn results/cinn-train/train.log \
                        --out results/train --title "PP-YOLOE_plus-S"
"""
import argparse
import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

STEP = re.compile(r"\[Train\]\[Epoch \d+|Epoch: \[\d+\] \[|epoch: \[\d+")
LOSS = re.compile(r"\bloss: ([0-9.]+)")


def parse(log):
    out = []
    for line in Path(log).read_text(errors="ignore").splitlines():
        if "Eval" in line or "eval" in line:
            continue
        if not STEP.search(line):
            continue
        m = LOSS.search(line)
        if m:
            out.append(float(m.group(1)))
    if not out:
        raise SystemExit(f"no training loss lines found in {log}")
    return out


p = argparse.ArgumentParser()
p.add_argument("--dynamic", type=Path, required=True)
p.add_argument("--cinn", type=Path, required=True)
p.add_argument("--out", type=Path, required=True)
p.add_argument("--title", default="train loss: dynamic vs CINN")
a = p.parse_args()
a.out.mkdir(parents=True, exist_ok=True)

dyn = parse(a.dynamic)
cinn = parse(a.cinn)
n = min(len(dyn), len(cinn))

csv_path = a.out / "loss_curve.csv"
with csv_path.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["step", "dynamic_loss", "cinn_loss"])
    for i in range(max(len(dyn), len(cinn))):
        w.writerow(
            [
                i + 1,
                f"{dyn[i]:.6f}" if i < len(dyn) else "",
                f"{cinn[i]:.6f}" if i < len(cinn) else "",
            ]
        )

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(range(1, len(dyn) + 1), dyn, color="#1f77b4", lw=1.2, label="dynamic")
ax.plot(range(1, len(cinn) + 1), cinn, color="#d62728", lw=1.2, alpha=0.8, label="CINN")
ax.set_xlabel("training step")
ax.set_ylabel("loss")
ax.grid(True, alpha=0.3)
ax.legend()
plt.title(f"{a.title} ({n} steps, MetaX)")
fig.tight_layout()
png_path = a.out / "loss_curve.png"
fig.savefig(png_path, dpi=130)

print(f"dynamic {len(dyn)} steps ({dyn[0]:.4f} -> {dyn[-1]:.4f})")
print(f"cinn    {len(cinn)} steps ({cinn[0]:.4f} -> {cinn[-1]:.4f})")
print(f"wrote {csv_path}")
print(f"wrote {png_path}")
