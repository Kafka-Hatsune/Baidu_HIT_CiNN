"""Entry wrapper: dispatch an inference or training run to its impl script."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

p = argparse.ArgumentParser()
p.add_argument("--task", choices=["infer", "train"], required=True)
p.add_argument("--mode", choices=["dynamic", "cinn"], required=True)
p.add_argument("--output", required=True)
p.add_argument("--checkpoint", default="")
p.add_argument("--warmup", default=os.environ.get("WARMUP", "5"))
p.add_argument("--iters", default=os.environ.get("ITERS", "50"))
p.add_argument("--device", default=os.environ.get("PADDLE_DEVICE", "auto"))
a = p.parse_args()

root = Path(__file__).parent
if a.task == "infer":
    cmd = [
        sys.executable,
        str(root / "infer_impl.py"),
        "--mode",
        a.mode,
        "--output",
        a.output,
        "--warmup",
        a.warmup,
        "--iters",
        a.iters,
        "--device",
        a.device,
    ]
    inputs = os.environ.get("INPUTS_NPZ", "")
    if a.checkpoint:
        cmd += ["--checkpoint", a.checkpoint]
    if inputs:
        cmd += ["--inputs", inputs]
else:
    cmd = [
        sys.executable,
        str(root / "train_impl.py"),
        "--mode",
        a.mode,
        "--output",
        a.output,
        "--warmup",
        a.warmup,
        "--iters",
        a.iters,
        "--device",
        a.device,
    ]
    batch = os.environ.get("BATCH_NPZ", "")
    if a.checkpoint:
        cmd += ["--checkpoint", a.checkpoint]
    if batch:
        cmd += ["--batch", batch]
subprocess.run(cmd, check=True)
