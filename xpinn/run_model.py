"""Entry wrapper: dispatch a single inference run to infer_impl.py."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

p = argparse.ArgumentParser()
p.add_argument("--task", choices=["infer"], required=True)
p.add_argument("--mode", choices=["dynamic", "cinn"], required=True)
p.add_argument("--output", required=True)
p.add_argument("--checkpoint", default="")
p.add_argument("--inputs", default=os.environ.get("INPUTS_NPZ", ""))
p.add_argument("--warmup", default=os.environ.get("WARMUP", "5"))
p.add_argument("--iters", default=os.environ.get("ITERS", "50"))
p.add_argument("--device", default=os.environ.get("PADDLE_DEVICE", "auto"))
a = p.parse_args()

root = Path(__file__).parent
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
if a.checkpoint:
    cmd += ["--checkpoint", a.checkpoint]
if a.inputs:
    cmd += ["--inputs", a.inputs]
subprocess.run(cmd, check=True)
