"""Run one PaddleScience CINN test case in dynamic or CINN mode.

The case definition (architecture + synthetic batch) is taken from
$PADDLE_PSCI_DIR/test/cinn/cases so this directory stays in sync with the
upstream PaddleScience CINN suite instead of duplicating model code.

Weights are initialised on CPU with a fixed seed, then moved to the target
device: the CPU RNG stream is platform independent, so A100 and MetaX start
from bit-identical parameters. The dynamic run dumps init.pdparams and
inputs.npz; the CINN run loads them, so both modes (and both platforms)
execute exactly the same model and batch.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

root = Path(__file__).parent

p = argparse.ArgumentParser()
p.add_argument("--mode", choices=["dynamic", "cinn"], required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument(
    "--checkpoint",
    type=Path,
    help="init.pdparams dumped by a dynamic run; generated deterministically if absent",
)
p.add_argument(
    "--inputs",
    type=Path,
    help="inputs.npz dumped by another run/platform",
)
p.add_argument("--warmup", type=int, default=5)
p.add_argument("--iters", type=int, default=50)
p.add_argument(
    "--device",
    default=os.environ.get("PADDLE_DEVICE", "auto"),
    help="auto|gpu:0|metax_gpu:0|cpu; auto prefers CUDA, then custom device",
)
args = p.parse_args()
args.output.mkdir(parents=True, exist_ok=False)

spec = json.loads((root / "model_spec.json").read_text())
ps_dir = os.environ.get("PADDLE_PSCI_DIR")
if not ps_dir:
    raise SystemExit("Set PADDLE_PSCI_DIR to the PaddleScience checkout")
sys.path.insert(0, ps_dir)
sys.path.insert(0, os.path.join(ps_dir, "test", "cinn"))
os.environ.setdefault("FLAGS_enable_pir_api", "1")
os.environ.setdefault("FLAGS_cudnn_deterministic", "true")
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"

import importlib

import paddle

case = importlib.import_module(f"cases.{spec['case']}")


def resolve_device(requested):
    if requested != "auto":
        return requested
    if paddle.device.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0:
        return "gpu:0"
    custom = paddle.device.get_all_custom_device_type() or []
    if custom:
        return f"{custom[0]}:0"
    return "cpu"


target = resolve_device(args.device)

# --- model: initialise on CPU so the RNG stream does not depend on the device
paddle.set_device("cpu")
paddle.seed(spec["seed"])
model = case.build_model()
if args.checkpoint:
    model.set_state_dict(paddle.load(str(args.checkpoint)))
else:
    paddle.save(model.state_dict(), str(args.output / "init.pdparams"))

# --- batch: generated on CPU as well, or replayed from another run
if args.inputs:
    with np.load(args.inputs) as stored:
        inputs_np = {name: stored[name] for name in stored.files}
else:
    inputs, _ = case.build_batch()
    inputs_np = {name: value.numpy() for name, value in inputs.items()}
    del inputs
np.savez(args.output / "inputs.npz", **inputs_np)

paddle.set_device(target)
model.eval()
model.to(device=target)
inputs = {name: paddle.to_tensor(value) for name, value in inputs_np.items()}
if hasattr(case, "prepare"):
    case.prepare(model)

# --- execution
has_case_forward = hasattr(case, "forward")
if args.mode == "cinn":
    if not paddle.is_compiled_with_cinn():
        raise SystemExit("paddle is not compiled with CINN; rebuild with WITH_CINN=ON")

    if has_case_forward:

        class Net(paddle.nn.Layer):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner

            def forward(self, data):
                return case.forward(self.inner, data)

    else:

        class Net(paddle.nn.Layer):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner

            def forward(self, data):
                return self.inner(data)

    net = paddle.jit.to_static(
        Net(model),
        input_spec=[
            {
                name: paddle.static.InputSpec(shape=t.shape, dtype=t.dtype)
                for name, t in inputs.items()
            }
        ],
        full_graph=True,
        backend="CINN",
    )
else:
    net = model
    if has_case_forward:
        forward = lambda data: case.forward(model, data)
    else:
        forward = model

with paddle.no_grad():
    paddle.device.synchronize()
    start = time.perf_counter()
    out = forward(inputs) if args.mode == "dynamic" else net(inputs)
    paddle.device.synchronize()
    first_call_ms = (time.perf_counter() - start) * 1000

    for _ in range(args.warmup):
        out = forward(inputs) if args.mode == "dynamic" else net(inputs)
    paddle.device.synchronize()

    start = time.perf_counter()
    for _ in range(args.iters):
        out = forward(inputs) if args.mode == "dynamic" else net(inputs)
    paddle.device.synchronize()
    steady_ms = (time.perf_counter() - start) * 1000 / args.iters

flat = {}


def flatten(obj, prefix=""):
    if paddle.is_tensor(obj):
        flat[prefix or "out"] = obj.numpy()
    elif isinstance(obj, dict):
        for key, value in obj.items():
            flatten(value, f"{prefix}.{key}" if prefix else str(key))
    elif isinstance(obj, (list, tuple)):
        for index, value in enumerate(obj):
            flatten(value, f"{prefix}[{index}]")


flatten(out)
np.savez(args.output / "outputs.npz", **flat)


def digest_of(mapping):
    digest = hashlib.sha256()
    for name in sorted(mapping):
        digest.update(name.encode())
        digest.update(np.ascontiguousarray(mapping[name]).tobytes())
    return digest.hexdigest()


weights = {name: value.numpy() for name, value in model.state_dict().items()}
report = dict(
    case=spec["case"],
    mode=args.mode,
    model=spec.get("label", spec["case"]),
    device=target,
    paddle_version=paddle.__version__,
    paddle_compiled_with_cinn=paddle.is_compiled_with_cinn(),
    ppsci_version=getattr(importlib.import_module("ppsci"), "__version__", "unknown"),
    ppsci_commit=os.popen(f"git -C {ps_dir} rev-parse HEAD").read().strip(),
    seed=spec["seed"],
    weights_sha256=digest_of(weights),
    inputs_sha256=digest_of(inputs_np),
    feeds={k: list(v.shape) for k, v in inputs_np.items()},
    outputs={k: list(v.shape) for k, v in flat.items()},
    warmup=args.warmup,
    iters=args.iters,
    first_call_ms=first_call_ms,
    steady_batch_ms=steady_ms,
    batch_per_second=1000.0 / steady_ms,
)
(args.output / "inference.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report), flush=True)
