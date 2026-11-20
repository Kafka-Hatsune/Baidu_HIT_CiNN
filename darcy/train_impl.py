"""Run one PaddleScience CINN test case as a short training job.

The model, batch and loss come from $PADDLE_PSCI_DIR/test/cinn/cases, so the
training semantics (Adam, learning rate, PDE residual loss, dropout disabled)
match PaddleScience's own CINN harness. Weights and the batch are created on
CPU with a fixed seed and dumped, so dynamic and CINN runs (and A100 vs MetaX)
start from bit-identical parameters and data and their per-step losses can be
compared.

Steps are split like the upstream harness: `warmup` leading steps are excluded
from the timing, `iters` more steps are timed; every step's loss is recorded.
"""
import argparse
import hashlib
import json
import os
import statistics
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
    "--batch",
    type=Path,
    help="batch.npz (inputs/..., labels/...) dumped by another run/platform",
)
p.add_argument("--warmup", type=int, default=5)
p.add_argument("--iters", type=int, default=500)
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


def disable_dropout(model):
    """Dropout draws differ between dynamic and static mode, so set p=0."""
    dropout_types = (
        paddle.nn.Dropout,
        paddle.nn.Dropout2D,
        paddle.nn.Dropout3D,
        paddle.nn.AlphaDropout,
    )
    count = 0
    for layer in model.sublayers(include_self=True):
        if isinstance(layer, dropout_types):
            layer.p = 0.0
            count += 1
    return count


def save_batch(path, inputs, labels):
    arrays, flags = {}, {}
    for group, mapping in (("inputs", inputs), ("labels", labels)):
        for key, value in mapping.items():
            name = f"{group}/{key}"
            arrays[name] = value.numpy()
            flags[name] = not value.stop_gradient
    np.savez(path, **arrays)
    path.with_suffix(".json").write_text(json.dumps(flags))


def load_batch(path):
    flags = json.loads(path.with_suffix(".json").read_text())
    inputs, labels = {}, {}
    with np.load(path) as stored:
        for name in stored.files:
            group, key = name.split("/", 1)
            tensor = paddle.to_tensor(stored[name])
            tensor.stop_gradient = not flags[name]
            (inputs if group == "inputs" else labels)[key] = tensor
    return inputs, labels


class LossNet(paddle.nn.Layer):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, inputs, labels):
        return case.compute_loss(self.inner, inputs, labels)


target = resolve_device(args.device)

# --- model and batch on CPU, so the RNG stream is platform independent
paddle.set_device("cpu")
paddle.seed(spec["seed"])
np.random.seed(spec["seed"])
model = case.build_model()
dropout_layers = disable_dropout(model)
if args.checkpoint:
    model.set_state_dict(paddle.load(str(args.checkpoint)))
else:
    paddle.save(model.state_dict(), str(args.output / "init.pdparams"))

if args.batch:
    batch_path = args.batch
    # load_batch needs the flags next to the npz; copy them into the output dir
    flags_src = batch_path.with_suffix(".json")
    if flags_src.is_file() and flags_src != (args.output / "batch.json"):
        (args.output / "batch.json").write_text(flags_src.read_text())
        batch_path = args.output / "batch.npz"
        with np.load(args.batch) as stored:
            np.savez(batch_path, **{key: stored[key] for key in stored.files})
else:
    inputs_cpu, labels_cpu = case.build_batch()
    save_batch(args.output / "batch.npz", inputs_cpu, labels_cpu)
    del inputs_cpu, labels_cpu
    batch_path = args.output / "batch.npz"

paddle.set_device(target)
model.train()
model.to(device=target)
inputs, labels = load_batch(batch_path)
if hasattr(case, "prepare"):
    case.prepare(model)


def digest_of(mapping):
    digest = hashlib.sha256()
    for name in sorted(mapping):
        digest.update(name.encode())
        digest.update(np.ascontiguousarray(mapping[name]).tobytes())
    return digest.hexdigest()


init_weights = {name: value.numpy() for name, value in model.state_dict().items()}
init_weights_sha256 = digest_of(init_weights)
del init_weights

net = LossNet(model)
if args.mode == "cinn":
    if not paddle.is_compiled_with_cinn():
        raise SystemExit("paddle is not compiled with CINN; rebuild with WITH_CINN=ON")

    def spec_of(mapping):
        return {
            name: paddle.static.InputSpec(shape=t.shape, dtype=t.dtype)
            for name, t in mapping.items()
        }

    net = paddle.jit.to_static(
        net,
        input_spec=[spec_of(inputs), spec_of(labels)],
        full_graph=True,
        backend="CINN",
    )

learning_rate = getattr(case, "LEARNING_RATE", 1e-3)
optimizer = paddle.optimizer.Adam(learning_rate, parameters=model.parameters())
grad_inputs = [t for t in inputs.values() if not t.stop_gradient]
total_iters = args.iters + args.warmup

times, losses = [], []
for step in range(total_iters):
    start = time.perf_counter()
    loss = net(inputs, labels)
    loss.backward()
    optimizer.step()
    optimizer.clear_grad()
    for tensor in grad_inputs:
        tensor.clear_gradient()
    paddle.device.synchronize()
    times.append(time.perf_counter() - start)
    losses.append(float(loss))

steady = times[args.warmup:]
np.savez(args.output / "losses.npz", loss=np.asarray(losses, dtype="float64"))
paddle.save(model.state_dict(), str(args.output / "final.pdparams"))


final_weights = {name: value.numpy() for name, value in model.state_dict().items()}
report = dict(
    case=spec["case"],
    mode=args.mode,
    task="train",
    model=spec.get("label", spec["case"]),
    device=target,
    paddle_version=paddle.__version__,
    paddle_compiled_with_cinn=paddle.is_compiled_with_cinn(),
    ppsci_version=getattr(importlib.import_module("ppsci"), "__version__", "unknown"),
    ppsci_commit=os.popen(f"git -C {ps_dir} rev-parse HEAD").read().strip(),
    seed=spec["seed"],
    learning_rate=learning_rate,
    dropout_layers_disabled=dropout_layers,
    warmup=args.warmup,
    iters=args.iters,
    total_steps=total_iters,
    init_weights_sha256=init_weights_sha256,
    final_weights_sha256=digest_of(final_weights),
    batch_sha256=hashlib.sha256(batch_path.read_bytes()).hexdigest(),
    first_step_ms=times[0] * 1000,
    steady_step_ms=statistics.mean(steady) * 1000,
    steady_median_ms=statistics.median(steady) * 1000,
    losses=losses,
)
(args.output / "train.json").write_text(json.dumps(report, indent=2))
print(
    json.dumps({key: value for key, value in report.items() if key != "losses"}),
    flush=True,
)
