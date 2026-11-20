"""Run a PaddleX official inference model in dynamic (PIR) or CINN mode.

The graph comes from the PaddleX official inference artifact
(inference.json + inference.pdiparams) and is executed through
paddle.inference, which is what PaddleX's PaddleStaticRunner does with
`enable_cinn` toggled per mode. Pre/post-processing is deliberately out of
scope: both modes receive the same raw tensors so only graph execution is
compared.

The entry is device agnostic: CUDA builds use gpu:0, Metax builds use the
first registered custom device (metax_gpu:0), and PADDLE_DEVICE overrides
the choice. Inputs are seeded and also dumped to inputs.npz so the other
platform can replay the exact same tensors (--inputs).
"""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

root = Path(__file__).parent

p = argparse.ArgumentParser()
p.add_argument("--mode", choices=["dynamic", "cinn"], required=True)
p.add_argument("--weights", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--warmup", type=int, default=5)
p.add_argument("--iters", type=int, default=50)
p.add_argument(
    "--device",
    default=os.environ.get("PADDLE_DEVICE", "auto"),
    help="auto|gpu:0|metax_gpu:0|cpu; auto prefers CUDA, then custom device",
)
p.add_argument(
    "--inputs",
    type=Path,
    help="reuse inputs.npz produced by another platform/run",
)
args = p.parse_args()
args.output.mkdir(parents=True, exist_ok=False)

spec = json.loads((root / "model_spec.json").read_text())
os.environ.setdefault("FLAGS_enable_pir_api", "1")
os.environ.setdefault("FLAGS_cudnn_deterministic", "true")
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"

import paddle
import paddle.inference as pi


def resolve_device(requested):
    if requested != "auto":
        return requested
    if paddle.device.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0:
        return "gpu:0"
    custom = paddle.device.get_all_custom_device_type() or []
    if custom:
        return f"{custom[0]}:0"
    return "cpu"


device = resolve_device(args.device)
device_kind, _, device_index = device.partition(":")
device_id = int(device_index or 0)

config = pi.Config(
    str(args.weights / "inference.json"),
    str(args.weights / "inference.pdiparams"),
)
if device_kind == "gpu":
    config.enable_use_gpu(100, device_id)
    config.disable_mkldnn()
elif device_kind == "cpu":
    config.disable_gpu()
else:
    config.enable_custom_device(device_kind, device_id)
config.enable_new_ir(True)
config.enable_new_executor()
config.set_optimization_level(3)
if args.mode == "cinn":
    if not paddle.is_compiled_with_cinn():
        raise SystemExit("paddle is not compiled with CINN; rebuild with WITH_CINN=ON")
    config.enable_cinn()

predictor = pi.create_predictor(config)

input_names = predictor.get_input_names()
if sorted(input_names) != sorted(spec["feeds"]):
    raise SystemExit(
        f"graph inputs {sorted(input_names)} do not match model_spec feeds "
        f"{sorted(spec['feeds'])}"
    )

if args.inputs:
    with np.load(args.inputs) as stored:
        feeds = {name: stored[name] for name in stored.files}
    missing = set(input_names) - set(feeds)
    if missing:
        raise SystemExit(f"{args.inputs} misses graph inputs {sorted(missing)}")
    feeds = {name: feeds[name] for name in input_names}
else:
    rng = np.random.default_rng(spec.get("seed", 20261119))
    feeds = {}
    for name, feed in spec["feeds"].items():
        dtype = feed.get("dtype", "float32")
        if "asset" in feed:
            with np.load(root / feed["asset"]) as stored:
                arr = stored[feed.get("key", "image")]
            if list(arr.shape) != feed["shape"]:
                raise SystemExit(
                    f"asset {feed['asset']} shape {list(arr.shape)} != "
                    f"model_spec shape {feed['shape']}"
                )
            feeds[name] = (arr.astype(dtype) * feed.get("scale", 1.0)).astype(dtype)
        elif "constant" in feed:
            feeds[name] = np.asarray(feed["constant"], dtype=dtype)
        else:
            feeds[name] = rng.random(feed["shape"], dtype=np.float64).astype(dtype)

for name in input_names:
    predictor.get_input_handle(name).copy_from_cpu(feeds[name])
np.savez(args.output / "inputs.npz", **feeds)

np.random.seed(spec.get("seed", 20261119))

with paddle.no_grad():
    paddle.device.synchronize()
    start = time.perf_counter()
    predictor.run()
    paddle.device.synchronize()
    first_call_ms = (time.perf_counter() - start) * 1000

    for _ in range(args.warmup):
        predictor.run()
    paddle.device.synchronize()

    start = time.perf_counter()
    for _ in range(args.iters):
        predictor.run()
    paddle.device.synchronize()
    steady_ms = (time.perf_counter() - start) * 1000 / args.iters

output_names = predictor.get_output_names()
outputs = {
    name: predictor.get_output_handle(name).copy_to_cpu() for name in output_names
}
np.savez(args.output / "outputs.npz", **outputs)

digest = hashlib.sha256()
for name in sorted(feeds):
    digest.update(name.encode())
    digest.update(np.ascontiguousarray(feeds[name]).tobytes())

import paddlex

report = dict(
    model=spec["model_name"],
    mode=args.mode,
    weights=str(args.weights),
    device=device,
    paddle_version=paddle.__version__,
    paddle_compiled_with_cinn=paddle.is_compiled_with_cinn(),
    paddlex_version=paddlex.__version__,
    seed=spec.get("seed", 20261119),
    inputs_sha256=digest.hexdigest(),
    feeds={k: list(v.shape) for k, v in feeds.items()},
    outputs={k: list(v.shape) for k, v in outputs.items()},
    warmup=args.warmup,
    iters=args.iters,
    first_call_ms=first_call_ms,
    steady_batch_ms=steady_ms,
    batch_per_second=1000.0 / steady_ms,
)
(args.output / "inference.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report))
