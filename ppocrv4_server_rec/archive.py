#!/usr/bin/env python
"""Platform detection and archiving of small alignment results.

Local runs write everything under results/<mode>-<task>; this script copies the
small artifacts used for cross-platform analysis into results/<platform>/, the
only part of results/ that is committed (see .gitignore).

    archive.py device     # gpu | metax_gpu | cpu  (PADDLE_DEVICE not set)
    archive.py platform   # a100 | metax-c500 | ...  (CINN_PLATFORM overrides)
    archive.py train --parser clas|ocr|det --mode M --rc N --log LOG --dest DIR
    archive.py infer --src results/<mode>-infer --log LOG --dest DIR
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import sys

# Custom devices have no queryable product name; map the device type instead.
CUSTOM_PLATFORMS = {"metax_gpu": "metax-c500"}
# Inference outputs above this size stay local (e.g. OCR det probability maps).
MAX_OUTPUTS_BYTES = int(os.environ.get("ARCHIVE_MAX_OUTPUTS_BYTES", 2 << 20))

CLAS_RE = re.compile(
    r"\[Train\]\[Epoch (\d+)/(\d+)\]\[Iter: (\d+)/(\d+)\].*?\bloss: ([-+0-9.eE]+|nan|inf),.*?batch_cost: ([-+0-9.eE]+)s"
)
OCR_RE = re.compile(
    r"global_step: (\d+).*?\bloss: ([-+0-9.eE]+|nan|inf),.*?avg_batch_cost: ([-+0-9.eE]+) s"
)
DET_RE = re.compile(
    r"Epoch: \[(\d+)\] \[\s*(\d+)/(\d+)\].*?\bloss: ([-+0-9.eE]+|nan|inf).*?batch_cost: ([-+0-9.eE]+)"
)


def parse_clas(text):
    for line in text.splitlines():
        if "[Avg]" in line:
            continue
        m = CLAS_RE.search(line)
        if m:
            epoch, _total, it, per, loss, cost = m.groups()
            yield (int(epoch) - 1) * int(per) + int(it), float(loss), float(cost)


def parse_ocr(text):
    for m in OCR_RE.finditer(text):
        step, loss, cost = m.groups()
        yield int(step), float(loss), float(cost)


def parse_det(text):
    for m in DET_RE.finditer(text):
        epoch, it, per, loss, cost = m.groups()
        yield int(epoch) * int(per) + int(it), float(loss), float(cost)


PARSERS = {"clas": parse_clas, "ocr": parse_ocr, "det": parse_det}


def device_kind():
    """Same preference as infer_impl.resolve_device: CUDA, then custom device."""
    import paddle

    if paddle.device.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0:
        return "gpu"
    custom = paddle.device.get_all_custom_device_type() or []
    return custom[0] if custom else "cpu"


def platform_name():
    if os.environ.get("CINN_PLATFORM"):
        return os.environ["CINN_PLATFORM"]
    kind = device_kind()
    if kind == "gpu":
        import paddle

        name = paddle.device.cuda.get_device_name(0)  # e.g. NVIDIA A100-SXM4-40GB
        m = re.search(r"\b([A-Z]+\d+[A-Z]*)\b", name.replace("NVIDIA", ""))
        return m.group(1).lower() if m else re.sub(r"\W+", "-", name).strip("-").lower()
    return CUSTOM_PLATFORMS.get(kind, kind)


def train_status(text, losses, rc):
    if any(v != v or v in (float("inf"), float("-inf")) for v in losses):
        return "diverged(nan)"
    if rc == 0:
        return "ok"
    if "AssertionError" in text:
        return "interrupted(loss-assert)"
    return f"failed(rc={rc})"


def rel_to_model(path):
    path = os.path.abspath(path)
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.relpath(path, here) if path.startswith(here + os.sep) else path


def archive_train(a):
    import numpy as np

    log_bytes = open(a.log, "rb").read()
    text = log_bytes.decode(errors="ignore")
    rows = list(PARSERS[a.parser](text))
    os.makedirs(a.dest, exist_ok=True)
    shutil.copyfile(a.log, os.path.join(a.dest, "train.log"))
    if not rows:
        print(f"archive: no loss lines in {a.log}; copied log only", file=sys.stderr)
        return
    steps, losses, costs = (list(c) for c in zip(*rows))
    loss_arr = np.asarray(losses, dtype="float64")
    step_arr = np.asarray(steps, dtype="int64")
    np.savez(os.path.join(a.dest, "losses.npz"), loss=loss_arr, step=step_arr)
    first_ms = costs[0] * 1000.0
    steady_ms = float(np.mean(costs[1:]) * 1000.0) if len(costs) > 1 else first_ms
    report = {
        "model": os.path.basename(os.path.dirname(os.path.abspath(__file__))),
        "mode": a.mode,
        "device": a.device,
        "status": train_status(text, losses, a.rc),
        "steps_logged": len(losses),
        "loss_first": float(loss_arr[0]),
        "loss_last": float(loss_arr[-1]),
        "first_step_ms": round(first_ms, 4),
        "steady_step_ms": round(steady_ms, 4),
        # median is robust to CINN recompiles on new shapes, which inflate the mean
        "median_step_ms": round(float(np.median(costs[1:] or costs) * 1000.0), 4),
        "losses_sha256": hashlib.sha256(
            np.stack([loss_arr, step_arr.astype("float64")]).tobytes()
        ).hexdigest(),
        "log_sha256": hashlib.sha256(log_bytes).hexdigest(),
        "source_log": rel_to_model(a.log),
    }
    with open(os.path.join(a.dest, "train.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"archive: {json.dumps(report)}")


def archive_infer(a):
    os.makedirs(a.dest, exist_ok=True)
    shutil.copyfile(os.path.join(a.src, "inference.json"), os.path.join(a.dest, "inference.json"))
    # The archived inference log keeps the train.log name used by every results/<platform>.
    shutil.copyfile(a.log, os.path.join(a.dest, "train.log"))
    outputs = os.path.join(a.src, "outputs.npz")
    if os.path.isfile(outputs):
        if os.path.getsize(outputs) <= MAX_OUTPUTS_BYTES:
            shutil.copyfile(outputs, os.path.join(a.dest, "outputs.npz"))
        else:
            print(f"archive: {outputs} exceeds {MAX_OUTPUTS_BYTES} bytes, kept local only", file=sys.stderr)
    print(f"archive: {a.src} -> {a.dest}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("device")
    sub.add_parser("platform")
    t = sub.add_parser("train")
    t.add_argument("--parser", choices=sorted(PARSERS), required=True)
    t.add_argument("--mode", choices=["dynamic", "cinn"], required=True)
    t.add_argument("--device", required=True)
    t.add_argument("--rc", type=int, required=True)
    t.add_argument("--log", required=True)
    t.add_argument("--dest", required=True)
    i = sub.add_parser("infer")
    i.add_argument("--src", required=True)
    i.add_argument("--log", required=True)
    i.add_argument("--dest", required=True)
    a = p.parse_args()

    if a.cmd == "device":
        print(device_kind())
    elif a.cmd == "platform":
        print(platform_name())
    elif a.cmd == "train":
        archive_train(a)
    else:
        archive_infer(a)


if __name__ == "__main__":
    main()
