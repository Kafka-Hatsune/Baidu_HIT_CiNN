#!/usr/bin/env python3
"""Summarize eager<->static boundary and host-sync cost from a Paddle trace.

Reads a chrome-tracing JSON produced by run_model.py --profile-output and
reports, normalized per recorded ProfileStep:

  - device kernel time (sum of cat=Kernel)
  - cinn_jit: count + host time + per-event host us   (per fused-subgraph entry/exit)
  - interpreter_core_run: count + host time           (per static-graph entry)
  - Dy2StGradNode: count + host time                  (static->eager backward bridge)
  - host sync: cudaStreamSynchronize + cudaDeviceSynchronize count + time
  - kernel-launch API time (cudaLaunchKernel / Runtime API *)

The "per-event host us" of cinn_jit / interpreter_core_run is the number to
compare across platforms: single-digit us on A100 CUDA vs tens of us on Metax.

Usage:
  python parse_boundary.py <trace.json> [<trace2.json> ...]
"""
import collections
import glob
import json
import re
import sys


def load_events(path):
    with open(path) as fh:
        data = json.load(fh)
    return data.get("traceEvents", data if isinstance(data, list) else [])


def n_steps(events):
    return sum(
        1
        for e in events
        if e.get("ph") == "X"
        and e.get("cat") == "ProfileStep"
        and (e.get("dur") or 0) > 1000  # real recorded step, not the empty closed ones
    )


def base(name):
    return str(name).split("[")[0]


def summarize(path):
    events = load_events(path)
    steps = max(n_steps(events), 1)

    def acc(pred):
        c = 0
        dur = 0.0
        for e in events:
            if e.get("ph") != "X":
                continue
            if pred(e):
                c += 1
                dur += e.get("dur", 0) or 0
        return c, dur  # dur in us

    kern_c, kern = acc(lambda e: e.get("cat") == "Kernel")
    jit_c, jit = acc(lambda e: base(e.get("name")) == "cinn_jit")
    ic_c, ic = acc(lambda e: base(e.get("name")) == "interpreter_core_run")
    dy_c, dy = acc(lambda e: base(e.get("name")) == "Global_Dy2StGradNode")
    sync_c, sync = acc(
        lambda e: base(e.get("name")).startswith("cudaStreamSynchronize")
        or base(e.get("name")).startswith("cudaDeviceSynchronize")
    )
    launch_c, launch = acc(
        lambda e: e.get("cat") == "CudaRuntime"
        and (
            base(e.get("name")).startswith("cudaLaunchKernel")
            or base(e.get("name")).startswith("Runtime API")
            or base(e.get("name")).startswith("metax Runtime API")
        )
    )

    def per_step(x):
        return x / steps

    def per_evt_us(dur, c):
        return (dur / c) if c else 0.0

    return {
        "trace": path,
        "recorded_steps": steps,
        "kernel_ms_per_step": per_step(kern) / 1000,
        "kernel_count_per_step": per_step(kern_c),
        "cinn_jit_count_per_step": per_step(jit_c),
        "cinn_jit_host_us_per_step": per_step(jit),
        "cinn_jit_host_us_per_event": per_evt_us(jit, jit_c),
        "interp_count_per_step": per_step(ic_c),
        "interp_host_us_per_event": per_evt_us(ic, ic_c),
        "dy2st_count_per_step": per_step(dy_c),
        "dy2st_host_us_per_event": per_evt_us(dy, dy_c),
        "sync_count_per_step": per_step(sync_c),
        "sync_us_per_step": per_step(sync),
        "launch_api_us_per_step": per_step(launch),
    }


def main():
    paths = []
    for a in sys.argv[1:]:
        paths.extend(sorted(glob.glob(a)))
    if not paths:
        print("usage: parse_boundary.py <trace.json> ...", file=sys.stderr)
        sys.exit(2)
    rows = [summarize(p) for p in paths]
    keys = [
        "recorded_steps",
        "kernel_ms_per_step",
        "cinn_jit_count_per_step",
        "cinn_jit_host_us_per_event",
        "interp_count_per_step",
        "interp_host_us_per_event",
        "dy2st_count_per_step",
        "dy2st_host_us_per_event",
        "sync_count_per_step",
        "sync_us_per_step",
        "launch_api_us_per_step",
    ]
    for r in rows:
        print("\n==", r["trace"])
        for k in keys:
            v = r[k]
            print(f"  {k:32s} {v:.3f}" if isinstance(v, float) else f"  {k:32s} {v}")
    with open("boundary_summary.json", "w") as fh:
        json.dump(rows, fh, indent=2)
    print("\nwrote boundary_summary.json")


if __name__ == "__main__":
    main()

