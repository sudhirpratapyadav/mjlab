"""Probe capture-mode / sync-behaviour variants for the CCD graph-capture segfault.

Tests, per variant, whether capturing a mujoco_warp step with a convex geom survives:
  - default   : plain wp.ScopedCapture()  (known to crash)
  - synced    : wp.synchronize() + force_module_load before capture
  - verify    : warp verify_cuda / synchronous mode enabled
  - relaxed   : capture on an explicit non-default stream

Also prints driver/toolkit versions, which is a prime suspect (warp JITs against
CUDA 12.9 while the node driver reports 12.4).

Usage: python probe_capture_modes.py --geom cylinder --mode default
"""

import argparse
import faulthandler
import os
import sys

faulthandler.enable()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from repro_geom_collision import build_xml  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--geom", default="cylinder")
    ap.add_argument("--partner", default="box")
    ap.add_argument("--mode", default="default",
                    choices=["default", "synced", "verify", "stream"])
    ap.add_argument("--nworld", type=int, default=8)
    args = ap.parse_args()

    import warp as wp

    if args.mode == "verify":
        wp.config.verify_cuda = True
        wp.config.verify_fp = True

    import mujoco

    import mujoco_warp as mjw

    wp.init()
    dev = wp.get_device()
    print(f"[modes] geom={args.geom} mode={args.mode} arch={dev.arch}", flush=True)
    try:
        drv = wp.context.runtime.driver_version
        tk = wp.context.runtime.toolkit_version
        print(f"[modes] driver_version={drv} toolkit_version={tk}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[modes] version probe failed: {e}", flush=True)

    mj_model = mujoco.MjModel.from_xml_string(build_xml(args.geom, args.partner))
    mj_data = mujoco.MjData(mj_model)
    mujoco.mj_forward(mj_model, mj_data)
    m = mjw.put_model(mj_model)
    d = mjw.put_data(mj_model, mj_data, nworld=args.nworld)

    if args.mode in ("synced", "verify"):
        mjw.step(m, d)
        wp.synchronize()
        wp.force_load(device=dev)
        wp.synchronize()
        print("[modes] pre-capture sync + force_load done", flush=True)

    print("[modes] capturing...", flush=True)
    if args.mode == "stream":
        stream = wp.Stream(dev)
        with wp.ScopedStream(stream):
            with wp.ScopedCapture(stream=stream) as cap:
                mjw.step(m, d)
        graph = cap.graph
    else:
        with wp.ScopedCapture() as cap:
            mjw.step(m, d)
        graph = cap.graph

    print("[modes] captured; launching...", flush=True)
    for _ in range(10):
        wp.capture_launch(graph)
    wp.synchronize()
    print("[modes] PASS", flush=True)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
