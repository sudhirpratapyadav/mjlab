"""Probe: does WARMING UP (stepping once eagerly) before graph capture fix the crash?

Hypothesis: the segfault is not a bad cylinder/mesh kernel, but module load / JIT
compilation happening *inside* the CUDA-graph capture. mujoco_warp builds CCD kernels
lazily per geom-pair (`ccd_kernel_builder`), so the first time a convex pair is seen
is inside `wp.ScopedCapture()` -> illegal work during capture -> SIGSEGV.

If this hypothesis holds, a single eager `mjw.step()` before capture (which forces the
module to load) makes the exact same capture succeed.

Usage: python probe_warmup.py --geom cylinder [--warmup/--no-warmup]
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
    ap.add_argument("--nworld", type=int, default=8)
    ap.add_argument("--warmup", dest="warmup", action="store_true", default=True)
    ap.add_argument("--no-warmup", dest="warmup", action="store_false")
    args = ap.parse_args()

    import mujoco
    import warp as wp

    import mujoco_warp as mjw

    wp.init()
    print(f"[warmup-probe] geom={args.geom} warmup={args.warmup} "
          f"arch={wp.get_device().arch}", flush=True)

    mj_model = mujoco.MjModel.from_xml_string(build_xml(args.geom, args.partner))
    mj_data = mujoco.MjData(mj_model)
    mujoco.mj_forward(mj_model, mj_data)

    m = mjw.put_model(mj_model)
    d = mjw.put_data(mj_model, mj_data, nworld=args.nworld)

    if args.warmup:
        print("[warmup-probe] eager warmup step (forces module load)...", flush=True)
        mjw.step(m, d)
        wp.synchronize()
        print("[warmup-probe] warmup done", flush=True)

    print("[warmup-probe] capturing graph...", flush=True)
    with wp.ScopedCapture() as cap:
        mjw.step(m, d)
    print("[warmup-probe] captured; launching...", flush=True)
    for _ in range(10):
        wp.capture_launch(cap.graph)
    wp.synchronize()
    print("[warmup-probe] PASS", flush=True)

    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
