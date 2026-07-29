"""Isolated sm_80 collision-kernel repro — NO mjlab dependency.

Builds a minimal two-body MJCF (one static floor + one free object of a given geom
type), puts it through the exact mujoco-warp path mjlab uses (put_model/put_data →
step under CUDA-graph capture), and reports pass/segfault.

One geom type per PROCESS (a segfault kills the process), so the driver
`run_matrix.py` re-invokes this file once per case as a subprocess.

Usage:
    python repro_geom_collision.py --geom cylinder [--nstep 20] [--nworld 8]
                                   [--no-graph] [--pair cylinder:box]
Exit codes: 0 = ok, non-zero (139 = SIGSEGV) = crash.
"""

import argparse
import ctypes
import faulthandler
import os
import sys

faulthandler.enable()

# Geom snippets: (attrs for the *free* object, attrs for the *static* partner).
# The partner defaults to a box "table" — matching the mjlab scenes.
GEOMS = {
    "box": 'type="box" size="0.02 0.02 0.02"',
    "sphere": 'type="sphere" size="0.02"',
    "capsule": 'type="capsule" size="0.02 0.02"',
    "cylinder": 'type="cylinder" size="0.02 0.02"',
    "ellipsoid": 'type="ellipsoid" size="0.02 0.03 0.015"',
    "disc": 'type="cylinder" size="0.03 0.005"',  # short cylinder = the mjlab disc
    "mesh": 'type="mesh" mesh="tetra"',
}

MESH_ASSET = """
  <asset>
    <mesh name="tetra" vertex="0 0 0  0.04 0 0  0 0.04 0  0 0 0.04"/>
  </asset>
"""


def build_xml(free_geom: str, static_geom: str) -> str:
    needs_mesh = "mesh" in (free_geom, static_geom)
    return f"""
<mujoco model="sm80_repro">
  <option timestep="0.005"/>
  {MESH_ASSET if needs_mesh else ""}
  <worldbody>
    <body name="ground">
      <geom name="static" {GEOMS[static_geom]} pos="0 0 0"/>
    </body>
    <body name="obj" pos="0 0 0.05">
      <freejoint/>
      <geom name="free" {GEOMS[free_geom]} mass="0.1"/>
    </body>
  </worldbody>
</mujoco>
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--geom", default="cylinder", help="free-object geom type")
    ap.add_argument("--partner", default="box", help="static partner geom type")
    ap.add_argument("--nstep", type=int, default=20)
    ap.add_argument("--nworld", type=int, default=8)
    ap.add_argument(
        "--no-graph",
        action="store_true",
        help="skip CUDA-graph capture (step eagerly) to isolate capture vs kernel",
    )
    ap.add_argument("--xml-only", action="store_true", help="print XML and exit")
    args = ap.parse_args()

    xml = build_xml(args.geom, args.partner)
    if args.xml_only:
        print(xml)
        return 0

    import mujoco
    import warp as wp

    import mujoco_warp as mjw

    print(f"[repro] geom={args.geom} partner={args.partner} "
          f"nworld={args.nworld} graph={not args.no_graph}", flush=True)
    print(f"[repro] warp={wp.config.version} mujoco={mujoco.__version__}", flush=True)

    wp.init()
    dev = wp.get_device()
    print(f"[repro] device={dev} arch={getattr(dev, 'arch', '?')}", flush=True)

    mj_model = mujoco.MjModel.from_xml_string(xml)
    mj_data = mujoco.MjData(mj_model)
    mujoco.mj_forward(mj_model, mj_data)

    gtypes = [int(t) for t in mj_model.geom_type]
    print(f"[repro] geom_type ids={gtypes} (mjGEOM: 2=sphere 3=capsule 5=cylinder "
          f"4=ellipsoid 6=box 7=mesh)", flush=True)

    print("[repro] put_model...", flush=True)
    m = mjw.put_model(mj_model)
    print("[repro] put_data...", flush=True)
    d = mjw.put_data(mj_model, mj_data, nworld=args.nworld)

    if args.no_graph:
        print("[repro] stepping eagerly (no graph capture)...", flush=True)
        for i in range(args.nstep):
            mjw.step(m, d)
            wp.synchronize()
            print(f"[repro]   step {i} ok", flush=True)
    else:
        print("[repro] capturing CUDA graph...", flush=True)
        with wp.ScopedCapture() as capture:
            mjw.step(m, d)
        graph = capture.graph
        print("[repro] graph captured; launching...", flush=True)
        for i in range(args.nstep):
            wp.capture_launch(graph)
        wp.synchronize()
        print(f"[repro] {args.nstep} graph launches ok", flush=True)

    # sanity: contacts actually generated (otherwise the kernel was never exercised)
    ncon = int(d.ncon.numpy()[0]) if hasattr(d, "ncon") else -1
    print(f"[repro] ncon={ncon}", flush=True)
    print("[repro] PASS", flush=True)

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)  # skip crashing warp/torch teardown


if __name__ == "__main__":
    raise SystemExit(main())
