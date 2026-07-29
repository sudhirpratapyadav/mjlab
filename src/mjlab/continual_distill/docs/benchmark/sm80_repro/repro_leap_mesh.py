"""Repro against the REAL LEAP hand asset (mesh colliders re-enabled).

The synthetic matrix proves the sm_80 graph-capture bug on toy geoms. This validates
the same conclusion on the actual asset that forced the mesh->box workaround: it loads
the vendored LEAP XML, RE-ENABLES collision on the mesh geoms that
leap_constants.py::get_spec currently disables, and graph-captures a step.

  old warp (1.11.0.dev20251124) -> expected SEGFAULT
  new warp (>=1.15)             -> expected PASS

Usage: python repro_leap_mesh.py [--xml <path>] [--no-graph]
"""

import argparse
import faulthandler
import glob
import os
import sys

faulthandler.enable()

DEFAULT_XML_GLOBS = [
    os.path.expanduser("~/sudhir/mjlab/src/mjlab/asset_zoo/robots/leap_hand/**/*.xml"),
]


def find_leap_xml() -> str | None:
    for pat in DEFAULT_XML_GLOBS:
        for p in sorted(glob.glob(pat, recursive=True)):
            base = os.path.basename(p).lower()
            if "leap" in base and "scene" not in base:
                return p
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", default=None)
    ap.add_argument("--no-graph", action="store_true")
    ap.add_argument("--nworld", type=int, default=4)
    args = ap.parse_args()

    xml = args.xml or find_leap_xml()
    if not xml or not os.path.exists(xml):
        print(f"[leap] ERROR: LEAP xml not found (looked: {DEFAULT_XML_GLOBS})")
        return 2
    print(f"[leap] xml={xml}", flush=True)

    import mujoco
    import warp as wp

    import mujoco_warp as mjw

    wp.init()
    print(f"[leap] warp={wp.config.version} arch={wp.get_device().arch}", flush=True)

    spec = mujoco.MjSpec.from_file(xml)

    # RE-ENABLE mesh collision (undo the workaround) so we exercise the real path.
    n_mesh = 0
    for g in spec.geoms:
        if g.type == mujoco.mjtGeom.mjGEOM_MESH:
            g.contype, g.conaffinity = 1, 1
            n_mesh += 1
    print(f"[leap] re-enabled collision on {n_mesh} mesh geoms", flush=True)

    mj_model = spec.compile()
    mj_data = mujoco.MjData(mj_model)
    mujoco.mj_forward(mj_model, mj_data)

    collidable_mesh = sum(
        1
        for i in range(mj_model.ngeom)
        if mj_model.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH
        and (mj_model.geom_contype[i] or mj_model.geom_conaffinity[i])
    )
    print(f"[leap] ngeom={mj_model.ngeom} collidable_mesh_geoms={collidable_mesh}",
          flush=True)

    m = mjw.put_model(mj_model)
    d = mjw.put_data(mj_model, mj_data, nworld=args.nworld)

    if args.no_graph:
        print("[leap] stepping eagerly...", flush=True)
        for _ in range(5):
            mjw.step(m, d)
        wp.synchronize()
    else:
        print("[leap] capturing CUDA graph (crash point on old warp)...", flush=True)
        with wp.ScopedCapture() as cap:
            mjw.step(m, d)
        print("[leap] captured; launching...", flush=True)
        for _ in range(10):
            wp.capture_launch(cap.graph)
        wp.synchronize()

    print("[leap] PASS", flush=True)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
