"""Measure the Class A reachable workspace, and audit every task's spawn ranges against it.

Two jobs:

1. ``--measure``: forward-kinematics sampling over the Franka's joint limits to derive
   the reachable envelope. This is where the numbers in
   ``tasks/manipulation/workspace.py`` come from; re-run it if the robot model changes.

2. default: build every Class A task and report where its objects actually spawn, with
   the fraction of comfortable top-down grasp poses available at that location. That
   fraction is the honest measure of whether a placement is sane — a spawn box in the
   reach tail is only enterable near singularities, which constrains grasp orientation
   and fights the joint-limit penalty.

Usage:
    python -m mjlab.scripts.audit_workspace --measure
    python -m mjlab.scripts.audit_workspace
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import numpy as np
import torch
import tyro

import mjlab  # noqa: F401  (import side-effect: registers task packages)
from mjlab.tasks.manipulation import benchmark, workspace
from mjlab.tasks.manipulation.taxonomy import Embodiment
from mjlab.tasks.registry import load_env_cfg, load_taxonomy

_TOPDOWN_COS = -0.85
"""EE z-axis vs world +z. -0.85 is within ~32 deg of straight down."""

_FLOOR_TOLERANCE = 0.02
"""Metres a geom may dip below z=0 before it counts as buried (rbound is a conservative
bounding sphere, so a little slack avoids false positives on rounded geoms)."""

# Entities that are DELIBERATELY outside the direct-grasp envelope, with the bound that
# does apply instead. Tool-pull's whole premise is that the puck cannot be reached by
# hand — it must be dragged in with the stick — so holding it to GRASP_RADIAL_MAX would
# delete the task. It is still bounded: it must sit inside the stick's effective
# extension, or the task becomes unsolvable in the other direction.
_REACH_EXEMPT: dict[tuple[str, str], tuple[float, float, str]] = {
  ("Mjlab-Tool-Pull-Franka", "puck"): (
    0.58,
    0.75,
    "out of direct reach by design; must stay inside the stick's extension",
  ),
}


@functools.lru_cache(maxsize=1)
def _sample_workspace(n: int = 400_000, joint_frac: float = 0.90):
  """FK-sample the arm; return (positions, downness) for the gripper site."""
  import mujoco

  from mjlab.asset_zoo.robots.franka_emika_panda.franka_constants import FRANKA_XML

  model = mujoco.MjModel.from_xml_path(str(FRANKA_XML))
  data = mujoco.MjData(model)
  site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripper")

  lo, hi = model.jnt_range[:7, 0], model.jnt_range[:7, 1]
  mid, half = (lo + hi) / 2, (hi - lo) / 2 * joint_frac
  rng = np.random.default_rng(0)
  q = rng.uniform(mid - half, mid + half, size=(n, 7))

  pos = np.empty((n, 3))
  down = np.empty(n)
  for i in range(n):
    data.qpos[:7] = q[i]
    mujoco.mj_kinematics(model, data)
    pos[i] = data.site_xpos[site_id]
    down[i] = data.site_xmat[site_id].reshape(3, 3)[2, 2]
  return pos, down


def _grasp_pose_fraction(x0, x1, y0, y1, site_z=0.10, tol=0.04) -> float:
  """Percent of top-down poses at grasp height that land inside this box."""
  pos, down = _sample_workspace()
  at_height = (down < _TOPDOWN_COS) & (np.abs(pos[:, 2] - site_z) < tol)
  g = pos[at_height]
  if len(g) == 0:
    return 0.0
  inside = (g[:, 0] >= x0) & (g[:, 0] <= x1) & (g[:, 1] >= y0) & (g[:, 1] <= y1)
  return float(inside.mean() * 100)


def measure() -> None:
  pos, down = _sample_workspace()
  print("Class A (Franka) reachable workspace — gripper site, 90% joint range\n")
  print("TOP-DOWN radial reach envelope by site height:")
  print(f"  {'z band':<24} {'p5':>7} {'p50':>7} {'p90':>7} {'p95':>7}")
  for z0, z1, label in [
    (0.06, 0.14, "grasp (floor object)"),
    (0.14, 0.25, "low carry"),
    (0.25, 0.40, "mid carry"),
    (0.40, 0.60, "mechanism height"),
  ]:
    sel = (down < _TOPDOWN_COS) & (pos[:, 2] >= z0) & (pos[:, 2] < z1)
    if sel.sum() < 50:
      continue
    r = np.linalg.norm(pos[sel][:, :2], axis=1)
    p = np.percentile(r, [5, 50, 90, 95])
    print(f"  {label:<24} {p[0]:7.3f} {p[1]:7.3f} {p[2]:7.3f} {p[3]:7.3f}")

  sel = (pos[:, 2] >= 0.45) & (pos[:, 2] < 0.60) & (pos[:, 0] > 0)
  r = np.linalg.norm(pos[sel][:, :2], axis=1)
  p = np.percentile(r, [5, 50, 90, 95])
  print("\nANY-orientation radial reach at mechanism height (z 0.45-0.60):")
  print(f"  p5={p[0]:.3f} p50={p[1]:.3f} p90={p[2]:.3f} p95={p[3]:.3f}")

  print("\nDensity of comfortable top-down grasp poses (higher = more approach freedom):")
  for label, box in [
    ("workspace.GRASP_*", (*workspace.GRASP_X_RANGE, *workspace.GRASP_Y_RANGE)),
    ("old lift/stack", (0.60, 0.80, -0.15, 0.15)),
    ("old tool puck", (0.78, 0.88, -0.10, 0.10)),
  ]:
    print(f"  {label:<22} {_grasp_pose_fraction(*box):5.1f}%")


def _geom_half_height(model, data, gid: int) -> float:
  """Vertical half-extent of a geom in WORLD frame, accounting for its rotation."""
  import mujoco

  gtype = model.geom_type[gid]
  size = model.geom_size[gid]
  if gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
    return float(size[0])
  if gtype == mujoco.mjtGeom.mjGEOM_PLANE:
    return 0.0
  rot = data.geom_xmat[gid].reshape(3, 3)
  if gtype == mujoco.mjtGeom.mjGEOM_BOX:
    half = size[:3]
  elif gtype in (mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_CAPSULE):
    half = np.array([size[0], size[0], size[1] + (size[0] if gtype ==
                     mujoco.mjtGeom.mjGEOM_CAPSULE else 0.0)])
  elif gtype == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
    half = size[:3]
  else:  # mesh and anything exotic: fall back to the conservative bound
    return float(model.geom_rbound[gid])
  # Projection of the box's half-extents onto world z.
  return float(np.abs(rot[2, :]) @ half)


def _floor_penetration(task_id: str) -> dict[str, float]:
  """How far each entity's lowest geom sits BELOW the ground plane, in metres.

  Radial distance from the base says nothing about vertical placement, so a mechanism
  can be perfectly in reach while half of it is buried in the floor. Articulated
  mechanisms hang DOWNWARD from a mocap mount (the door panel drops 0.94m below its
  mount), and Class A scenes have no table or wall for them to hang from, so the mount
  height has to account for the asset's own downward extent.
  """
  import mujoco

  from mjlab.envs import ManagerBasedRlEnv

  cfg = load_env_cfg(task_id, test=True)
  cfg.scene.num_envs = 1
  env = ManagerBasedRlEnv(cfg, device="cpu")
  try:
    env.reset()
    model = env.sim.mj_model
    data = mujoco.MjData(model)
    data.qpos[:] = env.sim.data.qpos[0].cpu().numpy()
    mujoco.mj_forward(model, data)
    out: dict[str, float] = {}
    for gid in range(model.ngeom):
      name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
      if "/" not in name:
        continue
      entity = name.split("/", 1)[0]
      if entity in ("robot", "terrain"):
        continue
      # Use the geom's true vertical half-extent, not geom_rbound: rbound is a
      # bounding SPHERE, which massively over-reports for flat/elongated geoms (a
      # 4cm cube resting correctly on the floor would look 2cm buried).
      half_z = _geom_half_height(model, data, gid)
      low = float(data.geom_xpos[gid][2] - half_z)
      out[entity] = min(out.get(entity, 0.0), low)
    return out
  finally:
    env.close()


def _object_positions(task_id: str, num_envs: int = 64) -> dict[str, np.ndarray]:
  """Reset a task and return each free/articulated entity's position, env-local."""
  from mjlab.envs import ManagerBasedRlEnv

  cfg = load_env_cfg(task_id, test=True)
  cfg.scene.num_envs = num_envs
  env = ManagerBasedRlEnv(cfg, device="cpu")
  try:
    env.reset()
    origins = env.scene.env_origins
    out = {}
    for name in cfg.scene.entities:
      if name in ("robot", "mocap_goal"):
        continue
      entity = env.scene[name]
      if "object_site" in entity.site_names:
        idx = entity.site_names.index("object_site")
        p = entity.data.site_pos_w[:, idx]
      else:
        p = entity.data.root_link_pos_w
      out[name] = (p - origins).cpu().numpy()
    return out
  finally:
    env.close()


@dataclass(frozen=True)
class AuditConfig:
  measure: bool = False
  """Print the FK-measured reach envelope instead of auditing tasks."""
  num_envs: int = 64
  keyword: str | None = None


def main(cfg: AuditConfig) -> None:
  if cfg.measure:
    measure()
    return

  tagged = benchmark.all_benchmark_tasks()
  task_ids = sorted(
    t
    for t, tax in tagged.items()
    if tax.embodiment is Embodiment.ARM_GRIPPER
    and (not cfg.keyword or cfg.keyword.lower() in t.lower())
  )

  print(f"Auditing {len(task_ids)} Class A tasks against workspace.py\n")
  print(f"{'task':<34} {'entity':<12} {'x':>13} {'y':>13} {'radial':>7} {'free%':>6}  flags")
  problems: list[str] = []
  for task_id in task_ids:
    try:
      positions = _object_positions(task_id, cfg.num_envs)
      buried = _floor_penetration(task_id)
    except Exception as exc:  # noqa: BLE001 — report and continue
      print(f"{task_id:<34} ERROR {type(exc).__name__}: {exc}")
      continue
    for name, p in positions.items():
      x0, x1 = float(p[:, 0].min()), float(p[:, 0].max())
      y0, y1 = float(p[:, 1].min()), float(p[:, 1].max())
      radial = float(np.linalg.norm(p[:, :2], axis=1).max())
      z_mean = float(p[:, 2].mean())
      is_mechanism = z_mean > 0.25
      free = _grasp_pose_fraction(x0, x1, y0, y1)

      flags = []
      sunk = buried.get(name, 0.0)
      if sunk < -_FLOOR_TOLERANCE:
        flags.append(f"below-floor({sunk:.2f}m)")
      exempt = _REACH_EXEMPT.get((task_id, name))
      note = ""
      if exempt is not None:
        lo, hi, why = exempt
        if not (lo <= radial <= hi):
          flags.append(f"exempt-but-outside[{lo},{hi}]")
        note = f" (exempt: {why})"
      elif is_mechanism:
        if radial > workspace.MECHANISM_HANDLE_RADIAL_MAX:
          flags.append(f"radial>{workspace.MECHANISM_HANDLE_RADIAL_MAX}")
      else:
        if radial > workspace.GRASP_RADIAL_MAX:
          flags.append(f"radial>{workspace.GRASP_RADIAL_MAX}")
        if free < 3.0:
          flags.append("sparse-reach")
      tag = ",".join(flags) if flags else f"ok{note}"
      if flags:
        problems.append(f"{task_id}/{name}: {tag}")
      print(
        f"{task_id:<34} {name:<12} [{x0:5.2f},{x1:5.2f}] [{y0:5.2f},{y1:5.2f}] "
        f"{radial:7.3f} {free:5.1f}%  {tag}"
      )

  print(f"\n{len(problems)} placement problem(s)")
  for p in problems:
    print(f"  {p}")


if __name__ == "__main__":
  main(tyro.cli(AuditConfig))
