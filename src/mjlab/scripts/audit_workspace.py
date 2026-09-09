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

_OVERLAP_TOLERANCE = 0.001
"""Metres two entities may interpenetrate at reset before it counts as a bug.

Much tighter than ``_FLOOR_TOLERANCE``: that one is loose because ``rbound`` is a
conservative bounding sphere, whereas contact ``dist`` is exact. 1 mm is contact
solver noise; anything deeper is a placement error. The board-in-wall bug this check
was written for was 13.4 mm, which a 0.02 threshold would have waved through."""

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
  ("Mjlab-Throw-To-Bin-Franka", "container"): (
    0.75,
    0.95,
    "beyond the arm's stretch BY DESIGN; a bin inside reach degenerates the task "
    "to place-in-container. Bounded so the ballistic arc stays achievable.",
  ),
}

# Tasks whose objects are approached from the SIDE (a poke, a straddle, a face push),
# never pinched top-down. The top-down grasp-pose fraction is meaningless for them —
# it reported "sparse-reach" on placements that are entirely comfortable for the
# approach they actually use. They are still bound by the radial ceiling.
_SIDE_APPROACH: dict[tuple[str, str], str] = {
  ("Mjlab-Topple-Block-Franka", "block"): "poked on the near face above its CoM",
  ("Mjlab-Cage-Drag-Franka", "cube"): "straddled with open fingers, never pinched",
  ("Mjlab-Pivot-Lift-Franka", "board"): "pinched standing on edge against the wall",
  ("Mjlab-Edge-Grasp-Franka", "plate"): (
    "pushed on its far face where it spawns; the PINCH happens later at the "
    "overhang, which is audited as goal:edge_grasp (radial 0.30)"
  ),
}

# Static fixtures: extrinsic-dexterity furniture the arm works AGAINST but never
# grasps. Scoring them by grasp-pose freedom is meaningless — the question for a
# fixture is only whether the arm can reach the object at it, which is covered by
# that object's own row. They remain bound by the radial ceiling so a fixture cannot
# drift somewhere the arm could never work against it.
_FIXTURES: dict[tuple[str, str], str] = {
  ("Mjlab-Pivot-Lift-Franka", "wall"): "pushed against, never grasped",
  ("Mjlab-Edge-Grasp-Franka", "ledge"): "the plate's support surface, never grasped",
}

# Command goals that are deliberately outside the reach envelope. Same rule as
# _REACH_EXEMPT: exempt from the ceiling, still BOUNDED, and justified in writing.
_GOAL_EXEMPT: dict[tuple[str, str], tuple[float, float, str]] = {
  ("Mjlab-Strike-Slide-Franka", "strike_slide"): (
    0.85,
    1.10,
    "past the arm's ~0.85 m stretch BY DESIGN; a reachable goal degenerates the "
    "task to a quasi-static push",
  ),
  ("Mjlab-Reach-Target-Franka", "reach_target"): (
    0.0,
    0.75,
    "the goal IS the reach probe — this task exists to span the envelope, and it "
    "carries nothing, so the carry ceiling does not apply. ONE-SIDED, unlike "
    "tool-pull and throw-to-bin, whose whole band is deliberately out of reach so a "
    "floor is meaningful for them; reach's band runs from the near envelope "
    "(radial 0.40) out to the far one, so only the ceiling constrains it. The floor "
    "was 0.60, copied from that two-sided pattern; audit_workspace only ever tested "
    "the MAX radial so it never fired, but verify_task tests every sampled goal and "
    "flagged it (2026-09-09, W1-a). CAVEAT: the far end "
    "(0.73) sits at the top-down p90 (0.726), i.e. in the sparse tail this module "
    "otherwise warns against. Left as-is deliberately because Reach has trained "
    "teachers; trimming it would invalidate them. Revisit if those are retrained.",
  ),
  ("Mjlab-Throw-To-Bin-Franka", "place_in_container"): (
    0.75,
    0.95,
    "the bin is beyond reach BY DESIGN; the object must be released ballistically",
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


_APPROACH_OFFSET = 0.08
"""Height of the gripper SITE above an object's centre during a top-down approach.

The metric's old hardcoded ``site_z=0.10`` silently assumed a floor-resting object
(centre z ~0.02). That is wrong for anything raised: the edge-grasp plate sits on a
0.10 m ledge, so its pinch happens with the site near 0.19, and scoring it at 0.10
sampled a slice of the envelope the task never uses — reporting "sparse-reach" on a
placement that is comfortable at the height it is actually grasped.

0.02 + 0.08 = 0.10 reproduces the original default exactly for floor objects."""


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
  elif gtype == mujoco.mjtGeom.mjGEOM_MESH:
    # EXACT, from the compiled mesh vertices. `geom_rbound` (used here before any
    # asset had a mesh) is a bounding SPHERE, so a flat 200x200x10mm plate reports a
    # 0.14m half-height instead of 0.005m. That is not a cosmetic over-estimate: this
    # function feeds `_floor_penetration` AND
    # tests/test_workspace_placement.py::test_mechanism_drops_are_swept_over_the_joint_range,
    # so every mechanism that grows a textured visual mesh would demand a much higher
    # `MECHANISM_DROP_BELOW_MOUNT` and get mounted well above the band its task was
    # tuned for. MuJoCo re-frames mesh vertices (`geom_xpos` sits at the mesh frame,
    # not the authored origin), so project the stored vertices on the world z row.
    mid = int(model.geom_dataid[gid])
    adr = int(model.mesh_vertadr[mid])
    num = int(model.mesh_vertnum[mid])
    verts = np.asarray(model.mesh_vert[adr : adr + num], dtype=float)
    return float(-(verts @ rot[2, :]).min())
  else:  # anything exotic: fall back to the conservative bound
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
    # Read geom positions from the LIVE sim. Mocap bodies are positioned by
    # write_mocap_pose_to_sim, which does NOT show up in qpos — rebuilding an MjData
    # from qpos alone leaves every mechanism at its MJCF authoring pose and reports
    # phantom floor penetration.
    geom_xpos = env.sim.data.geom_xpos[0].cpu().numpy()
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
      low = float(geom_xpos[gid][2] - half_z)
      out[entity] = min(out.get(entity, 0.0), low)
    return out
  finally:
    env.close()


def _entity_overlaps(task_id: str, trials: int = 12) -> dict[str, float]:
  """Deepest object<->object interpenetration at RESET, per geom pair, in metres.

  ``_floor_penetration`` catches an entity buried in the ground and the earlier
  placement audit covered robot<->object, but nothing checked one placed entity
  against another. That gap hid a real bug: Pivot-Lift spawned its board up to
  13.4 mm inside the wall it was supposed to be pushed against, so the episode began
  in contact with no travel available. Fixtures (wall, ledge, container) are written
  per-env by their command AFTER the object spawn, so an overlap here is a config
  error, not a physics transient.

  Negative ``dist`` is penetration. Terrain contacts are skipped — that is
  ``_floor_penetration``'s job, and a resting object legitimately touches the floor.
  """
  import mujoco

  from mjlab.envs import ManagerBasedRlEnv

  cfg = load_env_cfg(task_id, test=True)
  cfg.scene.num_envs = 1
  env = ManagerBasedRlEnv(cfg, device="cpu")
  worst: dict[str, float] = {}
  try:
    model = env.sim.mj_model
    for _ in range(trials):
      env.reset()
      # Mirror the LIVE sim into plain MuJoCo: mocap fixtures are written by
      # write_mocap_pose_to_sim and never appear in qpos, so a data built from qpos
      # alone would test every fixture at its MJCF authoring pose.
      data = mujoco.MjData(model)
      data.qpos[:] = env.sim.data.qpos[0].cpu().numpy()
      data.mocap_pos[:] = env.sim.data.mocap_pos[0].cpu().numpy()
      data.mocap_quat[:] = env.sim.data.mocap_quat[0].cpu().numpy()
      mujoco.mj_forward(model, data)
      for i in range(data.ncon):
        con = data.contact[i]
        n1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, con.geom1) or "?"
        n2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, con.geom2) or "?"
        if "terrain" in n1 or "terrain" in n2:
          continue
        key = f"{n1}<->{n2}"
        worst[key] = min(worst.get(key, 0.0), float(con.dist))
    return worst
  finally:
    env.close()


def _goal_positions(task_id: str, num_envs: int = 64) -> dict[str, np.ndarray]:
  """Reset a task and return each command's sampled goal point, env-local.

  Goals were never audited: only spawned entities were. A goal outside the envelope
  is just as much a task-design bug as an object outside it — the policy is asked to
  bring the object somewhere the arm cannot follow. Every command in this suite
  exposes its goal as ``target_pos`` in WORLD frame, so the env origin comes off here.
  """
  from mjlab.envs import ManagerBasedRlEnv

  cfg = load_env_cfg(task_id, test=True)
  cfg.scene.num_envs = num_envs
  env = ManagerBasedRlEnv(cfg, device="cpu")
  try:
    env.reset()
    # Step once before reading. Not every command populates target_pos in
    # _resample_command: StackingCommand derives it from the base object's live pose
    # in _update_metrics, so straight after reset its target_pos is still all zeros
    # and the audit would report the negated env origins as the "goal".
    env.step(torch.zeros(env.num_envs, env.action_manager.total_action_dim))
    origins = env.scene.env_origins
    out = {}
    for name, term in env.command_manager._terms.items():
      target = getattr(term, "target_pos", None)
      if target is None:
        continue
      out[name] = (target - origins).cpu().numpy()
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
      goals = _goal_positions(task_id, cfg.num_envs)
      overlaps = _entity_overlaps(task_id)
    except Exception as exc:  # noqa: BLE001 — report and continue
      print(f"{task_id:<34} ERROR {type(exc).__name__}: {exc}")
      continue
    for name, p in positions.items():
      x0, x1 = float(p[:, 0].min()), float(p[:, 0].max())
      y0, y1 = float(p[:, 1].min()), float(p[:, 1].max())
      radial = float(np.linalg.norm(p[:, :2], axis=1).max())
      z_mean = float(p[:, 2].mean())
      is_mechanism = name in workspace.MECHANISM_DROP_BELOW_MOUNT
      # Score the envelope at the height this object is actually grasped, not at a
      # fixed floor-grasp height. See _APPROACH_OFFSET.
      free = _grasp_pose_fraction(
        x0, x1, y0, y1, site_z=z_mean + _APPROACH_OFFSET
      )

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
        side = _SIDE_APPROACH.get((task_id, name))
        fixture = _FIXTURES.get((task_id, name))
        # A FIXTURE is never grasped -- it is pushed against or rested on -- so the
        # top-down GRASP ceiling is the wrong bound for it. ``verify_task.g4_g5``
        # already gives _FIXTURES entries GRASP_RADIAL_MAX + 0.05 (it imports this very
        # table to stay in sync); this branch was still applying the grasp ceiling, so
        # the two tools disagreed on the same entity. Matched to verify_task. (W1-c)
        ceiling = workspace.GRASP_RADIAL_MAX + (0.05 if fixture is not None else 0.0)
        if radial > ceiling:
          flags.append(f"radial>{ceiling}")
        if fixture is not None:
          note = f" (fixture: {fixture})"
        elif side is not None:
          note = f" (side-approach: {side})"
        elif free < 3.0:
          flags.append("sparse-reach")
      tag = ",".join(flags) if flags else f"ok{note}"
      if flags:
        problems.append(f"{task_id}/{name}: {tag}")
      print(
        f"{task_id:<34} {name:<12} [{x0:5.2f},{x1:5.2f}] [{y0:5.2f},{y1:5.2f}] "
        f"{radial:7.3f} {free:5.1f}%  {tag}"
      )

    # --- Command goals -----------------------------------------------------------
    for name, g in goals.items():
      gx0, gx1 = float(g[:, 0].min()), float(g[:, 0].max())
      gy0, gy1 = float(g[:, 1].min()), float(g[:, 1].max())
      gradial = float(np.linalg.norm(g[:, :2], axis=1).max())
      gflags = []
      gnote = ""
      gexempt = _GOAL_EXEMPT.get((task_id, name))
      # A mechanism goal is the end of the member's own arc, reached with any
      # orientation and nothing in hand — a looser bound than a carry goal.
      is_mech_goal = any(e in workspace.MECHANISM_DROP_BELOW_MOUNT for e in positions)
      ceiling = (
        workspace.MECHANISM_GOAL_RADIAL_MAX
        if is_mech_goal
        else workspace.GOAL_RADIAL_MAX
      )
      if gexempt is not None:
        lo, hi, why = gexempt
        if not (lo <= gradial <= hi):
          gflags.append(f"exempt-but-outside[{lo},{hi}]")
        gnote = f" (exempt: {why})"
      elif gradial > ceiling:
        gflags.append(f"goal-radial>{ceiling}")
      gtag = ",".join(gflags) if gflags else f"ok{gnote}"
      if gflags:
        problems.append(f"{task_id}/goal:{name}: {gtag}")
      print(
        f"{task_id:<34} {'goal:' + name:<12} [{gx0:5.2f},{gx1:5.2f}] "
        f"[{gy0:5.2f},{gy1:5.2f}] {gradial:7.3f} {'':>5}   {gtag}"
      )

    # --- Object <-> object interpenetration at reset ------------------------------
    for pair, dist in sorted(overlaps.items(), key=lambda kv: kv[1]):
      if dist < -_OVERLAP_TOLERANCE:
        problems.append(f"{task_id}/{pair}: overlap({dist:.4f}m)")
        print(f"{task_id:<34} {pair:<12} spawns interpenetrating: dist={dist:+.4f} m")

  print(f"\n{len(problems)} placement problem(s)")
  for p in problems:
    print(f"  {p}")


if __name__ == "__main__":
  main(tyro.cli(AuditConfig))
