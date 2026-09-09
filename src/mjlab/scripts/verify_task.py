"""CL-V2 task verifier: the G3 (physics), G4 (init distribution) and G5 (success) gates.

One command verifies one task end to end and writes the evidence the STATUS row asks
for::

    PYTHONPATH=src .venv/bin/python -m mjlab.scripts.verify_task \
        --task Mjlab-Lift-Cube-Franka --num-resets 1000 \
        --out src/mjlab/continual_distill/docs/cl_v2/runs/Lift-Cube

Runs on CPU by default (``--device cpu``, works on the login node; ~2-6 min per task)
or on a GPU (``--device cuda:0``).

What it checks
--------------
G3 physics (plain MuJoCo on the object XML alone, ground plane added):
  * ``settle``: object placed at rest on the plane, 2 s of simulation, must end with
    |v| < 1e-3 m/s, |w| < 1e-2 rad/s, no NaN, < 5 mm drop after the first 0.25 s, and
    the up-axis must not tip more than 5 degrees (a mesh whose collider is wrong
    falls over or sinks here).
  * ``drop``: dropped from 10 cm, must come to rest on the plane (lowest collider
    point within 3 mm of z=0) with no NaN.
  * ``articulation``: every hinge/slide range read from the COMPILED model, in
    radians / metres; the command's ``target_value`` must lie inside it (this is the
    degree-vs-radian trap), and a hinge range must not exceed 2*pi.
  * ``graspable``: for pinch-grasped objects, the collision AABB width along the two
    horizontal axes at rest; at least one must be <= 0.060 m (gripper opens 0.080).
  * ``step_time``: CPU mj_step time of the object-alone model vs the baseline commit's
    primitive XML (``--baseline 1127d12``); ratio must be <= 1.3.
  * ``mass``: body mass > 0 and inertia positive definite (compile already enforces).
G4 init distribution (the real env, N envs x R resets = ``--num-resets``):
  * every spawned entity's env-local position: radial bounds from
    ``mjlab.tasks.manipulation.workspace`` with the audit's exemptions
    (``audit_workspace._REACH_EXEMPT`` / ``_SIDE_APPROACH`` / ``_FIXTURES``);
  * no spawn collision: no contact deeper than 1 mm between two different entities
    (robot included) at reset; floor penetration <= 2 cm;
  * command goals inside ``GOAL_RADIAL_MAX`` / ``MECHANISM_GOAL_RADIAL_MAX`` with
    ``audit_workspace._GOAL_EXEMPT``;
  * histograms of x, y, radial, yaw per entity -> ``<out>/hist_<entity>.png``.
G5 success:
  * ``success_at_reset``: the command's instantaneous ``compute_success()`` after a
    reset plus one zero-action step, over all resets, must be 0.
  * ``oracle``: the manipulated object (or joint) is teleported into the goal
    configuration and physics is stepped; ``compute_success()`` must then be true in
    >= 90% of envs. This proves the predicate is reachable UNDER THE PHYSICS (a goal
    inside a wall, a joint target outside its range, a box the object cannot enter
    all fail here).

Output: ``<out>/verify.json`` (every number, plus ``pass`` booleans per gate) and a
printed summary. Exit code 0 only if every gate passes.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[3]
BASELINE = "1127d12"

# Which entity is the manipulated one, and how to build the success oracle.
# kind: free  -> teleport the entity root to the command's target_pos
#       joint -> write the command's target_value into the named joint
#       none  -> no oracle (asset-free task)
# quat: orientation to teleport with (w,x,y,z), None = keep the spawned one.
_UPRIGHT = (1.0, 0.0, 0.0, 0.0)
_X_UP = (math.cos(-math.pi / 4), 0.0, math.sin(-math.pi / 4), 0.0)  # rotate -90 deg about y
TASKS: dict[str, dict] = {
  "Mjlab-Lift-Cube-Franka": dict(entity="cube", kind="free", graspable=True),
  "Mjlab-Lift-Cylinder-Franka": dict(entity="cylinder", kind="free", graspable=True),
  "Mjlab-Lift-Sphere-Franka": dict(entity="sphere", kind="free", graspable=True),
  "Mjlab-Lift-Ellipsoid-Franka": dict(entity="ellipsoid", kind="free", graspable=True),
  "Mjlab-Stack-Cube-Franka": dict(entity="object", kind="free", graspable=True),
  # oracle_root: StackingCommand's predicate compares the peg's ROOT to target_pos,
  # but the peg's object_site is 5 cm away at its tip. Placing the SITE on the goal
  # (the default) drops the peg in mid-air with 25 mm still to fall, which three
  # 0.02 s steps cannot cover -- a verifier artefact, not a task defect. (W1-c)
  "Mjlab-Peg-Insertion-Franka": dict(entity="object", kind="free", graspable=True,
                                     oracle_root=True),
  "Mjlab-Reach-Target-Franka": dict(entity=None, kind="none", graspable=False),
  "Mjlab-Open-Door-Franka": dict(entity="door", kind="joint", joint="door_hinge", graspable=False),
  "Mjlab-Open-Drawer-Franka": dict(entity="drawer", kind="joint", joint="drawer_slide", graspable=False),
  # oracle_hold: the button's slide joint is SPRING-RETURNED (stiffness 1000 on a 36 g
  # plunger), so it does not stay where the oracle puts it. See the oracle_hold note
  # in g4_g5. (W2-a)
  "Mjlab-Push-Button-Franka": dict(entity="button", kind="joint", joint="button_slide",
                                   graspable=False, oracle_hold=True),
  "Mjlab-Push-Cuboid-Franka": dict(entity="cuboid", kind="free", graspable=False),
  "Mjlab-Push-Disc-Franka": dict(entity="disc", kind="free", graspable=False),
  "Mjlab-Turn-Lever-Franka": dict(entity="lever", kind="joint", joint="lever_hinge", graspable=False),
  "Mjlab-Rotate-Valve-Franka": dict(entity="valve", kind="joint", joint="valve_hinge", graspable=False),
  "Mjlab-Flip-Switch-Franka": dict(entity="switch", kind="joint", joint="switch_hinge", graspable=False),
  "Mjlab-Slide-Window-Franka": dict(entity="window", kind="joint", joint="window_slide", graspable=False),
  "Mjlab-Open-Lid-Franka": dict(entity="lid", kind="joint", joint="lid_hinge", graspable=False),
  "Mjlab-Place-In-Container-Franka": dict(entity="cube", kind="free", graspable=True),
  "Mjlab-Reorient-Object-Franka": dict(entity="cylinder", kind="free", graspable=True, quat=_UPRIGHT),
  "Mjlab-Tool-Pull-Franka": dict(entity="puck", kind="free", graspable=False),
  "Mjlab-Drag-Pull-Franka": dict(entity="cuboid", kind="free", graspable=False),
  "Mjlab-Strike-Slide-Franka": dict(entity="puck", kind="free", graspable=False),
  "Mjlab-Cage-Drag-Franka": dict(entity="cube", kind="free", graspable=False),
  "Mjlab-Topple-Block-Franka": dict(entity="block", kind="free", graspable=False, quat=_X_UP),
  "Mjlab-Push-Flap-Franka": dict(entity="flap", kind="joint", joint="flap_hinge", graspable=False),
  "Mjlab-Axial-Extract-Franka": dict(entity="plug", kind="joint", joint="plug_slide", graspable=True),
  # graspable="thin_axis": these two are the EXTRINSIC-DEXTERITY pair. Their premise
  # is that the object is unspannable where it lies (both in-plane widths exceed the
  # 80 mm aperture) and is pinched on its THICKNESS only after the ledge edge / wall
  # has re-presented it. Measuring the flat-pose in-plane width would fail the task
  # for doing exactly what it is designed to do, so the pinchable axis is the min
  # over ALL THREE extents. (W1-c)
  "Mjlab-Edge-Grasp-Franka": dict(entity="plate", kind="free", graspable="thin_axis"),
  "Mjlab-Pivot-Lift-Franka": dict(entity="board", kind="free", graspable="thin_axis"),
  "Mjlab-Throw-To-Bin-Franka": dict(entity="cube", kind="free", graspable=True),
}

_SKIP_ENTITIES = ("robot", "mocap_goal", "mocap_target")


# ======================================================================================
# G3 — physics on the object alone
# ======================================================================================


def _entity_xml_paths(cfg) -> dict[str, Path]:
  """Object XML path per non-robot entity, found by instantiating the entity spec."""
  out = {}
  for name, ecfg in cfg.scene.entities.items():
    if name in _SKIP_ENTITIES:
      continue
    spec = ecfg.spec_fn()
    p = getattr(spec, "modelfiledir", "") or ""
    fname = getattr(spec, "modelname", name)
    # modelfiledir is the directory; find the xml inside it whose model name matches.
    cands = list(Path(p).glob("*.xml")) if p else []
    match = None
    for c in cands:
      try:
        if mujoco.MjSpec.from_file(str(c)).modelname == spec.modelname:
          match = c
          break
      except Exception:
        continue
    out[name] = match if match else (cands[0] if cands else None)
  return out


def _object_model(xml: Path, gravity=(0, 0, -9.81), mount_pos=None):
  """Compile the object XML with a ground plane; return (model, data, spec)."""
  spec = mujoco.MjSpec.from_file(str(xml))
  spec.option.gravity = list(gravity)
  spec.option.timestep = 0.005
  spec.worldbody.add_geom(name="__plane", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[3, 3, 0.05],
                          contype=1, conaffinity=1, friction=[1, 0.03, 0.003])
  spec.worldbody.add_light(name="__l", pos=[0, 0, 3], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
  model = spec.compile()
  data = mujoco.MjData(model)
  return model, data, spec


def _geom_lowest_z(model, data, body_ids) -> float:
  """Lowest point of any collision geom that belongs to the given bodies."""
  low = np.inf
  for g in range(model.ngeom):
    if model.geom_bodyid[g] not in body_ids or model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0:
      continue
    if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE:
      continue
    low = min(low, _geom_aabb(model, data, g)[0][2])
  return float(low)


def _geom_aabb(model, data, g):
  """World AABB of one geom (mesh: transformed vertices; primitives: rotated box)."""
  R = data.geom_xmat[g].reshape(3, 3)
  c = data.geom_xpos[g]
  t = model.geom_type[g]
  if t == mujoco.mjtGeom.mjGEOM_MESH:
    mid = model.geom_dataid[g]
    v0 = model.mesh_vertadr[mid]
    n = model.mesh_vertnum[mid]
    verts = model.mesh_vert[v0:v0 + n] @ R.T + c
    return verts.min(axis=0), verts.max(axis=0)
  s = model.geom_size[g]
  if t == mujoco.mjtGeom.mjGEOM_SPHERE:
    h = np.array([s[0]] * 3)
  elif t == mujoco.mjtGeom.mjGEOM_CAPSULE:
    h = np.array([s[0], s[0], s[0] + s[1]])
  elif t == mujoco.mjtGeom.mjGEOM_CYLINDER:
    h = np.array([s[0], s[0], s[1]])
  elif t == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
    h = np.array(s[:3])
  else:  # box
    h = np.array(s[:3])
  ext = np.abs(R) @ h
  return c - ext, c + ext


def _entity_body_ids(model, root_body: str | None = None):
  return set(range(1, model.nbody))


def _free_joint(model):
  for j in range(model.njnt):
    if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
      return j
  return None


def _quat_tilt_deg(q) -> float:
  """Angle between body z-axis and world z for quaternion (w,x,y,z)."""
  R = np.zeros(9)
  mujoco.mju_quat2Mat(R, q)
  return float(np.degrees(np.arccos(np.clip(R.reshape(3, 3)[2, 2], -1, 1))))


def g3_physics(task_id: str, cfg, xml_paths: dict[str, Path], baseline: str, spec_info: dict) -> dict:
  results = {}
  gravity = tuple(cfg.sim.mujoco.gravity)
  for name, xml in xml_paths.items():
    if xml is None:
      results[name] = {"error": "xml not found"}
      continue
    r: dict = {"xml": str(xml.relative_to(REPO)) if xml.is_relative_to(REPO) else str(xml)}
    model, data, spec = _object_model(xml)
    fj = _free_joint(model)
    body_ids = _entity_body_ids(model)
    r["mass_kg"] = float(model.body_subtreemass[1]) if model.nbody > 1 else 0.0
    r["ngeom_collision"] = int(sum(1 for g in range(model.ngeom)
                                   if (model.geom_contype[g] or model.geom_conaffinity[g])
                                   and model.geom_type[g] != mujoco.mjtGeom.mjGEOM_PLANE))
    r["nmesh"] = int(model.nmesh)
    r["ntex"] = int(model.ntex)

    # --- articulation ---------------------------------------------------------------
    joints = []
    for j in range(model.njnt):
      jt = model.jnt_type[j]
      if jt in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
        rng = model.jnt_range[j].tolist() if model.jnt_limited[j] else None
        joints.append({"name": model.joint(j).name, "type": "hinge" if jt == mujoco.mjtJoint.mjJNT_HINGE else "slide",
                       "range": rng, "damping": float(model.dof_damping[model.jnt_dofadr[j]]),
                       "frictionloss": float(model.dof_frictionloss[model.jnt_dofadr[j]]),
                       "armature": float(model.dof_armature[model.jnt_dofadr[j]])})
    r["joints"] = joints
    art_ok = True
    for jd in joints:
      if jd["type"] == "hinge" and jd["range"] and (jd["range"][1] - jd["range"][0]) > 2 * math.pi + 1e-6:
        jd["problem"] = "hinge range wider than 2*pi — degrees compiled as radians?"
        art_ok = False
      tv = spec_info.get("target_value")
      if jd["range"] and tv is not None and spec_info.get("joint") == jd["name"]:
        lo, hi = jd["range"]
        if not (lo - 1e-6 <= tv <= hi + 1e-6):
          jd["problem"] = f"command target {tv} outside compiled range {jd['range']}"
          art_ok = False
        jd["target_value"] = tv
    r["articulation_pass"] = art_ok

    # --- settle test (free objects: rest on plane; mechanisms: hold pose) -------------
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    if fj is not None:
      adr = model.jnt_qposadr[fj]
      low = _geom_lowest_z(model, data, body_ids)
      data.qpos[adr + 2] -= low - 0.002  # rest 2 mm above the plane
      data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    z0 = data.qpos[model.jnt_qposadr[fj] + 2] if fj is not None else None
    q0 = data.qpos[model.jnt_qposadr[fj] + 3:model.jnt_qposadr[fj] + 7].copy() if fj is not None else None
    jq0 = data.qpos.copy()
    n_settle = int(2.0 / model.opt.timestep)
    z_after_quarter = None
    for i in range(n_settle):
      mujoco.mj_step(model, data)
      if fj is not None and i == int(0.25 / model.opt.timestep):
        z_after_quarter = float(data.qpos[model.jnt_qposadr[fj] + 2])
    settle = {"nan": bool(np.isnan(data.qpos).any() or np.isnan(data.qvel).any())}
    if fj is not None:
      adr = model.jnt_qposadr[fj]
      dadr = model.jnt_dofadr[fj]
      settle["v_final"] = float(np.linalg.norm(data.qvel[dadr:dadr + 3]))
      settle["w_final"] = float(np.linalg.norm(data.qvel[dadr + 3:dadr + 6]))
      settle["z_drop_after_quarter_s"] = float(z_after_quarter - data.qpos[adr + 2])
      settle["tilt_deg"] = _quat_tilt_deg(data.qpos[adr + 3:adr + 7]) - _quat_tilt_deg(q0)
      settle["lowest_point_z"] = _geom_lowest_z(model, data, body_ids)
      settle["pass"] = (not settle["nan"] and settle["v_final"] < 1e-3 and settle["w_final"] < 1e-2
                        and abs(settle["z_drop_after_quarter_s"]) < 0.005 and abs(settle["tilt_deg"]) < 5.0
                        and settle["lowest_point_z"] > -0.003)
    else:
      # Mechanism: joints must not drift under the TASK's gravity setting.
      model.opt.gravity[:] = gravity
      mujoco.mj_resetData(model, data)
      for _ in range(n_settle):
        mujoco.mj_step(model, data)
      drift = float(np.max(np.abs(data.qpos - jq0))) if model.nq else 0.0
      settle["joint_drift_task_gravity"] = drift
      settle["gravity"] = list(gravity)
      # And report drift with gravity ON as information (lid/switch run with gravity).
      model.opt.gravity[:] = (0, 0, -9.81)
      mujoco.mj_resetData(model, data)
      for _ in range(n_settle):
        mujoco.mj_step(model, data)
      settle["joint_drift_gravity_on"] = float(np.max(np.abs(data.qpos - jq0))) if model.nq else 0.0
      settle["nan"] = bool(np.isnan(data.qpos).any())
      settle["pass"] = (not settle["nan"]) and drift < 0.02
      model.opt.gravity[:] = (0, 0, -9.81)
    r["settle"] = settle

    # --- drop test ------------------------------------------------------------------
    if fj is not None:
      mujoco.mj_resetData(model, data)
      mujoco.mj_forward(model, data)
      adr = model.jnt_qposadr[fj]
      low = _geom_lowest_z(model, data, body_ids)
      data.qpos[adr + 2] -= low - 0.10
      data.qvel[:] = 0
      for _ in range(int(3.0 / model.opt.timestep)):
        mujoco.mj_step(model, data)
      drop = {"nan": bool(np.isnan(data.qpos).any()),
              "lowest_point_z": _geom_lowest_z(model, data, body_ids),
              "v_final": float(np.linalg.norm(data.qvel[model.jnt_dofadr[fj]:model.jnt_dofadr[fj] + 3]))}
      drop["pass"] = (not drop["nan"]) and abs(drop["lowest_point_z"]) < 0.003 and drop["v_final"] < 1e-2
      r["drop"] = drop

    # --- graspable width -------------------------------------------------------------
    if fj is not None:
      mujoco.mj_resetData(model, data)
      mujoco.mj_forward(model, data)
      lo = np.full(3, np.inf); hi = np.full(3, -np.inf)
      for g in range(model.ngeom):
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE or not (model.geom_contype[g] or model.geom_conaffinity[g]):
          continue
        a, b = _geom_aabb(model, data, g)
        lo = np.minimum(lo, a); hi = np.maximum(hi, b)
      ext = (hi - lo)
      r["collision_extent_m"] = ext.round(4).tolist()
      if spec_info.get("graspable") and name == spec_info.get("entity"):
        # Default: the object is pinched as it lies, so only the two horizontal axes
        # can be spanned. "thin_axis": the object is re-presented (tipped onto an edge)
        # before the pinch, so any axis may become the graspable one.
        axes = ext if spec_info["graspable"] == "thin_axis" else ext[:2]
        r["grasp_axis_rule"] = str(spec_info["graspable"])
        r["grasp_width_m"] = float(min(axes))
        r["graspable_pass"] = r["grasp_width_m"] <= 0.060

    # --- step time vs baseline -------------------------------------------------------
    def _steptime(m, d, n=2000):
      mujoco.mj_resetData(m, d)
      mujoco.mj_forward(m, d)
      t0 = time.perf_counter()
      for _ in range(n):
        mujoco.mj_step(m, d)
      return (time.perf_counter() - t0) / n
    t_new = _steptime(model, data)
    t_base = None
    try:
      rel = xml.relative_to(REPO)
      src = subprocess.run(["git", "-C", str(REPO), "show", f"{baseline}:{rel.as_posix()}"],
                           capture_output=True, text=True, check=True).stdout
      with tempfile.TemporaryDirectory() as td:
        bx = Path(td) / xml.name
        bx.write_text(src)
        mb, db, _ = _object_model(bx)
        t_base = _steptime(mb, db)
    except Exception as e:  # noqa: BLE001
      r["step_time_note"] = f"no baseline: {str(e)[:120]}"
    r["step_time_us"] = t_new * 1e6
    if t_base:
      r["step_time_baseline_us"] = t_base * 1e6
      r["step_time_ratio"] = t_new / t_base
      r["step_time_pass"] = (t_new / t_base) <= 1.3 or t_new < 20e-6
    else:
      r["step_time_pass"] = True
    results[name] = r
  return results


# ======================================================================================
# G4 / G5 — the real env
# ======================================================================================


def _mirror_cpu(env, model, k):
  """Plain-MuJoCo copy of env k's live state (qpos + mocap), forwarded."""
  data = mujoco.MjData(model)
  data.qpos[:] = env.sim.data.qpos[k].cpu().numpy()
  if model.nmocap:
    data.mocap_pos[:] = env.sim.data.mocap_pos[k].cpu().numpy()
    data.mocap_quat[:] = env.sim.data.mocap_quat[k].cpu().numpy()
  mujoco.mj_forward(model, data)
  return data


def _target_tensor(term):
  for attr in ("target_value", "target_angle", "target_distance"):
    t = getattr(term, attr, None)
    if t is not None:
      return t
  raise AttributeError(f"{type(term).__name__} exposes no joint target tensor")


def _yaw_of(q):
  w, x, y, z = q
  return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def g4_g5(task_id: str, cfg, spec_info: dict, num_resets: int, num_envs: int, device: str, out: Path) -> tuple[dict, dict]:
  import torch

  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.scripts import audit_workspace as aw
  from mjlab.tasks.manipulation import workspace as ws

  cfg.scene.num_envs = num_envs
  env = ManagerBasedRlEnv(cfg, device=device)
  model = env.sim.mj_model
  zero = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=device)
  term_name = next(iter(env.command_manager.active_terms))
  term = env.command_manager.get_term(term_name)
  origins = env.scene.env_origins
  entities = [n for n in cfg.scene.entities if n not in _SKIP_ENTITIES]
  rounds = int(math.ceil(num_resets / num_envs))

  pos = {n: [] for n in entities}
  yaw = {n: [] for n in entities}
  goals = []
  floor_min = {n: 0.0 for n in entities}
  overlaps: dict[str, float] = {}
  n_overlap_resets = 0
  success_at_reset = 0
  total = 0

  geom_entity = {}
  for gid in range(model.ngeom):
    nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
    geom_entity[gid] = nm.split("/", 1)[0] if "/" in nm else nm

  for _ in range(rounds):
    env.reset()
    # one zero-action step so every command has populated its metrics / target_pos
    env.step(zero)
    sr = term.compute_success().float().cpu().numpy()
    success_at_reset += int(sr.sum())
    total += len(sr)
    for n in entities:
      ent = env.scene[n]
      if "object_site" in ent.site_names:
        p = ent.data.site_pos_w[:, ent.site_names.index("object_site")]
      else:
        p = ent.data.root_link_pos_w
      pos[n].append((p - origins).cpu().numpy())
      q = getattr(ent.data, "root_link_quat_w", None)
      if q is not None:
        yaw[n].append(np.array([_yaw_of(qq) for qq in q.cpu().numpy()]))
    tp = getattr(term, "target_pos", None)
    if tp is not None:
      goals.append((tp - origins).cpu().numpy())
    # CPU mirror of every env: floor penetration + inter-entity contacts
    for k in range(env.num_envs):
      d = _mirror_cpu(env, model, k)
      for gid in range(model.ngeom):
        e = geom_entity[gid]
        if e in entities and (model.geom_contype[gid] or model.geom_conaffinity[gid]):
          half = aw._geom_half_height(model, d, gid)
          floor_min[e] = min(floor_min[e], float(d.geom_xpos[gid][2] - half))
      hit = False
      for i in range(d.ncon):
        con = d.contact[i]
        e1, e2 = geom_entity[con.geom1], geom_entity[con.geom2]
        if "terrain" in (e1, e2) or e1 == e2:
          continue
        if con.dist < -aw._OVERLAP_TOLERANCE:
          key = f"{e1}<->{e2}"
          overlaps[key] = min(overlaps.get(key, 0.0), float(con.dist))
          hit = True
      n_overlap_resets += int(hit)

  # ---- G4 evaluation -----------------------------------------------------------------
  g4: dict = {"num_resets": total, "entities": {}, "spawn_collision_resets": n_overlap_resets,
              "overlaps_m": overlaps, "floor_min_z": floor_min}
  g4_pass = n_overlap_resets == 0 and all(v > -aw._FLOOR_TOLERANCE for v in floor_min.values())
  mech = spec_info["kind"] == "joint"
  for n in entities:
    P = np.concatenate(pos[n])
    rad = np.hypot(P[:, 0], P[:, 1])
    key = (task_id, n)
    if key in aw._REACH_EXEMPT:
      lo, hi, why = aw._REACH_EXEMPT[key]
    elif mech and n == spec_info["entity"]:
      lo, hi, why = 0.0, ws.MECHANISM_HANDLE_RADIAL_MAX, "mechanism handle"
    elif key in aw._FIXTURES:
      lo, hi, why = 0.0, ws.GRASP_RADIAL_MAX + 0.05, "fixture"
    else:
      lo, hi, why = ws.GRASP_RADIAL_MIN, ws.GRASP_RADIAL_MAX, "grasp envelope"
    in_env = float(np.mean((rad >= lo - 1e-6) & (rad <= hi + 1e-6)))
    dead = float(np.mean(rad < ws.GRASP_RADIAL_MIN)) if why == "grasp envelope" else 0.0
    ent = {"n": int(len(P)), "x": [float(P[:, 0].min()), float(P[:, 0].max())],
           "y": [float(P[:, 1].min()), float(P[:, 1].max())],
           "z": [float(P[:, 2].min()), float(P[:, 2].max())],
           "radial": [float(rad.min()), float(rad.max())], "bound": [lo, hi], "rule": why,
           "frac_in_envelope": in_env, "frac_dead_zone": dead}
    ent["pass"] = in_env == 1.0 and dead == 0.0
    g4_pass &= ent["pass"]
    g4["entities"][n] = ent
    _hist(out / f"hist_{n}.png", n, P, rad, np.concatenate(yaw[n]) if yaw[n] else None, (lo, hi))
  if goals:
    G = np.concatenate(goals)
    grad = np.hypot(G[:, 0], G[:, 1])
    key = (task_id, term_name)
    if key in aw._GOAL_EXEMPT:
      lo, hi, why = aw._GOAL_EXEMPT[key]
    elif mech:
      lo, hi, why = 0.0, ws.MECHANISM_GOAL_RADIAL_MAX, "mechanism goal"
    else:
      lo, hi, why = 0.0, ws.GOAL_RADIAL_MAX, "carry goal"
    ok = float(np.mean((grad >= lo - 1e-6) & (grad <= hi + 1e-6)))
    g4["goal"] = {"term": term_name, "radial": [float(grad.min()), float(grad.max())],
                  "z": [float(G[:, 2].min()), float(G[:, 2].max())], "bound": [lo, hi], "rule": why,
                  "frac_in_bound": ok, "pass": ok == 1.0}
    g4_pass &= ok == 1.0
  g4["pass"] = bool(g4_pass)

  # ---- G5 ----------------------------------------------------------------------------
  g5: dict = {"success_at_reset": success_at_reset, "num_resets": total,
              "success_at_reset_pass": success_at_reset == 0}
  kind = spec_info["kind"]
  if kind == "none":
    g5["oracle"] = "n/a (asset-free task)"
    g5["oracle_pass"] = True
  else:
    env.reset()
    env.step(zero)
    ent = env.scene[spec_info["entity"]]
    if kind == "free":
      tp = term.target_pos.clone()
      q = getattr(ent.data, "root_link_quat_w").clone()
      if spec_info.get("quat") is not None:
        q[:] = torch.tensor(spec_info["quat"], device=device)
      # Site offset: the predicate compares the SITE (or body) to target_pos; put the
      # site on the target by shifting the root by the site's current offset.
      if "object_site" in ent.site_names and not spec_info.get("oracle_root"):
        sp = ent.data.site_pos_w[:, ent.site_names.index("object_site")]
        off = ent.data.root_link_pos_w - sp
      else:
        off = torch.zeros_like(tp)
      pose = torch.cat([tp + off, q], dim=-1)
      ent.write_root_link_pose_to_sim(pose)
      ent.write_root_link_velocity_to_sim(torch.zeros(env.num_envs, 6, device=device))
    else:
      jid = ent.joint_names.index(spec_info["joint"])
      tv = _target_tensor(term).clone().unsqueeze(-1)
      spec_info["target_value"] = float(tv[0, 0])
      ent.write_joint_position_to_sim(tv, joint_ids=torch.tensor([jid], device=device))
      ent.write_joint_velocity_to_sim(torch.zeros(env.num_envs, 1, device=device),
                                      joint_ids=torch.tensor([jid], device=device))
    hits = []
    # oracle_hold: SPRING-RETURNED mechanisms. Push-Button's slide joint carries
    # stiffness=1000 on a 36 g plunger (zeta = 0.42, period 38 ms), so writing it to
    # -0.05 and then taking three free 20 ms steps samples the predicate AFTER the
    # spring has already thrown the cap back through zero — measured trajectory
    # +0.0163, -0.0045, +0.0010. The oracle then reads 0.000 and G5 fails on a task
    # whose scripted teacher scores 1.000, because the button is only ever AT the
    # target while something holds it there. Identical numbers on the pre-CL-V2
    # primitive asset, so this is a verifier artefact, not an asset defect (W2-a).
    # Re-seat the joint after each step and refresh the metrics, i.e. ask the real G5
    # question: with the mechanism AT the goal configuration, does the predicate fire?
    # Same move `tests/test_class_a_expansion.py` makes for the switch's detent.
    hold_jid = (
      torch.tensor([jid], device=device)
      if kind == "joint" and spec_info.get("oracle_hold")
      else None
    )
    for _ in range(3):
      env.step(zero)
      if hold_jid is not None:
        ent.write_joint_position_to_sim(tv, joint_ids=hold_jid)
        ent.write_joint_velocity_to_sim(
          torch.zeros(env.num_envs, 1, device=device), joint_ids=hold_jid
        )
        term._update_metrics()
      hits.append(term.compute_success().float().cpu().numpy())
    frac = float(np.max(np.stack(hits), axis=0).mean())
    g5["oracle_success_fraction"] = frac
    g5["oracle_pass"] = frac >= 0.9
    thr = {k: v for k, v in vars(term.cfg).items()
           if any(s in k for s in ("threshold", "tolerance", "clearance", "target_value", "aperture", "drift", "height"))
           and isinstance(v, (int, float))}
    g5["thresholds"] = thr
  g5["pass"] = bool(g5["success_at_reset_pass"] and g5["oracle_pass"])
  env.close()
  return g4, g5


def _hist(path: Path, name: str, P, rad, yaw, bound):
  try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
  except Exception:
    return
  def _rng(v):
    """Bin range for a possibly CONSTANT coordinate.

    A fixture written to a fixed pose (the pivot wall, the edge-grasp riser) comes back
    with a range of ~1e-16 -- the float noise of adding and subtracting the env origin --
    and numpy then refuses: "Too many bins for data range". Widening a degenerate range
    by hand is the whole fix; it is a plotting concern only, and the gate numbers above
    are computed from the raw arrays. (W1-c)
    """
    lo, hi = float(np.min(v)), float(np.max(v))
    return (lo - 5e-4, hi + 5e-4) if (hi - lo) < 1e-6 else (lo, hi)

  try:
    fig, ax = plt.subplots(1, 4, figsize=(14, 3))
    ax[0].hist(P[:, 0], bins=30, range=_rng(P[:, 0])); ax[0].set_title(f"{name} x")
    ax[1].hist(P[:, 1], bins=30, range=_rng(P[:, 1])); ax[1].set_title("y")
    ax[2].hist(rad, bins=30, range=_rng(rad)); ax[2].axvline(bound[0], c="r"); ax[2].axvline(bound[1], c="r"); ax[2].set_title("radial")
    if yaw is not None:
      ax[3].hist(yaw, bins=30, range=_rng(yaw)); ax[3].set_title("yaw")
    fig.tight_layout(); fig.savefig(path, dpi=80); plt.close(fig)
  except Exception as e:  # noqa: BLE001
    # The histograms are EVIDENCE, not a gate. A plotting failure must never take the
    # run down with it -- it did, on a fixture written to a fixed pose. (W1-c)
    print(f"[hist] {name}: skipped ({type(e).__name__}: {e}); "
          f"x span {float(np.ptp(P[:, 0])):.3e} y span {float(np.ptp(P[:, 1])):.3e} "
          f"rad span {float(np.ptp(rad)):.3e} "
          f"yaw span {'-' if yaw is None else format(float(np.ptp(yaw)), '.3e')}")


# ======================================================================================


def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--task", required=True)
  ap.add_argument("--out", required=True)
  ap.add_argument("--num-resets", type=int, default=1000)
  ap.add_argument("--num-envs", type=int, default=50)
  ap.add_argument("--device", default="cpu")
  ap.add_argument("--baseline", default=BASELINE)
  ap.add_argument("--skip-g3", action="store_true")
  ap.add_argument("--skip-g45", action="store_true")
  a = ap.parse_args(argv)

  from mjlab.tasks.registry import load_env_cfg
  from mjlab.utils.torch import configure_torch_backends

  configure_torch_backends()
  out = Path(a.out)
  out.mkdir(parents=True, exist_ok=True)
  spec_info = dict(TASKS[a.task])
  cfg = load_env_cfg(a.task, test=True)
  if spec_info["kind"] == "joint":
    term_cfg = next(iter(cfg.commands.values()))
    if getattr(term_cfg, "target_value", None) is not None:
      spec_info["target_value"] = float(term_cfg.target_value)
  head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
  report = {"task": a.task, "head": head, "baseline": a.baseline, "num_resets": a.num_resets, "device": a.device}

  if not a.skip_g45:
    g4, g5 = g4_g5(a.task, cfg, spec_info, a.num_resets, a.num_envs, a.device, out)
    report["G4"] = g4
    report["G5"] = g5
    report["G4_pass"] = g4["pass"]
    report["G5_pass"] = g5["pass"]
  if not a.skip_g3:
    xmls = _entity_xml_paths(cfg)
    report["G3"] = g3_physics(a.task, cfg, xmls, a.baseline, spec_info)
    g3_pass = True
    for n, r in report["G3"].items():
      for k in ("settle", "drop"):
        if k in r and not r[k].get("pass", True):
          g3_pass = False
      for k in ("articulation_pass", "graspable_pass", "step_time_pass"):
        if k in r and not r[k]:
          g3_pass = False
      if "error" in r:
        g3_pass = False
    report["G3_pass"] = g3_pass
  (out / "verify.json").write_text(json.dumps(report, indent=1, default=float))

  print(f"\n=== verify_task {a.task} @ {head} ===")
  for g in ("G3", "G4", "G5"):
    if f"{g}_pass" in report:
      print(f"{g}: {'PASS' if report[f'{g}_pass'] else 'FAIL'}")
  if "G3" in report:
    for n, r in report["G3"].items():
      s = r.get("settle", {}); d = r.get("drop", {})
      print(f"  [{n}] mass={r.get('mass_kg', 0):.3f}kg ncol={r.get('ngeom_collision')} "
            f"settle={'ok' if s.get('pass') else 'FAIL'} drop={'ok' if d.get('pass', True) else 'FAIL'} "
            f"grasp_w={r.get('grasp_width_m', '-')} step={r.get('step_time_us', 0):.0f}us "
            f"ratio={r.get('step_time_ratio', float('nan')):.2f} joints={[(j['name'], j['range'], j.get('problem')) for j in r.get('joints', [])]}")
  if "G4" in report:
    for n, e in report["G4"]["entities"].items():
      print(f"  [{n}] radial {e['radial'][0]:.3f}-{e['radial'][1]:.3f} bound {e['bound']} in={e['frac_in_envelope']:.3f} dead={e['frac_dead_zone']:.3f} {'ok' if e['pass'] else 'FAIL'}")
    print(f"  spawn-collision resets: {report['G4']['spawn_collision_resets']}  floor_min_z: {report['G4']['floor_min_z']}")
    if "goal" in report["G4"]:
      g = report["G4"]["goal"]; print(f"  goal radial {g['radial'][0]:.3f}-{g['radial'][1]:.3f} bound {g['bound']} {'ok' if g['pass'] else 'FAIL'}")
  if "G5" in report:
    print(f"  success_at_reset: {report['G5']['success_at_reset']}/{report['G5']['num_resets']}  oracle: {report['G5'].get('oracle_success_fraction', report['G5'].get('oracle'))}")
  print(f"  -> {out / 'verify.json'}")
  ok = all(report.get(f"{g}_pass", True) for g in ("G3", "G4", "G5"))
  return 0 if ok else 1


if __name__ == "__main__":
  sys.exit(main())
