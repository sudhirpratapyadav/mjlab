"""Task geometry invariants, independent of teacher performance."""

import importlib

import mujoco
import numpy as np
import pytest
import torch

from mjlab import MJLAB_SRC_PATH
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.manipulation.benchmark import all_benchmark_tasks
from mjlab.tasks.manipulation.taxonomy import Embodiment
from mjlab.tasks.registry import load_env_cfg

TASKS = [
  name
  for name, tax in all_benchmark_tasks().items()
  if tax.embodiment is Embodiment.ARM_GRIPPER
]


@pytest.mark.parametrize("task", TASKS)
def test_goal_shape_pose_and_collision_masks(task):
  cfg = load_env_cfg(task, test=True)
  cfg.scene.num_envs = 2
  env = ManagerBasedRlEnv(cfg, device="cpu")
  try:
    env.reset()
    env.sim.forward()
    marker = env.scene["mocap_goal"].cfg.spec_fn().compile()
    assert np.all((marker.geom_rgba[:, 3] > 0) & (marker.geom_rgba[:, 3] < 1))
    assert not np.any(marker.geom_contype | marker.geom_conaffinity)
    for command in env.command_manager._terms.values():
      if not hasattr(command, "_goal_geometry"):
        continue
      asset = getattr(command, "asset", None)
      if asset is None:
        asset = next(
          getattr(command, k)
          for k in ("door", "drawer", "button")
          if hasattr(command, k)
        )
      value = next(
        getattr(command, k)
        for k in ("target_value", "target_angle", "target_distance")
        if hasattr(command, k)
      )
      asset.write_joint_position_to_sim(value[:, None])
      env.sim.forward()
      site = asset.data.site_pos_w[:, asset.site_names.index("object_site")]
      torch.testing.assert_close(site, command.target_pos, atol=1e-5, rtol=0)
      # The replica is site-centered, with the moving body's orientation.
      actual_quat = asset.data.body_link_quat_w[:, asset.body_names.index("handle")]
      goal_quat = env.scene["mocap_goal"].data.root_link_quat_w
      assert torch.all(torch.abs((actual_quat * goal_quat).sum(-1)) > 1 - 1e-5)
    for _ in range(3):
      env.step(torch.zeros((env.num_envs, env.action_manager.total_action_dim)))
      assert torch.isfinite(env.sim.data.qpos).all()
    robot = env.scene["robot"].cfg.spec_fn().compile()
    for body_name in (
      "link1",
      "link2",
      "link3",
      "link4",
      "link5",
      "link6",
      "link7",
      "hand",
      "left_finger",
      "right_finger",
    ):
      body = robot.body(body_name).id
      geoms = np.flatnonzero((robot.geom_bodyid == body) & (robot.geom_group == 3))
      assert np.any(
        (robot.geom_contype[geoms] & 1) | (robot.geom_conaffinity[geoms] & 2)
      ), body_name
  finally:
    env.close()


MECHANISMS = sorted(
  (MJLAB_SRC_PATH / "asset_zoo/objects/articulated").glob("*/*_constants.py")
)


@pytest.mark.parametrize("path", MECHANISMS, ids=lambda p: p.parent.name)
def test_mechanism_sweep_clears_housing(path):
  name = path.parent.name
  module = importlib.import_module(
    f"mjlab.asset_zoo.objects.articulated.{name}.{name}_constants"
  )
  model = getattr(module, f"get_{name}_spec")().compile()
  data = mujoco.MjData(model)
  moving = model.body("handle").id
  colliders = np.flatnonzero(model.geom_contype | model.geom_conaffinity)
  fixed = [i for i in colliders if model.geom_bodyid[i] != moving]
  parts = [i for i in colliders if model.geom_bodyid[i] == moving]
  lo, hi = model.jnt_range[0]
  for value in np.linspace(lo, hi, 65):
    data.qpos[0] = value
    mujoco.mj_forward(model, data)
    # Query distance explicitly: parent-child filtering must not hide penetration.
    for a in fixed:
      for b in parts:
        distance = mujoco.mj_geomDistance(model, data, a, b, 0.05, None)
        assert distance >= -1e-5, (
          name,
          model.geom(a).name,
          model.geom(b).name,
          value,
          distance,
        )


def test_arm_wrist_and_palm_generate_object_contacts():
  from mjlab.asset_zoo.robots.franka_emika_panda.franka_constants import get_spec

  spec = get_spec()
  probe = spec.worldbody.add_body(name="contact_probe", mocap=True)
  probe.add_geom(
    name="probe",
    type=mujoco.mjtGeom.mjGEOM_SPHERE,
    size=[0.015, 0, 0],
    contype=2,
    conaffinity=1,
  )
  model = spec.compile()
  data = mujoco.MjData(model)
  data.qpos[:7] = [0, 0.3, 0, -1.57079, 0, 2, -0.7853]
  data.qpos[7:] = 0.04
  probe_geom = model.geom("probe").id
  for name in (
    "link1",
    "link4",
    "link6",
    "link7",
    "hand",
    "left_finger",
    "right_finger",
  ):
    mujoco.mj_forward(model, data)
    body = model.body(name).id
    geoms = np.flatnonzero((model.geom_bodyid == body) & (model.geom_contype != 0))
    geom = geoms[0]
    mesh = model.geom_dataid[geom]
    vertex = model.mesh_vert[model.mesh_vertadr[mesh]]
    # Intersect the hull surface, rather than putting the probe at a degenerate
    # coincident centroid (some GJK/EPA implementations return zero there).
    data.mocap_pos[0] = (
      data.geom_xpos[geom] + data.geom_xmat[geom].reshape(3, 3) @ vertex
    )
    mujoco.mj_forward(model, data)
    assert any(
      probe_geom in c.geom and any(g in geoms for g in c.geom) and c.dist < 0
      for c in data.contact
    ), name


ASSET_MODULES = sorted(
  (MJLAB_SRC_PATH / "asset_zoo/objects").glob("*/*/*_constants.py")
)


@pytest.mark.parametrize("path", ASSET_MODULES, ids=lambda p: p.parent.name)
def test_goal_replica_geometry_coincides_with_source(path):
  from mjlab.asset_zoo.objects.goal import make_goal_spec

  relative = path.relative_to(MJLAB_SRC_PATH).with_suffix("")
  module = importlib.import_module("mjlab." + ".".join(relative.parts))
  makers = [
    getattr(module, name)
    for name in vars(module)
    if name.startswith("get_") and name.endswith("_spec") and "mocap" not in name
  ]
  for maker in makers:
    spec = maker()
    if spec.site("object_site") is None:
      continue
    model = spec.compile()
    data = mujoco.MjData(model)
    for j in range(model.njnt):
      if model.jnt_type[j] in (
        mujoco.mjtJoint.mjJNT_HINGE,
        mujoco.mjtJoint.mjJNT_SLIDE,
      ):
        data.qpos[model.jnt_qposadr[j]] = model.jnt_range[j].mean()
    mujoco.mj_forward(model, data)
    site = model.site("object_site").id
    body = model.site_bodyid[site]
    visible = np.flatnonzero(
      (model.geom_bodyid == body)
      & (model.geom_group != 3)
      & (model.geom_rgba[:, 3] > 0)
    )
    if not len(visible):
      visible = np.flatnonzero(model.geom_bodyid == body)
    goal = make_goal_spec(spec).compile()
    gd = mujoco.MjData(goal)
    gd.mocap_pos[0] = data.site_xpos[site]
    gd.mocap_quat[0] = data.xquat[body]
    mujoco.mj_forward(goal, gd)
    assert goal.ngeom == len(visible)
    np.testing.assert_array_equal(goal.geom_type, model.geom_type[visible])
    np.testing.assert_allclose(gd.geom_xpos, data.geom_xpos[visible], atol=1e-6)
    np.testing.assert_allclose(gd.geom_xmat, data.geom_xmat[visible], atol=1e-6)
    np.testing.assert_allclose(goal.geom_size, model.geom_size[visible], atol=1e-6)
