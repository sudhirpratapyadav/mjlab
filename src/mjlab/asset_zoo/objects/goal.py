"""Visual-only goal replicas, anchored at the manipulated object's tracking site."""

import mujoco
import numpy as np

GOAL_RGBA = (1.0, 0.5, 0.0, 0.28)


def make_goal_spec(source: mujoco.MjSpec) -> mujoco.MjSpec:
  """Copy the moving part's actual visible geometry, without physics or textures.

  The mocap origin is object_site; its axes are the moving body's axes. Keeping
  original mesh transforms (rather than compiled mesh-centroid transforms) avoids
  applying MuJoCo's mesh recentering twice.
  """
  site = source.site("object_site")
  body = site.parent
  geoms = list(body.geoms)
  visible = [g for g in geoms if g.group != 3 and g.rgba[3] > 0]
  if not visible:
    visible = geoms
  records = [
    dict(
      name=f"goal_{i}",
      type=g.type,
      size=g.size.copy(),
      pos=g.pos.copy() - site.pos,
      quat=g.quat.copy(),
      meshname=g.meshname,
      rgba=GOAL_RGBA,
      group=2,
      contype=0,
      conaffinity=0,
      mass=0,
    )
    for i, g in enumerate(visible)
  ]
  for b in list(source.worldbody.bodies):
    source.delete(b)
  for k in list(source.keys):
    source.delete(k)
  marker = source.worldbody.add_body(name="mocap_goal", mocap=True)
  for record in records:
    marker.add_geom(**record)
  return source


def joint_goal_geometry(spec: mujoco.MjSpec, joint_name: str):
  """Joint axis, pivot and zero-pose site/body transform in the mount frame."""
  model = spec.compile()
  data = mujoco.MjData(model)
  data.qpos[:] = 0
  mujoco.mj_forward(model, data)
  joint = model.joint(joint_name).id
  site = model.site("object_site").id
  body = model.site_bodyid[site]
  root = body
  while model.body_parentid[root] != 0:
    root = model.body_parentid[root]
  rotation = data.xmat[root].reshape(3, 3)
  origin = data.xpos[root]
  quat = np.empty(4)
  mujoco.mju_mat2Quat(quat, (rotation.T @ data.xmat[body].reshape(3, 3)).ravel())
  return (
    int(model.jnt_type[joint]),
    rotation.T @ data.xaxis[joint],
    rotation.T @ (data.xanchor[joint] - origin),
    rotation.T @ (data.site_xpos[site] - origin),
    quat,
  )


def object_support_points(spec: mujoco.MjSpec) -> np.ndarray:
  """Conservative collider corners in the tracking-site/body frame."""
  model = spec.compile()
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  site = model.site("object_site").id
  body = model.site_bodyid[site]
  rotation = data.xmat[body].reshape(3, 3)
  points = []
  from itertools import product

  corners = np.array(list(product((-1, 1), repeat=3)))
  for g in range(model.ngeom):
    if model.geom_bodyid[g] != body or not (
      model.geom_contype[g] or model.geom_conaffinity[g]
    ):
      continue
    if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
      mesh = model.geom_dataid[g]
      start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
      vertices = model.mesh_vert[start : start + count]
    else:
      size = model.geom_size[g].copy()
      if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE:
        size[:] = size[0]
      elif model.geom_type[g] in (
        mujoco.mjtGeom.mjGEOM_CYLINDER,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
      ):
        size = np.array(
          [
            size[0],
            size[0],
            size[1]
            + (size[0] if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE else 0),
          ]
        )
      vertices = corners * size
    world = vertices @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]
    points.extend((world - data.site_xpos[site]) @ rotation)
  return np.asarray(points)


def make_reach_goal_spec() -> mujoco.MjSpec:
  """A position-only target has no object shape or requested orientation."""
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="mocap_goal", mocap=True)
  body.add_geom(
    name="reach_goal",
    type=mujoco.mjtGeom.mjGEOM_SPHERE,
    size=[0.03, 0, 0],
    rgba=GOAL_RGBA,
    contype=0,
    conaffinity=0,
  )
  return spec
