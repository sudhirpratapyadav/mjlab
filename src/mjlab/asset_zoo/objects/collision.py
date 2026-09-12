"""Collision shells for visible housings with a shaft bore."""

import math

import mujoco


def add_collision_shell(spec, body, name, *, axis, lo, hi, inner, outer, segments=12):
  """Convex annular sectors preserve the bore instead of filling it with a hull."""
  for i in range(segments):
    angles = (2 * math.pi * i / segments, 2 * math.pi * (i + 1) / segments)
    vertices = []
    for axial in (lo, hi):
      for radius in (inner, outer):
        for angle in angles:
          point = [radius * math.cos(angle), radius * math.sin(angle)]
          point.insert(axis, axial)
          vertices.extend(point)
    mesh = f"{name}_{i}"
    spec.add_mesh(name=mesh, uservert=vertices)
    body.add_geom(
      name=mesh,
      type=mujoco.mjtGeom.mjGEOM_MESH,
      meshname=mesh,
      group=3,
      contype=2,
      conaffinity=1,
      mass=0,
      friction=[1, 0.03, 0.003],
      solref=[0.01, 1],
    )
