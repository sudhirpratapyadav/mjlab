"""CL-V2 asset pipeline: download -> normalise -> colliders -> MJCF package.

One tool so that every task owner packages a realistic asset the same way.  It
covers the three online sources that are reachable from the login node (Poly
Haven, Google Scanned Objects via Gazebo Fuel, YCB) plus two texture libraries
(ambientCG, Poly Haven textures) for Blender/trimesh-built assets, and it turns
any textured mesh into the on-disk layout the object XMLs expect::

    <asset_dir>/<name>_vis.obj      visual mesh, UVs, <= --max-tris triangles
    <asset_dir>/<name>_tex.png      albedo (AO baked in when available), <= --tex-res
    <asset_dir>/<name>_col_NN.obj   convex colliders (CoACD pieces, hull, or none)
    <asset_dir>/<name>_package.json what was done: scale, extents, mass, CoM, inertia,
                                    hull count, tri count, source, license

and prints the MJCF ``<asset>`` / ``<geom>`` / ``<inertial>`` snippets to paste
into the object XML.  Raw downloads live OUTSIDE the repo in ``~/assets_raw``.

Usage (CLI)::

    python -m mjlab.scripts.asset_pipeline fetch polyhaven:cardboard_box_01 --res 1k
    python -m mjlab.scripts.asset_pipeline fetch gso:Wooden_ABC_123_Blocks_50_pack
    python -m mjlab.scripts.asset_pipeline fetch ycb:036_wood_block
    python -m mjlab.scripts.asset_pipeline fetch ambientcg:Wood051 --res 1K
    python -m mjlab.scripts.asset_pipeline fetch polyhaven-texture:wood_table_001
    python -m mjlab.scripts.asset_pipeline inspect ~/assets_raw/ycb/036_wood_block/google_16k/textured.obj
    python -m mjlab.scripts.asset_pipeline package <mesh> --name cube \
        --out src/mjlab/asset_zoo/objects/free/cube/xmls/assets \
        --target-extent 0.04 0.04 0.04 --origin centroid --collider box --density 600

Python API: :func:`load_mesh`, :func:`normalise`, :func:`decimate`,
:func:`decompose`, :func:`package`.

MuJoCo facts this encodes (verified 2026-09-09 on mujoco 3.11.1):
  * textures must be PNG (JPG is rejected as "custom binary file format");
  * OBJ texcoords are honoured; the visual mesh is a separate geom with
    ``contype=0 conaffinity=0`` and the colliders are ``group=3`` invisible geoms;
  * a mesh geom collides as its convex hull, so every collider piece must be convex;
  * the classic renderer has no normal/roughness maps, so AO is baked into albedo.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

RAW_ROOT = Path(os.environ.get("MJLAB_ASSETS_RAW", Path.home() / "assets_raw"))
USER_AGENT = "Mozilla/5.0 mjlab-asset-pipeline"

# --------------------------------------------------------------------------------------
# Download helpers
# --------------------------------------------------------------------------------------


def _curl(url: str, dst: Path) -> Path:
  dst.parent.mkdir(parents=True, exist_ok=True)
  subprocess.run(
    ["curl", "-sSL", "-A", USER_AGENT, "-o", str(dst), url], check=True, timeout=600
  )
  if dst.stat().st_size < 200:
    txt = dst.read_text(errors="replace")
    if "<Error>" in txt or "Forbidden" in txt or "Not Found" in txt:
      raise RuntimeError(f"download of {url} failed: {txt[:200]}")
  return dst


def _curl_json(url: str) -> Any:
  out = subprocess.run(
    ["curl", "-sSL", "-A", USER_AGENT, url], check=True, capture_output=True, timeout=120
  )
  return json.loads(out.stdout)


def fetch_polyhaven(asset_id: str, res: str = "1k") -> Path:
  """Poly Haven model (CC0). Returns the .gltf path. Dimensions are in metres."""
  root = RAW_ROOT / "polyhaven" / asset_id
  files = _curl_json(f"https://api.polyhaven.com/files/{asset_id}")
  info = _curl_json(f"https://api.polyhaven.com/info/{asset_id}")
  (root).mkdir(parents=True, exist_ok=True)
  (root / "files.json").write_text(json.dumps(files, indent=1))
  (root / "info.json").write_text(json.dumps(info, indent=1))
  g = files["gltf"][res]["gltf"]
  gltf = _curl(g["url"], root / f"{asset_id}.gltf")
  for name, entry in g["include"].items():
    _curl(entry["url"], root / name)
  print(f"[polyhaven] {asset_id}: dims(mm)={info.get('dimensions')} -> {gltf}")
  return gltf


def fetch_polyhaven_texture(asset_id: str, res: str = "1k") -> Path:
  """Poly Haven PBR texture set (CC0): returns the directory with the JPGs."""
  root = RAW_ROOT / "polyhaven_tex" / asset_id
  files = _curl_json(f"https://api.polyhaven.com/files/{asset_id}")
  root.mkdir(parents=True, exist_ok=True)
  got = []
  for role in ("Diffuse", "AO", "Rough", "nor_gl", "Displacement"):
    if role in files and res in files[role]:
      entry = files[role][res].get("jpg") or files[role][res].get("png")
      if entry:
        ext = "jpg" if "jpg" in files[role][res] else "png"
        got.append(_curl(entry["url"], root / f"{asset_id}_{role.lower()}_{res}.{ext}"))
  print(f"[polyhaven-texture] {asset_id}: {[p.name for p in got]}")
  return root


def fetch_gso(name: str) -> Path:
  """Google Scanned Objects via Gazebo Fuel (CC-BY 4.0). Returns meshes/model.obj."""
  root = RAW_ROOT / "gso" / name
  zpath = RAW_ROOT / "gso" / f"{name}.zip"
  if not (root / "meshes" / "model.obj").exists():
    _curl(f"https://fuel.gazebosim.org/1.0/GoogleResearch/models/{name}.zip", zpath)
    root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zpath) as z:
      z.extractall(root)
  obj = root / "meshes" / "model.obj"
  print(f"[gso] {name} -> {obj}")
  return obj


def fetch_ycb(name: str, variant: str = "google_16k") -> Path:
  """YCB object (textured scan). `name` like '036_wood_block'. Returns textured.obj."""
  root = RAW_ROOT / "ycb"
  tgz = root / f"{name}_{variant}.tgz"
  obj = root / name / variant / "textured.obj"
  if not obj.exists():
    _curl(f"https://ycb-benchmarks.s3.amazonaws.com/data/google/{name}_{variant}.tgz", tgz)
    with tarfile.open(tgz) as t:
      t.extractall(root)
  print(f"[ycb] {name} -> {obj}")
  return obj


def fetch_ambientcg(asset_id: str, res: str = "1K") -> Path:
  """ambientCG texture set (CC0), e.g. Wood051. Returns the extracted directory."""
  root = RAW_ROOT / "ambientcg" / f"{asset_id}_{res}"
  if not root.exists():
    zpath = RAW_ROOT / "ambientcg" / f"{asset_id}_{res}-JPG.zip"
    _curl(f"https://ambientcg.com/get?file={asset_id}_{res}-JPG.zip", zpath)
    root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zpath) as z:
      z.extractall(root)
  print(f"[ambientcg] {asset_id}: {sorted(p.name for p in root.iterdir())}")
  return root


def fetch(spec: str, res: str | None = None) -> Path:
  src, _, ident = spec.partition(":")
  if src == "polyhaven":
    return fetch_polyhaven(ident, res or "1k")
  if src == "polyhaven-texture":
    return fetch_polyhaven_texture(ident, res or "1k")
  if src == "gso":
    return fetch_gso(ident)
  if src == "ycb":
    return fetch_ycb(ident, res or "google_16k")
  if src == "ambientcg":
    return fetch_ambientcg(ident, res or "1K")
  raise ValueError(f"unknown source {src!r}; use polyhaven|polyhaven-texture|gso|ycb|ambientcg")


# --------------------------------------------------------------------------------------
# Mesh helpers
# --------------------------------------------------------------------------------------


def load_mesh(path: str | Path, geometry: str | None = None):
  """Load any trimesh-readable file as ONE textured Trimesh.

  A glTF/OBJ scene with several geometries is concatenated when they share one
  texture image; otherwise pass ``geometry=<name>`` (names printed in the error).
  """
  import trimesh

  path = Path(path)
  loaded = trimesh.load(str(path), process=False)
  if isinstance(loaded, trimesh.Trimesh):
    return loaded
  geoms = dict(loaded.geometry)
  if geometry is not None:
    return _apply_scene_transform(loaded, geometry)
  if len(geoms) == 1:
    name = next(iter(geoms))
    return _apply_scene_transform(loaded, name)
  images = {}
  for n, g in geoms.items():
    img = _get_image(g)
    images[n] = None if img is None else (img.size, img.tobytes()[:64])
  if len(set(map(str, images.values()))) == 1:
    parts = [_apply_scene_transform(loaded, n) for n in geoms]
    return trimesh.util.concatenate(parts)
  raise ValueError(
    f"{path} has {len(geoms)} geometries with different textures; pass geometry= one of "
    f"{list(geoms)}"
  )


def _apply_scene_transform(scene, name: str):
  """Return geometry `name` baked into world coordinates of the scene graph."""
  import trimesh

  g = scene.geometry[name].copy()
  # Find the node using this geometry and apply its world transform.
  for node in scene.graph.nodes_geometry:
    T, gname = scene.graph[node]
    if gname == name:
      g.apply_transform(T)
      break
  if not isinstance(g, trimesh.Trimesh):
    raise TypeError(f"geometry {name} is {type(g)}, not a Trimesh")
  return g


def _get_image(mesh):
  """Albedo image of a textured mesh (SimpleMaterial.image or PBRMaterial.baseColorTexture)."""
  mat = getattr(mesh.visual, "material", None)
  if mat is None:
    return None
  img = getattr(mat, "image", None)
  if img is None:
    img = getattr(mat, "baseColorTexture", None)
  return img


def inspect_mesh(mesh) -> dict[str, Any]:
  ext = (mesh.bounds[1] - mesh.bounds[0]).tolist()
  uv = getattr(mesh.visual, "uv", None)
  img = _get_image(mesh)
  info = {
    "vertices": int(len(mesh.vertices)),
    "faces": int(len(mesh.faces)),
    "extent_m": [round(e, 4) for e in ext],
    "bounds": mesh.bounds.round(4).tolist(),
    "watertight": bool(mesh.is_watertight),
    "volume_m3": float(mesh.volume) if mesh.is_watertight else None,
    "hull_volume_m3": float(mesh.convex_hull.volume),
    "has_uv": uv is not None and len(uv) == len(mesh.vertices),
    "texture_size": None if img is None else list(img.size),
  }
  return info


def normalise(
  mesh,
  scale: float | None = None,
  target_extent: tuple[float, float, float] | None = None,
  rotate_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
  up: str = "z",
  origin: str = "centroid",
  translate: tuple[float, float, float] = (0.0, 0.0, 0.0),
):
  """Scale/rotate/re-origin a mesh in place and return it.

  ``up='y'`` first converts a glTF Y-up mesh to Z-up. ``rotate_deg`` is applied
  (x, y, z, extrinsic, degrees) after that. Scaling: explicit ``scale`` or a
  ``target_extent`` (per-axis, use ``None``/0 to leave an axis unconstrained; the
  scale is uniform, chosen from the first constrained axis). ``origin`` puts the body
  frame at: ``centroid`` (mesh mass centre), ``bbox`` (bounding-box centre),
  ``bottom`` (bbox centre in xy, lowest point in z), ``keep`` (leave as is).
  ``translate`` is a final offset.
  """
  import trimesh

  if up == "y":
    mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
  for axis, ang in zip(([1, 0, 0], [0, 1, 0], [0, 0, 1]), rotate_deg):
    if ang:
      mesh.apply_transform(trimesh.transformations.rotation_matrix(np.deg2rad(ang), axis))
  if scale is None and target_extent is not None:
    ext = mesh.bounds[1] - mesh.bounds[0]
    for i, t in enumerate(target_extent):
      if t:
        scale = float(t) / float(ext[i])
        break
  if scale is not None and scale != 1.0:
    mesh.apply_scale(scale)
  if origin == "centroid":
    c = mesh.center_mass if mesh.is_watertight else mesh.centroid
    mesh.apply_translation(-np.asarray(c))
  elif origin == "bbox":
    mesh.apply_translation(-mesh.bounds.mean(axis=0))
  elif origin == "bottom":
    c = mesh.bounds.mean(axis=0)
    mesh.apply_translation(-np.array([c[0], c[1], mesh.bounds[0][2]]))
  elif origin != "keep":
    raise ValueError(origin)
  if any(translate):
    mesh.apply_translation(np.asarray(translate, dtype=float))
  return mesh


def decimate(mesh, max_tris: int):
  """Reduce to <= max_tris while keeping UVs (Blender if available, else trimesh)."""
  if len(mesh.faces) <= max_tris:
    return mesh
  blender = Path.home() / "tools" / "blender" / "blender"
  if blender.exists():
    return _decimate_blender(mesh, max_tris, blender)
  # Fallback: quadric decimation drops UVs; only acceptable for untextured meshes.
  print(f"[decimate] WARNING: no Blender at {blender}; trimesh decimation drops UVs")
  return mesh.simplify_quadric_decimation(max_tris)


def _decimate_blender(mesh, max_tris: int, blender: Path):
  import tempfile

  import trimesh

  with tempfile.TemporaryDirectory() as td:
    src = Path(td) / "in.obj"
    dst = Path(td) / "out.obj"
    _write_obj(mesh, src, texture_name=None)
    ratio = max_tris / len(mesh.faces)
    script = Path(td) / "dec.py"
    script.write_text(
      "import bpy\n"
      "bpy.ops.wm.read_factory_settings(use_empty=True)\n"
      f"bpy.ops.wm.obj_import(filepath={str(src)!r})\n"
      "for o in bpy.context.scene.objects:\n"
      "    if o.type=='MESH':\n"
      "        bpy.context.view_layer.objects.active=o\n"
      "        m=o.modifiers.new('dec','DECIMATE'); m.ratio=" + repr(ratio) + "\n"
      "        bpy.ops.object.modifier_apply(modifier='dec')\n"
      f"bpy.ops.wm.obj_export(filepath={str(dst)!r}, export_materials=False, export_uv=True, export_normals=False)\n"
    )
    subprocess.run([str(blender), "-b", "--python", str(script)], check=True,
                   capture_output=True, timeout=1200)
    out = trimesh.load(str(dst), process=False, force="mesh")
    # Re-attach the texture image.
    img = _get_image(mesh)
    if img is not None and getattr(out.visual, "uv", None) is not None:
      out.visual = trimesh.visual.TextureVisuals(uv=out.visual.uv, image=img)
    print(f"[decimate] blender: {len(mesh.faces)} -> {len(out.faces)} tris")
    return out


def decompose(mesh, max_hulls: int = 8, threshold: float = 0.05, resolution: int = 2000,
              seed: int = 0):
  """CoACD convex decomposition -> list of convex Trimesh pieces."""
  import coacd
  import trimesh

  cm = coacd.Mesh(np.asarray(mesh.vertices, dtype=np.float64),
                  np.asarray(mesh.faces, dtype=np.int64))
  parts = coacd.run_coacd(cm, threshold=threshold, max_convex_hull=max_hulls,
                          resolution=resolution, seed=seed, max_ch_vertex=64)
  hulls = []
  for v, f in parts:
    h = trimesh.Trimesh(np.asarray(v), np.asarray(f), process=True).convex_hull
    if h.volume > 1e-9:
      hulls.append(h)
  hulls.sort(key=lambda h: -h.volume)
  return hulls[:max_hulls]


# --------------------------------------------------------------------------------------
# Writers
# --------------------------------------------------------------------------------------


def _write_obj(mesh, path: Path, texture_name: str | None) -> None:
  """Minimal OBJ writer: v / vt / f with per-vertex UVs (MuJoCo-compatible)."""
  uv = getattr(mesh.visual, "uv", None)
  has_uv = uv is not None and len(uv) == len(mesh.vertices)
  lines = [f"# generated by mjlab asset_pipeline; {len(mesh.faces)} faces"]
  if texture_name:
    lines.append(f"mtllib {path.stem}.mtl")
  for v in mesh.vertices:
    lines.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}")
  if has_uv:
    for t in uv:
      lines.append(f"vt {t[0]:.6f} {t[1]:.6f}")
  if texture_name:
    lines.append("usemtl mat")
  for f in mesh.faces:
    a, b, c = (int(i) + 1 for i in f)
    if has_uv:
      lines.append(f"f {a}/{a} {b}/{b} {c}/{c}")
    else:
      lines.append(f"f {a} {b} {c}")
  path.write_text("\n".join(lines) + "\n")
  if texture_name:
    (path.parent / f"{path.stem}.mtl").write_text(
      f"newmtl mat\nKd 1 1 1\nmap_Kd {texture_name}\n"
    )


def _write_texture(mesh, path: Path, tex_res: int, ao_image=None) -> bool:
  from PIL import Image

  img = _get_image(mesh)
  if img is None:
    return False
  img = img.convert("RGB")
  if ao_image is not None:
    # Poly Haven "arm" maps pack AO/Roughness/Metalness in R/G/B: use R only. A plain
    # AO map is greyscale, and its R channel is the same thing.
    ao = ao_image.convert("RGB").resize(img.size).split()[0]
    aof = np.asarray(ao).astype(np.float32) / 255.0
    aof = 0.35 + 0.65 * aof  # keep crevices readable, never fully black
    arr = np.asarray(img).astype(np.float32) * aof[..., None]
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
  if max(img.size) > tex_res:
    s = tex_res / max(img.size)
    img = img.resize((max(1, int(img.size[0] * s)), max(1, int(img.size[1] * s))),
                     Image.LANCZOS)
  img.save(path, optimize=True)
  mean = float(np.asarray(img).mean())
  if mean < 40:
    print(f"[texture] WARNING: albedo is very dark (mean {mean:.0f}/255) — check the AO/albedo inputs")
  return True


def _inertial_xml(mesh, density: float | None, mass: float | None, indent: str = "      "):
  """Compute mass / CoM / inertia from the (watertight or hull) mesh."""
  import trimesh

  m = mesh if mesh.is_watertight else mesh.convex_hull
  vol = float(m.volume)
  if mass is None:
    if density is None:
      raise ValueError("give --density or --mass")
    mass = density * vol
  m2 = m.copy()
  m2.density = mass / vol
  com = m2.center_mass
  I = m2.moment_inertia  # about CoM, in the mesh frame
  # Principal axes so we can emit diaginertia + quaternion (always valid for MuJoCo).
  w, V = np.linalg.eigh(I)
  if np.linalg.det(V) < 0:
    V[:, 0] *= -1
  q = trimesh.transformations.quaternion_from_matrix(
    np.block([[V, np.zeros((3, 1))], [np.zeros((1, 3)), np.ones((1, 1))]])
  )  # (w, x, y, z)
  w = np.maximum(w, 1e-9)
  xml = (f'{indent}<inertial pos="{com[0]:.5f} {com[1]:.5f} {com[2]:.5f}" '
         f'quat="{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}" mass="{mass:.5f}" '
         f'diaginertia="{w[0]:.3e} {w[1]:.3e} {w[2]:.3e}"/>')
  return xml, {"mass_kg": mass, "volume_m3": vol, "com": com.round(5).tolist(),
               "diaginertia": [float(x) for x in w]}


def package(
  mesh,
  name: str,
  out_dir: str | Path,
  collider: str = "coacd",
  max_hulls: int = 8,
  coacd_threshold: float = 0.05,
  max_tris: int = 20000,
  tex_res: int = 2048,
  density: float | None = None,
  mass: float | None = None,
  ao_image=None,
  source: str = "",
  license: str = "",
  friction: str = "1 0.03 0.003",
  extra_collider_meshes: list | None = None,
) -> dict[str, Any]:
  """Write the asset files and return a dict with the MJCF snippets and stats."""
  import trimesh

  out_dir = Path(out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)
  for old in out_dir.glob(f"{name}_*"):
    old.unlink()
  mesh = decimate(mesh, max_tris)
  tex_name = f"{name}_tex.png"
  has_tex = _write_texture(mesh, out_dir / tex_name, tex_res, ao_image)
  _write_obj(mesh, out_dir / f"{name}_vis.obj", tex_name if has_tex else None)

  hulls: list = []
  if collider == "coacd":
    hulls = decompose(mesh, max_hulls=max_hulls, threshold=coacd_threshold)
  elif collider == "hull":
    hulls = [mesh.convex_hull]
  elif collider in ("box", "none"):
    hulls = []
  else:
    raise ValueError(collider)
  if extra_collider_meshes:
    hulls = list(hulls) + list(extra_collider_meshes)
  for i, h in enumerate(hulls):
    _write_obj(h, out_dir / f"{name}_col_{i:02d}.obj", None)

  inertial_xml, inert = _inertial_xml(mesh, density, mass)
  ext = (mesh.bounds[1] - mesh.bounds[0])
  bb_c = mesh.bounds.mean(axis=0)

  asset_lines = [f'    <mesh name="{name}_vis" file="{name}_vis.obj"/>']
  if has_tex:
    asset_lines.insert(0, f'    <texture name="{name}_tex" type="2d" file="{tex_name}"/>')
    asset_lines.insert(1, f'    <material name="{name}_mat" texture="{name}_tex" '
                          'specular="0.15" shininess="0.15"/>')
  for i in range(len(hulls)):
    asset_lines.append(f'    <mesh name="{name}_col_{i:02d}" file="{name}_col_{i:02d}.obj"/>')

  geom_lines = [
    f'      <geom name="{name}_visual" type="mesh" mesh="{name}_vis"'
    + (f' material="{name}_mat"' if has_tex else "")
    + ' contype="0" conaffinity="0" group="2" mass="0"/>'
  ]
  for i in range(len(hulls)):
    geom_lines.append(
      f'      <geom name="{name}_col_{i:02d}" type="mesh" mesh="{name}_col_{i:02d}" '
      f'group="3" rgba="0.5 0.5 0.5 0.3" condim="3" friction="{friction}" '
      'contype="1" conaffinity="1" solref="0.01 1" mass="0"/>'
    )
  if collider == "box":
    geom_lines.append(
      f'      <geom name="{name}_col_00" type="box" pos="{bb_c[0]:.4f} {bb_c[1]:.4f} {bb_c[2]:.4f}" '
      f'size="{ext[0]/2:.4f} {ext[1]/2:.4f} {ext[2]/2:.4f}" group="3" rgba="0.5 0.5 0.5 0.3" '
      f'condim="3" friction="{friction}" contype="1" conaffinity="1" solref="0.01 1" mass="0"/>'
    )

  info = {
    "name": name,
    "source": source,
    "license": license,
    "visual_tris": int(len(mesh.faces)),
    "extent_m": ext.round(4).tolist(),
    "bounds": mesh.bounds.round(4).tolist(),
    "collider": collider,
    "n_colliders": len(hulls) if collider != "box" else 1,
    "collider_tris": [int(len(h.faces)) for h in hulls],
    "texture": tex_name if has_tex else None,
    **inert,
    "asset_xml": "\n".join(asset_lines),
    "geom_xml": "\n".join(geom_lines),
    "inertial_xml": inertial_xml,
    "files": sorted(p.name for p in out_dir.glob(f"{name}_*")),
    "bytes": sum(p.stat().st_size for p in out_dir.glob(f"{name}_*")),
  }
  (out_dir / f"{name}_package.json").write_text(json.dumps(info, indent=1))
  return info


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def _main(argv: list[str] | None = None) -> int:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  sub = ap.add_subparsers(dest="cmd", required=True)

  f = sub.add_parser("fetch", help="download a source asset into ~/assets_raw")
  f.add_argument("spec", help="polyhaven:<id> | gso:<name> | ycb:<name> | ambientcg:<id> | polyhaven-texture:<id>")
  f.add_argument("--res", default=None)

  i = sub.add_parser("inspect", help="print mesh stats")
  i.add_argument("mesh")
  i.add_argument("--geometry", default=None)

  p = sub.add_parser("package", help="write vis/tex/colliders + MJCF snippets")
  p.add_argument("mesh")
  p.add_argument("--geometry", default=None)
  p.add_argument("--name", required=True)
  p.add_argument("--out", required=True)
  p.add_argument("--scale", type=float, default=None)
  p.add_argument("--target-extent", type=float, nargs=3, default=None,
                 help="desired x y z extent in metres (0 = unconstrained); uniform scale")
  p.add_argument("--rotate", type=float, nargs=3, default=(0, 0, 0), help="deg about x y z")
  p.add_argument("--up", default="z", choices=["z", "y"])
  p.add_argument("--origin", default="centroid", choices=["centroid", "bbox", "bottom", "keep"])
  p.add_argument("--translate", type=float, nargs=3, default=(0, 0, 0))
  p.add_argument("--collider", default="coacd", choices=["coacd", "hull", "box", "none"])
  p.add_argument("--max-hulls", type=int, default=8)
  p.add_argument("--coacd-threshold", type=float, default=0.05)
  p.add_argument("--max-tris", type=int, default=20000)
  p.add_argument("--tex-res", type=int, default=2048)
  p.add_argument("--density", type=float, default=None, help="kg/m^3")
  p.add_argument("--mass", type=float, default=None, help="kg (overrides density)")
  p.add_argument("--ao", default=None, help="AO image to bake into albedo")
  p.add_argument("--source", default="")
  p.add_argument("--license", default="")
  p.add_argument("--friction", default="1 0.03 0.003")

  a = ap.parse_args(argv)
  if a.cmd == "fetch":
    print(fetch(a.spec, a.res))
    return 0
  if a.cmd == "inspect":
    m = load_mesh(a.mesh, a.geometry)
    print(json.dumps(inspect_mesh(m), indent=1))
    return 0
  if a.cmd == "package":
    from PIL import Image

    m = load_mesh(a.mesh, a.geometry)
    print("[package] input:", json.dumps(inspect_mesh(m)))
    m = normalise(m, scale=a.scale, target_extent=a.target_extent, rotate_deg=tuple(a.rotate),
                  up=a.up, origin=a.origin, translate=tuple(a.translate))
    ao = Image.open(a.ao) if a.ao else None
    info = package(m, a.name, a.out, collider=a.collider, max_hulls=a.max_hulls,
                   coacd_threshold=a.coacd_threshold, max_tris=a.max_tris, tex_res=a.tex_res,
                   density=a.density, mass=a.mass, ao_image=ao, source=a.source,
                   license=a.license, friction=a.friction)
    print("[package] extent(m):", info["extent_m"], "mass(kg):", round(info["mass_kg"], 4),
          "colliders:", info["n_colliders"], "tris:", info["visual_tris"],
          "bytes:", info["bytes"])
    print("\n  <asset>\n" + info["asset_xml"] + "\n  </asset>\n")
    print(info["inertial_xml"])
    print(info["geom_xml"])
    return 0
  return 1


if __name__ == "__main__":
  sys.exit(_main())
