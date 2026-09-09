# plug — provenance (CL-V2, W1-c, 2026-09-09)

Task: `Mjlab-Axial-Extract-Franka`.

## What it is

A **CEE 7/4 "Schuko" mains plug seated in a surface-mount power-outlet box**, face up —
the desk/floor-box geometry where a plug really is pulled straight up out of its socket.

| part | real dimensions | modelled as |
|---|---|---|
| socket outlet box | 80 x 80 mm faceplate, 100 mm deep back box, 50 mm round recess 5 mm deep, two 4.8 mm pin holes on a 19 mm pitch | one watertight trimesh solid (CSG), 402 tris |
| plug moulding | round, 50 mm across, 34 mm thick (Schuko bodies are 30-36 mm), sitting 5 mm into the recess so 29 mm stands proud | lathe (`trimesh.creation.revolve`), 480 tris |
| pins | 2 x 4.8 mm dia x 19 mm, 19 mm pitch | two visual cylinder geoms, brass rgba |

50 mm across is inside the 60 mm pinch limit (aperture 80 mm) and round, so the pinch is
orientation-free — which matters because the teacher's wrist yaw is not tightly held.

## Sources / licences

| file | source | licence |
|---|---|---|
| `assets/socket_tex.png`, `assets/plug_tex.png` | ambientCG texture `Plastic010`, re-tinted (warm white / black), 512^2 | CC0 |
| `assets/socket_vis.obj`, `assets/plug_vis.obj` | built with trimesh + manifold3d by `scratchpad/build/build_plug.py`; box- and cylinder-projected UVs | CC0 (own work) |

Masses: socket 0.30 kg (mocap, so kinematic — the number is cosmetic); plug moulding
0.070 kg (ABS, 1050 kg/m^3) + 0.006 kg of pins = 0.076 kg on the sliding `handle` body
(the primitive's was 0.092 kg).

## What was deliberately NOT changed

* the four socket collider boxes `plug_body`, `plug_wall_{nx,py,ny}` — same 80 x 80 mm
  outer footprint, same 32 mm square bore, same +-0.05 m z extent. So:
  - `workspace.MECHANISM_DROP_BELOW_MOUNT["plug"]` stays **0.050** (verified by
    `tests/test_workspace_placement.py::test_mechanism_drops_are_swept_over_the_joint_range`,
    which re-sweeps the joint range from this XML);
  - the `ee_plug_collision` contact sensor's geom pattern `plug_body|plug_wall_.*`
    (`axial_extract_env_cfg.py`) still matches exactly the static geoms and nothing else;
  - `franka_axial_extract_env_cfg`'s `reset_plug_position` band is untouched.
* `object_site` / `base_site` stay at `0 0 0.062`, so `HOVER_LEAD`, `SEAT_TOL`,
  `FLOOR_MIN_Z` and every other `classical/axial_extract.py` constant are unchanged.
* `plug_slide`: `range="0 0.12"`, `damping="1.0"`, **`frictionloss="4.0"`** — the ~4 N
  breakaway is the task premise (a real Schuko pull-out force is 40-100 N, far past what
  the Franka's position servo delivers, so the calibrated 4 N is kept deliberately).
* `<compiler angle="radian" .../>` is written explicitly.

## Changed

* `plug_shaft`: cylinder r 0.014 / half-height 0.05 (a 100 mm barrel) -> r 0.012 /
  half-height 0.0095 at z=+0.0355, i.e. the bounding cylinder of the two real pins. It is
  `contype=2` so it only ever collides with the robot, it lives inside the bore, and the
  swept low point is still the socket walls — so nothing downstream moves. The old
  100 mm barrel would have hung 80 mm of invisible collider below a plug whose visible
  pins are 19 mm once extracted.
* `handle` (the pinched geom): cylinder r 0.02 / half 0.012 -> r 0.025 / half 0.017, i.e.
  40 mm x 24 mm -> the real 50 mm x 34 mm moulding. Still centred at +0.062.

## Known visual/collision mismatches (both unreachable)

* the 50 mm round recess vs the 32 mm square bore: the collider fills the four corners of
  the recess ring (r in [0.016, 0.025], z in [0.045, 0.05]). The plug moulding occupies
  that space whenever it is seated, and the fingers close from OUTSIDE r = 0.025, so
  nothing can touch it.
* the faceplate's pin holes are 5 mm while the bore is 32 mm square: invisible void under
  a visually solid faceplate, permanently covered by the moulding.
