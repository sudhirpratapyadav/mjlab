# cube — provenance

| | |
|---|---|
| **Asset** | YCB `077_rubiks_cube`, `google_16k` textured scan |
| **Source** | YCB Object and Model Set — https://www.ycbbenchmarks.com/ (mirror: `https://ycb-benchmarks.s3.amazonaws.com/data/google/077_rubiks_cube_google_16k.tgz`) |
| **License** | CC-BY 4.0 |
| **Raw download** | `~/assets_raw/ycb/077_rubiks_cube/google_16k/textured.obj` (outside the repo) |
| **Used by** | Lift-Cube, Stack-Cube (the stacked object), Place-In-Container, Throw-To-Bin, Cage-Drag |

## Real dimensions vs the spec

The scan arrives at real scale but parked at an arbitrary yaw: its axis-aligned
bounding box reads 75.7 x 76.1 x 57.9 mm. A yaw search
(`~/cl_v2_work/W1-a/find_yaw.py`, minimum xy bbox area) puts the true cube at
**63.00 deg**, giving **59.0 x 58.1 x 57.9 mm** — the 57 mm standard 3x3 cube, so the
scan needs no unit correction, only de-rotation.

## What was changed and why

| Step | Value | Reason |
|---|---|---|
| Rotate about z | +63.00 deg | de-rotate the scan so the cube's faces are axis-aligned; every downstream half-extent constant assumes it |
| Uniform scale | 0.7797x (x extent 59.0 mm -> 46.0 mm) | **the one substantive change.** The scanned 57 mm cube is wider than Cage-Drag's `aperture_min = 0.055` latch, which would make that task unwinnable by construction. 46 mm is a real "mini 3x3" product size, is under the 60 mm pinch ceiling the 80 mm Franka gripper imposes (`verify_task` enforces this), and leaves 9 mm of aperture margin. Final extents **46.0 x 45.3 x 45.1 mm** |
| Origin | bounding-box centre | every constant that consumes this asset (`z=(h,h)` spawn heights, `stack_height`, `HALF_WIDTH`, `OBJ_CENTER_Z`) is expressed against the geometric centre of the collision box |
| Decimation | 16384 -> 12000 tris | budget is 20k; Blender 4.2 collapse decimation |
| Texture | 4096^2 -> 1024^2 PNG | budget; MuJoCo 3.11 takes PNG only. The atlas is mostly unused black, so the pipeline's "albedo is very dark" warning is a false positive — the stickers are fully saturated |
| Collider | one `box`, half-extents 0.0230 x 0.0226 x 0.0226 | a cube collides as a box: exact, and the cheapest primitive pair in mujoco_warp |
| Mass | 0.050 kg | published mass of the scanned 57 mm cube is 0.094 kg -> density 508 kg/m^3 -> 0.0494 kg at 46 mm. (Coincides with the primitive's 0.05 kg, so mass-dependent dynamics keep their cl25 baseline.) |

Friction / `condim` / `solref` / `contype` / `conaffinity` were carried over verbatim
from the primitive geom.

## Size budget

`xmls/assets/` = 1.18 MB (limit 5 MB).
