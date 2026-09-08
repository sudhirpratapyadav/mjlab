# CODE_MAP.md — mjlab plumbing contract for the Franka asset-replacement work

HEAD at survey time: `1127d12`. All paths relative to `/ihub/homedirs/svs_ald/sudhir/mjlab`.

## 1. SCENE ASSEMBLY — how an object XML reaches the compiled model

### Chain
1. `<obj>_constants.py::get_<obj>_spec()` calls `mujoco.MjSpec.from_file(str(<OBJ>_XML))` — e.g. `src/mjlab/asset_zoo/objects/free/cube/cube_constants.py:26`, `src/mjlab/asset_zoo/objects/articulated/drawer/drawer_constants.py:30`.
2. `get_<obj>_cfg()` returns `EntityCfg(spec_fn=get_<obj>_spec, init_state=...)` (`cube_constants.py:50`, `drawer_constants.py:70`).
3. `Entity.__init__` wraps it: `self._spec = auto_wrap_fixed_base_mocap(cfg.spec_fn)()` (`src/mjlab/entity/entity.py:127`).
4. `Scene._add_entities` attaches each entity spec into the scene spec with a prefix:
   `frame = self._spec.worldbody.add_frame(); self._spec.attach(ent.spec, prefix=f"{ent_name}/", frame=frame)` (`src/mjlab/scene/scene.py:156-157`).
5. Terrain attaches with `prefix=""` (`scene/scene.py:165-166`). Scene root spec is `mujoco.MjSpec.from_file(src/mjlab/scene/scene.xml)` (`scene/scene.py:36`). `Scene.compile()` → `self._spec.compile()` (`scene/scene.py:46`).

### `auto_wrap_fixed_base_mocap` (`src/mjlab/utils/spec.py:9-38`)
If a spec has no freejoint and its root body is not already `mocap`, it builds a **fresh `mujoco.MjSpec()`**, adds body `mocap_base` (mocap=True), and `wrapper_spec.attach(child=original_spec, prefix="", frame=...)`. Verified: assets survive this wrap — `Entity(get_franka_robot_cfg()).spec.assets` has 68 entries, `root_body.name == "mocap_base"`, `is_mocap == True`.
13 object XMLs already declare `mocap="true"` and skip the wrap: `articulated/{lid,valve,flap,button,drawer,window,plug,door,switch,lever}`, `free/{container,ledge,wall}`.

### Mesh / texture path resolution — the precedent (Franka)
`src/mjlab/asset_zoo/robots/franka_emika_panda/franka_constants.py:34-37`:
```python
def get_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(FRANKA_XML))
    spec.assets = get_assets(spec.meshdir)   # meshdir == "assets"
    return spec
```
`get_assets` (`franka_constants.py:21-31`) uses `mjlab.utils.os.update_assets` (`src/mjlab/utils/os.py:7-35`), which globs a directory and keys each file as `f"{meshdir}/{f.name}"` (so `"assets/link0.stl"`), plus `assets["hand.xml"]` for the `<include>`d sub-file.

`panda.xml:2` is `<compiler angle="radian" meshdir="assets" autolimits="true"/>`; meshes live in `.../xmls/assets/*.{stl,obj}`.

**Two independently working mechanisms, both verified empirically:**

- **A. In-memory assets dict.** `spec.assets` (68 entries) is merged into the parent on `parent.attach(child, prefix=..., frame=...)`; parent went 0 → 68 assets, then compiled: `nmesh=67, ngeom=80, nmeshvert=396791`, max verts on one mesh 44200.
- **B. No assets dict at all.** `mujoco.MjSpec.from_file(panda.xml)` with `len(spec.assets)==0`, attached into the scene spec, still compiled with `nmesh=67`. MuJoCo preserves the child's `modelfiledir` (`'.../franka_emika_panda/xmls/'`) through `attach`, and resolves `meshdir`-relative `file=` from disk.

**Practical rule for downstream agents:** you can add `<asset><mesh file="assets/foo.obj"/></asset>` + `<compiler meshdir="assets"/>` to an object XML and just `MjSpec.from_file` it — it will compile inside the scene. Setting `spec.assets = update_assets({}, xml_dir/"assets", spec.meshdir)` (the Franka pattern) is still recommended: it is required for `Scene.to_zip` / `Entity.to_zip` (`scene/scene.py:48-61`, `entity/entity.py:353-356`) and for any path where the model is shipped away from the source tree. **Prefer it — mirror `franka_constants.get_assets` exactly.**

Notes:
- `texturedir` on the Franka spec is `''`; textures would need their own dir (or reuse `meshdir`, keying assets under that prefix). `update_assets` takes any prefix string.
- `spec.copy_during_attach` is **write-only** in this MuJoCo build (`ValueError: copy_during_attach can only be set.`). Nothing in mjlab sets it; leave it alone.
- `Scene.to_zip` docstring warns the produced zip may need `assetdir="assets"` added manually (`scene/scene.py:48-59`).
- Per-entity spec editors exist and run before attach: `EntityCfg.lights/cameras/textures/materials/collisions` → `Entity._apply_spec_editors` (`entity/entity.py:142-153`), types in `src/mjlab/utils/spec_config.py` (`TextureCfg:63`, `MaterialCfg` ~:106, `LightCfg:241`). **No manipulation task currently uses any of them.**
- `SceneCfg.spec_fn` (`scene/scene.py:24`, applied at `:42-43`) is a last-chance hook to mutate the assembled scene spec. Also unused by manipulation.
- Entity name accessors strip the attach prefix: `joint_names/body_names/geom_names/site_names` all do `name.split("/")[-1]` (`entity/entity.py:227-249`). **So keeping body/site/joint names identical keeps every cfg/command/reward lookup working unchanged.**

MuJoCo 3.11.1, mujoco_warp 3.11.0.

## 2. SIMULATION BACKEND

- The stepper is **mujoco_warp (MJWarp) on GPU**, not plain MuJoCo. `src/mjlab/sim/sim.py:101` `class Simulation: """GPU-accelerated MuJoCo simulation powered by MJWarp."""`; `mjwarp.put_model` (`:120`), `mjwarp.put_data(..., nworld=num_envs, nconmax, njmax)` (`:129-135`), `mjwarp.step` (`:209`), with CUDA-graph capture when the device is CUDA + mempool (`:140-157`).
- `mujoco.MjModel` is still compiled and kept (`sim.py:113-116`) and is what the offscreen renderer and the classical teachers' internal IK use.

### Supported collision type-pairs (installed mujoco_warp 3.11.0)
`.venv/lib/python3.13/site-packages/mujoco_warp/_src/collision_driver.py:47-81` — `MJ_COLLISION_TABLE`. Every pair below is supported; `PRIMITIVE` = analytic kernel, `CONVEX` = GJK/EPA on convex hulls.

| pair | kind | line |
|---|---|---|
| plane–{sphere,capsule,ellipsoid,cylinder,box,**mesh**} | PRIMITIVE | 48-53 |
| hfield–{sphere,capsule,ellipsoid,cylinder,box,**mesh**} | CONVEX | 54-59 |
| sphere–{sphere,capsule,cylinder,box} | PRIMITIVE | 60-64 |
| sphere–ellipsoid, **sphere–mesh** | CONVEX | 62, 65 |
| capsule–{capsule,box} | PRIMITIVE | 66, 69 |
| capsule–{ellipsoid,cylinder,**mesh**} | CONVEX | 67,68,70 |
| ellipsoid–{ellipsoid,cylinder,box,**mesh**} | CONVEX | 71-74 |
| cylinder–{cylinder,box,**mesh**} | CONVEX | 75-77 |
| box–box | CONVEX (→ PRIMITIVE if `DisableBit.NATIVECCD`) | 78, 870 |
| **box–mesh**, **mesh–mesh** | CONVEX | 79-80 |

**Every mesh pairing needed for this work (mesh–plane, mesh–box, mesh–mesh, mesh–sphere, mesh–cylinder, mesh–capsule, mesh–ellipsoid) is supported.** Dispatch: `_narrowphase` (`collision_driver.py:872-884`) splits the table into `convex_pairs` / `primitive_pairs`.

**Mesh collider constraints**
- Collision is **convex only**: mesh geoms go through GJK/EPA on the hull. Concave shapes must be decomposed into multiple convex mesh geoms (the coacd use case) or approximated with primitives. MuJoCo itself computes the hull at compile time.
- No hard vertex cap found in the installed source; the Franka's largest mesh is 44200 verts and compiles/steps fine. Contact points per pair are capped by `MJ_MAXCONPAIR = mujoco.mjMAXCONPAIR` (`_src/types.py:27`, used `collision_convex.py:57-58, 426-430`) — a **very** high-poly hull does not create unbounded contacts, but it does cost GJK iterations. Recommended practice: keep collision hulls low-poly (visual mesh separate, `class="visual"` `contype=0 conaffinity=0`, exactly as `panda.xml:24-27` does) and use a decimated/convex-decomposed collision mesh.
- `condim`, `solref`, `solimp`, `friction`, `margin`, `gap`, `contype`/`conaffinity` are ordinary MjModel fields consumed by MJWarp's constraint solver — they are honored the same for mesh geoms as for primitives. All existing object XMLs already set `condim="3"` (ellipsoid uses `condim="6"`), `friction`, `solref="0.01 1"`, `contype="1" conaffinity="1"` — **carry these onto the mesh geoms verbatim.**

### nconmax / njmax
- `SimulationCfg.nconmax` / `njmax` (`src/mjlab/sim/sim.py:81-90`), forwarded to `mjwarp.put_data` (`sim.py:133-134`). `None` ⇒ MJWarp heuristics `_default_nconmax` / `_default_njmax` (`mujoco_warp/_src/io.py:1282-1305`), which are tiny (base 45 / 53 contacts).
- Per-task values (all in `src/mjlab/tasks/manipulation/*_env_cfg.py`):
  - **Free-object / graspable tasks: `nconmax=200, njmax=1000`** — lift(`lift_object_env_cfg.py:312-313`), stack(`:230-231`), push_cube/cuboid/disc, reach, place_in_container, reorient, tool_pull, drag_pull, strike_slide, cage_drag, topple_block, edge_grasp, pivot_lift.
  - **Articulated-mechanism tasks: `nconmax=60, njmax=650`** — door(`open_door_env_cfg.py:349-350`), drawer, button, lever, valve, switch, window, lid, flap, plug/axial_extract.
- **⚠️ The 60/650 budget is tight.** Replacing a 3-box mechanism with textured meshes (especially decomposed convex hulls) will add geoms and contacts. If MJWarp reports ncon overflow, raise `nconmax` for that task's `SimulationCfg`; contacts beyond `nconmax` are silently skipped (`collision_driver.py` docstring ~:900).
- Other sim settings: `timestep=0.005` everywhere; `decimation=4` ⇒ control dt 0.02 s (`lift_object_env_cfg.py:322`). Gravity is **disabled** (`(0,0,0)`) for door, drawer, button, lever, valve, window, flap, plug; **enabled** for lid and switch (see comments at `open_lid_env_cfg.py:350`, `flip_switch_env_cfg.py:350`).

## 3. TASK TABLE — 29 registered Franka tasks

Registry: `src/mjlab/tasks/manipulation/config/franka/__init__.py`. Env cfg fns: `src/mjlab/tasks/manipulation/config/franka/env_cfgs.py` (line numbers in col 2). Commands: `src/mjlab/tasks/manipulation/mdp/commands.py` (`C:` line refs). Episode length = **train** value (`play` ⇒ 1e9; several tasks set 5.0 in `test`).

**All success predicates are latched** via `episode_success = torch.maximum(episode_success, at_goal)` and reset per-env in `_resample_command`; `compute_success()` returns the *instantaneous* predicate. Cage-Drag additionally latches a **minimum** (`min_aperture`, `C:1952`).

Two placement mechanisms only:
- **CMD** — the command term's `_resample_command` samples/writes the pose (`_spawn_object`, `C:1433`, or the task's own sampler).
- **EVT** — a `reset_<asset>_position` EventTermCfg using `mdp.reset_root_state_uniform`, whose `pose_range` the Franka cfg overrides. Used by **all 10 articulated mechanisms only**.

| # | task id | env cfg fn (env_cfgs.py) | asset XML(s) | names referenced | placement | success predicate / threshold / latched | ep len (s) | geometry constants that move with the asset |
|---|---|---|---|---|---|---|---|---|
|1|Mjlab-Lift-Cube-Franka|`franka_lift_cube_env_cfg` :166|`objects/free/cube/xmls/cube.xml` + mocap goal (`cube_constants.py:31`)|body `cube`, joint `cube_joint`, geom `cube_geom`, site `object_site`; robot site `gripper`; geoms `left_finger_pad|right_finger_pad`|CMD `LiftingCommand._resample_command` `C:83`, ranges :186-201|`LiftingCommand.compute_success` `C:79`; `success_threshold=0.05` m `C:169`; latched `C:58-61`|20 (test 5)|spawn `z=(0.02,0.05)` = half-height :190; cube half-extent 0.02; mocap-goal geom `size=[0.02]*3` `cube_constants.py:38`|
|2|Mjlab-Lift-Cylinder-Franka|`franka_lift_cylinder_env_cfg` :254|`free/cylinder/xmls/cylinder.xml`|body `cylinder`, joint `cylinder_joint`, site `object_site`|CMD, :283-296|same, 0.05 m|20 (5)|cylinder `size="0.02 0.02"` (r, half-h); goal geom `cylinder_constants.py:38`|
|3|Mjlab-Lift-Sphere-Franka|`franka_lift_sphere_env_cfg` :440 → `_franka_lift_object_env_cfg` :360|`free/sphere/xmls/sphere.xml`|body `sphere`, joint `sphere_joint`, site `object_site`|CMD, :383-391|same, 0.05 m|20 (5)|r=0.022; goal `size=[0.022,0,0]`|
|4|Mjlab-Lift-Ellipsoid-Franka|`franka_lift_ellipsoid_env_cfg` :447|`free/ellipsoid/xmls/ellipsoid.xml`|body `ellipsoid`, joint `ellipsoid_joint`, site `object_site`|CMD|same, 0.05 m|20 (5)|semi-axes 0.035/0.018/0.018; `condim="6"`|
|5|Mjlab-Stack-Cube-Franka|`franka_stack_cube_env_cfg` :454|cube.xml (`object`) + cuboid.xml (`base`)|entities named `object`,`base`; sites `object_site` on each|CMD `StackingCommand._resample_command` `C:1078`; ranges :481-495|`StackingCommand` `C:1076`; xy `success_threshold=0.03` **AND** `height_threshold=0.02` `C:1129-1130`; latched `C:1064`|20 (5)|**`stack_height=0.035` :480 = cube half-h 0.02 + cuboid half-h 0.015**; spawn z 0.02 / 0.015|
|6|Mjlab-Peg-Insertion-Franka|`franka_peg_insertion_env_cfg` :536|`free/peg_in_hole/xmls/peg.xml` + `hole_board.xml`|`object`=peg (`peg_joint`, `object_site` at `0 0 -0.05`), `base`=hole_board (`hole_board_joint`, walls `hole_wall_{px,nx,py,ny}`)|CMD|Stacking; `success_threshold=0.015`, `height_threshold=0.03` :568-569|20 (5)|**`stack_height=0.01` :567**; peg spawn `z=(0.05,0.05)` = half-length :577; peg 0.012×0.012×0.05; hole gap 0.03 sq, board half-h 0.015|
|7|Mjlab-Reach-Target-Franka|`franka_reach_target_env_cfg` :625|none (mocap goal only)|robot site `gripper`|CMD `ReachingCommand` `C:895` (target only)|`ReachingCommand.compute_success` `C:939`; 0.05 m `C:985`; latched `C:934`|20 (5)|none (asset-free)|
|8|Mjlab-Open-Door-Franka|`franka_open_door_env_cfg` :673|`articulated/door/xmls/door.xml`|bodies `door_base`(mocap),`handle`; joint `door_hinge`; geoms `door_body`,`door_panel`,`handle`; sites `base_site`,`object_site`|**EVT** `reset_door_position` :714-718 → x(0.48,0.52) y(-0.30,-0.20) z=`_mech_z("door")`|`OpenDoorCommand` `C:276`; target 1.5708 rad (90°) `C:289`; `success_threshold=0.1` rad `C:360`; latched `C:253`|3|door.xml has **no `<compiler angle="radian"/>` ⇒ `range="0 90"` is DEGREES**; handle at `(-0.04, 0.25, 0)` from base (mount is shifted −y for this, :711-713); `MECHANISM_DROP_BELOW_MOUNT["door"]=0.800` (`workspace.py:154`); goal marker geom `size=[0.01,0.01,0.08]` (`door_constants.py:44`)|
|9|Mjlab-Open-Drawer-Franka|`franka_open_drawer_env_cfg` :770|`articulated/drawer/xmls/drawer.xml`|bodies `drawer_base`,`handle`; joint `drawer_slide` (range −0.25..0); geoms `drawer_body`,`drawer_panel`,`handle`; sites `base_site`,`object_site`|**EVT** `reset_drawer_position` :807-811 → x(0.46,0.56) y(±0.10) z=`_mech_z("drawer")`|`OpenDrawerCommand` `C:455`; target −0.25 m `C:469`; `success_threshold=0.02` m `C:521`; latched `C:433`|3|handle offset `(-0.04, 0, 0)` (cfg comment :804-806); slide stroke 0.25; drop 0.300 (`workspace.py:155`); goal geom `[0.01,0.08,0.01]`|
|10|Mjlab-Push-Button-Franka|`franka_push_button_env_cfg` :863|`articulated/button/xmls/button.xml`|bodies `button_base`,`handle`; joint `button_slide` (−0.05..0); geoms `button_body`,`button_panel`,`handle`; sites `base_site`(0,0,0.1),`object_site`(0,0,0.1)|**EVT** `reset_button_position` :900-903 → x(0.44,0.48) y(±0.10) z=`_mech_z("button")`|`PushButtonCommand` `C:616`; target −0.05 m `C:630`; `success_threshold=0.02` m `C:682`; latched `C:594`|3|cap sits `+0.1` above mount; drop 0.030; goal geom `[0.03,0.03,0.005]`|
|11|Mjlab-Push-Cuboid-Franka|`franka_push_cuboid_env_cfg` :956|`free/cuboid/xmls/cuboid.xml`|body `cuboid`, joint `cuboid_joint`, site `object_site`|CMD `PushingCommand._resample_command` `C:767`; ranges :983-995|`PushingCommand.compute_success` `C:763`; `success_threshold=0.02` (`push_cuboid_env_cfg.py:136`); latched `C:745`|3|spawn `z=(0.015,0.015)` = half-h :986; `goal_z_height=0.015` (`push_cuboid_env_cfg.py:137`); cuboid 0.04×0.04×0.015|
|12|Mjlab-Push-Disc-Franka|`franka_push_disc_env_cfg` :1041|`free/disc/xmls/disc.xml`|body `disc`, joint `disc_joint`, site `object_site`|CMD, :1066-1077|Pushing; default `success_threshold=0.05` `C:856`; latched|20 (test 5)|spawn `z=(0.05,0.05)` :1069; `goal_z_height=0.03` (`push_disc_env_cfg.py:136`); disc `size="0.02 0.02"`|
|13|Mjlab-Turn-Lever-Franka|`franka_turn_lever_env_cfg` :1191|`articulated/lever/xmls/lever.xml`|joint `lever_hinge`; geoms `lever_body`,`lever_hub`,`handle`; sites `base_site`(-0.05,0.12,0),`object_site`|**EVT** `reset_lever_position` :1209-1213 x(0.50,0.56) y(−0.20,−0.04)|`_ArticulationJointCommand.compute_success` `C:1245` (joint-value, `directional=True` `C:1294`); target **−1.5707963 rad**, thr **0.15 rad** `C:1323-1325`|3|`goal_marker_offset=(0,-0.12,-0.12)` `C:1326`; handle at `(-0.05,0.07,0)` size `0.012 0.07 0.012`; drop 0.150|
|14|Mjlab-Rotate-Valve-Franka|`franka_rotate_valve_env_cfg` :1218|`articulated/valve/xmls/valve.xml`|joint `valve_hinge`; geoms `valve_hub`,`handle`,`valve_spoke_b`; sites `base_site`(-0.04,0.09,0),`object_site`|**EVT** :1235-1239 x(0.49,0.55) y(−0.17,−0.01)|target **4.712389 rad (270°)**, thr **0.2 rad** `C:1346-1348`|**8.0** (:1241)|marker offset `(0,-0.09,0.09)`; spoke half-length 0.055, hub r 0.02; drop 0.150|
|15|Mjlab-Flip-Switch-Franka|`franka_flip_switch_env_cfg` :1246|`articulated/switch/xmls/switch.xml`|joint `switch_hinge`; geoms `switch_body`,`handle`,`switch_weight`; sites `base_site`(0,0,0.062),`object_site`|**EVT** :1263-1267 x(0.46,0.52) y(±0.08)|target **+0.5235988 (30°)** from `init_value=-0.7853982 (−45°)`, thr **0.15 rad** `C:1368-1370`|3|marker offset `(0,0,0.04)`; toggle 0.012×0.012×0.03 at z 0.026; **gravity ON** — the detent is a weighted inverted pendulum, so mass/inertia of the new mesh matters; drop 0.060|
|16|Mjlab-Slide-Window-Franka|`franka_slide_window_env_cfg` :1272|`articulated/window/xmls/window.xml`|joint `window_slide`; geoms `window_body`,`window_pane`,`handle`; sites `base_site`(-0.04,-0.10,0),`object_site`|**EVT** :1289-1293 x(0.50,0.56) y(0.05,0.15)|target **0.22 m**, thr **0.03 m** `C:1389-1391`|3|marker offset `(0,0.22,0)`; pane 0.01×0.15×0.2; handle 0.012×0.012×0.07; drop 0.220|
|17|Mjlab-Open-Lid-Franka|`franka_open_lid_env_cfg` :1298|`articulated/lid/xmls/lid.xml`|joint `lid_hinge` (axis 0 1 0, pos `0.10 0 0.04`); geoms `lid_body`,`lid_wall_{far,near,left,right}`,`lid_panel`,`handle`; sites `base_site`(-0.09,0,0.05),`object_site`|**EVT** :1320-1324 x(0.52,0.59) y(±0.08)|target **−1.308997 (−75°)**, thr **0.2 rad** `C:1411-1413`|3|marker offset `(0.10,0,0.12)`; **`MECHANISM_DROP_BELOW_MOUNT["lid"]=0.157` is a SWEPT value** — see `tests/test_workspace_placement.py:134`; **gravity ON**|
|18|Mjlab-Place-In-Container-Franka|`franka_place_in_container_env_cfg` :1329|`free/cube/xmls/cube.xml` + `free/container/xmls/container.xml`|`cube`(`cube_joint`,`object_site`); `container_base`(mocap), geoms `container_floor`,`container_wall_{px,nx,py,ny}`, sites `object_site`(0,0,0.02),`base_site`|CMD `PlaceInContainerCommand._resample_command` `C:1547+`; container written per-env `C:1560-1590`; ranges :1379-1394|`PlaceInContainerCommand.compute_success` `C:1547`; containment: lateral <0.055, below `rim_height=0.05`, above `-floor_tolerance=0.04`, `settle_speed<0.12` `C:1604-1614`; latched `C:1531`|20|**`lateral_tolerance=0.055` derives from walls at ±0.062**; goal read from `container.spec.sites["object_site"].pos` `C:1578-1582` — the site offset is read from the spec, so keep it; bin inner span ~0.124, rim 0.07 above floor|
|19|Mjlab-Reorient-Object-Franka|`franka_reorient_object_env_cfg` :1401|`free/cylinder/xmls/cylinder.xml`|body `cylinder`, `object_site`|CMD `ReorientObjectCommand` `C:1627`; spawn :1447-1456 (roll=π/2, lying down)|`ReorientObjectCommand.compute_success` `C:1718`; `angle_threshold=0.35` rad **AND** `max_drift=0.18` m `C:1772-1773`; latched `C:1703`|20|`body_axis=(0,0,1)`, `target_axis=(0,0,1)` `C:1764-1770`; spawn z 0.025; **`_pad = 0.05` :1445 is the cylinder half-length** used to inset the spawn box|
|20|Mjlab-Tool-Pull-Franka|`franka_tool_pull_env_cfg` :1463|`free/puck/xmls/puck.xml` + `free/stick/xmls/stick.xml`|`puck`(`puck_joint`,`object_site`); `stick`(`stick_joint`, geoms `stick_shaft`,`stick_hook`, sites `object_site` at `-0.09 0 0`, `tool_tip_site` at `0.12 0.02 0`)|CMD `ToolPullCommand` `C:1780`; ranges :1526-1540|`ToolPullCommand.compute_success` `C:1859`; `success_threshold=0.07` m `C:1905`; stage-1 `grasp_threshold=0.06` latched separately `C:1836-1838`; success latched `C:1844`|**12.0** (:1542)|**`goal_offset=(0.42, 0.0, 0.012)` `C:1897` — z MUST equal the puck half-thickness (0.012)**; puck r=0.035 h=0.012; stick shaft 0.13×0.011×0.011, hook at `(0.12,0.035,0)`; spawn z 0.012 both; **`+0.09` x offset in `tool_spawn_range` :1537 is the stick `object_site` offset**|
|21|Mjlab-Drag-Pull-Franka|`franka_drag_pull_env_cfg` :1558|`free/cuboid/xmls/cuboid.xml`|body `cuboid`, `object_site`|CMD Pushing; ranges :1583-1596|Pushing; `success_threshold=0.03` (`drag_pull_env_cfg.py:148`); latched|3|**`_ENGAGE_INSET = 0.04` :1578 is literally the cuboid x half-extent**; spawn z 0.015; `goal_z_height=0.015`|
|22|Mjlab-Strike-Slide-Franka|`franka_strike_slide_env_cfg` :1604|`free/puck/xmls/puck.xml`|body `puck`, `object_site`|CMD Pushing; ranges :1623-1636|Pushing; `success_threshold=0.08` (`strike_slide_env_cfg.py:147`); latched|4|spawn z 0.012 = puck half-thickness; `goal_z_height=0.012`; goal band x(0.88,1.05) is deliberately out of reach; puck friction governs slide distance|
|23|Mjlab-Cage-Drag-Franka|`franka_cage_drag_env_cfg` :1641|`free/cube/xmls/cube.xml`|body `cube`, `object_site`; robot joints `finger_joint1/2`|CMD `CageDragCommand` `C:1922`; ranges :1660-1670|`CageDragCommand.compute_success` `C:1976`: goal <`success_threshold=0.03` **AND** `min_aperture > aperture_min=0.055` over whole episode `C:1989-1994`; both latches reset `C:1982-1984`|4|**`aperture_min=0.055` is calibrated against the 0.04-wide cube and the 0.08 max Franka aperture** — if the caged object gets wider this must be re-derived; spawn z 0.02|
|24|Mjlab-Topple-Block-Franka|`franka_topple_block_env_cfg` :1676|`free/block/xmls/block.xml`|body `block`, joint `block_joint`, geom `block_geom`, `object_site`|CMD `ReorientObjectCommand`; spawn :1710-1716|Reorient; `angle_threshold=0.35` rad, `max_drift=0.30` (`topple_block_env_cfg.py:147-149`); `body_axis=(1,0,0)`, `symmetric_axis=True` (:146-147); latched|4|**`_half = 0.07` (yaw-swept half-extent), `_travel = 0.09` (~half-height), spawn `z=(0.09,0.09)` :1698-1714** — all three are block half-extents `0.05 0.07 0.09`; `assert _poke_far + _half + _travel < 1.0` :1717 ties into `object_out_of_bounds` x_bounds; block width > 0.08 aperture is the task's premise|
|25|Mjlab-Push-Flap-Franka|`franka_push_flap_env_cfg` :1723|`articulated/flap/xmls/flap.xml`|joint `flap_hinge`; geoms `flap_body`,`flap_panel`; sites `base_site`(-0.012,0.12,0),`object_site`(-0.012,0.18,0)|**EVT** `reset_flap_position` :1743-1747 x(0.44,0.50) y(−0.22,−0.12)|target **−1.2217305 (−70°)**, thr **0.15 rad** `C:2016-2022`|3|`goal_marker_offset=(0.11,-0.08,0)`; **flap.xml `range` was fixed to `-80.2 0` because it compiles in DEGREES** (no `<compiler angle="radian"/>`) — see `docs/cl25/phase_1/STATUS.md:231`. **Do not reintroduce that bug.** Push point 0.18 along +y drives the mount band; drop 0.160|
|26|Mjlab-Axial-Extract-Franka|`franka_axial_extract_env_cfg` :1752|`articulated/plug/xmls/plug.xml`|joint `plug_slide`; geoms `plug_body`,`plug_wall_{nx,py,ny}`,`plug_shaft`,`handle`; sites `base_site`(0,0,0.062),`object_site`|**EVT** `reset_plug_position` :1771-1775 x(0.40,0.48) y(±0.10)|target **0.10 m** (stop at 0.12), thr **0.02 m** `C:2045-2047`|4|marker offset `(0,0,0.10)`; **plug head is at `+0.062` above mount** (:1766-1769); shaft cylinder r 0.014 h 0.05, head r 0.02 h 0.012; the joint's `frictionloss` (breakaway) is the task premise; drop 0.050|
|27|Mjlab-Edge-Grasp-Franka|`franka_edge_grasp_env_cfg` :1780|`free/plate/xmls/plate.xml` + `free/ledge/xmls/ledge.xml`|`plate`(`plate_joint`,`object_site`); `ledge_base`(mocap), geom `ledge_geom`, sites `object_site`(0,0,0.10),`base_site`|CMD `EdgeGraspCommand._resample_command` `C:2119+` — writes ledge mocap **and** plate on top|`EdgeGraspCommand.compute_success` `C:2185`: `lift > lift_clearance=0.04` above ledge top **AND** `drift < max_drift=0.40`; latched `C:2160`|6|**`ledge_top_height=0.10` MUST match ledge.xml (geom half-h 0.05 at pos z 0.05)** `C:2199`; `plate_rest_offset=0.010` = plate half-thickness `C:2201`; `goal_offset=(-0.13,0,0.18)` `C:2207`; `plate_rel_x=(-0.02,0.04)`, `plate_rel_y=(±0.07)`; ledge 0.09×0.13×0.05, plate 0.05×0.045×0.008|
|28|Mjlab-Pivot-Lift-Franka|`franka_pivot_lift_env_cfg` :1807|`free/board/xmls/board.xml` + `free/wall/xmls/wall.xml`|`board`(`board_joint`,`object_site`); `wall_base`(mocap), geom `wall_geom`, `object_site`|CMD `PivotLiftCommand` `C:2235` (= Lifting + per-env wall mocap write `C:2248-2268`)|Lifting; `success_threshold=0.05` (`pivot_lift_env_cfg.py:147`); latched|6|**`wall_spawn_range x=0.50` with wall half-thickness 0.015 ⇒ inner face at 0.485**, paired against board yaw-swept half-extent `0.06·cos(0.3)+0.05·sin(0.3)=0.072` — the whole derivation is in `C:2275-2298`; board spawn `z=(0.011,0.011)` = half-thickness; board 0.06×0.05×0.010; wall 0.015×0.20×0.075|
|29|Mjlab-Throw-To-Bin-Franka|`franka_throw_to_bin_env_cfg` :1834 (reuses `make_place_in_container_env_cfg`)|cube.xml + container.xml|same as #18|CMD `PlaceInContainerCommand`; ranges :1861-1868|containment + settle, same thresholds as #18; latched|**5.0** (:1871)|`container_spawn_range x=(0.78,0.90)` deliberately beyond reach; `object_out_of_bounds` widened to x(0.0,1.4) y(±0.7) :1869-1870; cube spawn z 0.03|

**Shared per-task wiring** (applies to every row): action `robot_joint_pos` scale = `FRANKA_ACTION_SCALE` (0.04), 8-D; EE site is `"gripper"`; fingertip friction events target geoms `left_finger_pad|right_finger_pad`; `ee_ground_collision` contact sensor primary pattern `"link7"`; `cfg.viewer.body_name="link0"`. Terminations: `time_out`, `illegal_contact(ee_ground_collision)` (`mdp/terminations.py:15`), `object_out_of_bounds(x_bounds=(0,1), y_bounds=(±0.5))` (`mdp/terminations.py:20`) — the last two are popped in `test` mode.

**Observation vector is 60-D and fixed** (`lift_object_env_cfg.py:28-108`, layout documented at `classical/push_button.py:10-21`):
`0:9 joint_pos_rel | 9:18 joint_vel_rel | 18:21 object_pos | 21:25 object_quat | 25:28 gripper_pos | 28:34 gripper_rot6d | 34:40 object_rot6d | 40:43 gripper_to_object | 43:46 object_to_goal | 46:52 goal_orientation_diff | 52:60 control_qpos_diff`.
`object_pos` for articulated assets is the **`object_site`** world position; for free assets the object body/site. **Keeping `object_site` at the same offset keeps every teacher and policy valid.**

## 4. TEACHERS — `src/mjlab/continual_distill/classical/`

### `base.py`
`ClassicalPolicyBase` (`base.py:67`): loads a **private CPU `mujoco.MjModel.from_xml_path(panda.xml)`** (`:96`, path helper `:39-45`) for FK (`_fk` `:206`) and site Jacobians (`_ik_step` `:222`), runs damped-least-squares IK (`ik_iters=30` `:88`, `damping=0.2`, `kp_task=1.0`, `max_dq=0.05`, `max_pos_err=0.08`), and converts to the env's action convention `action[:7] = (q_des - default_qpos[:7]) / FRANKA_ACTION_SCALE`, `action[7] = gripper_a` (`:186-190`). Interface: subclass implements `_target_error(i, obs_i) -> (pos_err(3,), target_rot (3,3)|(3,)|None, gripper_action)` (`:193`). `__call__(obs (N,60)) -> (N,8)` (`:130`). `reset(env_ids)` clears `_phase`, `_phase_steps`, `_q_cmd` (`:119`).
Two default postures, and **picking the wrong one puts FK at a fantasy configuration**: `NEUTRAL_QPOS` (`:29`) for door/drawer/button/lever/valve/switch/window/lid/flap/plug envs (`get_franka_robot_cfg_neutral`), `HOME_QPOS` (`:32`) for lift/push/stack/etc (`get_franka_robot_cfg`).
Teachers must use **relative** obs terms only (`gripper_to_object` 40:43, `object_to_goal` 43:46) — absolute terms carry the per-env origin offset. Enforced by `tests/test_classical_teachers.py:66` (`test_teachers_use_relative_observations_only`).

### Registration
`classical/__init__.py:54-84` — the dict `CLASSICAL_POLICIES: {task_id: PolicyClass}`, described in-file as "the single source of truth". 29 entries, exactly the 29 registered Franka tasks. Adding a teacher = import the class + add one dict entry + add to `__all__`.

### Evaluation commands
Harness: `classical/test_classical.py` (67 lines). Args: `--task --num-envs (16) --num-episodes (2) --device (cuda:0)`. It reads success **during** the loop (`test_classical.py:52-55`) because the env auto-resets on timeout.

**The exact cl25 n=128 protocol** (`docs/cl25/phase_1/LOGS.md:288-293`, repeated at `:1181-1183`, `:1333`; protocol stated at `docs/cl25/phase_1/STATUS.md:40` "Competence bar: SR >= 0.90, measured at n = 128"):
```bash
RUN_GPU=2 srun --jobid=20277 --overlap -n1 --cpus-per-task=2 --export=ALL,RUN_GPU \
  bash -c 'export CUDA_VISIBLE_DEVICES=$RUN_GPU; cd /ihub/homedirs/svs_ald/sudhir/mjlab; \
  PYTHONPATH=src .venv/bin/python -m mjlab.continual_distill.classical.test_classical \
  --task <TASK_ID> --num-envs 32 --num-episodes 4'
```
i.e. **n = 32 envs × 4 episodes = 128**. Iterate at n=32 (`--num-envs 32 --num-episodes 1`).

Other entry points: `collect_classical_dataset.py` (`--task --num-samples 153600 --num-envs 512 --action-std 0.2 --output-dir`, writes `data.pkl`+`teacher.pkl`); `debug_rollout.py` (`--task --num-envs 4 --print-envs 2 --print-every 10`; **its own local POLICIES dict only covers 4 tasks**, `debug_rollout.py:25-30`).

### Geometry constants per teacher (all in `classical/<task>.py`)
- `axial_extract.py`: `HOVER_LEAD=0.13`, `SEAT_TOL=0.014`, `PULL_LEAD=0.05`, `FLOOR_MIN_Z=0.030`, `DESCENT_RATE=0.035`, `CLOSE_STEPS=14`.
- `cage_drag.py`: `RIDE_HEIGHT=0.026`, `HALF_WIDTH=0.02` (cube), `PAD_OFFSET=0.010`, `EFFECTIVE_STANDOFF=0.05`, `STANDOFF=0.004`, `BEHIND=0.08`, `FINGER_JOINT_MAX=0.04`, `HOVER_HEIGHT=0.15`.
- `drag_pull.py`: `RIDE_HEIGHT=0.020`, `HALF_WIDTH=0.04` (cuboid), `PAD_RADIUS=0.010`, `STANDOFF=0.004`, `BEHIND=0.055`.
- `edge_grasp.py`: `GOAL_OFFSET=[-0.13,0,0.18]`, `PLATE_HALF_X=0.05`, `LEDGE_HALF_X=0.09`, `BEHIND=PLATE_HALF_X+PAD_RADIUS+STANDOFF≈0.066`, `PUSH_HOVER_HEIGHT=0.13`, `OVERHANG_TARGET=0.032`.
- `flip_switch.py`: `PUSH_STANDOFF=0.030`, `CONTACT_Z=-0.012`, `HOVER_X=0.07`, `STROKE_X=0.16`, `STROKE_Z=-0.03`.
- `lift_object.py`: `GRASP_SITE_Z=0.045`, `OBJ_CENTER_Z=0.020`, `HOVER_SITE_Z=0.22`, `SEAT_TOL=0.012`; per-object overrides at `:245` (cylinder), `:252` (sphere, `CLOSE_STEPS=18`), `:259` (ellipsoid, `OBJ_CENTER_Z=0.022`, `ALIGN_TOL=0.008`).
- `open_door.py`: `STANDOFF=0.11`, `BAR_OUT=0.02` (2 cm handle bar), `SEAT_TOL=0.028`, `SLIP_DIST=0.10`, `MAX_WAYPOINT=0.22`, `PULL_DTHETA=22°`.
- `open_drawer.py`: `TIP_DROP=0.004`, `HOVER_Z=0.10`, `PANEL_BIAS_X=-0.015`, `SEATED_TOL=0.035`, `PULL_STEP=0.12`, `SEAT_OVERSHOOT=0.04`, `UNHOOK_Z=0.06`.
- `open_lid.py`: `FACE_STANDOFF=0.030`, `CONTACT_FRAC=0.95`, `HOVER=0.09`, `LEAD_ANGLE=0.9`.
- `peg_insertion.py`: `GRASP_UP=0.075`, `INSERT_DEPTH=0.025`, `CARRY_TIMEOUT=100`.
- `pivot_lift.py`: `BOARD_HALF_X=0.06`, `BOARD_HALF_Z=0.010`, `BOARD_CENTER_Z=0.011`, `STANDOFF≈0.076`, `PIVOT_DEPTH=0.55`, `CLEAR_UP=0.16`.
- `place_in_container.py`: `DROP_HEIGHT=0.10`, `DAMP_STEPS=25`, `SETTLE_STEPS=60`.
- `push_button.py`: `HOVER_HEIGHT=0.07`, `PRESS_DEPTH=0.03`, `_AXIS_DOWN=[0.20,0,-0.980]`.
- `push_cuboid.py`: `RIDE_HEIGHT=0.028`, `BEHIND=0.055`, `HALF_WIDTH=0.04`, `PAD_RADIUS=0.010`, `STANDOFF=0.004`, `ADVANCE=0.085`, `GOAL_TOL=0.015`.
- `push_disc.py`: `OBJ_CENTER_Z=0.020` (disc half-height), `PUSH_SITE_Z=0.050`, `BEHIND=0.045`, `ADVANCE=0.05`.
- `push_flap.py`: `FACE_STANDOFF=0.030`, `PUSH_DEPTH=0.12`, `CONTACT_FRAC=0.92`, `HOVER=0.09`.
- `reach_target.py`: no geometry constants (EMA only).
- `reorient_object.py`: `ROTATE_LIFT=0.10`, `ROTATE_STEPS=55`, `SETTLE_STEPS=45`.
- `rotate_valve.py`: `FACE_STANDOFF=0.014`, `CONTACT_FRAC=0.78`, `PINCH_FRAC=0.80`, `HANDOFF_ANGLE=1.40`, `LEAD_ANGLE=0.9`.
- `slide_window.py`: `PUSH_STANDOFF=0.028`, `CONTACT_Z=-0.015`, `HOVER_Y=0.10`, `PUSH_STEP=0.12`, `SEAT_OVERSHOOT=0.04`.
- `stack_object.py`: phase enum only; no half-sizes (uses obs vectors).
- `strike_slide.py`: `PUCK_HALF_HEIGHT=0.012`, `PUCK_RADIUS=0.035`, `RIDE_HEIGHT=0.026`, `STANDOFF=0.06`, `STRIKE_DEPTH=0.45`, `WINDUP_MIN/MAX=0.05/0.18`.
- `throw_to_bin.py`: `GRASP_SITE_Z=0.045`, `OBJ_CENTER_Z=0.020`, `DROP_HEIGHT=0.12`, `FLOOR_MIN_Z=0.030`.
- `tool_pull.py`: `GOAL_LOCAL=[0.42,0,0.012]`, `RIDE_Z=0.034`, `BEHIND=0.052`, `BEHIND_BIAS=1.8`, `PUSH_STEP=0.09`, `GOAL_TOL=0.045`.
- `topple_block.py`: `FACE_HALF_X=0.05`, `CONTACT_Z=0.055` ("top face is +0.09"), `STANDOFF=0.08`, `PUNCH_DEPTH=0.32`, `HOVER_Z=0.10`.
- `turn_lever.py`: `FACE_STANDOFF=0.018`, `CONTACT_FRAC=0.90`, `RESEAT_TOL=0.06`, `LEAD_ANGLE=0.60`.

**Any teacher constant naming a half-extent, ride/contact height, standoff, or face offset must be re-derived if the corresponding asset's collision geometry changes.**

## 5. RENDERING

- `src/mjlab/viewer/offscreen_renderer.py` uses **`mujoco.Renderer`** (`:47-49`), a separate CPU `MjData` (`:19`), and `mujoco.mj_forward` per frame (`:73`). It syncs `qpos`, `qvel`, and — critically — `mocap_pos`/`mocap_quat` (`:68-71`, with a comment explaining that without it every mocap-mounted asset renders at the env origin). It composites up to `_MAX_ENVS = 32` envs via `mjv_addGeoms` (`:83-101`). Sets `model.vis.global_.offheight/offwidth` from cfg (`:27-28`), and can kill shadows/reflections (`:30-33`).
- Camera: `_setup_camera` (`:106`) — `WORLD` ⇒ free camera; also `ASSET_ROOT`/`ASSET_BODY` tracking modes. `ViewerConfig` defaults `width=320, height=240` (`src/mjlab/viewer/viewer_config.py:28-29`).
- `render_rollout.py` (`src/mjlab/continual_distill/classical/render_rollout.py`) is two-phase: (1) unrendered batched stats (`run_stats_phase` :131), (2) single-env rendered (`run_render_phase` :198). Framing overrides at `:99-113`: `origin_type=WORLD`, `elevation=-28`, `azimuth=150`, `lookat=(0.3,0,0.4)`, `distance=1.6`, then `_autoframe_camera(env, RecordConfig())` (`:224`, defined `src/mjlab/scripts/record_task_videos.py:83`) which fits the camera to env-0's geom bounding box using `geom_xpos ± geom_rbound`, **ignoring the ground plane** (`record_task_videos.py:105-117`).
- **`MUJOCO_GL=egl` is required and it must run on a GPU node**: `render_rollout.py:41` (`MUJOCO_GL=egl PYTHONPATH=src .venv/bin/python -m ...`) and `:48` "Must run on a GPU node with `MUJOCO_GL=egl` (the login node has no EGL device)." No osmesa path is wired up; the login node cannot render.
- Video writing: **moviepy** `ImageSequenceClip(...).write_videofile(codec="libx264", audio=False)` (`render_rollout.py:173-179`); `imageio.v2.imwrite` for `thumb.jpg` and `--debug` PNG frames (`:234, :264, :289-293`). FPS resampled from the 50 Hz control rate to `--fps 30` by `_resample_fps` (`:165`). Defaults `--fps 30 --width 640 --height 480` (`:324-326`). Outputs `teacher.mp4`, `failure.mp4`, `thumb.jpg`, `result.json`.
- **Lighting / skybox / ground today:**
  - `src/mjlab/scene/scene.xml` is the only `<visual>` block: `<headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>`, `haze`, `<global azimuth="135" elevation="-25" offwidth="1920" offheight="1080"/>`, `<quality shadowsize="8192"/>`, `<statistic meansize="0.03"/>`.
  - **There are no `<light>` elements and no skybox anywhere in the manipulation scenes.** `spec_config.LightCfg` (`src/mjlab/utils/spec_config.py:241`) and `TextureCfg` with `type="skybox"` (`:68`) exist but grep shows **zero** uses under `asset_zoo/` or `tasks/manipulation/`. Terrain `add_lights` defaults to `False` (`src/mjlab/terrains/terrain_generator.py:59`, `config.py:56`).
  - Ground: `TerrainImporterCfg(terrain_type="plane")` for every manipulation task (`lift_object_env_cfg.py:291`), producing one `mjGEOM_PLANE` geom named `terrain` with a procedural checker texture `groundplane` (`rgb1=(0.2,0.3,0.4)`, `rgb2=(0.1,0.2,0.3)`, `markrgb=(0.8,0.8,0.8)`, 300×300) and material `groundplane` (`texrepeat=(4,4)`, `reflectance=0.2`) — `src/mjlab/terrains/terrain_importer.py:13-31, 150-160`.
  - **Implication:** everything you see today is headlight-lit, unshadowed, and untextured except the checker floor. Adding realistic mesh/texture assets will *look* flat unless a light and/or skybox is also added — either via `EntityCfg.lights/textures` on one entity, or via `SceneCfg.spec_fn`.
- Publish path: `src/mjlab/continual_distill/classical/publish_rollout.sh` — `rsync` a local dir to `untu_vps:~/sudhir/continual_learning/phase1/<Task-Id>/`, served at `https://cl.untuai.com/phase1/<Task-Id>/`. Usage: `publish_rollout.sh <Task-Id> <local-dir>`; index reads each `result.json` client-side, no rebuild.

## 6. AUDIT + TESTS

### `src/mjlab/scripts/audit_workspace.py`
Two modes (`AuditConfig` `:376-380`): `--measure` (FK sampling to regenerate the envelope, `measure()` `:167`, `_sample_workspace` `:119` over 400k samples at 90% of joint range) and the default per-task audit (`main` `:383`) over all `ARM_GRIPPER` tasks, optionally `--keyword`, `--num-envs 64`.
Checks, per task: object spawn radial vs `workspace.GRASP_RADIAL_MIN/MAX` and top-down grasp-pose fraction (`_object_positions` `:349`, `_grasp_pose_fraction` `:156`); goal radial vs `GOAL_RADIAL_MAX`/`MECHANISM_GOAL_RADIAL_MAX` (`_goal_positions` `:317`); **floor penetration** (`_floor_penetration` `:224`, tolerance `_FLOOR_TOLERANCE = 0.02` `:47`); **entity interpenetration at reset** (`_entity_overlaps` `:270`, `_OVERLAP_TOLERANCE = 0.001` `:39`). Exemption tables: `_REACH_EXEMPT` `:56` (tool-pull puck 0.58-0.75; throw-to-bin container 0.75-0.95), `_SIDE_APPROACH` `:71` (topple/cage/pivot/edge), `_GOAL_EXEMPT` `:94`.
Invoke: `python -m mjlab.scripts.audit_workspace` (add `--measure` to regenerate the envelope). **Both the floor and overlap checks use `geom_rbound` / contact distances, so they will react to new mesh geometry — run this after every asset swap.**

⚠️ `workspace.py:180` and `workspace.py:194` tell you to regenerate `MECHANISM_DROP_BELOW_MOUNT` with `audit_workspace --measure-drops`, but **that flag does not exist** in `audit_workspace.py` (only `--measure`). `tests/test_workspace_placement.py:134` (`test_mechanism_drops_are_swept_over_the_joint_range`) recomputes the swept drop directly from `src/mjlab/asset_zoo/objects/articulated/<asset>/xmls/<asset>.xml` and asserts the recorded value covers it — use that test as the regeneration oracle, or implement the flag.

### Tests (`tests/`, run with `make test` / `uv run pytest`, `make test-cpu` sets `FORCE_CPU=1`)
- `tests/test_classical_teachers.py` — `test_registry_covers_every_class_a_task` (:35, exact bijection between `CLASSICAL_POLICIES` and the tagged Class-A set), `test_teacher_produces_valid_actions` (:43, parametrized over all Class-A tasks: builds env on CPU with `test=True`, 5 steps, asserts shape `(N, action_dim)` and finiteness), `test_teachers_use_relative_observations_only` (:66). Explicitly **not** an accuracy measurement (docstring :1-11).
- `tests/test_workspace_placement.py` — the placement gate: `test_objects_spawn_within_reach` (:67), `test_objects_spawn_per_env` (:93), **`test_nothing_is_buried_in_the_floor` (:110)**, **`test_mechanism_drops_are_swept_over_the_joint_range` (:134)**, `test_mechanism_mount_heights_clear_the_floor` (:175), `test_grasp_box_corners_respect_the_radial_ceiling` (:184), `test_grasp_box_rejects_impossible_lateral_band` (:205). `REACH_EXEMPT` at :30 must be kept in sync with the audit script.
- `tests/test_class_a_expansion.py` — success-predicate tests for tasks 13-20: `test_articulation_success_predicate_fires` (:74), `test_place_in_container_requires_actual_containment` (:121), `test_reorient_scores_orientation_not_position` (:141), `test_every_manipulated_entity_is_observable` (:158), `test_flip_switch_detent_is_actually_bistable` (:195), `test_tool_pull_goal_is_physically_reachable` (:235).
- `tests/test_class_a_wave1.py` — tasks 21-29: `test_drag_pull_goal_is_nearer_than_the_object` (:129), `test_strike_slide_goal_is_beyond_the_reach_envelope` (:150), `test_cage_drag_pinching_voids_the_episode` (:172), `test_topple_scores_either_landing_face` (:216), plus interface checks (:65, asserts action scale 0.04).
- `tests/test_asset_zoo.py` only compiles G1/GO1 — **there is no compile test for the manipulation objects.** Adding one per new mesh asset would be cheap insurance.
- `tests/test_task_configs.py`, `tests/test_benchmark_taxonomy.py`, `tests/test_scene.py`, `tests/test_entity.py` also touch this area.

### benchmark-smoke and the manifest
- `benchmark-smoke` is a console script (`pyproject.toml:62` → `mjlab.scripts.benchmark_smoke:main`). Also `python -m mjlab.scripts.benchmark_smoke [--keyword X] [--num-envs 8] [--steps 20] [--isolate]`. It builds/resets/steps every tagged benchmark task with `test=True` and zero actions, checking finite reward and obs and reporting obs/action dims (`src/mjlab/scripts/benchmark_smoke.py:38-62`). **It will NOT catch a placement or burial bug** — that is exactly what `audit_workspace` + `test_workspace_placement` exist for (see `docs/benchmark/WORKSPACE_FIX.md:189`).
- Manifest: `src/mjlab/continual_distill/docs/benchmark/manifest.json`, currently `num_tasks: 37`, keys `num_tasks, num_distinct_skills, counts_by_embodiment, counts_by_skill, counts_by_fragility, tasks`. Regenerate with (per `docs/benchmark/AUTHORING_GUIDE.md:70`):
  ```
  python -c "from mjlab.tasks.manipulation import benchmark; \
    benchmark.export_manifest('src/mjlab/continual_distill/docs/benchmark/manifest.json')"
  ```
  (`benchmark.export_manifest` at `src/mjlab/tasks/manipulation/benchmark.py:146`.)
  **The manifest holds taxonomy metadata only — no geometry.** A pure asset swap that preserves task IDs and taxonomy does **not** require regenerating it. Regenerate only if you add/remove/retag tasks.

**Post-swap gate for each task agent (minimum):**
1. `python -m mjlab.scripts.benchmark_smoke --keyword <TaskName>`
2. `python -m mjlab.scripts.audit_workspace --keyword <TaskName>` → 0 problems
3. `FORCE_CPU=1 uv run pytest tests/test_workspace_placement.py -k <TaskName>` and `tests/test_classical_teachers.py -k <TaskName>`
4. `test_classical --task <ID> --num-envs 32 --num-episodes 4` vs the SR recorded in `docs/cl25/phase_1/STATUS.md`

## 7. PACKAGING

- Manager: **uv**, on PATH at `/ihub/homedirs/svs_ald/.local/bin/uv`, version **0.8.22**. Backend `uv_build` (`pyproject.toml:1-3`); project `mjlab` 0.1.0, `requires-python = ">=3.10,<3.14"`; venv is Python 3.13 at `./.venv`.
- Runtime deps at `pyproject.toml:33-50` (`mujoco>=3.11`, `mujoco-warp>=3.11`, `warp-lang>=1.14`, `torch>=2.7.0`, `trimesh>=4.8.3`, `moviepy`, `viser`, `rsl-rl-lib`, `wandb`, …). Dev group at `:65-75`. Extra `cu128` at `:76-77`. Custom indexes at `:85-106` (`nvidia`, `pytorch-cu128`, `mujoco` = `https://py.mujoco.org`, all `explicit = true`). `[tool.uv].required-environments` pins darwin/arm64 + linux/x86_64 (`:79-83`).
- **To add a package (e.g. coacd) correctly:**
  ```bash
  cd /ihub/homedirs/svs_ald/sudhir/mjlab
  uv add coacd            # edits pyproject.toml + uv.lock + installs into .venv
  ```
  `uv add` resolves against the existing lock and updates only what it must — it does not blow away pinned versions. If it should be dev-only (a mesh-preprocessing tool, not a runtime dep — **likely the right call for coacd**):
  ```bash
  uv add --group dev coacd
  ```
  To install without touching the lock at all (throwaway, e.g. one-off decomposition script): `uv run --with coacd python your_script.py`.
  **Do not** `pip install` into `.venv` — it desyncs `uv.lock`. Re-sync everything with `make sync` (`uv sync --all-extras --all-packages --group dev`).
- Ad-hoc runs in the repo consistently use `PYTHONPATH=src .venv/bin/python -m <module>` (see every command in `docs/cl25/phase_1/LOGS.md`), not `uv run` — keep that convention for cluster `srun` lines.
- Tooling: `make format` (ruff), `make type` (ty + pyright), `make test` / `make test-cpu` / `make test-fast`, `make build` (wheel + `tests/smoke_test.py` in isolation).

---

### Cheat-sheet of the invariants that must not break
1. **Names**: body, joint, geom, and especially `object_site` / `base_site` names and their **offsets** — commands index them by name (`C:1231`, `C:1264-1266`, `C:1578`) and Entity strips the `<entity>/` prefix (`entity/entity.py:227-249`).
2. **Half-extents baked into cfgs**: every `z=(h,h)` spawn value, `stack_height`, `ledge_top_height`, `plate_rest_offset`, `goal_offset.z` for tool-pull, `_ENGAGE_INSET`, topple's `_half`/`_travel`, `aperture_min`, `lateral_tolerance`.
3. **`MECHANISM_DROP_BELOW_MOUNT`** (`workspace.py:148-163`) — swept over the joint range, gates `_mech_z()` in the Franka cfgs.
4. **Radian vs degree in MJCF** — object XMLs have no `<compiler angle="radian"/>`, so bare `range="0 90"` means degrees (door), and `flap.xml` was already burned by this once.
5. **Mocap-ness**: 13 assets rely on `mocap="true"` root bodies written per-env by commands; keep it or `auto_wrap_fixed_base_mocap` will silently add a `mocap_base` wrapper body and change `root_body.name`.
6. **`nconmax=60, njmax=650`** on the ten articulated tasks — the most likely thing to overflow after a mesh swap.
7. **Convex-only collision** in mujoco_warp — concave meshes need convex decomposition into multiple geoms.