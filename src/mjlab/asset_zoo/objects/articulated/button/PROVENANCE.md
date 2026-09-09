# Push-Button — asset provenance (CL-V2, W2-a, 2026-09-09)

**Asset:** BUILT, not downloaded. PartNet-Mobility is unreachable from the cluster
(STATUS decision D8), so the mechanism was modelled in headless **Blender 4.2.23 LTS**
(`~/tools/blender/blender`) and textured from ambientCG.
Build script: `~/assets_raw/blender/w2a/build_button.py` (+ `mechlib.py`, `pack.py`).

**Real object:** an industrial mushroom-head **plunger / palm button** on a bolted
steel control panel (Schneider XB4 / Eaton M22 family look).

| part | mesh | texture (source) | license |
|---|---|---|---|
| control panel, 200 x 200 x 20 mm, 4 M10 bolts | `button_panel_vis.obj` | ambientCG `Metal009` (brushed steel) | CC0 1.0 |
| guide barrel, 50 mm dia x 41 mm | `button_housing_vis.obj` | ambientCG `Metal029` (matte black), luminance lifted to mean 0.22 so it is not crushed to black by MuJoCo's headlight | CC0 1.0 |
| warning collar, 61 mm dia x 5 mm | `button_collar_vis.obj` | ambientCG `PaintedMetal001` (safety yellow, chipped) | CC0 1.0 |
| plunger rod, 21 mm dia x 62 mm + shoulder | `button_rod_vis.obj` | ambientCG `Metal032` (bright brushed steel) | CC0 1.0 |
| mushroom cap, 45 mm dia x 18 mm | `button_cap_vis.obj` | ambientCG `Plastic007`, re-tinted to signal red (0.90, 0.08, 0.06) at mean 0.80 — luminance kept, so the moulded-plastic micro-detail survives | CC0 1.0 |

Texture URLs: `https://ambientcg.com/view?id=<Metal009|Metal029|PaintedMetal001|Metal032|Plastic007>`
(all CC0 1.0, 1K JPG sets, converted to 512x512 PNG by `asset_pipeline.package`).

## Real dimensions vs. the spec

| dimension | real product | this asset | note |
|---|---|---|---|
| cap diameter | 40 mm (XB4-BS) | 45 mm | inside the brief's 40-50 mm band |
| cap height | 14-18 mm | 18 mm | |
| barrel diameter | 30-40 mm | 50 mm | scaled up with the cap |
| panel | 3-6 mm sheet | 20 mm | UNCHANGED from the primitive asset: `button_body` is the contact-sensor geom and its half-extents set `MECHANISM_DROP_BELOW_MOUNT["button"] = 0.030` |
| **stroke** | **5-10 mm** | **50 mm** | **the one non-real dimension — see below** |

## The 50 mm stroke (decision, W2-a)

`button_slide` range is -0.05..0 and `PushButtonCommand` targets -0.05 m with a 0.02 m
threshold. A real mushroom head travels 5-10 mm, so a literal mushroom button at 50 mm
would be a lie. The task's *motion profile* is "press down 5 cm" and the brief lets the
owner decide: **the stroke is KEPT** (shortening it would have meant editing the shared
`commands.py` target and threshold, changing the benchmark's difficulty for a purely
cosmetic gain) and the **geometry was changed to make 50 mm honest**: the cap rides on a
62 mm bright-steel plunger rod that retracts into a 41 mm black guide barrel. That is a
real object class (a palm/plunger button), not a flush mushroom head.

Clearances checked over the whole travel: at full press the cap underside sits at
z = +0.041 and the collar top at z = +0.036 — 5 mm of daylight, nothing interpenetrates.

## What was scaled / rotated / kept

Nothing was scaled: every part was modelled directly in the MJCF body-local frame in
metres. Frozen and unchanged: body names `button_base` (mocap) / `handle`, joint
`button_slide`, collider geom names `button_body` and `handle`, sites `base_site` and
`object_site` at (0, 0, 0.100), and the panel collider's contact parameters.

Changed: the cap collider is now a **cylinder** (r 0.0225, half-height 0.009) instead of
a 60 x 60 x 10 mm box, matching the round cap. Its mass is pinned at **36 g** — the same
moving mass the primitive box had, so the spring (`stiffness=1000`) / damper
(`damping=5`) response of the press is bit-for-bit the old one, and 36 g is also a fair
mass for a moulded cap on a steel rod.
