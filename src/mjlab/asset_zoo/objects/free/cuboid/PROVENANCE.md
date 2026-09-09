# cuboid — provenance

| | |
|---|---|
| **Asset** | YCB `009_gelatin_box`, `google_16k` textured scan |
| **Source** | YCB Object and Model Set — https://www.ycbbenchmarks.com/ (mirror: `https://ycb-benchmarks.s3.amazonaws.com/data/google/009_gelatin_box_google_16k.tgz`) |
| **License** | CC-BY 4.0 |
| **Raw download** | `~/assets_raw/ycb/009_gelatin_box/google_16k/textured.obj` (outside the repo) |
| **Used by** | Push-Cuboid, Drag-Pull, Stack-Cube (the base) |

## Real dimensions vs the spec

The scan is at real scale but parked at an arbitrary yaw (axis-aligned bbox
89.4 x 101.1 x 30.1 mm). The yaw search puts the true box at **76.85 deg**, giving
**89.2 x 72.9 x 30.1 mm** — the real gelatin carton. The primitive it replaces was
80 x 80 x 30 mm, so this is a genuinely close real-object substitute; **no scaling
was applied**.

## What was changed and why

| Step | Value | Reason |
|---|---|---|
| Rotate about z | +166.85 deg | 76.85 deg de-rotates the scan; the extra **90 deg** puts the box's LONG (89 mm) side across the push/drag direction and its SHORT (73 mm) side along it. Both halves of that help: a wider contact face gives the pusher more cross-track tolerance, and a smaller along-axis half-extent shrinks Drag-Pull's `_ENGAGE_INSET`, so the reachable spawn band *grows* instead of shrinking |
| Scale | none (1.0x) | already real scale, and within 12% of the primitive on every axis |
| Origin | bounding-box centre | half-extents read straight off the collision box |
| Decimation | 16384 -> 11999 tris | budget 20k |
| Texture | 4096^2 -> 1024^2 PNG | budget; PNG only |
| Collider | one `box`, half-extents 0.0365 x 0.0446 x 0.0150 | flat carton; box-box is the cheapest pair, and step time stays at the primitive baseline |
| Mass | 0.097 kg | the published product mass. **This is ~2x the primitive's 0.050 kg** and therefore ~2x the sliding friction (0.95 N vs 0.49 N) the Push-Cuboid and Drag-Pull teachers work against — a real physics change, measured in G7, not hidden |

Half-**height** is 0.0150 m, the same as the primitive, so every `z=(0.015, 0.015)`
spawn value, `goal_z_height` and stack z in the env cfgs is unchanged. Only the
horizontal half-widths moved, from a single 0.04 to 0.0365 (x) / 0.0446 (y).

Friction / `condim` / `solref` / `contype` / `conaffinity` carried over verbatim.

## Size budget

`xmls/assets/` = 1.5 MB (limit 5 MB).
