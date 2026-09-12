"""Scripted OpenDrawer teacher policy — top-down HOOK strategy.

Physics refresh (2026-09-10): seat 12 mm down with a 5 mm outward bias,
limit pull lead to 60 mm, and reseat when the captured contact is lost. These
parameters were re-evaluated with wrist, palm and full finger collisions active.
Historical calibration notes below refer to the earlier collision model.

Geometry (drawer.xml): the handle is a horizontal bar 16cm wide (y), 2cm tall
(z in [-0.01, 0.01]), 2cm thick (x), sitting ~1cm in front of the drawer panel
— the gap between the bar's rear face (x=-0.03) and the panel's front face
(x=-0.02) is only 1cm. The slide joint opens along -x; success = drawer pulled
to within 2cm of the -0.25m stop (i.e. >=0.23m), latched over the episode.

Strategy: NOT a side cage-grasp (the old approach, ~36%). Instead, come straight
DOWN from above with the fingers CLOSED, and drop the fingertip into the 1cm slot
behind the bar. The closed fingertip pad (~1.75cm across its closing axis) is
wider than the 1cm slot, so it wedges — a friction-free geometric hook. Pulling
-x then drives the pad's rear face into the bar's rear face (normal force, not
tangential friction), so the domain-randomized low fingertip friction (down to
0.3) is irrelevant. We deliberately bias the descent into the panel face (a large
40x20cm wall, impossible to miss) and let the tip slide down into the corner.

Only relative obs are used (offsets cancel): gto = obs[40:43] (handle - gripper),
o2g = obs[43:46] (goal - handle). Robot starts at NEUTRAL_QPOS (+-10deg reset).
"""

from __future__ import annotations

import numpy as np

from mjlab.continual_distill.classical.base import ClassicalPolicyBase

# Approach axis: EE z-axis points straight DOWN (-world z), so the fingers hang
# vertically and the closed fingertip drops into the slot behind the bar. Yaw is
# left free (axis-only orientation target), so the solver keeps the wrist near
# its posture default and only the approach axis is constrained.
_DOWN_AXIS = np.array([0.0, 0.0, -1.0])

# The 'gripper' site sits ~0.4cm ABOVE the closed fingertip pads (measured from
# panda.xml with the wrist oriented top-down), so the site is essentially AT the
# tip — no tool offset needed. When the site z-axis points straight down, the
# finger-closing axis is world y (along the 16cm bar), so closed fingers
# straddle along the bar and the tip drops into the 1cm slot in x. Verified.
# CL-V2 RE-DERIVATION. Measured from panda.xml: the closed collision pads span
# -0.0046 .. +0.0118 along the site's approach axis, so the true site -> lowest-pad
# offset is 11.8 mm, not the 4 mm the comment above claims. Substituting the true value
# measured WORSE (n=16, CPU: 0.688 vs 0.750), so 0.004 stays — it is a tuned descent
# bias that lands the pad on the bar/panel corner, not a tool offset. Documented rather
# than "fixed".
TIP_DROP = 0.012  # metres, gripper-site -> closed-fingertip along -z

# Lateral/depth standoffs, all in the offset-free gto frame.
# CL-V2 RE-DERIVATION, against the new cabinet (asset_zoo/.../drawer/PROVENANCE.md):
# the bar's half-height is 0.010 and the carcass top slab's underside is 0.130 above the
# handle, so the hover point at +0.10 clears the bar and keeps the hand capsule (which
# reaches site_z + 0.11) 20 mm under the slab. Unchanged.
HOVER_Z = 0.10  # hover this far above the bar before descending
# The hook slot is bit-identical to cl25 (bar rear face x = -0.030, drawer front face
# x = -0.020, 10 mm slot, 17.6 mm closed pad), so this re-derives to its old value. The
# sign is outward, away from the cabinet: the pad lands in front of the bar's top edge
# and slides back into the corner. Verified, not copied.
PANEL_BIAS_X = -0.005  # bias the descent ~1.5cm into the panel -> seats in corner
# xy alignment tol before starting the descent. Loose on purpose: the bar is
# 16cm wide in y and the panel-bias makes x self-seat into the corner, so a ~5cm
# xy offset still drops the tip into the slot. A tight tol (0.03) stranded the
# arm hovering at its ~3.5cm DLS steady-state bias for the whole episode.
# CL-V2: the cabinet is now a real 400 mm-deep carcass with a top slab, not cl25's
# 20 mm-thick floating wall, so the arm can no longer swing over it on its way down.
# Measured: with the top slab collidable the teacher drops 1.000 -> 0.625 (n=16), and
# the contact log shows the FINGER PADS clipping `drawer_body` on the way in. The
# approach is therefore staged: first a standoff well in FRONT of the cabinet face,
# then straight down. FRONT_STANDOFF puts the pads 0.14 m clear of the carcass front
# plane, which sits 0.04 m behind the bar.
FRONT_STANDOFF = 0.14
FRONT_TOL = 0.06
ALIGN_TOL = 0.055
SEATED_TOL = 0.035  # tip within this of slot depth => hooked, start pulling (slot
# geometry unchanged from cl25, so this re-derives unchanged)
EMA_ALPHA = 0.4  # smooth the noisy gto (obs noise ~+-1.4cm effective)
INTEG_GAIN = 0.25  # integral nulls the DLS steady-state bias on the descent
# Consecutive in-tolerance steps required before phase 1 -> 2 (both the initial
# hook-in and, after a pop-out, the re-hook). Was 2. Instrumented (2026-09-08):
# every failing env reaches phase 2 exactly once, pops out exactly once
# (`unhooks=1`), and DOES re-enter phase 2 a second time (`entries_ph2=2`) before
# the episode ends -- so the hook mechanism itself is not the residual, the
# RECOVERY COST is. On a 150-step budget for a 0.23m pull, the extra settle step
# on top of the re-descent measurably eats into the pull window: several
# near-misses landed the drawer within 1-2mm of the success band (e.g.
# -0.2285m / -0.2280m against a -0.23m cutoff) with `final_phase=2`, i.e. still
# actively pulling when time ran out. Same fix as turn_lever.py's SETTLE, same
# reasoning: dropping to 1 measured 0.711->0.883 (128) with no regression (a
# single in-tolerance step is not a false positive here either -- SEATED_TOL
# (0.035) already does the real filtering).
DESCEND_SETTLE = 1

PULL_STEP = 0.06  # aggressive -x pull; 150-step budget, 0.23m stroke (the drawer's
# 0.25 m travel and 3 s episode are both unchanged in CL-V2, so this re-derives
# unchanged)
GOAL_TOL = 0.015
SEAT_OVERSHOOT = 0.04  # keep driving -x this far past goal to seat on the stop
UNHOOK_Z = 0.06  # tip risen this far above slot while pulling => popped out
# CL-V3 (2026-09-09, W1-N): the two residual failures at n=128 on the v3 spec (0.984) were
# both a pop-out in the first ~10 steps of PULL where the hand ended up 15 cm in FRONT of
# the bar and 4-5 cm BELOW it (gto = (+0.15, 0, +0.04..0.06)) -- the z-only gate above never
# fired (tip_err z 0.03-0.06, under UNHOOK_Z) and the teacher spent the remaining 85 steps
# "pulling" air while the drawer coasted to -0.07..-0.10 m. Pop-outs are now detected on the
# FULL seat vector and recovered through the hover stage (straight back up over the bar,
# then the normal descent), which is the path the phase-1 instrumentation showed re-hooks
# reliably: `entries_ph2=2` on every failing env there.
UNHOOK_TOL = 0.065  # |site -> seat| while pulling => popped out (any direction)
GRIPPER_CLOSED = -1.0


class OpenDrawerClassicalPolicy(ClassicalPolicyBase):
  """Top-down hook: drop a closed fingertip behind the bar and pull -x."""

  # Tried 0.18 (faster, on the theory that more of the residual is throughput):
  # measured WORSE, 0.771 (96) vs 0.812 (96) at DESCEND_SETTLE=1. A faster
  # descent/pull destabilises the hook (more pop-outs) enough to outweigh the
  # extra speed budget buys. Reverted; not retried.
  max_dq = 0.15  # 150-step budget: brisk descent + 0.23m pull

  def reset(self, env_ids=None) -> None:
    super().reset(env_ids)
    if env_ids is None:
      self._settle = np.zeros(self.num_envs, dtype=np.int64)
      self._gto_ema = np.zeros((self.num_envs, 3))
      self._ema_init = np.zeros(self.num_envs, dtype=bool)
      self._integ = np.zeros((self.num_envs, 3))
      self._staged = np.zeros(self.num_envs, dtype=bool)
    else:
      self._settle[env_ids] = 0
      self._ema_init[env_ids] = False
      self._integ[env_ids] = 0.0
      self._staged[env_ids] = False

  def _target_error(self, i: int, obs_i: np.ndarray):
    gto_raw = obs_i[40:43]  # handle - gripper (bar center)
    o2g = obs_i[43:46]  # goal - handle  (points -x, magnitude = remaining stroke)
    if not self._ema_init[i]:
      self._gto_ema[i] = gto_raw
      self._ema_init[i] = True
    else:
      self._gto_ema[i] = EMA_ALPHA * gto_raw + (1 - EMA_ALPHA) * self._gto_ema[i]
    gto = self._gto_ema[i]

    # Vector from the gripper site to the desired fingertip seat point: the bar
    # center, biased into the panel corner, with the tip dropped TIP_DROP below
    # the site. gto already targets the bar center relative to the site; add the
    # tool offset so the *fingertip* (not the site) lands there.
    tip_offset = np.array([0.0, 0.0, -TIP_DROP])  # site -> closed tip, world frame
    seat = gto + np.array([PANEL_BIAS_X, 0.0, 0.0]) - tip_offset

    gripper_a = GRIPPER_CLOSED  # fingers closed throughout: pads act as one hook

    if self._phase[i] == 0:
      if not self._staged[i]:
        # Stage A: come in to a standoff in FRONT of the cabinet, at hover height, so
        # the descent never crosses the carcass top slab.
        pos_err = seat + np.array([-FRONT_STANDOFF, 0.0, HOVER_Z])
        if np.linalg.norm(pos_err) < FRONT_TOL:
          self._staged[i] = True
      else:
        # Stage B: slide in over the bar, HOVER_Z up. Get xy aligned first so the
        # descent goes down a clean vertical line into the slot.
        pos_err = seat + np.array([0.0, 0.0, HOVER_Z])
        if np.linalg.norm(pos_err[:2]) < ALIGN_TOL:
          self._phase[i] = 1
          self._phase_steps[i] = 0
    elif self._phase[i] == 1:
      # Descend into the slot. Integral action nulls the DLS steady-state bias
      # so the tip actually reaches slot depth instead of hovering short.
      raw_err = seat
      self._integ[i] = np.clip(self._integ[i] + INTEG_GAIN * raw_err, -0.06, 0.06)
      pos_err = raw_err + self._integ[i]
      if np.linalg.norm(raw_err) < SEATED_TOL:
        self._settle[i] += 1
        if self._settle[i] >= DESCEND_SETTLE:
          self._phase[i] = 2
          self._phase_steps[i] = 0
      else:
        self._settle[i] = 0
    else:
      # Pull -x along o2g until the slide seats on the stop. Re-descend if the
      # hook popped out of the slot (tip rose well above slot height).
      tip_err = seat  # residual site->seat error; +z means tip is too high
      if tip_err[2] > UNHOOK_Z or np.linalg.norm(tip_err) > UNHOOK_TOL:
        # Popped out: go back to the hover stage (over the bar at HOVER_Z, xy aligned)
        # and re-descend. Entering DESCEND directly from below / in front of the bar
        # would drive the pad into the bar's front face instead of over it.
        self._phase[i] = 0
        self._phase_steps[i] = 0
        self._staged[i] = True
        self._settle[i] = 0
        self._integ[i] = 0.0
        return seat + np.array([0.0, 0.0, HOVER_Z]), _DOWN_AXIS, gripper_a
      dist = np.linalg.norm(o2g)
      # Hold the seat (keep the tip in the slot) while translating along o2g.
      anchor = seat + self._integ[i]
      if dist < GOAL_TOL:
        # At goal: keep driving a little further -x to seat firmly on the stop.
        pos_err = anchor + o2g / (dist + 1e-8) * SEAT_OVERSHOOT if dist > 1e-6 \
          else anchor + np.array([-SEAT_OVERSHOOT, 0.0, 0.0])
      else:
        pos_err = anchor + o2g / (dist + 1e-8) * min(dist + SEAT_OVERSHOOT, PULL_STEP)

    return pos_err, _DOWN_AXIS, gripper_a
