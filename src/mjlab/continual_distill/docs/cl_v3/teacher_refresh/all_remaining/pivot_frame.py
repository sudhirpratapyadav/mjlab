"""Scripted PivotLift teacher -- ride the board up the gripper's own ramp, then pinch
(CL-V3, W1-D3; rewritten from W1-D/W1-D2's pusher).

TASK
----
Chopping board 0.120 (x) x 0.219 (y) x 0.020 (z), 0.17 kg, friction 0.7, flat on the floor
at x 0.38-0.41, yaw +-0.15, in front of a STATIC brick wall (box x 0.508-0.613, y +-0.25,
z 0-0.15, friction 0.9).  Goal: an airborne point (x 0.32-0.48, y +-0.15, z 0.15-0.30),
success ||board - goal|| < 0.05 latched over the 300-step episode.  Flat on the floor the
board cannot be pinched (both in-plane widths exceed the 80 mm aperture, the 20 mm
thickness is flush with the floor), so it has to be stood up first.

THE GEOMETRY THAT DECIDES THE STRATEGY (W1-D3, measured / derived)
------------------------------------------------------------------
* The board pivots about its FAR-BOTTOM edge, jammed in the floor/wall corner at x 0.508.
  Its far-top corner rides up the wall face, so at tilt ``th`` the board's near (upper)
  edge is at x = 0.508 - 0.12 cos th, z = 0.12 sin th and the wedge between the board's
  wall-side face and the wall, ``d`` below that edge, is (0.12 - d) cot th wide.
* The board therefore NEVER stands clear of the wall: at th = 90 deg it is flush against
  it and its top edge (z = 0.12) is still 3 cm BELOW the wall's top (0.15).  A top-down
  pinch of the standing board -- what the task description imagines -- is geometrically
  impossible: the far pad has nowhere to go.  The pinch has to happen at a MODERATE tilt,
  and W1-D2's teacher, which pushed to 63 deg and then let go to re-approach, was aiming
  at a configuration that (a) cannot be gripped and (b) does not exist: with the COM on
  the robot side of the pivot the board is unstable at every angle below 90 and falls back
  flat in ~0.1 s (I = mL^2/3 = 8.2e-4, torque m g L/2 cos th -> 20-90 rad/s^2).
* So the hand that tips the board must also be the hand that holds it and the hand that
  grips it, without ever breaking contact.  MEASURED tipping condition, pushing the near
  end face with a pad whose inner face is inclined at ``psi`` while the hand CLIMBS (so
  the pad slides up relative to the board and friction acts UP the face):
      N [0.02 sin psi + 0.12 cos psi + mu (0.12 sin psi - 0.02 cos psi)] >= m g 0.06
  -> N ~ 0.8 N at psi 55-80 deg with mu = 0.7 (the board's own friction; contacts combine
  by max, so the randomized pad friction 0.3-1.5 can only help).  WITHOUT the climb the
  friction reverses and the same push self-locks: 0.02 sin psi + 0.12 cos psi must then
  beat mu (0.12 sin psi - 0.02 cos psi), which fails for every psi above ~50 deg and needs
  6-15 N below it.  The climb is not a refinement, it is the mechanism.

STRATEGY -- the gripper's lower pad is a RAMP
---------------------------------------------
Hold the hand pitched by ``psi`` about the board's own azimuth, jaws part-open: the lower
finger's inner face is then a ramp inclined at ``psi``, facing up and toward the wall, and
the upper finger rides 3-6 cm clear above the board.
  0 HOVER   above the push point at z 0.22 -- OVER the wall, whose 0.15 top the old
            teacher's 0.10 hover sat ON for whole episodes (W1-D2) -- walking an xy
            waypoint in at 8 mm/step.
  1 DESCEND waypoint descent until the ramp's tip is just below the board's near-top edge.
  2 TIP     press the ramp against the board's near end and CLIMB.  ``psi`` is scheduled
            from PSI_MAX down to PSI_MIN as the observed tilt grows, and the tracked
            contact point walks from the near-TOP corner (best moment arm while the board
            is flat) to a point D_GRIP below the near edge on the board's wall-side face,
            so the ramp ends up flat under the board.  A persistent lead along +c (press)
            and -a (drag up the ramp) is what puts the friction in the tipping direction.
  3 SEAT    hold the board resting on the ramp at PSI_MIN.
  4 CLOSE   squeeze.  Closing walks the lower pad 6 mm further into the board and brings
            the upper pad down onto its other face; the board finishes centred between the
            pads with ~10 mm of interference per side.
  5 LIFT    climb, then drive ``object_to_goal``.

Observations (60-D): 0:9 joint_pos_rel, 34:40 object_orientation (rows 1-2 of the board's
rotation matrix; local x = column 0 = the 120 mm axis, local z = column 2 = the 20 mm
thickness), 40:43 gripper_to_object (board centre - site), 43:46 object_to_goal.
"""

from __future__ import annotations

import numpy as np

from mjlab.continual_distill.classical.base import (
  FRANKA_ACTION_SCALE,
  HOME_QPOS,
  ClassicalPolicyBase,
)

GRIPPER_OPEN = 1.0
GRIPPER_CLOSED = -1.0
# Finger target = 0.04 * (action + 1) (per-finger, m); a = q/0.04 - 1.
# W1-D4, DERIVED from the panda XML and then measured.  Pad box half-sizes (0.0088 site-x,
# 0.0076 along the closing axis, 0.0082 along the approach axis), centre at (q + 0.0076)
# along c and +0.0037 along a.  With the hand pitched psi the two finger tips are at
#   lower  z = site_z - (q + 0.0152) cos psi - 0.0119 sin psi
#   upper  z = site_z + q cos psi         - 0.0119 sin psi
# so the STAGGER between them is only (2q + 0.0152) cos psi.  Requiring the lower pad to
# reach the floor guard AND the upper finger to clear the board's 20 mm top face by 3 mm
# gives (2q + 0.0152) cos psi >= 0.018.  At the old PSI_MAX = 85 deg that needs
# q >= 0.096 -- impossible -- which is why the trace showed the UPPER finger standing on
# the board's top face and pressing it 7 mm into the floor while the "ramp" never touched
# the end face.  The docstring's "the upper finger rides clear above the board" was 1.6 mm
# of clearance, not 30-60.  q = 0.040 (jaws fully open) with psi <= 76 deg satisfies it:
# at psi 70 the site sits at 0.035, the lower pad bottom at 0.005 and the upper finger tip
# at 0.038.
Q_STRADDLE = 0.040
GRIPPER_STRADDLE = Q_STRADDLE / 0.04 - 1.0

P_HOVER = 0
P_DESCEND = 1
P_TIP = 2
P_SEAT = 3
P_CLOSE = 4
P_LIFT = 5
P_SLIDE = 6                  # W1-D4: push the flat board to the wall BEFORE tipping
PHASE_NAMES = {0: "HOVER", 1: "DESCEND", 2: "TIP", 3: "SEAT", 4: "CLOSE", 5: "LIFT",
               6: "SLIDE"}

JOINT_LIMITS = np.array(
  [[-2.8973, 2.8973], [-1.7628, 1.7628], [-2.8973, 2.8973], [-3.0718, -0.0698],
   [-2.8973, 2.8973], [-0.0175, 3.7525], [-2.8973, 2.8973]]
)

BOARD_HALF_X = 0.06
BOARD_HALF_Z = 0.01

# -- hand pitch schedule ---------------------------------------------------------------
# PSI_MAX: nearly-vertical hand while the board is flat.  The ramp's tip corner then sits
# only 2.7-4 mm above the pad's LOWEST corner, so the pad-floor guard (which holds the
# lowest corner at PAD_FLOOR_MIN) still leaves the tip below the board's 20 mm top face --
# at PSI 55 the same guard puts the tip 1 mm ABOVE it and the hand rides over the board.
# PSI_MIN: the tilt the board is gripped at.  Bounded above by the wedge (the upper pad's
# outer face sits 25 mm off the board's mid-plane at the grip, and the wedge D_GRIP below
# the near edge is (0.12 - D_GRIP) cot th) and by the hand capsule, which is 0.20 m long
# along the closing axis and clips the wall's top-front edge for th above ~62 deg.
PSI_MAX = np.radians(70.0)
PSI_MIN = np.radians(55.0)
TILT_TARGET = np.radians(55.0)
S_FACE = 0.004               # where the tracked board point sits on the ramp face,
                             # measured along the approach axis from the site (the face
                             # spans -0.0045 .. +0.0119)
D_GRIP = 0.020               # final contact depth below the board's near edge
PAD_OUT = 0.0152             # pad outer face offset beyond the finger opening q (MEASURED
                             # on the panda XML: pad centre at q + 0.0076, half 0.0076)
PUSH_FACE_OUT = 0            # 1 = push with the upper pad's OUTER face (closing load)
# APPROACH FROM BEHIND, not from above the contact point.  MEASURED: descending straight
# onto the near-top corner puts the pad's inner face AT the corner's x, and the moment the
# fingers move (contact pushes them by 10-30 mm, see _ramp_site) the pad walks forward onto
# the board's TOP face and sits there -- traced, the site parked 15 mm past the corner with
# the board untouched for the rest of the episode in 6/8 envs.  Hover and descend a pad's
# width further back and let the TIP phase close the gap horizontally.
APPROACH_BACK = 0.0          # MEASURED at 0.032: it did NOT help (ee_ground_collision
                             # 0/8 -> 3/8 and max tilt 3-17 deg -> 2-9 deg), because the
                             # extra 3 cm of horizontal travel has to be made up inside
                             # TIP and the hand arrives with the fingers already displaced.
                             # Kept as a knob; the finger displacement is the real defect.

HOVER_Z = 0.22               # clears the wall's 0.15 top plus the pad hang
ALIGN_TOL = 0.012
ALIGN_SETTLE = 2
ALIGN_TIMEOUT = 35
APPROACH_RATE = 0.008
APPROACH_LEASH = 0.030
INTEG_GAIN = 0.08
INTEG_CLIP = 0.03
INTEG_BAND = 0.04
DESCENT_RATE = 0.005
DESCENT_RATE_FAST = 0.014
DESCENT_SLOW_BAND = 0.05
DESCENT_LEASH = 0.020
SEAT_TOL = 0.006
DESCEND_TIMEOUT = 45
# THE TIP IS A TILT WAYPOINT, NOT A LEAD.  MEASURED (first version of this file): a
# persistent Cartesian lead along -a (up the board's face) is not blocked by anything, so
# with cmd_ref = "command" the position loop integrates it and the hand simply CLIMBS --
# traced, the site left the board behind at z 0.045-0.081 with the board still flat in 8/8
# envs.  What is bounded is the board's own ARC: the contact point rides a circle of radius
# 0.12 about the far-bottom edge (which the wall pins), so the teacher walks a TILT
# waypoint along that arc at TILT_RATE, leashed to TILT_LEASH ahead of the observed tilt.
# A board that will not come only ever sees a bounded offset (0.12 * TILT_LEASH of arc)
# plus the normal press, and the pad stays on the 16 mm ramp face instead of running away.
TILT_RATE = 0.020            # rad/step the commanded tilt walks (55 deg in ~48 steps)
# The board has to be JAMMED before it can be tipped.  Its pivot is the far-bottom edge,
# and that edge only stays put once the wall holds it: on open floor the same push just
# slides the board and any tilt it picks up flattens out again (measured -- the board
# advanced 2-3 cm of the 5-9 cm it had to travel and the tilt stalled at 3-17 deg).  The
# wall's inner face is at a FIXED base-frame x in every env (wall_spawn_range x = 0.5605,
# half-thickness 0.0525), and the pivot's base-frame x is FK(joints) + gripper_to_object
# + the board's own half-extents -- all relative terms.
WALL_FACE_X = 0.508
JAM_MARGIN = 0.012           # start tipping once the pivot is this close to the face
TILT_LEASH = 0.09            # rad the command may lead the observed tilt (7 mm of arc)
TIP_PRESS = 0.015            # m of command lead along the contact normal.  MEASURED: 0.040
                             # generates >8 N -- it pried the fingers fully open (commanded
                             # 0.016 per finger, observed 0.0385), because the ramp is the
                             # lower pad's INNER face and its reaction is exactly the
                             # finger's opening direction.  The tip needs ~1 N.
CMD_LEAD_MAX = 0.05          # m the joint command's site may lead the actual site
TIP_RATE = 0.012             # m/step cap on the commanded site step during the tip
TIP_NORM_RATE = 0.006        # m/step cap along the contact normal (the press)
TIP_TAN_RATE = 0.006         # m/step cap along the face (the climb + position error)
TIP_TIMEOUT = 70
# W1-D4, MEASURED (pv_trace.py, n=8, 300 steps): the board NEVER REACHES THE WALL.  Its
# centre moved from x 0.388 to 0.391 over 100 steps of TIP while the site sat at x 0.349
# -- 15 mm PAST the board's near-top corner (0.328 + the 0.015 press) with the pads at
# z 0.017 against a board top of 0.020, i.e. the pusher was a 3 mm sliver riding the top
# corner.  The tilt waypoint is gated on the pivot reaching WALL_FACE_X - JAM_MARGIN, so
# with the board 60 mm short of the wall the gate never opened and TIP degenerated into a
# 200-step press.  Nothing downstream can work until the board is jammed, so the slide is
# now its own phase with its own bounded waypoint: the lead is the board's OWN remaining
# travel to the wall (never a runaway), the tracked contact point is the MIDDLE of the
# 20 mm end face rather than its top corner, and the floor guard is relaxed for the two
# pushing phases so the pad can actually get down onto that face.
SLIDE_TIMEOUT = 90           # phase timeouts now sum to 35+45+90+70+18+16+14 = 288 < 300
# The slide is a WAYPOINT ON THE BOARD, not a Cartesian lead on the hand.  MEASURED
# (W1-D4): a 30 mm lead along the azimuth, clipped only by the remaining travel to the
# wall, is a lead in a direction the board does not actually block (it slides, slowly), so
# the position loop integrates it and the pusher walks 16 mm PAST the board's near end
# face and stands on its top -- the board was pressed 4-7 mm INTO the floor (root z 0.0023
# against a 0.0099 rest height), and a pad rubbing down a face pins the object so it
# cannot slide at all.  The commanded pivot x now walks toward the wall at SLIDE_RATE but
# is leashed SLIDE_LEASH ahead of the OBSERVED pivot, so the pusher can never lead the
# board by more than a leash however stubborn the board is.
SLIDE_LEASH = 0.014          # swept: 0.008 -> 4 ground collisions per 8 envs and 4 envs past 20 deg; 0.014 -> 2 and 5
SLIDE_RATE = 0.010           # m/step cap on the commanded site step during the slide
SLIDE_FLOOR_MIN = 0.012      # pad-corner floor guard while pushing (0.015 held the pad
                             # bottom at 0.015 against a 20 mm face: 3 mm of overlap)
PUSH_FACE_Z = 0.5            # tracked point on the end face, as a fraction of its height
                             # (1.0 = the near-top corner, the old behaviour)
CLIMB_FACE_RATE = 0.030      # per step, in units of the fraction above (0.02 m of face)
CLIMB_FACE_MAX = 0.90        # stop 2 mm below the near-top corner
SEAT_PRESS = 0.018
SEAT_STEPS = 18
CLOSE_STEPS = 16
APERTURE_HELD = 0.012        # 20 mm board -> finger sum ~0.020; < 0.012 = missed
APERTURE_EMA = 0.3
CLIMB_STEPS = 14
CLIMB_ERR = 0.10
HELD_TOL = 0.07
CARRY_INTEG_GAIN = 0.15
CARRY_INTEG_CLIP = 0.05
YAW_RATE = 0.15
EMA_ALPHA = 0.35
EMA_ALPHA_LOW = 0.2
RESET_JOINT_JUMP = 0.25
RESET_OBJ_JUMP = 0.15
LEAD_HOVER = 0.12
LEAD_DESCEND = 0.08
LEAD_TIP = 0.14
MAX_ATTEMPTS = 2


def _board_rot(obs_i: np.ndarray) -> np.ndarray:
  """Board rotation matrix (columns = local axes in world) from rows 1-2 of R."""
  r1 = obs_i[34:37]
  r2 = obs_i[37:40]
  r1 = r1 / (np.linalg.norm(r1) + 1e-9)
  r2 = r2 / (np.linalg.norm(r2) + 1e-9)
  r0 = np.cross(r1, r2)
  return np.stack([r0, r1, r2], axis=0)


def _board_frame(obs_i: np.ndarray):
  """(x_loc, z_loc, tilt): the board's 120 mm axis oriented AWAY from the robot, its
  thickness axis oriented UP, and the tilt of the near end above horizontal."""
  Rb = _board_rot(obs_i)
  x_loc = Rb[:, 0].copy()
  if x_loc[0] < 0:
    x_loc = -x_loc
  z_loc = Rb[:, 2].copy()
  if z_loc[2] < 0:
    z_loc = -z_loc
  # x_loc points at the wall, so it pitches DOWN as the near end comes up.
  tilt = float(np.arcsin(np.clip(-x_loc[2], -1.0, 1.0)))
  return x_loc, z_loc, tilt


def _hand_axes(psi: float, n_hat: np.ndarray):
  """(approach axis, closing axis) for a hand pitched ``psi`` below horizontal in the
  board's own azimuth plane.  The closing axis is the board's normal at tilt ``psi``, so
  the lower finger's inner face is a ramp the board's face can come to rest on."""
  up = np.array([0.0, 0.0, 1.0])
  a_hat = np.cos(psi) * n_hat - np.sin(psi) * up
  c_hat = np.sin(psi) * n_hat + np.cos(psi) * up
  return a_hat, c_hat


class PivotLiftClassicalPolicy(ClassicalPolicyBase):
  """Ride the board up the lower pad's ramp, seat it on the ramp, pinch, lift."""

  DEFAULT_QPOS = HOME_QPOS
  max_dq = 0.05
  step_gain = 0.5
  orientation_weight = 0.5
  cmd_lead_max = LEAD_HOVER
  # M4, MEASURED (W1-D2, physics-free sweep of the base DLS): posture_weight 0.005 stops
  # the solve 2.6-3.3 cm SHORT in radius and ~1 cm HIGH on low, near targets (r 0.24-0.36
  # at z 0.02-0.03) -- exactly the push point here, and exactly the "IK dead zone" phase-1
  # recorded.  With 0 the same solve lands on them to 1e-3.
  posture_weight = 0.0
  # W1-D2, MEASURED: with cmd_ref = "actual" the base rebuilds the target from
  # FK(previous COMMAND) + err every step, so the servo's sag is ADDED to the descent each
  # step.  Referencing to the command makes the position loop an integrator instead.  ALL
  # axes, not just z: with the lateral error referenced to the ACTUAL site the loop's fixed
  # point is "position error == servo sag" (0.11 rad of shoulder droop = 10-17 cm at this
  # stretch) and the hand parks short and stays there.
  cmd_ref = "command"
  cmd_ref_axes = None
  PAD_FLOOR_MIN = 0.015
  PHASE_NAMES = PHASE_NAMES

  def reset(self, env_ids=None) -> None:
    super().reset(env_ids)
    if env_ids is None:
      n = self.num_envs
      self._gto_ema = np.zeros((n, 3))
      self._ema_init = np.zeros(n, dtype=bool)
      self._prev_gto = np.zeros((n, 3))
      self._prev_q = np.full((n, 7), np.nan)
      self._integ = np.zeros((n, 3))
      self._settle = np.zeros(n, dtype=np.int64)
      self._c_target = np.tile(np.array([1.0, 0.0, 0.0]), (n, 1))
      self._ap_ema = np.full(n, 0.08)
      self._ap_init = np.zeros(n, dtype=bool)
      self._attempts = np.zeros(n, dtype=np.int64)
      self._hold = np.zeros((n, 3))
      self._tilt_ema = np.zeros(n)
      self._twp = np.zeros(n)
      self._xwp = np.zeros(n)
      self._fwp = np.zeros(n)
      self._zwp = np.full(n, np.nan)
      self._wp = np.zeros((n, 3))
      self._wp_ok = np.zeros(n, dtype=bool)
      self.debug = [[] for _ in range(n)]
    ids = range(self.num_envs) if env_ids is None else env_ids
    for i in ids:
      self._clear(i)
      self._prev_q[i] = np.nan
      self._attempts[i] = 0
      self._tilt_ema[i] = 0.0

  def _clear(self, i: int) -> None:
    self._zwp[i] = np.nan
    self._wp_ok[i] = False
    self._ema_init[i] = False
    self._integ[i] = 0.0
    self._settle[i] = 0
    self._twp[i] = 0.0
    self._xwp[i] = 0.0
    self._fwp[i] = PUSH_FACE_Z

  def _rewind(self, i: int, reason: str = "", phase: int = P_HOVER) -> None:
    self.debug[i].append((int(self._phase[i]), reason))
    self._phase[i] = phase
    self._phase_steps[i] = 0
    self._clear(i)

  def _goto(self, i: int, phase: int) -> None:
    self._phase[i] = phase
    self._phase_steps[i] = 0
    self._clear(i)

  def _gto(self, i: int, obs_i: np.ndarray) -> np.ndarray:
    raw = obs_i[40:43]
    if not self._ema_init[i]:
      self._gto_ema[i] = raw
      self._ema_init[i] = True
    else:
      a = EMA_ALPHA_LOW if self._phase[i] in (P_DESCEND, P_SEAT, P_CLOSE) else EMA_ALPHA
      self._gto_ema[i] = a * raw + (1 - a) * self._gto_ema[i]
    return self._gto_ema[i]

  def _update_aperture(self, i: int, obs_i: np.ndarray) -> None:
    raw = float(self.default_qpos[7] + obs_i[7] + self.default_qpos[8] + obs_i[8])
    if not self._ap_init[i]:
      self._ap_ema[i] = raw
      self._ap_init[i] = True
    else:
      self._ap_ema[i] = APERTURE_EMA * raw + (1 - APERTURE_EMA) * self._ap_ema[i]

  def _detect_reset(self, i: int, obs_i: np.ndarray) -> None:
    q7 = obs_i[0:7]
    raw = obs_i[40:43]
    jumped = False
    if not np.any(np.isnan(self._prev_q[i])):
      if np.max(np.abs(q7 - self._prev_q[i])) > RESET_JOINT_JUMP:
        jumped = True
    if self._ema_init[i] and np.linalg.norm(raw - self._prev_gto[i]) > RESET_OBJ_JUMP:
      jumped = True
    self._prev_q[i] = q7
    self._prev_gto[i] = raw
    if jumped:
      self._rewind(i, "reset")
      self._attempts[i] = 0
      self._tilt_ema[i] = 0.0

  # -- action ----------------------------------------------------------------------------

  def _act_single(self, i: int, obs_i: np.ndarray) -> np.ndarray:
    self._update_aperture(i, obs_i)
    self._detect_reset(i, obs_i)
    ph = self._phase[i]
    q_abs = self.default_qpos + obs_i[0:9]
    self.cmd_lead_max = {
      P_DESCEND: LEAD_DESCEND, P_TIP: LEAD_TIP, P_SEAT: LEAD_TIP, P_CLOSE: LEAD_DESCEND
    }.get(ph, LEAD_HOVER)
    a = super()._act_single(i, obs_i)
    return a

  def _integrate(self, i: int, err: np.ndarray, gain=INTEG_GAIN, clip=INTEG_CLIP) -> None:
    lateral = np.array([err[0], err[1], 0.0])
    if np.linalg.norm(lateral) < INTEG_BAND:
      self._integ[i] = np.clip(self._integ[i] + gain * lateral, -clip, clip)

  def _pad_min_z(self) -> float:
    """Lowest corner of the two finger pads at the pose last passed to ``_fk``."""
    if not hasattr(self, "_pad_gids"):
      self._pad_gids = [self.model.geom(n).id for n in ("left_finger_pad", "right_finger_pad")]
      self._pad_hs = [self.model.geom_size[g].copy() for g in self._pad_gids]
    best = 1e9
    for g, hs in zip(self._pad_gids, self._pad_hs):
      pos = self.data.geom_xpos[g]
      R = self.data.geom_xmat[g].reshape(3, 3)
      best = min(best, float(pos[2] - abs(R[2, 0]) * hs[0] - abs(R[2, 1]) * hs[1]
                             - abs(R[2, 2]) * hs[2]))
    return best

  def _guard(self, err: np.ndarray, q_abs: np.ndarray) -> np.ndarray:
    """Floor guard on the true lowest pad corner (the pads hang 12 mm below the site only
    with the hand exactly vertical; pitched by 55-78 deg the outer corner drops 17-28 mm).

    W1-D4: this file runs ``cmd_ref = "command"`` with ``cmd_ref_axes = None``, so the base
    builds its IK target from FK(previous COMMAND) + err.  A guard evaluated only at the
    ACTUAL pose therefore lets the COMMAND sit a servo sag below the floor and the hand
    follows it down -- measured, ``ee_ground_collision`` in 5-6 of 8 envs, which
    TERMINATES the episode and is why the tip phase almost never got to run.  Strike-Slide
    already bounds both references; do the same here.
    """
    self._fk(q_abs)
    pad = self._pad_min_z()
    if self.cmd_lead_max > 0.0 and not np.any(np.isnan(self._q_cmd[self._guard_env])):
      q_ref = q_abs.copy()
      q_ref[:7] = np.clip(self._q_cmd[self._guard_env], q_abs[:7] - self.cmd_lead_max,
                          q_abs[:7] + self.cmd_lead_max)
      self._fk(q_ref)
      pad = min(pad, self._pad_min_z())
      self._fk(q_abs)
    err = err.copy()
    if pad + err[2] < self.PAD_FLOOR_MIN:
      err[2] = self.PAD_FLOOR_MIN - pad
    return err

  # -- geometry --------------------------------------------------------------------------

  def _contact_plan(self, i: int, obs_i: np.ndarray, tilt_cmd: float | None = None,
                    face_frac: float | None = None):
    """(target point on the board relative to the site, approach axis, closing axis).

    Everything is anchored on the PIVOT -- the board's far-bottom edge, which the wall
    pins -- not on its centre, because the centre rises as the board tips.  The tracked
    point walks from the board's near-TOP corner (tilt 0: the ramp pushes the top of the
    20 mm end face, the largest moment arm the geometry allows) to a point D_GRIP below
    the near edge on the ROBOT-side face (tilt = TILT_TARGET: the ramp is then flat under
    the board and the pinch closes on it)."""
    gto = self._gto(i, obs_i)
    x_obs, z_obs, tilt = _board_frame(obs_i)
    self._tilt_ema[i] = 0.3 * tilt + 0.7 * self._tilt_ema[i]
    pivot = gto + BOARD_HALF_X * x_obs - BOARD_HALF_Z * z_obs
    n_hat = np.array([x_obs[0], x_obs[1], 0.0])
    n_hat = n_hat / (np.linalg.norm(n_hat) + 1e-9)
    th = self._tilt_ema[i] if tilt_cmd is None else tilt_cmd
    t = float(np.clip(th / TILT_TARGET, 0.0, 1.0))
    psi = PSI_MAX + (PSI_MIN - PSI_MAX) * t
    a_hat, c_hat = _hand_axes(psi, n_hat)
    up = np.array([0.0, 0.0, 1.0])
    x_cmd = np.cos(th) * n_hat - np.sin(th) * up
    z_cmd = np.sin(th) * n_hat + np.cos(th) * up
    depth = D_GRIP * t
    ff = PUSH_FACE_Z if face_frac is None else face_frac
    face_z = 2.0 * BOARD_HALF_Z * (ff + (1.0 - ff) * t)
    p = pivot - (2.0 * BOARD_HALF_X - depth) * x_cmd + face_z * (1.0 - t) * z_cmd
    return p, a_hat, c_hat

  def _cmd_pos(self, i: int, q_abs: np.ndarray) -> np.ndarray:
    """Site position of the joint command the base will integrate from (same clipping the
    base applies), so a LEAD can be expressed as a bounded offset instead of a runaway."""
    if np.any(np.isnan(self._q_cmd[i])):
      return self._fk(q_abs)[0]
    q_ref = q_abs.copy()
    q_ref[:7] = np.clip(self._q_cmd[i], q_abs[:7] - self.cmd_lead_max,
                        q_abs[:7] + self.cmd_lead_max)
    return self._fk(q_ref)[0]

  def _ramp_site(self, i: int, p: np.ndarray, a_hat: np.ndarray,
                 c_hat: np.ndarray) -> np.ndarray:
    """Site position (relative to the current site) that puts ``p`` on the lower pad's
    inner face at S_FACE along the approach axis.

    The face's offset from the site is the OBSERVED finger position, not the commanded
    one.  MEASURED: contact against the board pushes the fingers off their target (the
    ramp is the pad's inner face, so its reaction is the finger's opening direction, and
    a pad that gets on top of the board is pushed the other way) -- traced, the aperture
    collapsed from a commanded 0.032 to 0.000, which walks the pad 16 mm forward and puts
    it on the board's top face, and the hand then rode over the board for the rest of the
    episode.  Referencing the geometry to the aperture that IS keeps the ramp where the
    plan wants it whatever the fingers are doing."""
    q = float(np.clip(0.5 * self._ap_ema[i], 0.004, 0.040))
    if PUSH_FACE_OUT and self._phase[i] in (P_SLIDE, P_TIP):
      # The upper finger's OUTER face (pad outer surface at q + PAD_OUT along +c).  Its
      # reaction on the finger is -c, i.e. the CLOSING direction, so contact cannot pry
      # the jaws open the way the inner-face ramp does.
      return p - (q + PAD_OUT) * c_hat - S_FACE * a_hat
    return p + q * c_hat - S_FACE * a_hat

  def _walk_xy(self, i: int, p_now: np.ndarray, tgt: np.ndarray) -> np.ndarray:
    """xy waypoint at APPROACH_RATE, leashed to the actual site.  MEASURED (W1-D2):
    handing the raw 17 cm error to the position loop makes it a high-gain integrator with
    a long dead time -- the hand sat motionless ~30 steps then covered 18 cm in 12."""
    if not self._wp_ok[i]:
      self._wp[i] = p_now.copy()
      self._wp_ok[i] = True
    dxy = tgt[:2] - self._wp[i][:2]
    n_xy = float(np.linalg.norm(dxy))
    if n_xy > APPROACH_RATE:
      dxy = dxy * (APPROACH_RATE / n_xy)
    self._wp[i][:2] = self._wp[i][:2] + dxy
    lead_xy = self._wp[i][:2] - p_now[:2]
    n_lead = float(np.linalg.norm(lead_xy))
    if n_lead > APPROACH_LEASH:
      self._wp[i][:2] = p_now[:2] + lead_xy * (APPROACH_LEASH / n_lead)
    return self._wp[i][:2] - p_now[:2]

  # -- phases ----------------------------------------------------------------------------

  def _target_error(self, i: int, obs_i: np.ndarray):
    err, axis, grip = self._target_error_axis(i, obs_i)
    if np.asarray(axis).shape == (3,):
      z = axis / max(np.linalg.norm(axis), 1e-9)
      y = self._c_target[i] - np.dot(self._c_target[i], z) * z
      if np.linalg.norm(y) > 1e-6:
        y /= np.linalg.norm(y)
        axis = np.column_stack([np.cross(y,z), y,z])
    return err, axis, grip

  def _target_error_axis(self, i: int, obs_i: np.ndarray):
    q_abs = self.default_qpos + obs_i[0:9]
    self._guard_env = i
    ph = self._phase[i]
    o2g = obs_i[43:46]

    if ph == P_LIFT:
      self._c_target[i] = np.array([1.0, 0.0, 0.0])
      rot = np.array([0.0, 0.0, -1.0])
      gto = self._gto(i, obs_i)
      if self._phase_steps[i] < CLIMB_STEPS:
        return np.array([0.0, 0.0, CLIMB_ERR]), rot, GRIPPER_CLOSED
      if np.linalg.norm(gto) > HELD_TOL and self._attempts[i] < MAX_ATTEMPTS:
        self._attempts[i] += 1
        self._rewind(i, f"lift gto={np.round(gto, 3).tolist()}")
        return np.array([0.0, 0.0, 0.05]), rot, GRIPPER_OPEN
      self._integrate(i, o2g, CARRY_INTEG_GAIN, CARRY_INTEG_CLIP)
      return o2g + self._integ[i], rot, GRIPPER_CLOSED

    p, a_hat, c_hat = self._contact_plan(i, obs_i)
    self._c_target[i] = c_hat
    tgt_rel = self._ramp_site(i, p, a_hat, c_hat)
    if ph in (P_HOVER, P_DESCEND):
      n_back = np.array([c_hat[0], c_hat[1], 0.0])
      tgt_rel = tgt_rel - APPROACH_BACK * n_back / (np.linalg.norm(n_back) + 1e-9)
    p_now, _R = self._fk(q_abs)
    up = np.array([0.0, 0.0, 1.0])

    if ph == P_HOVER:
      err = tgt_rel + up * (HOVER_Z - (p_now[2] + tgt_rel[2]))
      err_xy = self._walk_xy(i, p_now, p_now + err)
      err = np.array([err_xy[0], err_xy[1], err[2]])
      self._integrate(i, err)
      if np.linalg.norm(err[:2]) < ALIGN_TOL and abs(err[2]) < 0.04:
        self._settle[i] += 1
      else:
        self._settle[i] = 0
      if self._settle[i] >= ALIGN_SETTLE or self._phase_steps[i] > ALIGN_TIMEOUT:
        self._goto(i, P_DESCEND)
      return self._guard(err + self._integ[i], q_abs), a_hat, GRIPPER_STRADDLE

    if ph == P_DESCEND:
      # Rate-limiting the ERROR does not rate-limit a descent (the DLS realises ~40 % of a
      # small request): walk a leashed z waypoint and hand the servo the FULL error to it.
      z_goal = p_now[2] + tgt_rel[2]
      if not np.isfinite(self._zwp[i]):
        self._zwp[i] = p_now[2]
      rate = DESCENT_RATE_FAST if (self._zwp[i] - z_goal) > DESCENT_SLOW_BAND else DESCENT_RATE
      self._zwp[i] = max(z_goal, self._zwp[i] - rate, p_now[2] - DESCENT_LEASH)
      self._integrate(i, tgt_rel)
      cmd = tgt_rel + self._integ[i]
      cmd[2] = self._zwp[i] - p_now[2]
      if abs(p_now[2] - z_goal) < SEAT_TOL and np.linalg.norm(tgt_rel[:2]) < ALIGN_TOL:
        self._settle[i] += 1
      else:
        self._settle[i] = 0
      # UNCONDITIONAL step-count escape (phase-1 M3).
      if self._settle[i] >= 2 or self._phase_steps[i] > DESCEND_TIMEOUT:
        self._goto(i, P_SLIDE)
      return self._guard(cmd, q_abs), a_hat, GRIPPER_STRADDLE

    if ph == P_SLIDE:
      self.PAD_FLOOR_MIN = SLIDE_FLOOR_MIN
      p, a_hat, c_hat = self._contact_plan(i, obs_i, tilt_cmd=0.0)
      self._c_target[i] = c_hat
      n_hat = np.array([c_hat[0], c_hat[1], 0.0])
      n_hat = n_hat / (np.linalg.norm(n_hat) + 1e-9)
      pivot_x = float(p_now[0] + self._gto(i, obs_i)[0] + BOARD_HALF_X)
      jam_x = float(WALL_FACE_X - JAM_MARGIN)
      if self._xwp[i] <= 0.0:
        self._xwp[i] = pivot_x
      self._xwp[i] = min(self._xwp[i] + SLIDE_RATE, pivot_x + SLIDE_LEASH, jam_x)
      self._xwp[i] = max(self._xwp[i], pivot_x)
      cmd = self._ramp_site(i, p, a_hat, c_hat) + n_hat * float(self._xwp[i] - pivot_x)
      self._integrate(i, cmd)
      cmd = cmd + self._integ[i]
      need = jam_x - pivot_x
      n = float(np.linalg.norm(cmd))
      if n > SLIDE_RATE:
        cmd = cmd * (SLIDE_RATE / n)
      if need <= 0.0 or self._phase_steps[i] > SLIDE_TIMEOUT:
        self._goto(i, P_TIP)
      return self._guard(cmd, q_abs), a_hat, GRIPPER_STRADDLE

    if ph in (P_TIP, P_SEAT):
      self.PAD_FLOOR_MIN = SLIDE_FLOOR_MIN if ph == P_TIP else 0.015
      if ph == P_TIP:
        pivot_x = float(p_now[0] + self._gto(i, obs_i)[0] + BOARD_HALF_X)
        if pivot_x >= WALL_FACE_X - JAM_MARGIN:
          self._twp[i] = min(self._twp[i] + TILT_RATE, self._tilt_ema[i] + TILT_LEASH,
                             TILT_TARGET)
        self._twp[i] = max(self._twp[i], self._tilt_ema[i])
        # THE CLIMB.  MEASURED (W1-D4): with the slide fixed the board reaches the wall
        # and the ramp sits correctly on its 20 mm end face -- and nothing happens, tilt
        # 0.2-3 deg for 70 steps in 4 of 8 envs.  The tilt waypoint is leashed 0.09 rad
        # ahead of the OBSERVED tilt, and while the board is flat that caps the commanded
        # contact height at the board's own top corner, so the only thing driving the
        # board is a near-horizontal press -- which cannot tip it (a horizontal force at
        # the 10 mm mid-face needs m g 0.06 / 0.01 = 10 N).  The leash is a ratchet with
        # nothing to start it.  What starts it is FRICTION: the ramp climbing the end face
        # drags the near edge up with mu N at a 0.12 m arm, so ~1 N does the job.  The
        # climb is its own waypoint, walking the tracked point from mid-face up to just
        # below the near-top corner, which is exactly as far as it can go before the pad
        # slides over the top.
        self._fwp[i] = min(self._fwp[i] + CLIMB_FACE_RATE, CLIMB_FACE_MAX)
        press = TIP_PRESS
      else:
        # Keep the leash in SEAT too.  MEASURED: forcing the commanded tilt to
        # TILT_TARGET when the board is still flat drives the hand to a pose meant for a
        # standing board -- the pads reached z 0.0001-0.003 and ee_ground_collision fired
        # in 7/8 envs.
        self._twp[i] = min(TILT_TARGET, self._tilt_ema[i] + TILT_LEASH)
        press = SEAT_PRESS
      p, a_hat, c_hat = self._contact_plan(i, obs_i, tilt_cmd=float(self._twp[i]),
                                           face_frac=float(self._fwp[i]))
      self._c_target[i] = c_hat
      # The press is a COMMAND offset along the contact normal, but c_hat has a vertical
      # component (21 % at PSI_MAX) and while the board is still flat that offset lifts the
      # ramp's tip ABOVE the board's 20 mm top face and the hand rides over it (measured:
      # site z 0.0375, ramp tip 0.0215 vs a board top of 0.0202, board untouched in 8/8).
      # Below the target tilt the press is therefore laid down horizontally, which is the
      # right direction anyway: what it has to push then is the vertical end face.
      pv = c_hat.copy()
      pv[2] -= (1.0 - float(self._twp[i] / TILT_TARGET)) * c_hat[2]
      cmd = self._ramp_site(i, p, a_hat, c_hat) + press * pv
      # Referenced to the ACTUAL site (an integrator that walks the command past the servo
      # sag until the hand arrives), but with the COMMAND's lead over the actual site
      # capped: the tilt waypoint above is what bounds how far the ramp may run ahead of
      # the board, so nothing here needs an unbounded Cartesian lead.
      lead = self._cmd_pos(i, q_abs) + cmd - p_now
      n_l = float(np.linalg.norm(lead))
      if n_l > CMD_LEAD_MAX:
        cmd = p_now + lead * (CMD_LEAD_MAX / n_l) - self._cmd_pos(i, q_abs)
      # SCALE THE PRESS AND THE CLIMB SEPARATELY.  MEASURED (W1-D2, in the PUSH phase of
      # this same file, and again here): scaling the WHOLE command vector down to a rate
      # divides the small component by the same factor as the big one, so the 0.2 mm/step
      # climb that has to drag the board up disappears behind a 15 mm press.  Split the
      # command into the contact NORMAL (the press) and the rest (the climb along the
      # face plus the position error) and cap each on its own.
      cn = float(np.dot(cmd, c_hat))
      ct = cmd - cn * c_hat
      cn = float(np.clip(cn, -TIP_NORM_RATE, TIP_NORM_RATE))
      n_t = float(np.linalg.norm(ct))
      if n_t > TIP_TAN_RATE:
        ct = ct * (TIP_TAN_RATE / n_t)
      cmd = cn * c_hat + ct
      if ph == P_TIP:
        if self._tilt_ema[i] >= TILT_TARGET - 0.05 or self._phase_steps[i] > TIP_TIMEOUT:
          self._goto(i, P_SEAT)
      elif self._phase_steps[i] >= SEAT_STEPS:
        self._hold[i] = p_now + self._ramp_site(i, p, a_hat, c_hat) + press * pv
        self._goto(i, P_CLOSE)
      return self._guard(cmd, q_abs), a_hat, GRIPPER_STRADDLE

    # P_CLOSE
    err = self._hold[i] - p_now
    if self._phase_steps[i] >= CLOSE_STEPS:
      if self._ap_ema[i] < APERTURE_HELD and self._attempts[i] < MAX_ATTEMPTS:
        self._attempts[i] += 1
        self._rewind(i, f"close_aperture {self._ap_ema[i]:.3f}", P_TIP)
        return np.array([0.0, 0.0, 0.02]), a_hat, GRIPPER_STRADDLE
      self._goto(i, P_LIFT)
    return self._guard(err, q_abs), a_hat, GRIPPER_CLOSED
