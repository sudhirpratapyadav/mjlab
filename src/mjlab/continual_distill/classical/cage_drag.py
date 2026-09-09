"""Scripted CageDrag teacher policy.

Motion profile: FORM-CLOSURE (CAGING) TRANSPORT. The gripper straddles the
cube with the fingers held OPEN and translates; the object is constrained
geometrically, never force-closed.

*** THE TRAP (read before touching GRIPPER_OPEN below) ***
``CageDragCommand`` (mdp/commands.py) latches ``min_aperture`` — the MINIMUM,
not maximum, of ``joint_pos[finger_joint1] + joint_pos[finger_joint2]`` over the
whole episode — and ``compute_success`` requires ``min_aperture > aperture_min``
(0.055) in addition to the usual goal-distance check:

    def compute_success(self) -> torch.Tensor:
      return (self.metrics["goal_error"] < self.cfg.success_threshold) & (
        self.min_aperture > self.cfg.aperture_min
      )

Every other latch in this repo is a success-MAXIMUM (``torch.maximum`` on
``episode_success``, monotonically easier to satisfy over time). This is the
opposite: the moment the combined finger opening dips below 0.055 ANYWHERE in
the episode, the run is voided PERMANENTLY, silently, with no error and no
recovery — closing the fingers even once, even briefly, even after a
successful placement, loses the episode.

The mitigation here is not cleverness, it is refusal: this policy NEVER emits
a gripper action below ``GRIPPER_OPEN`` and never varies it. ``_target_error``
returns the same constant every call, and ``__call__`` overrides ``action[7]``
unconditionally at the end as a second, independent guard (belt-and-braces —
the state machine has no path that could set it otherwise, but a future edit
should not be able to reintroduce one silently). Franka fully open is
joint 0.04 + 0.04 = 0.08 combined; commanding max-open the whole episode keeps
``min_aperture`` at ~0.08, 0.025 clear of the 0.055 threshold, independent of
contact forces (the cube, 0.05 kg, cannot force the position servo — kp=350 —
closed).

TRANSPORT MECHANISM. With the fingers pinned open (aperture ~0.08) the two
fingertip pads sit ~0.05 m either side of the ``gripper`` site (joint travel
0.04 + ~0.01 pad offset — see push_cuboid.py's measured PAD_RADIUS). The cube
is only 0.04 m wide (half-extent 0.02), so an aperture this wide does NOT by
itself cage the cube against a single flat face the way push_cuboid's CLOSED
fingers do — the object has ~2.5cm of clearance to spare on each side at
dead-centre. Two things make transport work regardless:

  1. Orientation. Unlike push_cuboid (axis-only, yaw free — a closed pusher is
     radially symmetric so yaw does not matter), here yaw MATTERS: the EE
     x-axis is the finger-opening axis (verified in stack_object.py's
     GEOMETRY note), so the wrist must be yawed to put that axis along the
     object->goal direction. ``down_frame(yaw)`` (copied locally from
     stack_object.py to avoid a cross-file dependency on another agent's
     file) builds the full rotation target.
  2. Contact-point servo. Exactly push_cuboid's v3 strategy (never command a
     point inside the object; servo the trailing contact face and creep it
     toward the goal), with the pusher standoff widened from
     push_cuboid's ~0.014 m (closed fingers) to ~0.05 m (open fingers) and the
     object half-width shrunk from the cuboid's 0.0405 to the cube's 0.023. The
     trailing pad is what drags the object; the leading pad (0.10 m further
     out) never gets close enough to interfere for any goal inside the task's
     workspace. This still counts as "caging" in the sense the task intends
     (transport happens via lateral straddling contact, not a force-closure
     grasp) even though only one pad typically loads at a time.

MEASURED MECHANISM FINDINGS (HEAD 1127d12).

1. RIDE_HEIGHT, same bug family as drag_pull.py. Starting from a naively
   scaled push_cuboid offset (0.033) the teacher stalled/oscillated; a small
   sweep at n=32 (0.022/0.026/0.030/0.038 -> 0.188/0.281/0.156-0.219/0.062)
   showed a clear peak at 0.026 (pads riding mid-face on the cube, half-height
   0.02 at the time); CL-V2 carries that 65%-of-half-height fraction onto the
   46 mm cube as 0.028. Confirms the general lesson from
   drag_pull/push_cuboid: this "pad height above object centre" family of
   constants does not port between objects even when the rest of the strategy
   does, and is worth a real sweep rather than a single guess.
2. YAW-LOCK MUST BE WHOLE-EPISODE, not per-phase. First attempt used
   axis-only orientation (yaw free) while far/descending and only committed
   to the full yaw-locked frame once the contact-servo phase started, on the
   open_door.py precedent ("a full 3x3 makes IK crawl" while far). MEASURED
   WORSE: 0.156 -> 0.000 at n=32 — the abrupt reorientation exactly at the
   phase boundary reproduced the same kind of violent wrist swing the lock
   was meant to prevent, just relocated to a different step. Reverted to
   locking the full frame from the very first control step of the episode
   (see the yaw-lock comment in ``_target_error``); this is what ships.
3. What the yaw lock actually fixes, confirmed by instrumentation: with yaw
   recomputed every step from the live (noisy) direction, one env's `along`
   (healthy around +0.03..+0.08) spiked to -0.32 mid-episode (the gripper
   swept through the cube) and `min_aperture` read 0.053, BELOW aperture_min
   (0.055) — a genuine physical over-limit event from contact/momentum,
   despite the commanded gripper action being pinned open the entire time.
   After the fix this class of violent event was not reproduced in spot
   checks, though isolated smaller dips (e.g. one env at 0.032, still a
   real — if rarer — violation) were still observed; this is reflected
   directly in the measured SR below, not a separate hidden failure.

FINAL MEASURED SR: **0.141 (n=128)**, HEAD 1127d12. Below the 0.90 bar. The
residual failure is a mix of: (a) the same friction-limited "position servo
holds a steady offset but does not always generate enough sustained force"
mechanism as push_cuboid/drag_pull, now compounded by a much smaller object
(cube half-width 0.02 vs cuboid's 0.04) giving less margin for cross-track
error before the trailing pad slips off the object's edge; (b) residual
aperture violations from hard, glancing contact during the hover/descend
approach (see finding 3) which void an otherwise-successful transport
after the fact — this is inherent to a wide-open, momentum-sensitive
contact geometry and was only partially mitigated by the yaw lock, not
eliminated. See LOGS.md for the full run history.

Observation layout (60 dims, identical structure/order to drag_pull/
push_cuboid, object=cube, command=cage_drag):
  [ 0: 9] robot_joint_pos          [ 9:18] robot_joint_vel
  [18:21] object_pos               [21:25] object_quat
  [25:28] gripper_pos              [28:34] gripper_orientation
  [34:40] object_orientation
  [40:43] gripper_to_object  <- USE  [43:46] object_to_goal      <- USE
  [46:52] goal_orientation_diff     [52:60] control_qpos_diff
"""

from __future__ import annotations

import numpy as np

from mjlab.continual_distill.classical.base import HOME_QPOS, ClassicalPolicyBase

# Never anything else. See the module docstring's "THE TRAP" section.
GRIPPER_OPEN = 1.0


def down_frame(yaw: float) -> np.ndarray:
  """Full EE rotation matrix: z-axis straight down, finger axis (EE x) at
  ``yaw`` in the world xy-plane. Local copy of stack_object.py's helper (not
  imported, to keep this file self-contained per the one-agent-one-file rule).
  """
  c, s = np.cos(yaw), np.sin(yaw)
  ex = np.array([c, s, 0.0])
  ez = np.array([0.0, 0.0, -1.0])
  ey = np.cross(ez, ex)
  return np.column_stack([ex, ey, ez])


HOVER_HEIGHT = 0.15
# Site height above the cube CENTRE (centre at z=0.0226, half-height 0.0226,
# full height 0.045). push_cuboid found the pad-below-site offset to be ~0.013 m for
# its (different) fingertip geometry when closed; the same pads at max-open
# sit slightly lower again once the joint offset is folded in. Start from the
# same relative offset and recalibrate from a measured rollout if the
# fingertip pads graze the top of the cube or miss it.
# Re-derived for the 46 mm cube (half-height 0.0226, was 0.020). The measured
# optimum 0.026 on the 40 mm cube put the pads (~0.013 below the site) at 0.013
# above the centre = 65% of the half-height; holding that fraction on the new cube
# gives 0.65 * 0.0226 + 0.013 = 0.0277.
RIDE_HEIGHT = 0.028
# Stand-off on the far side of the object (opposite the goal) while hovering /
# descending, before the contact-point servo takes over.
BEHIND = 0.08
ALIGN_TOL = 0.04
# Success is |object - goal| < 0.03 (success_threshold in make_cage_drag_env_cfg).
GOAL_TOL = 0.025
DESCENT_RATE = 0.05

# -- contact-point servo (see cage_drag module docstring, transport mechanism) -
# cube.xml: YCB 077_rubiks_cube at 46 mm, collision box 0.046 x 0.0453 x 0.0451,
# so 0.023 half-width along the pushed axis (was 0.02 on the 40 mm primitive).
# The wider face is a net GAIN here: the module docstring names "less margin for
# cross-track error before the trailing pad slips off the object's edge" as one of
# the two residual failure modes, and this asset gives 15% more of that margin.
HALF_WIDTH = 0.023
# Fingertip contact offset from the ``gripper`` site with fingers held at
# MAX OPEN: joint travel (0.04) + the same pad protrusion push_cuboid measured
# at joint=0 (0.010) — the pad sits at a roughly fixed offset from the finger
# body, which itself sits at the joint value from the site.
FINGER_JOINT_MAX = 0.04
PAD_OFFSET = 0.010
EFFECTIVE_STANDOFF = FINGER_JOINT_MAX + PAD_OFFSET  # ~0.05
STANDOFF = 0.004  # clearance between the pad and the cube face at zero advance
GAIN = 0.60
STEP_MIN = 0.012
STEP_MAX = 0.030
FINE_ZONE = 0.06
FINE_GAIN = 0.35
CROSS_GAIN = 0.8
# Minimum "behind-ness" before the trailing pad is considered to have lost
# contact and the pusher re-seats. See push_cuboid.py for the derivation.
MIN_BEHIND = 0.012
RESEAT_STEPS = 12
FLOOR_MIN_Z = 0.030


class CageDragClassicalPolicy(ClassicalPolicyBase):
  """Straddle the cube with fingers pinned open and drag it to the goal.

  The gripper action is a hard constant (GRIPPER_OPEN) for the entire
  episode — see the module docstring. Never edit ``_target_error`` to return
  anything else in slot 7, and never remove the ``__call__`` override below.
  """

  DEFAULT_QPOS = HOME_QPOS  # cage_drag env uses get_franka_robot_cfg (home)
  max_dq = 0.15
  step_clip_mode = "per_joint"

  def reset(self, env_ids=None) -> None:
    super().reset(env_ids)
    if env_ids is None:
      self._reseat = np.zeros(self.num_envs, dtype=np.int64)
      self._yaw = np.zeros(self.num_envs, dtype=np.float64)
      self._yaw_locked = np.zeros(self.num_envs, dtype=bool)
    else:
      self._reseat[env_ids] = 0
      self._yaw_locked[env_ids] = False

  def __call__(self, obs: np.ndarray) -> np.ndarray:
    actions = super().__call__(obs)
    # Second, independent guard against ever closing the fingers (belt and
    # braces on top of _target_error always returning GRIPPER_OPEN — see the
    # module docstring's "THE TRAP" section). This assert is intentionally a
    # hard failure, not a clamp: silently clamping would hide a state-machine
    # bug that ALSO needs fixing, since a bug that only gets caught here would
    # still have driven a wrong contact geometry for a step.
    assert np.all(actions[:, 7] >= GRIPPER_OPEN - 1e-6), (
      "cage_drag policy attempted to close the gripper — this would silently "
      "void the episode via CageDragCommand's min-aperture latch"
    )
    actions[:, 7] = GRIPPER_OPEN
    return actions

  def _target_error(self, i: int, obs_i: np.ndarray):
    gto = obs_i[40:43]  # object - gripper
    o2g = obs_i[43:46]  # goal - object

    self.cmd_lead_max = 0.0

    o2g_xy = o2g[:2]
    dist_goal = np.linalg.norm(o2g_xy)
    d = o2g_xy / (dist_goal + 1e-8)

    # WRIST-YAW LOCK. Orientation is a FULL rotation target here (unlike
    # push_cuboid/drag_pull's axis-only DOWN, which leaves yaw free) because
    # the finger-opening axis must sit along the push direction for the
    # straddle to work at all — see the module docstring. Recomputing yaw
    # every step from the live (noisy, +-1cm) object/goal observation is
    # dangerous: as dist_goal shrinks near the goal, atan2 becomes unstable
    # and can flip by a large angle between consecutive steps, commanding a
    # violent wrist swing. MEASURED: with yaw recomputed every step, one env's
    # `along` (component of object-minus-gripper on the push axis, healthy
    # around +0.03..+0.08) spiked to -0.32 mid-episode — the gripper swept
    # THROUGH the cube at speed — and `min_aperture` dropped to 0.053, BELOW
    # aperture_min (0.055), i.e. a real, physical aperture violation despite
    # the gripper action being pinned open the entire time (contact/momentum
    # can still force the position servo briefly closed on a hard enough
    # strike). Locking yaw once, on the first control step of each episode
    # (the direction is well-defined then — full dist_goal available, no
    # near-goal noise) and holding it fixed removes the whiplash. It does NOT
    # need to track the live direction precisely: the push only needs to be
    # roughly aligned with object->goal for the trailing pad to stay in
    # contact; CROSS_GAIN below corrects small residual misalignment in
    # translation.
    if not self._yaw_locked[i]:
      self._yaw[i] = float(np.arctan2(d[1], d[0]))
      self._yaw_locked[i] = True
    # MEASURED (see module docstring, TRANSPORT MECHANISM / yaw-lock note):
    # switching orientation targets between phases (axis-only while
    # approaching, full frame once shepherding) was tried and made things
    # WORSE (0.156 -> 0.000 at n=32) — the abrupt reorientation exactly AT the
    # phase boundary produced the same kind of violent swing the yaw lock
    # itself was meant to prevent, just relocated to a different step. The
    # full yaw-locked frame is used for the WHOLE episode instead, from the
    # first control step (when the direction estimate is most reliable).
    target_rot = down_frame(self._yaw[i])
    behind = np.array([-d[0] * BEHIND, -d[1] * BEHIND, 0.0])

    if dist_goal < GOAL_TOL or self._phase[i] == 3:
      self._phase[i] = 3
      pos_err = gto + behind + np.array([0.0, 0.0, HOVER_HEIGHT])
    elif self._phase[i] == 0:
      pos_err = gto + behind + np.array([0.0, 0.0, HOVER_HEIGHT])
      if np.linalg.norm(pos_err[:2]) < ALIGN_TOL and abs(pos_err[2]) < 0.08:
        self._phase[i] = 1
    elif self._phase[i] == 1:
      pos_err = gto + behind + np.array([0.0, 0.0, RIDE_HEIGHT])
      if np.linalg.norm(pos_err) < ALIGN_TOL:
        self._phase[i] = 2
    else:
      along = float(gto[:2] @ d)
      n = np.array([-d[1], d[0]])
      cross = float(gto[:2] @ n)

      if self._reseat[i] > 0:
        self._reseat[i] -= 1
        pos_err = gto + 1.6 * behind + np.array([0.0, 0.0, RIDE_HEIGHT])
      elif along < MIN_BEHIND:
        self._reseat[i] = RESEAT_STEPS
        pos_err = gto + 1.6 * behind + np.array([0.0, 0.0, RIDE_HEIGHT])
      else:
        seat_back = HALF_WIDTH + EFFECTIVE_STANDOFF + STANDOFF
        step = float(np.clip(GAIN * dist_goal, STEP_MIN, STEP_MAX))
        if dist_goal < FINE_ZONE:
          step = min(step, dist_goal * FINE_GAIN)
        pos_err = gto + np.array([
          -d[0] * seat_back + d[0] * step - n[0] * cross * CROSS_GAIN,
          -d[1] * seat_back + d[1] * step - n[1] * cross * CROSS_GAIN,
          RIDE_HEIGHT,
        ])

    pos_err[2] = max(pos_err[2], -DESCENT_RATE)

    q_abs = self.default_qpos + obs_i[0:9]
    z = float(self._fk(q_abs)[0][2])
    if z + pos_err[2] < FLOOR_MIN_Z:
      pos_err[2] = FLOOR_MIN_Z - z
    return pos_err, target_rot, GRIPPER_OPEN
