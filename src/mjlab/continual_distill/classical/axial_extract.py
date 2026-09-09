"""Scripted AxialExtract teacher policy.

GEOMETRY (plug.xml, measured)
------------------------------
``plug_base`` is a static mocap mount whose x/y is drawn per-env (x in
[0.40, 0.48], y in [-0.10, 0.10] -- ``franka_axial_extract_env_cfg``'s
``reset_plug_position`` event) and whose z is fixed at 0.08
(``workspace.min_mechanism_mount_z("plug")`` = ``MECHANISM_DROP_BELOW_MOUNT["plug"]``
(0.05) + ``MECHANISM_FLOOR_CLEARANCE`` (0.03)). Four static box walls
(contype=2/conaffinity=1 -- collide with the robot but NOT with each other or the
shaft, see plug.xml's comment) form a socket around a vertical bore, top at
mount_z + 0.05 = 0.13. A ``handle`` body carries the slide joint (axis +z, range
[0, 0.12], damping 1.0, ``frictionloss`` 4.0 -- nothing moves until the pull
exceeds the ~4N breakaway) and the pinchable head: a 4cm-diameter, 2.4cm-tall
cylinder whose CENTRE sits 0.062m above (mount + current slide displacement q).
At q=0 (spawn) the head therefore sits at absolute z = mount_z + 0.062 = 0.142 --
resting right on top of the socket walls (top z = 0.13), comfortably inside the
Franka's measured 0.06-0.14 top-down grasp band (see workspace.py). Diameter
0.04 is well under the 0.08 aperture. **The head's x/y NEVER changes** -- the
slide joint only moves along +z -- so once gripped there is no lateral tracking
problem, only a vertical pull.

SUCCESS (AxialExtractCommand, an ``_ArticulationJointCommand``)
-----------------------------------------------------------------
Success is measured on the SLIDE JOINT VALUE directly
(``self.asset.data.joint_pos[:, joint_idx]``), NOT on any Cartesian distance:
``target_value=0.10`` (near the 0.12 physical stop), ``directional=True``
(default -- overshoot counts as success, only shortfall is penalised),
``success_threshold=0.02``. So the episode is won the instant q >= 0.08 -- 80%
of the way to the target, 67% of the way to the stop -- and success LATCHES
(``torch.maximum`` in ``_update_metrics``), so a single crossing is enough even
if the grip subsequently slips or the arm is pulled back down afterward.

Mechanism runs gravity-OFF (CONTEXT.md sec 6: every mocap-mounted mechanism
task does), and the slide joint has no return spring -- so once broken free
there is nothing pulling the plug back down; only our own policy could undo it.

STRATEGY -- pinch, break the friction, pull straight up
---------------------------------------------------------
A four-phase top-down pick, structurally the same spine as
``lift_object.py`` (hover -> vertical descend -> close -> lift) but with NO
carry: the pull target is purely vertical, and because the head cannot move
laterally, the pull phase does not need to re-track a moving object in x/y at
all -- it only has to keep applying a SUSTAINED upward lead. The lead is a
constant offset ABOVE the current tracked head position, recomputed fresh
every step from the (smoothed) relative observation -- exactly the mechanism
``open_drawer.py`` uses to pull an articulated joint at speed: the commanded
waypoint never "arrives" (it is always PULL_LEAD metres above wherever the
head currently reads), so the position servo keeps generating force against
the joint's static friction instead of relaxing to zero error the moment the
wrist's own compliance absorbs a few millimetres.

OBSERVATIONS: 60-D layout (see ``axial_extract_env_cfg.py``, term-by-term).
Only RELATIVE terms are used, so the per-env scene-origin offset cancels:
    [40:43] gripper_to_object   <- plug head (object_site) - gripper site
    [43:46] object_to_goal      <- target_pos - plug head
``target_pos`` is a Cartesian proxy the command computes once per episode at
``base_site + (0, 0, 0.10)`` (AxialExtractCommandCfg.goal_marker_offset),
i.e. exactly where the head sits when q=0.10 -- it is NOT used by this policy
(success depends on the joint value, not on reaching it), but is documented
here for completeness. Absolute terms (object_pos, gripper_pos, obs[18:21],
obs[25:28]) carry the per-env scene origin and are never used.

Robot starts at NEUTRAL_QPOS (this task uses ``get_franka_robot_cfg_neutral``,
the same convention as the door/drawer/button/flap group), with a +-10deg
joint reset offset each episode.
"""

from __future__ import annotations

import numpy as np

from mjlab.continual_distill.classical.base import NEUTRAL_QPOS, ClassicalPolicyBase

_DOWN_AXIS = np.array([0.0, 0.0, -1.0])
GRIPPER_OPEN = 1.0
GRIPPER_CLOSED = -1.0

# Phases.
P_HOVER = 0
P_DESCEND = 1
P_CLOSE = 2
P_PULL = 3

# Site altitude above the head while hovering/aligning in xy before committing
# to a vertical descent -- same "get high and centred first" idiom as
# lift_object.py, to keep the descent a clean vertical line.
HOVER_LEAD = 0.13
ALIGN_TOL = 0.020
ALIGN_TIMEOUT = 45
SEAT_TOL = 0.014
DESCENT_RATE = 0.035
CLOSE_STEPS = 14
# Sustained upward lead during the pull: a constant RELATIVE offset above the
# CURRENT tracked head position (see module docstring for why this, and not
# the raw remaining ``object_to_goal``, is what generates a persistent,
# non-vanishing servo force against the joint's frictionloss).
PULL_LEAD = 0.05
EMA_ALPHA = 0.4
# If gripper_to_object jumps by more than this in one step, the episode
# auto-reset under us (mirrors the check in lift_object.py / stack_object.py):
# without rewinding, the state machine keeps "pulling" a socket it never
# actually gripped in the new episode.
RESET_JUMP = 0.12
# General-purpose floor guard (workspace never approaches the true ground here
# -- everything happens at site z >= ~0.14 -- but kept as a cheap safety net,
# same value as every other task's general case; NOT the peg-insertion 0.022
# special case, which does not apply here).
FLOOR_MIN_Z = 0.030


class AxialExtractClassicalPolicy(ClassicalPolicyBase):
  """Top-down pinch of the plug head; sustained vertical pull past breakaway."""

  DEFAULT_QPOS = NEUTRAL_QPOS  # axial_extract uses get_franka_robot_cfg_neutral
  max_dq = 0.15
  orientation_weight = 0.3

  def __init__(self, num_envs: int):
    super().__init__(num_envs)
    # MEASURED MECHANISM BUG, not a tuning knob: with NEUTRAL_QPOS's own posture
    # (joint2 = -1.0) as the posture-regularization target, the DLS descent to
    # this task's grasp height (site z ~= 0.142, at ANY mount x in [0.30, 0.48])
    # reliably converges to a genuine STATIONARY POINT of the weighted
    # least-squares cost ~2.2cm short of the target -- confirmed independent of
    # damping (0.02-0.2), kp_task (1-2) and x -- i.e. a local-minimum "wrong IK
    # branch" trap, not a rate-limit or a real joint/collision limit (joint4
    # sits at -2.5..-2.6 of a [-3.07,-0.07] range, 0.5rad of headroom unused).
    # DEFAULT_QPOS must stay NEUTRAL_QPOS (obs[0:9] is joint_pos_rel against
    # THIS task's own robot cfg, get_franka_robot_cfg_neutral -- changing it
    # would silently misread every observation). But the POSTURE TARGET is a
    # separate, low-weight (0.005) tie-breaker in the null space, not tied to
    # obs correctness. Biasing only its joint2 component toward HOME_QPOS's
    # value (0.3, vs NEUTRAL's -1.0) tips the solver into a different
    # (elbow-down) branch that closes to within 0.4-0.6cm of the target across
    # the whole mount x range -- comfortably inside SEAT_TOL (0.012). Measured
    # in isolation (module-free FK probe, not this class) before being kept
    # here; changing joint7's sign alone (the other big NEUTRAL/HOME
    # difference) had NO effect, so only joint2 is touched.
    self._posture_target[1] = 0.3

  def reset(self, env_ids=None) -> None:
    super().reset(env_ids)
    if env_ids is None:
      self._ema = np.zeros((self.num_envs, 3))
      self._ema_ok = np.zeros(self.num_envs, dtype=bool)
      self._settle = np.zeros(self.num_envs, dtype=np.int64)
      self._prev_gto = np.zeros((self.num_envs, 3))
    else:
      self._ema_ok[env_ids] = False
      self._settle[env_ids] = 0

  def _rewind(self, i: int) -> None:
    self._phase[i] = P_HOVER
    self._phase_steps[i] = 0
    self._settle[i] = 0
    self._ema_ok[i] = False

  def _smooth(self, i: int, raw: np.ndarray) -> np.ndarray:
    if not self._ema_ok[i]:
      self._ema[i] = raw
      self._ema_ok[i] = True
    else:
      self._ema[i] = EMA_ALPHA * raw + (1 - EMA_ALPHA) * self._ema[i]
    return self._ema[i]

  def _guard(self, err: np.ndarray, obs_i: np.ndarray) -> np.ndarray:
    """Clamp the commanded displacement so the site never targets below the
    floor. Site height recovered via FK on the arm's OWN joints (robot-base
    frame), so it carries no per-env scene-origin offset -- legal in a way
    reading an absolute observation term is not (see base.py docstring)."""
    q_abs = self.default_qpos + obs_i[0:9]
    z = float(self._fk(q_abs)[0][2])
    if z + err[2] < FLOOR_MIN_Z:
      err = err.copy()
      err[2] = FLOOR_MIN_Z - z
    return err

  def _target_error(self, i: int, obs_i: np.ndarray):
    raw_gto = obs_i[40:43]  # plug head - gripper
    if self._ema_ok[i] and np.linalg.norm(raw_gto - self._prev_gto[i]) > RESET_JUMP:
      self._rewind(i)
    self._prev_gto[i] = raw_gto

    gto = self._smooth(i, raw_gto)
    phase = self._phase[i]

    if phase == P_HOVER:
      pos_err = gto + np.array([0.0, 0.0, HOVER_LEAD])
      if np.linalg.norm(pos_err[:2]) < ALIGN_TOL and abs(pos_err[2]) < 0.05:
        self._settle[i] += 1
        if self._settle[i] >= 2:
          self._phase[i] = P_DESCEND
          self._phase_steps[i] = 0
          self._settle[i] = 0
      else:
        self._settle[i] = max(0, self._settle[i] - 1)
      if self._phase_steps[i] > ALIGN_TIMEOUT:
        self._phase[i] = P_DESCEND
        self._phase_steps[i] = 0
        self._settle[i] = 0
      return self._guard(pos_err, obs_i), _DOWN_AXIS, GRIPPER_OPEN

    if phase == P_DESCEND:
      # NEGATIVE RESULT, recorded so nobody re-tries it: lateral INTEGRAL
      # action here -- the idiom every other multi-phase teacher in this
      # package uses to null the DLS steady-state bias (open_drawer.py,
      # rotate_valve.py, turn_lever.py, tool_pull.py, stack_object.py's carry)
      # -- was tried and measured WORSE, not better, at every gain/clip tried
      # (0.22/0.05, 0.08/0.02; xy-only or all 3 axes): 0.78 -> 0.03-0.28 (n=32).
      # Symptom differed from the plain-proportional failure mode too: without
      # the integrator, P_PULL failures were a clean bimodal split (grip fully
      # holds and extracts, or completely misses, max_q ~= 0); WITH it, most
      # envs got a small nonzero q (0.001-0.06) and then slipped -- i.e. the
      # correction was landing a WEAKER, off-centre pinch on the 2cm-radius
      # cylindrical head rather than no pinch at all, and that partial grip
      # could not survive the sustained pull. Carrying the integrator into
      # P_CLOSE/P_PULL made it worse again (dragging the site sideways right
      # as the pinch formed, or working an established grip loose as the arm's
      # pose -- and hence the true kinematic bias -- changed during the
      # ascent), but even DESCEND-only integration underperformed the plain
      # version. Kept at plain proportional plus the DESCENT_RATE clamp.
      pos_err = gto.copy()
      pos_err[2] = max(pos_err[2], -DESCENT_RATE)
      if np.linalg.norm(gto) < SEAT_TOL:
        self._settle[i] += 1
        if self._settle[i] >= 2:
          self._phase[i] = P_CLOSE
          self._phase_steps[i] = 0
      else:
        self._settle[i] = 0
      if self._phase_steps[i] > 120:  # stalled: squeeze anyway rather than idle
        self._phase[i] = P_CLOSE
        self._phase_steps[i] = 0
      return self._guard(pos_err, obs_i), _DOWN_AXIS, GRIPPER_OPEN

    if phase == P_CLOSE:
      # Mild residual correction while squeezing -- see lift_object.py's
      # P_CLOSE for the same "do not command zero, do not command full gto
      # either" compromise (zero lets the arm sag; full gto keeps chasing
      # sensor noise).
      pos_err = gto * 0.5
      if self._phase_steps[i] >= CLOSE_STEPS:
        self._phase[i] = P_PULL
        self._phase_steps[i] = 0
      return self._guard(pos_err, obs_i), _DOWN_AXIS, GRIPPER_CLOSED

    # P_PULL: sustained upward lead, recomputed every step from the fresh
    # smoothed relative position -- see PULL_LEAD in the module docstring.
    # Lateral term is the RAW gto, not integral-corrected: once genuinely
    # gripped the head cannot move laterally (see module docstring), so gto's
    # xy should already read ~0, and MEASURED carrying the stale
    # descend-phase integrator through the pull instead actively worked the
    # grip loose as the arm's pose (and hence the true kinematic bias) changed
    # during the ascent -- symptom was many envs breaking friction (small
    # nonzero q) then slipping well short of the 0.08 threshold.
    pos_err = np.array([gto[0], gto[1], gto[2] + PULL_LEAD])
    return self._guard(pos_err, obs_i), _DOWN_AXIS, GRIPPER_CLOSED
