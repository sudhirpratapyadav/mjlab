"""Scripted FlipSwitch teacher — BALLISTIC COMMIT past the over-centre point.

Geometry (switch.xml, verified live): a hinge about world +y (the mount has no
yaw randomisation), range [-45 deg, +45 deg], starting at -45 deg. The toggle is
an INVERTED PENDULUM: a light shaft with a 50 g weight at body-local
(0, 0, 0.062), well above the pivot. Centre is therefore an UNSTABLE equilibrium
— gravity torque grows with tilt, so the toggle accelerates toward whichever stop
it is leaning to. ``object_site`` is at the weight. Measured with the toggle at
its -45 deg start: the tip sits at z = 0.134, about 4.4 cm toward the robot from
the pivot; the pivot itself is at z = 0.152.

Success: hinge past +30 deg (shortfall-only, threshold 0.15 rad), latched.

Strategy — this is the one task where servoing to a goal is the WRONG answer. The
detent (here, the over-centre weight) pushes back on any gentle quasi-static push,
and a policy that converges on a target pose stalls at the balance point. So:
align the closed pad on the toggle's -x face at tip height, then execute a single
open-loop BALLISTIC stroke — a fixed +x waypoint driven at the solver's maximum
stride, held for a fixed number of steps regardless of what the observation says.
Once the tip crosses centre, the inverted-pendulum torque finishes the flip on its
own and the +45 deg stop catches it, so the stroke does not need to be accurate,
only decisive and committed. The follow-through is what buys the last 30 deg.

There is no grasp anywhere in this policy, so the domain-randomised fingertip
friction (as low as 0.3) is irrelevant: the contact is pure face-normal push.

Only relative terms are used (gto = obs[40:43], the toggle tip minus the gripper),
so per-env scene-origin offsets cancel.
"""

from __future__ import annotations

import numpy as np

from mjlab.continual_distill.classical.base import ClassicalPolicyBase

# EE z-axis tilted forward from straight down. A strictly vertical wrist at this
# low, far-out pose sits near the edge of the Franka workspace; the forward tilt
# also points the pad's face more squarely into the toggle's -x side.
_APPROACH_AXIS = np.array([0.55, 0.0, -0.835])

GRIPPER_CLOSED = -1.0

# Standoff from the tip to the pad's pushing face, along -x.
PUSH_STANDOFF = 0.030
# Contact slightly BELOW the tip: nearer the shaft, so the pad cannot skid over
# the top of the weight block during the stroke.
CONTACT_Z = -0.012
HOVER_X = 0.07
HOVER_Z = 0.05
ALIGN_TOL = 0.035  # y-z alignment before closing the last of the -x gap
CONTACT_TOL = 0.05
# Signed seat gates (see phase 1). The pad must be within SEAT_Z_TOL vertically
# and no further than SEAT_X_MAX short of the face, while still being BEHIND it.
SEAT_Z_TOL = 0.030
SEAT_X_MAX = 0.055
EMA_ALPHA = 0.4
INTEG_GAIN = 0.22
SETTLE = 3
# Hard timeouts: this task has only 150 steps and the commit stroke must fire.
# Without them a persistent DLS steady-state bias parks the arm in an approach
# phase forever (measured 0/32 before these were added).
ALIGN_TIMEOUT = 60
SEAT_TIMEOUT = 40

# The commit stroke. STROKE_X is far enough past the toggle that the solver never
# converges and keeps driving; STROKE_STEPS is the follow-through that carries the
# toggle from centre to the +45 stop after gravity takes over.
STROKE_X = 0.16
STROKE_Z = -0.03  # slightly downward: the tip arcs DOWN in x as it crosses centre
STROKE_STEPS = 30


class FlipSwitchClassicalPolicy(ClassicalPolicyBase):
  """Align on the toggle's near face, then commit with one ballistic +x stroke."""

  max_dq = 0.20  # deliberately high: the stroke must be fast, not accurate

  def reset(self, env_ids=None) -> None:
    super().reset(env_ids)
    if env_ids is None:
      self._settle = np.zeros(self.num_envs, dtype=np.int64)
      self._ema = np.zeros((self.num_envs, 3))
      self._ema_init = np.zeros(self.num_envs, dtype=bool)
      self._integ = np.zeros((self.num_envs, 3))
    else:
      self._settle[env_ids] = 0
      self._ema_init[env_ids] = False
      self._integ[env_ids] = 0.0

  def _target_error(self, i: int, obs_i: np.ndarray):
    gto_raw = obs_i[40:43]
    if not self._ema_init[i]:
      self._ema[i] = gto_raw
      self._ema_init[i] = True
    else:
      self._ema[i] = EMA_ALPHA * gto_raw + (1 - EMA_ALPHA) * self._ema[i]
    gto = self._ema[i]

    # Where the gripper site must sit to have its pad on the toggle's -x face.
    seat = gto + np.array([-PUSH_STANDOFF, 0.0, CONTACT_Z])

    gripper_a = GRIPPER_CLOSED

    if self._phase[i] == 0:
      # Stand off in -x and above; align in y and z before approaching.
      hover = seat + np.array([-HOVER_X, 0.0, HOVER_Z])
      self._integ[i] = np.clip(self._integ[i] + INTEG_GAIN * hover, -0.08, 0.08)
      pos_err = hover + self._integ[i]
      # Gate on lateral (y) alignment only, plus a hard timeout. The DLS solve
      # keeps a few-cm steady-state bias at this low, far-out pose, so a tolerance
      # on the full y-z residual never trips and the arm hovers for the whole
      # episode — the exact failure this replaced (measured: 0/32).
      if np.abs(hover[1]) < ALIGN_TOL or self._phase_steps[i] > ALIGN_TIMEOUT:
        self._phase[i] = 1
        self._phase_steps[i] = 0
        self._integ[i] = 0.0
    elif self._phase[i] == 1:
      # Close the -x gap to just short of contact.
      #
      # THE GATE IS DIRECTIONAL, and that is the whole point of this phase.
      # It used to fire on ``norm(seat) < CONTACT_TOL`` -- an UNSIGNED 3D
      # residual, which is equally satisfied by a hand 5cm SHORT of the seat
      # point and by one 5cm PAST it, on the far side of the toggle. Measured
      # over 32 episodes, that is exactly what the misses were doing: at stroke
      # entry the failures sat at gto.x ~ -0.05 (gripper 5cm beyond the toggle)
      # and gto.z ~ +0.06..+0.10 (hand well below the tip), so the +x ballistic
      # stroke swept through empty air on the far side and the hinge never moved
      # at all (max angle stayed at the -45 deg start). The successes sat at
      # gto.x ~ +0.05, gto.z ~ 0.00. Same stroke, opposite side of the switch.
      #
      # So gate on the SIGNED axes instead: the pad must still need to travel
      # +x to reach the face (never already past it), and must be aligned in y
      # and z. Only then is a blind +x stroke guaranteed to hit something.
      # A SUSTAINED closed-loop press was tried here instead of handing off to
      # the stroke (drive +x with a constant lead so the servo never converges,
      # while holding y/z closed-loop). It measured 0.500 against a 0.531
      # baseline, i.e. no better, and it is recorded here so it is not retried:
      # instrumenting the hinge angle does show many successes flipping the
      # toggle during the approach rather than during the stroke, but converting
      # that observation into a deliberate press did not pay.
      raw = seat
      # Integrate the lateral/vertical axes only. Integrating x as well is what
      # drove the hand through the switch plane in the first place: the x error
      # is deliberately held slightly positive here, so an x integrator winds up
      # without bound until it overshoots.
      lat = np.array([0.0, raw[1], raw[2]])
      self._integ[i] = np.clip(self._integ[i] + INTEG_GAIN * lat, -0.06, 0.06)
      aligned = abs(raw[1]) < ALIGN_TOL and abs(raw[2]) < SEAT_Z_TOL
      behind = raw[0] > 0.0  # the face is still ahead of the pad in +x
      close = raw[0] < SEAT_X_MAX
      if aligned and behind and close:
        self._settle[i] += 1
      else:
        self._settle[i] = 0
      pos_err = raw + self._integ[i]
      # Never command past the seat point in +x: approach it, do not shoot it.
      pos_err[0] = min(pos_err[0], max(raw[0], 0.0))
      # Hand off to the ballistic stroke once seated, or when the window expires.
      if self._settle[i] >= SETTLE:
        self._phase[i] = 2
        self._phase_steps[i] = 0
        self._integ[i] = 0.0
      elif self._phase_steps[i] > SEAT_TIMEOUT:
        if behind and abs(raw[1]) < 0.06:
          self._phase[i] = 2
        else:
          self._phase[i] = 0
        self._phase_steps[i] = 0
        self._integ[i] = 0.0
        self._settle[i] = 0
    else:
      # BALLISTIC COMMIT. Open-loop on purpose: the waypoint is a fixed +x offset
      # from the gripper's own position, NOT from the toggle, so the command never
      # converges and never softens as the toggle retreats. Feedback here would be
      # actively harmful — it is what lets a policy stall at the balance point.
      pos_err = np.array([STROKE_X, 0.0, STROKE_Z])
      if self._phase_steps[i] > STROKE_STEPS:
        # Stroke spent. Back off and re-seat for another attempt if the flip
        # failed; if it succeeded, success is already latched.
        self._phase[i] = 0
        self._phase_steps[i] = 0
        self._settle[i] = 0
        self._integ[i] = 0.0

    return pos_err, _APPROACH_AXIS, gripper_a
