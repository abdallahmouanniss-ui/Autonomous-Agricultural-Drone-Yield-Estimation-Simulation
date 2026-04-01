# src/physics.py
# Point-mass quadrotor physics model — ported from DiffPhysDrone
# (Zhang et al., Nature Machine Intelligence 2025).
#
# What this file does:
#   1. PointMassModel  — discrete-time integrator that predicts drone
#      position and velocity one timestep ahead, given a thrust command.
#      Used by the SITL sensor injector to produce realistic fake sensor
#      readings without Gazebo, and optionally by the brain for lookahead.
#
#   2. CommandSmoother — exponential moving average filter on all velocity
#      commands sent to ArduPilot.  Prevents high-frequency jitter from
#      sensor noise from causing oscillation.  α = CMD_EMA_ALPHA = 1/15.
#
#   3. LatencyBuffer   — circular buffer that delays commands by
#      CONTROL_LATENCY_S to replicate the measured 33 ms actuator lag.
#      Makes SITL behave much closer to real hardware.
#
#   4. obstacle_cost() — soft-plus barrier loss function from the paper.
#      Replaces the binary if/else in lateral_nudge() with a smooth,
#      differentiable cost that scales proportionally to how fast the
#      drone is approaching an obstacle and how close it is.

import math
import time
import collections
from typing import Tuple

import config as cfg


# ── 1. Point-mass dynamics ─────────────────────────────────────────────────
class PointMassModel:
    """
    Discrete-time point-mass quadrotor model from DiffPhysDrone.

    Equations (from the paper, simplified for body-NED frame):
        a = (thrust_vec / mass) - gravity_vec - (drag_coeff / mass) * v
        v_next = v + a * dt
        p_next = p + v * dt + 0.5 * a * dt²

    The model is intentionally simple — the paper demonstrates this is
    sufficient for sim-to-real transfer without full rigid-body dynamics.

    State: position (x, y, z) and velocity (vx, vy, vz) in metres / m/s.
    All coordinates are NED (North-East-Down), matching ArduPilot convention.
    """

    def __init__(self):
        self.pos = [0.0, 0.0, -cfg.TAKEOFF_ALT_M]  # x, y, z (NED; z<0=up)
        self.vel = [0.0, 0.0, 0.0]                  # vx, vy, vz m/s

    def reset(self, x: float = 0.0, y: float = 0.0, alt_m: float = 3.5):
        """Reset to a starting position (altitude as positive AGL metres)."""
        self.pos = [x, y, -alt_m]
        self.vel = [0.0, 0.0, 0.0]

    def step(self, vx_cmd: float, vy_cmd: float, vz_cmd: float,
             dt: float) -> Tuple[list, list]:
        """
        Advance the model by one timestep dt (seconds).

        Args:
            vx_cmd: commanded forward velocity (m/s, body frame)
            vy_cmd: commanded right velocity (m/s, body frame)
            vz_cmd: commanded down velocity (m/s, body frame; positive=down)
            dt:     timestep in seconds

        Returns:
            (position, velocity) after the step
        """
        # Convert body-frame velocity commands to world-frame acceleration.
        # We treat the commanded velocity as a target; the EMA smoother and
        # attitude controller track it with the calibrated gain λ = ATTITUDE_GAIN.
        lam = cfg.ATTITUDE_GAIN

        # Acceleration = gain * (cmd - current_vel) - drag * vel
        drag_coeff = cfg.DRAG_COEFF
        mass       = cfg.MASS_KG

        ax = lam * (vx_cmd - self.vel[0]) - (drag_coeff / mass) * self.vel[0]
        ay = lam * (vy_cmd - self.vel[1]) - (drag_coeff / mass) * self.vel[1]
        # Z: gravity is already handled by ArduPilot's altitude controller;
        # we only model horizontal dynamics for sensor injection.
        az = lam * (vz_cmd - self.vel[2]) - (drag_coeff / mass) * self.vel[2]

        # Clamp: real quadrotor can't exceed ~3g horizontal
        max_a = 3.0 * cfg.GRAVITY_MS2
        ax = max(-max_a, min(max_a, ax))
        ay = max(-max_a, min(max_a, ay))
        az = max(-max_a, min(max_a, az))

        # Euler integration (matches DiffPhysDrone discrete-time eqs)
        self.vel[0] += ax * dt
        self.vel[1] += ay * dt
        self.vel[2] += az * dt

        self.pos[0] += self.vel[0] * dt + 0.5 * ax * dt * dt
        self.pos[1] += self.vel[1] * dt + 0.5 * ay * dt * dt
        self.pos[2] += self.vel[2] * dt + 0.5 * az * dt * dt

        return list(self.pos), list(self.vel)

    @property
    def altitude_m(self) -> float:
        """Current altitude in metres AGL (positive = above ground)."""
        return -self.pos[2]

    @property
    def speed_ms(self) -> float:
        """Current horizontal speed magnitude."""
        return math.hypot(self.vel[0], self.vel[1])


# ── 2. EMA command smoother ────────────────────────────────────────────────
class CommandSmoother:
    """
    Exponential moving average filter on velocity commands.

    From DiffPhysDrone: the inner loop uses τ = 1/15 ≈ 0.067 s time
    constant for the EMA on control inputs, which eliminates high-frequency
    jitter without adding meaningful lag at the 1.2 m/s cruise speed.

    α = CMD_EMA_ALPHA = 1/15 (paper calibrated value)
    output_t = α * input_t + (1 - α) * output_{t-1}
    """

    def __init__(self, alpha: float = None):
        self._alpha = alpha if alpha is not None else cfg.CMD_EMA_ALPHA
        self._vx = 0.0
        self._vy = 0.0
        self._vz = 0.0

    def smooth(self, vx: float, vy: float,
               vz: float = 0.0) -> Tuple[float, float, float]:
        """Apply EMA and return smoothed (vx, vy, vz)."""
        a = self._alpha
        self._vx = a * vx + (1.0 - a) * self._vx
        self._vy = a * vy + (1.0 - a) * self._vy
        self._vz = a * vz + (1.0 - a) * self._vz
        return self._vx, self._vy, self._vz

    def reset(self):
        self._vx = self._vy = self._vz = 0.0


# ── 3. Latency buffer ──────────────────────────────────────────────────────
class LatencyBuffer:
    """
    Circular buffer that delays commands by CONTROL_LATENCY_S.

    DiffPhysDrone measured 33 ms of actuator lag on their real platform.
    Injecting this delay into SITL makes simulated response curves match
    real-flight data closely (see Fig. S1 in their supplementary).

    In production (real hardware) this class is instantiated but bypassed —
    ArduPilot itself introduces the real hardware latency.
    """

    def __init__(self, latency_s: float = None, loop_hz: float = None):
        latency = latency_s if latency_s is not None else cfg.CONTROL_LATENCY_S
        hz      = loop_hz   if loop_hz   is not None else cfg.LOOP_HZ
        depth   = max(1, round(latency * hz))
        # Pre-fill with zeros so the buffer is ready from tick 0
        self._buf: collections.deque = collections.deque(
            [(0.0, 0.0, 0.0)] * depth, maxlen=depth)

    def push_and_get(self, vx: float, vy: float,
                     vz: float = 0.0) -> Tuple[float, float, float]:
        """
        Push the current command and return the command from `latency` ticks ago.
        The caller should USE the returned value, not the input.
        """
        self._buf.append((vx, vy, vz))
        return self._buf[0]   # oldest entry (left side of deque)


# ── 4. Obstacle cost (DiffPhysDrone avoidance loss) ───────────────────────
def obstacle_cost(distance_m: float, approach_speed_ms: float) -> float:
    """
    Smooth obstacle-avoidance cost from DiffPhysDrone.

    Original formulation (Zhang et al. eq. for L_obs):
        r  = closest obstacle distance
        vc = approach speed toward that obstacle (positive = closing)
        L  = vc * (β₁ * max(0, margin - r)² + softplus(β₂ * (margin - r)))

    The paper uses β₂ = 32 for a sharp but differentiable barrier.

    We use this as a scalar "urgency" value:
        — near 0.0 when the drone is far from any wall
        — rises steeply as it approaches within OBSTACLE_MARGIN_M
        — scaled by approach speed so a slow drift costs less than a fast one

    Args:
        distance_m:      measured distance to nearest wall (metres)
        approach_speed_ms: rate at which distance is shrinking (m/s, ≥ 0)

    Returns:
        cost ≥ 0.  Used by lateral_nudge_diffphys() to decide nudge strength.
    """
    margin = cfg.OBSTACLE_MARGIN_M
    b1     = cfg.OBSTACLE_BETA1
    b2     = cfg.OBSTACLE_BETA2

    delta = margin - distance_m   # positive when inside the danger zone

    truncated_quad = b1 * max(0.0, delta) ** 2
    # Numerically stable softplus: ln(1 + exp(x))
    x = b2 * delta
    if x > 20.0:
        softplus_val = x            # avoid overflow; linear approximation
    elif x < -20.0:
        softplus_val = 0.0          # negligible
    else:
        softplus_val = math.log1p(math.exp(x))

    barrier = truncated_quad + softplus_val
    cost    = max(0.0, approach_speed_ms) * barrier
    return cost


def lateral_nudge_diffphys(us_left: float | None,
                            us_right: float | None,
                            vel_left: float,
                            vel_right: float) -> float:
    """
    DiffPhysDrone-inspired lateral correction.

    Replaces the binary if/else potential field with a smooth cost-based
    correction that:
      1. Computes the obstacle cost for each wall using the softplus barrier.
      2. Returns the net lateral velocity command proportional to
         (right_cost - left_cost), so corrections fade smoothly to zero
         as the drone centres itself, preventing oscillation.

    Args:
        us_left:   left ultrasonic reading (metres), or None if stale
        us_right:  right ultrasonic reading (metres), or None if stale
        vel_left:  current lateral velocity component toward left wall (m/s)
        vel_right: current lateral velocity component toward right wall (m/s)

    Returns:
        vy command (m/s): positive = nudge right, negative = nudge left
    """
    INF = 99.0   # treat a missing sensor as "wall very far away" (safe side)

    L = us_left  if us_left  is not None else INF
    R = us_right if us_right is not None else INF

    cost_L = obstacle_cost(L, max(0.0,  vel_left))   # closing on left
    cost_R = obstacle_cost(R, max(0.0, vel_right))   # closing on right

    # Net nudge: push away from the higher-cost side
    # Scale by LATERAL_NUDGE_MS so the maximum command matches config
    net = cost_R - cost_L   # positive → right wall is more threatening → nudge left
    # Note sign: net > 0 means right wall closer, so we nudge left (negative vy)
    nudge = -net * cfg.LATERAL_NUDGE_MS

    # Hard clamp: never command more than 2× the baseline nudge authority
    max_cmd = cfg.LATERAL_NUDGE_MS * 2.0
    return max(-max_cmd, min(max_cmd, nudge))