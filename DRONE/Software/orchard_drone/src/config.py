# src/config.py
# Single source of truth for every tunable parameter.
# DiffPhysDrone contributions: MASS_KG, DRAG_COEFF, CMD_EMA_ALPHA,
#   ATTITUDE_GAIN, CONTROL_LATENCY_S, OBSTACLE_BETA1, OBSTACLE_BETA2
# (Zhang et al., Nature Machine Intelligence 2025)

import os

# ── Connection ─────────────────────────────────────────────────────────────
# Switch FC_CONNECTION between SITL and production without touching any
# other file.  Set env var SITL=1 to auto-switch.
_SITL = os.getenv("SITL", "0") == "1"
FC_CONNECTION  = "udpin:0.0.0.0:14550" if _SITL else "/dev/ttyAMA0"
FC_BAUD        = 921600

# ── Sensor GPIO (BCM numbering) ────────────────────────────────────────────
US_LEFT_TRIG   = 23
US_LEFT_ECHO   = 24
US_RIGHT_TRIG  = 27
US_RIGHT_ECHO  = 22
LIDAR_SERIAL   = "/dev/ttyUSB0"
LIDAR_BAUD     = 115200
CAMERA_INDEX   = 0

# ── Flight parameters ──────────────────────────────────────────────────────
CRUISE_SPEED_MS  = 1.2    # m/s forward cruise speed
LATERAL_NUDGE_MS = 0.3    # m/s baseline lateral correction authority
TAKEOFF_ALT_M    = 3.5    # metres AGL target altitude

# ── DiffPhysDrone physics constants ───────────────────────────────────────
# Sourced from Zhang et al. calibration on a real 365 g quadrotor.
# These govern the point-mass simulator in physics.py and the command
# smoother in navigation.py.
MASS_KG            = 0.365   # vehicle mass (grams→kg); use your actual mass
DRAG_COEFF         = 0.08    # linear air drag N/(m/s); tuned by the paper
GRAVITY_MS2        = 9.81
ATTITUDE_GAIN      = 13.0    # proportional gain of inner attitude loop
CMD_EMA_ALPHA      = 1.0/15  # τ = 1/15 exponential moving average constant
CONTROL_LATENCY_S  = 0.033   # 33 ms measured actuator latency

# Obstacle-avoidance soft-plus barrier (DiffPhysDrone loss function params)
OBSTACLE_BETA1     = 1.0     # truncated-quadratic weight
OBSTACLE_BETA2     = 32.0    # soft-plus sharpness (paper: β₂ = 32)
OBSTACLE_MARGIN_M  = 0.30    # clearance below which cost activates (metres)

# ── Corridor sensor fusion ─────────────────────────────────────────────────
US_WALL_CLOSE_M    = 1.0     # nudge if side sensor reads below this
US_ROW_END_M       = 4.0     # side sensor above this = open space
LIDAR_ROW_END_M    = 15.0    # front LiDAR above this = row end
SENSOR_TIMEOUT_S   = 2.0     # stale data window before fail-safe
LOOP_HZ            = 20
LOOP_PERIOD        = 1.0 / LOOP_HZ

# ── Vision ─────────────────────────────────────────────────────────────────
TRUNK_CENTER_TOL_PX   = 45     # px offset tolerance for shutter trigger
TREE_CLOSE_BBOX_RATIO = 0.30   # bbox_h / frame_h threshold = "adjacent"
SHUTTER_COOLDOWN_S    = 1.5
FRAME_W, FRAME_H      = 640, 480
YOLO_MODEL_PATH       = "models/orchard_yolo.pt"
YOLO_CONF_THRESHOLD   = 0.45
GREEN_DENSITY_THRESHOLD = 0.15

# ── Maneuver ───────────────────────────────────────────────────────────────
ROW_CLEAR_DIST_M  = 4.0    # fly forward this far after row-end detected
LATERAL_SHIFT_M   = 5.0    # lateral shift to next row
YAW_TURN_DEG      = 180    # U-turn anglea
