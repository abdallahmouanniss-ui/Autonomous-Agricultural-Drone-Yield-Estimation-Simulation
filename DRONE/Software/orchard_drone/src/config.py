# src/config.py
# Single source of truth. Set SIM_MODE = True when running in Gazebo SITL.
# Set SIM_MODE = False for physical hardware deployment.
# Can also be overridden at launch: SIM_MODE=1 python src/main.py

import os

# ── Master simulation toggle ───────────────────────────────────────────────
# True  → UDP connection, MAVLink virtual sensors, mock/GStreamer camera
# False → Serial connection, GPIO sensors, physical camera
SIM_MODE: bool = os.getenv("SIM_MODE", "1") == "1"

# ── Flight controller connection ───────────────────────────────────────────
FC_CONNECTION: str = "udpin:0.0.0.0:14552" if SIM_MODE else "/dev/ttyAMA0"
FC_BAUD:       int = 921600   # ignored for UDP but kept for parity

# ── Sensor hardware (GPIO BCM — only used when SIM_MODE = False) ───────────
US_LEFT_TRIG  = 23
US_LEFT_ECHO  = 24
US_RIGHT_TRIG = 27
US_RIGHT_ECHO = 22
LIDAR_SERIAL  = "/dev/ttyUSB0"
LIDAR_BAUD    = 115200

# ── Camera ─────────────────────────────────────────────────────────────────
# SIM_MODE=False: integer index passed to cv2.VideoCapture
# SIM_MODE=True:  CAMERA_SIM_SOURCE is tried first.
#   Set to a GStreamer pipeline string, e.g.:
#     "udpsrc port=5600 ! ... ! appsink"
#   or a video file path for offline testing:
#     "/path/to/test_orchard.mp4"
#   Leave as empty string "" to use the built-in MockVision (safe fallback).
CAMERA_INDEX      = 0
CAMERA_SIM_SOURCE = os.getenv("CAMERA_SIM_SOURCE", "")

# ── Flight parameters ──────────────────────────────────────────────────────
CRUISE_SPEED_MS  = 1.2
LATERAL_NUDGE_MS = 0.3
TAKEOFF_ALT_M    = 3.5

# ── DiffPhysDrone physics constants ───────────────────────────────────────
MASS_KG           = 0.365
DRAG_COEFF        = 0.08
GRAVITY_MS2       = 9.81
ATTITUDE_GAIN     = 13.0
CMD_EMA_ALPHA     = 1.0 / 15
CONTROL_LATENCY_S = 0.033
OBSTACLE_BETA1    = 1.0
OBSTACLE_BETA2    = 32.0
OBSTACLE_MARGIN_M = 0.30

# ── Corridor sensor fusion ─────────────────────────────────────────────────
US_WALL_CLOSE_M = 1.0
US_ROW_END_M    = 4.0
LIDAR_ROW_END_M = 15.0

# Stale-sensor timeout.
# In SITL the MAVLink sensor thread refreshes every message cycle (~5 Hz).
# Give it 10 s in sim, 2 s on real hardware (serial/GPIO runs at 15-100 Hz).
SENSOR_TIMEOUT_S: float = 10.0 if SIM_MODE else 2.0

LOOP_HZ     = 20
LOOP_PERIOD = 1.0 / LOOP_HZ

# ── Vision thresholds ──────────────────────────────────────────────────────
TRUNK_CENTER_TOL_PX   = 45
TREE_CLOSE_BBOX_RATIO = 0.30
SHUTTER_COOLDOWN_S    = 1.5
FRAME_W, FRAME_H      = 640, 480
YOLO_MODEL_PATH       = "models/orchard_yolo.pt"
YOLO_CONF_THRESHOLD   = 0.45
GREEN_DENSITY_THRESHOLD = 0.15

# ── Manoeuvre parameters ───────────────────────────────────────────────────
ROW_CLEAR_DIST_M = 4.0
LATERAL_SHIFT_M  = 5.0
YAW_TURN_DEG     = 180
