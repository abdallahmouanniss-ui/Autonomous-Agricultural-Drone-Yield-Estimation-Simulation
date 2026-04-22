# src/sensors.py
# Hardware Abstraction Layer for all distance sensors.
#
# SIM_MODE = False → GPIO (HC-SR04) and serial (TFmini) polling threads
# SIM_MODE = True  → MAVLinkSensor objects fed by DISTANCE_SENSOR messages
#                    forwarded from FlightController._watchdog_loop via
#                    SensorHub.ingest_mavlink(msg).
#
# ── Physics-based Gazebo data flow (SIM_MODE=True) ──────────────────────
#
#  Gazebo Ray sensor
#    → libgazebo_ros_ray_sensor.so
#    → ROS sensor_msgs/Range topic
#    → libArduPilotPlugin.so <rangefinder> bridge
#    → ArduPilot SITL (internal)
#    → MAVLink DISTANCE_SENSOR message (UDP → port 14552)
#    → FlightController._watchdog_loop recv_match()
#    → SensorHub.ingest_mavlink(msg)
#    → MAVLinkSensor.ingest(msg)   ← matches on msg.orientation
#    → MAVLinkSensor._update(distance_m)
#    → MAVLinkSensor.read() / age()  ← consumed by FusionData each tick
#
# ── Orientation mapping (ArduPilot MAV_SENSOR_ROTATION_* integers) ──────
#
#   Gazebo sensor   ROS topic            RNGFND_ORIENT  Python constant
#   ─────────────── ──────────────────── ─────────────  ──────────────
#   lidar_front     /drone/lidar_front   0 (NONE/Fwd)   ORIENT_FORWARD = 0
#   sonar_left      /drone/sonar_left    6 (YAW_270/L)  ORIENT_LEFT    = 6
#   sonar_right     /drone/sonar_right   2 (YAW_90/R)   ORIENT_RIGHT   = 2
#
# No UDP listener on any port exists in this file.  The sole data path is
# the FC watchdog calling ingest_mavlink().
# ────────────────────────────────────────────────────────────────────────

import threading
import time
import logging
import math
from typing import Optional, Callable

import config as cfg

log = logging.getLogger("sensors")

# ── Conditional hardware imports ───────────────────────────────────────────
try:
    import RPi.GPIO as GPIO
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    _GPIO_OK = True
except ImportError:
    log.warning("RPi.GPIO unavailable — ultrasonic sensors will use MAVLink or return None")
    _GPIO_OK = False

try:
    import serial as pyserial
    _SERIAL_OK = True
except ImportError:
    log.warning("pyserial unavailable — TFmini will use MAVLink or return None")
    _SERIAL_OK = False


# ══════════════════════════════════════════════════════════════════════════
# Base sensor
# ══════════════════════════════════════════════════════════════════════════

class _SensorBase:
    """
    Thread-safe base. Every sensor exposes:
      .read()    → latest float value or None
      .age()     → seconds since last successful reading (math.inf if never)
      .start() / .stop() / .cleanup()
    """

    def __init__(self, name: str, poll_hz: float = 20.0):
        self.name = name
        self._value: Optional[float] = None
        self._last_update = 0.0          # 0.0 sentinel → age() returns inf
        self._lock = threading.Lock()
        self._running = False
        self._poll_period = 1.0 / poll_hz
        self._thread = threading.Thread(
            target=self._loop, name=f"sensor-{name}", daemon=True)

    def start(self):
        self._running = True
        self._thread.start()
        log.info(f"[{self.name}] thread started")

    def stop(self):
        self._running = False

    def read(self) -> Optional[float]:
        with self._lock:
            return self._value

    def age(self) -> float:
        with self._lock:
            if self._last_update == 0.0:
                return math.inf
            return time.monotonic() - self._last_update

    def _update(self, value: float):
        with self._lock:
            self._value = value
            self._last_update = time.monotonic()

    def _loop(self):
        raise NotImplementedError

    def cleanup(self):
        pass


# ══════════════════════════════════════════════════════════════════════════
# SIMULATION sensors — fed by MAVLink DISTANCE_SENSOR messages
# ══════════════════════════════════════════════════════════════════════════

class MAVLinkSensor(_SensorBase):
    """
    Virtual sensor for SITL / physics-based Gazebo simulation.

    Does NOT run its own polling loop.  FlightController._watchdog_loop
    calls SensorHub.ingest_mavlink(msg) for every incoming MAVLink message;
    that method calls .ingest(msg) on each MAVLinkSensor instance.

    Matching key: msg.orientation must equal self.expected_orientation.

    ArduPilot MAV_SENSOR_ROTATION_* orientation integers
    (these are the values ArduPilot writes into the DISTANCE_SENSOR.orientation
    field based on the RNGFND#_ORIENT parameter you set):

        ORIENT_FORWARD = 0   MAV_SENSOR_ROTATION_NONE      → lidar_front
        ORIENT_RIGHT   = 2   MAV_SENSOR_ROTATION_YAW_90    → sonar_right
        ORIENT_LEFT    = 6   MAV_SENSOR_ROTATION_YAW_270   → sonar_left

    These integers are the authoritative source of truth.  They must match:
      1. The <orientation> values inside <rangefinder> blocks in orchard.sdf
      2. The RNGFND#_ORIENT MAVProxy params set before flight
      3. The expected_orientation passed to each MAVLinkSensor constructor
    """

    # MAVLink MAV_SENSOR_ROTATION_* integer values
    ORIENT_FORWARD = 0   # forward  — LiDAR front
    ORIENT_RIGHT   = 2   # right    — HC-SR04 right
    ORIENT_LEFT    = 6   # left     — HC-SR04 left

    def __init__(self, name: str, expected_orientation: int):
        super().__init__(name, poll_hz=1.0)   # poll_hz irrelevant — no active loop
        self.expected_orientation = expected_orientation
        # Replace the thread target with a harmless idle so start() doesn't block
        self._thread = threading.Thread(
            target=self._loop, name=f"sensor-{name}", daemon=True)

    def _loop(self):
        """No-op idle loop — this sensor is push-fed via ingest()."""
        while self._running:
            time.sleep(1.0)

    def ingest(self, msg) -> bool:
        """
        Called by SensorHub.ingest_mavlink() for every incoming MAVLink message.

        Returns True  → this sensor consumed the message (caller should break).
        Returns False → message not for this sensor (wrong type or orientation).

        Guards:
        - msg type must be DISTANCE_SENSOR
        - msg.orientation must match expected_orientation
        - distance must be > 0 cm (ArduPilot emits 0 on sensor init/timeout)
        - distance must be within the sensor's stated max_distance
        """
        if msg.get_type() != "DISTANCE_SENSOR":
            return False
        if msg.orientation != self.expected_orientation:
            return False

        # ArduPilot DISTANCE_SENSOR.current_distance is in centimetres
        distance_cm = msg.current_distance
        if distance_cm <= 0:
            # Zero reading = sensor not yet ready or out-of-range low; skip.
            return True   # still "consumed" — no other sensor should process it

        distance_m = distance_cm / 100.0

        # Soft-clamp to the sensor's own declared maximum so stray reflections
        # don't feed 65535 cm (the ArduPilot "no return" sentinel) upstream.
        max_m = msg.max_distance / 100.0
        if distance_m > max_m:
            return True   # out-of-range high — consumed but not stored

        self._update(distance_m)
        return True


# ══════════════════════════════════════════════════════════════════════════
# REAL HARDWARE sensors
# ══════════════════════════════════════════════════════════════════════════

class UltrasonicSensor(_SensorBase):
    """HC-SR04 GPIO sensor. Returns metres in [0.02, 4.5] or None."""

    MAX_M       = 4.5
    MIN_M       = 0.02
    SPEED_SOUND = 343.0
    TRIG_PULSE_S = 1e-5
    TIMEOUT_S   = 0.025

    def __init__(self, name: str, trig: int, echo: int):
        super().__init__(name, poll_hz=15.0)
        self.trig = trig
        self.echo = echo
        if _GPIO_OK:
            GPIO.setup(self.trig, GPIO.OUT, initial=GPIO.LOW)
            GPIO.setup(self.echo, GPIO.IN)

    def _measure(self) -> Optional[float]:
        if not _GPIO_OK:
            return None
        GPIO.output(self.trig, GPIO.HIGH)
        time.sleep(self.TRIG_PULSE_S)
        GPIO.output(self.trig, GPIO.LOW)

        deadline = time.monotonic() + self.TIMEOUT_S
        while GPIO.input(self.echo) == GPIO.LOW:
            if time.monotonic() > deadline:
                return None
        t0 = time.monotonic()
        while GPIO.input(self.echo) == GPIO.HIGH:
            if time.monotonic() > deadline:
                return None
        dist = (time.monotonic() - t0) * self.SPEED_SOUND / 2.0
        return round(dist, 4) if self.MIN_M <= dist <= self.MAX_M else None

    def _loop(self):
        while self._running:
            t0 = time.monotonic()
            v = self._measure()
            if v is not None:
                self._update(v)
            time.sleep(max(0.0, self._poll_period - (time.monotonic() - t0)))

    def cleanup(self):
        if _GPIO_OK:
            GPIO.cleanup([self.trig, self.echo])


class TFminiLiDAR(_SensorBase):
    """Benewake TFmini UART LiDAR. Returns metres or None."""

    HDR     = 0x59
    FLEN    = 9
    MIN_STR = 100

    def __init__(self, port: str = "/dev/ttyUSB0", baud: int = 115200):
        super().__init__("TFmini", poll_hz=100.0)
        self.port   = port
        self.baud   = baud
        self._serial = None

    def _open(self) -> bool:
        if not _SERIAL_OK:
            return False
        try:
            self._serial = pyserial.Serial(self.port, self.baud, timeout=0.1)
            log.info(f"[TFmini] {self.port} @ {self.baud}")
            return True
        except Exception as e:
            log.error(f"[TFmini] open failed: {e}")
            return False

    def _parse(self, buf: bytes) -> Optional[float]:
        if len(buf) < self.FLEN:
            return None
        if buf[0] != self.HDR or buf[1] != self.HDR:
            return None
        if (sum(buf[:8]) & 0xFF) != buf[8]:
            return None
        dist_cm  = buf[2] | (buf[3] << 8)
        strength = buf[4] | (buf[5] << 8)
        if strength < self.MIN_STR or dist_cm == 0:
            return None
        return dist_cm / 100.0

    def _loop(self):
        if not self._open():
            log.warning("[TFmini] no serial — stub mode (returns None)")
            while self._running:
                time.sleep(0.5)
            return

        buf = b""
        while self._running:
            try:
                buf += self._serial.read(self._serial.in_waiting or 1)
                while len(buf) >= self.FLEN:
                    idx = buf.find(bytes([self.HDR, self.HDR]))
                    if idx == -1:
                        buf = b""
                        break
                    if idx > 0:
                        buf = buf[idx:]
                    d = self._parse(buf[:self.FLEN])
                    if d is not None:
                        self._update(d)
                    buf = buf[self.FLEN:]
            except Exception as e:
                log.error(f"[TFmini] read error: {e}")
                time.sleep(0.5)

    def cleanup(self):
        if self._serial and self._serial.is_open:
            self._serial.close()


# ══════════════════════════════════════════════════════════════════════════
# Sensor Hub
# ══════════════════════════════════════════════════════════════════════════

class SensorHub:
    """
    Owns all sensor instances and exposes a unified interface to the brain.

    SIM_MODE = True:
        lidar, us_left, us_right → MAVLinkSensor objects.
        Data arrives via SensorHub.ingest_mavlink(msg), called by
        FlightController._watchdog_loop for every incoming MAVLink message.
        No GPIO, no serial, no UDP socket is opened.

    SIM_MODE = False:
        lidar    → TFminiLiDAR  (serial UART)
        us_left  → UltrasonicSensor (GPIO BCM pins from config)
        us_right → UltrasonicSensor (GPIO BCM pins from config)
        Each runs its own polling daemon thread.
    """

    def __init__(self):
        if cfg.SIM_MODE:
            log.info("SensorHub: SIMULATION mode — MAVLink virtual sensors "
                     "(Gazebo → ArduPilot → DISTANCE_SENSOR)")
            self.lidar    = MAVLinkSensor("TFmini_sim",  MAVLinkSensor.ORIENT_FORWARD)
            self.us_left  = MAVLinkSensor("US_LEFT_sim", MAVLinkSensor.ORIENT_LEFT)
            self.us_right = MAVLinkSensor("US_RIGHT_sim",MAVLinkSensor.ORIENT_RIGHT)
        else:
            log.info("SensorHub: HARDWARE mode — GPIO and serial sensors")
            self.lidar    = TFminiLiDAR(cfg.LIDAR_SERIAL, cfg.LIDAR_BAUD)
            self.us_left  = UltrasonicSensor("US_LEFT",
                                             cfg.US_LEFT_TRIG, cfg.US_LEFT_ECHO)
            self.us_right = UltrasonicSensor("US_RIGHT",
                                             cfg.US_RIGHT_TRIG, cfg.US_RIGHT_ECHO)

        self._all = [self.lidar, self.us_left, self.us_right]

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def start_all(self):
        for s in self._all:
            s.start()

    def stop_all(self):
        for s in self._all:
            s.stop()
            s.cleanup()

    # ── Staleness check ────────────────────────────────────────────────────

    def any_stale(self, timeout_s: float) -> list:
        """Returns names of sensors whose last update is older than timeout_s."""
        return [s.name for s in self._all if s.age() > timeout_s]

    # ── MAVLink ingestion (SIM_MODE data path) ─────────────────────────────

    def ingest_mavlink(self, msg) -> None:
        """
        Called by FlightController._watchdog_loop for EVERY incoming MAVLink
        message (not just DISTANCE_SENSOR — the sensor filters internally).

        In SIM_MODE: iterates MAVLinkSensor objects; the first whose
        expected_orientation matches msg.orientation consumes the message.

        In hardware mode: no-op (hardware sensors run their own threads).

        This is the ONLY mechanism by which sensor data enters the system
        in SIM_MODE.  There is no UDP listener on any port in this file.
        """
        if not cfg.SIM_MODE:
            return
        for sensor in self._all:
            if isinstance(sensor, MAVLinkSensor):
                if sensor.ingest(msg):
                    break   # each DISTANCE_SENSOR message belongs to exactly one sensor
