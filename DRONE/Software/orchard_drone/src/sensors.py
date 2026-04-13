# src/sensors.py
# Hardware Abstraction Layer for all distance sensors.
#
# SIM_MODE = False  → GPIO (HC-SR04) and serial (TFmini) threads
# SIM_MODE = True   → MAVLinkSensor threads that read DISTANCE_SENSOR
#                     messages forwarded by FlightController.watchdog_loop
#
# The key design: in SITL every sensor is a MAVLinkSensor whose _update()
# is called by the FC watchdog whenever a DISTANCE_SENSOR message arrives.
# This keeps age() fresh so the stale-sensor failsafe never fires falsely.
#
# Gazebo sends DISTANCE_SENSOR messages for the simulated LiDAR/sonar
# via the ArduPilot SITL bridge. Each sensor is identified by sensor_id
# (orientation field in the MAVLink message):
#   id 0  = forward  → TFmini (front LiDAR)
#   id 6  = left     → HC-SR04 left
#   id 2  = right    → HC-SR04 right
# These match ArduPilot's RNGFND orientation conventions.

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
      .read()  → latest float value or None
      .age()   → seconds since last successful reading
      .start() / .stop() / .cleanup()
    """

    def __init__(self, name: str, poll_hz: float = 20.0):
        self.name          = name
        self._value: Optional[float] = None
        self._last_update  = 0.0       # initialised here — age() always safe
        self._lock         = threading.Lock()
        self._running      = False
        self._poll_period  = 1.0 / poll_hz
        self._thread       = threading.Thread(
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
            self._value       = value
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
    Virtual sensor for SITL.

    Does NOT run its own polling loop. Instead, FlightController.watchdog_loop
    calls .ingest(msg) whenever a DISTANCE_SENSOR MAVLink message arrives
    whose orientation matches this sensor's expected_orientation.

    Gazebo/ArduPilot DISTANCE_SENSOR orientations:
      MAV_SENSOR_ROTATION_NONE (0)          → forward (TFmini front LiDAR)
      MAV_SENSOR_ROTATION_YAW_270 (6)       → left  (HC-SR04 left)
      MAV_SENSOR_ROTATION_YAW_90  (2)       → right (HC-SR04 right)
    """

    # MAVLink MAV_SENSOR_ROTATION_* values
    ORIENT_FORWARD = 0
    ORIENT_RIGHT   = 2
    ORIENT_LEFT    = 6

    def __init__(self, name: str, expected_orientation: int):
        # poll_hz irrelevant — no background loop
        super().__init__(name, poll_hz=1.0)
        self.expected_orientation = expected_orientation
        # Override the thread to a no-op so start() doesn't block
        self._thread = threading.Thread(
            target=self._loop, name=f"sensor-{name}", daemon=True)

    def _loop(self):
        """No-op — this sensor is push-fed via ingest()."""
        while self._running:
            time.sleep(1.0)

    def ingest(self, msg) -> bool:
        """
        Called by FC watchdog for every DISTANCE_SENSOR message.
        Returns True if this message was consumed by this sensor.
        """
        if msg.get_type() != "DISTANCE_SENSOR":
            return False
        if msg.orientation != self.expected_orientation:
            return False
        # ArduPilot reports distance in centimetres
        distance_m = msg.current_distance / 100.0
        if distance_m > 0:
            self._update(distance_m)
        return True


# ══════════════════════════════════════════════════════════════════════════
# REAL HARDWARE sensors
# ══════════════════════════════════════════════════════════════════════════
class UltrasonicSensor(_SensorBase):
    """HC-SR04. Returns metres [0.02, 4.5] or None."""

    MAX_M        = 4.5
    MIN_M        = 0.02
    SPEED_SOUND  = 343.0
    TRIG_PULSE_S = 1e-5
    TIMEOUT_S    = 0.025

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
            v  = self._measure()
            if v is not None:
                self._update(v)
            time.sleep(max(0.0, self._poll_period - (time.monotonic() - t0)))

    def cleanup(self):
        if _GPIO_OK:
            GPIO.cleanup([self.trig, self.echo])


class TFminiLiDAR(_SensorBase):
    """Benewake TFmini UART. Returns metres or None."""

    HDR     = 0x59
    FLEN    = 9
    MIN_STR = 100

    def __init__(self, port: str = "/dev/ttyUSB0", baud: int = 115200):
        super().__init__("TFmini", poll_hz=100.0)
        self.port    = port
        self.baud    = baud
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
    Owns all sensor instances and exposes a unified interface.

    SIM_MODE = True:
        lidar, us_left, us_right are MAVLinkSensor objects.
        They are fed by FlightController.watchdog_loop via
        SensorHub.ingest_mavlink(msg).  No GPIO or serial is touched.

    SIM_MODE = False:
        lidar, us_left, us_right are TFminiLiDAR / UltrasonicSensor objects
        running their own polling threads.
    """

    def __init__(self):
        if cfg.SIM_MODE:
            log.info("SensorHub: SIMULATION mode — using MAVLink virtual sensors")
            self.lidar    = MAVLinkSensor("TFmini_sim",
                                          MAVLinkSensor.ORIENT_FORWARD)
            self.us_left  = MAVLinkSensor("US_LEFT_sim",
                                          MAVLinkSensor.ORIENT_LEFT)
            self.us_right = MAVLinkSensor("US_RIGHT_sim",
                                          MAVLinkSensor.ORIENT_RIGHT)
        else:
            log.info("SensorHub: HARDWARE mode — using GPIO and serial sensors")
            self.lidar    = TFminiLiDAR(cfg.LIDAR_SERIAL, cfg.LIDAR_BAUD)
            self.us_left  = UltrasonicSensor("US_LEFT",
                                             cfg.US_LEFT_TRIG, cfg.US_LEFT_ECHO)
            self.us_right = UltrasonicSensor("US_RIGHT",
                                             cfg.US_RIGHT_TRIG, cfg.US_RIGHT_ECHO)

        self._all = [self.lidar, self.us_left, self.us_right]

    def start_all(self):
        for s in self._all:
            s.start()

    def stop_all(self):
        for s in self._all:
            s.stop()
            s.cleanup()

    def any_stale(self, timeout_s: float) -> list:
        """Returns names of sensors whose last update is older than timeout_s."""
        return [s.name for s in self._all if s.age() > timeout_s]

    def ingest_mavlink(self, msg) -> None:
        """
        Called by FlightController.watchdog_loop for every incoming MAVLink
        message.  In SIM_MODE, MAVLinkSensor objects consume DISTANCE_SENSOR
        messages that match their orientation.  In hardware mode this is a
        no-op — hardware sensors run their own threads.
        """
        if not cfg.SIM_MODE:
            return
        for sensor in self._all:
            if isinstance(sensor, MAVLinkSensor):
                if sensor.ingest(msg):
                    break   # each message belongs to exactly one sensor
