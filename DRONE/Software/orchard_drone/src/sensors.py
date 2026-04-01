# src/sensors.py
# Non-blocking hardware drivers.
# Each sensor runs in its own daemon thread.
# SensorHub owns only hardware sensors (LiDAR + 2x ultrasonic).
# CameraVision is NOT here — RowVision in vision.py owns the camera.

import threading
import time
import logging
import math

log = logging.getLogger("sensors")

try:
    import RPi.GPIO as GPIO  # type: ignore
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    _GPIO_OK = True
except ImportError:
    log.warning("RPi.GPIO unavailable — ultrasonics return None (SITL mode)")
    _GPIO_OK = False

try:
    import serial as pyserial
    _SERIAL_OK = True
except ImportError:
    log.warning("pyserial unavailable — TFmini returns None (SITL mode)")
    _SERIAL_OK = False


class _SensorBase:
    """Thread-safe base for all sensor drivers."""

    def __init__(self, name: str, poll_hz: float = 20.0):
        self.name          = name
        self._value        = None
        self._last_update  = 0.0   # initialised here so age() is always safe
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

    def read(self):
        with self._lock:
            return self._value

    def age(self) -> float:
        with self._lock:
            if self._last_update == 0.0:
                return math.inf
            return time.monotonic() - self._last_update

    def _update(self, value):
        with self._lock:
            self._value       = value
            self._last_update = time.monotonic()

    def _loop(self):
        raise NotImplementedError

    def cleanup(self):
        pass


class UltrasonicSensor(_SensorBase):
    """
    HC-SR04 driver.
    Returns distance in metres [0.02, 4.5], or None on timeout/out-of-range.
    Assumes 3.3 V logic / voltage divider on ECHO pin.
    """

    MAX_M        = 4.5
    MIN_M        = 0.02
    SPEED_SOUND  = 343.0     # m/s at 20°C
    TRIG_PULSE_S = 1e-5      # 10 µs
    TIMEOUT_S    = 0.025     # 25 ms ≈ max range

    def __init__(self, name: str, trig: int, echo: int):
        super().__init__(name, poll_hz=15.0)
        self.trig = trig
        self.echo = echo
        if _GPIO_OK:
            GPIO.setup(self.trig, GPIO.OUT, initial=GPIO.LOW)
            GPIO.setup(self.echo, GPIO.IN)

    def _measure(self) -> float | None:
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
            t0  = time.monotonic()
            val = self._measure()
            if val is not None:
                self._update(val)
            time.sleep(max(0.0, self._poll_period - (time.monotonic() - t0)))

    def cleanup(self):
        if _GPIO_OK:
            GPIO.cleanup([self.trig, self.echo])


class TFminiLiDAR(_SensorBase):
    """
    Benewake TFmini UART driver.
    Parses 9-byte frames: [0x59, 0x59, dL, dH, strL, strH, _, _, cksum]
    Returns distance in metres, or None on bad frame / low signal.
    """

    HDR      = 0x59
    FLEN     = 9
    MIN_STR  = 100   # minimum signal strength for a valid reading

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

    def _parse(self, buf: bytes) -> float | None:
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
            log.warning("[TFmini] SITL stub — returning None")
            while self._running:
                time.sleep(0.1)
            return
        buf = b""
        while self._running:
            try:
                chunk = self._serial.read(self._serial.in_waiting or 1)
                buf  += chunk
                while len(buf) >= self.FLEN:
                    idx = buf.find(bytes([self.HDR, self.HDR]))
                    if idx == -1:
                        buf = b""
                        break
                    if idx > 0:
                        buf = buf[idx:]
                    dist = self._parse(buf[:self.FLEN])
                    if dist is not None:
                        self._update(dist)
                    buf = buf[self.FLEN:]
            except Exception as e:
                log.error(f"[TFmini] read error: {e}")
                time.sleep(0.5)

    def cleanup(self):
        if self._serial and self._serial.is_open:
            self._serial.close()


class SensorHub:
    """
    Owns all hardware sensor threads.
    Camera is intentionally excluded — RowVision in vision.py owns it,
    preventing false stale-sensor faults from an unused CameraVision thread.
    """

    def __init__(self, cfg_module):
        self.lidar    = TFminiLiDAR(cfg_module.LIDAR_SERIAL,
                                     cfg_module.LIDAR_BAUD)
        self.us_left  = UltrasonicSensor("US_LEFT",
                                          cfg_module.US_LEFT_TRIG,
                                          cfg_module.US_LEFT_ECHO)
        self.us_right = UltrasonicSensor("US_RIGHT",
                                          cfg_module.US_RIGHT_TRIG,
                                          cfg_module.US_RIGHT_ECHO)
        self._all = [self.lidar, self.us_left, self.us_right]

    def start_all(self):
        for s in self._all:
            s.start()

    def stop_all(self):
        for s in self._all:
            s.stop()
            s.cleanup()

    def any_stale(self, timeout_s: float) -> list[str]:
        return [s.name for s in self._all if s.age() > timeout_s]