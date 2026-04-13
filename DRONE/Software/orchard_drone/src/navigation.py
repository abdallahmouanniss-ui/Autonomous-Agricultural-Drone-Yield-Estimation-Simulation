# src/navigation.py
# Thread-safe MAVLink wrapper for ArduPilot GUIDED mode.
#
# HAL changes:
#   - Connection string and baud driven entirely by cfg.SIM_MODE / cfg.FC_CONNECTION
#   - EMA smoother always active; latency buffer active only in SIM_MODE
#   - watchdog_loop dispatches DISTANCE_SENSOR messages to SensorHub.ingest_mavlink
#     so virtual sensors stay fresh in SITL without a second UDP socket
#   - register_sensor_hub() wires the SensorHub callback after construction

import math
import time
import threading
import logging
from pymavlink import mavutil

from physics import CommandSmoother, LatencyBuffer
import config as cfg

log = logging.getLogger("navigation")


class FlightController:

    def __init__(self, connection_str: str = None, baud: int = None):
        conn = connection_str or cfg.FC_CONNECTION
        baud = baud           or cfg.FC_BAUD
        log.info(f"FC connecting → {conn}  (SIM_MODE={cfg.SIM_MODE})")

        self.mav = mavutil.mavlink_connection(
            conn, baud=baud,
            source_system=255, source_component=0)
        self._lock = threading.Lock()

        # Telemetry cache
        self._pos     = None
        self._heading = 0.0
        self._voltage = 0.0
        self._armed   = False
        self._mode    = ""

        # Distance-move state
        self._dmove_start = None
        self._dmove_dist  = 0.0

        # DiffPhysDrone command pipeline
        self._smoother = CommandSmoother()
        self._latbuf   = LatencyBuffer()

        # SensorHub callback — registered after construction via
        # register_sensor_hub(). None until then.
        self._sensor_hub = None

        self._running  = True
        self._watchdog = threading.Thread(
            target=self._watchdog_loop, name="mav-watchdog", daemon=True)
        self._wait_heartbeat()
        self._watchdog.start()

    def register_sensor_hub(self, hub) -> None:
        """
        Wire the SensorHub so the watchdog can forward DISTANCE_SENSOR
        messages to virtual sensors.  Must be called before the mission loop.
        """
        self._sensor_hub = hub
        log.info("FC: SensorHub registered for MAVLink sensor ingestion")

    # ── Watchdog ───────────────────────────────────────────────────────────

    def _wait_heartbeat(self, timeout: float = 30.0):
        log.info("Waiting for heartbeat…")
        self.mav.wait_heartbeat(timeout=timeout)
        log.info(f"Heartbeat OK  sys={self.mav.target_system} "
                 f"comp={self.mav.target_component}")

    def _watchdog_loop(self):
        last_hb = time.monotonic()
        while self._running:
            msg = self.mav.recv_match(blocking=False)
            if msg:
                t = msg.get_type()

                # ── Telemetry cache ────────────────────────────────────────
                if t == "GLOBAL_POSITION_INT":
                    self._pos = (msg.lat / 1e7,
                                 msg.lon / 1e7,
                                 msg.relative_alt / 1000.0)
                    self._heading = msg.hdg / 100.0
                elif t == "SYS_STATUS":
                    self._voltage = msg.voltage_battery / 1000.0
                elif t == "HEARTBEAT":
                    self._armed = bool(
                        msg.base_mode &
                        mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
                    self._mode = mavutil.mode_string_v10(msg)

                # ── Forward to virtual sensors (SITL only) ─────────────────
                # This keeps MAVLinkSensor.age() fresh without a second socket.
                if self._sensor_hub is not None:
                    self._sensor_hub.ingest_mavlink(msg)

            # GCS heartbeat every 1 s
            if time.monotonic() - last_hb >= 1.0:
                with self._lock:
                    self.mav.mav.heartbeat_send(
                        mavutil.mavlink.MAV_TYPE_GCS,
                        mavutil.mavlink.MAV_AUTOPILOT_INVALID,
                        0, 0, 0)
                last_hb = time.monotonic()

            time.sleep(0.005)   # ~200 Hz drain

    # ── Telemetry properties ───────────────────────────────────────────────

    @property
    def position(self):
        return self._pos

    @property
    def heading(self):
        return self._heading

    @property
    def battery_volts(self):
        return self._voltage

    @property
    def is_armed(self):
        return self._armed

    # ── Mode / Arm ─────────────────────────────────────────────────────────

    def set_mode(self, mode_name: str):
        mode_id = self.mav.mode_mapping().get(mode_name)
        if mode_id is None:
            log.error(f"Unknown mode: {mode_name}")
            return
        with self._lock:
            self.mav.mav.set_mode_send(
                self.mav.target_system,
                mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                mode_id)
        log.info(f"Mode → {mode_name}")

    def arm(self, force: bool = False):
        with self._lock:
            self.mav.mav.command_long_send(
                self.mav.target_system, self.mav.target_component,
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                0, 1, 21196 if force else 0, 0, 0, 0, 0, 0)
        log.info("Arm sent")

    def disarm(self):
        with self._lock:
            self.mav.mav.command_long_send(
                self.mav.target_system, self.mav.target_component,
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                0, 0, 0, 0, 0, 0, 0, 0)

    # ── Takeoff ────────────────────────────────────────────────────────────

    def takeoff(self, altitude_m: float):
        with self._lock:
            self.mav.mav.command_long_send(
                self.mav.target_system, self.mav.target_component,
                mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
                0, 0, 0, 0, 0, 0, 0, altitude_m)
        log.info(f"Takeoff → {altitude_m} m")

    def reached_altitude(self, target_m: float, tol_m: float = 0.4) -> bool:
        if self._pos is None:
            return False
        return self._pos[2] >= (target_m - tol_m)

    # ── Velocity — EMA + DiffPhysDrone latency buffer ─────────────────────

    def send_body_velocity(self, vx: float, vy: float, vz: float = 0.0):
        """
        EMA smoothing always active (removes sensor jitter).
        Latency buffer active only in SIM_MODE (replicates 33 ms actuator lag).
        """
        svx, svy, svz = self._smoother.smooth(vx, vy, vz)
        if cfg.SIM_MODE:
            svx, svy, svz = self._latbuf.push_and_get(svx, svy, svz)
        with self._lock:
            self.mav.mav.set_position_target_local_ned_send(
                0,
                self.mav.target_system, self.mav.target_component,
                mavutil.mavlink.MAV_FRAME_BODY_NED,
                0b0000_1111_1100_0111,
                0, 0, 0,
                svx, svy, svz,
                0, 0, 0,
                0, 0)

    def stop(self):
        self._smoother.reset()
        self.send_body_velocity(0.0, 0.0, 0.0)

    def loiter(self):
        self.set_mode("LOITER")
        log.warning("LOITER engaged (failsafe)")

    # ── Yaw ────────────────────────────────────────────────────────────────

    def condition_yaw(self, angle_deg: float, relative: bool = True,
                      rate_dps: float = 40.0):
        direction = 1 if angle_deg >= 0 else -1
        with self._lock:
            self.mav.mav.command_long_send(
                self.mav.target_system, self.mav.target_component,
                mavutil.mavlink.MAV_CMD_CONDITION_YAW,
                0, abs(angle_deg), rate_dps, direction,
                1 if relative else 0, 0, 0, 0)
        log.info(f"Yaw {angle_deg}° ({'rel' if relative else 'abs'})")

    def yaw_complete_in(self, angle_deg: float, rate_dps: float = 40.0) -> float:
        return abs(angle_deg) / rate_dps + 0.5

    # ── Distance move ──────────────────────────────────────────────────────

    def start_distance_move(self, vx: float, vy: float, dist_m: float):
        self._dmove_start = self._pos
        self._dmove_dist  = dist_m

    def distance_move_complete(self) -> bool:
        if self._dmove_start is None or self._pos is None:
            return False
        return self._haversine(self._dmove_start, self._pos) >= self._dmove_dist

    # ── Camera shutter ─────────────────────────────────────────────────────

    def trigger_camera(self):
        with self._lock:
            self.mav.mav.command_long_send(
                self.mav.target_system, self.mav.target_component,
                mavutil.mavlink.MAV_CMD_DO_DIGICAM_CONTROL,
                0, 0, 0, 0, 0, 1, 0, 0)
        log.info("Shutter")

    # ── RTL ────────────────────────────────────────────────────────────────

    def rtl(self):
        self.set_mode("RTL")
        log.info("RTL")

    # ── Teardown ───────────────────────────────────────────────────────────

    def close(self):
        self._running = False
        self.mav.close()
        log.info("MAVLink closed")

    # ── Utility ────────────────────────────────────────────────────────────

    @staticmethod
    def _haversine(p1, p2) -> float:
        R = 6_371_000.0
        lat1, lon1 = math.radians(p1[0]), math.radians(p1[1])
        lat2, lon2 = math.radians(p2[0]), math.radians(p2[1])
        dlat, dlon = lat2-lat1, lon2-lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
        return R * 2 * math.asin(math.sqrt(a))
