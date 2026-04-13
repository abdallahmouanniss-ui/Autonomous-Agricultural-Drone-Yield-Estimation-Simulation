# src/main.py
# Async mission brain — HAL-aware state machine.
#
# HAL changes vs previous version:
#   1. Uses build_vision() factory from vision.py — no direct RowVision import
#   2. Registers SensorHub with FlightController after construction so the
#      watchdog can forward DISTANCE_SENSOR messages to virtual sensors
#   3. FusionData.stale_sensors is suppressed in SIM_MODE when sensors have
#      never received a reading (prevents false FAULT at mission start)
#   4. _check_failsafes distinguishes SIM vs hardware sensor staleness
#   5. SIM_MODE printed prominently in the startup log

import asyncio
import time
import logging
import sys
import os
import math

import config as cfg

sys.path.insert(0, os.path.dirname(__file__))

from sensors    import SensorHub
from navigation import FlightController
from vision     import build_vision, VisionResult, MockVision
from physics    import PointMassModel, lateral_nudge_diffphys

# ── Logging ────────────────────────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("logs/mission.log"),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger("brain")


# ══════════════════════════════════════════════════════════════════════════
# States
# ══════════════════════════════════════════════════════════════════════════
class S:
    TAKEOFF       = "TAKEOFF"
    ALLEY_FOLLOW  = "ALLEY_FOLLOW"
    EXIT_MANEUVER = "EXIT_MANEUVER"
    SEARCH_NEXT   = "SEARCH_NEXT"
    RTL           = "RTL"
    FAULT         = "FAULT"
    DONE          = "DONE"


# ══════════════════════════════════════════════════════════════════════════
# Sensor fusion snapshot
# ══════════════════════════════════════════════════════════════════════════
class FusionData:
    """
    One consistent snapshot of all hardware sensor readings per loop tick.

    Staleness handling:
      SIM_MODE=False  → standard: any sensor over SENSOR_TIMEOUT_S is stale
      SIM_MODE=True   → MAVLink sensors are fed by the FC watchdog.
                        In the first few seconds before Gazebo sends any
                        DISTANCE_SENSOR messages, age() is inf.  We give the
                        system a 15 s grace window at startup before reporting
                        stale sensors, preventing false FAULT at boot.
    """

    __slots__ = ("lidar", "us_left", "us_right", "stale_sensors")

    # Class-level startup timestamp so grace window is shared across ticks
    _startup_time: float = time.monotonic()
    _SIM_GRACE_S:  float = 15.0   # seconds to suppress stale-sensor faults

    def __init__(self, hub: SensorHub, timeout: float):
        self.lidar    = hub.lidar.read()
        self.us_left  = hub.us_left.read()
        self.us_right = hub.us_right.read()

        raw_stale = hub.any_stale(timeout)

        if cfg.SIM_MODE:
            elapsed = time.monotonic() - FusionData._startup_time
            if elapsed < FusionData._SIM_GRACE_S:
                # Still in startup grace — suppress stale reports
                self.stale_sensors = []
            else:
                self.stale_sensors = raw_stale
        else:
            self.stale_sensors = raw_stale

    def row_end_detected(self, vision: VisionResult) -> bool:
        """
        Triple-gate exit: LiDAR opens up AND one side opens up AND green drops.
        In SIM_MODE with no camera (MockVision), green_density is always 0.40
        so the density gate never falsely triggers a row-end.
        """
        lidar_open  = (self.lidar is not None and
                       self.lidar > cfg.LIDAR_ROW_END_M)
        left_open   = (self.us_left  is not None and
                       self.us_left  > cfg.US_ROW_END_M)
        right_open  = (self.us_right is not None and
                       self.us_right > cfg.US_ROW_END_M)
        side_open   = left_open or right_open
        density_low = (vision is not None and
                       vision.green_density < cfg.GREEN_DENSITY_THRESHOLD)
        return lidar_open and side_open and density_low

    def lateral_nudge(self) -> float:
        """
        DiffPhysDrone smooth corridor centering.
        If sensors are None (SITL before first DISTANCE_SENSOR arrives),
        lateral_nudge_diffphys treats them as 99 m — no correction commanded.
        """
        L = self.us_left  if self.us_left  is not None else 99.0
        R = self.us_right if self.us_right is not None else 99.0
        approach_L = max(0.0, (cfg.US_WALL_CLOSE_M - L) * cfg.CRUISE_SPEED_MS)
        approach_R = max(0.0, (cfg.US_WALL_CLOSE_M - R) * cfg.CRUISE_SPEED_MS)
        return lateral_nudge_diffphys(
            self.us_left, self.us_right, approach_L, approach_R)


# ══════════════════════════════════════════════════════════════════════════
# Mission brain
# ══════════════════════════════════════════════════════════════════════════
class OrchardBrain:

    def __init__(self):
        log.info("=" * 60)
        log.info(f"  Orchard Drone  |  SIM_MODE = {cfg.SIM_MODE}")
        log.info(f"  FC connection  = {cfg.FC_CONNECTION}")
        log.info(f"  Sensor timeout = {cfg.SENSOR_TIMEOUT_S} s")
        log.info("=" * 60)

        # 1. Flight controller (connection string from config)
        self.fc = FlightController()

        # 2. Sensor hub (MAVLink virtual or GPIO/serial hardware)
        self.hub = SensorHub()

        # 3. Wire sensor hub to FC watchdog BEFORE sensors start
        #    so no DISTANCE_SENSOR message is missed
        self.fc.register_sensor_hub(self.hub)

        # 4. Vision (MockVision, RowVision+GStreamer, or RowVision+device)
        self.vision = build_vision()

        # 5. DiffPhysDrone point-mass model for SITL diagnostic logging
        self._phys = PointMassModel()

        self.state    = S.TAKEOFF
        self._state_t = time.monotonic()

        self._rows             = 0
        self._photos_taken     = 0
        self._row_photo_start  = 0
        self._last_shutter     = 0.0
        self._yaw_deadline     = None
        self._dmove_init       = False

    # ── Helpers ────────────────────────────────────────────────────────────

    def _go(self, new_state: str):
        log.info(f"  [{self.state}] → [{new_state}]")
        self.state         = new_state
        self._state_t      = time.monotonic()
        self._dmove_init   = False
        self._yaw_deadline = None

    def _time_in_state(self) -> float:
        return time.monotonic() - self._state_t

    # ── Failsafes ──────────────────────────────────────────────────────────

    def _check_failsafes(self, fd: FusionData) -> bool:
        """
        Returns True if a failsafe was triggered (caller skips normal logic).

        SIM_MODE notes:
          - fd.stale_sensors is empty during the 15 s grace window, then
            reflects real MAVLink sensor freshness
          - Battery voltage is 0.0 V in SITL until SYS_STATUS arrives;
            we skip the battery check when voltage == 0.0 to avoid false RTL
          - Vision staleness: MockVision.age() is always < 1 s — no warning
        """
        if self.state in (S.RTL, S.FAULT, S.DONE):
            return False

        # Hardware / virtual sensor staleness
        if fd.stale_sensors:
            log.error(f"STALE SENSORS: {fd.stale_sensors} → LOITER")
            self.fc.loiter()
            self._go(S.FAULT)
            return True

        # Vision staleness (non-critical — just disables shutter)
        if self.vision.age() > cfg.SENSOR_TIMEOUT_S:
            log.warning("Vision stale — shutter disabled this tick")

        # Battery (skip when voltage is 0.0 — SITL before first SYS_STATUS)
        v = self.fc.battery_volts
        if 0.0 < v < 13.5:
            log.error(f"LOW BATTERY {v:.2f} V → RTL")
            self._go(S.RTL)
            return True

        return False

    # ── Camera shutter ─────────────────────────────────────────────────────

    def _maybe_shutter(self, vision: VisionResult):
        """
        In SIM_MODE with MockVision, vision.best_tree is always None so
        tree_centred and tree_close_enough are always False — no false triggers.
        """
        if vision is None:
            return
        now = time.monotonic()
        if now - self._last_shutter < cfg.SHUTTER_COOLDOWN_S:
            return
        if vision.tree_centred and vision.tree_close_enough:
            t = vision.best_tree
            log.info(f"SHUTTER off={vision.best_offset_px}px "
                     f"conf={t.confidence:.2f} "
                     f"dist~{t.distance_hint:.1f}m "
                     f"n={self._photos_taken + 1}")
            self.fc.trigger_camera()
            self._last_shutter  = now
            self._photos_taken += 1

    # ── State handlers ─────────────────────────────────────────────────────

    async def _state_takeoff(self, fd: FusionData, vision: VisionResult):
        if self._time_in_state() < 0.5:
            self.fc.set_mode("GUIDED")
            await asyncio.sleep(0.2)
            self.fc.arm()
            await asyncio.sleep(1.0)
            self.fc.takeoff(cfg.TAKEOFF_ALT_M)
            self._phys.reset(alt_m=cfg.TAKEOFF_ALT_M)
            log.info(f"Takeoff → {cfg.TAKEOFF_ALT_M} m")
        if self.fc.reached_altitude(cfg.TAKEOFF_ALT_M):
            log.info("Altitude OK → ALLEY_FOLLOW")
            self._go(S.ALLEY_FOLLOW)

    async def _state_alley_follow(self, fd: FusionData, vision: VisionResult):
        nudge = fd.lateral_nudge()
        self.fc.send_body_velocity(cfg.CRUISE_SPEED_MS, nudge, 0.0)

        if cfg.SIM_MODE:
            self._phys.step(cfg.CRUISE_SPEED_MS, nudge, 0.0, cfg.LOOP_PERIOD)

        self._maybe_shutter(vision)

        if fd.row_end_detected(vision):
            photos = self._photos_taken - self._row_photo_start
            log.info(f"Row end → EXIT_MANEUVER  (photos this row: {photos})")
            self._row_photo_start = self._photos_taken
            self._go(S.EXIT_MANEUVER)

    async def _state_exit_maneuver(self, fd: FusionData, vision: VisionResult):
        if not self._dmove_init:
            self.fc.start_distance_move(
                cfg.CRUISE_SPEED_MS, 0.0, cfg.ROW_CLEAR_DIST_M)
            self._dmove_init = True
        if not self.fc.distance_move_complete():
            self.fc.send_body_velocity(cfg.CRUISE_SPEED_MS, 0.0, 0.0)
            return
        if self._yaw_deadline is None:
            self.fc.stop()
            self.fc.condition_yaw(cfg.YAW_TURN_DEG, relative=True)
            self._yaw_deadline = (time.monotonic() +
                                  self.fc.yaw_complete_in(cfg.YAW_TURN_DEG))
            log.info("Yaw 180° in progress")
        if time.monotonic() >= self._yaw_deadline:
            self._rows += 1
            log.info(f"Row {self._rows} complete")
            self._go(S.SEARCH_NEXT)

    async def _state_search_next(self, fd: FusionData, vision: VisionResult):
        if not self._dmove_init:
            self.fc.start_distance_move(
                0.0, cfg.LATERAL_SHIFT_M, cfg.LATERAL_SHIFT_M)
            self._dmove_init = True
        if not self.fc.distance_move_complete():
            self.fc.send_body_velocity(0.0, cfg.CRUISE_SPEED_MS, 0.0)
            return
        self.fc.stop()
        yolo_density = (vision is not None and
                        vision.green_density >= cfg.GREEN_DENSITY_THRESHOLD)
        us_wall      = ((fd.us_left  is not None and
                         fd.us_left  < cfg.US_ROW_END_M) or
                        (fd.us_right is not None and
                         fd.us_right < cfg.US_ROW_END_M))
        yolo_trees   = vision is not None and len(vision.trees) > 0
        if yolo_density or us_wall or yolo_trees:
            log.info(f"New row found (density={yolo_density} "
                     f"wall={us_wall} trees={yolo_trees})")
            self._go(S.ALLEY_FOLLOW)
        else:
            log.info(f"No more rows. Rows={self._rows} Photos={self._photos_taken}")
            self._go(S.RTL)

    async def _state_rtl(self, fd: FusionData, vision: VisionResult):
        self.fc.rtl()
        self._go(S.DONE)

    async def _state_fault(self, fd: FusionData, vision: VisionResult):
        if not fd.stale_sensors:
            log.info("Sensors recovered → ALLEY_FOLLOW")
            self.fc.set_mode("GUIDED")
            self._go(S.ALLEY_FOLLOW)
        elif self._time_in_state() > 10.0:
            log.error("Fault unresolved → RTL")
            self._go(S.RTL)

    # ── Main loop ──────────────────────────────────────────────────────────

    async def run(self):
        self.hub.start_all()
        self.vision.start()
        await asyncio.sleep(2.0)   # let threads warm up

        dispatch = {
            S.TAKEOFF:       self._state_takeoff,
            S.ALLEY_FOLLOW:  self._state_alley_follow,
            S.EXIT_MANEUVER: self._state_exit_maneuver,
            S.SEARCH_NEXT:   self._state_search_next,
            S.RTL:           self._state_rtl,
            S.FAULT:         self._state_fault,
        }

        try:
            while self.state != S.DONE:
                tick_start = time.monotonic()
                fd     = FusionData(self.hub, cfg.SENSOR_TIMEOUT_S)
                vision = self.vision.read()
                if not self._check_failsafes(fd):
                    handler = dispatch.get(self.state)
                    if handler:
                        await handler(fd, vision)
                elapsed = time.monotonic() - tick_start
                await asyncio.sleep(max(0.0, cfg.LOOP_PERIOD - elapsed))

        except asyncio.CancelledError:
            log.warning("Mission cancelled")
        except KeyboardInterrupt:
            log.warning("Operator interrupt → RTL")
            self.fc.rtl()
        finally:
            self.fc.stop()
            self.hub.stop_all()
            self.vision.stop()
            self.vision.cleanup()
            self.fc.close()
            log.info(f"Mission ended — rows={self._rows} photos={self._photos_taken}")


# ── Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    brain = OrchardBrain()
    asyncio.run(brain.run())
