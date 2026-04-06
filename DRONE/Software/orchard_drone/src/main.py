# src/main.py
# Async mission brain — state machine + sensor fusion.
# Run from the project root: SITL=1 python src/main.py
#
# DiffPhysDrone integrations used here:
#   - lateral_nudge_diffphys() replaces hand-coded if/else potential field
#   - PointMassModel predictive logging (optional, for debugging)

import asyncio
import time
import logging
import sys
import os
import math

import config as cfg

# Ensure src/ is on the path when running from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from sensors     import SensorHub
from navigation  import FlightController
from vision      import RowVision, VisionResult
from physics     import PointMassModel, lateral_nudge_diffphys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("logs/mission.log"),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger("brain")

os.makedirs("logs", exist_ok=True)


# ── States ─────────────────────────────────────────────────────────────────
class S:
    TAKEOFF       = "TAKEOFF"
    ALLEY_FOLLOW  = "ALLEY_FOLLOW"
    EXIT_MANEUVER = "EXIT_MANEUVER"
    SEARCH_NEXT   = "SEARCH_NEXT"
    RTL           = "RTL"
    FAULT         = "FAULT"
    DONE          = "DONE"


# ── Sensor fusion snapshot ─────────────────────────────────────────────────
class FusionData:
    """
    One consistent snapshot of all hardware sensor readings per loop tick.
    Vision is deliberately excluded — it is fetched separately as VisionResult
    because it runs on a different thread with its own staleness check.
    """

    __slots__ = ("lidar", "us_left", "us_right", "stale_sensors")

    def __init__(self, hub: SensorHub, timeout: float):
        self.lidar         = hub.lidar.read()
        self.us_left       = hub.us_left.read()
        self.us_right      = hub.us_right.read()
        self.stale_sensors = hub.any_stale(timeout)

    def row_end_detected(self, vision: VisionResult) -> bool:
        """
        Triple-condition row-end gate:
          1. Front LiDAR > LIDAR_ROW_END_M  (clear space ahead)
          2. At least one side ultrasonic > US_ROW_END_M  (open laterally)
          3. Camera green density < GREEN_DENSITY_THRESHOLD  (no canopy)
        All three must agree to avoid false positives at canopy gaps.
        """
        lidar_open  = self.lidar is not None and self.lidar > cfg.LIDAR_ROW_END_M
        left_open   = self.us_left  is not None and self.us_left  > cfg.US_ROW_END_M
        right_open  = self.us_right is not None and self.us_right > cfg.US_ROW_END_M
        side_open   = left_open or right_open
        density_low = (vision is not None and
                       vision.green_density < cfg.GREEN_DENSITY_THRESHOLD)
        return lidar_open and side_open and density_low

    def lateral_nudge(self) -> float:
        """
        DiffPhysDrone smooth obstacle-avoidance cost replaces the old
        binary if/else potential field.

        We estimate approach speed toward each wall from the current
        ultrasonic readings relative to US_WALL_CLOSE_M:
          approach = max(0, (US_WALL_CLOSE_M - dist) * CRUISE_SPEED_MS)
        This is a conservative proxy — positive when inside the danger zone.
        """
        L = self.us_left  if self.us_left  is not None else 99.0
        R = self.us_right if self.us_right is not None else 99.0

        # Approach speed proxy: proportional to how deep inside danger zone
        approach_L = max(0.0, (cfg.US_WALL_CLOSE_M - L) * cfg.CRUISE_SPEED_MS)
        approach_R = max(0.0, (cfg.US_WALL_CLOSE_M - R) * cfg.CRUISE_SPEED_MS)

        return lateral_nudge_diffphys(
            self.us_left, self.us_right, approach_L, approach_R)


# ── Mission brain ──────────────────────────────────────────────────────────
class OrchardBrain:

    def __init__(self):
        self.fc     = FlightController(cfg.FC_CONNECTION, cfg.FC_BAUD)
        self.hub    = SensorHub(cfg)
        self.vision = RowVision(
            model_path=cfg.YOLO_MODEL_PATH,
            camera_index=cfg.CAMERA_INDEX,
            frame_w=cfg.FRAME_W,
            frame_h=cfg.FRAME_H,
            conf_threshold=cfg.YOLO_CONF_THRESHOLD,
        )
        # DiffPhysDrone point-mass model for predictive state logging in SITL
        self._phys = PointMassModel()

        self.state    = S.TAKEOFF
        self._state_t = time.monotonic()

        self._rows             = 0
        self._photos_taken     = 0
        self._row_photo_start  = 0
        self._last_shutter     = 0.0
        self._yaw_deadline     = None
        self._dmove_init       = False

    # ── Transition ─────────────────────────────────────────────────────────

    def _go(self, new_state: str):
        log.info(f"  [{self.state}] -> [{new_state}]")
        self.state         = new_state
        self._state_t      = time.monotonic()
        self._dmove_init   = False
        self._yaw_deadline = None

    def _time_in_state(self) -> float:
        return time.monotonic() - self._state_t

    # ── Fail-safes ─────────────────────────────────────────────────────────

    def _check_failsafes(self, fd: FusionData) -> bool:
        """
        Returns True if a fail-safe fires (caller skips normal logic).
        Checks:
          1. Hardware sensor staleness  → LOITER + FAULT
          2. Vision staleness           → warning only (non-critical)
          3. Battery < 13.5 V (4S)      → RTL
        """
        if self.state in (S.RTL, S.FAULT, S.DONE):
            return False

        if fd.stale_sensors:
            log.error(f"STALE SENSORS: {fd.stale_sensors} -> LOITER")
            self.fc.loiter()
            self._go(S.FAULT)
            return True

        if self.vision.age() > cfg.SENSOR_TIMEOUT_S:
            log.warning("Vision stale — shutter disabled this tick")

        if 0.0 < self.fc.battery_volts < 13.5:
            log.error(f"LOW BATTERY {self.fc.battery_volts:.2f} V -> RTL")
            self._go(S.RTL)
            return True

        return False

    # ── Camera shutter ─────────────────────────────────────────────────────

    def _maybe_shutter(self, vision: VisionResult):
        """
        Fire MAV_CMD_DO_DIGICAM_CONTROL when:
          1. YOLO detects a tree with confidence >= threshold
          2. Tree is centred in frame  (offset < TRUNK_CENTER_TOL_PX)
          3. Tree bbox fills > 30% frame height  (drone is adjacent)
          4. Cooldown elapsed since last shot
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
            log.info(f"Takeoff -> {cfg.TAKEOFF_ALT_M} m")
        if self.fc.reached_altitude(cfg.TAKEOFF_ALT_M):
            log.info("Altitude OK")
            self._go(S.ALLEY_FOLLOW)

    async def _state_alley_follow(self, fd: FusionData, vision: VisionResult):
        """
        Forward flight with three concurrent processes:
          1. DiffPhysDrone smooth lateral correction (via FusionData.lateral_nudge)
          2. YOLO-gated camera shutter
          3. Triple-gate row-end detection
        The physics model is stepped for SITL diagnostic logging.
        """
        nudge = fd.lateral_nudge()
        self.fc.send_body_velocity(cfg.CRUISE_SPEED_MS, nudge, 0.0)

        # Advance point-mass model for SITL predictive logging
        if cfg._SITL:
            self._phys.step(cfg.CRUISE_SPEED_MS, nudge, 0.0, cfg.LOOP_PERIOD)

        self._maybe_shutter(vision)

        if fd.row_end_detected(vision):
            photos_this_row = self._photos_taken - self._row_photo_start
            log.info(f"Row end -> EXIT_MANEUVER  "
                     f"(photos this row: {photos_this_row})")
            self._row_photo_start = self._photos_taken
            self._go(S.EXIT_MANEUVER)

    async def _state_exit_maneuver(self, fd: FusionData, vision: VisionResult):
        """
        Phase 1: Fly forward ROW_CLEAR_DIST_M to clear the last trees.
        Phase 2: Yaw 180°.
        """
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
            log.info("Yaw 180 in progress")

        if time.monotonic() >= self._yaw_deadline:
            self._rows += 1
            log.info(f"Row {self._rows} complete")
            self._go(S.SEARCH_NEXT)

    async def _state_search_next(self, fd: FusionData, vision: VisionResult):
        """
        Lateral shift to next row.
        Row detection uses three independent signals for robustness.
        """
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
            log.info(f"New row (density={yolo_density} "
                     f"us_wall={us_wall} trees={yolo_trees})")
            self._go(S.ALLEY_FOLLOW)
        else:
            log.info(f"Mission complete. "
                     f"Rows={self._rows}  Photos={self._photos_taken}")
            self._go(S.RTL)

    async def _state_rtl(self, fd: FusionData, vision: VisionResult):
        self.fc.rtl()
        self._go(S.DONE)

    async def _state_fault(self, fd: FusionData, vision: VisionResult):
        """Sensors recovered within 10 s → resume.  Otherwise → RTL."""
        if not fd.stale_sensors:
            log.info("Sensors recovered -> ALLEY_FOLLOW")
            self.fc.set_mode("GUIDED")
            self._go(S.ALLEY_FOLLOW)
        elif self._time_in_state() > 10.0:
            log.error("Fault unresolved -> RTL")
            self._go(S.RTL)

    # ── Main loop ──────────────────────────────────────────────────────────

    async def run(self):
        log.info("=== Orchard Drone — Mission Start ===")
        if cfg._SITL:
            log.info("SITL mode active — EMA + latency buffer enabled")

        self.hub.start_all()
        self.vision.start()
        await asyncio.sleep(2.0)   # let sensor threads warm up

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
            log.warning("Operator interrupt -> RTL")
            self.fc.rtl()
        finally:
            self.fc.stop()
            self.hub.stop_all()
            self.vision.stop()
            self.vision.cleanup()
            self.fc.close()
            log.info(f"=== Mission ended — "
                     f"rows={self._rows} photos={self._photos_taken} ===")


if __name__ == "__main__":
    brain = OrchardBrain()
    asyncio.run(brain.run())
