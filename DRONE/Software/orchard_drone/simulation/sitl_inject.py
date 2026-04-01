# simulation/sitl_inject.py
# SITL sensor injector — runs alongside main.py in SITL mode.
# Uses the DiffPhysDrone PointMassModel to generate realistic synthetic
# sensor readings that evolve with the simulated drone's position,
# instead of using fixed stub values.
#
# Usage:
#   Terminal 1: SITL=1 python src/main.py
#   Terminal 2: python simulation/sitl_inject.py
#
# The injector reads the shared physics model state from a file
# (sitl_state.json written by main.py) and writes sensor values back
# to a UDP socket that the SensorHub can read.
#
# For basic testing you can also run it standalone — it will print
# what the sensors would read at each timestep.

import sys
import os
import time
import math
import json
import threading

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import config as cfg
from physics import PointMassModel

# ── Row geometry (must match orchard.sdf) ─────────────────────────────────
ROW_LENGTH_M   = 20.0
ROW_WIDTH_M    = 5.0    # alley width between tree rows
NUM_ROWS       = 3
TREE_SPACING_M = 4.0    # trees every 4 m along each row


class SITLInjector:
    """
    Simulates all sensor readings using the DiffPhysDrone point-mass model.

    Sensor models:
      - Front LiDAR:  distance to the row end (or open space past row_end)
      - Left/Right US: distance to each tree wall in the corridor
      - Camera:         green density based on position along row
    """

    def __init__(self):
        self._phys = PointMassModel()
        self._phys.reset(x=0.0, y=ROW_WIDTH_M / 2.0, alt_m=cfg.TAKEOFF_ALT_M)
        self._vx   = 0.0
        self._vy   = 0.0
        self._row  = 0      # current row index (0-based)
        self._dir  = 1      # +1 = forward along X, -1 = backward
        self._lock = threading.Lock()
        self._running = True

    def update_command(self, vx: float, vy: float):
        with self._lock:
            self._vx = vx
            self._vy = vy

    def _lidar_distance(self, x: float) -> float:
        """Front LiDAR: distance to the row-end wall along current direction."""
        if self._dir == 1:
            dist = ROW_LENGTH_M - x
        else:
            dist = x
        # Once outside the row, return a large open-space value
        return max(0.05, dist) if 0 <= x <= ROW_LENGTH_M else 22.0

    def _us_distance(self, y: float, side: str) -> float:
        """
        Side ultrasonic: distance to nearest tree wall.
        The alley has tree walls at y=0 and y=ROW_WIDTH_M.
        """
        if side == "left":
            # Left wall is at y=0 in local corridor frame
            dist = y
        else:
            dist = ROW_WIDTH_M - y
        # Add realistic noise: ±2 cm
        noise = (hash((side, round(y * 1000))) % 40 - 20) / 1000.0
        return max(0.02, dist + noise)

    def _green_density(self, x: float) -> float:
        """
        Green density: high (0.6) inside row, drops to near-zero at row end.
        Models a smooth transition over the last 3 m of the row.
        """
        margin = 3.0
        if self._dir == 1:
            dist_to_end = ROW_LENGTH_M - x
        else:
            dist_to_end = x
        if dist_to_end > margin:
            return 0.60
        return max(0.02, 0.60 * (dist_to_end / margin))

    def step(self, dt: float) -> dict:
        with self._lock:
            vx, vy = self._vx, self._vy

        pos, vel = self._phys.step(vx, vy, 0.0, dt)
        x = pos[0]
        y = pos[1]

        sensors = {
            "lidar_m":       round(self._lidar_distance(x), 3),
            "us_left_m":     round(self._us_distance(y, "left"), 3),
            "us_right_m":    round(self._us_distance(y, "right"), 3),
            "green_density": round(self._green_density(x), 3),
            "alt_m":         round(self._phys.altitude_m, 3),
            "vx":            round(vel[0], 3),
            "vy":            round(vel[1], 3),
            "pos_x":         round(x, 3),
            "pos_y":         round(y, 3),
        }
        return sensors

    def run_print_loop(self):
        """Standalone mode: print sensor readings to stdout at 20 Hz."""
        print("SITL Injector running (standalone print mode)")
        print(f"{'lidar':>8} {'US_L':>7} {'US_R':>7} {'green':>7} "
              f"{'alt':>6} {'x':>6} {'y':>6}")
        print("-" * 58)

        # Simulate: forward at cruise speed for 20 m, then open space
        self.update_command(cfg.CRUISE_SPEED_MS, 0.0)
        t = 0.0

        while self._running:
            dt      = cfg.LOOP_PERIOD
            sensors = self.step(dt)
            t      += dt

            # Auto-stop demo after 30 seconds
            if t > 30.0:
                print("\n[Injector] 30 s complete — stopping demo")
                break

            print(f"{sensors['lidar_m']:8.2f} "
                  f"{sensors['us_left_m']:7.2f} "
                  f"{sensors['us_right_m']:7.2f} "
                  f"{sensors['green_density']:7.3f} "
                  f"{sensors['alt_m']:6.2f} "
                  f"{sensors['pos_x']:6.2f} "
                  f"{sensors['pos_y']:6.2f}")
            time.sleep(dt)


if __name__ == "__main__":
    inj = SITLInjector()
    try:
        inj.run_print_loop()
    except KeyboardInterrupt:
        print("\nInjector stopped")