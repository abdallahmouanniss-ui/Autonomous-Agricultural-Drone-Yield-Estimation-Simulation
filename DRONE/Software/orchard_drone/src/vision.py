# src/vision.py
# Hardware Abstraction Layer for the forward-facing camera.
#
# SIM_MODE = False → cv2.VideoCapture(CAMERA_INDEX)  physical USB/CSI cam
# SIM_MODE = True  → one of three paths, tried in order:
#   1. GStreamer pipeline string (cfg.CAMERA_SIM_SOURCE starts with "gst-")
#   2. Video file or RTSP URL  (cfg.CAMERA_SIM_SOURCE is a non-empty string)
#   3. MockVision              (cfg.CAMERA_SIM_SOURCE == "")
#      MockVision never crashes. It returns a neutral VisionResult every tick
#      so the mission loop keeps flying without false triggers.
#
# Thread safety:
#   RowVision._loop runs in a daemon thread.
#   All public methods (read, age, latest_frame) are protected by _lock.
#   The main asyncio loop only calls read() — never touches OpenCV directly.

import cv2
import numpy as np
import threading
import time
import logging
from dataclasses import dataclass, field
from typing import Optional

import config as cfg

log = logging.getLogger("vision")

try:
    from ultralytics import YOLO
    _YOLO_OK = True
except ImportError:
    log.warning("ultralytics not installed — contour fallback active")
    _YOLO_OK = False


# ══════════════════════════════════════════════════════════════════════════
# Data classes
# ══════════════════════════════════════════════════════════════════════════
@dataclass
class TreeDetection:
    cx: int
    cy: int
    width: int
    height: int
    confidence: float
    distance_hint: float = 0.0


@dataclass
class VisionResult:
    green_density: float = 0.0
    trees: list = field(default_factory=list)
    best_tree: Optional[TreeDetection] = None
    frame_w: int = 640
    frame_h: int = 480

    @property
    def best_offset_px(self) -> Optional[int]:
        if self.best_tree is None:
            return None
        return self.best_tree.cx - (self.frame_w // 2)

    @property
    def tree_centred(self) -> bool:
        offset = self.best_offset_px
        return offset is not None and abs(offset) <= cfg.TRUNK_CENTER_TOL_PX

    @property
    def tree_close_enough(self) -> bool:
        if self.best_tree is None:
            return False
        return self.best_tree.height > (self.frame_h * cfg.TREE_CLOSE_BBOX_RATIO)


# ══════════════════════════════════════════════════════════════════════════
# Mock vision — safe fallback for headless SITL with no camera
# ══════════════════════════════════════════════════════════════════════════
class MockVision:
    """
    Returns a neutral VisionResult (no trees, 0.4 green density) every tick.
    Green density of 0.4 keeps the drone flying forward without triggering
    a false row-end detection (threshold is 0.15).
    age() always returns a small value so vision is never marked stale.
    """

    GREEN_DENSITY_WHILE_FLYING = 0.40

    def __init__(self):
        self._lock        = threading.Lock()
        self._last_update = time.monotonic()
        self._running     = False
        self._thread      = threading.Thread(
            target=self._loop, name="vision-mock", daemon=True)
        log.info("[MockVision] headless mock active — no camera required")

    def start(self):
        self._running = True
        self._thread.start()

    def stop(self):
        self._running = False

    def _loop(self):
        """Refreshes the timestamp so age() stays low."""
        while self._running:
            with self._lock:
                self._last_update = time.monotonic()
            time.sleep(0.1)     # 10 Hz refresh

    def read(self) -> VisionResult:
        return VisionResult(
            green_density=self.GREEN_DENSITY_WHILE_FLYING,
            trees=[],
            best_tree=None,
            frame_w=cfg.FRAME_W,
            frame_h=cfg.FRAME_H,
        )

    def latest_frame(self) -> Optional[np.ndarray]:
        return None

    def age(self) -> float:
        with self._lock:
            return time.monotonic() - self._last_update

    def cleanup(self):
        pass


# ══════════════════════════════════════════════════════════════════════════
# Real vision — wraps cv2.VideoCapture (hardware or video source)
# ══════════════════════════════════════════════════════════════════════════
class RowVision:
    """
    Runs YOLO/contour detection in a daemon thread.
    Accepts either an integer device index or a string (GStreamer / file / URL).
    """

    GREEN_LO = np.array([32, 45, 40])
    GREEN_HI = np.array([88, 255, 255])

    TREE_CLASS_NAMES    = {"tree", "trunk", "potted plant"}
    TREE_CLASS_IDS_COCO = {63}   # COCO80: "potted plant"

    def __init__(self,
                 source,                      # int index or str pipeline/path
                 frame_w: int  = 640,
                 frame_h: int  = 480,
                 conf_threshold: float = 0.45,
                 model_path: str = None):
        self.source         = source
        self.frame_w        = frame_w
        self.frame_h        = frame_h
        self.conf_threshold = conf_threshold
        self._result: Optional[VisionResult] = None
        self._raw_frame     = None
        self._lock          = threading.Lock()
        self._running       = False
        self._cap           = None
        self._last_update   = 0.0   # safe before thread starts

        self._model = None
        if _YOLO_OK and model_path:
            try:
                self._model = YOLO(model_path)
                log.info(f"[RowVision] YOLOv8 loaded: {model_path}")
            except Exception as e:
                log.warning(f"[RowVision] YOLO load failed ({e}) — contour fallback")

        self._thread = threading.Thread(
            target=self._loop, name="vision", daemon=True)

    def start(self):
        self._running = True
        self._thread.start()

    def stop(self):
        self._running = False

    def read(self) -> Optional[VisionResult]:
        with self._lock:
            return self._result

    def latest_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            return self._raw_frame.copy() if self._raw_frame is not None else None

    def age(self) -> float:
        with self._lock:
            return float('inf') if self._last_update == 0.0 \
                else time.monotonic() - self._last_update

    def _open_capture(self) -> Optional[cv2.VideoCapture]:
        """
        Open the video source. Handles:
          - integer index         → physical device
          - "gst-..." string      → GStreamer pipeline with cv2.CAP_GSTREAMER
          - any other string      → video file, RTSP, UDP stream
        Returns None if the source cannot be opened.
        """
        try:
            if isinstance(self.source, str) and self.source.startswith("gst-"):
                pipeline = self.source[4:]   # strip "gst-" prefix
                cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            elif isinstance(self.source, str):
                cap = cv2.VideoCapture(self.source)
            else:
                cap = cv2.VideoCapture(int(self.source))

            if not cap.isOpened():
                log.error(f"[RowVision] cannot open source: {self.source!r}")
                return None

            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.frame_w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_h)
            log.info(f"[RowVision] capture opened: {self.source!r}")
            return cap
        except Exception as e:
            log.error(f"[RowVision] capture error: {e}")
            return None

    def _loop(self):
        self._cap = self._open_capture()
        if self._cap is None:
            log.error("[RowVision] capture failed — thread exiting. "
                      "Use MockVision or set CAMERA_SIM_SOURCE.")
            return

        while self._running:
            t0 = time.monotonic()
            ret, frame = self._cap.read()
            if not ret:
                time.sleep(0.05)
                continue
            result = self._process(frame)
            with self._lock:
                self._result      = result
                self._raw_frame   = frame
                self._last_update = time.monotonic()
            time.sleep(max(0.0, 1.0/20 - (time.monotonic() - t0)))

    def _process(self, frame: np.ndarray) -> VisionResult:
        r = VisionResult(frame_w=self.frame_w, frame_h=self.frame_h)
        r.green_density = self._green_density(frame)
        r.trees         = (self._detect_yolo(frame) if self._model
                           else self._detect_contours(frame))
        if r.trees:
            r.best_tree = self._pick_best(r.trees)
        return r

    def _detect_yolo(self, frame: np.ndarray) -> list:
        detections = []
        try:
            res = self._model(frame, verbose=False,
                              conf=self.conf_threshold)[0]
        except Exception as e:
            log.debug(f"YOLO error: {e}")
            return detections
        for box in res.boxes:
            cls_id   = int(box.cls[0].item())
            cls_name = res.names.get(cls_id, "").lower()
            if (cls_name not in self.TREE_CLASS_NAMES and
                    cls_id not in self.TREE_CLASS_IDS_COCO):
                continue
            conf         = float(box.conf[0].item())
            x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
            bw, bh       = x2-x1, y2-y1
            cx, cy       = x1+bw//2, y1+bh//2
            dist         = (2.5 * 600) / max(bh, 1)
            detections.append(TreeDetection(cx=cx, cy=cy, width=bw,
                                            height=bh, confidence=conf,
                                            distance_hint=dist))
        return detections

    def _detect_contours(self, frame: np.ndarray) -> list:
        detections = []
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur  = cv2.GaussianBlur(gray, (5,5), 0)
        edges = cv2.Canny(blur, 40, 120)
        cnts, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in cnts:
            area = cv2.contourArea(cnt)
            if area < 800:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            if (h / max(w,1)) < 1.8:
                continue
            cx, cy = x+w//2, y+h//2
            detections.append(TreeDetection(cx=cx, cy=cy, width=w, height=h,
                                            confidence=1.0,
                                            distance_hint=(2.5*600)/max(h,1)))
        return detections

    def _pick_best(self, trees: list) -> TreeDetection:
        cx0      = self.frame_w // 2
        max_area = max(t.width * t.height for t in trees)
        max_off  = max(abs(t.cx - cx0) for t in trees) or 1
        def score(t):
            a = (t.width*t.height) / max_area
            c = 1.0 - abs(t.cx-cx0) / max_off
            return 0.5*a + 0.5*c
        return max(trees, key=score)

    def _green_density(self, frame: np.ndarray) -> float:
        roi  = frame[:int(self.frame_h*0.7), :]
        hsv  = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.GREEN_LO, self.GREEN_HI)
        return float(np.count_nonzero(mask)) / float(mask.size)

    def cleanup(self):
        if self._cap:
            self._cap.release()


# ══════════════════════════════════════════════════════════════════════════
# Factory function — called by main.py
# ══════════════════════════════════════════════════════════════════════════
def build_vision():
    """
    Returns the right vision object based on SIM_MODE and CAMERA_SIM_SOURCE.

    Decision tree:
      SIM_MODE=False  → RowVision(CAMERA_INDEX)
      SIM_MODE=True and CAMERA_SIM_SOURCE=""    → MockVision
      SIM_MODE=True and CAMERA_SIM_SOURCE set   → RowVision(CAMERA_SIM_SOURCE)
    """
    if not cfg.SIM_MODE:
        log.info("[vision] HARDWARE mode → cv2.VideoCapture(%d)" % cfg.CAMERA_INDEX)
        return RowVision(
            source=cfg.CAMERA_INDEX,
            frame_w=cfg.FRAME_W,
            frame_h=cfg.FRAME_H,
            conf_threshold=cfg.YOLO_CONF_THRESHOLD,
            model_path=cfg.YOLO_MODEL_PATH,
        )

    if cfg.CAMERA_SIM_SOURCE:
        log.info(f"[vision] SIM mode → RowVision({cfg.CAMERA_SIM_SOURCE!r})")
        return RowVision(
            source=cfg.CAMERA_SIM_SOURCE,
            frame_w=cfg.FRAME_W,
            frame_h=cfg.FRAME_H,
            conf_threshold=cfg.YOLO_CONF_THRESHOLD,
            model_path=cfg.YOLO_MODEL_PATH,
        )

    log.info("[vision] SIM mode → MockVision (no camera required)")
    return MockVision()
