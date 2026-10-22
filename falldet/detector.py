"""
Core pipeline (Fig. 1): frame -> YOLOv8 person detection -> MediaPipe Pose on
the person ROI -> torso tilt angle (Eq. 1) or aspect ratio -> persistence /
latch -> alert.

Each detected person gets a ByteTrack ID so fall state (and the MediaPipe
tracker) is kept per person instead of globally.
"""

import os
import time
from dataclasses import dataclass, field
from typing import Optional

import cv2

from .geometry import (aspect_ratio, is_fall_angle, is_fall_aspect,
                       torso_angle_from_landmarks)
from .state import FallStateMachine


@dataclass
class DetectorConfig:
    method: str = "angle"            # "angle" (proposed) or "aspect" (baseline)
    angle_thresh: float = 45.0       # Sec. III-F
    ar_thresh: float = 1.0           # Sec. III-E
    persist_frames: int = 10         # N
    latch_frames: int = 30           # M
    weights: str = "models/yolov8n.pt"
    conf: float = 0.3                # YOLO person confidence (low: lying people score lower)
    imgsz: int = 640                 # YOLO inference size
    pose_complexity: int = 1         # MediaPipe 0 / 1 / 2
    min_visibility: float = 0.4      # landmark visibility needed for Eq. 1
    crop_pad: float = 0.10           # ROI padding so the whole body is in the crop
    # MediaPipe's person detector rarely finds a lying body; if pose fails on a
    # wide box, retry on the ROI rotated by +-90 deg and map landmarks back
    rotate_fallback: bool = True
    rotate_min_ar: float = 0.7
    track: bool = True               # per-person ByteTrack IDs
    max_people: int = 6              # cap on pose instances
    stale_frames: int = 45           # drop a track's state after this many unseen frames
    device: Optional[str] = None     # e.g. "cpu", "0"


@dataclass
class PersonResult:
    track_id: int
    box: tuple                       # x1, y1, x2, y2 in frame pixels
    angle: Optional[float]
    ar: Optional[float]
    flagged: Optional[bool]          # raw per-frame fall condition, None = unknown
    in_fall: bool                    # after persistence + latch
    newly_confirmed: bool
    landmarks: object = None         # MediaPipe landmark list (crop-normalised)
    crop_box: tuple = None           # padded ROI the landmarks refer to


@dataclass
class FrameResult:
    persons: list = field(default_factory=list)
    fall: bool = False               # any person in FALL state
    newly_confirmed: bool = False    # a new fall event started this frame
    timings: dict = field(default_factory=dict)


def _merge_flags(a, b):
    """Combine flags of boxes sharing one state: any True wins, None = unknown."""
    if a is True or b is True:
        return True
    if a is False or b is False:
        return False
    return None


def _unrotate_landmarks(landmarks, rot):
    """Map landmarks found on a 90-deg rotated crop back to the crop's frame."""
    from mediapipe.framework.formats import landmark_pb2

    out = landmark_pb2.NormalizedLandmarkList()
    out.CopyFrom(landmarks)
    for lm in out.landmark:
        x, y = lm.x, lm.y
        if rot == cv2.ROTATE_90_CLOCKWISE:
            lm.x, lm.y = y, 1.0 - x
        else:  # ROTATE_90_COUNTERCLOCKWISE
            lm.x, lm.y = 1.0 - y, x
    return out


class _Track:
    def __init__(self, cfg, pose_factory):
        self.sm = FallStateMachine(cfg.persist_frames, cfg.latch_frames)
        self.pose = pose_factory() if pose_factory else None
        self.last_seen = 0

    def close(self):
        if self.pose is not None:
            self.pose.close()


class FallDetector:
    def __init__(self, cfg: DetectorConfig = None):
        self.cfg = cfg or DetectorConfig()
        if self.cfg.method not in ("angle", "aspect"):
            raise ValueError(f"unknown method: {self.cfg.method}")

        from ultralytics import YOLO
        weights = self.cfg.weights
        if not os.path.exists(weights):
            # bare name (e.g. "yolov8n.pt") -> ultralytics downloads it
            weights = os.path.basename(weights)
        self.model = YOLO(weights, task="detect")  # .pt or exported dir (OpenVINO / NCNN)

        self._mp_pose = None
        self._static_pose = None
        if self.cfg.method == "angle":
            import mediapipe as mp
            self._mp_pose = mp.solutions.pose
            if self.cfg.rotate_fallback:
                self._static_pose = self._mp_pose.Pose(
                    static_image_mode=True, model_complexity=self.cfg.pose_complexity,
                    min_detection_confidence=0.5)

        self._tracks = {}
        self._frame_idx = 0
        # once a fall has been confirmed for any person in the current clip
        self.confirmed_this_run = False

    # ------------------------------------------------------------------ #

    def _new_pose(self):
        return self._mp_pose.Pose(
            static_image_mode=False,
            model_complexity=self.cfg.pose_complexity,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def _get_track(self, tid):
        tr = self._tracks.get(tid)
        if tr is None:
            if len(self._tracks) >= self.cfg.max_people:
                oldest = min(self._tracks, key=lambda k: self._tracks[k].last_seen)
                self._tracks.pop(oldest).close()
            factory = self._new_pose if self._mp_pose is not None else None
            tr = _Track(self.cfg, factory)
            self._tracks[tid] = tr
        tr.last_seen = self._frame_idx
        return tr

    def reset(self):
        """Clear all state between independent clips (eval mode)."""
        for tr in self._tracks.values():
            tr.close()
        self._tracks.clear()
        self._frame_idx = 0
        self.confirmed_this_run = False
        # fresh ByteTrack state for the next video
        predictor = getattr(self.model, "predictor", None)
        for t in getattr(predictor, "trackers", None) or []:
            try:
                t.reset()
            except Exception:
                pass

    def close(self):
        for tr in self._tracks.values():
            tr.close()
        self._tracks.clear()
        if self._static_pose is not None:
            self._static_pose.close()
            self._static_pose = None

    # ------------------------------------------------------------------ #

    def _detect(self, frame):
        kw = dict(classes=[0], conf=self.cfg.conf, imgsz=self.cfg.imgsz, verbose=False)
        if self.cfg.device is not None:
            kw["device"] = self.cfg.device
        if self.cfg.track:
            res = self.model.track(frame, persist=True, tracker="bytetrack.yaml", **kw)
        else:
            res = self.model.predict(frame, **kw)

        out = []
        if not res or res[0].boxes is None or len(res[0].boxes) == 0:
            return out
        boxes = res[0].boxes
        xyxy = boxes.xyxy.cpu().numpy()
        ids = boxes.id.int().cpu().tolist() if (self.cfg.track and boxes.id is not None) else None
        for i, b in enumerate(xyxy):
            # untracked boxes all share id 0 -> one global state (original behaviour)
            tid = ids[i] if ids is not None else 0
            out.append((tid, b))
        return out

    @staticmethod
    def _expand(box, pad, frame_shape, square=False):
        h, w = frame_shape[:2]
        x1, y1, x2, y2 = box
        bw, bh = (x2 - x1) * (1 + 2 * pad), (y2 - y1) * (1 + 2 * pad)
        if square:
            bw = bh = max(bw, bh)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        return (int(max(0, cx - bw / 2)), int(max(0, cy - bh / 2)),
                int(min(w, cx + bw / 2)), int(min(h, cy + bh / 2)))

    def _pose_angle(self, track, frame, box):
        """
        Torso angle for one person. Returns (angle, landmarks, crop_box);
        landmarks are normalised to crop_box.
        """
        x1, y1, x2, y2 = box
        crop_box = self._expand(box, self.cfg.crop_pad, frame.shape)
        cx1, cy1, cx2, cy2 = crop_box
        if cx2 <= cx1 or cy2 <= cy1:
            return None, None, None
        rgb = cv2.cvtColor(frame[cy1:cy2, cx1:cx2], cv2.COLOR_BGR2RGB)

        result = track.pose.process(rgb)
        if result.pose_landmarks:
            lms = result.pose_landmarks
            angle = torso_angle_from_landmarks(lms.landmark, cx2 - cx1, cy2 - cy1,
                                               self.cfg.min_visibility)
            if angle is not None:
                return angle, lms, crop_box

        # Fallback for wide (possibly lying) boxes: a larger square crop with
        # more context, tried upright and rotated +-90 deg.
        if self._static_pose is None or (x2 - x1) / max(1, y2 - y1) < self.cfg.rotate_min_ar:
            return None, None, crop_box

        big = self._expand(box, 0.5, frame.shape, square=True)
        bx1, by1, bx2, by2 = big
        brgb = cv2.cvtColor(frame[by1:by2, bx1:bx2], cv2.COLOR_BGR2RGB)
        for rot in (None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE):
            img = brgb if rot is None else cv2.rotate(brgb, rot)
            res = self._static_pose.process(img)
            if not res.pose_landmarks:
                continue
            lms = res.pose_landmarks if rot is None else _unrotate_landmarks(res.pose_landmarks, rot)
            angle = torso_angle_from_landmarks(lms.landmark, bx2 - bx1, by2 - by1,
                                               self.cfg.min_visibility)
            if angle is not None:
                return angle, lms, big
        return None, None, crop_box

    def process(self, frame):
        """Run the full pipeline on one BGR frame (not modified)."""
        self._frame_idx += 1
        fr = FrameResult()
        h, w = frame.shape[:2]

        t0 = time.perf_counter()
        detections = self._detect(frame)
        t1 = time.perf_counter()

        # several untracked boxes share one state machine -> combine flags first
        combined = {}
        for tid, b in detections:
            x1, y1, x2, y2 = (int(max(0, b[0])), int(max(0, b[1])),
                              int(min(w, b[2])), int(min(h, b[3])))
            bw, bh = x2 - x1, y2 - y1
            if bw <= 0 or bh <= 0:
                continue
            track = self._get_track(tid)

            ar = aspect_ratio(bw, bh)
            angle, lms, crop_box = None, None, None
            if self.cfg.method == "angle":
                angle, lms, crop_box = self._pose_angle(track, frame, (x1, y1, x2, y2))
                # no usable landmarks -> unknown, not "upright"
                flagged = None if angle is None else is_fall_angle(angle, self.cfg.angle_thresh)
            else:
                flagged = is_fall_aspect(ar, self.cfg.ar_thresh)

            p = PersonResult(tid, (x1, y1, x2, y2), angle, ar, flagged,
                             False, False, lms, crop_box)
            fr.persons.append(p)
            combined[tid] = _merge_flags(combined.get(tid), flagged)
        t2 = time.perf_counter()

        # persistence / latch per track; tracks not seen this frame are "unknown"
        status = {}
        for tid, tr in list(self._tracks.items()):
            if tid in combined:
                status[tid] = tr.sm.update(combined[tid])
            elif self._frame_idx - tr.last_seen > self.cfg.stale_frames:
                self._tracks.pop(tid).close()
            else:
                in_fall, _ = tr.sm.update(None)
                fr.fall = fr.fall or in_fall

        for p in fr.persons:
            p.in_fall, p.newly_confirmed = status.get(p.track_id, (False, False))
            fr.fall = fr.fall or p.in_fall
            fr.newly_confirmed = fr.newly_confirmed or p.newly_confirmed

        if fr.newly_confirmed:
            self.confirmed_this_run = True

        fr.timings = {"detect_ms": (t1 - t0) * 1000, "pose_ms": (t2 - t1) * 1000,
                      "total_ms": (time.perf_counter() - t0) * 1000}
        return fr


# ---------------------------------------------------------------------- #
# Drawing
# ---------------------------------------------------------------------- #

GREEN, RED, CYAN, WHITE = (0, 200, 0), (0, 0, 255), (255, 255, 0), (255, 255, 255)


def annotate(frame, fr: FrameResult, method="angle", fps=None, draw_skeleton=True):
    """Return an annotated copy of frame (boxes, IDs, angle, skeleton, banner)."""
    vis = frame.copy()

    if draw_skeleton:
        try:
            import mediapipe as mp
            drawer, conns = mp.solutions.drawing_utils, mp.solutions.pose.POSE_CONNECTIONS
        except Exception:
            drawer = None
    else:
        drawer = None

    for p in fr.persons:
        x1, y1, x2, y2 = p.box
        color = RED if (p.in_fall or p.flagged) else GREEN
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        if drawer is not None and p.landmarks is not None and p.crop_box is not None:
            cx1, cy1, cx2, cy2 = p.crop_box
            # OpenCV can't draw into a non-contiguous slice -> draw on a copy, paste back
            roi = vis[cy1:cy2, cx1:cx2].copy()
            drawer.draw_landmarks(roi, p.landmarks, conns)
            vis[cy1:cy2, cx1:cx2] = roi

        if method == "angle":
            val = "--" if p.angle is None else f"{p.angle:.0f} deg"
        else:
            val = "--" if p.ar is None else f"AR {p.ar:.2f}"
        label = f"#{p.track_id} {val}"
        if p.in_fall:
            label += " FALL"
        cv2.putText(vis, label, (x1, max(15, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    status = "FALL" if fr.fall else "OK"
    header = f"method={method}  state={status}"
    if fps is not None:
        header += f"  FPS={fps:.1f}"
    cv2.putText(vis, header, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, CYAN, 2)
    if fr.fall:
        cv2.putText(vis, "FALL DETECTED", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, RED, 3)
    return vis
