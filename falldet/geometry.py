"""
Geometry used by the two fall rules in the paper.

- Sec. III-F, Eq. (1): torso tilt angle from shoulder/hip mid-points.
- Sec. III-E: bounding-box aspect ratio baseline (AR = W / H).
"""

import math

# MediaPipe Pose landmark indices
L_SHOULDER, R_SHOULDER = 11, 12
L_HIP, R_HIP = 23, 24

EPS = 1e-6

# Table II angle bins (degrees)
ANGLE_BINS = [
    (0, 15, "Upright / Standing"),
    (15, 30, "Slight Bend"),
    (30, 45, "Moderate Lean / Sitting"),
    (45, 60, "High Incline / Pre-fall"),
    (60, 75, "Near-Horizontal / Falling"),
    (75, 90, "Fully Horizontal (Fallen)"),
]


def midpoint(a, b):
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


def torso_angle(shoulder_mid, hip_mid):
    """
    Eq. (1): theta = atan( |x_hip - x_shoulder| / (|y_hip - y_shoulder| + eps) )

    0 deg = upright torso, 90 deg = horizontal torso.
    Points are in pixels (x right, y down).
    """
    dx = abs(hip_mid[0] - shoulder_mid[0])
    dy = abs(hip_mid[1] - shoulder_mid[1])
    if dx == 0 and dy == 0:
        return None
    return math.degrees(math.atan2(dx, dy + EPS))


def torso_angle_from_landmarks(landmarks, width, height, min_visibility=0.4):
    """
    Torso angle from MediaPipe landmarks normalised to an image of size
    (width, height). Returns None if a needed landmark is missing or not
    visible enough.
    """
    try:
        ls, rs = landmarks[L_SHOULDER], landmarks[R_SHOULDER]
        lh, rh = landmarks[L_HIP], landmarks[R_HIP]
    except (IndexError, TypeError):
        return None

    if min(ls.visibility, rs.visibility, lh.visibility, rh.visibility) < min_visibility:
        return None

    shoulder_mid = midpoint((ls.x * width, ls.y * height), (rs.x * width, rs.y * height))
    hip_mid = midpoint((lh.x * width, lh.y * height), (rh.x * width, rh.y * height))
    return torso_angle(shoulder_mid, hip_mid)


def is_fall_angle(angle, thresh=45.0):
    """Decision rule of Sec. III-F: theta >= 45 deg -> fall (horizontal)."""
    return angle is not None and angle >= thresh


def aspect_ratio(box_w, box_h):
    """Sec. III-E baseline: AR = W / H."""
    if box_h <= 0:
        return None
    return box_w / box_h


def is_fall_aspect(ar, thresh=1.0):
    """Baseline rule: box wider than tall -> possibly fallen."""
    return ar is not None and ar > thresh


def angle_bin(angle):
    """Index into ANGLE_BINS for an angle in [0, 90], or None."""
    if angle is None:
        return None
    for i, (lo, hi, _) in enumerate(ANGLE_BINS):
        if lo <= angle < hi or (i == len(ANGLE_BINS) - 1 and angle <= hi):
            return i
    return None
