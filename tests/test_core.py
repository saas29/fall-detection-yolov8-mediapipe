"""
Unit tests for the model-free parts (Eq. 1, decision rules, persistence /
latch, metrics, label parsing).

    python -m unittest discover tests
"""

import csv
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from falldet.geometry import (ANGLE_BINS, R_HIP, R_SHOULDER, L_HIP, L_SHOULDER,  # noqa: E402
                              angle_bin, aspect_ratio, is_fall_angle, is_fall_aspect,
                              torso_angle, torso_angle_from_landmarks)
from falldet.metrics import compute_metrics  # noqa: E402
from falldet.state import FallStateMachine  # noqa: E402
from falldet.evaluate import load_frame_labels, load_labels  # noqa: E402


def _lm(x, y, vis=1.0):
    return SimpleNamespace(x=x, y=y, visibility=vis)


def _landmarks(shoulder, hip, vis=1.0):
    lms = [_lm(0, 0, 0)] * 33
    lms = list(lms)
    lms[L_SHOULDER] = lms[R_SHOULDER] = _lm(*shoulder, vis)
    lms[L_HIP] = lms[R_HIP] = _lm(*hip, vis)
    return lms


class TestEquation1(unittest.TestCase):
    def test_upright_is_zero(self):
        self.assertAlmostEqual(torso_angle((100, 50), (100, 150)), 0.0, places=3)

    def test_horizontal_is_ninety(self):
        self.assertAlmostEqual(torso_angle((50, 100), (150, 100)), 90.0, places=3)

    def test_diagonal_is_45(self):
        self.assertAlmostEqual(torso_angle((0, 0), (100, 100)), 45.0, places=3)

    def test_symmetric_and_degenerate(self):
        self.assertAlmostEqual(torso_angle((0, 0), (30, 100)), torso_angle((30, 0), (0, 100)))
        self.assertIsNone(torso_angle((5, 5), (5, 5)))

    def test_landmarks_use_pixel_scale(self):
        # normalised dx=0.5, dy=0.5 but crop is 2x wider than tall -> > 45 deg
        lms = _landmarks((0.25, 0.25), (0.75, 0.75))
        self.assertAlmostEqual(torso_angle_from_landmarks(lms, 200, 100), 63.435, places=2)
        self.assertAlmostEqual(torso_angle_from_landmarks(lms, 100, 100), 45.0, places=3)

    def test_low_visibility_returns_none(self):
        lms = _landmarks((0.5, 0.2), (0.5, 0.8), vis=0.1)
        self.assertIsNone(torso_angle_from_landmarks(lms, 100, 100))


class TestRules(unittest.TestCase):
    def test_angle_rule_inclusive(self):
        self.assertTrue(is_fall_angle(45.0))
        self.assertFalse(is_fall_angle(44.9))
        self.assertFalse(is_fall_angle(None))

    def test_aspect_rule(self):
        self.assertTrue(is_fall_aspect(aspect_ratio(200, 100)))
        self.assertFalse(is_fall_aspect(aspect_ratio(50, 100)))
        self.assertIsNone(aspect_ratio(10, 0))

    def test_angle_bins(self):
        self.assertEqual(angle_bin(0), 0)
        self.assertEqual(angle_bin(44.9), 2)
        self.assertEqual(angle_bin(45), 3)
        self.assertEqual(angle_bin(90), len(ANGLE_BINS) - 1)
        self.assertIsNone(angle_bin(None))


class TestPersistenceLatch(unittest.TestCase):
    def test_needs_n_consecutive(self):
        sm = FallStateMachine(persist_frames=3, latch_frames=5)
        outs = [sm.update(f) for f in [1, 1, 0, 1, 1]]
        self.assertFalse(any(o[0] for o in outs))
        self.assertFalse(sm.confirmed_ever)
        in_fall, new = sm.update(1)
        self.assertTrue(in_fall)
        self.assertTrue(new)

    def test_latch_holds_m_frames(self):
        sm = FallStateMachine(persist_frames=2, latch_frames=4)
        sm.update(1)
        sm.update(1)  # confirmed, frame 1 of latch
        states = [sm.update(0)[0] for _ in range(5)]
        self.assertEqual(states, [True, True, True, False, False])

    def test_one_event_while_lying(self):
        sm = FallStateMachine(persist_frames=2, latch_frames=3)
        news = [sm.update(1)[1] for _ in range(20)]
        self.assertEqual(sum(news), 1)
        self.assertEqual(sm.confirmations, 1)

    def test_unknown_frames_hold_counter(self):
        sm = FallStateMachine(persist_frames=4, latch_frames=3)
        for f in [True, True, None, None, True]:
            self.assertFalse(sm.update(f)[0])
        self.assertEqual(sm.consec, 3)
        self.assertTrue(sm.update(True)[1])
        # once confirmed, unknown frames (person hidden on the floor) keep FALL on
        self.assertTrue(all(sm.update(None)[0] for _ in range(10)))
        # an explicit upright frame ends it after the latch
        states = [sm.update(False)[0] for _ in range(5)]
        self.assertEqual(states, [True, True, False, False, False])

    def test_reset(self):
        sm = FallStateMachine(2, 3)
        sm.update(1); sm.update(1)
        sm.reset()
        self.assertFalse(sm.confirmed_ever)
        self.assertEqual(sm.update(0), (False, False))


class TestRotationFallback(unittest.TestCase):
    def test_unrotate_restores_coordinates(self):
        try:
            import cv2
            from mediapipe.framework.formats import landmark_pb2
            from falldet.detector import _unrotate_landmarks
        except ImportError:
            self.skipTest("mediapipe not installed")

        w, h = 200, 100
        pts = [(30, 20), (170, 80)]  # pixel points in the original crop
        img_like = (h, w)
        for rot in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE):
            lst = landmark_pb2.NormalizedLandmarkList()
            for (x, y) in pts:
                # where the point lands in the rotated (h x w -> w x h) image
                if rot == cv2.ROTATE_90_CLOCKWISE:
                    xr, yr = (img_like[0] - y) / h, x / w
                else:
                    xr, yr = y / h, (img_like[1] - x) / w
                lst.landmark.add(x=xr, y=yr)
            back = _unrotate_landmarks(lst, rot)
            for (x, y), lm in zip(pts, back.landmark):
                self.assertAlmostEqual(lm.x * w, x, places=4)
                self.assertAlmostEqual(lm.y * h, y, places=4)


class TestMetrics(unittest.TestCase):
    def test_known_values(self):
        m = compute_metrics([1, 1, 1, 0, 0, 0], [1, 1, 0, 0, 0, 1])
        self.assertEqual((m["tp"], m["tn"], m["fp"], m["fn"]), (2, 2, 1, 1))
        self.assertAlmostEqual(m["accuracy"], 4 / 6)
        self.assertAlmostEqual(m["precision"], 2 / 3)
        self.assertAlmostEqual(m["recall"], 2 / 3)
        self.assertAlmostEqual(m["f1"], 2 / 3)

    def test_empty_and_no_positives(self):
        self.assertEqual(compute_metrics([], [])["accuracy"], 0.0)
        m = compute_metrics([0, 0], [0, 0])
        self.assertEqual(m["accuracy"], 1.0)
        self.assertEqual(m["f1"], 0.0)


class TestLabels(unittest.TestCase):
    def test_load_labels_and_frames(self):
        with tempfile.TemporaryDirectory() as d:
            lp = os.path.join(d, "l.csv")
            with open(lp, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["video", "label", "view"])
                w.writerow(["a.mp4", 1, "side"])
                w.writerow(["b.mp4", 0, ""])
            rows = load_labels(lp)
            self.assertEqual(rows[0], {"video": "a.mp4", "label": 1, "view": "side"})
            self.assertIsNone(rows[1]["view"])

            fp = os.path.join(d, "f.csv")
            with open(fp, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["video", "frame", "label"])
                w.writerow(["a.mp4", 3, 1])
                w.writerow(["a.mp4", 4, -1])
            fl = load_frame_labels(fp)
            self.assertEqual(fl["a.mp4"], {3: 1, 4: -1})


if __name__ == "__main__":
    unittest.main()
