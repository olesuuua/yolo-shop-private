"""Focused tests for the scale-aware pre-OCR quality gate (CPU only).

Gate: accept when full-resolution Laplacian variance >= OCR_SHARPNESS_MIN
OR normalized@OCR_SHARPNESS_NORM_HEIGHT variance >= OCR_SHARPNESS_NORM_MIN.
Thresholds come from reports/quality-gate/eval-table.md (tuning + held-out).
"""
import unittest

import cv2
import numpy as np

from config import (OCR_MIN_CROP_HEIGHT, OCR_MIN_CROP_WIDTH,
                    OCR_SHARPNESS_MIN, OCR_SHARPNESS_NORM_HEIGHT,
                    OCR_SHARPNESS_NORM_MIN)
from vision import (crop_sharpness, crop_sharpness_at_height,
                    quality_gate_accepts)


def sharp_image(height=1600, width=758):
    image = np.full((height, width, 3), 255, np.uint8)
    cv2.putText(image, "AKBA TEST 123", (30, height // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 5)
    for x in range(0, width, 40):
        cv2.line(image, (x, 0), (x, height), (0, 0, 0), 2)
    return image


class MeasurementTests(unittest.TestCase):
    def test_config_bounds_sane(self):
        self.assertTrue(OCR_SHARPNESS_MIN > 0)
        self.assertTrue(OCR_SHARPNESS_NORM_MIN > 0)
        self.assertTrue(OCR_SHARPNESS_NORM_HEIGHT >= 600)
        self.assertEqual(OCR_SHARPNESS_NORM_HEIGHT, 1000)
        self.assertEqual(OCR_SHARPNESS_MIN, 15.0)
        self.assertEqual(OCR_SHARPNESS_NORM_MIN, 30.0)

    def test_normalized_is_deterministic_and_monotonic_with_blur(self):
        image = sharp_image()
        first = crop_sharpness_at_height(image, 1000)
        second = crop_sharpness_at_height(image, 1000)
        self.assertEqual(first, second)
        blurred = cv2.GaussianBlur(image, (21, 21), 0)
        self.assertGreater(first, 0)
        self.assertGreater(
            first, crop_sharpness_at_height(blurred, 1000),
            "heavy blur must score lower at the same canonical scale")

    def test_measurement_handles_size_variants(self):
        for height, width in [(1650, 758), (1211, 541), (748, 463),
                              (557, 368)]:
            image = sharp_image(height, width)
            full = crop_sharpness(image)
            normed = crop_sharpness_at_height(image, 1000)
            self.assertTrue(np.isfinite(full) and full > 0)
            self.assertTrue(np.isfinite(normed) and normed > 0)

    def test_malformed_empty_safe_fail_closed(self):
        self.assertEqual(crop_sharpness_at_height(None, 1000), 0.0)
        self.assertEqual(
            crop_sharpness_at_height(np.zeros((0, 0, 3), np.uint8), 1000),
            0.0)
        self.assertEqual(
            crop_sharpness_at_height(np.zeros((4, 4, 3), np.uint8), 1000),
            0.0)
        self.assertFalse(quality_gate_accepts(0.0, 0.0))
        self.assertFalse(quality_gate_accepts(None, None))
        self.assertFalse(quality_gate_accepts(float("nan"), float("nan")))


class GateLogicTests(unittest.TestCase):
    def test_headline_saint_spring_rescued(self):
        # 388:2:31: 758x1650, full 12.37 (reject), norm@1000 34.88 (rescue).
        self.assertLess(12.37, OCR_SHARPNESS_MIN)
        self.assertGreaterEqual(34.88, OCR_SHARPNESS_NORM_MIN)
        self.assertTrue(quality_gate_accepts(12.37, 34.88))

    def test_genuinely_blurry_stays_rejected(self):
        # 398:6:49-like: full ~4.1, norm ~7.6 band buys mostly empty OCR.
        self.assertFalse(quality_gate_accepts(4.07, 7.6))

    def test_small_sharp_preserved_via_full(self):
        # 437:4:122-like: 541x1211, full 16.19, norm 18.9. A pure
        # normalized gate would drop this AKBA hit; the union keeps it.
        self.assertTrue(quality_gate_accepts(16.19, 18.9))
        self.assertGreaterEqual(16.19, OCR_SHARPNESS_MIN)

    def test_current_accepts_are_subset_of_union(self):
        # Union never regresses a current accept by construction.
        for full, normed in [(15.0, 0.0), (45.0, 12.0), (16.7, 11.1)]:
            if full >= OCR_SHARPNESS_MIN:
                self.assertTrue(quality_gate_accepts(full, normed))

    def test_minimum_crop_size_unchanged(self):
        self.assertEqual(OCR_MIN_CROP_WIDTH, 60)
        self.assertEqual(OCR_MIN_CROP_HEIGHT, 80)


if __name__ == "__main__":
    unittest.main()
