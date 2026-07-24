#!/usr/bin/env python3
"""Camera-free validation for the printable AprilTag generator."""

from __future__ import annotations

import importlib.util
import re
import sys
import tempfile
import unittest
from pathlib import Path

sys.dont_write_bytecode = True

import cv2
import numpy as np
from PIL import Image


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "generate_apriltag_prints.py"
)
SPEC = importlib.util.spec_from_file_location("generate_apriltag_prints", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
generator = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = generator
SPEC.loader.exec_module(generator)


def detect_ids(image: np.ndarray, dictionary):
    if hasattr(cv2.aruco, "ArucoDetector"):
        parameters = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(dictionary, parameters)
        corners, ids, _ = detector.detectMarkers(image)
    else:
        parameters = (
            cv2.aruco.DetectorParameters_create()
            if hasattr(cv2.aruco, "DetectorParameters_create")
            else cv2.aruco.DetectorParameters()
        )
        corners, ids, _ = cv2.aruco.detectMarkers(
            image, dictionary, parameters=parameters
        )
    return corners, ids


class GenerateAprilTagPrintsTest(unittest.TestCase):
    def test_default_tags_are_detectable_and_physically_sized(self):
        # A lower DPI keeps the test fast; all dimensions still use the same
        # physical-size calculations as the 600-DPI default.
        dpi = 100
        with tempfile.TemporaryDirectory() as temporary_directory:
            result = generator.generate_prints(
                dpi=dpi, output_dir=Path(temporary_directory)
            )

            self.assertEqual(result.dictionary_name, "APRILTAG_36h11")
            self.assertEqual([page.tag_id for page in result.pages], [0, 1])
            self.assertEqual(len(list(Path(temporary_directory).glob("*.png"))), 2)
            self.assertEqual(len(list(Path(temporary_directory).glob("*.pdf"))), 1)
            self.assertTrue(result.pdf_path.is_file())

            _, dictionary = generator.get_dictionary(result.dictionary_name)
            expected_page_size = (
                round(210.0 * dpi / generator.MM_PER_INCH),
                round(297.0 * dpi / generator.MM_PER_INCH),
            )
            expected_marker_size = round(12.0 * dpi / generator.CM_PER_INCH)
            expected_modules = int(dictionary.markerSize) + 2
            expected_quiet = round(expected_marker_size / expected_modules)

            for page in result.pages:
                self.assertEqual(page.marker_size_cm, 12.0)
                self.assertEqual(page.page_size_px, expected_page_size)
                self.assertEqual(page.module_count, expected_modules)
                self.assertEqual(page.quiet_zone_px, expected_quiet)

                with Image.open(page.png_path) as png:
                    self.assertEqual(png.size, expected_page_size)
                    png_dpi = png.info.get("dpi")
                    self.assertIsNotNone(png_dpi)
                    self.assertAlmostEqual(png_dpi[0], dpi, delta=0.1)
                    self.assertAlmostEqual(png_dpi[1], dpi, delta=0.1)
                    image = np.asarray(png.convert("L"))

                left, top, right, bottom = page.marker_box
                self.assertEqual(right - left, expected_marker_size)
                self.assertEqual(bottom - top, expected_marker_size)

                # The marker's complete outer edge is black.
                self.assertEqual(int(image[top, left:right].max()), 0)
                self.assertEqual(int(image[bottom - 1, left:right].max()), 0)
                self.assertEqual(int(image[top:bottom, left].max()), 0)
                self.assertEqual(int(image[top:bottom, right - 1].max()), 0)

                # Exactly one module is reserved as uninterrupted white space
                # on every side; cut guides are drawn outside this square.
                quiet = page.quiet_zone_px
                self.assertEqual(int(image[top:bottom, left - quiet:left].min()), 255)
                self.assertEqual(int(image[top:bottom, right:right + quiet].min()), 255)
                self.assertEqual(int(image[top - quiet:top, left:right].min()), 255)
                self.assertEqual(int(image[bottom:bottom + quiet, left:right].min()), 255)

                corners, detected_ids = detect_ids(image, dictionary)
                self.assertIsNotNone(detected_ids)
                self.assertEqual(detected_ids.reshape(-1).tolist(), [page.tag_id])
                detected = np.asarray(corners[0]).reshape(4, 2)
                side_lengths = [
                    np.linalg.norm(detected[(index + 1) % 4] - detected[index])
                    for index in range(4)
                ]
                self.assertAlmostEqual(
                    float(np.mean(side_lengths)), expected_marker_size - 1,
                    delta=3.0,
                )

            # Pillow leaves page dictionaries uncompressed, so this checks the
            # combined output without adding a PDF-reader dependency.
            pdf_data = result.pdf_path.read_bytes()
            page_objects = re.findall(rb"/Type\s*/Page\b", pdf_data)
            self.assertEqual(len(page_objects), 2)
            media_boxes = re.findall(
                rb"/MediaBox\s*\[\s*0\s+0\s+([0-9.]+)\s+([0-9.]+)\s*\]",
                pdf_data,
            )
            self.assertEqual(len(media_boxes), 2)
            expected_points = tuple(
                pixels * 72.0 / dpi for pixels in expected_page_size
            )
            for width, height in media_boxes:
                self.assertAlmostEqual(
                    float(width), expected_points[0], delta=0.02
                )
                self.assertAlmostEqual(
                    float(height), expected_points[1], delta=0.02
                )

    def test_cli_accepts_per_id_sizes_and_letter_paper(self):
        parser = generator.build_parser()
        args = parser.parse_args(
            [
                "--dictionary", "APRILTAG_25h9",
                "--ids", "3", "4",
                "--sizes-cm", "8.5", "9.5",
                "--paper", "Letter",
                "--dpi", "300",
                "--output-dir", "/tmp/apriltag-test",
            ]
        )
        self.assertEqual(args.dictionary, "APRILTAG_25h9")
        self.assertEqual(args.ids, [3, 4])
        self.assertEqual(args.sizes_cm, [8.5, 9.5])
        self.assertEqual(args.paper, "LETTER")
        self.assertEqual(args.dpi, 300)


if __name__ == "__main__":
    unittest.main()
