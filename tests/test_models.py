"""
Enhetstester för kärndataklasser i src/core/models.py (ToFReading och BarcodeReading).
Verifierar standardvärden, is_valid()-metod, invalid()-fabriker,
serialisering (to_dict/from_dict) och tidslogik.
"""

import time
import unittest
import tempfile
from pathlib import Path

from src.core.models import ToFReading, BarcodeReading, CameraFrame, CameraFramePair
import numpy as np
import cv2


class TestToFReading(unittest.TestCase):
    """Testsvit för ToFReading dataklassen."""

    def test_default_values(self):
        """Verifiera att standardvärden är säkra och ogiltiga vid tom initiering."""
        reading = ToFReading()
        self.assertEqual(reading.distance_mm, -1.0)
        self.assertFalse(reading.is_valid())
        self.assertEqual(reading.timestamp, 0.0)
        self.assertEqual(reading.samples, 0)
        self.assertEqual(reading.sample_method, "")
        self.assertEqual(reading.zone, "")
        self.assertEqual(reading.zone_method, "")
        self.assertEqual(reading.valid_zones, 0)
        # En tom/ogiltig läsning ska alltid räknas som utgången
        self.assertTrue(reading.is_expired())

    def test_invalid_factory(self):
        """Verifiera att ToFReading.invalid() skapar ett säkert, ogiltigt objekt utan argument."""
        inv = ToFReading.invalid()
        self.assertEqual(inv.distance_mm, -1.0)
        self.assertFalse(inv.is_valid())
        self.assertEqual(inv.samples, 0)
        self.assertEqual(inv.sample_method, "")
        self.assertEqual(inv.zone, "")
        self.assertEqual(inv.zone_method, "")
        self.assertEqual(inv.valid_zones, 0)
        self.assertTrue(inv.is_expired())

    def test_to_dict_and_from_dict_roundtrip(self):
        """Verifiera att data bevaras 100% korrekt genom to_dict och from_dict."""
        original = ToFReading(
            distance_mm=952.456,
            timestamp=1728842400.0,
            samples=5,
            sample_method="mean",
            zone="center",
            zone_method="median",
            valid_zones=16,
            _is_valid=True
        )

        self.assertTrue(original.is_valid())

        data = original.to_dict()
        # Verifiera avrundning till 2 decimaler
        self.assertEqual(data["distance_mm"], 952.46)
        self.assertTrue(data["is_valid"])
        self.assertEqual(data["sample_method"], "mean")
        self.assertEqual(data["zone_method"], "median")

        # Återskapa från dictionaryn
        restored = ToFReading.from_dict(data)
        self.assertEqual(restored.distance_mm, 952.46)
        self.assertTrue(restored.is_valid())
        self.assertEqual(restored.timestamp, 1728842400.0)
        self.assertEqual(restored.samples, 5)
        self.assertEqual(restored.sample_method, "mean")
        self.assertEqual(restored.zone, "center")
        self.assertEqual(restored.zone_method, "median")
        self.assertEqual(restored.valid_zones, 16)

    def test_from_dict_malformed_data(self):
        """Verifiera att felaktiga typer och saknade fält aldrig kraschar koden."""
        # Test 1: Tom dict
        r1 = ToFReading.from_dict({})
        self.assertFalse(r1.is_valid())
        self.assertEqual(r1.distance_mm, -1.0)

        # Test 2: Värden är None istället för siffror
        bad_data_none = {
            "distance_mm": None,
            "is_valid": True,
            "timestamp": None,
            "samples": None
        }
        r2 = ToFReading.from_dict(bad_data_none)
        self.assertFalse(r2.is_valid())

        # Test 3: Värden är felaktiga strängar
        bad_data_str = {
            "distance_mm": "inte_en_siffra",
            "is_valid": True
        }
        r3 = ToFReading.from_dict(bad_data_str)
        self.assertFalse(r3.is_valid())

        # Test 4: Indata är inte ens en dict
        r4 = ToFReading.from_dict(None)  # type: ignore
        self.assertFalse(r4.is_valid())

    def test_is_expired_logic(self):
        """Verifiera tidslogiken med timmar."""
        now = time.time()

        # En giltig mätning gjord för 2 timmar sedan
        recent_reading = ToFReading(
            distance_mm=950.0,
            valid_zones=16,
            _is_valid=True,
            timestamp=now - (2 * 3600)
        )
        self.assertTrue(recent_reading.is_valid())
        self.assertFalse(recent_reading.is_expired(max_age_hours=24.0))
        self.assertTrue(recent_reading.is_expired(max_age_hours=1.0))

        # En mätning gjord för 10 dagar sedan (240 timmar)
        old_reading = ToFReading(
            distance_mm=950.0,
            valid_zones=16,
            _is_valid=True,
            timestamp=now - (240 * 3600)
        )
        self.assertTrue(old_reading.is_expired())


class TestBarcodeReading(unittest.TestCase):
    """Testsvit för BarcodeReading dataklassen."""

    def test_default_values(self):
        """Verifiera standardvärden för BarcodeReading."""
        b = BarcodeReading()
        self.assertEqual(b.code, "")
        self.assertFalse(b.is_valid())
        self.assertFalse(b.is_used)

    def test_invalid_factory(self):
        """Verifiera att BarcodeReading.invalid() skapar ett säkert, förbrukat tomt objekt."""
        inv = BarcodeReading.invalid()
        self.assertEqual(inv.code, "")
        self.assertFalse(inv.is_valid())
        self.assertTrue(inv.is_used)

    def test_valid_barcode(self):
        """Verifiera att en kod med tecken och is_used=False är giltig."""
        b = BarcodeReading(code="7312345678901", is_used=False)
        self.assertTrue(b.is_valid())

        # Om koden markeras som använd ska den inte längre vara giltig
        b.is_used = True
        self.assertFalse(b.is_valid())

    def test_to_dict_and_from_dict(self):
        """Verifiera serialisering för BarcodeReading."""
        original = BarcodeReading(
            code="ABC-12345",
            timestamp=1728842400.0,
            is_used=False
        )
        data = original.to_dict()
        self.assertEqual(data["code"], "ABC-12345")
        self.assertTrue(data["is_valid"])

        restored = BarcodeReading.from_dict(data)
        self.assertEqual(restored.code, "ABC-12345")
        self.assertTrue(restored.is_valid())

    def test_from_dict_malformed(self):
        """Verifiera att felaktig dict ger BarcodeReading.invalid()."""
        r = BarcodeReading.from_dict(None)  # type: ignore
        self.assertFalse(r.is_valid())
        self.assertEqual(r.code, "")


class TestCameraFrame(unittest.TestCase):
    """Testsvit för CameraFrame dataklassen."""

    def test_default_values(self):
        """Verifiera att en tom CameraFrame har säkra standardvärden och är ogiltig."""
        frame = CameraFrame()
        self.assertIsNone(frame.image)
        self.assertEqual(frame.camera_id, "")
        self.assertFalse(frame.is_valid())
        self.assertEqual(frame.resolution, (0, 0))
        self.assertEqual(frame.exposure_us, 0)
        self.assertEqual(frame.gain, 0.0)

    def test_invalid_factory(self):
        """Verifiera att invalid() skapar en ogiltig instans utan argument."""
        inv = CameraFrame.invalid()
        self.assertIsNone(inv.image)
        self.assertEqual(inv.camera_id, "")
        self.assertFalse(inv.is_valid())
        self.assertGreater(inv.timestamp, 0.0)

    def test_valid_frame(self):
        """Verifiera att en komplett instans med bildmatris är giltig."""
        dummy_img = np.zeros((1232, 1640, 3), dtype=np.uint8)
        frame = CameraFrame(
            image=dummy_img,
            camera_id="CAM0",
            timestamp=time.time(),
            resolution=(1640, 1232),
            exposure_us=25000,
            gain=4.0,
            samples=1,
            sample_method="none",
            _is_valid=True,
        )
        self.assertTrue(frame.is_valid())
        self.assertEqual(frame.camera_id, "CAM0")
        self.assertEqual(frame.exposure_us, 25000)

    def test_to_dict_and_from_dict(self):
        """Verifiera serialisering av metadata till/från dict."""
        dummy_img = np.zeros((480, 640), dtype=np.uint8)
        original = CameraFrame(
            image=dummy_img,
            camera_id="CAM1",
            timestamp=1728842400.0,
            resolution=(640, 480),
            exposure_us=30000,
            gain=2.5,
            samples=3,
            sample_method="median",
            _is_valid=True,
        )
        data = original.to_dict()
        self.assertEqual(data["camera_id"], "CAM1")
        self.assertTrue(data["is_valid"])
        self.assertTrue(data["has_image"])
        self.assertEqual(data["image_shape"], [480, 640])
        self.assertEqual(data["image_dtype"], "uint8")

        # Återskapa från dict
        restored = CameraFrame.from_dict(data, image=dummy_img)
        self.assertTrue(restored.is_valid())
        self.assertEqual(restored.camera_id, "CAM1")
        self.assertEqual(restored.exposure_us, 30000)
        self.assertEqual(restored.sample_method, "median")

    def test_from_dict_malformed(self):
        """Verifiera att felaktiga indata till from_dict ger CameraFrame.invalid()."""
        inv = CameraFrame.from_dict(None)  # type: ignore
        self.assertFalse(inv.is_valid())
        self.assertIsNone(inv.image)

    def test_str_representation(self):
        """Verifiera att __str__ returnerar en läsbar sträng."""
        frame = CameraFrame.invalid()
        self.assertIn("CameraFrame", str(frame))

    def test_to_jpeg_valid_image(self):
        """Verifiera att to_jpeg sparar giltiga JPEG-filer med rätt namn och innehåll."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            dummy_img = np.full((100, 150, 3), 128, dtype=np.uint8)
            frame = CameraFrame(
                image=dummy_img,
                camera_id="CAM0",
                timestamp=1728842400.0,
                resolution=(150, 100),
                exposure_us=25000,
                gain=4.0,
                _is_valid=True,
            )

            # Test 1: Spara med automatisk namngivning i katalog
            saved = frame.to_jpeg(path=tmp_path)
            self.assertIsNotNone(saved)
            self.assertTrue(saved.exists())
            self.assertEqual(saved.name, "1728842400_CAM0.jpg")

            # Kontrollera att filen går att läsa och har rätt dimensioner
            read_img = cv2.imread(str(saved))
            self.assertIsNotNone(read_img)
            self.assertEqual(read_img.shape, (100, 150, 3))

            # Test 2: Spara med explicit filnamn
            explicit = frame.to_jpeg(path=tmp_path / "custom_test.jpg")
            self.assertIsNotNone(explicit)
            self.assertTrue(explicit.exists())
            self.assertEqual(explicit.name, "custom_test.jpg")

    def test_to_jpeg_invalid_frame(self):
        """Verifiera att to_jpeg returnerar None för en ogiltig bildram."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            inv = CameraFrame.invalid()
            self.assertIsNone(inv.to_jpeg(path=tmp_dir))


class TestCameraFramePair(unittest.TestCase):
    """Testsvit för CameraFramePair dataklassen."""

    def test_default_values(self):
        """Verifiera att standard stereopar har ogiltiga bildramar."""
        pair = CameraFramePair()
        self.assertFalse(pair.is_valid())
        self.assertFalse(pair.left.is_valid())
        self.assertFalse(pair.right.is_valid())
        self.assertEqual(pair.timestamp, 0.0)

    def test_invalid_factory(self):
        """Verifiera att invalid() skapar ett säkert ogiltigt stereopar."""
        inv = CameraFramePair.invalid()
        self.assertFalse(inv.is_valid())
        self.assertGreater(inv.timestamp, 0.0)

    def test_valid_pair(self):
        """Verifiera att stereoparet är giltigt om och endast om BÅDA bilderna är giltiga."""
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        f_left = CameraFrame(image=img, camera_id="CAM1", _is_valid=True)
        f_right = CameraFrame(image=img, camera_id="CAM0", _is_valid=True)

        pair = CameraFramePair(left=f_left, right=f_right, timestamp=1728842400.0)
        self.assertTrue(pair.is_valid())

        # Om bara ena är giltig ska hela paret vara ogiltigt
        half_pair = CameraFramePair(left=f_left, right=CameraFrame.invalid())
        self.assertFalse(half_pair.is_valid())

    def test_to_dict_and_from_dict(self):
        """Verifiera roundtrip serialisering för CameraFramePair."""
        img = np.full((20, 30, 3), 50, dtype=np.uint8)
        f_left = CameraFrame(image=img, camera_id="CAM1", resolution=(30, 20), _is_valid=True)
        f_right = CameraFrame(image=img, camera_id="CAM0", resolution=(30, 20), _is_valid=True)

        original = CameraFramePair(left=f_left, right=f_right, timestamp=1728842400.0)
        data = original.to_dict()
        self.assertTrue(data["is_valid"])
        self.assertEqual(data["timestamp"], 1728842400.0)

        restored = CameraFramePair.from_dict(data, left_image=img, right_image=img)
        self.assertTrue(restored.is_valid())
        self.assertEqual(restored.left.camera_id, "CAM1")
        self.assertEqual(restored.right.camera_id, "CAM0")

    def test_from_dict_malformed(self):
        """Verifiera att felaktiga indata till from_dict ger ogiltigt stereopar."""
        inv = CameraFramePair.from_dict(None)  # type: ignore
        self.assertFalse(inv.is_valid())

    def test_to_jpeg(self):
        """Verifiera att to_jpeg sparar båda bilderna i paret."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            img = np.full((20, 20, 3), 100, dtype=np.uint8)
            f_left = CameraFrame(image=img, camera_id="CAM1", timestamp=1000.0, _is_valid=True)
            f_right = CameraFrame(image=img, camera_id="CAM0", timestamp=1000.0, _is_valid=True)
            pair = CameraFramePair(left=f_left, right=f_right, timestamp=1000.0)

            left_path, right_path = pair.to_jpeg(path=tmp_path)
            self.assertIsNotNone(left_path)
            self.assertIsNotNone(right_path)
            self.assertTrue(left_path.exists())
            self.assertTrue(right_path.exists())
            self.assertIn("CAM1", left_path.name)
            self.assertIn("CAM0", right_path.name)

    def test_str_representation(self):
        """Verifiera att __str__ returnerar en läsbar sträng."""
        pair = CameraFramePair.invalid()
        self.assertIn("CameraFramePair", str(pair))




if __name__ == "__main__":
    unittest.main()
