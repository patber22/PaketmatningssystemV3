"""
Enhetstester för kärndataklasser i src/core/models.py (ToFReading och BarcodeReading).
Verifierar standardvärden, is_valid()-metod, invalid()-fabriker,
serialisering (to_dict/from_dict) och tidslogik.
"""

import time
import unittest
import tempfile
from pathlib import Path

from src.core.models import (
    ToFReading,
    BarcodeReading,
    CameraFrame,
    CameraFramePair,
    WeightReading,
    PhotometricSweep,
    VisionCapture,
    StereoDepthResult,
    PhotometricResult,
)
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


class TestWeightReading(unittest.TestCase):
    """Testsvit för WeightReading dataklassen."""

    def test_default_values(self):
        """Verifiera standardvärden för tom WeightReading."""
        w = WeightReading()
        self.assertEqual(w.weight_g, -1.0)
        self.assertFalse(w.is_valid())
        self.assertFalse(w.is_stable)
        self.assertEqual(w.timestamp, 0.0)
        self.assertEqual(w.samples, 1)
        self.assertEqual(w.raw_value, 0)

    def test_invalid_factory(self):
        """Verifiera att invalid() skapar en ogiltig instans utan argument."""
        inv = WeightReading.invalid()
        self.assertEqual(inv.weight_g, -1.0)
        self.assertFalse(inv.is_valid())
        self.assertFalse(inv.is_stable)
        self.assertGreater(inv.timestamp, 0.0)

    def test_valid_readings(self):
        """Verifiera giltighetskriterier (0 gram och positiva vikter)."""
        # 0.0g är en giltig tarerad mätning
        w_zero = WeightReading(weight_g=0.0, _is_valid=True)
        self.assertTrue(w_zero.is_valid())

        # Positiv vikt
        w_pkg = WeightReading(weight_g=1250.5, is_stable=True, _is_valid=True)
        self.assertTrue(w_pkg.is_valid())
        self.assertTrue(w_pkg.is_stable)

        # Negativ vikt ska räknas som ogiltig
        w_neg = WeightReading(weight_g=-5.0, _is_valid=True)
        self.assertFalse(w_neg.is_valid())

    def test_to_dict_and_from_dict(self):
        """Verifiera serialisering av WeightReading till och från dict."""
        original = WeightReading(
            weight_g=2450.78,
            is_stable=True,
            timestamp=1728842400.0,
            samples=5,
            raw_value=123456,
            _is_valid=True,
        )
        data = original.to_dict()
        self.assertEqual(data["weight_g"], 2450.8)
        self.assertTrue(data["is_stable"])
        self.assertTrue(data["is_valid"])
        self.assertEqual(data["raw_value"], 123456)
        self.assertEqual(data["samples"], 5)

        restored = WeightReading.from_dict(data)
        self.assertTrue(restored.is_valid())
        self.assertEqual(restored.weight_g, 2450.8)
        self.assertTrue(restored.is_stable)
        self.assertEqual(restored.raw_value, 123456)

    def test_from_dict_malformed(self):
        """Verifiera att felaktiga indata till from_dict ger ogiltigt objekt."""
        inv = WeightReading.from_dict(None)  # type: ignore
        self.assertFalse(inv.is_valid())
        self.assertEqual(inv.weight_g, -1.0)

    def test_str_representation(self):
        """Verifiera att __str__ returnerar en läsbar sträng."""
        w = WeightReading.invalid()
        self.assertIn("WeightReading", str(w))


class TestPhotometricSweep(unittest.TestCase):
    """Testsvit för PhotometricSweep dataklassen."""

    def _create_valid_pair(self, ts=1000.0) -> CameraFramePair:
        img = np.full((10, 10, 3), 128, dtype=np.uint8)
        f_left = CameraFrame(image=img, camera_id="CAM1", timestamp=ts, _is_valid=True)
        f_right = CameraFrame(image=img, camera_id="CAM0", timestamp=ts, _is_valid=True)
        return CameraFramePair(left=f_left, right=f_right, timestamp=ts)

    def test_default_values(self):
        """Verifiera att standardvärden är ogiltiga."""
        sweep = PhotometricSweep()
        self.assertFalse(sweep.is_valid())
        self.assertEqual(sweep.timestamp, 0.0)
        self.assertFalse(sweep.ch1.is_valid())
        self.assertFalse(sweep.ch4.is_valid())

    def test_invalid_factory(self):
        """Verifiera att invalid() fabriksmetoden returnerar ogiltigt svep."""
        inv = PhotometricSweep.invalid()
        self.assertFalse(inv.is_valid())
        self.assertGreater(inv.timestamp, 0.0)

    def test_valid_sweep(self):
        """Verifiera att is_valid() returnerar True endast om ALLA 4 kanaler är giltiga."""
        pair1 = self._create_valid_pair()
        pair2 = self._create_valid_pair()
        pair3 = self._create_valid_pair()
        pair4 = self._create_valid_pair()

        # Alla 4 giltiga -> True
        sweep = PhotometricSweep(ch1=pair1, ch2=pair2, ch3=pair3, ch4=pair4, timestamp=1000.0)
        self.assertTrue(sweep.is_valid())

        # En kanal ogiltig -> False
        sweep_incomplete = PhotometricSweep(ch1=pair1, ch2=pair2, ch3=pair3, ch4=CameraFramePair.invalid())
        self.assertFalse(sweep_incomplete.is_valid())

    def test_get_channel(self):
        """Verifiera åtkomst till kanaler via get_channel()."""
        pair1 = self._create_valid_pair(1.0)
        pair2 = self._create_valid_pair(2.0)
        pair3 = self._create_valid_pair(3.0)
        pair4 = self._create_valid_pair(4.0)
        sweep = PhotometricSweep(ch1=pair1, ch2=pair2, ch3=pair3, ch4=pair4)

        self.assertEqual(sweep.get_channel(1).timestamp, 1.0)
        self.assertEqual(sweep.get_channel(2).timestamp, 2.0)
        self.assertEqual(sweep.get_channel(3).timestamp, 3.0)
        self.assertEqual(sweep.get_channel(4).timestamp, 4.0)

        with self.assertRaises(ValueError):
            sweep.get_channel(5)

    def test_to_dict_and_from_dict(self):
        """Verifiera serialisering av metadata utan bildmatriser."""
        pair1 = self._create_valid_pair()
        pair2 = self._create_valid_pair()
        pair3 = self._create_valid_pair()
        pair4 = self._create_valid_pair()
        original = PhotometricSweep(ch1=pair1, ch2=pair2, ch3=pair3, ch4=pair4, timestamp=1700000000.0)

        data = original.to_dict()
        self.assertTrue(data["is_valid"])
        self.assertEqual(data["timestamp"], 1700000000.0)
        self.assertIn("ch1", data)
        self.assertIn("ch4", data)

        restored = PhotometricSweep.from_dict(data)
        self.assertFalse(restored.is_valid())  # Inga råa bildmatriser följer med i metadata-dict
        self.assertEqual(restored.timestamp, 1700000000.0)

    def test_to_jpeg(self):
        """Verifiera att to_jpeg sparar alla 4 kanalers bildpar till disk."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            pair1 = self._create_valid_pair(100.0)
            pair2 = self._create_valid_pair(200.0)
            pair3 = self._create_valid_pair(300.0)
            pair4 = self._create_valid_pair(400.0)
            sweep = PhotometricSweep(ch1=pair1, ch2=pair2, ch3=pair3, ch4=pair4)

            saved = sweep.to_jpeg(path=tmp_path)
            self.assertEqual(len(saved), 4)
            for ch in [1, 2, 3, 4]:
                left_p, right_p = saved[ch]
                self.assertIsNotNone(left_p)
                self.assertIsNotNone(right_p)
                self.assertTrue(left_p.exists())
                self.assertTrue(right_p.exists())

    def test_str_representation(self):
        """Verifiera __str__ representation."""
        sweep = PhotometricSweep.invalid()
        self.assertIn("PhotometricSweep", str(sweep))


class TestVisionCapture(unittest.TestCase):
    """Testsvit för VisionCapture dataklassen."""

    def _create_valid_pair(self, ts=1000.0) -> CameraFramePair:
        img = np.full((10, 10, 3), 128, dtype=np.uint8)
        f_left = CameraFrame(image=img, camera_id="CAM1", timestamp=ts, _is_valid=True)
        f_right = CameraFrame(image=img, camera_id="CAM0", timestamp=ts, _is_valid=True)
        return CameraFramePair(left=f_left, right=f_right, timestamp=ts)

    def test_default_values(self):
        """Verifiera att standardvärden är ogiltiga."""
        cap = VisionCapture()
        self.assertFalse(cap.is_valid())
        self.assertEqual(cap.timestamp, 0.0)

    def test_invalid_factory(self):
        """Verifiera att invalid() fabriksmetoden returnerar ogiltigt objekt."""
        inv = VisionCapture.invalid()
        self.assertFalse(inv.is_valid())
        self.assertGreater(inv.timestamp, 0.0)

    def test_valid_capture(self):
        """Verifiera att is_valid() kräver både giltigt stereopar och fotometriskt svep."""
        stereo = self._create_valid_pair(100.0)
        p1 = self._create_valid_pair(101.0)
        p2 = self._create_valid_pair(102.0)
        p3 = self._create_valid_pair(103.0)
        p4 = self._create_valid_pair(104.0)
        photo = PhotometricSweep(ch1=p1, ch2=p2, ch3=p3, ch4=p4, timestamp=105.0)

        # Båda giltiga -> True
        cap = VisionCapture(stereo=stereo, photometric=photo, timestamp=100.0)
        self.assertTrue(cap.is_valid())

        # Stereopar ogiltigt -> False
        cap_bad_stereo = VisionCapture(stereo=CameraFramePair.invalid(), photometric=photo)
        self.assertFalse(cap_bad_stereo.is_valid())

        # Svep ogiltigt -> False
        cap_bad_photo = VisionCapture(stereo=stereo, photometric=PhotometricSweep.invalid())
        self.assertFalse(cap_bad_photo.is_valid())

    def test_to_dict_and_from_dict(self):
        """Verifiera serialisering av VisionCapture metadata."""
        stereo = self._create_valid_pair(100.0)
        photo = PhotometricSweep(
            ch1=self._create_valid_pair(101.0),
            ch2=self._create_valid_pair(102.0),
            ch3=self._create_valid_pair(103.0),
            ch4=self._create_valid_pair(104.0),
            timestamp=105.0,
        )
        original = VisionCapture(stereo=stereo, photometric=photo, timestamp=1700000000.0)

        data = original.to_dict()
        self.assertTrue(data["is_valid"])
        self.assertEqual(data["timestamp"], 1700000000.0)
        self.assertIn("stereo", data)
        self.assertIn("photometric", data)

        restored = VisionCapture.from_dict(data)
        self.assertEqual(restored.timestamp, 1700000000.0)

    def test_to_jpeg(self):
        """Verifiera att to_jpeg sparar både stereo och fotometriska bilder."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            stereo = self._create_valid_pair(100.0)
            photo = PhotometricSweep(
                ch1=self._create_valid_pair(101.0),
                ch2=self._create_valid_pair(102.0),
                ch3=self._create_valid_pair(103.0),
                ch4=self._create_valid_pair(104.0),
                timestamp=105.0,
            )
            cap = VisionCapture(stereo=stereo, photometric=photo, timestamp=100.0)

            saved = cap.to_jpeg(path=tmp_path)
            self.assertIn("stereo", saved)
            self.assertIn("photometric", saved)
            left_s, right_s = saved["stereo"]
            self.assertTrue(left_s.exists())
            self.assertTrue(right_s.exists())

    def test_str_representation(self):
        """Verifiera __str__ representation."""
        cap = VisionCapture.invalid()
        self.assertIn("VisionCapture", str(cap))


class TestStereoDepthResult(unittest.TestCase):
    """Testsvit för StereoDepthResult dataklassen."""

    def test_default_values(self):
        """Verifiera att standardvärden är säkra och ogiltiga vid tom initiering."""
        res = StereoDepthResult()
        self.assertFalse(res.is_valid())
        self.assertIsNone(res.disparity)
        self.assertIsNone(res.point_cloud)
        self.assertIsNone(res.rectified_left)
        self.assertEqual(res.timestamp, 0.0)
        self.assertEqual(res.execution_time_ms, 0.0)

    def test_invalid_factory(self):
        """Verifiera att StereoDepthResult.invalid() skapar ett ogiltigt objekt."""
        inv = StereoDepthResult.invalid()
        self.assertFalse(inv.is_valid())
        self.assertIsNone(inv.disparity)
        data = inv.to_dict()
        self.assertFalse(data["is_valid"])

    def test_valid_operations(self):
        """Verifiera maskning, punktmolnsfiltrering och to_dict metadata."""
        h, w = 10, 10
        # Skapa disparitet där 70 pixlar är giltiga (>0) och 30 är ogiltiga (-1)
        disp = np.full((h, w), -1.0, dtype=np.float32)
        disp[:7, :] = 25.5

        # Skapa punktmoln där giltiga pixlar har Z mellan 600 och 700 mm
        pc = np.zeros((h, w, 3), dtype=np.float32)
        pc[:7, :, 0] = np.arange(w)
        pc[:7, :, 1] = np.arange(7)[:, None]
        pc[:7, :, 2] = 650.0

        rect = np.full((h, w, 3), 128, dtype=np.uint8)

        res = StereoDepthResult(
            disparity=disp,
            point_cloud=pc,
            rectified_left=rect,
            timestamp=12345.0,
            execution_time_ms=35.5,
            _is_valid=True,
        )

        self.assertTrue(res.is_valid())

        # Testa giltighetsmask
        mask = res.get_valid_mask()
        self.assertEqual(mask.shape, (h, w))
        self.assertEqual(int(np.count_nonzero(mask)), 70)

        # Testa get_valid_points
        valid_pts = res.get_valid_points()
        self.assertEqual(valid_pts.shape, (70, 3))
        self.assertEqual(valid_pts[0, 2], 650.0)

        # Testa to_dict
        d = res.to_dict()
        self.assertTrue(d["is_valid"])
        self.assertEqual(d["resolution"], [w, h])
        self.assertEqual(d["valid_pixels_count"], 70)
        self.assertEqual(d["valid_pixels_percent"], 70.0)
        self.assertEqual(d["depth_stats"]["min_mm"], 650.0)
        self.assertEqual(d["depth_stats"]["max_mm"], 650.0)
        self.assertEqual(d["depth_stats"]["mean_mm"], 650.0)

    def test_to_jpeg(self):
        """Verifiera att to_jpeg sparar rektifierad bild och färgkodad disparitet."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            disp = np.full((20, 20), 15.0, dtype=np.float32)
            disp[0, 0] = -1.0  # en ogiltig pixel
            pc = np.full((20, 20, 3), 600.0, dtype=np.float32)
            rect = np.full((20, 20, 3), 100, dtype=np.uint8)

            res = StereoDepthResult(
                disparity=disp,
                point_cloud=pc,
                rectified_left=rect,
                _is_valid=True,
            )

            saved = res.to_jpeg(tmp_path)
            self.assertIsNotNone(saved["rectified_left"])
            self.assertIsNotNone(saved["disparity"])
            self.assertTrue(saved["rectified_left"].exists())
            self.assertTrue(saved["disparity"].exists())

    def test_str_representation(self):
        """Verifiera __str__ representation."""
        res = StereoDepthResult.invalid()
        self.assertIn("StereoDepthResult", str(res))


class TestPhotometricResult(unittest.TestCase):
    """Testsvit för PhotometricResult dataklassen."""

    def test_default_values(self):
        """Verifiera standardvärden vid tom initiering."""
        res = PhotometricResult()
        self.assertFalse(res.is_valid())
        self.assertIsNone(res.normals)
        self.assertIsNone(res.albedo)
        self.assertEqual(res.camera_id, "")
        self.assertEqual(res.execution_time_ms, 0.0)

    def test_invalid_factory(self):
        """Verifiera PhotometricResult.invalid() fabriken."""
        inv = PhotometricResult.invalid()
        self.assertFalse(inv.is_valid())
        data = inv.to_dict()
        self.assertFalse(data["is_valid"])

    def test_valid_operations(self):
        """Verifiera maskning, metadata och serialisering på giltigt resultat."""
        h, w = 10, 10
        normals = np.zeros((h, w, 3), dtype=np.float32)
        normals[:, :, 2] = 1.0  # Pekar rakt upp
        albedo = np.full((h, w), 0.8, dtype=np.float32)
        albedo[:2, :] = 0.0     # 2 rader ogiltiga

        res = PhotometricResult(
            normals=normals,
            albedo=albedo,
            camera_id="left_cam1",
            timestamp=123.0,
            execution_time_ms=14.5,
            _is_valid=True,
        )

        self.assertTrue(res.is_valid())
        mask = res.get_valid_mask()
        self.assertEqual(np.count_nonzero(mask), 80)

        d = res.to_dict()
        self.assertTrue(d["is_valid"])
        self.assertEqual(d["camera_id"], "left_cam1")
        self.assertEqual(d["valid_pixels_count"], 80)
        self.assertEqual(d["valid_pixels_percent"], 80.0)
        self.assertEqual(d["albedo_stats"]["mean"], 0.8)

    def test_to_jpeg(self):
        """Verifiera export av normalkarta (RGB) och albedo."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            normals = np.zeros((10, 10, 3), dtype=np.float32)
            normals[:, :, 2] = 1.0
            albedo = np.full((10, 10), 0.5, dtype=np.float32)

            res = PhotometricResult(
                normals=normals,
                albedo=albedo,
                camera_id="left_cam1",
                _is_valid=True,
            )

            saved = res.to_jpeg(tmp_path)
            self.assertIsNotNone(saved["normals_rgb"])
            self.assertIsNotNone(saved["albedo"])
            self.assertTrue(saved["normals_rgb"].exists())
            self.assertTrue(saved["albedo"].exists())

    def test_str_representation(self):
        """Verifiera __str__ representation."""
        res = PhotometricResult.invalid()
        self.assertIn("PhotometricResult", str(res))


if __name__ == "__main__":
    unittest.main()
