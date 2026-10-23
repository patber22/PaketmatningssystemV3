"""
Enhetstester för Camera-klassen (src/hardware/camera.py).

Verifierar camera_id validering, koppling mot config, frånkopplat beteende,
mockad bildtagning (samples=1 och multi-sample median/mean), exponeringsändring,
context manager och resursfrigöring utan krav på fysisk Raspberry Pi.
"""

import unittest
from unittest.mock import MagicMock
import numpy as np

from src.hardware.camera import Camera
from src.core.config import CAMERAS
from src.core.models import CameraFrame


class TestCamera(unittest.TestCase):
    """Testsvit för NoIR Camera-drivrutinen."""

    def test_camera_id_required(self):
        """Verifiera att konstruktorn kräver camera_id."""
        with self.assertRaises(TypeError):
            Camera()  # type: ignore

    def test_invalid_camera_id_raises_value_error(self):
        """Verifiera att ogiltigt camera_id kastar ValueError."""
        for invalid_id in ["CAM99", "camera", "CSI0", "", "CAM2"]:
            with self.assertRaises(ValueError):
                Camera(camera_id=invalid_id, verbose=False, auto_connect=False)

    def test_valid_attributes_match_config(self):
        """Verifiera att CAM0 och CAM1 hämtar rätt port och beskrivning från config.py."""
        for cid in ["CAM0", "CAM1"]:
            cam = Camera(camera_id=cid, verbose=False, auto_connect=False)
            expected = CAMERAS[cid]
            self.assertEqual(cam.camera_id, cid)
            self.assertEqual(cam.port, expected["port"])
            self.assertEqual(cam.description, expected["description"])
            self.assertEqual(cam.exposure_us, 25000)
            self.assertEqual(cam.gain, 4.0)
            self.assertFalse(cam.is_connected)

    def test_disconnected_behavior(self):
        """Verifiera att frånkopplad kamera returnerar invalid CameraFrame och inte kraschar."""
        cam = Camera(camera_id="CAM0", verbose=False, auto_connect=False)
        self.assertFalse(cam.is_connected)

        # set_exposure ska returnera False
        self.assertFalse(cam.set_exposure(30000, 2.0))

        # capture ska returnera CameraFrame.invalid()
        frame = cam.capture()
        self.assertIsInstance(frame, CameraFrame)
        self.assertFalse(frame.is_valid())

    def test_mocked_capture_single_sample(self):
        """Verifiera bildtagning av 1 sample via mockad Picamera2."""
        cam = Camera(camera_id="CAM0", verbose=False, auto_connect=False)
        mock_picam = MagicMock()
        cam._picam2 = mock_picam
        cam.is_connected = True

        dummy_img = np.zeros((1232, 1640, 3), dtype=np.uint8)
        mock_picam.capture_array.return_value = dummy_img

        frame = cam.capture(samples=1)
        self.assertTrue(frame.is_valid())
        self.assertEqual(frame.camera_id, "CAM0")
        self.assertEqual(frame.samples, 1)
        self.assertEqual(frame.sample_method, "none")
        self.assertEqual(frame.image.shape, (1232, 1640, 3))
        mock_picam.capture_array.assert_called_once_with("main")

    def test_mocked_capture_multi_sample_median(self):
        """Verifiera att multi-sampling med median beräknar rätt pixelvärden."""
        cam = Camera(camera_id="CAM1", verbose=False, auto_connect=False)
        mock_picam = MagicMock()
        cam._picam2 = mock_picam
        cam.is_connected = True

        # Tre bilder med olika intensiteter: 10, 30, 90 (median ska bli 30)
        img1 = np.full((10, 10, 3), 10, dtype=np.uint8)
        img2 = np.full((10, 10, 3), 30, dtype=np.uint8)
        img3 = np.full((10, 10, 3), 90, dtype=np.uint8)
        mock_picam.capture_array.side_effect = [img1, img2, img3]

        frame = cam.capture(samples=3, method="median")
        self.assertTrue(frame.is_valid())
        self.assertEqual(frame.camera_id, "CAM1")
        self.assertEqual(frame.samples, 3)
        self.assertEqual(frame.sample_method, "median")
        self.assertEqual(frame.image[0, 0, 0], 30)
        self.assertEqual(mock_picam.capture_array.call_count, 3)

    def test_mocked_set_exposure(self):
        """Verifiera att set_exposure uppdaterar kontroller på aktiv kamera."""
        cam = Camera(camera_id="CAM0", verbose=False, auto_connect=False)
        mock_picam = MagicMock()
        cam._picam2 = mock_picam
        cam.is_connected = True

        self.assertTrue(cam.set_exposure(exposure_us=35000, gain=6.0))
        self.assertEqual(cam.exposure_us, 35000)
        self.assertEqual(cam.gain, 6.0)
        mock_picam.set_controls.assert_called_once_with({
            "ExposureTime": 35000,
            "AnalogueGain": 6.0,
            "AeEnable": False,
        })

    def test_context_manager_cleanup(self):
        """Verifiera att context manager garanterat stänger kameran vid avslut."""
        mock_picam = MagicMock()
        with Camera(camera_id="CAM0", verbose=False, auto_connect=False) as cam:
            cam._picam2 = mock_picam
            cam.is_connected = True

        self.assertFalse(cam.is_connected)
        self.assertIsNone(cam._picam2)
        mock_picam.stop.assert_called_once()
        mock_picam.close.assert_called_once()

    def test_repr(self):
        """Verifiera att __repr__ formaterar sträng korrekt."""
        cam = Camera(camera_id="CAM0", verbose=False, auto_connect=False)
        r = repr(cam)
        self.assertIn("Camera", r)
        self.assertIn("CAM0", r)


if __name__ == "__main__":
    unittest.main()
