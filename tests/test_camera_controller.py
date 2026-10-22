"""
Enhetstester för CameraController-fasaden (src/hardware/camera_controller.py).

Verifierar initiering av båda kamerasensorerna (CAM0 Höger & CAM1 Vänster),
validering av camera_id, frånkopplat beteende, gemensam exponeringsstyrning,
mockad bildtagning (capture_single och capture_both till CameraFramePair)
och automatisk resursfrigöring i context manager.
"""

import unittest
from unittest.mock import MagicMock
import numpy as np

from src.hardware.camera_controller import CameraController
from src.core.models import CameraFrame, CameraFramePair


class TestCameraController(unittest.TestCase):
    """Testsvit för CameraController-fasaden."""

    def test_initialization(self):
        """Verifiera att båda kamerorna CAM0 och CAM1 skapas och konfigureras likvärdigt."""
        ctrl = CameraController(verbose=False, auto_connect=False)
        self.assertEqual(sorted(list(ctrl._cameras.keys())), ["CAM0", "CAM1"])
        self.assertFalse(ctrl.is_connected)
        self.assertEqual(ctrl.exposure_us, 25000)
        self.assertEqual(ctrl.gain, 4.0)

    def test_invalid_camera_id_raises_value_error(self):
        """Verifiera att ogiltigt camera_id kastar ValueError vid enskilda operationer."""
        ctrl = CameraController(verbose=False, auto_connect=False)
        for invalid_id in ["CAM99", "invalid", "CSI0", ""]:
            with self.assertRaises(ValueError):
                ctrl.capture_single(invalid_id)
            with self.assertRaises(ValueError):
                ctrl.set_single_exposure(invalid_id, 20000, 2.0)

    def test_disconnected_behavior(self):
        """Verifiera att frånkopplad controller returnerar säkra ogiltiga objekt och inte kraschar."""
        ctrl = CameraController(verbose=False, auto_connect=False)
        self.assertFalse(ctrl.is_connected)

        # set_exposure ska returnera False
        self.assertFalse(ctrl.set_exposure(30000, 3.0))

        # capture_single ska returnera ogiltig CameraFrame
        single_frame = ctrl.capture_single("CAM0")
        self.assertIsInstance(single_frame, CameraFrame)
        self.assertFalse(single_frame.is_valid())

        # capture_both ska returnera ogiltig CameraFramePair
        pair = ctrl.capture_both()
        self.assertIsInstance(pair, CameraFramePair)
        self.assertFalse(pair.is_valid())

    def test_mocked_set_exposure_both(self):
        """Verifiera att set_exposure uppdaterar båda kamerorna samtidigt."""
        ctrl = CameraController(verbose=False, auto_connect=False)

        mock_cam0 = MagicMock()
        mock_cam1 = MagicMock()
        mock_cam0.set_exposure.return_value = True
        mock_cam1.set_exposure.return_value = True

        ctrl._cameras["CAM0"] = mock_cam0
        ctrl._cameras["CAM1"] = mock_cam1

        self.assertTrue(ctrl.set_exposure(exposure_us=30000, gain=5.0))
        self.assertEqual(ctrl.exposure_us, 30000)
        self.assertEqual(ctrl.gain, 5.0)
        mock_cam0.set_exposure.assert_called_once_with(30000, 5.0)
        mock_cam1.set_exposure.assert_called_once_with(30000, 5.0)

    def test_mocked_capture_single(self):
        """Verifiera att capture_single anropar vald kamera och returnerar dess bildruta."""
        ctrl = CameraController(verbose=False, auto_connect=False)

        mock_cam0 = MagicMock()
        dummy_frame = CameraFrame(
            image=np.zeros((10, 10, 3), dtype=np.uint8),
            camera_id="CAM0",
            _is_valid=True
        )
        mock_cam0.capture.return_value = dummy_frame
        ctrl._cameras["CAM0"] = mock_cam0

        frame = ctrl.capture_single("CAM0", samples=1)
        self.assertTrue(frame.is_valid())
        self.assertEqual(frame.camera_id, "CAM0")
        mock_cam0.capture.assert_called_once_with(samples=1, method="median")

    def test_mocked_capture_both_returns_camera_frame_pair(self):
        """Verifiera att capture_both returnerar CameraFramePair med korrekt left/right-bindning."""
        ctrl = CameraController(verbose=False, auto_connect=False)

        # Skapa mockade bildramar
        dummy_left = CameraFrame(
            image=np.zeros((10, 10, 3), dtype=np.uint8),
            camera_id="CAM1",
            _is_valid=True
        )
        dummy_right = CameraFrame(
            image=np.zeros((10, 10, 3), dtype=np.uint8),
            camera_id="CAM0",
            _is_valid=True
        )

        mock_cam0 = MagicMock()
        mock_cam1 = MagicMock()
        mock_cam0.capture.return_value = dummy_right
        mock_cam1.capture.return_value = dummy_left

        ctrl._cameras["CAM0"] = mock_cam0
        ctrl._cameras["CAM1"] = mock_cam1

        pair = ctrl.capture_both(samples=3, method="mean")

        self.assertIsInstance(pair, CameraFramePair)
        self.assertTrue(pair.is_valid())
        # Kontrollera strikt semantisk bindning: left=CAM1 (Vänster), right=CAM0 (Höger)
        self.assertEqual(pair.left.camera_id, "CAM1")
        self.assertEqual(pair.right.camera_id, "CAM0")
        mock_cam1.capture.assert_called_once_with(samples=3, method="mean")
        mock_cam0.capture.assert_called_once_with(samples=3, method="mean")

    def test_context_manager_cleanup(self):
        """Verifiera att context manager anropar disconnect() på båda kamerorna."""
        mock_cam0 = MagicMock()
        mock_cam1 = MagicMock()

        with CameraController(verbose=False, auto_connect=False) as ctrl:
            ctrl._cameras["CAM0"] = mock_cam0
            ctrl._cameras["CAM1"] = mock_cam1

        mock_cam0.disconnect.assert_called_once()
        mock_cam1.disconnect.assert_called_once()

    def test_repr(self):
        """Verifiera att __repr__ formaterar strängen korrekt."""
        ctrl = CameraController(verbose=False, auto_connect=False)
        r = repr(ctrl)
        self.assertIn("CameraController", r)
        self.assertIn("Frånkopplad", r)


if __name__ == "__main__":
    unittest.main()
