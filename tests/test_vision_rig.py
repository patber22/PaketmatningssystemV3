"""
Enhetstester för VisionRig fasaden (src/hardware/vision_rig.py).

Testar initiering, synkroniserad belysning, stereobildtagning,
fotometriska svep och kompletta mätcykler med mockade kontroller.
"""

import unittest
from unittest.mock import MagicMock, call

import numpy as np

from src.core.models import CameraFrame, CameraFramePair, PhotometricSweep, VisionCapture
from src.hardware.vision_rig import VisionRig


class TestVisionRig(unittest.TestCase):
    """Testsvit för VisionRig-klassen."""

    def _create_mock_pair(self, ts: float = 1000.0) -> CameraFramePair:
        img = np.full((10, 10, 3), 100, dtype=np.uint8)
        f_left = CameraFrame(image=img, camera_id="CAM1", timestamp=ts, _is_valid=True)
        f_right = CameraFrame(image=img, camera_id="CAM0", timestamp=ts, _is_valid=True)
        return CameraFramePair(left=f_left, right=f_right, timestamp=ts)

    def setUp(self):
        """Skapa mockade kontroller och en VisionRig-instans."""
        self.mock_cameras = MagicMock()
        self.mock_leds = MagicMock()

        self.mock_cameras.is_connected = True
        self.mock_cameras.exposure_us = 15000
        self.mock_cameras.gain = 2.0
        self.mock_cameras.capture_both.return_value = self._create_mock_pair()

        self.mock_leds.is_connected = True

        self.rig = VisionRig(
            camera_controller=self.mock_cameras,
            led_controller=self.mock_leds,
            stereo_exposure_us=15000,
            stereo_gain=2.0,
            photo_exposure_us=35000,
            photo_gain=4.0,
            settle_time_s=0.0,  # 0s för snabba enhetstester
            verbose=False,
            auto_connect=False,
        )

    def tearDown(self):
        """Stäng ner VisionRig."""
        self.rig.disconnect()

    def test_default_init(self):
        """Verifiera standardvärden och profiler vid initiering."""
        self.assertEqual(self.rig.stereo_exposure_us, 15000)
        self.assertEqual(self.rig.stereo_gain, 2.0)
        self.assertEqual(self.rig.photo_exposure_us, 35000)
        self.assertEqual(self.rig.photo_gain, 4.0)
        self.assertEqual(self.rig.settle_time_s, 0.0)

    def test_connect_and_disconnect(self):
        """Verifiera anrop till connect och disconnect på kontroller."""
        self.mock_cameras.connect.return_value = True
        self.mock_leds.connect.return_value = True

        self.assertTrue(self.rig.connect())
        self.mock_cameras.connect.assert_called_once()
        self.mock_leds.connect.assert_called_once()

        self.rig.disconnect()
        self.mock_leds.all_off.assert_called()
        self.mock_leds.disconnect.assert_called_once()
        self.mock_cameras.disconnect.assert_called_once()

    def test_is_connected_property(self):
        """Verifiera att is_connected kräver att BÅDE kameror och LEDs är anslutna."""
        self.mock_cameras.is_connected = True
        self.mock_leds.is_connected = True
        self.assertTrue(self.rig.is_connected)

        self.mock_cameras.is_connected = False
        self.assertFalse(self.rig.is_connected)

        self.mock_cameras.is_connected = True
        self.mock_leds.is_connected = False
        self.assertFalse(self.rig.is_connected)

    def test_set_exposure(self):
        """Verifiera vidarebefordran av set_exposure till kamerakontrollern."""
        self.rig.set_exposure(20000, 3.5)
        self.mock_cameras.set_exposure.assert_called_with(20000, 3.5)

    def test_capture_stereo_all_leds(self):
        """Verifiera att capture_stereo('all') tänder alla LEDs, tar bild och släcker."""
        pair = self.rig.capture_stereo(led_mode="all", samples=2, method="mean")

        self.assertTrue(pair.is_valid())
        self.mock_leds.all_on.assert_called_once()
        self.mock_cameras.capture_both.assert_called_with(samples=2, method="mean")
        self.mock_leds.all_off.assert_called()

    def test_capture_stereo_ambient(self):
        """Verifiera att capture_stereo('ambient') släcker alla LEDs och tar bild."""
        pair = self.rig.capture_stereo(led_mode="ambient", samples=1, method="median")

        self.assertTrue(pair.is_valid())
        self.mock_leds.all_off.assert_called()
        self.mock_cameras.capture_both.assert_called_with(samples=1, method="median")

    def test_capture_stereo_invalid_mode(self):
        """Verifiera att ogiltigt belysningsläge kastar ValueError."""
        with self.assertRaises(ValueError):
            self.rig.capture_stereo(led_mode="disco")

    def test_capture_photometric_sweep_success(self):
        """Verifiera att svepet stegar igenom Kanal 1-4 och returnerar PhotometricSweep."""
        sweep = self.rig.capture_photometric_sweep(samples=1, method="median")

        self.assertIsInstance(sweep, PhotometricSweep)
        self.assertTrue(sweep.is_valid())

        # Verifiera att Kanal 1, 2, 3, 4 tändes och släcktes i ordning
        self.mock_leds.on.assert_has_calls([call(1), call(2), call(3), call(4)])
        self.mock_leds.off.assert_has_calls([call(1), call(2), call(3), call(4)])
        self.mock_leds.all_off.assert_called()

        # Verifiera 4 anrop till capture_both
        self.assertEqual(self.mock_cameras.capture_both.call_count, 4)

    def test_capture_photometric_sweep_error_handling(self):
        """Verifiera att fel under svepet fångas och returnerar ogiltigt svep."""
        self.mock_cameras.capture_both.side_effect = RuntimeError("Kamerafel!")

        sweep = self.rig.capture_photometric_sweep()
        self.assertIsInstance(sweep, PhotometricSweep)
        self.assertFalse(sweep.is_valid())
        # Släckning ska ändå garanteras
        self.mock_leds.all_off.assert_called()

    def test_capture_measurement_cycle(self):
        """Verifiera att mätcykeln returnerar en giltig VisionCapture med båda momenten."""
        cap = self.rig.capture_measurement_cycle(stereo_samples=1, photo_samples=1)

        self.assertIsInstance(cap, VisionCapture)
        self.assertTrue(cap.is_valid())
        self.assertTrue(cap.stereo.is_valid())
        self.assertTrue(cap.photometric.is_valid())

    def test_context_manager(self):
        """Verifiera att with VisionRig() stänger ner och släcker vid avslut."""
        with self.rig as r:
            self.assertEqual(r, self.rig)

        self.mock_leds.disconnect.assert_called_once()
        self.mock_cameras.disconnect.assert_called_once()


if __name__ == "__main__":
    unittest.main()
