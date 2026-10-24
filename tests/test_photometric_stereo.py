"""
Enhetstester för PhotometricStereoEstimator (src/algorithms/photometric_stereo.py).
"""

from pathlib import Path
import unittest
import cv2
import numpy as np

from src.algorithms.photometric_stereo import PhotometricStereoEstimator
from src.core.models import (
    CameraFrame,
    CameraFramePair,
    PhotometricSweep,
    PhotometricResult,
)


class TestPhotometricStereoEstimator(unittest.TestCase):
    """Testsvit för Fotometrisk Stereoberäkning."""

    def setUp(self):
        """Initiera standardinstans med tyst loggning för enhetstester."""
        self.estimator = PhotometricStereoEstimator(
            camera_id="left_cam1",
            min_albedo=0.03,
            verbose=False,
        )

    def test_init_defaults(self):
        """Verifiera korrekt laddning av ljusmatris och pseudoinvers."""
        self.assertEqual(self.estimator.camera_id, "left_cam1")
        self.assertEqual(self.estimator.L.shape, (4, 3))
        self.assertEqual(self.estimator.L_pinv.shape, (3, 4))
        self.assertFalse(self.estimator.verbose)

    def test_init_right_camera(self):
        """Verifiera att instansiering för höger kamera (CAM0) fungerar."""
        est_right = PhotometricStereoEstimator(camera_id="right_cam0", verbose=False)
        self.assertEqual(est_right.camera_id, "right_cam0")
        self.assertEqual(est_right.L.shape, (4, 3))
        self.assertEqual(est_right.L_pinv.shape, (3, 4))

    def test_init_invalid_camera(self):
        """Verifiera att okända kameror kastar ValueError."""
        with self.assertRaises(ValueError):
            PhotometricStereoEstimator(camera_id="cam_99_fantasy")

    def test_synthetic_math_recovery(self):
        """
        Matematiskt test: Skapa kända intensiteter I = L * (rho * N) och
        verifiera att algoritmen återskapar exakt samma N och rho.
        """
        h, w = 10, 10
        # Förväntad normal: svagt lutande yta
        expected_N = np.array([0.2, -0.3, 0.9327379], dtype=np.float32)
        expected_N /= np.linalg.norm(expected_N)
        expected_albedo = 0.75

        g_expected = expected_albedo * expected_N # (3,)

        # Syntetiska intensiteter I = L * g
        I_vals = self.estimator.L @ g_expected # (4,)

        # Skapa (H, W, 4) intensitetsmatris
        I_tensor = np.zeros((h, w, 4), dtype=np.float32)
        I_tensor[:, :] = I_vals

        normals, albedo, valid_mask = self.estimator._compute_normals_and_albedo(I_tensor)

        self.assertTrue(np.all(valid_mask))
        np.testing.assert_allclose(albedo, expected_albedo, atol=1e-4)
        np.testing.assert_allclose(normals[:, :, 0], expected_N[0], atol=1e-4)
        np.testing.assert_allclose(normals[:, :, 1], expected_N[1], atol=1e-4)
        np.testing.assert_allclose(normals[:, :, 2], expected_N[2], atol=1e-4)

    def test_gradients(self):
        """Verifiera beräkning av ytgradienter (p = -nx/nz, q = -ny/nz)."""
        h, w = 4, 4
        normals = np.zeros((h, w, 3), dtype=np.float32)
        # Sätt yta med nx=0.1, ny=-0.2, nz=0.9
        normals[:, :, 0] = 0.1
        normals[:, :, 1] = -0.2
        normals[:, :, 2] = 0.9
        valid_mask = np.ones((h, w), dtype=bool)

        p, q = self.estimator._compute_gradients(normals, valid_mask)
        expected_p = -0.1 / 0.9
        expected_q = 0.2 / 0.9

        np.testing.assert_allclose(p, expected_p, atol=1e-5)
        np.testing.assert_allclose(q, expected_q, atol=1e-5)

    def _create_mock_sweep(self, h=60, w=80) -> PhotometricSweep:
        """Hjälpmetod för att bygga ett syntetiskt giltigt PhotometricSweep."""
        channels = []
        for ch_idx in range(1, 5):
            img_l = np.full((h, w, 3), 100 + ch_idx * 20, dtype=np.uint8)
            img_r = np.full((h, w, 3), 100 + ch_idx * 20, dtype=np.uint8)
            fl = CameraFrame(camera_id="CAM1", image=img_l, _is_valid=True)
            fr = CameraFrame(camera_id="CAM0", image=img_r, _is_valid=True)
            channels.append(CameraFramePair(left=fl, right=fr, timestamp=10.0))

        return PhotometricSweep(
            ch1=channels[0],
            ch2=channels[1],
            ch3=channels[2],
            ch4=channels[3],
            timestamp=10.0,
        )

    def test_estimate_valid_sweep(self):
        """Verifiera estimate() med ett syntetiskt PhotometricSweep."""
        sweep = self._create_mock_sweep(h=60, w=80)
        res = self.estimator.estimate(sweep)

        self.assertTrue(res.is_valid())
        self.assertEqual(res.camera_id, "left_cam1")
        self.assertEqual(res.normals.shape, (60, 80, 3))
        self.assertEqual(res.albedo.shape, (60, 80))
        self.assertEqual(res.p_gradient.shape, (60, 80))
        self.assertEqual(res.q_gradient.shape, (60, 80))
        self.assertGreater(res.execution_time_ms, 0.0)

    def test_estimate_invalid_sweep(self):
        """Verifiera att ogiltigt PhotometricSweep returnerar PhotometricResult.invalid()."""
        # 1. Invalid sweep objekt
        inv_sweep = PhotometricSweep.invalid()
        res1 = self.estimator.estimate(inv_sweep)
        self.assertFalse(res1.is_valid())

        # 2. None
        res2 = self.estimator.estimate(None)
        self.assertFalse(res2.is_valid())

    def test_estimate_with_real_images(self):
        """Kör PhotometricStereoEstimator på de sparade bilderna från kalibreringen."""
        p_l = [Path(f"data/calibration/photometric/raw_sweep_left_cam1_ch{i}.jpg") for i in range(1, 5)]
        p_r = [Path(f"data/calibration/photometric/raw_sweep_right_cam0_ch{i}.jpg") for i in range(1, 5)]

        all_exist = all(p.exists() for p in p_l + p_r)
        if not all_exist:
            self.skipTest("Sparade kalibreringsbilder saknas lokalt.")

        pairs = []
        for i in range(4):
            img_l = cv2.imread(str(p_l[i]))
            img_r = cv2.imread(str(p_r[i]))
            fl = CameraFrame(camera_id="CAM1", image=img_l, _is_valid=True)
            fr = CameraFrame(camera_id="CAM0", image=img_r, _is_valid=True)
            pairs.append(CameraFramePair(left=fl, right=fr, timestamp=123.45))

        sweep = PhotometricSweep(
            ch1=pairs[0],
            ch2=pairs[1],
            ch3=pairs[2],
            ch4=pairs[3],
            timestamp=123.45,
        )

        res = self.estimator.estimate(sweep)
        self.assertTrue(res.is_valid())
        self.assertEqual(res.normals.shape, (1232, 1640, 3))
        self.assertEqual(res.albedo.shape, (1232, 1640))

        # Kontrollera metadata
        info = res.to_dict()
        self.assertTrue(info["is_valid"])
        self.assertGreater(info["valid_pixels_count"], 1000000)
        self.assertGreater(info["albedo_stats"]["mean"], 0.1)


if __name__ == "__main__":
    unittest.main()
