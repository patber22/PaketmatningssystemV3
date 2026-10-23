"""
Enhetstester för StereoDepthEstimator (src/algorithms/stereo_depth.py).
"""

import json
from pathlib import Path
import unittest
import numpy as np

from src.algorithms.stereo_depth import StereoDepthEstimator
from src.core.models import CameraFrame, CameraFramePair, StereoDepthResult


class TestStereoDepthEstimator(unittest.TestCase):
    """Testsvit för Stereodjupsberäknaren."""

    @classmethod
    def setUpClass(cls):
        """Läs in den riktiga kalibreringsfilen för referenstester."""
        cls.calib_path = Path("config/calibration/data/stereo.json")
        if not cls.calib_path.exists():
            # Fallback till absolut sökväg om testet körs från annan arbetskatalog
            cls.calib_path = Path(__file__).resolve().parents[1] / "config/calibration/data/stereo.json"

        with open(cls.calib_path, "r", encoding="utf-8") as f:
            cls.calib_data = json.load(f)

    def setUp(self):
        """Initiera en standardinstans för testerna."""
        self.estimator = StereoDepthEstimator(
            calib_source=self.calib_data,
            min_disparity=64,
            num_disparities=192,
            block_size=5,
        )

    def test_init_from_file(self):
        """Verifiera att instansen kan initieras direkt från fil."""
        estimator = StereoDepthEstimator(calib_source=str(self.calib_path))
        self.assertEqual(estimator.width, 1640)
        self.assertEqual(estimator.height, 1232)
        self.assertEqual(estimator.Q.shape, (4, 4))
        self.assertEqual(estimator.map1_left.shape, (1232, 1640))

    def test_init_file_not_found(self):
        """Verifiera att FileNotFoundError kastas om filen inte finns."""
        with self.assertRaises(FileNotFoundError):
            StereoDepthEstimator(calib_source="non_existent_stereo_calib.json")

    def test_param_validation(self):
        """Verifiera att ogiltiga SGBM-parametrar fångas upp tidigt."""
        # num_disparities inte delbart med 16
        with self.assertRaises(ValueError):
            StereoDepthEstimator(calib_source=self.calib_data, num_disparities=100)

        # block_size jämnt tal
        with self.assertRaises(ValueError):
            StereoDepthEstimator(calib_source=self.calib_data, block_size=4)

        # block_size < 1
        with self.assertRaises(ValueError):
            StereoDepthEstimator(calib_source=self.calib_data, block_size=-1)

    def test_update_matcher_params(self):
        """Verifiera att matchningsparametrar kan uppdateras dynamiskt."""
        self.estimator.update_matcher_params(min_disparity=32, num_disparities=128, block_size=7)
        self.assertEqual(self.estimator.min_disparity, 32)
        self.assertEqual(self.estimator.num_disparities, 128)
        self.assertEqual(self.estimator.block_size, 7)

        # Kontrollera att ogiltiga värden i uppdatering också fångas
        with self.assertRaises(ValueError):
            self.estimator.update_matcher_params(num_disparities=55)

    def test_rectify(self):
        """Verifiera att bildrektifiering bevarar dimensioner och reagerar på fel."""
        w, h = self.estimator.width, self.estimator.height
        img_l = np.zeros((h, w, 3), dtype=np.uint8)
        img_r = np.zeros((h, w, 3), dtype=np.uint8)

        rect_l, rect_r = self.estimator.rectify(img_l, img_r)
        self.assertEqual(rect_l.shape, (h, w, 3))
        self.assertEqual(rect_r.shape, (h, w, 3))

        # Kontrollera tom inmatning
        with self.assertRaises(ValueError):
            self.estimator.rectify(np.array([]), img_r)

    def test_compute_disparity(self):
        """Verifiera disparitetsberäkning på syntetiska bilder."""
        w, h = self.estimator.width, self.estimator.height
        # Skapa syntetiskt mönster med textur så att SGBM kan hitta korrespondenser
        img_l = np.random.randint(0, 255, (h, w), dtype=np.uint8)
        img_r = np.roll(img_l, -80, axis=1)  # Förskjut 80 pixlar (inom 64..256 disparitetsintervall)

        disp = self.estimator.compute_disparity(img_l, img_r)
        self.assertEqual(disp.shape, (h, w))
        self.assertEqual(disp.dtype, np.float32)

        # Kontrollera tom inmatning
        with self.assertRaises(ValueError):
            self.estimator.compute_disparity(np.array([]), img_r)

    def test_reproject_to_3d(self):
        """Verifiera reprojicering till 3D och att ogiltiga dispariteter nollställs."""
        w, h = self.estimator.width, self.estimator.height
        disp = np.full((h, w), -1.0, dtype=np.float32)
        # Sätt en giltig region med 100 px disparitet
        disp[100:200, 100:200] = 100.0

        pc = self.estimator.reproject_to_3d(disp)
        self.assertEqual(pc.shape, (h, w, 3))
        self.assertEqual(pc.dtype, np.float32)

        # Ogiltiga dispariteter (-1.0) ska vara nollställda till 0.0
        self.assertTrue(np.all(pc[0:50, 0:50] == 0.0))

        # Giltig region ska ha ett positivt metriskt Z-avstånd
        valid_z = pc[150, 150, 2]
        self.assertGreater(valid_z, 500.0)
        self.assertLess(valid_z, 1500.0)

    def test_estimate_with_numpy_arrays(self):
        """Verifiera estimate() med rena numpy-arrayer."""
        w, h = self.estimator.width, self.estimator.height
        img_l = np.full((h, w, 3), 128, dtype=np.uint8)
        img_r = np.full((h, w, 3), 128, dtype=np.uint8)

        res = self.estimator.estimate(img_l, img_r)
        self.assertTrue(res.is_valid())
        self.assertIsNotNone(res.disparity)
        self.assertIsNotNone(res.point_cloud)
        self.assertIsNotNone(res.rectified_left)
        self.assertGreater(res.execution_time_ms, 0.0)
        self.assertGreater(res.timestamp, 0.0)

    def test_estimate_with_camera_frame_pair(self):
        """Verifiera estimate() med CameraFramePair domänobjekt."""
        w, h = self.estimator.width, self.estimator.height
        frame_l = CameraFrame(
            camera_id="CAM1",
            timestamp=100.0,
            resolution=(w, h),
            image=np.full((h, w, 3), 128, dtype=np.uint8),
            _is_valid=True,
        )
        frame_r = CameraFrame(
            camera_id="CAM0",
            timestamp=100.0,
            resolution=(w, h),
            image=np.full((h, w, 3), 128, dtype=np.uint8),
            _is_valid=True,
        )
        pair = CameraFramePair(left=frame_l, right=frame_r, timestamp=100.0)

        res = self.estimator.estimate(pair)
        self.assertTrue(res.is_valid())
        self.assertEqual(res.disparity.shape, (h, w))
        self.assertEqual(res.point_cloud.shape, (h, w, 3))

    def test_estimate_with_invalid_inputs(self):
        """Verifiera att felaktig inmatning returnerar ett säkert invalid-objekt."""
        # 1. Invalid CameraFramePair
        invalid_pair = CameraFramePair.invalid()
        res1 = self.estimator.estimate(invalid_pair)
        self.assertFalse(res1.is_valid())

        # 2. None
        res2 = self.estimator.estimate(None, None)
        self.assertFalse(res2.is_valid())

        # 3. Fel upplösning (inte 1640x1232)
        wrong_size = np.zeros((100, 100, 3), dtype=np.uint8)
        res3 = self.estimator.estimate(wrong_size, wrong_size)
        self.assertFalse(res3.is_valid())


if __name__ == "__main__":
    unittest.main()
