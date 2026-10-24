"""
Enhetstester för StereoDepthEstimator (src/algorithms/stereo_depth.py).
"""

import unittest
import numpy as np

from src.algorithms.stereo_depth import StereoDepthEstimator
from src.core.models import CameraFrame, CameraFramePair


class TestStereoDepthEstimator(unittest.TestCase):
    """Testsvit för Stereodjupsberäknaren (minimalistisk och deterministisk)."""

    def setUp(self):
        """Initiera en standardinstans för testerna."""
        self.estimator = StereoDepthEstimator(
            min_disparity=64,
            num_disparities=192,
            block_size=5,
            mode="fast",
            verbose=False,
        )

    def test_init_defaults(self):
        """Verifiera att instansen laddar standardkalibrering och sätter korrekta dimensioner."""
        self.assertEqual(self.estimator.width, 1640)
        self.assertEqual(self.estimator.height, 1232)
        self.assertEqual(self.estimator.Q.shape, (4, 4))
        self.assertEqual(self.estimator.mode, "fast")
        self.assertFalse(self.estimator.verbose)

    def test_init_modes(self):
        """Verifiera att 'fast' och 'precise' accepteras, medan okända lägen fångas."""
        est_precise = StereoDepthEstimator(mode="precise", verbose=False)
        self.assertEqual(est_precise.mode, "precise")

        with self.assertRaises(ValueError):
            StereoDepthEstimator(mode="ultra_super_mode")

    def test_param_validation(self):
        """Verifiera att felaktiga disparitets- och blockparametrar fångas tidigt."""
        # num_disparities inte delbart med 16
        with self.assertRaises(ValueError):
            StereoDepthEstimator(num_disparities=100)

        # block_size jämnt tal
        with self.assertRaises(ValueError):
            StereoDepthEstimator(block_size=4)

        # block_size < 1
        with self.assertRaises(ValueError):
            StereoDepthEstimator(block_size=-1)

    def test_estimate_with_valid_pair(self):
        """Verifiera att estimate() med ett giltigt CameraFramePair returnerar ett giltigt StereoDepthResult."""
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
        self.assertEqual(res.rectified_left.shape, (h, w, 3))
        self.assertGreater(res.execution_time_ms, 0.0)
        self.assertGreater(res.timestamp, 0.0)

    def test_estimate_with_invalid_inputs(self):
        """Verifiera att felaktiga indata returnerar StereoDepthResult.invalid()."""
        # 1. Ogiltigt CameraFramePair
        invalid_pair = CameraFramePair.invalid()
        res1 = self.estimator.estimate(invalid_pair)
        self.assertFalse(res1.is_valid())

        # 2. Fel typ (t.ex. None eller ren numpy-array)
        res2 = self.estimator.estimate(None)
        self.assertFalse(res2.is_valid())

        # 3. Fel upplösning (inte 1640x1232)
        wrong_frame_l = CameraFrame(
            camera_id="CAM1",
            resolution=(100, 100),
            image=np.zeros((100, 100, 3), dtype=np.uint8),
            _is_valid=True,
        )
        wrong_frame_r = CameraFrame(
            camera_id="CAM0",
            resolution=(100, 100),
            image=np.zeros((100, 100, 3), dtype=np.uint8),
            _is_valid=True,
        )
        wrong_pair = CameraFramePair(left=wrong_frame_l, right=wrong_frame_r)
        res3 = self.estimator.estimate(wrong_pair)
        self.assertFalse(res3.is_valid())

    def test_internal_helpers(self):
        """Verifiera de interna beräkningsstegen (_rectify, _compute_disparity, _reproject_to_3d)."""
        w, h = self.estimator.width, self.estimator.height
        img_l = np.full((h, w, 3), 100, dtype=np.uint8)
        img_r = np.full((h, w, 3), 100, dtype=np.uint8)

        # 1. Rektifiering
        rect_l, rect_r = self.estimator._rectify(img_l, img_r)
        self.assertEqual(rect_l.shape, (h, w, 3))
        self.assertEqual(rect_r.shape, (h, w, 3))

        # 2. Disparitet
        disp = self.estimator._compute_disparity(rect_l, rect_r)
        self.assertEqual(disp.shape, (h, w))
        self.assertEqual(disp.dtype, np.float32)

        # 3. 3D-reprojicering
        synthetic_disp = np.full((h, w), -1.0, dtype=np.float32)
        synthetic_disp[100:200, 100:200] = 100.0  # 100 px giltig disparitet
        pc = self.estimator._reproject_to_3d(synthetic_disp)
        self.assertEqual(pc.shape, (h, w, 3))

        # Ogiltiga dispariteter (-1.0) ska vara nollställda till 0.0
        self.assertTrue(np.all(pc[0:50, 0:50] == 0.0))
        # Giltig disparitet ska ge positivt metriskt djup
        self.assertGreater(pc[150, 150, 2], 500.0)


if __name__ == "__main__":
    unittest.main()
