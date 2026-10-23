"""
Stereodjupsberäkning för Paketmätningssystem V3.

Klassen StereoDepthEstimator ansvarar för:
1. Rektifiering av stereobildpar med förberäknade rektifieringskartor.
2. Beräkning av tät disparitetskarta via OpenCV StereoSGBM (optimerad för CM5/ARM).
3. Reprojicering till metriskt 3D-punktmoln i millimeter med kalibrerad Q-matris.
4. Leverans av standardiserad StereoDepthResult dataklass.
"""

import json
from pathlib import Path
import time
from typing import Any, Dict, Optional, Tuple, Union
import cv2
import numpy as np

from src.core.models import CameraFrame, CameraFramePair, StereoDepthResult


class StereoDepthEstimator:
    """
    Räknar ut 3D-punktmoln och disparitet från ett kalibrerat stereokamerapar.
    
    Förberäknar remappingskartor och matcher vid initiering för minimal
    körtidslatens (< 200 ms på CM5).
    """

    def __init__(
        self,
        calib_source: Union[str, Path, dict] = "config/calibration/data/stereo.json",
        min_disparity: int = 64,
        num_disparities: int = 192,
        block_size: int = 5,
        uniqueness_ratio: int = 10,
        speckle_window_size: int = 100,
        speckle_range: int = 2,
        disp12_max_diff: int = 1,
        mode: int = cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    ):
        """
        Initierar StereoDepthEstimator och förberäknar rektifieringstabeller.

        Args:
            calib_source: Sökväg till stereo.json eller färdig dictionary med kalibreringsdata.
            min_disparity: Minsta disparitet att söka (t.ex. 64 för att spara beräkning mot oändligheten).
            num_disparities: Antal disparitetssteg att söka (måste vara delbart med 16).
            block_size: Matchningsblockets storlek i pixlar (udda tal, t.ex. 3, 5 eller 7).
            uniqueness_ratio: Tröskelvärde för unikhet i matchning (i procent, t.ex. 10).
            speckle_window_size: Max storlek på brusöar/speckles som ska filtreras bort.
            speckle_range: Max disparitetsvariation inom en brusö.
            disp12_max_diff: Max tillåten skillnad i vänster-höger-konsistenskontroll.
            mode: StereoSGBM-läge (standard STEREO_SGBM_MODE_SGBM_3WAY för snabb ARM-exekvering).
        """
        # 1. Ladda kalibreringsdata
        self.calib_data = self._load_calibration(calib_source)

        # 2. Extrahera upplösning och matriser
        self.width = int(self.calib_data["resolution"][0])
        self.height = int(self.calib_data["resolution"][1])

        cams = self.calib_data["cameras"]
        k_l = np.array(cams["left_cam1"]["camera_matrix_K"], dtype=np.float64)
        d_l = np.array(cams["left_cam1"]["dist_coeffs_D"], dtype=np.float64)
        r1 = np.array(cams["left_cam1"]["rectification_R1"], dtype=np.float64)
        p1 = np.array(cams["left_cam1"]["projection_P1"], dtype=np.float64)

        k_r = np.array(cams["right_cam0"]["camera_matrix_K"], dtype=np.float64)
        d_r = np.array(cams["right_cam0"]["dist_coeffs_D"], dtype=np.float64)
        r2 = np.array(cams["right_cam0"]["rectification_R2"], dtype=np.float64)
        p2 = np.array(cams["right_cam0"]["projection_P2"], dtype=np.float64)

        self.Q = np.array(self.calib_data["disparity_to_depth_Q"], dtype=np.float32)

        # 3. Förberäkna rektifieringskartor (körs enbart en gång vid start)
        img_size = (self.width, self.height)
        self.map1_left, self.map2_left = cv2.initUndistortRectifyMap(
            k_l, d_l, r1, p1, img_size, cv2.CV_32FC1
        )
        self.map1_right, self.map2_right = cv2.initUndistortRectifyMap(
            k_r, d_r, r2, p2, img_size, cv2.CV_32FC1
        )

        # 4. Konfigurera SGBM-matcher
        self.min_disparity = int(min_disparity)
        self.num_disparities = int(num_disparities)
        self.block_size = int(block_size)
        self.uniqueness_ratio = int(uniqueness_ratio)
        self.speckle_window_size = int(speckle_window_size)
        self.speckle_range = int(speckle_range)
        self.disp12_max_diff = int(disp12_max_diff)
        self.mode = int(mode)

        self._validate_params()
        self.matcher = self._create_matcher()

    def _load_calibration(self, source: Union[str, Path, dict]) -> dict:
        """Laddar och validerar struktur på kalibreringsdata."""
        if isinstance(source, dict):
            return source

        path = Path(source)
        if not path.is_absolute() and not path.exists():
            # Fallback relativt till projektets rotkatalog
            project_root = Path(__file__).resolve().parents[2]
            path = project_root / source

        if not path.exists():
            raise FileNotFoundError(f"Stereo-kalibreringsfil hittades inte: {path}")

        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _validate_params(self) -> None:
        """Verifierar att disparitets- och blockparametrar uppfyller OpenCV:s krav."""
        if self.num_disparities <= 0 or self.num_disparities % 16 != 0:
            raise ValueError(f"num_disparities ({self.num_disparities}) måste vara positivt och delbart med 16.")
        if self.block_size < 1 or self.block_size % 2 == 0:
            raise ValueError(f"block_size ({self.block_size}) måste vara ett udda heltal >= 1.")

    def _create_matcher(self) -> cv2.StereoSGBM:
        """Skapar instans av OpenCV StereoSGBM med aktuella parametrar."""
        # P1 och P2 styr jämnhetsstraff (anpassas för 1-kanals gråskala)
        p1_val = 8 * 1 * (self.block_size ** 2)
        p2_val = 32 * 1 * (self.block_size ** 2)

        return cv2.StereoSGBM_create(
            minDisparity=self.min_disparity,
            numDisparities=self.num_disparities,
            blockSize=self.block_size,
            P1=p1_val,
            P2=p2_val,
            disp12MaxDiff=self.disp12_max_diff,
            uniquenessRatio=self.uniqueness_ratio,
            speckleWindowSize=self.speckle_window_size,
            speckleRange=self.speckle_range,
            mode=self.mode,
        )

    def update_matcher_params(
        self,
        min_disparity: Optional[int] = None,
        num_disparities: Optional[int] = None,
        block_size: Optional[int] = None,
        uniqueness_ratio: Optional[int] = None,
        speckle_window_size: Optional[int] = None,
        speckle_range: Optional[int] = None,
        disp12_max_diff: Optional[int] = None,
        mode: Optional[int] = None,
    ) -> None:
        """
        Uppdaterar inställningar för matchern utan att behöva bygga om rektifieringstabellerna.
        """
        if min_disparity is not None:
            self.min_disparity = int(min_disparity)
        if num_disparities is not None:
            self.num_disparities = int(num_disparities)
        if block_size is not None:
            self.block_size = int(block_size)
        if uniqueness_ratio is not None:
            self.uniqueness_ratio = int(uniqueness_ratio)
        if speckle_window_size is not None:
            self.speckle_window_size = int(speckle_window_size)
        if speckle_range is not None:
            self.speckle_range = int(speckle_range)
        if disp12_max_diff is not None:
            self.disp12_max_diff = int(disp12_max_diff)
        if mode is not None:
            self.mode = int(mode)

        self._validate_params()
        self.matcher = self._create_matcher()

    def rectify(
        self, left: np.ndarray, right: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Rektifierar ett bildpar så att epipolarlinjerna blir perfekt horisontella.

        Args:
            left: Orektifierad vänsterbild (CAM1).
            right: Orektifierad högerbild (CAM0).

        Returns:
            Tuple med (rektifierad_vänster, rektifierad_höger).
        """
        if left is None or right is None or left.size == 0 or right.size == 0:
            raise ValueError("Vänster- och högerbild måste vara giltiga numpy-arrayer.")

        rect_l = cv2.remap(left, self.map1_left, self.map2_left, interpolation=cv2.INTER_LINEAR)
        rect_r = cv2.remap(right, self.map1_right, self.map2_right, interpolation=cv2.INTER_LINEAR)
        return rect_l, rect_r

    def compute_disparity(
        self, rect_left: np.ndarray, rect_right: np.ndarray
    ) -> np.ndarray:
        """
        Beräknar tät disparitetskarta från rektifierade bilder.

        Args:
            rect_left: Rektifierad vänsterbild.
            rect_right: Rektifierad högerbild.

        Returns:
            Disparitetskarta som float32-array (H, W). Ogiltiga pixlar är satta till -1.0.
        """
        if rect_left is None or rect_right is None or rect_left.size == 0 or rect_right.size == 0:
            raise ValueError("Rektifierade bilder saknas eller är tomma.")

        # Konvertera till 1-kanals gråskala för matchning om färg (minskar beräkningstid med 3x)
        if len(rect_left.shape) == 3 and rect_left.shape[2] == 3:
            gray_l = cv2.cvtColor(rect_left, cv2.COLOR_BGR2GRAY)
        else:
            gray_l = rect_left

        if len(rect_right.shape) == 3 and rect_right.shape[2] == 3:
            gray_r = cv2.cvtColor(rect_right, cv2.COLOR_BGR2GRAY)
        else:
            gray_r = rect_right

        # StereoSGBM compute returnerar 16-bitars heltal skalat med 16
        disp16 = self.matcher.compute(gray_l, gray_r)
        disp_f = disp16.astype(np.float32) / 16.0

        # Markera pixlar som inte matchades eller ligger under min_disparity som ogiltiga
        invalid_mask = (disp_f < float(self.min_disparity)) | (~np.isfinite(disp_f))
        disp_f[invalid_mask] = -1.0
        return disp_f

    def reproject_to_3d(self, disparity: np.ndarray) -> np.ndarray:
        """
        Reprojicerar disparitetskarta till ett metriskt 3D-punktmoln i millimeter.

        Args:
            disparity: Tät float32 disparitetskarta (H, W).

        Returns:
            Punktmoln som float32-array (H, W, 3) där kanaler är [X, Y, Z] i mm.
            Ogiltiga punkter (disparitet <= 0) nollställs.
        """
        if disparity is None or disparity.size == 0:
            raise ValueError("Disparitetskarta saknas eller är tom.")

        point_cloud = cv2.reprojectImageTo3D(disparity, self.Q)

        # Nollställ ogiltiga och icke-finita punkter för att undvika oändligheter i 3D
        invalid_mask = (disparity <= 0.0) | (~np.isfinite(disparity))
        point_cloud[invalid_mask] = 0.0
        return point_cloud

    def estimate(
        self,
        left: Union[np.ndarray, CameraFrame, CameraFramePair],
        right: Optional[Union[np.ndarray, CameraFrame]] = None,
    ) -> StereoDepthResult:
        """
        Komplett pipeline: rektifiering -> disparitet -> 3D-punktmoln.

        Accepterar antingen:
        - Ett färdigt CameraFramePair
        - Två separata CameraFrame-objekt
        - Två separata numpy bildmatriser (left, right)

        Returns:
            StereoDepthResult innehållande disparitet, punktmoln och exekveringstid.
            Vid fel returneras StereoDepthResult.invalid().
        """
        t0 = time.perf_counter()

        # Extrahera råa bildmatriser baserat på inmatningstyp
        if isinstance(left, CameraFramePair):
            if not left.is_valid():
                return StereoDepthResult.invalid()
            img_l = left.left.image
            img_r = left.right.image
        else:
            img_l = left.image if isinstance(left, CameraFrame) else left
            img_r = right.image if isinstance(right, CameraFrame) else right

        if img_l is None or img_r is None:
            return StereoDepthResult.invalid()

        try:
            # Kontrollera att bildens dimensioner stämmer överens med kalibreringen
            h, w = img_l.shape[:2]
            if (w, h) != (self.width, self.height):
                return StereoDepthResult.invalid()

            # 1. Rektifiera
            rect_l, rect_r = self.rectify(img_l, img_r)

            # 2. Beräkna disparitet
            disp = self.compute_disparity(rect_l, rect_r)

            # 3. Reprojicera till 3D
            point_cloud = self.reproject_to_3d(disp)

            exec_time_ms = (time.perf_counter() - t0) * 1000.0

            return StereoDepthResult(
                disparity=disp,
                point_cloud=point_cloud,
                rectified_left=rect_l,
                timestamp=time.time(),
                execution_time_ms=exec_time_ms,
                _is_valid=True,
            )
        except Exception:
            return StereoDepthResult.invalid()
