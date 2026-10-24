"""
Stereodjupsberäkning för Paketmätningssystem V3.

Klassen StereoDepthEstimator ansvarar för:
1. Rektifiering av NoIR-stereopar (CAM1/CAM0).
2. Beräkning av tät disparitetskarta via OpenCV StereoSGBM.
3. Reprojicering till metriskt 3D-punktmoln i millimeter med kalibrerad Q-matris.
4. Leverans av standardiserad StereoDepthResult dataklass.
"""

import json
from pathlib import Path
import time
from typing import Tuple
import cv2
import numpy as np

from src.core.models import CameraFramePair, StereoDepthResult

# Fast systemstandard för stereokalibrering
CALIB_PATH = Path(__file__).resolve().parents[2] / "config" / "calibration" / "data" / "stereo.json"

# Abstraherade SGBM-lägen
SGBM_MODES = {
    "fast": cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    "precise": cv2.STEREO_SGBM_MODE_HH,
}


class StereoDepthEstimator:
    """
    Räknar ut 3D-punktmoln och disparitet från ett NoIR-stereokamerapar.

    Deterministisk och immutabel efter initiering. Förberäknar remappingskartor
    och matcher vid start för minimal körtidslatens (< 200 ms på CM5).
    """

    def __init__(
        self,
        min_disparity: int = 64,
        num_disparities: int = 192,
        block_size: int = 5,
        mode: str = "fast",
        uniqueness_ratio: int = 10,
        speckle_window_size: int = 100,
        speckle_range: int = 2,
        disp12_max_diff: int = 1,
        verbose: bool = True,
    ):
        """
        Initierar StereoDepthEstimator och förberäknar rektifieringstabeller.

        Args:
            min_disparity: Minsta disparitet att söka (t.ex. 64 för att spara beräkning mot oändligheten).
            num_disparities: Antal disparitetssteg att söka (måste vara delbart med 16).
            block_size: Matchningsblockets storlek i pixlar (udda tal, t.ex. 3, 5 eller 7).
            mode: Beräkningsläge ('fast' för 3-vägs SGBM eller 'precise' för 8-vägs HH).
            uniqueness_ratio: Tröskelvärde för unikhet i matchning (i procent, standard 10).
            speckle_window_size: Max storlek på brusöar/speckles som ska filtreras bort.
            speckle_range: Max disparitetsvariation inom en brusö.
            disp12_max_diff: Max tillåten skillnad i vänster-höger-konsistenskontroll.
            verbose: Om True loggas varningar vid felaktiga indata eller beräkningsfel.
        """
        self.verbose = verbose

        # 1. Validera parametrar
        if num_disparities <= 0 or num_disparities % 16 != 0:
            raise ValueError(f"num_disparities ({num_disparities}) måste vara positivt och delbart med 16.")
        if block_size < 1 or block_size % 2 == 0:
            raise ValueError(f"block_size ({block_size}) måste vara ett udda heltal >= 1.")

        mode_key = mode.lower()
        if mode_key not in SGBM_MODES:
            raise ValueError(f"Okänt läge '{mode}'. Tillåtna lägen är: {list(SGBM_MODES.keys())}")

        self.min_disparity = int(min_disparity)
        self.num_disparities = int(num_disparities)
        self.block_size = int(block_size)
        self.mode = mode_key

        # 2. Ladda fast kalibreringsdata
        if not CALIB_PATH.exists():
            raise FileNotFoundError(f"Stereo-kalibreringsfil hittades inte: {CALIB_PATH}")

        with open(CALIB_PATH, "r", encoding="utf-8") as f:
            calib_data = json.load(f)

        # 3. Extrahera upplösning och matriser
        self.width = int(calib_data["resolution"][0])
        self.height = int(calib_data["resolution"][1])

        cams = calib_data["cameras"]
        k_l = np.array(cams["left_cam1"]["camera_matrix_K"], dtype=np.float64)
        d_l = np.array(cams["left_cam1"]["dist_coeffs_D"], dtype=np.float64)
        r1 = np.array(cams["left_cam1"]["rectification_R1"], dtype=np.float64)
        p1 = np.array(cams["left_cam1"]["projection_P1"], dtype=np.float64)

        k_r = np.array(cams["right_cam0"]["camera_matrix_K"], dtype=np.float64)
        d_r = np.array(cams["right_cam0"]["dist_coeffs_D"], dtype=np.float64)
        r2 = np.array(cams["right_cam0"]["rectification_R2"], dtype=np.float64)
        p2 = np.array(cams["right_cam0"]["projection_P2"], dtype=np.float64)

        self.Q = np.array(calib_data["disparity_to_depth_Q"], dtype=np.float32)

        # 4. Förberäkna rektifieringskartor (körs enbart en gång vid instansiering)
        img_size = (self.width, self.height)
        self._map1_left, self._map2_left = cv2.initUndistortRectifyMap(
            k_l, d_l, r1, p1, img_size, cv2.CV_32FC1
        )
        self._map1_right, self._map2_right = cv2.initUndistortRectifyMap(
            k_r, d_r, r2, p2, img_size, cv2.CV_32FC1
        )

        # 5. Konfigurera SGBM-matcher
        p1_val = 8 * 1 * (self.block_size ** 2)
        p2_val = 32 * 1 * (self.block_size ** 2)

        self._matcher = cv2.StereoSGBM_create(
            minDisparity=self.min_disparity,
            numDisparities=self.num_disparities,
            blockSize=self.block_size,
            P1=p1_val,
            P2=p2_val,
            disp12MaxDiff=int(disp12_max_diff),
            uniquenessRatio=int(uniqueness_ratio),
            speckleWindowSize=int(speckle_window_size),
            speckleRange=int(speckle_range),
            mode=SGBM_MODES[mode_key],
        )

    def _rectify(
        self, img_left: np.ndarray, img_right: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Intern rektifiering via förberäknade remappingskartor."""
        rect_l = cv2.remap(img_left, self._map1_left, self._map2_left, interpolation=cv2.INTER_LINEAR)
        rect_r = cv2.remap(img_right, self._map1_right, self._map2_right, interpolation=cv2.INTER_LINEAR)
        return rect_l, rect_r

    def _compute_disparity(
        self, rect_left: np.ndarray, rect_right: np.ndarray
    ) -> np.ndarray:
        """Intern disparitetsberäkning från rektifierade bilder."""
        # Konvertera till 1-kanals gråskala om färgbild (optimerar beräkningstid för ARM)
        if len(rect_left.shape) == 3 and rect_left.shape[2] == 3:
            gray_l = cv2.cvtColor(rect_left, cv2.COLOR_BGR2GRAY)
        else:
            gray_l = rect_left

        if len(rect_right.shape) == 3 and rect_right.shape[2] == 3:
            gray_r = cv2.cvtColor(rect_right, cv2.COLOR_BGR2GRAY)
        else:
            gray_r = rect_right

        # StereoSGBM returnerar 16-bitars heltal med 4 subpixel-bitar (därav / 16.0)
        disp16 = self._matcher.compute(gray_l, gray_r)
        disp_f = disp16.astype(np.float32) / 16.0

        # Markera pixlar under min_disparity eller icke-finita värden som ogiltiga (-1.0)
        invalid_mask = (disp_f < float(self.min_disparity)) | (~np.isfinite(disp_f))
        disp_f[invalid_mask] = -1.0
        return disp_f

    def _reproject_to_3d(self, disparity: np.ndarray) -> np.ndarray:
        """Intern reprojicering från disparitet till 3D-punktmoln i millimeter."""
        point_cloud = cv2.reprojectImageTo3D(disparity, self.Q)

        # Nollställ ogiltiga punkter för att undvika oändligheter i 3D-matrisen
        invalid_mask = (disparity <= 0.0) | (~np.isfinite(disparity))
        point_cloud[invalid_mask] = 0.0
        return point_cloud

    def estimate(self, pair: CameraFramePair) -> StereoDepthResult:
        """
        Komplett pipeline: Validerar CameraFramePair och beräknar metriskt 3D-stereodjup.

        Args:
            pair: Ett giltigt stereopar från NoIR-kamerorna (CAM1 vänster, CAM0 höger).

        Returns:
            StereoDepthResult med disparitet, metriskt 3D-punktmoln och exekveringstid.
            Vid ogiltigt par eller fel returneras StereoDepthResult.invalid().
        """
        t0 = time.perf_counter()

        # 1. Strikt kontroll av datatyp och validitet via modellens inbyggda funktion
        if not isinstance(pair, CameraFramePair) or not pair.is_valid():
            if self.verbose:
                print("[StereoDepthEstimator] Varning: Inmatat CameraFramePair är ogiltigt.")
            return StereoDepthResult.invalid()

        img_l = pair.left.image
        img_r = pair.right.image

        if not isinstance(img_l, np.ndarray) or not isinstance(img_r, np.ndarray):
            if self.verbose:
                print("[StereoDepthEstimator] Varning: Bildmatriser saknas i paret.")
            return StereoDepthResult.invalid()

        # 2. Kontrollera att bildens upplösning matchar kalibreringen
        h, w = img_l.shape[:2]
        if (w, h) != (self.width, self.height):
            if self.verbose:
                print(f"[StereoDepthEstimator] Varning: Upplösning ({w}x{h}) matchar inte kalibrerad ({self.width}x{self.height}).")
            return StereoDepthResult.invalid()

        try:
            # 3. Rektifiera
            rect_l, rect_r = self._rectify(img_l, img_r)

            # 4. Beräkna disparitet
            disp = self._compute_disparity(rect_l, rect_r)

            # 5. Reprojicera till metriskt punktmoln (mm)
            point_cloud = self._reproject_to_3d(disp)

            exec_time_ms = (time.perf_counter() - t0) * 1000.0

            return StereoDepthResult(
                disparity=disp,
                point_cloud=point_cloud,
                rectified_left=rect_l,
                timestamp=time.time(),
                execution_time_ms=exec_time_ms,
                _is_valid=True,
            )
        except Exception as exc:
            if self.verbose:
                print(f"[StereoDepthEstimator] Fel vid beräkning: {exc}")
            return StereoDepthResult.invalid()
