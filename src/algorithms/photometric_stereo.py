"""
Fotometrisk stereoberäkning för Paketmätningssystem V3.

Klassen PhotometricStereoEstimator ansvarar för:
1. Inläsning av kalibrerade ljusvektorer och relativa intensiteter från photometric.json.
2. Förberäkning av pseudoinversen (L_pinv) för blixtsnabb linjär algebra.
3. Extraktion av 4-kanals ljusintensiteter från ett PhotometricSweep.
4. Beräkning av ytnormaler N = (nx, ny, nz) och skuggfri albedo rho via Woodhams algoritm.
5. Beräkning av partiella ytlutningar (p, q-gradienter) för framtida ytintegration.
6. Leverans av standardiserad PhotometricResult dataklass.
"""

import json
from pathlib import Path
import time
from typing import Tuple
import cv2
import numpy as np

from src.core.models import PhotometricSweep, PhotometricResult

# Fast systemstandard för fotometrisk kalibrering
CALIB_PATH = Path(__file__).resolve().parents[2] / "config" / "calibration" / "data" / "photometric.json"

SUPPORTED_CAMERAS = ("left_cam1", "right_cam0")


class PhotometricStereoEstimator:
    """
    Beräknar ytnormaler och albedo från ett 4-kanals fotometriskt svep (PhotometricSweep).

    Deterministisk och immutabel efter initiering. Förberäknar ljusmatrisens pseudoinvers
    vid start för minimal körtidslatens (< 170 ms på CM5).
    """

    def __init__(
        self,
        camera_id: str = "left_cam1",
        min_albedo: float = 0.03,
        enable_outlier_rejection: bool = True,
        saturation_threshold: float = 0.90,
        verbose: bool = True,
    ):
        """
        Initierar PhotometricStereoEstimator för en specifik kamera.

        Args:
            camera_id: 'left_cam1' (primär) eller 'right_cam0'.
            min_albedo: Tröskelvärde för albedo under vilket normaler betraktas som ogiltiga/skugga.
            enable_outlier_rejection: Om True aktiveras 4-source outlier rejection mot spegelblänk.
            saturation_threshold: Intensitetsnivå [0..1] för att flagga spegelblänk (standard 0.90, ca 230/255).
            verbose: Om True loggas mikrotider och eventuella varningar vid körning.
        """
        self.verbose = verbose
        self.camera_id = camera_id.lower()
        self.min_albedo = float(min_albedo)
        self.enable_outlier_rejection = bool(enable_outlier_rejection)
        self.saturation_threshold = float(saturation_threshold)

        if self.camera_id not in SUPPORTED_CAMERAS:
            raise ValueError(
                f"Okänt camera_id '{camera_id}'. Tillåtna kameror är: {list(SUPPORTED_CAMERAS)}"
            )

        # 1. Ladda kalibreringsdata
        if not CALIB_PATH.exists():
            raise FileNotFoundError(f"Fotometrisk kalibreringsfil hittades inte: {CALIB_PATH}")

        with open(CALIB_PATH, "r", encoding="utf-8") as f:
            calib_data = json.load(f)

        cam_data = calib_data["cameras"].get(self.camera_id)
        if not cam_data:
            raise KeyError(f"Kameradata för '{self.camera_id}' saknas i {CALIB_PATH}")

        # 2. Hämta och skala ljusvektorer med relativa intensiteter
        vecs = np.array(cam_data["light_vectors"], dtype=np.float32)       # (4, 3)
        intens = np.array(cam_data["light_intensities"], dtype=np.float32) # (4,)

        if vecs.shape != (4, 3):
            raise ValueError(f"Förväntade 4x3 ljusvektorer, fick {vecs.shape}")

        # Ljusmatris L: rad k är intensitet_k * vektor_k
        self.L = vecs * intens[:, None]  # (4, 3)

        # 3. Förberäkna Moore-Penrose pseudoinvers: L_pinv = (L^T * L)^-1 * L^T
        self.L_pinv = np.linalg.pinv(self.L).astype(np.float32)  # (3, 4)

        # 4. Förberäkna 3-ljus tripletter för outlier rejection (triplett k utesluter lampa k)
        # Triplett 0: lampor (1, 2, 3)
        # Triplett 1: lampor (0, 2, 3)
        # Triplett 2: lampor (0, 1, 3)
        # Triplett 3: lampor (0, 1, 2)
        self._triplet_indices = [
            np.array([1, 2, 3], dtype=np.int32),
            np.array([0, 2, 3], dtype=np.int32),
            np.array([0, 1, 3], dtype=np.int32),
            np.array([0, 1, 2], dtype=np.int32),
        ]
        self._triplet_pinvs = np.stack(
            [np.linalg.pinv(self.L[self._triplet_indices[k]]).astype(np.float32) for k in range(4)],
            axis=0,
        )  # (4, 3, 3)

    def _prepare_intensities(self, sweep: PhotometricSweep) -> np.ndarray:
        """
        Extraherar de 4 bildkanalerna för den valda kameran och staplar till (H, W, 4) float32 [0..1].
        """
        if self.camera_id == "left_cam1":
            frames = [sweep.ch1.left, sweep.ch2.left, sweep.ch3.left, sweep.ch4.left]
        else:
            frames = [sweep.ch1.right, sweep.ch2.right, sweep.ch3.right, sweep.ch4.right]

        images = []
        for ch_idx, frame in enumerate(frames, start=1):
            img = frame.image
            if img is None or img.size == 0:
                raise ValueError(f"Kanal {ch_idx} saknar giltig bildmatris för {self.camera_id}.")

            # Konvertera till 1-kanals gråskala om färg
            if len(img.shape) == 3 and img.shape[2] == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img

            images.append(gray.astype(np.float32) / 255.0)

        # Stapla till en (H, W, 4) matris
        return np.stack(images, axis=-1)

    def _reject_outliers(self, I: np.ndarray, g: np.ndarray) -> np.ndarray:
        """
        Identifierar och ersätter estimat för pixlar med EXAKT EN mättad lampa (spegelblänk).
        Om 2 eller fler lampor är mättade behålls standard 4-ljuslösningen (eftersom minst 3
        rena lampor krävs för en deterministisk 3-ljuslösning; pixelhantering sker då via fusion).
        """
        # Räkna mättade kanaler per pixel
        sat_counts = np.sum(I >= self.saturation_threshold, axis=-1)

        # Aktivera endast där EXAKT 1 kanal är mättad
        single_sat = (sat_counts == 1)

        if not np.any(single_sat):
            return g

        g_out = g.copy()

        sat_I = I[single_sat]
        worst_sat = np.argmax(sat_I, axis=-1)
        g_sat = np.empty((sat_I.shape[0], 3), dtype=np.float32)

        for k in range(4):
            k_sel = (worst_sat == k)
            if np.any(k_sel):
                I_sub = sat_I[k_sel][:, self._triplet_indices[k]]
                g_sat[k_sel] = I_sub @ self._triplet_pinvs[k].T

        g_out[single_sat] = g_sat
        return g_out

    def _compute_normals_and_albedo(
        self, I: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Löser Woodhams minsta-kvadratmetod g = L_pinv * I vektoriserat över alla pixlar.

        Returns:
            Tuple av (normals, albedo, valid_mask).
        """
        # Tensor-multiplikation: g(y, x, :) = L_pinv (3, 4) @ I(y, x, :) (4,)
        # Resulterande g har form (H, W, 3)
        g = np.tensordot(I, self.L_pinv, axes=([2], [1])).astype(np.float32)

        # 4-source outlier rejection (spegelblänk och skuggor)
        if self.enable_outlier_rejection:
            g = self._reject_outliers(I, g)

        # Albedo rho = ||g||
        albedo = np.linalg.norm(g, axis=2).astype(np.float32)

        # Validitetsmask: Albedo över tröskelvärde och finit
        valid_mask = (albedo >= self.min_albedo) & np.isfinite(albedo)

        # Normaliserad enhetsnormal N = g / ||g||
        normals = np.zeros_like(g)
        normals[valid_mask] = g[valid_mask] / albedo[valid_mask, None]

        return normals, albedo, valid_mask

    def _compute_gradients(
        self, normals: np.ndarray, valid_mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Beräknar ytans partiella lutningar p = dZ/dx = -nx/nz och q = dZ/dy = -ny/nz.
        """
        nz = normals[:, :, 2]
        # Undvik nolldelning där nz är mycket liten eller ytan är vertikal
        safe_nz = np.where(nz > 0.05, nz, 1.0)
        usable = valid_mask & (nz > 0.05)

        p = np.where(usable, -normals[:, :, 0] / safe_nz, 0.0).astype(np.float32)
        q = np.where(usable, -normals[:, :, 1] / safe_nz, 0.0).astype(np.float32)

        return p, q

    def estimate(self, sweep: PhotometricSweep) -> PhotometricResult:
        """
        Komplett fotometrisk stereopipeline.

        Args:
            sweep: Ett komplett och giltigt PhotometricSweep med bilder för Kanal 1-4.

        Returns:
            PhotometricResult med normaler, albedo, gradienter och exekveringstid.
            Vid ogiltigt svep eller beräkningsfel returneras PhotometricResult.invalid().
        """
        t0 = time.perf_counter()

        # 1. Strikt kontroll av inmatning
        if not isinstance(sweep, PhotometricSweep) or not sweep.is_valid():
            if self.verbose:
                print(f"[PhotometricStereoEstimator] Varning: Inmatat PhotometricSweep är ogiltigt.")
            return PhotometricResult.invalid()

        try:
            # 2. Förbered intensitetsmatris (H, W, 4)
            t_prep_start = time.perf_counter()
            I = self._prepare_intensities(sweep)
            t_prep_ms = (time.perf_counter() - t_prep_start) * 1000.0

            # 3. Lös normaler och albedo
            t_solve_start = time.perf_counter()
            normals, albedo, valid_mask = self._compute_normals_and_albedo(I)
            t_solve_ms = (time.perf_counter() - t_solve_start) * 1000.0

            # 4. Beräkna ytgradienter (p, q)
            t_grad_start = time.perf_counter()
            p, q = self._compute_gradients(normals, valid_mask)
            t_grad_ms = (time.perf_counter() - t_grad_start) * 1000.0

            exec_time_ms = (time.perf_counter() - t0) * 1000.0

            if self.verbose:
                print(
                    f"[PhotometricStereoEstimator] Klar på {exec_time_ms:.1f} ms "
                    f"(prep: {t_prep_ms:.1f} ms, solve: {t_solve_ms:.1f} ms, grad: {t_grad_ms:.1f} ms)"
                )

            return PhotometricResult(
                normals=normals,
                albedo=albedo,
                p_gradient=p,
                q_gradient=q,
                camera_id=self.camera_id,
                timestamp=time.time(),
                execution_time_ms=exec_time_ms,
                _is_valid=True,
            )
        except Exception as exc:
            if self.verbose:
                print(f"[PhotometricStereoEstimator] Fel vid beräkning: {exc}")
            return PhotometricResult.invalid()
