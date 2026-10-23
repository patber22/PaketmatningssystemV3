"""
Stereokalibrering för NoIR-kamerorna (CAM1 Vänster och CAM0 Höger).

Instruktioner:
1. Ändra variablerna nedan (rader, kolumner, rutstorlek) så att de matchar din bild på iPaden.
2. Kör skriptet på pajen:
    python3 config/calibration/calibrate_stereo.py
3. Håll upp iPaden under kamerorna. När båda kamerorna ser mönstret trycker du Enter.
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

# ==============================================================================
# 1. KONFIGURATION (Ändra dessa direkt i filen så det matchar din iPad-bild)
# ==============================================================================
CHESSBOARD_COLS: int = 9         # Antal INRE hörn horisontellt
CHESSBOARD_ROWS: int = 6         # Antal INRE hörn vertikalt
SQUARE_SIZE_MM: float = 19.5     # Rutans bredd/höjd i millimeter (mät på din skärm)
TARGET_PAIRS: int = 15           # Antal bildpar att samla in

# Exponering för avläsning av skärm i rumsljus (släckta IR-lampor)
EXPOSURE_US: int = 40000         # 40 ms
GAIN: float = 4.0

# ==============================================================================
# SÖKVÄGAR OCH IMPORTER
# ==============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
from src.hardware.vision_rig import VisionRig

OUTPUT_DIR = PROJECT_ROOT / "config" / "calibration" / "data"
OUTPUT_FILE = OUTPUT_DIR / "stereo.json"
IMAGE_SAVE_DIR = PROJECT_ROOT / "data" / "calibration" / "stereo" / time.strftime("%Y%m%d_%H%M%S")


def find_corners(gray_img: np.ndarray, pattern_size: tuple):
    """Söker efter inre schackrutshörn och förfinar med subpixelprecision."""
    flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH
        + cv2.CALIB_CB_NORMALIZE_IMAGE
        + cv2.CALIB_CB_FAST_CHECK
    )
    found, corners = cv2.findChessboardCorners(gray_img, pattern_size, flags)
    if found and corners is not None:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
        corners_subpix = cv2.cornerSubPix(gray_img, corners, (11, 11), (-1, -1), criteria)
        return True, corners_subpix
    return False, None


def main():
    pattern_size = (CHESSBOARD_COLS, CHESSBOARD_ROWS)

    print("=" * 65)
    print("  STEREOKALIBRERING: NoIR-kameror")
    print("=" * 65)
    print(f"Mönster:      {CHESSBOARD_COLS}x{CHESSBOARD_ROWS} inre hörn")
    print(f"Rutstorlek:   {SQUARE_SIZE_MM:.1f} mm")
    print(f"Mål:          {TARGET_PAIRS} bildpar")
    print(f"Resultatfil:  {OUTPUT_FILE}\n")

    IMAGE_SAVE_DIR.mkdir(parents=True, exist_ok=True)

    print("Ansluter till kameror (släckta IR-lampor)...")
    with VisionRig(
        stereo_exposure_us=EXPOSURE_US,
        stereo_gain=GAIN,
        verbose=False,
    ) as rig:
        if not rig.is_connected:
            print("[FEL] Kunde inte ansluta till kamerorna.")
            return

        rig.leds.all_off()
        print("[OK] Redo! Håll upp iPaden under kamerorna i olika vinklar.\n")

        image_pairs = []
        pair_count = 0

        try:
            while pair_count < TARGET_PAIRS:
                # Ta ambient-bildpar
                pair = rig.capture_stereo(led_mode="ambient")
                if not pair.is_valid():
                    time.sleep(0.05)
                    continue

                img_l = pair.left.image
                img_r = pair.right.image

                gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
                gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

                found_l, _ = find_corners(gray_l, pattern_size)
                found_r, _ = find_corners(gray_r, pattern_size)

                status_l = "HITTAD" if found_l else "söker..."
                status_r = "HITTAD" if found_r else "söker..."
                ready_text = " [REDO! Tryck ENTER]" if (found_l and found_r) else ""

                # Skriv status på samma rad i terminalen
                sys.stdout.write(
                    f"\rSparade par: {pair_count}/{TARGET_PAIRS}  |  "
                    f"CAM1 (V): {status_l:8s}  |  CAM0 (H): {status_r:8s}{ready_text}   "
                )
                sys.stdout.flush()

                # När båda kamerorna ser mönstret, ge användaren möjlighet att spara
                if found_l and found_r:
                    print(f"\n[OK] Båda ser mönstret! Tryck ENTER för att spara (eller 'q' för att avsluta): ", end="")
                    cmd = input().strip().lower()
                    if cmd == "q":
                        print("Avslutar insamlingen.")
                        break

                    pair_count += 1
                    # Spara bilderna till disk
                    l_path = IMAGE_SAVE_DIR / f"pair_{pair_count:02d}_CAM1_left.jpg"
                    r_path = IMAGE_SAVE_DIR / f"pair_{pair_count:02d}_CAM0_right.jpg"
                    cv2.imwrite(str(l_path), img_l)
                    cv2.imwrite(str(r_path), img_r)

                    image_pairs.append((gray_l, gray_r))
                    print(f"  -> Sparade par {pair_count} ({l_path.name})")

                time.sleep(0.04)

        except KeyboardInterrupt:
            print("\nAvbröt med Ctrl+C.")

    if len(image_pairs) < 5:
        print(f"\n[AVBRUTET] För få bildpar ({len(image_pairs)} st). Behöver minst 5 för kalibrering.")
        return

    # ==============================================================================
    # 2. BERÄKNA KALIBRERINGEN
    # ==============================================================================
    print("\n" + "=" * 65)
    print(f"Beräknar stereokalibrering ur {len(image_pairs)} bildpar...")
    print("=" * 65)

    # Skapa 3D-koordinater för mönstrets inre hörn (Z = 0)
    objp = np.zeros((CHESSBOARD_ROWS * CHESSBOARD_COLS, 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESSBOARD_COLS, 0:CHESSBOARD_ROWS].T.reshape(-1, 2) * SQUARE_SIZE_MM

    objpoints = []
    imgpoints_l = []
    imgpoints_r = []

    h, w = image_pairs[0][0].shape[:2]
    img_size = (w, h)

    for gray_l, gray_r in image_pairs:
        ok_l, corners_l = find_corners(gray_l, pattern_size)
        ok_r, corners_r = find_corners(gray_r, pattern_size)
        if ok_l and ok_r:
            objpoints.append(objp)
            imgpoints_l.append(corners_l)
            imgpoints_r.append(corners_r)

    # 1. Individuell kalibrering
    print("[1/3] Kalibrerar individuella kameror...")
    ret_l, K_l, D_l, _, _ = cv2.calibrateCamera(objpoints, imgpoints_l, img_size, None, None)
    ret_r, K_r, D_r, _, _ = cv2.calibrateCamera(objpoints, imgpoints_r, img_size, None, None)
    print(f"  CAM1 (Vänster) RMS fel: {ret_l:.3f} px")
    print(f"  CAM0 (Höger)   RMS fel: {ret_r:.3f} px")

    # 2. Stereo extrinsics (R, T)
    print("\n[2/3] Beräknar stereoparametrar (R, T)...")
    stereo_flags = cv2.CALIB_FIX_INTRINSIC
    ret_s, K_l, D_l, K_r, D_r, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints_l, imgpoints_r,
        K_l, D_l, K_r, D_r, img_size,
        flags=stereo_flags
    )
    baseline_mm = float(np.linalg.norm(T))
    print(f"  Stereo RMS fel : {ret_s:.3f} px")
    print(f"  Kamerabaslinje : {baseline_mm:.1f} mm")

    # 3. Rektifiering (R1, R2, P1, P2, Q)
    print("\n[3/3] Beräknar rektifieringsmatriser (Q)...")
    R1, R2, P1, P2, Q, roi_l, roi_r = cv2.stereoRectify(
        K_l, D_l, K_r, D_r, img_size, R, T,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0.0
    )

    # 4. Spara till JSON
    calib_result = {
        "calibrated_at": datetime.now().isoformat(),
        "resolution": [w, h],
        "pattern": {
            "cols": CHESSBOARD_COLS,
            "rows": CHESSBOARD_ROWS,
            "square_size_mm": SQUARE_SIZE_MM,
            "pairs_used": len(objpoints),
        },
        "rms_errors": {
            "stereo_rms_px": float(ret_s),
            "left_rms_px": float(ret_l),
            "right_rms_px": float(ret_r),
        },
        "baseline_distance_mm": baseline_mm,
        "cameras": {
            "left_cam1": {
                "camera_matrix_K": K_l.tolist(),
                "dist_coeffs_D": D_l.flatten().tolist(),
                "rectification_R1": R1.tolist(),
                "projection_P1": P1.tolist(),
                "roi": list(roi_l),
            },
            "right_cam0": {
                "camera_matrix_K": K_r.tolist(),
                "dist_coeffs_D": D_r.flatten().tolist(),
                "rectification_R2": R2.tolist(),
                "projection_P2": P2.tolist(),
                "roi": list(roi_r),
            },
        },
        "extrinsics": {
            "rotation_matrix_R": R.tolist(),
            "translation_vector_T_mm": T.flatten().tolist(),
            "essential_matrix_E": E.tolist(),
            "fundamental_matrix_F": F.tolist(),
        },
        "disparity_to_depth_Q": Q.tolist(),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(calib_result, f, indent=2)

    print("\n" + "=" * 65)
    print("  KALIBRERINGEN SPARAD!")
    print(f"  Fil: {OUTPUT_FILE}")
    print("=" * 65)


if __name__ == "__main__":
    main()
