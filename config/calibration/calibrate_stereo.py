"""
Kalibreringsverktyg för stereokamerorna (CAM1 Vänster och CAM0 Höger).

Beräknar kamerornas inre parametrar (K1, K2, D1, D2), det fysiska avståndet/vinkeln
mellan dem (R, T) samt stereorektifierings- och projektionsmatriserna (R1, R2, P1, P2, Q).
Resultatet sparas till config/calibration/data/stereo.json.

Användning på Raspberry Pi:
1. Skapa ett mönster att visa på iPad (om du inte redan har ett):
    python3 config/calibration/calibrate_stereo.py --generate-pattern

2. Starta interaktiv kalibrering med iPad / schackbräde:
    python3 config/calibration/calibrate_stereo.py --square-size 25.0 --cols 9 --rows 6

3. Kalibrera från en tidigare sparad bildmapp:
    python3 config/calibration/calibrate_stereo.py --image-dir data/calibration/stereo/20261023_...
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Säkerställ att projektroten finns i sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from src.hardware.vision_rig import VisionRig

CALIBRATION_DATA_DIR = PROJECT_ROOT / "config" / "calibration" / "data"
OUTPUT_JSON_PATH = CALIBRATION_DATA_DIR / "stereo.json"
DEFAULT_IMAGE_SAVE_DIR = PROJECT_ROOT / "data" / "calibration" / "stereo"


def generate_pattern_image(
    cols: int = 9,
    rows: int = 6,
    square_pixels: int = 120,
    output_path: Optional[Path] = None,
) -> Path:
    """
    Genererar en högupplöst PNG-bild med ett schackmönster för visning på iPad.
    Cols och rows avser antalet INRE hörn (antalet rutor är cols+1 och rows+1).
    """
    out = output_path or (PROJECT_ROOT / "data" / "checkerboard_pattern.png")
    out.parent.mkdir(parents=True, exist_ok=True)

    total_cols = cols + 1
    total_rows = rows + 1
    border_squares = 1

    img_w = (total_cols + 2 * border_squares) * square_pixels
    img_h = (total_rows + 2 * border_squares) * square_pixels

    # Vit bakgrund med marginal
    img = np.full((img_h, img_w), 255, dtype=np.uint8)

    for r in range(total_rows):
        for c in range(total_cols):
            if (r + c) % 2 == 1:
                y1 = (r + border_squares) * square_pixels
                y2 = y1 + square_pixels
                x1 = (c + border_squares) * square_pixels
                x2 = x1 + square_pixels
                img[y1:y2, x1:x2] = 0

    cv2.imwrite(str(out), img)
    return out


def find_corners(
    img_gray: np.ndarray,
    pattern_size: Tuple[int, int],
) -> Tuple[bool, Optional[np.ndarray]]:
    """
    Söker efter inre schackrutshörn och förfinar dem med subpixel-precision.
    """
    flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH
        + cv2.CALIB_CB_NORMALIZE_IMAGE
        + cv2.CALIB_CB_FAST_CHECK
    )
    found, corners = cv2.findChessboardCorners(img_gray, pattern_size, flags)
    if found and corners is not None:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
        corners_subpix = cv2.cornerSubPix(
            img_gray, corners, (11, 11), (-1, -1), criteria
        )
        return True, corners_subpix
    return False, None


def run_interactive_capture(
    rig: VisionRig,
    pattern_size: Tuple[int, int],
    save_dir: Path,
    target_pairs: int = 20,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Interaktiv insamling i terminalen:
    Visar realtidsstatus för hörndetektion i båda kamerorna.
    När mönstret syns i båda kamerorna trycker användaren Enter för att spara paret.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    captured_pairs: List[Tuple[np.ndarray, np.ndarray]] = []

    print("\n" + "=" * 70)
    print("  INTERAKTIV BILDINSAMLING FÖR STEREOKALIBRERING")
    print("=" * 70)
    print(f"Mönster: {pattern_size[0]}x{pattern_size[1]} inre hörn.")
    print("Instruktioner:")
    print(" 1. Håll upp schackbrädet / iPaden under kamerorna i olika vinklar:")
    print("    - Plant mot mätbädden i olika positioner (mitt, hörn).")
    print("    - Lätt tiltat (~10-20 grader) åt vänster, höger, framåt och bakåt.")
    print("    - På olika höjder (bordsnivå upp till ~20 cm över bädden).")
    print(" 2. Tryck ENTER när båda kamerorna indikerar [HITTAD].")
    print(" 3. Skriv 'q' och tryck Enter när du samlat tillräckligt med par.\n")

    pair_idx = 0
    try:
        while len(captured_pairs) < target_pairs:
            # Ta ett ambient-stereopar (släckta lampor, ljust för skärm/rumsljus)
            frame_pair = rig.capture_stereo(led_mode="ambient")
            if not frame_pair.is_valid():
                time.sleep(0.05)
                continue

            img_l = frame_pair.left.image
            img_r = frame_pair.right.image

            gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
            gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

            found_l, _ = find_corners(gray_l, pattern_size)
            found_r, _ = find_corners(gray_r, pattern_size)

            status_l = "HITTAD" if found_l else "söker..."
            status_r = "HITTAD" if found_r else "söker..."
            both_ready = " [REDO! Tryck ENTER]" if (found_l and found_r) else ""

            # Uppdatera status på samma rad
            sys.stdout.write(
                f"\rInsamlade: {len(captured_pairs)}/{target_pairs}  |  "
                f"CAM1 (V): {status_l:8s}  |  CAM0 (H): {status_r:8s}{both_ready}   "
            )
            sys.stdout.flush()

            # Om båda hittades, ge användaren en chans att spara via tangenttryck
            if found_l and found_r:
                print(f"\n[OK] Par {pair_idx + 1} redo! Tryck ENTER för att spara (eller 'q' för att avbryta): ", end="")
                cmd = input().strip().lower()
                if cmd == "q":
                    print("Avslutar insamling.")
                    break

                # Spara bilderna till disk
                pair_idx += 1
                l_path = save_dir / f"pair_{pair_idx:02d}_CAM1_left.jpg"
                r_path = save_dir / f"pair_{pair_idx:02d}_CAM0_right.jpg"
                cv2.imwrite(str(l_path), img_l)
                cv2.imwrite(str(r_path), img_r)
                captured_pairs.append((gray_l, gray_r))
                print(f"  -> Sparat par {pair_idx} till disk ({l_path.name})")

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\nAvbröt insamling via Ctrl+C.")

    return captured_pairs


def calibrate_stereo_from_pairs(
    image_pairs: List[Tuple[np.ndarray, np.ndarray]],
    pattern_size: Tuple[int, int],
    square_size_mm: float,
) -> Dict:
    """
    Genomför fullständig stereokalibrering med OpenCV:
    1. calibrateCamera på Left och Right individuellt.
    2. stereoCalibrate för extrinsics (R, T, E, F).
    3. stereoRectify för rektifieringsmatriser (R1, R2, P1, P2, Q).
    """
    cols, rows = pattern_size
    # 3D punkter för schackbrädets inre hörn i mönstrets lokala koordinatsystem (Z = 0)
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size_mm

    objpoints: List[np.ndarray] = []
    imgpoints_l: List[np.ndarray] = []
    imgpoints_r: List[np.ndarray] = []

    print("\n" + "-" * 70)
    print(f"Analyserar {len(image_pairs)} stereobildpar för kalibrering...")
    print("-" * 70)

    h, w = image_pairs[0][0].shape[:2]
    img_size = (w, h)

    valid_count = 0
    for idx, (gray_l, gray_r) in enumerate(image_pairs):
        found_l, corners_l = find_corners(gray_l, pattern_size)
        found_r, corners_r = find_corners(gray_r, pattern_size)

        if found_l and found_r and corners_l is not None and corners_r is not None:
            objpoints.append(objp)
            imgpoints_l.append(corners_l)
            imgpoints_r.append(corners_r)
            valid_count += 1
            print(f"  Par {idx + 1:02d}: Hörn hittade och verifierade.")
        else:
            print(f"  Par {idx + 1:02d}: Kunde inte hitta alla hörn i båda kamerorna (hoppas över).")

    if valid_count < 5:
        raise ValueError(
            f"Endast {valid_count} giltiga par hittades. Minst 10 rekommenderas för god precision."
        )

    print(f"\n[1/3] Kalibrerar individuella kameror ({valid_count} par)...")
    calib_flags = cv2.CALIB_RATIONAL_MODEL
    ret_l, K_l, D_l, _, _ = cv2.calibrateCamera(
        objpoints, imgpoints_l, img_size, None, None, flags=calib_flags
    )
    ret_r, K_r, D_r, _, _ = cv2.calibrateCamera(
        objpoints, imgpoints_r, img_size, None, None, flags=calib_flags
    )
    print(f"  CAM1 (Vänster) RMS: {ret_l:.3f} px")
    print(f"  CAM0 (Höger)   RMS: {ret_r:.3f} px")

    print("\n[2/3] Kör stereoCalibrate för extrinsics (R, T)...")
    stereo_flags = cv2.CALIB_FIX_INTRINSIC
    criteria_stereo = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-6)

    ret_stereo, K_l, D_l, K_r, D_r, R, T, E, F = cv2.stereoCalibrate(
        objpoints,
        imgpoints_l,
        imgpoints_r,
        K_l,
        D_l,
        K_r,
        D_r,
        img_size,
        criteria=criteria_stereo,
        flags=stereo_flags,
    )
    baseline_mm = float(np.linalg.norm(T))
    print(f"  Stereo RMS reprojection error: {ret_stereo:.3f} px")
    print(f"  Beräknad baslinje (avstånd mellan kameror): {baseline_mm:.1f} mm")
    print(f"  Translationsvektor T (X, Y, Z mm): [{T[0][0]:.1f}, {T[1][0]:.1f}, {T[2][0]:.1f}]")

    print("\n[3/3] Beräknar rektifiering (R1, R2, P1, P2, Q)...")
    R1, R2, P1, P2, Q, roi_l, roi_r = cv2.stereoRectify(
        K_l, D_l, K_r, D_r, img_size, R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0.0
    )

    calibration_dict = {
        "calibrated_at": datetime.now().isoformat(),
        "resolution": [w, h],
        "pattern": {
            "cols": cols,
            "rows": rows,
            "square_size_mm": square_size_mm,
            "valid_pairs": valid_count,
        },
        "rms_errors": {
            "stereo_rms_px": float(ret_stereo),
            "left_rms_px": float(ret_l),
            "right_rms_px": float(ret_r),
        },
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
            "baseline_distance_mm": baseline_mm,
            "essential_matrix_E": E.tolist(),
            "fundamental_matrix_F": F.tolist(),
        },
        "disparity_to_depth_Q": Q.tolist(),
    }

    return calibration_dict


def main():
    parser = argparse.ArgumentParser(
        description="Kalibreringsverktyg för stereokameror (NoIR CAM1 och CAM0)."
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=9,
        help="Antal inre hörn horisontellt (standard: 9).",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=6,
        help="Antal inre hörn vertikalt (standard: 6).",
    )
    parser.add_argument(
        "--square-size",
        type=float,
        default=25.0,
        help="Rutstorlek i millimeter på mönstret (standard: 25.0 mm).",
    )
    parser.add_argument(
        "--exposure",
        type=int,
        default=40000,
        help="Slutartid i µs för kalibreringen i rumsljus/iPad (standard: 40000 µs = 40 ms).",
    )
    parser.add_argument(
        "--gain",
        type=float,
        default=4.0,
        help="Analog förstärkning för kalibreringen (standard: 4.0).",
    )
    parser.add_argument(
        "--pairs",
        type=int,
        default=15,
        help="Önskat antal bildpar att samla in (standard: 15).",
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default=None,
        help="Sökväg till tidigare sparad bildmapp med bildpar för offline-kalibrering.",
    )
    parser.add_argument(
        "--generate-pattern",
        action="store_true",
        help="Generera en bild med schackmönster för visning på skärm/iPad och avsluta.",
    )

    args = parser.parse_args()

    # Läge 1: Generera bildfil för iPad
    if args.generate_pattern:
        out = generate_pattern_image(cols=args.cols, rows=args.rows)
        print(f"[OK] Schackmönster skapat: {out}")
        print("Öppna denna bild i fullskärm på din iPad och mät en ruta med linjal.")
        return

    pattern_size = (args.cols, args.rows)

    # Läge 2: Kalibrera från befintlig bildmapp
    if args.image_dir:
        img_dir = Path(args.image_dir)
        if not img_dir.exists():
            print(f"[FEL] Mappen {img_dir} existerar inte.")
            return

        left_files = sorted(list(img_dir.glob("*_CAM1_left.jpg")))
        right_files = sorted(list(img_dir.glob("*_CAM0_right.jpg")))

        if len(left_files) != len(right_files) or len(left_files) == 0:
            print(f"[FEL] Hittade inte matchande bildpar i {img_dir}.")
            return

        image_pairs = []
        for l_p, r_p in zip(left_files, right_files):
            img_l = cv2.imread(str(l_p), cv2.IMREAD_GRAYSCALE)
            img_r = cv2.imread(str(r_p), cv2.IMREAD_GRAYSCALE)
            image_pairs.append((img_l, img_r))

    # Läge 3: Live insamling via VisionRig
    else:
        ts_str = time.strftime("%Y%m%d_%H%M%S")
        save_dir = DEFAULT_IMAGE_SAVE_DIR / ts_str
        print(f"Initierar kameror i ambient läge (släckta lampor, exp={args.exposure}µs, gain={args.gain})...")

        with VisionRig(
            stereo_exposure_us=args.exposure,
            stereo_gain=args.gain,
            verbose=False,
        ) as rig:
            if not rig.is_connected:
                print("[FEL] Kunde inte ansluta till kameror.")
                return

            rig.leds.all_off()
            image_pairs = run_interactive_capture(
                rig=rig,
                pattern_size=pattern_size,
                save_dir=save_dir,
                target_pairs=args.pairs,
            )

    if not image_pairs:
        print("[AVBRUTET] Inga bildpar samlades in.")
        return

    # Beräkna kalibrering
    try:
        calib_data = calibrate_stereo_from_pairs(
            image_pairs=image_pairs,
            pattern_size=pattern_size,
            square_size_mm=args.square_size,
        )

        CALIBRATION_DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(calib_data, f, indent=2)

        print("\n" + "=" * 70)
        print("  KALIBRERINGEN SPARAD OCH SLUTFÖRD!")
        print("=" * 70)
        print(f"Fil: {OUTPUT_JSON_PATH}")
        print(f"Stereo RMS Fel : {calib_data['rms_errors']['stereo_rms_px']:.3f} pixlar")
        print(f"Kamerabaslinje : {calib_data['extrinsics']['baseline_distance_mm']:.1f} mm")
        print("Klar för 3D-djupberäkning och rektifiering!\n")

    except Exception as e:
        print(f"\n[FEL] Kalibreringen misslyckades: {e}")


if __name__ == "__main__":
    main()
