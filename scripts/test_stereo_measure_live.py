"""
Live-test för snabbmätning av paket via 3D-stereo och automatiskt bordsplan.

Använder RANSAC-planpassning för att automatiskt detektera bordsytan,
vilket gör mätningen helt immun mot att kamerariggen lutar i förhållande till bordet.

Körs direkt på Raspberry Pi CM5 med:
    python3 scripts/test_stereo_measure_live.py
"""

import sys
import time
from pathlib import Path
from typing import Tuple
import cv2
import numpy as np

# Säkerställ att projektroten finns i sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.hardware.vision_rig import VisionRig
from src.algorithms.stereo_depth import StereoDepthEstimator

# ==============================================================================
# Konfiguration
# ==============================================================================
MIN_HEIGHT_MM = 15.0     # Minsta höjd över bordet (mm) för att räknas som paket
MAX_HEIGHT_MM = 350.0    # Max tillåten pakethöjd (mm)
OUTPUT_DIR = PROJECT_ROOT / "data" / "test_output" / "quick_measure"


def fit_table_plane_ransac(
    points: np.ndarray,
    threshold_mm: float = 6.0,
    iterations: int = 100,
    max_samples: int = 10000,
) -> Tuple[float, float, float]:
    """
    Hittar bordsplanet (Z = A*X + B*Y + C) robust med RANSAC.
    
    Gör mätningen helt oberoende av monteringsvinklar eller lutning mellan
    kameror och bordsskiva.
    """
    n_pts = len(points)
    sample_size = min(max_samples, n_pts)
    sample_idx = np.random.choice(n_pts, size=sample_size, replace=False)
    sample_pts = points[sample_idx]

    best_inliers = 0
    best_plane = None

    for _ in range(iterations):
        idx = np.random.choice(sample_size, 3, replace=False)
        p1, p2, p3 = sample_pts[idx]
        normal = np.cross(p2 - p1, p3 - p1)
        norm = np.linalg.norm(normal)
        if norm < 1e-6:
            continue
        normal /= norm
        if normal[2] < 0:
            normal = -normal
        d = -np.dot(normal, p1)

        dists = np.abs(sample_pts @ normal + d)
        inliers = int(np.count_nonzero(dists < threshold_mm))
        if inliers > best_inliers:
            best_inliers = inliers
            best_plane = (normal, d)

    if best_plane is None:
        raise RuntimeError("Kunde inte passa ett bordsplan mot punktmolnet.")

    normal, d = best_plane
    # Återanpassa med minsta-kvadratmetoden på alla inliers för maximal precision
    inliers_mask = np.abs(sample_pts @ normal + d) < threshold_mm
    inlier_pts = sample_pts[inliers_mask]

    X = inlier_pts[:, 0]
    Y = inlier_pts[:, 1]
    Z = inlier_pts[:, 2]
    A_mat = np.column_stack([X, Y, np.ones_like(X)])
    (A, B, C), _, _, _ = np.linalg.lstsq(A_mat, Z, rcond=None)

    return float(A), float(B), float(C)


def main():
    print("=" * 70)
    print("  Stereo 3D Snabbmätning (Live-test med Bordsplan)")
    print("=" * 70)

    # 1. Initiera stereodjupsberäknaren (optimerat sökintervall 128..256)
    print("[1/4] Initierar StereoDepthEstimator (min_disp=128, num_disp=128)...")
    estimator = StereoDepthEstimator(
        min_disparity=128,
        num_disparities=128,
        mode="fast",
        verbose=True,
    )

    # 2. Starta riggen och ta en bild med full IR-belysning
    print("\n[2/4] Startar VisionRig och tar stereobild (all LEDs on)...")
    with VisionRig(verbose=True) as rig:
        if not rig.is_connected:
            print("[FEL] Kunde inte ansluta till kameror och belysning.")
            return

        pair = rig.capture_stereo(led_mode="all")

    if not pair.is_valid():
        print("[FEL] Misslyckades med att fånga ett giltigt stereobildpar.")
        return

    # 3. Beräkna stereodjup och 3D-punktmoln
    print("\n[3/4] Beräknar 3D-stereodjup...")
    t_start = time.perf_counter()
    depth_res = estimator.estimate(pair)
    total_time_ms = (time.perf_counter() - t_start) * 1000.0

    if not depth_res.is_valid():
        print("[FEL] Stereodjupsberäkningen misslyckades.")
        return

    # 4. Automatisk bordsplansdetektering och paketsegmentering
    print("\n[4/4] Analyserar geometri och segmenterar paket...")
    valid_pts = depth_res.get_valid_points()
    total_pts_count = len(valid_pts)

    if total_pts_count < 1000:
        print("[FEL] För få giltiga 3D-punkter hittades i punktmolnet.")
        return

    # Detektera det fysiska bordsplanet med RANSAC
    A, B, C = fit_table_plane_ransac(valid_pts, threshold_mm=6.0)
    tilt_deg = float(np.degrees(np.arctan(np.sqrt(A**2 + B**2))))
    print(f"  Detekterat bordsplan: Z = {A:+.4f}*X {B:+.4f}*Y + {C:.1f} mm")
    print(f"  Bordsavstånd i centrum: {C:.1f} mm (Kamerans lutning: {tilt_deg:.1f}°)")

    # Beräkna höjden över bordet för varje punkt (planets Z minus punktens Z)
    z_plane = A * valid_pts[:, 0] + B * valid_pts[:, 1] + C
    heights_mm = z_plane - valid_pts[:, 2]

    # Maskera ut punkter som tillhör paketet (ovanför bordet)
    package_mask = (heights_mm >= MIN_HEIGHT_MM) & (heights_mm <= MAX_HEIGHT_MM)
    package_pts = valid_pts[package_mask]
    package_heights = heights_mm[package_mask]

    pkg_count = len(package_pts)
    table_count = int(np.count_nonzero(np.abs(heights_mm) < MIN_HEIGHT_MM))
    print(f"  Totalt antal 3D-punkter: {total_pts_count:,}")
    print(f"  Punkter på bordsytan:    {table_count:,} ({table_count/total_pts_count*100:.1f}%)")
    print(f"  Punkter på paketet:      {pkg_count:,} ({pkg_count/total_pts_count*100:.1f}%)")

    if pkg_count < 100:
        print("\n[INFO] Inget paket detekterades på bordet (> 15 mm).")
        print("Bordsytan är helt ren och isolerad!")
        return

    # A. Höjd: 99:e percentilen för att eliminera enstaka brus-spikar
    height_mm = float(np.percentile(package_heights, 99.0))

    # B. Längd och Bredd: 2D oriented bounding box på X/Y i millimeter
    pts_2d = package_pts[:, :2].astype(np.float32)

    # Rensa bort extrema 2D-outliers innan bounding box (2:a till 98:e percentilen)
    x_min, x_max = np.percentile(pts_2d[:, 0], [2.0, 98.0])
    y_min, y_max = np.percentile(pts_2d[:, 1], [2.0, 98.0])
    inlier_mask = (
        (pts_2d[:, 0] >= x_min) & (pts_2d[:, 0] <= x_max) &
        (pts_2d[:, 1] >= y_min) & (pts_2d[:, 1] <= y_max)
    )
    inlier_pts_2d = pts_2d[inlier_mask]

    rect = cv2.minAreaRect(inlier_pts_2d)
    _, (dim1, dim2), angle = rect
    length_mm = max(dim1, dim2)
    width_mm = min(dim1, dim2)

    # Spara felsökningsbilder
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    saved_paths = depth_res.to_jpeg(OUTPUT_DIR, prefix="quick_")

    print("\n" + "=" * 70)
    print("  UPPSKATTADE DIMENSIONER (Stereo 3D)")
    print("=" * 70)
    print(f"  Längd (L): {length_mm:6.1f} mm")
    print(f"  Bredd (B): {width_mm:6.1f} mm")
    print(f"  Höjd  (H): {height_mm:6.1f} mm")
    print("-" * 70)
    print(f"  Riktning (vinkel):       {angle:5.1f}°")
    print(f"  Total beräkningstid:     {total_time_ms:5.1f} ms")
    if saved_paths.get("disparity"):
        print(f"  Färgkodad djupkarta:     {saved_paths['disparity']}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
