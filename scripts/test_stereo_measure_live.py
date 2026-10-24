"""
Live-test för snabbmätning av paket via 3D-stereo och automatiskt bordsplan.

Använder:
1. RANSAC-planpassning för att hitta bordets lutande plan (Z = A*X + B*Y + C).
2. 2D-höjdkarta där bordets yta nollställs (höjd = 0 mm).
3. Kantmaskning och morfologisk filtrering som eliminerar kantartefakter och isolerar paketet.
4. Konturanalys som hittar det verkliga paketet och mäter L x B x H i millimeter.
5. Sparar höjdkarta, disparitetskarta och bild med grön bounding box.

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
BORDER_MARGIN_PX = 80    # Ignorera bildens yttersta pixlar (eliminerar SGBM kantbrus)
MIN_CONTOUR_AREA_PX = 1500  # Minsta pixelarea för att betraktas som paket
OUTPUT_DIR = PROJECT_ROOT / "data" / "test_output" / "quick_measure"


def fit_table_plane_ransac(
    points: np.ndarray,
    threshold_mm: float = 6.0,
    iterations: int = 100,
    max_samples: int = 10000,
) -> Tuple[float, float, float]:
    """
    Hittar bordsplanet (Z = A*X + B*Y + C) robust med RANSAC.
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
    # Återanpassa med minsta-kvadratmetoden på inliers
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
    print("  Stereo 3D Snabbmätning (Live-test med Bordsplan & Segmentering)")
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

    # 4. Geometrianalys & Paketsegmentering
    print("\n[4/4] Analyserar bordsplan och isolerar paket...")
    valid_pts = depth_res.get_valid_points()
    total_pts_count = len(valid_pts)

    if total_pts_count < 1000:
        print("[FEL] För få giltiga 3D-punkter i punktmolnet.")
        return

    # A. Detektera bordsplanet
    A, B, C = fit_table_plane_ransac(valid_pts, threshold_mm=6.0)
    tilt_deg = float(np.degrees(np.arctan(np.sqrt(A**2 + B**2))))
    print(f"  Detekterat bordsplan: Z = {A:+.4f}*X {B:+.4f}*Y + {C:.1f} mm")
    print(f"  Bordsavstånd i centrum: {C:.1f} mm (Kamerans lutning: {tilt_deg:.1f}°)")

    # B. Beräkna 2D-höjdkarta (höjd över bordsplanet per pixel)
    pc = depth_res.point_cloud
    valid_mask = depth_res.get_valid_mask()
    h, w = valid_mask.shape

    z_plane = A * pc[:, :, 0] + B * pc[:, :, 1] + C
    height_map_mm = np.zeros((h, w), dtype=np.float32)
    valid_pixels = valid_mask & (pc[:, :, 2] > 0)
    height_map_mm[valid_pixels] = z_plane[valid_pixels] - pc[:, :, 2][valid_pixels]

    # C. Skapa ren binärmask: Tröskla höjd och klipp bildkanter
    raw_binary = (height_map_mm >= MIN_HEIGHT_MM) & (height_map_mm <= MAX_HEIGHT_MM)
    m = BORDER_MARGIN_PX
    clean_binary = np.zeros((h, w), dtype=np.uint8)
    clean_binary[m:h-m, m:w-m] = raw_binary[m:h-m, m:w-m].astype(np.uint8) * 255

    # D. Morfologisk öppning: Tar bort små brusöar och ocklusionsskuggor
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    clean_binary = cv2.morphologyEx(clean_binary, cv2.MORPH_OPEN, kernel)
    clean_binary = cv2.morphologyEx(clean_binary, cv2.MORPH_CLOSE, kernel)

    # E. Hitta sammanhängande objekt (konturer)
    contours, _ = cv2.findContours(clean_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= MIN_CONTOUR_AREA_PX]

    if not valid_contours:
        print("\n[INFO] Inget paket detekterades på bordet (> 15 mm).")
        print("Bordet är helt rent och isolerat!")
        return

    # Välj den största sammanhängande konturen (det huvudsakliga paketet/boken)
    package_contour = max(valid_contours, key=cv2.contourArea)
    contour_area_px = cv2.contourArea(package_contour)

    # Skapa mask enbart för det detekterade paketet
    pkg_mask_2d = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(pkg_mask_2d, [package_contour], -1, 255, -1)

    # Hämta 3D-punkter och höjder enbart INUTI paketet
    pkg_points_3d = pc[(pkg_mask_2d == 255) & valid_pixels]
    pkg_heights = height_map_mm[(pkg_mask_2d == 255) & valid_pixels]

    print(f"  Detekterat objekt:      {contour_area_px:,.0f} pixlar i yta")
    print(f"  Giltiga 3D-paketpunkter: {len(pkg_points_3d):,}")

    # Höjd (H): 99:e percentilen för att undvika enstaka brusspikar
    height_mm = float(np.percentile(pkg_heights, 99.0))

    # Längd och Bredd (L x B): 2D oriented bounding box på X/Y
    pts_2d = pkg_points_3d[:, :2].astype(np.float32)
    # Rensa bort 2% yttersta punkter innan rektangelpassning
    x_min, x_max = np.percentile(pts_2d[:, 0], [2.0, 98.0])
    y_min, y_max = np.percentile(pts_2d[:, 1], [2.0, 98.0])
    inliers_mask = (
        (pts_2d[:, 0] >= x_min) & (pts_2d[:, 0] <= x_max) &
        (pts_2d[:, 1] >= y_min) & (pts_2d[:, 1] <= y_max)
    )
    inlier_pts_2d = pts_2d[inliers_mask]

    rect = cv2.minAreaRect(inlier_pts_2d)
    _, (dim1, dim2), angle = rect
    length_mm = max(dim1, dim2)
    width_mm = min(dim1, dim2)

    # ==============================================================================
    # Spara pedagogiska felsökningsbilder
    # ==============================================================================
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Rå disparitetskarta
    depth_res.to_jpeg(OUTPUT_DIR, prefix="quick_")

    # 2. Färgkodad HÖJDKARTA (Där bordet faktiskt är 0 mm och mörkt!)
    height_vis = np.zeros((h, w), dtype=np.uint8)
    h_clamped = np.clip(height_map_mm, 0, 250.0)
    height_vis[valid_pixels] = (h_clamped[valid_pixels] / 250.0 * 255.0).astype(np.uint8)
    height_color = cv2.applyColorMap(height_vis, cv2.COLORMAP_JET)
    height_color[~valid_pixels] = (0, 0, 0)
    # Rita ut kontur på höjdkartan
    cv2.drawContours(height_color, [package_contour], -1, (255, 255, 255), 2)
    cv2.imwrite(str(OUTPUT_DIR / "quick_height_map.jpg"), height_color)

    # 3. Kamerabild med grön paketkontur
    rect_left = depth_res.rectified_left.copy() if depth_res.rectified_left is not None else None
    if rect_left is not None:
        cv2.drawContours(rect_left, [package_contour], -1, (0, 255, 0), 3)
        cv2.imwrite(str(OUTPUT_DIR / "quick_detection_overlay.jpg"), rect_left)

    print("\n" + "=" * 70)
    print("  UPPSKATTADE DIMENSIONER (Stereo 3D)")
    print("=" * 70)
    print(f"  Längd (L): {length_mm:6.1f} mm")
    print(f"  Bredd (B): {width_mm:6.1f} mm")
    print(f"  Höjd  (H): {height_mm:6.1f} mm")
    print("-" * 70)
    print(f"  Riktning (vinkel):       {angle:5.1f}°")
    print(f"  Total beräkningstid:     {total_time_ms:5.1f} ms")
    print(f"  Sparade bilder:")
    print(f"    Höjdkarta (Plan-nollställd): {OUTPUT_DIR / 'quick_height_map.jpg'}")
    print(f"    Kamerabild med kontur:       {OUTPUT_DIR / 'quick_detection_overlay.jpg'}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
