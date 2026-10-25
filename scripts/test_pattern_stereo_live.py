"""
Live-test för aktiv mönsterstereo (Active Structured Texture Test).

Tänder en specifik LED (t.ex. Topp-LED med mönstertub på kanal 4),
tar ett stereobildpar, beräknar SGBM-disparitet och 3D-mått (L x B x H),
och sparar råbilder samt diagnostik för att validera mönstrets täckning.

Användning:
    python scripts/test_pattern_stereo_live.py --channel 4
    python scripts/test_pattern_stereo_live.py --channel 4 --exposure 35000
"""

import argparse
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
from src.core.models import CameraFramePair

# ==============================================================================
# Konfiguration
# ==============================================================================
MIN_HEIGHT_MM = 10.0        # Minsta höjd över bordet (mm) för att räknas som paket
MAX_HEIGHT_MM = 350.0       # Max tillåten pakethöjd (mm)
BORDER_MARGIN_PX = 80       # Ignorera bildens yttersta pixlar (SGBM kantbrus)
MIN_CONTOUR_AREA_PX = 1500  # Minsta pixelarea för paket
OUTPUT_DIR = PROJECT_ROOT / "data" / "test_output" / "pattern_test"


def fit_table_plane_ransac(
    points: np.ndarray,
    threshold_mm: float = 6.0,
    iterations: int = 100,
    max_samples: int = 10000,
) -> Tuple[float, float, float]:
    """Hittar bordsplanet (Z = A*X + B*Y + C) med RANSAC."""
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
        raise RuntimeError("Kunde inte anpassa bordsplan.")

    normal, d = best_plane
    inliers_mask = np.abs(sample_pts @ normal + d) < threshold_mm
    inlier_pts = sample_pts[inliers_mask]

    X = inlier_pts[:, 0]
    Y = inlier_pts[:, 1]
    Z = inlier_pts[:, 2]
    A_mat = np.column_stack([X, Y, np.ones_like(X)])
    (A, B, C), _, _, _ = np.linalg.lstsq(A_mat, Z, rcond=None)

    return float(A), float(B), float(C)


def main():
    parser = argparse.ArgumentParser(description="Testa aktiv mönsterstereo live")
    parser.add_argument("--channel", type=int, default=4, help="LED-kanal att tända (1-4, standard: 4)")
    parser.add_argument("--exposure", type=int, default=30000, help="Kameraexponering i mikrosekunder (standard: 30000 = 30 ms)")
    parser.add_argument("--gain", type=float, default=3.5, help="Analog förstärkning (standard: 3.5)")
    args = parser.parse_args()

    print("=" * 70)
    print("  Aktiv Mönsterstereo - Live Test")
    print("=" * 70)
    print(f"  Mönster-LED Kanal: {args.channel}")
    print(f"  Slutartid:         {args.exposure / 1000.0:.1f} ms")
    print(f"  Analog Gain:       {args.gain:.1f}x")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Initiera stereodjupsberäknaren
    print("\n[1/4] Initierar StereoDepthEstimator (128..256 disp)...")
    estimator = StereoDepthEstimator(
        min_disparity=128,
        num_disparities=128,
        block_size=5,
        mode="fast",
        verbose=False,
    )

    # 2. Starta riggen, tänd ENBART mönster-LED:en, ta bildpar
    print(f"\n[2/4] Startar VisionRig och tänder enbart LED-kanal {args.channel}...")
    with VisionRig(verbose=False) as rig:
        if not rig.is_connected:
            print("[FEL] Kunde inte ansluta till kameror eller LED-kontroller.")
            return

        # Släck alla först, sätt exponering
        rig.leds.all_off()
        rig.set_exposure(exposure_us=args.exposure, gain=args.gain)
        time.sleep(0.05)

        # Tänd mönster-LED
        rig.leds.on(args.channel)
        time.sleep(0.05)

        # Ta bildpar (left=CAM1, right=CAM0)
        pair = rig.cameras.capture_both(samples=1)

        # Släck direkt efter exponering
        rig.leds.all_off()

    if not pair.is_valid():
        print("[FEL] Kunde inte läsa in giltigt stereobildpar.")
        return

    # Spara råbilder så användaren kan se hur mönstret ser ut
    cv2.imwrite(str(OUTPUT_DIR / "raw_left_cam1.png"), pair.left.data)
    cv2.imwrite(str(OUTPUT_DIR / "raw_right_cam0.png"), pair.right.data)
    print(f"  [Sparat] Råbild Vänster: {OUTPUT_DIR / 'raw_left_cam1.png'}")
    print(f"  [Sparat] Råbild Höger:   {OUTPUT_DIR / 'raw_right_cam0.png'}")

    # 3. Beräkna stereodjup
    print("\n[3/4] Beräknar 3D-stereodjup (SGBM)...")
    t0 = time.perf_counter()
    depth_res = estimator.estimate(pair)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    print(f"  SGBM beräkningstid: {elapsed_ms:.1f} ms")

    if not depth_res.is_valid():
        print("[FEL] Stereoberäkningen misslyckades.")
        return

    valid_pts = depth_res.get_valid_points()
    total_valid = len(valid_pts)
    print(f"  Totalt antal giltiga 3D-punkter: {total_valid:,}")

    if total_valid < 1000:
        print("[FEL] För få giltiga punkter i punktmolnet. Kontrollera att mönstret syns.")
        return

    # 4. Analysera bordsplan och segmentera paketet
    print("\n[4/4] Analyserar bordsplan och söker efter paket...")
    A, B, C = fit_table_plane_ransac(valid_pts, threshold_mm=6.0)
    tilt_deg = float(np.degrees(np.arctan(np.sqrt(A**2 + B**2))))
    print(f"  Detekterat bord: Z = {A:+.4f}*X {B:+.4f}*Y + {C:.1f} mm (Lutning: {tilt_deg:.1f}°)")

    pc = depth_res.point_cloud
    valid_mask = depth_res.get_valid_mask()
    h, w = valid_mask.shape

    z_plane = A * pc[:, :, 0] + B * pc[:, :, 1] + C
    height_map_mm = np.zeros((h, w), dtype=np.float32)
    valid_pixels = valid_mask & (pc[:, :, 2] > 0)
    height_map_mm[valid_pixels] = z_plane[valid_pixels] - pc[:, :, 2][valid_pixels]

    # Skapa binärmask
    raw_binary = (height_map_mm >= MIN_HEIGHT_MM) & (height_map_mm <= MAX_HEIGHT_MM)
    m = BORDER_MARGIN_PX
    clean_binary = np.zeros((h, w), dtype=np.uint8)
    clean_binary[m:h-m, m:w-m] = raw_binary[m:h-m, m:w-m].astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    clean_binary = cv2.morphologyEx(clean_binary, cv2.MORPH_OPEN, kernel)
    clean_binary = cv2.morphologyEx(clean_binary, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(clean_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= MIN_CONTOUR_AREA_PX]

    # Spara färgkodad disparitetskarta
    disp = depth_res.disparity
    disp_vis = np.zeros_like(disp, dtype=np.uint8)
    valid_d = disp > 0
    if np.any(valid_d):
        d_min, d_max = np.percentile(disp[valid_d], 2), np.percentile(disp[valid_d], 98)
        d_scaled = np.clip((disp - d_min) / max(d_max - d_min, 1e-3) * 255.0, 0, 255).astype(np.uint8)
        disp_color = cv2.applyColorMap(d_scaled, cv2.COLORMAP_TURBO)
        disp_color[~valid_d] = 0
    else:
        disp_color = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.imwrite(str(OUTPUT_DIR / "disparity_map.png"), disp_color)

    # Spara höjdkarta
    height_vis = np.zeros((h, w), dtype=np.uint8)
    h_clamped = np.clip(height_map_mm, 0, 250.0)
    height_vis[valid_pixels] = (h_clamped[valid_pixels] / 250.0 * 255.0).astype(np.uint8)
    height_color = cv2.applyColorMap(height_vis, cv2.COLORMAP_JET)
    height_color[~valid_pixels] = 0
    cv2.imwrite(str(OUTPUT_DIR / "height_map.png"), height_color)

    if not valid_contours:
        print("\n[INFO] Inget paket detekterades med höjd > 10 mm.")
        print(f"Se {OUTPUT_DIR / 'disparity_map.png'} för att se vad kameran såg!")
        return

    package_contour = max(valid_contours, key=cv2.contourArea)
    contour_area_px = cv2.contourArea(package_contour)

    pkg_mask_2d = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(pkg_mask_2d, [package_contour], -1, 255, -1)

    pkg_points_3d = pc[(pkg_mask_2d == 255) & valid_pixels]
    pkg_heights = height_map_mm[(pkg_mask_2d == 255) & valid_pixels]

    # Beräkna täckningsgrad (fill rate)
    total_pkg_pixels = int(np.count_nonzero(pkg_mask_2d == 255))
    fill_rate_pct = (len(pkg_heights) / max(total_pkg_pixels, 1)) * 100.0

    # Höjd: 99:e percentilen
    height_mm = float(np.percentile(pkg_heights, 99.0))
    mean_h_mm = float(np.mean(pkg_heights))

    # Längd & Bredd via minAreaRect
    pts_2d = pkg_points_3d[:, :2].astype(np.float32)
    hull = cv2.convexHull(pts_2d)
    rect = cv2.minAreaRect(hull)
    _, (dim1, dim2), angle = rect
    length_mm = max(dim1, dim2)
    width_mm = min(dim1, dim2)

    # Rita overlay-bild
    rect_left = depth_res.rectified_left.copy() if depth_res.rectified_left is not None else np.zeros((h, w, 3), dtype=np.uint8)
    cv2.drawContours(rect_left, [package_contour], -1, (0, 255, 0), 2)
    box_2d = cv2.minAreaRect(package_contour)
    box_pts = cv2.boxPoints(box_2d).astype(np.int32)
    cv2.drawContours(rect_left, [box_pts], -1, (0, 0, 255), 2)

    # Lägg text på bilden
    cv2.putText(
        rect_left,
        f"L:{length_mm:.1f} B:{width_mm:.1f} H_max:{height_mm:.1f}mm (Fill:{fill_rate_pct:.0f}%)",
        (40, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(OUTPUT_DIR / "detection_overlay.png"), rect_left)

    print("\n" + "=" * 70)
    print("  RESULTAT AV MÖNSTERTEST")
    print("=" * 70)
    print(f"  Yttäckning på paketet (Fill Rate): {fill_rate_pct:5.1f} %")
    print(f"  Giltiga 3D-punkter på paketet:     {len(pkg_points_3d):,} st")
    print("-" * 70)
    print(f"  Längd (L):      {length_mm:6.1f} mm")
    print(f"  Bredd (B):      {width_mm:6.1f} mm")
    print(f"  Höjd max (H):   {height_mm:6.1f} mm  (Snitthöjd: {mean_h_mm:.1f} mm)")
    print(f"  Vinkel:         {angle:5.1f}°")
    print("-" * 70)
    print(f"  Alla bilder sparade i: {OUTPUT_DIR}")
    print(f"    - raw_left_cam1.png     (Råbild vänster)")
    print(f"    - raw_right_cam0.png    (Råbild höger)")
    print(f"    - disparity_map.png     (Färgkodad djupkarta)")
    print(f"    - height_map.png        (Höjdkarta)")
    print(f"    - detection_overlay.png (Kamerabild med mått & box)")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
