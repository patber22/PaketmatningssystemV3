"""
Fotometrisk kalibrering för Paketmätningssystem V3.

Beräknar ljusriktningsvektorer (L1..L4) och relativa lampstyrkor (s1..s4)
för båda kamerorna (CAM1 Vänster och CAM0 Höger) med hjälp av en bordtennisboll (40 mm).
Resultatet sparas till config/calibration/data/photometric.json.

Arkitektur:
1. Tar en bild med alla lampor tända samtidigt (led_mode="all") för jämn belysning.
2. Tar ett fotometriskt svep (Kanal 1 -> 2 -> 3 -> 4).
3. Detekterar sfären automatiskt (eller använder manuella koordinater vid behov).
4. Beräknar 3D-normalvektorer och relativa intensitetsskalor för alla 4 LED-kanaler.
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ==============================================================================
# 1. KONFIGURATION (Ändra vid behov)
# ==============================================================================
BALL_DIAMETER_MM: float = 40.0   # Standard bordtennisboll (40.0 mm)
BALL_RADIUS_PX: float = 40.5     # Nominell optisk radie i pixlar vid ~630 mm avstånd

# Manuella koordinater (cx, cy, radie_px). Sätt till None för automatisk detektering.
# Vid behov kan du låsa positionen manuellt här:
MANUAL_SPHERE_LEFT: Optional[Tuple[float, float, float]] = (869.0, 519.5, 40.5)
MANUAL_SPHERE_RIGHT: Optional[Tuple[float, float, float]] = (712.2, 547.2, 40.0)

# ==============================================================================
# SÖKVÄGAR OCH IMPORTER
# ==============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from src.core.models import PhotometricSweep
from src.hardware.vision_rig import VisionRig

OUTPUT_DIR = PROJECT_ROOT / "config" / "calibration" / "data"
OUTPUT_FILE = OUTPUT_DIR / "photometric.json"
DEBUG_IMG_DIR = PROJECT_ROOT / "data" / "calibration" / "photometric"


def find_sphere_circle(
    all_bgr: np.ndarray,
    camera_name: str,
    manual_sphere: Optional[Tuple[float, float, float]] = None,
) -> Optional[Tuple[float, float, float]]:
    """
    Hittar sfärens cirkel (cx, cy, radie) i bilden där alla lampor lyser samtidigt.
    Använder adaptiv tröskling och kontursökning anpassad för en 40 mm sfär.
    """
    if manual_sphere is not None:
        print(f"  -> {camera_name}: Använder manuellt angiven sfär: {manual_sphere}")
        return manual_sphere

    gray = cv2.cvtColor(all_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 1.5)
    h, w = gray.shape
    cx_nom, cy_nom = w // 2, h // 2
    margin_x, margin_y = 150, 100

    best_candidate = None
    best_score = -1e9

    # Tröskel-sweep kring bollens kantintensitet mot bädden
    for th_val in range(165, 186, 2):
        _, thresh = cv2.threshold(blur, th_val, 255, cv2.THRESH_BINARY)
        # Nollställ ytterkanter (bordskanter / bakgrund)
        thresh[:margin_y, :] = 0; thresh[-margin_y:, :] = 0
        thresh[:, :margin_x] = 0; thresh[:, -margin_x:] = 0

        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            area = cv2.contourArea(c)
            if not (2500 <= area <= 6500):
                continue
            perimeter = cv2.arcLength(c, True)
            if perimeter == 0:
                continue
            circ = 4 * np.pi * (area / (perimeter * perimeter))
            (cx, cy), radius = cv2.minEnclosingCircle(c)
            dist_to_center = np.hypot(cx - cx_nom, cy - cy_nom)

            if 34.0 <= radius <= 46.0 and circ > 0.55:
                # Hög cirkularitet och radie nära 40 px premieras
                r_diff = abs(radius - BALL_RADIUS_PX)
                score = (circ * 100.0) - (r_diff * 5.0) - (dist_to_center * 0.05)
                if score > best_score:
                    best_score = score
                    # Använd nominell BALL_RADIUS_PX om detekterad radie är nära
                    final_r = BALL_RADIUS_PX if 36.0 <= radius <= 44.0 else float(radius)
                    best_candidate = (float(cx), float(cy), final_r)

    return best_candidate


def compute_light_vectors_for_camera(
    images_gray: List[np.ndarray],
    cx: float,
    cy: float,
    radius: float,
    camera_name: str,
) -> Dict:
    """
    Beräknar de 4 ljusriktningsvektorerna (L1..L4) och relativa styrkor (s1..s4)
    för en kamera baserat på sfärens ytreflektioner i de 4 svepbilderna.
    """
    h, w = images_gray[0].shape
    y_coords, x_coords = np.ogrid[:h, :w]
    dist_sq = (x_coords - cx) ** 2 + (y_coords - cy) ** 2

    # Använd 92% av radien för att undvika skuggor precis vid bollens konturkant
    ball_mask = dist_sq <= (radius * 0.92) ** 2

    light_vectors = []
    flux_values = []
    highlight_points = []

    for ch_idx, img in enumerate(images_gray):
        masked_img = np.where(ball_mask, img, 0)
        valid_pixels = masked_img[ball_mask]

        if len(valid_pixels) == 0 or np.max(valid_pixels) < 10:
            raise ValueError(f"Ingen belysning detekterades på bollen för Kanal {ch_idx + 1} ({camera_name}).")

        # Hitta den ljusaste punkten (reflektionens topp / hotspot)
        # Använder tyngdpunkten för de 5% ljusaste pixlarna på bollen för brusimmunitet
        top_threshold = np.percentile(valid_pixels, 95)
        hotspot_mask = (masked_img >= top_threshold) & ball_mask

        M = cv2.moments(hotspot_mask.astype(np.uint8))
        if M["m00"] > 0:
            px = M["m10"] / M["m00"]
            py = M["m01"] / M["m00"]
        else:
            py, px = np.unravel_index(np.argmax(masked_img), masked_img.shape)

        highlight_points.append([float(px), float(py)])

        # Räkna ut sfärens 3D-normalvektor i den ljusaste punkten
        # (Nx, Ny pekar i bildplanet, Nz pekar rakt mot kameran)
        dx = (px - cx) / radius
        dy = (py - cy) / radius
        dz = np.sqrt(max(0.0, 1.0 - dx * dx - dy * dy))

        vec = np.array([dx, dy, dz], dtype=np.float64)
        norm = np.linalg.norm(vec)
        unit_l = vec / norm if norm > 0 else np.array([0.0, 0.0, 1.0])
        light_vectors.append(unit_l.tolist())

        # Total ljusflöde på sfären för relativ intensitet
        flux = float(np.sum(valid_pixels))
        flux_values.append(flux)

    # Normalisera intensiteterna kring 1.0
    mean_flux = np.mean(flux_values) if np.mean(flux_values) > 0 else 1.0
    intensities = [float(f / mean_flux) for f in flux_values]

    return {
        "sphere": {
            "center": [float(cx), float(cy)],
            "radius_px": float(radius),
        },
        "light_vectors": light_vectors,
        "light_intensities": intensities,
        "highlight_points": highlight_points,
    }


def main():
    print("=" * 65)
    print("  FOTOMETRISK KALIBRERING (Pingisboll / 4x IR LED)")
    print("=" * 65)
    print(f"Bolldiameter: {BALL_DIAMETER_MM:.1f} mm")
    print(f"Resultatfil:  {OUTPUT_FILE}\n")

    print("[1/4] Ansluter till VisionRig (kameror + 4-kanals LED)...")
    with VisionRig(verbose=False) as rig:
        if not rig.is_connected:
            print("[FEL] Kunde inte ansluta till hårdvaran i VisionRig.")
            return

        print("[OK] Hårdvara redo. Placera pingisbollen mitt på mätbädden.")
        print("     Tryck ENTER för att starta kalibreringen: ", end="")
        input()

        # [2/4] Bild med alla lampor tända för symmetrisk sfärdetektering
        print("\n[2/4] Tar bild med ALLA lampor tända (jämn sfärbelysning)...")
        all_pair = rig.capture_stereo(led_mode="all")
        if not all_pair.is_valid():
            print("[FEL] Kunde inte ta bild med alla lampor tända.")
            return

        # [3/4] Fotometriskt svep (Kanal 1 -> 2 -> 3 -> 4)
        print("[3/4] Tar fotometriskt svep (Kanal 1 -> 2 -> 3 -> 4)...")
        sweep = rig.capture_photometric_sweep()
        if not sweep.is_valid():
            print("[FEL] Fotometriska svepet misslyckades eller saknar kanaler.")
            return

    # Organisera bilderna per kamera
    cams_all = {
        "left_cam1": all_pair.left.image,
        "right_cam0": all_pair.right.image,
    }
    cams_sweep = {
        "left_cam1": [sweep.get_channel(ch).left.image for ch in [1, 2, 3, 4]],
        "right_cam0": [sweep.get_channel(ch).right.image for ch in [1, 2, 3, 4]],
    }

    DEBUG_IMG_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Spara alla råbilder omedelbart för full insyn
    for cam_name, img_all in cams_all.items():
        cv2.imwrite(str(DEBUG_IMG_DIR / f"raw_all_{cam_name}.jpg"), img_all)
    for cam_name, bgr_list in cams_sweep.items():
        for ch_idx, bgr in enumerate(bgr_list):
            cv2.imwrite(str(DEBUG_IMG_DIR / f"raw_sweep_{cam_name}_ch{ch_idx + 1}.jpg"), bgr)

    print("\n[4/4] Beräknar ljusvektorer för båda kamerorna...")
    calib_results = {
        "calibrated_at": datetime.now().isoformat(),
        "ball_diameter_mm": BALL_DIAMETER_MM,
        "cameras": {},
    }

    manual_spheres = {
        "left_cam1": MANUAL_SPHERE_LEFT,
        "right_cam0": MANUAL_SPHERE_RIGHT,
    }

    for cam_name in ["left_cam1", "right_cam0"]:
        img_all = cams_all[cam_name]
        bgr_sweep = cams_sweep[cam_name]
        grays_sweep = [cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) for im in bgr_sweep]

        circle = find_sphere_circle(img_all, cam_name, manual_spheres.get(cam_name))
        if circle is None:
            print(f"  [FEL] Kunde inte hitta pingisbollen i {cam_name}!")
            print(f"        Kontrollera {DEBUG_IMG_DIR / f'raw_all_{cam_name}.jpg'}.")
            print(f"        Tips: Du kan ange position manuellt via MANUAL_SPHERE_{'LEFT' if 'left' in cam_name else 'RIGHT'} i filens topp.")
            return

        cx, cy, radius = circle
        print(f"  -> {cam_name}: Sfär hittad på centrum ({cx:.1f}, {cy:.1f}), radie={radius:.1f} px")

        # Beräkna ljusvektorerna baserat på svepbilderna
        cam_calib = compute_light_vectors_for_camera(grays_sweep, cx, cy, radius, cam_name)
        calib_results["cameras"][cam_name] = cam_calib

        # Skapa en tydlig debug-bild på 'all_lit'-bilden med sfärcirkel och de 4 lampvektorerna
        debug_vis = img_all.copy()
        cv2.circle(debug_vis, (int(cx), int(cy)), int(radius), (0, 255, 0), 2)
        colors = [(0, 0, 255), (0, 255, 255), (255, 0, 0), (255, 0, 255)]
        for ch_idx, pt in enumerate(cam_calib["highlight_points"]):
            cv2.circle(debug_vis, (int(pt[0]), int(pt[1])), 4, colors[ch_idx], -1)
            cv2.putText(
                debug_vis,
                f"L{ch_idx + 1}",
                (int(pt[0]) + 6, int(pt[1]) - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                colors[ch_idx],
                1,
            )

        dbg_path = DEBUG_IMG_DIR / f"debug_{cam_name}.jpg"
        cv2.imwrite(str(dbg_path), debug_vis)

    # Spara resultatet till JSON
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(calib_results, f, indent=2)

    print("\n" + "=" * 65)
    print("  FOTOMETRISK KALIBRERING SPARAD!")
    print("=" * 65)
    print(f"Fil: {OUTPUT_FILE}")
    for cam_name, data in calib_results["cameras"].items():
        print(f"\n[{cam_name}] 3D-Ljusriktningar [Lx, Ly, Lz]:")
        for i, (vec, scale) in enumerate(zip(data["light_vectors"], data["light_intensities"])):
            print(f"  Lampa {i + 1}: [{vec[0]:6.3f}, {vec[1]:6.3f}, {vec[2]:6.3f}]  (Relativ styrka: {scale:.3f})")

    print(f"\nDebug-bilder sparade i: {DEBUG_IMG_DIR}")
    print("Klart för att bygga och köra Woodham fotometrisk stereo!\n")


if __name__ == "__main__":
    main()
