"""
Fotometrisk kalibrering för Paketmätningssystem V3.

Beräknar ljusriktningsvektorer (L1..L4) och relativa lampstyrkor (s1..s4)
för båda kamerorna (CAM1 Vänster och CAM0 Höger) med hjälp av en vit pingisboll
placerad på ett mörkt underlag (t.ex. 3D-skrivarens byggplatta).

Resultatet sparas till config/calibration/data/photometric.json.

Arkitektur:
1. Tar en bild med alla lampor tända samtidigt (led_mode="all") för jämn belysning.
2. Tvåstegsdetektering:
   - Mask 1: Identifierar det mörka underlaget och klipper bort den ljusa OSB-skivan.
   - Mask 2: Lokaliserar den vita bollen inuti underlaget och mäter dess centrum och radie dynamiskt.
3. Tar ett fotometriskt svep (Kanal 1 -> 2 -> 3 -> 4) och beräknar lampornas 3D-vektorer och styrkor.
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
) -> Tuple[Optional[Tuple[float, float, float]], Optional[Tuple[int, int, int, int]]]:
    """
    Tvåstegsdetektering av pingisbollen:
    1. Mask 1: Hittar det mörka underlaget (t.ex. svart byggplatta) och klipper bort OSB-bädden.
    2. Mask 2: Hittar den vita bollen inuti underlaget och mäter dess verkliga centrum och radie.
    """
    gray = cv2.cvtColor(all_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 2)

    # --------------------------------------------------------------------------
    # STEG 1: Hitta mörkt underlag (byggplatta/musmatta)
    # OSB-skivan är ljus (~130-150), mörk byggplatta är < 95
    # --------------------------------------------------------------------------
    plate_cnt = None
    bbox_dark = None
    plate_mask = np.zeros_like(gray, dtype=np.uint8)

    for dark_th in [95, 105, 85, 75]:
        _, dark_binary = cv2.threshold(blur, dark_th, 255, cv2.THRESH_BINARY_INV)
        cnts_dark, _ = cv2.findContours(dark_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        large_dark = [c for c in cnts_dark if cv2.contourArea(c) > 6000]
        if large_dark:
            plate_cnt = max(large_dark, key=cv2.contourArea)
            bx, by, bw, bh = cv2.boundingRect(plate_cnt)
            bbox_dark = (bx, by, bw, bh)
            cv2.drawContours(plate_mask, [plate_cnt], -1, 255, -1)
            break

    if plate_cnt is not None:
        print(f"  -> {camera_name}: Mörkt underlag identifierat på ({bbox_dark[0]}, {bbox_dark[1]}), storlek {bbox_dark[2]}x{bbox_dark[3]} px")
    else:
        print(f"  -> {camera_name}: [INFO] Inget separat mörkt underlag hittades, söker i hela bilden...")
        plate_mask[:, :] = 255

    # --------------------------------------------------------------------------
    # STEG 2: Hitta den vita bollen inuti det mörka underlaget
    # --------------------------------------------------------------------------
    white_mask = (blur > 115) & (plate_mask == 255)
    cnts_white, _ = cv2.findContours(white_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_ball = None
    best_circ = 0.0

    for c in cnts_white:
        area = cv2.contourArea(c)
        if area < 600 or area > 15000:
            continue
        perimeter = cv2.arcLength(c, True)
        if perimeter == 0:
            continue
        circ = 4 * np.pi * (area / (perimeter * perimeter))
        (cx, cy), radius = cv2.minEnclosingCircle(c)

        if 20.0 <= radius <= 70.0 and circ > 0.55:
            if circ > best_circ:
                best_circ = circ
                best_ball = (float(cx), float(cy), float(radius))

    return best_ball, bbox_dark


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

        # Totalt ljusflöde på sfären för relativ intensitet
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
    print(f"Resultatfil: {OUTPUT_FILE}\n")

    print("[1/4] Ansluter till VisionRig (kameror + 4-kanals LED)...")
    with VisionRig(verbose=False) as rig:
        if not rig.is_connected:
            print("[FEL] Kunde inte ansluta till hårdvaran i VisionRig.")
            return

        print("[OK] Hårdvara redo.")
        print("     Instruktion: Lägg mörkt underlag (t.ex. byggplatta) på bädden med vit pingisboll ovanpå.")
        print("     Tryck ENTER för att starta kalibreringen: ", end="")
        input()

        # [2/4] Bild med alla lampor tända för jämn sfärbelysning
        print("\n[2/4] Tar bild med ALLA lampor tända (jämn belysning)...")
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

    # Spara alla råbilder omedelbart för full insyn och felsökning
    for cam_name, img_all in cams_all.items():
        cv2.imwrite(str(DEBUG_IMG_DIR / f"raw_all_{cam_name}.jpg"), img_all)
    for cam_name, bgr_list in cams_sweep.items():
        for ch_idx, bgr in enumerate(bgr_list):
            cv2.imwrite(str(DEBUG_IMG_DIR / f"raw_sweep_{cam_name}_ch{ch_idx + 1}.jpg"), bgr)

    print("\n[4/4] Beräknar ljusvektorer för båda kamerorna...")
    calib_results = {
        "calibrated_at": datetime.now().isoformat(),
        "ball_diameter_mm": 40.0,
        "cameras": {},
    }

    for cam_name in ["left_cam1", "right_cam0"]:
        img_all = cams_all[cam_name]
        bgr_sweep = cams_sweep[cam_name]
        grays_sweep = [cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) for im in bgr_sweep]

        circle, bbox_dark = find_sphere_circle(img_all, cam_name)
        if circle is None:
            print(f"  [FEL] Kunde inte hitta pingisbollen i {cam_name}!")
            print(f"        Kontrollera {DEBUG_IMG_DIR / f'raw_all_{cam_name}.jpg'}.")
            return

        cx, cy, radius = circle
        print(f"  -> {cam_name}: Sfär hittad på centrum ({cx:.1f}, {cy:.1f}), radie={radius:.1f} px")

        # Beräkna ljusvektorerna baserat på svepbilderna
        cam_calib = compute_light_vectors_for_camera(grays_sweep, cx, cy, radius, cam_name)
        calib_results["cameras"][cam_name] = cam_calib

        # Skapa debug-bild med byggplattans box, sfärcirkel och lampvektorer
        debug_vis = img_all.copy()

        # Rita bounding box runt det mörka underlaget (om det hittades)
        if bbox_dark is not None:
            bx, by, bw, bh = bbox_dark
            cv2.rectangle(debug_vis, (bx, by), (bx + bw, by + bh), (255, 180, 0), 2)
            cv2.putText(
                debug_vis,
                "Morkt underlag",
                (bx + 8, by + 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 180, 0),
                2,
            )

        # Rita ut sfärcirkeln kring bollen
        cv2.circle(debug_vis, (int(cx), int(cy)), int(radius), (0, 255, 0), 2)
        cv2.putText(
            debug_vis,
            f"Boll r={radius:.1f}px",
            (int(cx) - 45, int(cy) - int(radius) - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )

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
