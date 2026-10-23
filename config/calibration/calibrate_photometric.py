"""
Fotometrisk kalibrering för Paketmätningssystem V3.

Beräknar ljusriktningsvektorer (L1..L4) och relativa lampstyrkor (s1..s4)
för båda kamerorna (CAM1 Vänster och CAM0 Höger) med hjälp av en vit pingisboll.
Resultatet sparas till config/calibration/data/photometric.json.

Instruktioner:
1. Placera en vit, matt pingisboll (40 mm) mitt på mätbädden.
2. Kör skriptet på Raspberry Pi:
    python3 config/calibration/calibrate_photometric.py
3. Skriptet kör ett fotometriskt svep, lokaliserar bollen och beräknar vektorerna.
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
MIN_RADIUS_PX: int = 28          # Minsta radie i pixlar (~30-50 px vid 600-700 mm)
MAX_RADIUS_PX: int = 55          # Största radie i pixlar
ROI_RADIUS_PX: int = 400         # Sökradie kring mitten (ignorerar hörn och bordskanter)
NUM_SWEEPS: int = 1              # Antal svep att medelvärdesbilda över (standard: 1)

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
    images_gray: List[np.ndarray],
    min_radius: int = MIN_RADIUS_PX,
    max_radius: int = MAX_RADIUS_PX,
    roi_radius: int = ROI_RADIUS_PX,
) -> Optional[Tuple[float, float, float]]:
    """
    Hittar sfärens cirkel (cx, cy, radie) på medelvärdesbilden inom en centrum-ROI.
    Ignorerar hörn, kanter och bakgrund utanför mätbäddens mittområde.
    """
    avg_gray = np.mean(images_gray, axis=0).astype(np.uint8)
    h, w = avg_gray.shape
    cx_nom, cy_nom = w // 2, h // 2

    # Maskera ut centrum-ROI (ignorerar allt utanför mitten av bädden)
    y_coords, x_coords = np.ogrid[:h, :w]
    roi_mask = ((x_coords - cx_nom) ** 2 + (y_coords - cy_nom) ** 2) <= (roi_radius ** 2)

    roi_img = np.zeros_like(avg_gray)
    roi_img[roi_mask] = avg_gray[roi_mask]
    blur = cv2.GaussianBlur(roi_img, (9, 9), 2)

    # Metod 1: HoughCircles i centrum-ROI
    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=60,
        param1=50,
        param2=24,
        minRadius=min_radius,
        maxRadius=max_radius,
    )
    if circles is not None and len(circles[0]) > 0:
        # Välj cirkeln närmast bildens mittpunkt
        best_circle = None
        min_dist = float("inf")
        for c in circles[0]:
            cx, cy, r = float(c[0]), float(c[1]), float(c[2])
            dist = np.hypot(cx - cx_nom, cy - cy_nom)
            if dist < min_dist:
                min_dist = dist
                best_circle = (cx, cy, r)
        if best_circle is not None:
            return best_circle

    # Metod 2 (Fallback): Tröskla ut ljusaste området i ROI och sök rundaste kontur
    roi_pixels = roi_img[roi_mask]
    if len(roi_pixels) == 0:
        return None
    thresh_val = max(140, int(np.percentile(roi_pixels, 92)))
    _, thresh = cv2.threshold(roi_img, thresh_val, 255, cv2.THRESH_BINARY)

    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_circle = None
    best_score = 0.0

    for c in cnts:
        area = cv2.contourArea(c)
        if area < np.pi * (min_radius ** 2):
            continue
        perimeter = cv2.arcLength(c, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        (cx, cy), radius = cv2.minEnclosingCircle(c)
        dist_to_center = np.hypot(cx - cx_nom, cy - cy_nom)

        if min_radius <= radius <= max_radius and circularity > 0.60 and dist_to_center <= roi_radius:
            score = circularity * (1.0 - (dist_to_center / roi_radius) * 0.5)
            if score > best_score:
                best_score = score
                best_circle = (float(cx), float(cy), float(radius))

    return best_circle


def compute_light_vectors_for_camera(
    images_gray: List[np.ndarray],
    cx: float,
    cy: float,
    radius: float,
    camera_name: str,
) -> Dict:
    """
    Beräknar de 4 ljusriktningsvektorerna (L1..L4) och relativa styrkor (s1..s4)
    för en kamera baserat på sfärens ytreflektioner.
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

        print("[OK] Hårdvara redo. Se till att pingisbollen ligger på mätbädden.")
        print("     Tryck ENTER för att köra ljussvepet: ", end="")
        input()

        # [2/4] Samla in fotometriskt ljussvep
        print("\n[2/4] Tar fotometriskt svep (Kanal 1 -> 2 -> 3 -> 4)...")
        sweep = rig.capture_photometric_sweep()
        if not sweep.is_valid():
            print("[FEL] Fotometriska svepet misslyckades eller saknar kanaler.")
            return

    # Separera de 4 bilderna per kamera
    # CAM1 = Vänster, CAM0 = Höger
    cams = {
        "left_cam1": [sweep.get_channel(ch).left.image for ch in [1, 2, 3, 4]],
        "right_cam0": [sweep.get_channel(ch).right.image for ch in [1, 2, 3, 4]],
    }

    print("\n[3/4] Detekterar pingisbollen och beräknar ljusvektorer för båda kamerorna...")
    DEBUG_IMG_DIR.mkdir(parents=True, exist_ok=True)
    calib_results = {
        "calibrated_at": datetime.now().isoformat(),
        "ball_diameter_mm": BALL_DIAMETER_MM,
        "cameras": {},
    }

    for cam_name, bgr_list in cams.items():
        grays = [cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) for im in bgr_list]

        # Medelvärdesbild för kontrast och cirkeldetektion (utan övermättning)
        avg_gray = np.mean(grays, axis=0).astype(np.uint8)

        circle = find_sphere_circle(grays)
        if circle is None:
            print(f"  [FEL] Kunde inte hitta pingisbollen i {cam_name} inom centrum-ROI!")
            print("        Kontrollera att bollen ligger i mitten av bädden och syns tydligt.")
            return

        cx, cy, radius = circle
        print(f"  -> {cam_name}: Bollen hittad på centrum ({cx:.1f}, {cy:.1f}), radie={radius:.1f} px")

        # Beräkna ljusvektorerna
        cam_calib = compute_light_vectors_for_camera(grays, cx, cy, radius, cam_name)
        calib_results["cameras"][cam_name] = cam_calib

        # Spara råbilder för varje belysningskanal för enkel inspektion
        for ch_idx, bgr in enumerate(bgr_list):
            cv2.imwrite(str(DEBUG_IMG_DIR / f"raw_{cam_name}_ch{ch_idx + 1}.jpg"), bgr)

        # Skapa en debug-bild som visar sök-ROI, bollen och de 4 belysningspunkterna
        debug_vis = cv2.cvtColor(avg_gray, cv2.COLOR_GRAY2BGR)
        h, w = avg_gray.shape
        cv2.circle(debug_vis, (w // 2, h // 2), ROI_RADIUS_PX, (80, 80, 80), 1)  # ROI-gräns
        cv2.circle(debug_vis, (int(cx), int(cy)), int(radius), (0, 255, 0), 2)   # Detekterad sfär
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

    # [4/4] Spara till JSON
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
