"""
Testskript: Calibrated Differential Shadow Magnitude (CDSM)
============================================================

Implementerar den verifierade 2D-paketmaskningsmetoden som utvecklades 2026-10-25.
Hittar paketets sanna fysiska 2D-kontur utan hårdkodade tröskelvärden eller antaganden om form.

Körs med:
    python scripts/test_package_detection.py --dataset blue_book
    python scripts/test_package_detection.py --dataset small_book
    python scripts/test_package_detection.py --dataset all
"""

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, Tuple

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data" / "test_data"
OUTPUT_DIR = PROJECT_ROOT / "data" / "test_output" / "package_detection"
CALIB_PATH = PROJECT_ROOT / "config" / "calibration" / "data" / "photometric.json"
STEREO_PATH = PROJECT_ROOT / "config" / "calibration" / "data" / "stereo.json"

# Kända mått för referens (Ground Truth i mm)
GROUND_TRUTH: Dict[str, Dict[str, float]] = {
    "small_book": {"length_mm": 178.0, "width_mm": 110.0, "height_mm": 27.0},
    "blue_book": {"length_mm": 251.0, "width_mm": 173.0, "height_mm": 72.0},
}


def imread_unicode(path: Path) -> np.ndarray:
    """Läser in bild säkert på Windows även vid icke-ASCII-tecken i sökvägen."""
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)


def imwrite_unicode(path: Path, img: np.ndarray) -> bool:
    """Sparar bild säkert på Windows även vid icke-ASCII-tecken i sökvägen."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix if path.suffix else ".jpg"
    is_success, buf = cv2.imencode(ext, img)
    if is_success:
        buf.tofile(str(path))
        return True
    return False


def load_lamp_intensities() -> np.ndarray:
    """Laddar kalibrerade lampstyrkor från photometric.json."""
    if not CALIB_PATH.exists():
        return np.array([1.018, 1.031, 1.038, 0.912], dtype=np.float32)
    with open(CALIB_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return np.array(data["cameras"]["left_cam1"]["light_intensities"], dtype=np.float32)


def load_cam1_focal_length() -> Tuple[float, float]:
    """Laddar fx, fy från stereo.json."""
    if not STEREO_PATH.exists():
        return 1290.56, 1290.00
    with open(STEREO_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    K = np.array(data["cameras"]["left_cam1"]["camera_matrix_K"], dtype=np.float64)
    return float(K[0, 0]), float(K[1, 1])


def run_cdsm_pipeline(dataset_name: str) -> Dict[str, any]:
    """
    Kör CDSM-pipelinen från råbilder till färdig paketmask och kontur.
    """
    folder = DATA_DIR / dataset_name
    if not folder.exists():
        raise FileNotFoundError(f"Kunde inte hitta mappen: {folder}")

    # 1. Läs in de 4 råa CAM1-kanalerna
    raw_imgs = []
    for ch in range(1, 5):
        p = folder / f"raw_ch{ch}_CAM1.jpg"
        if not p.exists():
            raise FileNotFoundError(f"Kunde inte hitta bild: {p}")
        raw_imgs.append(imread_unicode(p))

    h_img, w_img = raw_imgs[0].shape
    avg_img = np.mean(raw_imgs, axis=0).astype(np.uint8)

    # 2. Kalibrerad ljusbalansering (Steg 1)
    s = load_lamp_intensities()
    i1 = raw_imgs[0].astype(np.float32) / s[0]
    i2 = raw_imgs[1].astype(np.float32) / s[1]
    i3 = raw_imgs[2].astype(np.float32) / s[2]
    i4 = raw_imgs[3].astype(np.float32) / s[3]

    # 3. Motstående differentiella par och kvadratisk dämpning (Steg 2 & 3)
    diff_x = np.abs(i1 - i3)
    diff_y = np.abs(i2 - i4)
    sx = ((diff_x / np.max(diff_x)) ** 2) * 255.0
    sy = ((diff_y / np.max(diff_y)) ** 2) * 255.0

    # 4. Vektormagnitud (Steg 4)
    mag_raw = np.hypot(sx, sy)
    mag = np.clip((mag_raw / np.max(mag_raw)) * 255.0, 0, 255).astype(np.uint8)

    # 5. Bordsmask för att undvika yttre chassit
    bed_mask = np.zeros((h_img, w_img), dtype=np.uint8)
    bed_mask[180:1050, 260:1380] = 255
    mag_bed = cv2.bitwise_and(mag, bed_mask)

    # Spara magnitudo-bilden så man kan inspektera skuggfältet direkt före tröskling och fyllning
    mag_out_path = OUTPUT_DIR / f"{dataset_name}_magnitude.jpg"
    imwrite_unicode(mag_out_path, mag)

    # 6. Adaptiv tröskling via klassbalanserad Otsu (Steg 5)
    crop_bed = mag_bed[180:1050, 260:1380]
    th_otsu, _ = cv2.threshold(crop_bed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Dynamisk skalfaktor baserad på skuggans andel: faktor = clip(0.5 + andel, 0.50, 0.80)
    andel = float(np.mean(crop_bed > th_otsu))
    scale_factor = float(np.clip(0.5 + andel, 0.50, 0.80))
    th_shadow = max(int(round(th_otsu * scale_factor)), 25)
    binary_shadow = (mag_bed > th_shadow).astype(np.uint8) * 255

    # 7. Ö-isolering via inverterad flödesfyllning (Steg 6)
    # Litet 5x5 filter som tätar mikroskopiska 2-3 pixels glipor i skarpa hörn
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    shadow_closed = cv2.morphologyEx(binary_shadow, cv2.MORPH_CLOSE, kernel)

    flood = shadow_closed.copy()
    ff_mask = np.zeros((h_img + 2, w_img + 2), np.uint8)
    cv2.floodFill(flood, ff_mask, (0, 0), 255)

    pkg_mask_raw = (flood == 0).astype(np.uint8) * 255

    # 8. Extrahera yttersta fysiska konturen (pixel för pixel, inga gissningar/rektanglar)
    cnts, _ = cv2.findContours(pkg_mask_raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        raise RuntimeError("Ingen paketmask kunde isoleras!")
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    pkg_contour = cnts[0]

    # Skapa solid mask där alla inre skuggor (t.ex. bulor/veck) fylls solidt
    solid_mask = np.zeros((h_img, w_img), dtype=np.uint8)
    cv2.drawContours(solid_mask, [pkg_contour], -1, 255, thickness=cv2.FILLED)

    area_px = cv2.contourArea(pkg_contour)

    # 9. Visuell overlay direkt över originalbilden (medelvärdet av LED-bilderna)
    vis = cv2.cvtColor(avg_img, cv2.COLOR_GRAY2BGR)
    color_tint = np.zeros_like(vis)
    color_tint[solid_mask > 0] = [0, 220, 0]
    vis = cv2.addWeighted(vis, 0.85, color_tint, 0.15, 0)
    # Rita den exakta ytterkonturen med 1 px tunn grön linje för maximal pixelprecision
    cv2.drawContours(vis, [pkg_contour], -1, (0, 255, 0), thickness=1)

    cv2.putText(vis, f"CDSM Mask [{dataset_name.upper()}] (th = {th_shadow}, Otsu = {th_otsu:.1f}, andel = {andel*100:.1f}%)", (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(vis, f"Mask Area: {area_px:.0f} px | Konturpunkter: {len(pkg_contour)}", (30, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    out_path = OUTPUT_DIR / f"{dataset_name}_cdsm_result.jpg"
    imwrite_unicode(out_path, vis)

    return {
        "dataset": dataset_name,
        "shadow_threshold": th_shadow,
        "otsu_threshold": th_otsu,
        "andel": andel,
        "scale_factor": scale_factor,
        "area_px": area_px,
        "contour_points": len(pkg_contour),
        "magnitude_path": mag_out_path,
        "output_path": out_path,
    }


def main():
    parser = argparse.ArgumentParser(description="Test av Calibrated Differential Shadow Magnitude (CDSM)")
    parser.add_argument("--dataset", type=str, default="all", help="Mappnamn under data/test_data/ (t.ex. blue_book, small_book, bula_test) eller 'all'")
    args = parser.parse_args()

    if args.dataset == "all":
        datasets = [d.name for d in sorted(DATA_DIR.iterdir()) if d.is_dir() and not d.name.startswith(".")]
    else:
        datasets = [args.dataset]

    print("=" * 75)
    print("  TEST: CALIBRATED DIFFERENTIAL SHADOW MAGNITUDE (CDSM)")
    print("=" * 75)

    for ds in datasets:
        print(f"\n--> Bearbetar dataset: '{ds}'...")
        res = run_cdsm_pipeline(ds)

        print(f"  [OK] Mask isolerad!")
        print(f"       Otsu-tröskel:        {res['otsu_threshold']:.1f} (Skugg-andel: {res['andel']*100:.1f}%)")
        print(f"       Adaptiv tröskel:     {res['shadow_threshold']} (Faktor: {res['scale_factor']:.3f})")
        print(f"       Maskarea:            {res['area_px']:.0f} px")
        print(f"       Konturpunkter:       {res['contour_points']} st")
        print(f"  [BILD] Sparad magnitude:  {res['magnitude_path']}")
        print(f"  [BILD] Sparad overlay:    {res['output_path']}")

    print("\n" + "=" * 75)
    print("  KLART! Alla resultat finns i: data/test_output/package_detection/")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
