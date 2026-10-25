"""
Labskript: Utvärdering av Differentiell Skugganalys för CAM1
===========================================================

Detta skript undersöker hur skuggorna framträder runt paketen med hjälp av
differentiell bildsubtraktion mellan motstående lampor, utan att behöva
någon sparad referensbild på tomt bord:
  - Par 1: Lampa 1 vs Lampa 3 (Ch1 mot Ch3)
  - Par 2: Lampa 2 vs Lampa 4 (Ch2 mot Ch4)

Körs för både:
  - small_book (liten vit matt bok)
  - blue_book (stor blå matt bok)
"""

import sys
from pathlib import Path
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data" / "test_data"
OUTPUT_DIR = PROJECT_ROOT / "data" / "test_output" / "shadow_exploration"


def save_image(path: Path, img: np.ndarray):
    """Sparar bild säkert på Windows även med icke-ASCII-tecken i sökvägen."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower() if path.suffix else ".jpg"
    is_success, buf = cv2.imencode(ext, img)
    if is_success:
        buf.tofile(str(path))
    else:
        print(f"[VARNING] Kunde inte spara bild till: {path}")


def load_cam1_channels(folder: Path):
    """Laddar de fyra råa kanalbilderna för CAM1 som gråskala float32."""
    channels = []
    for ch in range(1, 5):
        p = folder / f"raw_ch{ch}_CAM1.jpg"
        if not p.exists():
            raise FileNotFoundError(f"Kunde inte hitta: {p}")
        img = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        channels.append(img.astype(np.float32))
    return channels


def filter_small_blobs(mask: np.ndarray, min_area: int = 800) -> np.ndarray:
    """Tar bort isolerade brusblobbar och kantreflektioner utanför bordet."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    clean = np.zeros_like(mask, dtype=bool)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            clean[labels == i] = True
    return clean


def analyze_shadows_for_dataset(name: str):
    print(f"\n[{name.upper()}] Läser in CAM1-kanaler...")
    folder = DATA_DIR / name
    out_dir = OUTPUT_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)

    i1, i2, i3, i4 = load_cam1_channels(folder)

    # Läs in kalibrerade lampstyrkor från photometric.json
    calib_file = PROJECT_ROOT / "config" / "calibration" / "data" / "photometric.json"
    if calib_file.exists():
        import json
        with open(calib_file, "r", encoding="utf-8") as f:
            intens = json.load(f)["cameras"]["left_cam1"]["light_intensities"]
            s = np.array(intens, dtype=np.float32)
    else:
        s = np.array([1.018, 1.031, 1.038, 0.912], dtype=np.float32)

    # Normalisera kanalerna med lampornas styrka för att eliminera obalans
    i1_n = i1 / s[0]
    i2_n = i2 / s[1]
    i3_n = i3 / s[2]
    i4_n = i4 / s[3]

    # 1. Bipolära differentialbilder (centrerade runt 128 = neutralgrå)
    diff_1_3 = np.clip(128.0 + (i1_n - i3_n), 0, 255).astype(np.uint8)
    diff_2_4 = np.clip(128.0 + (i2_n - i4_n), 0, 255).astype(np.uint8)

    save_image(out_dir / "1_diff_ch1_minus_ch3.jpg", diff_1_3)
    save_image(out_dir / "2_diff_ch2_minus_ch4.jpg", diff_2_4)

    # 2. De fyra individuella riktningsskuggorna
    s1 = np.clip((i3_n - i1_n) * 2.0, 0, 255).astype(np.uint8)  # Skugga från Lampa 1
    s2 = np.clip((i4_n - i2_n) * 2.0, 0, 255).astype(np.uint8)  # Skugga från Lampa 2
    s3 = np.clip((i1_n - i3_n) * 2.0, 0, 255).astype(np.uint8)  # Skugga från Lampa 3
    s4 = np.clip((i2_n - i4_n) * 2.0, 0, 255).astype(np.uint8)  # Skugga från Lampa 4

    save_image(out_dir / "3_shadow_led1.jpg", s1)
    save_image(out_dir / "4_shadow_led2.jpg", s2)
    save_image(out_dir / "5_shadow_led3.jpg", s3)
    save_image(out_dir / "6_shadow_led4.jpg", s4)

    # 3. Färgkodad översiktskarta över alla fyra skuggorna runt paketet
    base_gray = ((i1 + i2 + i3 + i4) / 4.0).astype(np.uint8)
    color_vis = cv2.cvtColor(base_gray // 2, cv2.COLOR_GRAY2BGR)

    # Generös bordsmask: Behåller hela det aktiva träbordet och klipper chassit utanför
    bed_mask = np.zeros_like(i1, dtype=bool)
    bed_mask[120:1090, 230:1450] = True

    # Relativ kontrast: Procentuellt ljusfall = (I_lit - I_dark) / I_lit
    r1 = (i3_n - i1_n) / np.maximum(i3_n, 1.0)
    r2 = (i4_n - i2_n) / np.maximum(i4_n, 1.0)
    r3 = (i1_n - i3_n) / np.maximum(i1_n, 1.0)
    r4 = (i2_n - i4_n) / np.maximum(i2_n, 1.0)

    # Kombinera procentuell tröskling (>45% ljusfall) med absolut differens (>40 ADU)
    ratio_th = 0.45
    diff_th = 40.0
    m1_raw = (r1 > ratio_th) & ((i3_n - i1_n) > diff_th) & bed_mask
    m2_raw = (r2 > ratio_th) & ((i4_n - i2_n) > diff_th) & bed_mask
    m3_raw = (r3 > ratio_th) & ((i1_n - i3_n) > diff_th) & bed_mask
    m4_raw = (r4 > ratio_th) & ((i2_n - i4_n) > diff_th) & bed_mask

    # Liten morfologisk öppning (5x5) för att städa kanter
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    m1 = cv2.morphologyEx(m1_raw.astype(np.uint8), cv2.MORPH_OPEN, kernel) > 0
    m2 = cv2.morphologyEx(m2_raw.astype(np.uint8), cv2.MORPH_OPEN, kernel) > 0
    m3 = cv2.morphologyEx(m3_raw.astype(np.uint8), cv2.MORPH_OPEN, kernel) > 0
    m4 = cv2.morphologyEx(m4_raw.astype(np.uint8), cv2.MORPH_OPEN, kernel) > 0

    # Blob-filtrering: Behåll endast sammanhängande skuggytor (>500 px)
    m1_clean = filter_small_blobs(m1, min_area=500)
    m2_clean = filter_small_blobs(m2, min_area=500)
    m3_clean = filter_small_blobs(m3, min_area=500)
    m4_clean = filter_small_blobs(m4, min_area=500)

    color_vis[m1_clean] = [0, 0, 255]     # Röd
    color_vis[m2_clean] = [0, 255, 0]     # Grön
    color_vis[m3_clean] = [255, 120, 0]   # Blå
    color_vis[m4_clean] = [0, 255, 255]   # Gul

    # Textinformation
    cv2.putText(color_vis, f"{name.upper()}: Rena skuggor (Procentuell >45% + Diff >40 + Bed ROI)", (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(color_vis, "Rod: LED1 | Gron: LED2 | Bla: LED3 | Gul: LED4", (30, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

    # Spara de individuellt renade binära skuggmaskerna som förlustfri PNG
    save_image(out_dir / "3_shadow_mask_led1.png", (m1_clean * 255).astype(np.uint8))
    save_image(out_dir / "4_shadow_mask_led2.png", (m2_clean * 255).astype(np.uint8))
    save_image(out_dir / "5_shadow_mask_led3.png", (m3_clean * 255).astype(np.uint8))
    save_image(out_dir / "6_shadow_mask_led4.png", (m4_clean * 255).astype(np.uint8))

    save_image(out_dir / "7_shadow_composite_color.jpg", color_vis)

    print(f"  [OK] Sparade alla skuggkartor till: data/test_output/shadow_exploration/{name}/")
    print(f"       Rensade skuggpixlar: LED1={m1_clean.sum():,}, LED2={m2_clean.sum():,}, LED3={m3_clean.sum():,}, LED4={m4_clean.sum():,}")


def main():
    print("=" * 80)
    print("  UNDERSÖKNING: DIFFERENTIELL SKUGGANALYS PÅ CAM1")
    print("=" * 80)

    analyze_shadows_for_dataset("small_book")
    analyze_shadows_for_dataset("blue_book")

    print("\n" + "=" * 80)
    print("  KLART! Öppna bilderna i: data/test_output/shadow_exploration/")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
