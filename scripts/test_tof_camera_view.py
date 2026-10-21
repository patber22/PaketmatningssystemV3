"""
Snabbtest: Se hur ST VL53L5CX 940nm VCSEL-lasern ser ut i NoIR-kameran (CAM0).

1. Tar en referensbild med ToF helt AV (test_tof_OFF.jpg).
2. Startar ToF-sensorn så att 940nm VCSEL-lasermatrisen sänder aktivt.
3. Tar en bild med ToF PÅ (test_tof_ON.jpg).
4. Stänger av sensorn.

Körs på pajen med:
    python3 scripts/test_tof_camera_view.py
"""

import sys
import time
import subprocess
from pathlib import Path

# Säkerställ att projektroten finns i sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.hardware.tof import ToFSensor

# Försök importera Picamera2
try:
    from picamera2 import Picamera2
    HAS_PICAM2 = True
except ImportError:
    HAS_PICAM2 = False


def capture_with_picam2(output_path: Path):
    """Tar bild via Picamera2."""
    picam2 = Picamera2(camera_num=0)
    config = picam2.create_still_configuration()
    picam2.configure(config)
    picam2.start()
    time.sleep(0.4)
    picam2.capture_file(str(output_path))
    picam2.stop()
    picam2.close()


def capture_with_rpicam_cli(output_path: Path):
    """Tar bild via rpicam-still CLI."""
    cmd = [
        "rpicam-still",
        "--camera", "0",
        "-t", "500",
        "-o", str(output_path),
        "--nopreview",
    ]
    subprocess.run(cmd, check=True)


def capture_cam0(output_path: Path):
    """Tar en bild med CAM0 med tillgängligt bibliotek."""
    if HAS_PICAM2:
        capture_with_picam2(output_path)
    else:
        capture_with_rpicam_cli(output_path)


def main():
    print("=" * 65)
    print("  Kameratest: ST VL53L5CX 940nm VCSEL-laser i NoIR (CAM0)")
    print("=" * 65)

    path_off = PROJECT_ROOT / "test_tof_OFF.jpg"
    path_on = PROJECT_ROOT / "test_tof_ON.jpg"

    # Steg 1: Bild med ToF AV
    print("\n📸 [1/2] Tar referensbild med ToF helt AV...")
    capture_cam0(path_off)
    print(f"   Sparad: {path_off.name} ({path_off.stat().st_size / 1024:.1f} KB)")

    # Steg 2: Starta ToF och ta bild med ToF PÅ
    print("\n⚡ [2/2] Startar ST VL53L5CX ToF-sensorn...")
    with ToFSensor(verbose=True) as tof:
        if not tof.is_connected:
            print("[Fel] Kunde inte ansluta till ToF-sensorn. Kontrollera I2C!")
            return

        print("   ToF är igång och sänder 940nm laserpulser!")
        print("   Väntar 0.5s för stabil mätström...")
        time.sleep(0.5)

        print("\n📸 Tar bild med NoIR CAM0 medan ToF-lasern sänder...")
        capture_cam0(path_on)
        print(f"   Sparad: {path_on.name} ({path_on.stat().st_size / 1024:.1f} KB)")

    print("\n" + "=" * 65)
    print("  Klart! ToF-lasern är avstängd igen.")
    print("  Jämför filerna i projektmappen:")
    print(f"    1. {path_off.name} (ToF AV)")
    print(f"    2. {path_on.name}  (ToF PÅ)")
    print("  (Tips: Om rummet är lite dunkelt syns laserpunkterna allra bäst!)")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
