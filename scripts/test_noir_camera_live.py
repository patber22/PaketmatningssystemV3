"""
Liveskript för att testa NoIR-kamera och 3W IR-lysdiod på Raspberry Pi CM5.
Tänder IR-lysdioden (Active-LOW på GPIO 17), knäpper en bild med CAM0,
släcker lysdioden och sparar bilden som 'test_ir_cam0.jpg'.

Körs på pajen med: python3 scripts/test_noir_camera_live.py
"""

import time
import sys
import subprocess
from pathlib import Path

# Säkerställ att projektroten finns i sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# GPIO-pinne för Kanal 1 (Standard: GPIO 5 = Fysiskt stift 29)
# Kan även överstyras via kommandoraden, t.ex: python3 scripts/test_noir_camera_live.py 5
STROBE_PIN = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 5
OUTPUT_FILENAME = PROJECT_ROOT / "test_ir_cam0.jpg"

# Försök importera gpiozero för Raspberry Pi 5/CM5 GPIO-styrning
try:
    from gpiozero import OutputDevice
    HAS_GPIOZERO = True
except ImportError:
    HAS_GPIOZERO = False

# Försök importera Picamera2
try:
    from picamera2 import Picamera2
    HAS_PICAM2 = True
except ImportError:
    HAS_PICAM2 = False


class SimpleStrobe:
    """Enkel Active-LOW GPIO-kontroller kompatibel med Raspberry Pi 5 / CM5."""

    def __init__(self, pin: int):
        self.pin = pin
        self._device = None
        if HAS_GPIOZERO:
            # active_high=False: on() drar pinnen till 0V (LOW = Tänd), off() sätter 3.3V (HIGH = Släckt)
            self._device = OutputDevice(self.pin, active_high=False, initial_value=False)
        else:
            # Fallback via officiella Pi 5 'pinctrl'-kommandot
            try:
                subprocess.run(["pinctrl", "set", str(self.pin), "op", "dh"], check=False)
            except (FileNotFoundError, PermissionError):
                pass

    def on(self):
        """Tänder lysdioden (drar pinnen till 0V)."""
        if self._device:
            self._device.on()
        else:
            try:
                subprocess.run(["pinctrl", "set", str(self.pin), "op", "dl"], check=False)
            except (FileNotFoundError, PermissionError):
                pass

    def off(self):
        """Släcker lysdioden (sätter pinnen till 3.3V)."""
        if self._device:
            self._device.off()
        else:
            try:
                subprocess.run(["pinctrl", "set", str(self.pin), "op", "dh"], check=False)
            except (FileNotFoundError, PermissionError):
                pass

    def close(self):
        self.off()
        if self._device:
            self._device.close()


def capture_with_picam2(output_path: Path, strobe: SimpleStrobe):
    """Tar bild via Picamera2 Python-biblioteket."""
    print("[Kamera] Initierar Picamera2 på CAM0 (CSI0)...")
    picam2 = Picamera2(camera_num=0)
    config = picam2.create_still_configuration()
    picam2.configure(config)
    picam2.start()

    print("[Kamera] Väntar 0.5s på sensorns AGC/AEC...")
    time.sleep(0.5)

    print("[Blixt] Tänder 3W IR-lysdioden...")
    strobe.on()
    time.sleep(0.05)  # 50 ms förstabilisering

    print(f"[Kamera] Exponerar och sparar bild till: {output_path.name}...")
    picam2.capture_file(str(output_path))

    print("[Blixt] Släcker 3W IR-lysdioden...")
    strobe.off()

    picam2.stop()
    picam2.close()


def capture_with_rpicam_cli(output_path: Path, strobe: SimpleStrobe):
    """Fallback via systemverktyget rpicam-still."""
    print("[Kamera] Kör via rpicam-still CLI (CAM0)...")
    print("[Blixt] Tänder 3W IR-lysdioden...")
    strobe.on()
    time.sleep(0.05)

    cmd = [
        "rpicam-still",
        "--camera", "0",
        "-t", "500",
        "-o", str(output_path),
        "--nopreview",
    ]
    subprocess.run(cmd, check=True)

    print("[Blixt] Släcker 3W IR-lysdioden...")
    strobe.off()


def main():
    print("\n=== Test: NoIR-kamera (CAM0) + 3W IR Blixt ===")
    print(f"Styrsignal: GPIO {STROBE_PIN} (Active-LOW, standard: Stift 29 / GPIO 5)")
    print(f"Målfil:     {OUTPUT_FILENAME.resolve()}\n")

    strobe = SimpleStrobe(STROBE_PIN)
    try:
        if HAS_PICAM2:
            capture_with_picam2(OUTPUT_FILENAME, strobe)
        else:
            capture_with_rpicam_cli(OUTPUT_FILENAME, strobe)

        print("\n Klar! Bilden är sparad.")
        print(f"Filstorlek: {OUTPUT_FILENAME.stat().st_size / 1024:.1f} KB")
        print("Du kan öppna bilden på pajen för att beundra ditt IR-ljus!\n")

    except Exception as e:
        strobe.off()
        print(f"\n Fel under fotografering: {e}")
    finally:
        strobe.close()


if __name__ == "__main__":
    main()
