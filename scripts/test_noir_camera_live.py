"""
Liveskript för att testa NoIR-kamera och 4x 3W 850nm IR-lysdioder på Raspberry Pi CM5.

Styrprincip: Active-LOW via YYNMOS-4 MOSFET (LOW/0V = Tänd, HIGH/3.3V = Släckt).
Pinnar hämtas direkt från src/core/config.py:
  - LED 1 (SE): GPIO 5  (Stift 29)
  - LED 2 (SW): GPIO 6  (Stift 31)
  - LED 3 (NW): GPIO 25 (Stift 22)
  - LED 4 (NE): GPIO 24 (Stift 18)

Användning:
  python3 scripts/test_noir_camera_live.py         # Tar alla 5 bilder i följd (LED 1-4 + ALLA)
  python3 scripts/test_noir_camera_live.py 1       # Tar enbart bild med LED 1
  python3 scripts/test_noir_camera_live.py all     # Tar enbart bild med alla lampor tända
"""

import sys
import time
import subprocess
from pathlib import Path

# Säkerställ att projektroten finns i sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Hämta hårdvarukonfiguration
try:
    from src.core.config import LED_CHANNELS
except ImportError:
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class PinMapping:
        bcm: int
        physical_pin: int
        description: str

    LED_CHANNELS = {
        1: PinMapping(bcm=5, physical_pin=29, description="Arm 1 (SE)"),
        2: PinMapping(bcm=6, physical_pin=31, description="Arm 2 (SW)"),
        3: PinMapping(bcm=25, physical_pin=22, description="Arm 3 (NW)"),
        4: PinMapping(bcm=24, physical_pin=18, description="Arm 4 (NE)"),
    }

# Försök importera gpiozero för GPIO-styrning
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


class StrobeController:
    """Styr alla 4 LED-kanaler individuellt eller gemensamt (Active-LOW)."""

    def __init__(self, channels_config: dict):
        self.channels_config = channels_config
        self.devices = {}

        for ch, mapping in self.channels_config.items():
            if HAS_GPIOZERO:
                try:
                    # active_high=False: on() -> 0V (tänd), off() -> 3.3V (släckt)
                    self.devices[ch] = OutputDevice(
                        mapping.bcm,
                        active_high=False,
                        initial_value=False
                    )
                except Exception as e:
                    print(f"[Varning] Kunde inte initiera GPIO {mapping.bcm} via gpiozero: {e}")
            else:
                # Fallback via systemkommandot 'pinctrl'
                try:
                    subprocess.run(["pinctrl", "set", str(mapping.bcm), "op", "dh"], check=False)
                except Exception:
                    pass

    def on(self, channel: int):
        """Tänder en specifik LED-kanal (drar GPIO till 0V)."""
        if channel in self.devices:
            self.devices[channel].on()
        elif channel in self.channels_config:
            bcm = self.channels_config[channel].bcm
            subprocess.run(["pinctrl", "set", str(bcm), "op", "dl"], check=False)

    def off(self, channel: int):
        """Släcker en specifik LED-kanal (sätter GPIO till 3.3V)."""
        if channel in self.devices:
            self.devices[channel].off()
        elif channel in self.channels_config:
            bcm = self.channels_config[channel].bcm
            subprocess.run(["pinctrl", "set", str(bcm), "op", "dh"], check=False)

    def all_on(self):
        """Tänder alla 4 LED-kanaler."""
        for ch in self.channels_config:
            self.on(ch)

    def all_off(self):
        """Släcker alla 4 LED-kanaler."""
        for ch in self.channels_config:
            self.off(ch)

    def close(self):
        """Säkerställer att alla lampor är släckta och släpper hårdvaruresurser."""
        self.all_off()
        for dev in self.devices.values():
            try:
                dev.close()
            except Exception:
                pass


def capture_sequence_picam2(strobe: StrobeController, shots: list):
    """Tar bildsekvens via Picamera2 Python-biblioteket."""
    print("[Kamera] Initierar Picamera2 på CAM0 (CSI0)...")
    picam2 = Picamera2(camera_num=0)
    config = picam2.create_still_configuration()
    picam2.configure(config)
    picam2.start()

    print("[Kamera] Startar sensorn och låter AGC/AEC stabiliseras (0.5s)...")
    time.sleep(0.5)

    for desc, channels, file_path in shots:
        print(f"\n--> Belysning: {desc}")
        # Tänd angivna lampor
        for ch in channels:
            strobe.on(ch)

        # Låt exponeringen anpassa sig efter ljusvinkeln
        time.sleep(0.4)

        print(f"    Tar bild med CAM0...")
        picam2.capture_file(str(file_path))
        print(f"    Sparad: {file_path.name} ({file_path.stat().st_size / 1024:.1f} KB)")

        # Släck igen
        for ch in channels:
            strobe.off(ch)

        time.sleep(0.15)

    picam2.stop()
    picam2.close()


def capture_sequence_rpicam_cli(strobe: StrobeController, shots: list):
    """Fallback via rpicam-still systemkommando."""
    print("[Kamera] Kör via rpicam-still CLI (CAM0)...")

    for desc, channels, file_path in shots:
        print(f"\n--> Belysning: {desc}")
        for ch in channels:
            strobe.on(ch)

        time.sleep(0.1)

        cmd = [
            "rpicam-still",
            "--camera", "0",
            "-t", "500",
            "-o", str(file_path),
            "--nopreview",
        ]
        subprocess.run(cmd, check=True)
        print(f"    Sparad: {file_path.name} ({file_path.stat().st_size / 1024:.1f} KB)")

        for ch in channels:
            strobe.off(ch)

        time.sleep(0.15)


def main():
    print("=" * 60)
    print("  Paketmätningssystem V3: Test av 4-vägs IR-stroboskop (CAM0)")
    print("=" * 60)

    all_shots = [
        ("LED 1: Sydost / SE (GPIO 5, Stift 29)", [1], PROJECT_ROOT / "test_ir_led1_se.jpg"),
        ("LED 2: Sydväst / SW (GPIO 6, Stift 31)", [2], PROJECT_ROOT / "test_ir_led2_sw.jpg"),
        ("LED 3: Nordväst / NW (GPIO 25, Stift 22)", [3], PROJECT_ROOT / "test_ir_led3_nw.jpg"),
        ("LED 4: Nordost / NE (GPIO 24, Stift 18)", [4], PROJECT_ROOT / "test_ir_led4_ne.jpg"),
        ("ALLA 4 LAMPOR TÄNDA SAMTIDIGT", [1, 2, 3, 4], PROJECT_ROOT / "test_ir_led_all.jpg"),
    ]

    # CLI-filtrering vid behov
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg in ["1", "2", "3", "4"]:
            ch = int(arg)
            shots = [s for s in all_shots if s[1] == [ch]]
        elif arg in ["all", "alla"]:
            shots = [all_shots[-1]]
        else:
            print(f"Okänt argument '{arg}'. Kör alla 5 bilder.")
            shots = all_shots
    else:
        shots = all_shots

    strobe = StrobeController(LED_CHANNELS)

    try:
        if HAS_PICAM2:
            capture_sequence_picam2(strobe, shots)
        else:
            capture_sequence_rpicam_cli(strobe, shots)

        print("\n" + "=" * 60)
        print("  Alla testbilder är tagna!")
        print("=" * 60)
        for _, _, p in shots:
            if p.exists():
                print(f"  * {p.name:<25} ({p.stat().st_size / 1024:.1f} KB)")
        print("\nBilderna finns i projektmappen.")

    except Exception as e:
        print(f"\n[Fel] Ett fel inträffade: {e}")
    finally:
        strobe.close()


if __name__ == "__main__":
    main()
