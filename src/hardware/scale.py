"""
Drivrutin för Paketmätningssystemets vågmodul (ESP32-C3 + HX711 via USB-serieport).

Läser asynkront ADC-råvärden från lastcellerna (Wheatstone-brygga) i en bakgrundstråd,
applicerar kalibreringsparametrar (tara och skalfaktor) och beräknar stabilitet (standardavvikelse).
Returnerar standardiserade WeightReading-objekt.
"""

import json
import os
import threading
import time
from collections import deque
from typing import Optional, List
import numpy as np
import serial
import serial.tools.list_ports

from src.core.models import WeightReading


class Scale:
    """
    Hanterar asynkron seriekommunikation och filtrering för vågmodulen.
    """

    DEFAULT_CONFIG_PATH = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "config",
            "calibration",
            "data",
            "scale.json",
        )
    )

    def __init__(
        self,
        port: Optional[str] = None,
        baudrate: int = 115200,
        config_path: Optional[str] = None,
        buffer_size: int = 10,
        stability_threshold_g: float = 5.0,
        verbose: bool = True,
        auto_connect: bool = True,
    ):
        self.port = port
        self.baudrate = baudrate
        self.config_path = config_path or self.DEFAULT_CONFIG_PATH
        self.buffer_size = buffer_size
        self.stability_threshold_g = stability_threshold_g
        self.verbose = verbose

        # Kalibreringsparametrar
        self.tare_offset_raw: float = 0.0
        self.scale_factor: float = 1.0  # counts per gram (delta_raw / gram)
        self.is_calibrated: bool = False

        # Serieports- och trådhantering
        self.ser: Optional[serial.Serial] = None
        self.is_connected: bool = False
        self.running: bool = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Buffertar för filtrering
        self._raw_buffer: deque = deque(maxlen=self.buffer_size)
        self._latest_raw: Optional[int] = None
        self._latest_time: float = 0.0

        # Läs in sparad kalibrering
        self.load_calibration()

        # Anslut automatiskt om begärt
        if auto_connect:
            self.connect()

    @staticmethod
    def find_port() -> Optional[str]:
        """
        Söker automatiskt efter ansluten ESP32-C3 vågmodul.
        Letar i första hand efter Espressif USB-JTAG/CDC (VID 0x303A).
        Fallback till vanliga USB-seriella enheter.
        """
        ports = serial.tools.list_ports.comports()

        # 1. Prioritera Espressif VID för ESP32-C3 USB CDC (0x303A)
        for p in ports:
            if getattr(p, "vid", None) == 0x303A:
                return p.device

        # 2. Fallback: vanliga beskrivningar för USB-serieportar
        for p in ports:
            desc = (p.description or "").lower()
            if any(term in desc for term in ("seriell usb", "usb-serial", "ch340", "cp210")):
                return p.device

        return None

    def load_calibration(self, path: Optional[str] = None) -> bool:
        """
        Läser in tara och skalfaktor från kalibreringsfilen.
        Returnerar True vid framgång, False om filen saknas eller är ogiltig.
        """
        target_path = path or self.config_path
        if not os.path.exists(target_path):
            if self.verbose:
                print(
                    f"[Scale] Ingen kalibreringsfil hittades på {target_path}. "
                    f"Kör calibration/calibrate_scale.py."
                )
            self.is_calibrated = False
            return False

        try:
            with open(target_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.tare_offset_raw = float(data.get("tare_offset_raw", 0.0))
            self.scale_factor = float(data.get("scale_factor", 1.0))
            self.is_calibrated = True

            if self.verbose:
                print(
                    f"[Scale] Kalibrering inläst från {target_path} "
                    f"(Tare: {self.tare_offset_raw:.0f}, Faktor: {self.scale_factor:.3f} cts/g)"
                )
            return True
        except Exception as e:
            if self.verbose:
                print(f"[Scale] Fel vid inläsning av kalibreringsfil ({target_path}): {e}")
            self.is_calibrated = False
            return False

    def connect(self) -> bool:
        """
        Ansluter till vågen och startar bakgrundstråden för kontinuerlig avläsning.
        Returnerar True vid anslutning, False vid fel. Kraschar aldrig.
        """
        if self.is_connected:
            return True

        target_port = self.port or self.find_port()
        if not target_port:
            if self.verbose:
                print("[Scale] Ingen serieport angiven och ingen ESP32-C3 våg hittades.")
            self.is_connected = False
            return False

        try:
            self.ser = serial.Serial(target_port, self.baudrate, timeout=1.0)
            self.port = target_port
            self.ser.reset_input_buffer()

            self.running = True
            self.is_connected = True
            self._thread = threading.Thread(
                target=self._read_loop,
                daemon=True,
                name="Scale_Serial_Reader",
            )
            self._thread.start()

            if self.verbose:
                print(f"[Scale] Ansluten till våg på {self.port} ({self.baudrate} baud).")
            return True
        except Exception as e:
            if self.verbose:
                print(f"[Scale] Kunde inte ansluta till {target_port}: {e}")
            self.ser = None
            self.is_connected = False
            return False

    def disconnect(self) -> None:
        """Kopplar ifrån vågen och avslutar bakgrundstråden säkert."""
        self.running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.5)
        self._thread = None

        if self.ser and self.ser.is_open:
            try:
                self.ser.close()
            except Exception:
                pass
        self.ser = None
        self.is_connected = False

        if self.verbose:
            print("[Scale] Frånkopplad.")

    def _read_loop(self) -> None:
        """Bakgrundstråd som kontinuerligt tar emot rådata från ESP32-C3."""
        while self.running:
            if not self.ser or not self.ser.is_open:
                time.sleep(0.1)
                continue

            try:
                line = self.ser.readline().decode("utf-8", errors="ignore").strip()
                if not line:
                    continue

                # Matcha formatet RAW:<tal> eller rent heltal
                raw_val = None
                if line.startswith("RAW:"):
                    raw_str = line.split("RAW:", 1)[1].strip()
                else:
                    raw_str = line

                try:
                    raw_val = int(raw_str)
                except ValueError:
                    continue

                now = time.time()
                with self._lock:
                    self._latest_raw = raw_val
                    self._latest_time = now
                    self._raw_buffer.append(raw_val)

            except serial.SerialException:
                if self.running:
                    if self.verbose:
                        print("[Scale] Serieportsfel, försöker igen om 0.5s...")
                    time.sleep(0.5)
            except Exception:
                pass

    def tare(self, samples: int = 15, timeout_s: float = 2.5) -> bool:
        """
        Sätter aktuell nollpunkt (tara) i minnet utan att ändra skalfaktorn eller skriva till fil.
        Väntar tills 'samples' observationer samlats in.
        """
        collected: List[int] = []
        start_time = time.time()

        while len(collected) < samples and (time.time() - start_time) < timeout_s:
            with self._lock:
                if self._latest_raw is not None and (not collected or collected[-1] != self._latest_raw):
                    collected.append(self._latest_raw)
            time.sleep(0.05)

        if not collected:
            if self.verbose:
                print("[Scale] Kunde inte tarera, inga mätvärden mottagna.")
            return False

        median_tare = float(np.median(collected))
        self.tare_offset_raw = median_tare
        if self.verbose:
            print(f"[Scale] Tarerad! Ny nollpunkt i minnet (råvärde): {self.tare_offset_raw:.0f}")
        return True

    def read(self, samples: int = 5) -> WeightReading:
        """
        Returnerar aktuell filtrerad mätning från vågen.
        Använder medianen av de senaste observationerna för spik- och brusfiltrering.
        """
        with self._lock:
            buf = list(self._raw_buffer)
            latest_raw = self._latest_raw
            latest_time = self._latest_time

        if not buf or latest_raw is None:
            return WeightReading.invalid()

        # Begränsa till de 'samples' senaste observationerna för medianen
        recent_samples = buf[-min(samples, len(buf)):]
        med_raw = float(np.median(recent_samples))

        # Kontrollera stabilitet (standardavvikelse på bufferten i gram)
        if len(buf) >= 4 and abs(self.scale_factor) > 1e-4:
            weights_in_buf = [(r - self.tare_offset_raw) / self.scale_factor for r in buf]
            std_g = float(np.std(weights_in_buf))
            is_stable = bool(std_g < self.stability_threshold_g)
        else:
            is_stable = False

        # Beräkna vikt i gram
        if abs(self.scale_factor) > 1e-4:
            weight_g = (med_raw - self.tare_offset_raw) / self.scale_factor
        else:
            weight_g = 0.0

        # Klipp bort små mikrogram-negativa fluktuationer kring nollan (+/- 2g dödband)
        if abs(weight_g) < 2.0:
            weight_g = 0.0

        # Sätt intern giltighetsflagga baserat på kalibrering och icke-negativ vikt
        _is_valid = bool(self.is_calibrated and weight_g >= 0.0)

        return WeightReading(
            weight_g=weight_g,
            is_stable=is_stable,
            timestamp=latest_time if latest_time > 0.0 else time.time(),
            samples=len(recent_samples),
            raw_value=int(med_raw),
            _is_valid=_is_valid,
        )

    def read_stable(self, timeout_s: float = 2.0, poll_interval_s: float = 0.05) -> WeightReading:
        """
        Väntar tills vågen har stabiliserats (is_stable=True) och returnerar mätningen.
        Om timeout nås returneras den senaste mätningen (med is_stable=False).
        """
        start = time.time()
        latest_reading = WeightReading.invalid()

        while (time.time() - start) < timeout_s:
            latest_reading = self.read()
            if latest_reading.is_valid() and latest_reading.is_stable:
                return latest_reading
            time.sleep(poll_interval_s)

        return latest_reading

    def __enter__(self) -> "Scale":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.disconnect()
