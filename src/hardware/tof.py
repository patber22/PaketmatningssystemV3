"""
Hårdvarudrivrutin för ST VL53L5CX Time-of-Flight (8x8 zoner) över Linux I2C.

Kör en asynkron bakgrundstråd som kontinuerligt pollar sensorn i 15 Hz för att
hålla en färsk 8x8-matris uppdaterad med 0 ms svarstid.
Garanterar trådsäkerhet via en intern Mutex (threading.Lock).
"""

import time
import threading
from typing import Optional, Tuple
import numpy as np

from src.core.models import ToFReading

# Försök importera ST:s officiella C-drivrutin för Linux I2C
try:
    import vl53l5cx_ctypes
    HAS_HARDWARE_DRIVER = True
except ImportError:
    HAS_HARDWARE_DRIVER = False


class ToFSensor:
    """
    Hanterar kontinuerlig I2C-avläsning och zon-aggregering för VL53L5CX.
    """

    def __init__(
        self,
        frequency_hz: int = 15,
        verbose: bool = True,
        auto_connect: bool = True,
    ):
        self.frequency_hz = frequency_hz
        self.verbose = verbose

        # Hårdvaru-handtag
        self._driver = None
        self.is_connected = False

        # Trådhantering och Mutex
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()

        # Senaste rådata (8x8)
        self._latest_distances: Optional[np.ndarray] = None
        self._latest_statuses: Optional[np.ndarray] = None
        self._latest_timestamp: float = 0.0

        if auto_connect:
            self.connect()

    def connect(self) -> bool:
        """
        Initierar I2C-anslutningen, konfigurerar 8x8-zoner och startar bakgrundstråden.
        Returnerar True vid framgång, False vid fel. Kraschar aldrig.
        """
        if self.is_connected:
            return True

        if not HAS_HARDWARE_DRIVER:
            if self.verbose:
                print("[ToFSensor] INFO: 'vl53l5cx_ctypes' saknas (körs på Windows/utvecklingsmiljö).")
                print("[ToFSensor] Sensorn är inte ansluten.")
            self.is_connected = False
            return False

        try:
            if self.verbose:
                print("[ToFSensor] Ansluter till VL53L5CX på I2C-1 (0x29)...")

            # Initiera ST C-drivrutinen (använder I2C-1 och 0x29 som standard)
            self._driver = vl53l5cx_ctypes.VL53L5CX()

            # Sätt 8x8 upplösning (64 zoner)
            self._driver.set_resolution(8 * 8)

            # Sätt mätfrekvens
            self._driver.set_ranging_frequency_hz(self.frequency_hz)

            # Starta kontinuerlig mätning i sensorn
            self._driver.start_ranging()

            self.is_connected = True
            self._running = True

            # Starta bakgrundstråden som pollar I2C
            self._thread = threading.Thread(
                target=self._poll_loop,
                daemon=True,
                name="ToF_I2C_Poller"
            )
            self._thread.start()

            if self.verbose:
                print(f"[ToFSensor] Ansluten och aktiv! Bakgrundstråd körs i {self.frequency_hz} Hz.")
            return True

        except Exception as e:
            if self.verbose:
                print(f"[ToFSensor] FEL: Kunde inte ansluta till sensorn: {e}")
            self.is_connected = False
            self._running = False
            self._driver = None
            return False

    def disconnect(self) -> None:
        """Stoppar bakgrundstråden och stänger I2C-kommunikationen säkert."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
            self._thread = None

        if self._driver and self.is_connected:
            try:
                self._driver.stop_ranging()
            except Exception:
                pass
            self._driver = None

        self.is_connected = False
        if self.verbose:
            print("[ToFSensor] Frånkopplad och avstängd.")

    def _poll_loop(self) -> None:
        """Bakgrundstråd: Pollar I2C efter nya mätningar och uppdaterar minnet trådsäkert."""
        sleep_interval = 1.0 / (self.frequency_hz * 2)

        while self._running and self._driver:
            try:
                if self._driver.data_ready():
                    data = self._driver.get_data()

                    # ST returnerar 64 element i rad, omforma till 8x8
                    dist_8x8 = np.array(data.distance_mm, dtype=np.float32).reshape((8, 8))
                    status_8x8 = np.array(data.target_status, dtype=np.uint8).reshape((8, 8))
                    now = time.time()

                    # Trådsäker uppdatering via Mutex
                    with self._lock:
                        self._latest_distances = dist_8x8
                        self._latest_statuses = status_8x8
                        self._latest_timestamp = now

                time.sleep(sleep_interval)
            except Exception as e:
                if self.verbose:
                    print(f"[ToFSensor] Varning i bakgrundstråd: {e}")
                time.sleep(0.05)

    def _filter_matrix(self, distances: np.ndarray, statuses: np.ndarray) -> np.ndarray:
        """
        Filtrerar bort ogiltiga mätningar baserat på ST-statuskoder.
        Zoner som inte är godkända sätts till np.nan.
        """
        cleaned = distances.copy().astype(np.float32)

        # ST status 5 och 9 indikerar 100% godkänd mätning med bra signal
        valid_mask = (statuses == 5) | (statuses == 9)
        valid_mask &= (cleaned > 0.0)

        cleaned[~valid_mask] = np.nan
        return cleaned

    def _extract_zone(self, matrix: np.ndarray, zone: str) -> np.ndarray:
        """Klipper ut önskat zon-område med match-case."""
        match zone.lower():
            case "spot":
                # De 4 allra innersta pixlarna (rad 3-4, kolumn 3-4)
                return matrix[3:5, 3:5]
            case "center":
                # De centrala 4x4 zonerna (rad 2-5, kolumn 2-5 = 16 zoner)
                return matrix[2:6, 2:6]
            case "all":
                # Alla 8x8 (64 zoner)
                return matrix
            case _:
                if self.verbose:
                    print(f"[ToFSensor] Varning: Okänd zon '{zone}', använder 'center'.")
                return matrix[2:6, 2:6]

    def _aggregate(self, values: np.ndarray, method: str) -> float:
        """Aggregerar värdena och ignorerar automatiskt alla NaN via match-case."""
        # Om hela zonen bara är NaN (alla zoner skymda)
        if np.all(np.isnan(values)):
            return -1.0

        match method.lower():
            case "median":
                return float(np.nanmedian(values))
            case "mean":
                return float(np.nanmean(values))
            case "min":
                return float(np.nanmin(values))
            case _:
                if self.verbose:
                    print(f"[ToFSensor] Varning: Okänd metod '{method}', använder 'median'.")
                return float(np.nanmedian(values))

    def read(
        self,
        samples: int = 1,
        method: str = "median",
        zone: str = "center"
    ) -> ToFReading:
        """
        Läser av det senaste mätvärdet från ToF-sensorn och returnerar en ToFReading.
        
        Om sensorn inte är ansluten returneras ett säkert ToFReading med
        distance_mm = -1.0 och is_valid = False.
        """
        if not self.is_connected or self._latest_distances is None:
            if self.verbose and not self.is_connected:
                print("[ToFSensor] FEL: Kan inte läsa, sensorn är inte ansluten.")
            return ToFReading(
                distance_mm=-1.0,
                is_valid=False,
                timestamp=time.time(),
                samples=samples,
                method=method,
                zone=zone,
                valid_zones=0,
            )

        # Hämta en trådsäker kopia av den senaste datan
        with self._lock:
            distances = self._latest_distances.copy()
            statuses = self._latest_statuses.copy()
            ts = self._latest_timestamp

        # 1. Filtrera status (sätter ogiltiga till np.nan)
        filtered_matrix = self._filter_matrix(distances, statuses)

        # 2. Välj zon
        zone_data = self._extract_zone(filtered_matrix, zone)

        # 3. Räkna giltiga zoner
        valid_count = int(np.count_nonzero(~np.isnan(zone_data)))

        # 4. Aggregera
        final_distance = self._aggregate(zone_data, method)
        is_valid = (final_distance > 0.0) and (valid_count > 0)

        return ToFReading(
            distance_mm=final_distance if is_valid else -1.0,
            is_valid=is_valid,
            timestamp=ts,
            samples=samples,
            method=method,
            zone=zone,
            valid_zones=valid_count,
        )

    def read_matrix_raw(self) -> Optional[np.ndarray]:
        """Returnerar en trådsäker kopia av den råa 8x8-matrisen i mm."""
        if not self.is_connected or self._latest_distances is None:
            return None
        with self._lock:
            return self._latest_distances.copy()

    def __enter__(self) -> "ToFSensor":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.disconnect()

