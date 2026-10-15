"""
Hårdvarudrivrutin för ST VL53L5CX Time-of-Flight (8x8 zoner) över Linux I2C.

Kör en asynkron bakgrundstråd som kontinuerligt pollar sensorn i 15 Hz för att
hålla en färsk 8x8-matris uppdaterad med 0 ms svarstid.
Garanterar trådsäkerhet via en intern Mutex (threading.Lock).
"""

import time
import threading
import warnings
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

        # Mätorder-synkronisering (servitör/kock)
        self._target_samples: int = 0
        self._collected_distances: list = []
        self._collected_statuses: list = []
        self._order_event = threading.Event()

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
        self._order_event.set()  # Väck eventuellt väntande read() så den inte blir hängande

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

                        # Om en mätorder är aktiv (servitören väntar på samples)
                        if self._target_samples > 0:
                            self._collected_distances.append(dist_8x8)
                            self._collected_statuses.append(status_8x8)

                            # Om vi nått önskat antal samples: slå på serveringsklockan!
                            if len(self._collected_distances) >= self._target_samples:
                                self._target_samples = 0
                                self._order_event.set()

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

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            match method.lower():
                case "median":
                    return float(np.nanmedian(values))
                case "mean" | "average":
                    return float(np.nanmean(values))
                case "min":
                    return float(np.nanmin(values))
                case _:
                    if self.verbose:
                        print(f"[ToFSensor] Varning: Okänd metod '{method}', använder 'median'.")
                    return float(np.nanmedian(values))

    def _temporal_aggregate(self, stack: np.ndarray, method: str) -> np.ndarray:
        """
        Kör Steg 1: Komprimerar tidsaxeln (axis 0) i en 3D-matris (N, 8, 8) till en 2D (8, 8)-matris.
        Hanterar automatiskt NaN-värden så att enstaka brusiga zoner i en mätning inte förstör helheten.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            match method.lower():
                case "mean" | "average":
                    result = np.nanmean(stack, axis=0)
                case "min":
                    result = np.nanmin(stack, axis=0)
                case "median" | _:
                    result = np.nanmedian(stack, axis=0)
        return result

    def read(
        self,
        samples: int = 1,
        sample_method: str = "median",
        zone: str = "center",
        zone_method: str = "median",
        timeout: Optional[float] = None,
        **kwargs
    ) -> ToFReading:
        """
        Läser av mätvärden från ToF-sensorn med tvåstegs brusreducering:
          Steg 1 (Tidsfilter): Samlar N bildrutor och komprimerar över tid med sample_method till en ren 8x8-matris.
          Steg 2 (Rumsfilter): Klipper ut specificerad zon och beräknar slutligt avstånd med zone_method.

        :param samples: Antal färska 8x8-bildrutor att samla in över tid (>= 1).
        :param sample_method: Metod för tidsbrus ('median', 'mean', 'min').
        :param zone: Zon-urval ('center', 'spot', 'all').
        :param zone_method: Metod för rumsaggregering ('median', 'mean', 'min').
        :param timeout: Max väntetid i sekunder (default: dynamisk baserad på samples och frekvens).
        :return: ToFReading med det beräknade mätvärdet och metadata.
        """
        # Bakåtkompatibilitet: om 'method' skickas som keyword-argument, tolka det som zone_method
        if "method" in kwargs:
            zone_method = kwargs["method"]

        samples = max(1, samples)

        if not self.is_connected:
            if self.verbose:
                print("[ToFSensor] FEL: Kan inte läsa, sensorn är inte ansluten.")
            return ToFReading(
                distance_mm=-1.0,
                is_valid=False,
                timestamp=time.time(),
                samples=samples,
                method=zone_method,
                zone=zone,
                valid_zones=0,
            )

        # Hämta de N matriserna
        if not self._running:
            # Offline / Mock-läge (t.ex. vid enhetstester där bakgrundstråden inte snurrar)
            with self._lock:
                if self._latest_distances is None or self._latest_statuses is None:
                    return ToFReading(
                        distance_mm=-1.0,
                        is_valid=False,
                        timestamp=time.time(),
                        samples=samples,
                        method=zone_method,
                        zone=zone,
                        valid_zones=0,
                    )
                distances_list = [self._latest_distances.copy() for _ in range(samples)]
                statuses_list = [self._latest_statuses.copy() for _ in range(samples)]
                ts = self._latest_timestamp
        else:
            # Aktiv hårdvara med bakgrundstråd: Lägg en mätorder till kocken
            if timeout is None:
                # Beräkna rimlig timeout: förväntad tid + 1.5 sekunders marginal
                timeout = (samples / self.frequency_hz) + 1.5

            with self._lock:
                self._collected_distances = []
                self._collected_statuses = []
                self._target_samples = samples
                self._order_event.clear()

            # Servitören väntar på att kocken slår på serveringsklockan
            completed = self._order_event.wait(timeout=timeout)

            if not completed:
                if self.verbose:
                    print(f"[ToFSensor] Timeout: Fick inte {samples} samples inom {timeout:.2f}s.")
                with self._lock:
                    self._target_samples = 0
                return ToFReading(
                    distance_mm=-1.0,
                    is_valid=False,
                    timestamp=time.time(),
                    samples=samples,
                    method=zone_method,
                    zone=zone,
                    valid_zones=0,
                )

            with self._lock:
                distances_list = [d.copy() for d in self._collected_distances]
                statuses_list = [s.copy() for s in self._collected_statuses]
                ts = self._latest_timestamp

        # 1. Filtrera varje bildruta mot ST:s hårdvarustatus (ogiltiga sätts till np.nan)
        filtered_frames = [
            self._filter_matrix(dist, stat)
            for dist, stat in zip(distances_list, statuses_list)
        ]
        stack = np.stack(filtered_frames, axis=0)  # Form: (N, 8, 8)

        # 2. STEG 1: Tidsfilter över de N matriserna -> ger en enda brusfri 8x8-matris
        temporal_matrix = self._temporal_aggregate(stack, sample_method)

        # 3. STEG 2: Rumsfilter (klipp ut zon och beräkna slutgiltigt millimeteravstånd)
        zone_data = self._extract_zone(temporal_matrix, zone)
        valid_count = int(np.count_nonzero(~np.isnan(zone_data)))

        final_distance = self._aggregate(zone_data, zone_method)
        is_valid = (final_distance > 0.0) and (valid_count > 0)

        return ToFReading(
            distance_mm=final_distance if is_valid else -1.0,
            is_valid=is_valid,
            timestamp=ts,
            samples=samples,
            method=zone_method,
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

