"""
Hårdvarudrivrutin för USB HID-streckkodsläsare (Honeywell MS5145 Eclipse).

Läser råa tangentbordshändelser asynkront via Linux evdev-undersystemet.
Detta förhindrar att scannade tecken skrivs ut i terminalen och tillåter
bakgrundsläsning utan att ett fönster eller terminal behöver ha fokus.

Garanterar trådsäkerhet via intern Mutex (threading.Lock) och implementerar
ett konsumentmönster (peek/consume/clear) för att förhindra spökkopplingar.
"""

import time
import threading
from typing import Optional, Callable

from src.core.models import BarcodeReading

# Försök importera evdev för Linux input-händelser
try:
    import evdev
    from evdev import ecodes
    import select
    HAS_EVDEV = True
except ImportError:
    HAS_EVDEV = False


class BarcodeScanner:
    """
    Asynkron drivrutin för USB-streckkodsläsare via Linux evdev.
    """

    def __init__(
        self,
        device_path: Optional[str] = None,
        name_filter: str = "Honeywell",
        grab_device: bool = True,
        on_scan_callback: Optional[Callable[[BarcodeReading], None]] = None,
        verbose: bool = True,
        auto_connect: bool = True,
    ):
        """
        Initierar streckkodsläsaren.

        :param device_path: Specifik sökväg (t.ex. '/dev/input/event0'). Om None söks enheten automatiskt.
        :param name_filter: Substräng att matcha enhetsnamnet mot (default: 'Honeywell').
        :param grab_device: True för att ta exklusiv kontroll (hindrar tecken från att läcka till terminalen).
        :param on_scan_callback: Frivillig callback-funktion som anropas vid godkänd scanning.
        :param verbose: True för att skriva ut statusmeddelanden i konsolen.
        :param auto_connect: True för att ansluta direkt vid initiering.
        """
        self.device_path = device_path
        self.name_filter = name_filter
        self.grab_device = grab_device
        self.on_scan_callback = on_scan_callback
        self.verbose = verbose

        # Enhet och anslutningsstatus
        self._device = None
        self.is_connected = False

        # Senaste inlästa streckkod (startar som ogiltig, aldrig None)
        self._latest_reading: BarcodeReading = BarcodeReading.invalid()

        # Trådhantering och Mutex
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()

        if auto_connect:
            self.connect()

    def connect(self) -> bool:
        """
        Upptäcker USB-scannern, tar exklusiv kontroll och startar bakgrundstråden.
        Returnerar True vid framgång, False vid fel. Kraschar aldrig.
        """
        if self.is_connected:
            return True

        if not HAS_EVDEV:
            if self.verbose:
                print("[BarcodeScanner] INFO: 'evdev' saknas (körs på Windows/utvecklingsmiljö).")
                print("[BarcodeScanner] Scannern är inte ansluten.")
            self.is_connected = False
            return False

        try:
            target_device = None

            # Om specifik sökväg angetts, använd den direkt
            if self.device_path:
                target_device = evdev.InputDevice(self.device_path)
            else:
                # Sök bland anslutna evdev-enheter
                devices = [evdev.InputDevice(path) for path in evdev.list_devices()]
                for dev in devices:
                    # Matcha mot angivet filter eller kända generiska namn
                    dev_name = dev.name.lower()
                    filter_lower = self.name_filter.lower()
                    if (
                        filter_lower in dev_name
                        or "metrologic" in dev_name
                        or "honeywell" in dev_name
                        or "barcode" in dev_name
                    ):
                        target_device = dev
                        break

            if target_device is None:
                if self.verbose:
                    print(f"[BarcodeScanner] Ingen scanner hittades som matchar '{self.name_filter}'.")
                self.is_connected = False
                return False

            self._device = target_device

            # Ta exklusivt lås på enheten så att tangenttryck inte går till konsolen
            if self.grab_device:
                try:
                    self._device.grab()
                except Exception as e:
                    if self.verbose:
                        print(f"[BarcodeScanner] Varning: Kunde inte ta exklusiv kontroll (grab): {e}")

            self.is_connected = True
            self._running = True

            # Starta bakgrundstråd som lyssnar på inkommande knapphändelser
            self._thread = threading.Thread(
                target=self._listen_loop,
                daemon=True,
                name="Barcode_Listener"
            )
            self._thread.start()

            if self.verbose:
                print(f"[BarcodeScanner] Ansluten till '{self._device.name}' på {self._device.path}!")
            return True

        except Exception as e:
            if self.verbose:
                print(f"[BarcodeScanner] Fel vid anslutning: {e}")
            self.is_connected = False
            self._device = None
            return False

    def _listen_loop(self) -> None:
        """
        Bakgrundstråd som lyssnar på tangenttryck från scannern via evdev och bygger streckkoden.
        """
        buffer = []
        shift_active = False

        # Tangentkoder till tecken
        key_map = {
            ecodes.KEY_0: ("0", ")"),
            ecodes.KEY_1: ("1", "!"),
            ecodes.KEY_2: ("2", "@"),
            ecodes.KEY_3: ("3", "#"),
            ecodes.KEY_4: ("4", "$"),
            ecodes.KEY_5: ("5", "%"),
            ecodes.KEY_6: ("6", "^"),
            ecodes.KEY_7: ("7", "&"),
            ecodes.KEY_8: ("8", "*"),
            ecodes.KEY_9: ("9", "("),
            ecodes.KEY_A: ("a", "A"),
            ecodes.KEY_B: ("b", "B"),
            ecodes.KEY_C: ("c", "C"),
            ecodes.KEY_D: ("d", "D"),
            ecodes.KEY_E: ("e", "E"),
            ecodes.KEY_F: ("f", "F"),
            ecodes.KEY_G: ("g", "G"),
            ecodes.KEY_H: ("h", "H"),
            ecodes.KEY_I: ("i", "I"),
            ecodes.KEY_J: ("j", "J"),
            ecodes.KEY_K: ("k", "K"),
            ecodes.KEY_L: ("l", "L"),
            ecodes.KEY_M: ("m", "M"),
            ecodes.KEY_N: ("n", "N"),
            ecodes.KEY_O: ("o", "O"),
            ecodes.KEY_P: ("p", "P"),
            ecodes.KEY_Q: ("q", "Q"),
            ecodes.KEY_R: ("r", "R"),
            ecodes.KEY_S: ("s", "S"),
            ecodes.KEY_T: ("t", "T"),
            ecodes.KEY_U: ("u", "U"),
            ecodes.KEY_V: ("v", "V"),
            ecodes.KEY_W: ("w", "W"),
            ecodes.KEY_X: ("x", "X"),
            ecodes.KEY_Y: ("y", "Y"),
            ecodes.KEY_Z: ("z", "Z"),
            ecodes.KEY_MINUS: ("-", "_"),
            ecodes.KEY_SLASH: ("/", "?"),
            ecodes.KEY_DOT: (".", ">"),
            ecodes.KEY_SPACE: (" ", " "),
        }

        while self._running and self._device is not None:
            try:
                # Använd select med timeout så att tråden kan avslutas mjukt
                r, _, _ = select.select([self._device.fd], [], [], 0.5)
                if not r:
                    continue

                for event in self._device.read():
                    if event.type != ecodes.EV_KEY:
                        continue

                    # Spåra Shift-tangenter (tryck ner = 1, hålls = 2, släpps = 0)
                    if event.code in (ecodes.KEY_LEFTSHIFT, ecodes.KEY_RIGHTSHIFT):
                        shift_active = (event.value in (1, 2))
                        continue

                    # Reagera endast på nedtryckning (value == 1)
                    if event.value != 1:
                        continue

                    # Om Enter trycks är streckkoden färdigläst
                    if event.code in (ecodes.KEY_ENTER, ecodes.KEY_KPENTER):
                        code_str = "".join(buffer).strip()
                        buffer.clear()

                        if code_str:
                            reading = BarcodeReading(
                                code=code_str,
                                timestamp=time.time(),
                                is_used=False
                            )
                            with self._lock:
                                self._latest_reading = reading

                            if self.verbose:
                                print(f"[BarcodeScanner] Scannad kod: '{code_str}'")

                            # Trigga eventuell callback
                            if self.on_scan_callback:
                                try:
                                    self.on_scan_callback(reading)
                                except Exception as cb_err:
                                    if self.verbose:
                                        print(f"[BarcodeScanner] Fel i on_scan_callback: {cb_err}")

                    elif event.code in key_map:
                        char_normal, char_shift = key_map[event.code]
                        buffer.append(char_shift if shift_active else char_normal)

            except Exception as e:
                if self._running and self.verbose:
                    print(f"[BarcodeScanner] Avbrott i läsloop: {e}")
                break

        self.is_connected = False

    def peek(self) -> BarcodeReading:
        """
        Inspekterar den senaste inlästa streckkoden utan att markera den som använd.
        Returnerar BarcodeReading.invalid() om ingen scanner är ansluten eller ingen färsk kod finns.
        """
        if not self.is_connected:
            if self.verbose:
                print("[BarcodeScanner] peek(): Scannern är inte ansluten.")
            return BarcodeReading.invalid()

        with self._lock:
            if not self._latest_reading.is_valid():
                if self.verbose:
                    print("[BarcodeScanner] peek(): Ingen färsk streckkod tillgänglig.")
                return BarcodeReading.invalid()
            return self._latest_reading

    def consume(self) -> BarcodeReading:
        """
        Hämtar och förbrukar den senaste streckkoden.
        Återställer intern läsning till BarcodeReading.invalid() så att koden inte kan läsas igen.
        Returnerar den giltiga koden, eller BarcodeReading.invalid() om scannern inte är ansluten
        eller om ingen färsk kod fanns.
        """
        if not self.is_connected:
            if self.verbose:
                print("[BarcodeScanner] consume(): Scannern är inte ansluten.")
            return BarcodeReading.invalid()

        with self._lock:
            if not self._latest_reading.is_valid():
                if self.verbose:
                    print("[BarcodeScanner] consume(): Ingen streckkod att förbruka.")
                return BarcodeReading.invalid()

            reading = self._latest_reading
            self._latest_reading = BarcodeReading.invalid()
            return reading

    def clear(self) -> None:
        """
        Ångra/rensa scanning: Förbrukar och kasserar den senaste koden.
        """
        self.consume()

    def disconnect(self) -> None:
        """
        Kopplar ifrån scannern, släpper exklusivt lås och stänger bakgrundstråden mjukt.
        """
        self._running = False

        if self._device is not None:
            try:
                if self.grab_device:
                    self._device.ungrab()
                self._device.close()
            except Exception:
                pass
            self._device = None

        self.is_connected = False

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)

        if self.verbose:
            print("[BarcodeScanner] Frånkopplad.")

    def __enter__(self) -> "BarcodeScanner":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.disconnect()
