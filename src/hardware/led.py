"""
Hårdvarudrivrutin för en enskild 3W 850nm IR LED (Kanal 1-4) via Active-LOW GPIO.

Styrning sker via YYNMOS-4 MOSFET med optokopplare:
- LOW (0V): Tänd (aktiv)
- HIGH (3.3V): Släckt (inaktiv)
"""

from typing import Optional
from src.core.config import LED_CHANNELS, LED_ACTIVE_HIGH, PinMapping

try:
    from gpiozero import OutputDevice
    HAS_GPIOZERO = True
except ImportError:
    HAS_GPIOZERO = False


class LED:
    """
    Representerar en enskild fysisk IR-lysdiod (kanal 1-4).
    """

    def __init__(
        self,
        channel: int,
        verbose: bool = True,
        auto_connect: bool = True,
    ):
        """
        Kräver explicit angivande av kanalnummer (1, 2, 3 eller 4).
        Styrprincip (Active-LOW) hämtas globalt från src/core/config.py.
        """
        if channel not in LED_CHANNELS:
            raise ValueError(f"Ogiltig LED-kanal {channel}. Välj mellan {list(LED_CHANNELS.keys())}.")

        self.channel: int = channel
        self.verbose: bool = verbose
        self.active_high: bool = LED_ACTIVE_HIGH
        self.mapping: PinMapping = LED_CHANNELS[channel]

        self.bcm_pin: int = self.mapping.bcm
        self.physical_pin: int = self.mapping.physical_pin
        self.description: str = self.mapping.description

        self.is_connected: bool = False
        self._is_on: bool = False
        self._device: Optional[OutputDevice] = None

        if auto_connect:
            self.connect()

    def connect(self) -> bool:
        """
        Initierar GPIO-anslutningen för lysdioden.
        Returnerar True vid lyckad hårdvarukoppling, annars False. Kraschar aldrig.
        """
        if self.is_connected:
            return True

        if not HAS_GPIOZERO:
            if self.verbose:
                print(f"[LED {self.channel}] INFO: 'gpiozero' saknas (körs på Windows/utvecklingsmiljö).")
                print(f"[LED {self.channel}] Lampan är inte ansluten till fysisk hårdvara.")
            self.is_connected = False
            return False

        try:
            # active_high=False: on() sätter 0V (tänd), off() sätter 3.3V (släckt)
            # initial_value=False: startar alltid som släckt (3.3V på pinnen)
            self._device = OutputDevice(
                self.bcm_pin,
                active_high=self.active_high,
                initial_value=False
            )
            self.is_connected = True
            self._is_on = False
            if self.verbose:
                print(f"[LED {self.channel}] Ansluten på GPIO {self.bcm_pin} (Stift {self.physical_pin}, {self.description}).")
            return True

        except Exception as e:
            if self.verbose:
                print(f"[LED {self.channel}] FEL: Kunde inte initiera GPIO {self.bcm_pin}: {e}")
            self.is_connected = False
            self._device = None
            return False

    def disconnect(self) -> None:
        """Släcker lampan och frigör GPIO-resursen."""
        if self._device is not None:
            try:
                self._device.off()
                self._device.close()
            except Exception:
                pass
            self._device = None

        self._is_on = False
        self.is_connected = False

    def on(self) -> bool:
        """
        Tänder lysdioden.
        Returnerar True om kommandot utfördes på ansluten hårdvara, annars False.
        """
        if not self.is_connected or self._device is None:
            if self.verbose:
                print(f"[LED {self.channel}] FEL: Kan inte tända lampan, den är inte ansluten.")
            return False

        try:
            self._device.on()
            self._is_on = True
            return True
        except Exception as e:
            if self.verbose:
                print(f"[LED {self.channel}] FEL vid tändning: {e}")
            self._is_on = False
            return False

    def off(self) -> bool:
        """
        Släcker lysdioden.
        Returnerar True om kommandot utfördes på ansluten hårdvara, annars False.
        """
        if not self.is_connected or self._device is None:
            if self.verbose:
                print(f"[LED {self.channel}] FEL: Kan inte släcka lampan, den är inte ansluten.")
            return False

        try:
            self._device.off()
            self._is_on = False
            return True
        except Exception as e:
            if self.verbose:
                print(f"[LED {self.channel}] FEL vid släckning: {e}")
            return False

    @property
    def is_on(self) -> bool:
        """Returnerar True om lampan för närvarande är tänd."""
        return self._is_on

    def __enter__(self) -> "LED":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.disconnect()

    def __repr__(self) -> str:
        status = "TÄND" if self._is_on else "SLÄCKT"
        conn = "Ansluten" if self.is_connected else "Frånkopplad"
        return f"LED(Kanal {self.channel}, GPIO {self.bcm_pin}, {conn}, {status})"
