"""
Kontroller och fasad för hela 3W 850nm IR LED-klustret (Kanal 1-4).

Tillhandahåller grupp- och enskild styrning för alla 4 lampor:
- all_on() / all_off() för stereofotografering
- on(channel) / off(channel) för fotometrisk stereo
- Context manager (with LEDController() as leds:) för garanterad nedsläckning
"""

from typing import Dict, List
from src.core.config import LED_CHANNELS
from src.hardware.led import LED


class LEDController:
    """
    Fasad som hanterar och orkestrerar alla 4 fysiska LED-armar.
    """

    def __init__(self, verbose: bool = True, auto_connect: bool = True):
        self.verbose: bool = verbose
        self._channels: Dict[int, LED] = {
            ch: LED(channel=ch, verbose=verbose, auto_connect=auto_connect)
            for ch in LED_CHANNELS.keys()
        }

    def connect(self) -> bool:
        """
        Initierar anslutningen för alla 4 LED-kanaler.
        Returnerar True om alla lyckades ansluta.
        """
        results = [led.connect() for led in self._channels.values()]
        return all(results)

    def disconnect(self) -> None:
        """Släcker alla lampor och frigör hårdvaruresurserna."""
        for led in self._channels.values():
            led.disconnect()

    @property
    def is_connected(self) -> bool:
        """Returnerar True om alla 4 LED-kanaler är fysiskt anslutna."""
        return all(led.is_connected for led in self._channels.values())

    def on(self, channel: int) -> bool:
        """
        Tänder en specifik LED-kanal (1-4).
        Returnerar True om lampan tändes, annars False.
        """
        if channel not in self._channels:
            raise ValueError(f"Ogiltig LED-kanal {channel}. Välj mellan {list(self._channels.keys())}.")
        return self._channels[channel].on()

    def off(self, channel: int) -> bool:
        """
        Släcker en specifik LED-kanal (1-4).
        Returnerar True om lampan släcktes, annars False.
        """
        if channel not in self._channels:
            raise ValueError(f"Ogiltig LED-kanal {channel}. Välj mellan {list(self._channels.keys())}.")
        return self._channels[channel].off()

    def is_on(self, channel: int) -> bool:
        """Returnerar True om den angivna kanalen är tänd."""
        if channel not in self._channels:
            raise ValueError(f"Ogiltig LED-kanal {channel}. Välj mellan {list(self._channels.keys())}.")
        return self._channels[channel].is_on

    def all_on(self) -> bool:
        """
        Tänder alla 4 lampor samtidigt.
        Returnerar True om alla lampor tändes.
        """
        results = [led.on() for led in self._channels.values()]
        return all(results)

    def all_off(self) -> bool:
        """
        Släcker alla 4 lampor samtidigt.
        Returnerar True om alla lampor släcktes.
        """
        results = [led.off() for led in self._channels.values()]
        return all(results)

    def __enter__(self) -> "LEDController":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.disconnect()

    def __repr__(self) -> str:
        active = [ch for ch, led in self._channels.items() if led.is_on]
        conn = "Ansluten" if self.is_connected else "Frånkopplad"
        return f"LEDController({conn}, Aktiva kanaler={active})"
