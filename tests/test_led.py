"""
Enhetstester för klassen LED (src/hardware/led.py).

Verifierar obligatoriskt kanalnummer, config-uppslag, beteende i frånkopplat läge,
mockad hårdvarustyrning, felhantering och context manager utan krav på fysisk Raspberry Pi.
"""

import unittest
from unittest.mock import MagicMock

from src.hardware.led import LED
from src.core.config import LED_CHANNELS


class TestLED(unittest.TestCase):
    """Testsvit för LED-aktuatorn."""

    def test_channel_is_required(self):
        """Verifiera att konstruktorn kräver kanalnummer och inte har någon farlig default."""
        with self.assertRaises(TypeError):
            LED()  # type: ignore

    def test_invalid_channel_raises_value_error(self):
        """Verifiera att ogiltiga kanalnummer (t.ex. 0, 5, -1, 99) kastar ValueError."""
        for invalid_ch in [0, 5, -1, 99, 100]:
            with self.assertRaises(ValueError):
                LED(channel=invalid_ch, verbose=False, auto_connect=False)

    def test_valid_channel_attributes_match_config(self):
        """Verifiera att kanal 1-4 hämtar rätt BCM, stift och beskrivning från src/core/config.py."""
        for ch in [1, 2, 3, 4]:
            led = LED(channel=ch, verbose=False, auto_connect=False)
            expected = LED_CHANNELS[ch]
            self.assertEqual(led.channel, ch)
            self.assertEqual(led.bcm_pin, expected.bcm)
            self.assertEqual(led.physical_pin, expected.physical_pin)
            self.assertEqual(led.description, expected.description)
            self.assertFalse(led.active_high)
            self.assertFalse(led.is_connected)
            self.assertFalse(led.is_on)

    def test_disconnected_behavior(self):
        """Verifiera att on() och off() returnerar False när lampan inte är ansluten till hårdvara."""
        led = LED(channel=1, verbose=False, auto_connect=False)
        self.assertFalse(led.is_connected)

        # on() och off() ska returnera False och inte ändra is_on
        self.assertFalse(led.on())
        self.assertFalse(led.is_on)

        self.assertFalse(led.off())
        self.assertFalse(led.is_on)

    def test_mocked_hardware_lifecycle(self):
        """Verifiera tändning, släckning och frånkoppling när enhetsobjekt finns."""
        led = LED(channel=1, verbose=False, auto_connect=False)

        mock_device = MagicMock()
        led._device = mock_device
        led.is_connected = True

        # Tänd lampan
        self.assertTrue(led.on())
        self.assertTrue(led.is_on)
        mock_device.on.assert_called_once()

        # Släck lampan
        self.assertTrue(led.off())
        self.assertFalse(led.is_on)
        mock_device.off.assert_called_once()

        # Frånkoppling ska släcka och stänga
        led.disconnect()
        self.assertFalse(led.is_connected)
        self.assertFalse(led.is_on)
        self.assertIsNone(led._device)
        mock_device.close.assert_called_once()

    def test_device_exception_handling(self):
        """Verifiera att hårdvarufel i OutputDevice fångas mjukt och returnerar False."""
        led = LED(channel=2, verbose=False, auto_connect=False)
        mock_device = MagicMock()
        mock_device.on.side_effect = RuntimeError("GPIO bus locked")
        mock_device.off.side_effect = RuntimeError("GPIO bus locked")

        led._device = mock_device
        led.is_connected = True

        # on() kraschar inte
        result_on = led.on()
        self.assertFalse(result_on)
        self.assertFalse(led.is_on)

        # off() kraschar inte
        result_off = led.off()
        self.assertFalse(result_off)

    def test_context_manager_cleanup(self):
        """Verifiera att with-blocket anropar disconnect() vid avslut."""
        mock_device = MagicMock()
        with LED(channel=3, verbose=False, auto_connect=False) as led:
            led._device = mock_device
            led.is_connected = True
            led.on()
            self.assertTrue(led.is_on)

        # Efter with-blocket ska led vara frånkopplad och släckt
        self.assertFalse(led.is_connected)
        self.assertFalse(led.is_on)
        self.assertIsNone(led._device)
        mock_device.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
