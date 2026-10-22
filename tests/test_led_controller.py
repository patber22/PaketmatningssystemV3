"""
Enhetstester för klassen LEDController (src/hardware/led_controller.py).

Verifierar fasadstyrning, kanalvalidering, enskild och gruppstyrning (all_on/all_off),
mockad hårdvarustyrning och automatisk uppstädning i context manager.
"""

import unittest
from unittest.mock import MagicMock

from src.hardware.led_controller import LEDController


class TestLEDController(unittest.TestCase):
    """Testsvit för LEDController-fasaden."""

    def test_initialization(self):
        """Verifiera att controllern initierar exakt de 4 kanalerna 1, 2, 3 och 4."""
        ctrl = LEDController(verbose=False, auto_connect=False)
        self.assertEqual(sorted(list(ctrl._channels.keys())), [1, 2, 3, 4])
        self.assertFalse(ctrl.is_connected)
        for ch in [1, 2, 3, 4]:
            self.assertFalse(ctrl.is_on(ch))

    def test_invalid_channel_raises_value_error(self):
        """Verifiera att ogiltiga kanalnummer kastar ValueError för on, off och is_on."""
        ctrl = LEDController(verbose=False, auto_connect=False)
        for invalid_ch in [0, 5, -1, 99]:
            with self.assertRaises(ValueError):
                ctrl.on(invalid_ch)
            with self.assertRaises(ValueError):
                ctrl.off(invalid_ch)
            with self.assertRaises(ValueError):
                ctrl.is_on(invalid_ch)

    def test_disconnected_behavior(self):
        """Verifiera att kommandona returnerar False när hårdvara saknas."""
        ctrl = LEDController(verbose=False, auto_connect=False)
        self.assertFalse(ctrl.is_connected)
        self.assertFalse(ctrl.on(1))
        self.assertFalse(ctrl.off(1))
        self.assertFalse(ctrl.all_on())
        self.assertFalse(ctrl.all_off())

    def test_mocked_individual_channels(self):
        """Verifiera att enskild kanal kan tändas och släckas oberoende av de andra."""
        ctrl = LEDController(verbose=False, auto_connect=False)

        mock_devices = {}
        for ch in [1, 2, 3, 4]:
            dev = MagicMock()
            ctrl._channels[ch]._device = dev
            ctrl._channels[ch].is_connected = True
            mock_devices[ch] = dev

        self.assertTrue(ctrl.is_connected)

        # Tänd enbart kanal 2
        self.assertTrue(ctrl.on(2))
        self.assertTrue(ctrl.is_on(2))
        self.assertFalse(ctrl.is_on(1))
        self.assertFalse(ctrl.is_on(3))
        self.assertFalse(ctrl.is_on(4))
        mock_devices[2].on.assert_called_once()
        mock_devices[1].on.assert_not_called()

        # Släck kanal 2
        self.assertTrue(ctrl.off(2))
        self.assertFalse(ctrl.is_on(2))
        mock_devices[2].off.assert_called_once()

    def test_mocked_all_on_and_all_off(self):
        """Verifiera att all_on() och all_off() styr samtliga 4 kanaler."""
        ctrl = LEDController(verbose=False, auto_connect=False)

        mock_devices = {}
        for ch in [1, 2, 3, 4]:
            dev = MagicMock()
            ctrl._channels[ch]._device = dev
            ctrl._channels[ch].is_connected = True
            mock_devices[ch] = dev

        # Tänd alla 4
        self.assertTrue(ctrl.all_on())
        for ch in [1, 2, 3, 4]:
            self.assertTrue(ctrl.is_on(ch))
            mock_devices[ch].on.assert_called_once()

        # Släck alla 4
        self.assertTrue(ctrl.all_off())
        for ch in [1, 2, 3, 4]:
            self.assertFalse(ctrl.is_on(ch))
            mock_devices[ch].off.assert_called_once()

    def test_context_manager_cleanup(self):
        """Verifiera att context manager garanterat släcker och kopplar ifrån alla lampor."""
        mock_devices = {}
        with LEDController(verbose=False, auto_connect=False) as ctrl:
            for ch in [1, 2, 3, 4]:
                dev = MagicMock()
                ctrl._channels[ch]._device = dev
                ctrl._channels[ch].is_connected = True
                mock_devices[ch] = dev

            ctrl.all_on()
            self.assertTrue(ctrl.is_on(1))

        # Efter with-blocket ska allt vara släckt och frånkopplat
        self.assertFalse(ctrl.is_connected)
        for ch in [1, 2, 3, 4]:
            self.assertFalse(ctrl.is_on(ch))
            mock_devices[ch].close.assert_called_once()

    def test_repr_output(self):
        """Verifiera att __repr__ formateras korrekt utan fel."""
        ctrl = LEDController(verbose=False, auto_connect=False)
        repr_str = repr(ctrl)
        self.assertIn("LEDController", repr_str)
        self.assertIn("Frånkopplad", repr_str)


if __name__ == "__main__":
    unittest.main()
