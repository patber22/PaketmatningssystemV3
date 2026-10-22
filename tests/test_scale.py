"""
Enhetstester för Scale-vagdrivrutinen (src/hardware/scale.py).

Testar initiering, kalibreringsinläsning, viktberäkningar, dödband,
stabilitetsdetektering och tarering med mockade mätvärden.
"""

import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from src.core.models import WeightReading
from src.hardware.scale import Scale


class TestScale(unittest.TestCase):
    """Testsvit för Scale-klassen."""

    def setUp(self):
        """Skapa en isolerad Scale-instans utan hårdvaruanslutning."""
        self.scale = Scale(
            auto_connect=False,
            verbose=False,
            config_path="/non/existent/path/scale.json",
        )

    def tearDown(self):
        """Stäng ner eventuella resurser efter varje test."""
        self.scale.disconnect()

    def test_default_init(self):
        """Verifiera standardtillstånd vid initiering utan hårdvara."""
        self.assertFalse(self.scale.is_connected)
        self.assertFalse(self.scale.is_calibrated)
        self.assertEqual(self.scale.tare_offset_raw, 0.0)
        self.assertEqual(self.scale.scale_factor, 1.0)
        self.assertEqual(self.scale.stability_threshold_g, 5.0)
        self.assertEqual(self.scale.buffer_size, 10)

    def test_load_calibration_success(self):
        """Verifiera att kalibreringsdata läses in korrekt från en giltig JSON-fil."""
        cal_data = {
            "tare_offset_raw": 12345.0,
            "scale_factor": 25.5,
            "reference_weight_g": 2000.0,
            "updated_at": "2026-10-22 12:00:00",
        }
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json", encoding="utf-8") as f:
            json.dump(cal_data, f)
            temp_path = f.name

        try:
            loaded = self.scale.load_calibration(temp_path)
            self.assertTrue(loaded)
            self.assertTrue(self.scale.is_calibrated)
            self.assertEqual(self.scale.tare_offset_raw, 12345.0)
            self.assertEqual(self.scale.scale_factor, 25.5)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_load_calibration_missing_or_corrupt(self):
        """Verifiera graciös hantering när kalibreringsfilen saknas eller är korrupt."""
        # Saknad fil
        self.assertFalse(self.scale.load_calibration("/ogiltig/fil/som/inte/finns.json"))
        self.assertFalse(self.scale.is_calibrated)

        # Korrupt fil
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json", encoding="utf-8") as f:
            f.write("{ogiltig json: ")
            temp_path = f.name

        try:
            self.assertFalse(self.scale.load_calibration(temp_path))
            self.assertFalse(self.scale.is_calibrated)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_read_empty_buffer(self):
        """Verifiera att read() returnerar en ogiltig WeightReading om bufferten är tom."""
        reading = self.scale.read()
        self.assertIsInstance(reading, WeightReading)
        self.assertFalse(reading.is_valid())
        self.assertEqual(reading.weight_g, -1.0)
        self.assertFalse(reading.is_stable)

    def test_read_weight_calculation(self):
        """Verifiera beräkning av vikt från råvärden via skalfaktor och tara."""
        self.scale.is_calibrated = True
        self.scale.tare_offset_raw = 1000.0
        self.scale.scale_factor = 10.0  # 10 counts per gram

        # Injicera råvärden som motsvarar (6000 - 1000) / 10 = 500.0 g
        with self.scale._lock:
            self.scale._raw_buffer.extend([6000, 6000, 6000, 6000, 6000])
            self.scale._latest_raw = 6000
            self.scale._latest_time = 1700000000.0

        reading = self.scale.read()
        self.assertTrue(reading.is_valid())
        self.assertAlmostEqual(reading.weight_g, 500.0, places=1)
        self.assertEqual(reading.raw_value, 6000)
        self.assertTrue(reading.is_stable)

    def test_read_deadband_clipping(self):
        """Verifiera att små fluktuationer under +/- 2.0g klipps till exakt 0.0g."""
        self.scale.is_calibrated = True
        self.scale.tare_offset_raw = 1000.0
        self.scale.scale_factor = 10.0

        # Råvärde 1015 motsvarar +1.5 g (inom dödbandet +/- 2.0g)
        with self.scale._lock:
            self.scale._raw_buffer.extend([1015, 1015, 1015, 1015])
            self.scale._latest_raw = 1015
            self.scale._latest_time = 1700000000.0

        reading = self.scale.read()
        self.assertEqual(reading.weight_g, 0.0)
        self.assertTrue(reading.is_valid())  # 0.0g är en giltig nollad mätning

    def test_read_negative_weight(self):
        """Verifiera att negativ vikt (under nollpunkten) markeras som ogiltig."""
        self.scale.is_calibrated = True
        self.scale.tare_offset_raw = 1000.0
        self.scale.scale_factor = 10.0

        # Råvärde 900 motsvarar -10.0 g
        with self.scale._lock:
            self.scale._raw_buffer.extend([900, 900, 900, 900])
            self.scale._latest_raw = 900
            self.scale._latest_time = 1700000000.0

        reading = self.scale.read()
        self.assertAlmostEqual(reading.weight_g, -10.0, places=1)
        self.assertFalse(reading.is_valid())  # Negativ vikt ska inte godkännas

    def test_read_zero_or_invalid_scale_factor(self):
        """Verifiera att read() inte kraschar om scale_factor är 0 (skydd mot nolldivision)."""
        self.scale.scale_factor = 0.0
        with self.scale._lock:
            self.scale._raw_buffer.extend([5000, 5000, 5000, 5000])
            self.scale._latest_raw = 5000
            self.scale._latest_time = 1700000000.0

        reading = self.scale.read()
        self.assertEqual(reading.weight_g, 0.0)
        self.assertFalse(reading.is_stable)

    def test_stability_detection(self):
        """Verifiera att is_stable kräver >= 4 mätningar och låg standardavvikelse."""
        self.scale.is_calibrated = True
        self.scale.tare_offset_raw = 0.0
        self.scale.scale_factor = 10.0
        self.scale.stability_threshold_g = 5.0

        # 1. Färre än 4 prover ska ge is_stable = False
        with self.scale._lock:
            self.scale._raw_buffer.clear()
            self.scale._raw_buffer.extend([1000, 1000, 1000])  # Endast 3 prover
            self.scale._latest_raw = 1000
            self.scale._latest_time = 1700000000.0

        reading = self.scale.read()
        self.assertFalse(reading.is_stable)

        # 2. 5 prover med hög spridning (brus/rörelse på vågen)
        with self.scale._lock:
            self.scale._raw_buffer.clear()
            # Motsvarar 100g, 200g, 50g, 180g -> stor spridning
            self.scale._raw_buffer.extend([1000, 2000, 500, 1800, 1200])
            self.scale._latest_raw = 1200
            self.scale._latest_time = 1700000000.0

        reading = self.scale.read()
        self.assertFalse(reading.is_stable)

        # 3. 5 prover med mycket låg spridning (stabil vikt)
        with self.scale._lock:
            self.scale._raw_buffer.clear()
            # Motsvarar ca 100.0g till 100.2g (brus under 1 gram)
            self.scale._raw_buffer.extend([1000, 1002, 1001, 1000, 1001])
            self.scale._latest_raw = 1001
            self.scale._latest_time = 1700000000.0

        reading = self.scale.read()
        self.assertTrue(reading.is_stable)

    def test_tare(self):
        """Verifiera att tare() samlar in värden och uppdaterar tare_offset_raw i minnet."""
        # Förbered mockade mätningar som uppdateras över tid
        self.scale._latest_raw = 5500

        # Kör tare med 3 prover
        success = self.scale.tare(samples=3, timeout_s=1.0)
        self.assertTrue(success)
        self.assertEqual(self.scale.tare_offset_raw, 5500.0)

    def test_tare_timeout(self):
        """Verifiera att tare() returnerar False om inga mätvärden kan samlas in."""
        self.scale._latest_raw = None
        success = self.scale.tare(samples=5, timeout_s=0.1)
        self.assertFalse(success)

    def test_read_stable_success(self):
        """Verifiera att read_stable returnerar den stabila mätningen direkt."""
        self.scale.is_calibrated = True
        self.scale.tare_offset_raw = 0.0
        self.scale.scale_factor = 10.0

        with self.scale._lock:
            self.scale._raw_buffer.extend([5000, 5001, 5000, 5002])
            self.scale._latest_raw = 5002
            self.scale._latest_time = 1700000000.0

        reading = self.scale.read_stable(timeout_s=0.5)
        self.assertTrue(reading.is_valid())
        self.assertTrue(reading.is_stable)

    def test_read_stable_timeout(self):
        """Verifiera att read_stable timeoutar och returnerar instabil mätning vid brus."""
        self.scale.is_calibrated = True
        self.scale.tare_offset_raw = 0.0
        self.scale.scale_factor = 10.0

        with self.scale._lock:
            self.scale._raw_buffer.extend([1000, 9000, 2000, 8000])  # Kraftigt instabil
            self.scale._latest_raw = 8000
            self.scale._latest_time = 1700000000.0

        reading = self.scale.read_stable(timeout_s=0.1, poll_interval_s=0.02)
        self.assertFalse(reading.is_stable)

    def test_context_manager(self):
        """Verifiera att context managern anropar disconnect() vid exit."""
        scale = Scale(auto_connect=False, verbose=False)
        scale.disconnect = MagicMock()

        with scale as s:
            self.assertEqual(s, scale)

        scale.disconnect.assert_called_once()


if __name__ == "__main__":
    unittest.main()
