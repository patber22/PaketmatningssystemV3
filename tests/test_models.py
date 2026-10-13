"""
Enhetstester för kärndataklasser i src/core/models.py.
Körs blixtsnabbt och automatiskt utan hårdvaruberoenden.
"""

import time
import unittest

from src.core.models import ToFReading


class TestToFReading(unittest.TestCase):
    """Testsvit för ToFReading dataklassen."""

    def test_default_values(self):
        """Verifiera att standardvärden är säkra och ogiltiga vid tom initiering."""
        reading = ToFReading()
        self.assertEqual(reading.distance_mm, 0.0)
        self.assertFalse(reading.is_valid)
        self.assertEqual(reading.timestamp, 0.0)
        self.assertEqual(reading.samples, 1)
        self.assertEqual(reading.method, "median")
        self.assertEqual(reading.zone, "center")
        self.assertEqual(reading.valid_zones, 0)
        # En tom/ogiltig läsning ska alltid räknas som utgången
        self.assertTrue(reading.is_expired())

    def test_to_dict_and_from_dict_roundtrip(self):
        """Verifiera att data bevaras 100% korrekt genom to_dict och from_dict."""
        original = ToFReading(
            distance_mm=952.456,
            is_valid=True,
            timestamp=1728842400.0,
            samples=5,
            method="median",
            zone="center",
            valid_zones=16
        )

        data = original.to_dict()
        # Verifiera avrundning till 2 decimaler
        self.assertEqual(data["distance_mm"], 952.46)
        self.assertTrue(data["is_valid"])

        # Återskapa från dictionaryn
        restored = ToFReading.from_dict(data)
        self.assertEqual(restored.distance_mm, 952.46)
        self.assertTrue(restored.is_valid)
        self.assertEqual(restored.timestamp, 1728842400.0)
        self.assertEqual(restored.samples, 5)
        self.assertEqual(restored.method, "median")
        self.assertEqual(restored.zone, "center")
        self.assertEqual(restored.valid_zones, 16)

    def test_from_dict_malformed_data(self):
        """Verifiera att felaktiga typer och saknade fält aldrig kraschar koden."""
        # Test 1: Tom dict
        r1 = ToFReading.from_dict({})
        self.assertFalse(r1.is_valid)
        self.assertEqual(r1.distance_mm, 0.0)

        # Test 2: Värden är None istället för siffror
        bad_data_none = {
            "distance_mm": None,
            "is_valid": True,
            "timestamp": None,
            "samples": None
        }
        r2 = ToFReading.from_dict(bad_data_none)
        self.assertFalse(r2.is_valid)

        # Test 3: Värden är felaktiga strängar
        bad_data_str = {
            "distance_mm": "inte_en_siffra",
            "is_valid": True
        }
        r3 = ToFReading.from_dict(bad_data_str)
        self.assertFalse(r3.is_valid)

        # Test 4: Indata är inte ens en dict (t.ex. None eller en lista)
        r4 = ToFReading.from_dict(None)  # type: ignore
        self.assertFalse(r4.is_valid)

    def test_is_expired_logic(self):
        """Verifiera tidslogiken med timmar."""
        now = time.time()

        # En mätning gjord för 2 timmar sedan
        recent_reading = ToFReading(
            distance_mm=950.0,
            is_valid=True,
            timestamp=now - (2 * 3600)
        )
        # Den ska INTE vara utgången om gränsen är 24 timmar
        self.assertFalse(recent_reading.is_expired(max_age_hours=24.0))
        # Men den SKA vara utgången om gränsen är 1 timme
        self.assertTrue(recent_reading.is_expired(max_age_hours=1.0))

        # En mätning gjord för 10 dagar sedan (240 timmar)
        old_reading = ToFReading(
            distance_mm=950.0,
            is_valid=True,
            timestamp=now - (240 * 3600)
        )
        # Ska vara utgången mot default 168 timmar (7 dagar)
        self.assertTrue(old_reading.is_expired())


if __name__ == "__main__":
    unittest.main()
