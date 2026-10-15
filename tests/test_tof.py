"""
Enhetstester för ToFSensor-klassen (src/hardware/tof.py).
Verifierar filtrering, zonurval, aggregering och felhantering isolerat.
Körs blixtsnabbt utan krav på fysisk I2C-hårdvara.
"""

import unittest
import numpy as np

from src.hardware.tof import ToFSensor
from src.core.models import ToFReading


class TestToFSensor(unittest.TestCase):
    """Testsvit för ToFSensor-klassens logik och databehandling."""

    def setUp(self):
        """Körs före varje test: skapa en frånkopplad sensorinstans."""
        self.sensor = ToFSensor(verbose=False, auto_connect=False)

    def test_disconnected_behavior(self):
        """Verifiera att en frånkopplad sensor ger säkra felvärden och aldrig kraschar."""
        self.assertFalse(self.sensor.is_connected)
        
        reading = self.sensor.read()
        self.assertIsInstance(reading, ToFReading)
        self.assertEqual(reading.distance_mm, -1.0)
        self.assertFalse(reading.is_valid)
        self.assertEqual(reading.valid_zones, 0)
        
        # Råmatris ska vara None när sensorn inte är ansluten
        self.assertIsNone(self.sensor.read_matrix_raw())

    def test_filter_matrix(self):
        """Verifiera att _filter_matrix sätter ogiltiga statuskoder och <= 0 till NaN."""
        # Skapa en 8x8 testmatris med 500 mm överallt
        distances = np.full((8, 8), 500.0, dtype=np.float32)
        # Sätt en pixel till 0 mm (ska bli NaN)
        distances[0, 0] = 0.0

        # Skapa statusmatris: 5 och 9 är giltiga, övriga ogiltiga
        statuses = np.zeros((8, 8), dtype=np.uint8)
        statuses[2, 2] = 5   # Giltig status
        statuses[3, 3] = 9   # Giltig status
        statuses[4, 4] = 255 # Ogiltig status (t.ex. ingen reflektion)
        statuses[0, 0] = 5   # Giltig status men avstånd 0 -> ska bli NaN

        filtered = self.sensor._filter_matrix(distances, statuses)

        # Kontrollera formen
        self.assertEqual(filtered.shape, (8, 8))
        # Status 5 och 9 med avstånd > 0 ska behållas
        self.assertEqual(filtered[2, 2], 500.0)
        self.assertEqual(filtered[3, 3], 500.0)
        # Ogiltig status ska bli NaN
        self.assertTrue(np.isnan(filtered[4, 4]))
        # Avstånd 0 ska bli NaN trots giltig status
        self.assertTrue(np.isnan(filtered[0, 0]))
        # Resterande zoner (status 0) ska vara NaN
        self.assertTrue(np.isnan(filtered[1, 1]))

    def test_extract_zone(self):
        """Verifiera att zon-urklippning skär ut rätt delar av matrisen via match-case."""
        # Skapa en 8x8 matris med talen 0 till 63
        matrix = np.arange(64, dtype=np.float32).reshape((8, 8))

        # Test 1: "center" (ska vara 4x4, rader 2-5, kolumner 2-5)
        center = self.sensor._extract_zone(matrix, "center")
        self.assertEqual(center.shape, (4, 4))
        self.assertEqual(center[0, 0], matrix[2, 2])
        self.assertEqual(center[3, 3], matrix[5, 5])

        # Test 2: "spot" (ska vara 2x2, rader 3-4, kolumner 3-4)
        spot = self.sensor._extract_zone(matrix, "spot")
        self.assertEqual(spot.shape, (2, 2))
        self.assertEqual(spot[0, 0], matrix[3, 3])
        self.assertEqual(spot[1, 1], matrix[4, 4])

        # Test 3: "all" (ska vara hela 8x8 matrisen)
        all_zones = self.sensor._extract_zone(matrix, "all")
        self.assertEqual(all_zones.shape, (8, 8))
        np.testing.assert_array_equal(all_zones, matrix)

        # Test 4: Okänd zon (ska falla tillbaka på "center")
        unknown = self.sensor._extract_zone(matrix, "ogiltig_zon")
        self.assertEqual(unknown.shape, (4, 4))

    def test_aggregate_methods(self):
        """Verifiera att median, mean och min räknar rätt och ignorerar NaN."""
        # Värden med en NaN inblandad: [100.0, 200.0, 300.0, NaN]
        data = np.array([100.0, 200.0, 300.0, np.nan], dtype=np.float32)

        self.assertEqual(self.sensor._aggregate(data, "median"), 200.0)
        self.assertEqual(self.sensor._aggregate(data, "mean"), 200.0)
        self.assertEqual(self.sensor._aggregate(data, "min"), 100.0)

        # Om alla värden är NaN (skymd sensor) -> ska returnera -1.0
        all_nan = np.full((4, 4), np.nan, dtype=np.float32)
        self.assertEqual(self.sensor._aggregate(all_nan, "median"), -1.0)

    def test_read_with_simulated_data(self):
        """Simulera en ansluten sensor med injicerad data och verifiera ToFReading."""
        # Injicera simulerad 8x8-data i sensorns minne
        mock_dist = np.full((8, 8), 750.0, dtype=np.float32)
        mock_status = np.full((8, 8), 5, dtype=np.uint8)

        # Gör 2 zoner i mitten ogiltiga för att testa filtrering
        mock_status[3, 3] = 0
        mock_status[4, 4] = 0

        self.sensor.is_connected = True
        self.sensor._latest_distances = mock_dist
        self.sensor._latest_statuses = mock_status
        self.sensor._latest_timestamp = 1729000000.0

        # Läs av mitten (4x4 = 16 zoner, 2 är ogiltiga -> ska bli 14 giltiga zoner)
        reading = self.sensor.read(method="median", zone="center")

        self.assertTrue(reading.is_valid)
        self.assertEqual(reading.distance_mm, 750.0)
        self.assertEqual(reading.valid_zones, 14)
        self.assertEqual(reading.zone, "center")
        self.assertEqual(reading.method, "median")

        # Testa även read_matrix_raw
        raw = self.sensor.read_matrix_raw()
        self.assertIsNotNone(raw)
        self.assertEqual(raw.shape, (8, 8))


if __name__ == "__main__":
    unittest.main()
