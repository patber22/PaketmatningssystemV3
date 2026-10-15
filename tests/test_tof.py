"""
Enhetstester för ToFSensor-klassen (src/hardware/tof.py).
Verifierar filtrering, tvåstegs brusreducering (tids- och rumsfilter),
mätorder-synkronisering och felhantering isolerat utan krav på I2C-hårdvara.
"""

import unittest
import threading
import time
import numpy as np

from src.hardware.tof import ToFSensor
from src.core.models import ToFReading


class TestToFSensor(unittest.TestCase):
    """Testsvit för ToFSensor-klassens logik och databehandling."""

    def setUp(self):
        """Körs före varje test: skapa en frånkopplad sensorinstans."""
        self.sensor = ToFSensor(verbose=False, auto_connect=False)

    def tearDown(self):
        """Körs efter varje test: säkerställ att sensorn kopplas ifrån."""
        self.sensor.disconnect()

    def test_disconnected_behavior(self):
        """Verifiera att en frånkopplad sensor ger säkra felvärden och aldrig kraschar."""
        self.assertFalse(self.sensor.is_connected)

        reading = self.sensor.read()
        self.assertIsInstance(reading, ToFReading)
        self.assertEqual(reading.distance_mm, -1.0)
        self.assertFalse(reading.is_valid())
        self.assertEqual(reading.valid_zones, 0)
        self.assertEqual(reading.samples, 0)

        # read_matrix ska returnera None när sensorn inte är ansluten
        self.assertIsNone(self.sensor.read_matrix())

    def test_filter_matrix(self):
        """Verifiera att _filter_matrix sätter ogiltiga statuskoder och <= 0 till NaN."""
        # Skapa en 8x8 testmatris med 500 mm överallt
        distances = np.full((8, 8), 500.0, dtype=np.float32)
        distances[0, 0] = 0.0  # 0 mm -> ska bli NaN

        # Skapa statusmatris: 5 och 9 är giltiga, övriga ogiltiga
        statuses = np.zeros((8, 8), dtype=np.uint8)
        statuses[2, 2] = 5   # Giltig status
        statuses[3, 3] = 9   # Giltig status
        statuses[4, 4] = 255 # Ogiltig status (ingen reflektion)
        statuses[0, 0] = 5   # Giltig status men 0 mm -> ska bli NaN

        filtered = self.sensor._filter_matrix(distances, statuses)

        self.assertEqual(filtered.shape, (8, 8))
        self.assertEqual(filtered[2, 2], 500.0)
        self.assertEqual(filtered[3, 3], 500.0)
        self.assertTrue(np.isnan(filtered[4, 4]))
        self.assertTrue(np.isnan(filtered[0, 0]))
        self.assertTrue(np.isnan(filtered[1, 1]))

    def test_extract_zone(self):
        """Verifiera att zon-urklippning skär ut rätt delar av matrisen via match-case."""
        matrix = np.arange(64, dtype=np.float32).reshape((8, 8))

        # Test 1: "center" (4x4, rader 2-5, kolumner 2-5)
        center = self.sensor._extract_zone(matrix, "center")
        self.assertEqual(center.shape, (4, 4))
        self.assertEqual(center[0, 0], matrix[2, 2])
        self.assertEqual(center[3, 3], matrix[5, 5])

        # Test 2: "spot" (2x2, rader 3-4, kolumner 3-4)
        spot = self.sensor._extract_zone(matrix, "spot")
        self.assertEqual(spot.shape, (2, 2))
        self.assertEqual(spot[0, 0], matrix[3, 3])
        self.assertEqual(spot[1, 1], matrix[4, 4])

        # Test 3: "all" (hela 8x8)
        all_zones = self.sensor._extract_zone(matrix, "all")
        self.assertEqual(all_zones.shape, (8, 8))
        np.testing.assert_array_equal(all_zones, matrix)

        # Test 4: Okänd zon (falla tillbaka på "center")
        unknown = self.sensor._extract_zone(matrix, "ogiltig_zon")
        self.assertEqual(unknown.shape, (4, 4))

    def test_aggregate_methods(self):
        """Verifiera att rumsaggregering räknar rätt och ignorerar NaN."""
        data = np.array([100.0, 200.0, 300.0, np.nan], dtype=np.float32)

        self.assertEqual(self.sensor._aggregate(data, "median"), 200.0)
        self.assertEqual(self.sensor._aggregate(data, "mean"), 200.0)
        self.assertEqual(self.sensor._aggregate(data, "min"), 100.0)

        # Om alla värden är NaN (skymd sensor) -> ska ge -1.0
        all_nan = np.full((4, 4), np.nan, dtype=np.float32)
        self.assertEqual(self.sensor._aggregate(all_nan, "median"), -1.0)

    def test_temporal_aggregate(self):
        """Verifiera att tidsfiltret komprimerar en 3D-stack (N, 8, 8) till (8, 8)."""
        # Skapa 3 tidssteg med värden 100, 200, 300 mm
        m1 = np.full((8, 8), 100.0, dtype=np.float32)
        m2 = np.full((8, 8), 200.0, dtype=np.float32)
        m3 = np.full((8, 8), 300.0, dtype=np.float32)
        # Sätt in en NaN i mätning 2 för en zon: tidsfiltret ska ignorera NaN
        m2[4, 4] = np.nan

        stack = np.stack([m1, m2, m3], axis=0)  # Form: (3, 8, 8)

        # Median
        med = self.sensor._temporal_aggregate(stack, "median")
        self.assertEqual(med.shape, (8, 8))
        self.assertEqual(med[0, 0], 200.0)
        # Pixel (4,4) hade [100.0, NaN, 300.0] -> medianen av 100 och 300 är 200.0
        self.assertEqual(med[4, 4], 200.0)

        # Mean
        avg = self.sensor._temporal_aggregate(stack, "mean")
        self.assertEqual(avg[0, 0], 200.0)

        # Min
        min_val = self.sensor._temporal_aggregate(stack, "min")
        self.assertEqual(min_val[0, 0], 100.0)

    def test_read_pipeline_with_order_event(self):
        """Verifiera hela pipeline från mätorder till färdig ToFReading via serveringsklockan."""
        self.sensor.is_connected = True
        self.sensor._running = True

        mock_dist = np.full((8, 8), 850.0, dtype=np.float32)
        mock_status = np.full((8, 8), 5, dtype=np.uint8)
        # Gör 2 zoner i mitten ogiltiga
        mock_status[3, 3] = 0
        mock_status[4, 4] = 0

        # Simulera att bakgrundstråden levererar mätningar
        def deliver_measurements():
            time.sleep(0.02)
            with self.sensor._lock:
                self.sensor._collected_distances = [mock_dist, mock_dist]
                self.sensor._collected_statuses = [mock_status, mock_status]
                self.sensor._target_samples = 0
                self.sensor._order_event.set()

        threading.Thread(target=deliver_measurements, daemon=True).start()

        reading = self.sensor.read(
            samples=2,
            sample_method="median",
            zone="center",
            zone_method="median",
            timeout=1.0
        )

        self.assertTrue(reading.is_valid())
        self.assertEqual(reading.distance_mm, 850.0)
        self.assertEqual(reading.samples, 2)
        self.assertEqual(reading.valid_zones, 14)  # 16 - 2 ogiltiga = 14
        self.assertEqual(reading.zone, "center")
        self.assertEqual(reading.sample_method, "median")
        self.assertEqual(reading.zone_method, "median")

    def test_read_matrix_timeout(self):
        """Verifiera att timeout ger None för read_matrix och ToFReading.invalid() för read."""
        self.sensor.is_connected = True
        self.sensor._running = True

        # Ingen bakgrundstråd slår på klockan -> ska timea ut efter 0.05 sekunder
        matrix = self.sensor.read_matrix(samples=3, timeout=0.05)
        self.assertIsNone(matrix)

        reading = self.sensor.read(samples=3, timeout=0.05)
        self.assertFalse(reading.is_valid())
        self.assertEqual(reading.distance_mm, -1.0)
        self.assertEqual(reading.samples, 0)

    def test_context_manager(self):
        """Verifiera att with-satsen fungerar och kopplar ifrån sensorn."""
        with ToFSensor(verbose=False, auto_connect=False) as s:
            self.assertIsInstance(s, ToFSensor)
            self.assertFalse(s.is_connected)

        self.assertFalse(s.is_connected)


if __name__ == "__main__":
    unittest.main()
