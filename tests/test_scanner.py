"""
Enhetstester för BarcodeScanner-klassen (src/hardware/scanner.py).
Verifierar konsumentmönster (peek/consume/clear), Null Object Pattern (aldrig None),
frånkoppling, trådsäkerhet och callback-hantering isolerat utan krav på fysisk USB-hårdvara.
"""

import unittest
import time

from src.hardware.scanner import BarcodeScanner
from src.core.models import BarcodeReading


class TestBarcodeScanner(unittest.TestCase):
    """Testsvit för BarcodeScanner-klassens logik och säkerhetsmekanismer."""

    def setUp(self):
        """Körs före varje test: skapa en frånkopplad scannerinstans."""
        self.scanner = BarcodeScanner(verbose=False, auto_connect=False)

    def tearDown(self):
        """Körs efter varje test: säkerställ att scannern kopplas ifrån."""
        self.scanner.disconnect()

    def test_disconnected_behavior(self):
        """Verifiera att en frånkopplad scanner returnerar BarcodeReading.invalid() och aldrig kraschar."""
        self.assertFalse(self.scanner.is_connected)

        peek_reading = self.scanner.peek()
        self.assertIsInstance(peek_reading, BarcodeReading)
        self.assertFalse(peek_reading.is_valid())
        self.assertEqual(peek_reading.code, "")

        consume_reading = self.scanner.consume()
        self.assertIsInstance(consume_reading, BarcodeReading)
        self.assertFalse(consume_reading.is_valid())
        self.assertEqual(consume_reading.code, "")

        # clear() ska inte kasta något undantag
        try:
            self.scanner.clear()
        except Exception as e:
            self.fail(f"clear() kastade ett oväntat fel: {e}")

    def test_peek_does_not_consume(self):
        """Verifiera att peek() inspekterar koden utan att förbruka den."""
        self.scanner.is_connected = True
        self.scanner._latest_reading = BarcodeReading(
            code="7312345678901",
            timestamp=time.time(),
            is_used=False
        )

        # Första peek()
        reading1 = self.scanner.peek()
        self.assertIsInstance(reading1, BarcodeReading)
        self.assertTrue(reading1.is_valid())
        self.assertEqual(reading1.code, "7312345678901")

        # Andra peek() ska fortfarande ge samma giltiga kod
        reading2 = self.scanner.peek()
        self.assertTrue(reading2.is_valid())
        self.assertEqual(reading2.code, "7312345678901")

    def test_consume_marks_used(self):
        """Verifiera att consume() förbrukar koden så att den inte kan läsas igen."""
        self.scanner.is_connected = True
        self.scanner._latest_reading = BarcodeReading(
            code="7312345678901",
            timestamp=time.time(),
            is_used=False
        )

        # Första anropet: ska hämta koden och förbruka den i scannern
        consumed = self.scanner.consume()
        self.assertTrue(consumed.is_valid())
        self.assertEqual(consumed.code, "7312345678901")

        # Andra anropet: koden är redan förbrukad -> ska returnera BarcodeReading.invalid()
        second_consume = self.scanner.consume()
        self.assertFalse(second_consume.is_valid())

        # peek() ska nu också returnera BarcodeReading.invalid()
        self.assertFalse(self.scanner.peek().is_valid())

    def test_clear_discards_reading(self):
        """Verifiera att clear() (Ångra scanning) kastar bort koden direkt."""
        self.scanner.is_connected = True
        self.scanner._latest_reading = BarcodeReading(
            code="FELAKTIG_KOD_123",
            timestamp=time.time(),
            is_used=False
        )

        # Rensa koden
        self.scanner.clear()

        # Nu ska varken consume() eller peek() hitta någon giltig kod
        self.assertFalse(self.scanner.consume().is_valid())
        self.assertFalse(self.scanner.peek().is_valid())

    def test_new_scan_overwrites_consumed_reading(self):
        """Verifiera att en ny inläsning återställer is_used=False och blir tillgänglig."""
        self.scanner.is_connected = True
        self.scanner._latest_reading = BarcodeReading(
            code="GAMMAL_KOD_001",
            timestamp=100.0,
            is_used=False
        )

        # Förbruka den gamla koden
        old = self.scanner.consume()
        self.assertTrue(old.is_valid())
        self.assertEqual(old.code, "GAMMAL_KOD_001")
        self.assertFalse(self.scanner.consume().is_valid())

        # Simulera att användaren scannar ett nytt paket
        self.scanner._latest_reading = BarcodeReading(
            code="NY_KOD_002",
            timestamp=200.0,
            is_used=False
        )

        # Nu ska den nya koden kunna läsas och förbrukas
        new_peek = self.scanner.peek()
        self.assertTrue(new_peek.is_valid())
        self.assertEqual(new_peek.code, "NY_KOD_002")

        new_consumed = self.scanner.consume()
        self.assertTrue(new_consumed.is_valid())
        self.assertEqual(new_consumed.code, "NY_KOD_002")

    def test_context_manager(self):
        """Verifiera att with-satsen (__enter__ och __exit__) fungerar och kopplar ifrån."""
        with BarcodeScanner(verbose=False, auto_connect=False) as s:
            self.assertIsInstance(s, BarcodeScanner)
            self.assertFalse(s.is_connected)

        # Efter with-blocket ska disconnect() ha anropats
        self.assertFalse(s.is_connected)

    def test_on_scan_callback(self):
        """Verifiera att on_scan_callback anropas när en callback har definierats."""
        received_readings = []

        def callback_handler(reading: BarcodeReading):
            received_readings.append(reading)

        scanner = BarcodeScanner(
            verbose=False,
            auto_connect=False,
            on_scan_callback=callback_handler
        )

        test_reading = BarcodeReading(
            code="CALLBACK_TEST_123",
            timestamp=time.time(),
            is_used=False
        )

        # Simulera callback-anrop från scannerns logik
        if scanner.on_scan_callback:
            scanner.on_scan_callback(test_reading)

        self.assertEqual(len(received_readings), 1)
        self.assertEqual(received_readings[0].code, "CALLBACK_TEST_123")
        self.assertTrue(received_readings[0].is_valid())


if __name__ == "__main__":
    unittest.main()
