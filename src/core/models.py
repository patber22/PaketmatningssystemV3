"""
Kärndataklasser och domänmodeller för Paketmätningssystem V3.
Håller standardiserade datastrukturer oberoende av hårdvarudrivrutiner.
"""

from dataclasses import dataclass
import time
from typing import Any, Dict


@dataclass
class ToFReading:
    """
    Representerar ett aggregerat mätvärde från Time-of-Flight (VL53L5CX).
    
    Används för bordskalibrering, referenshöjd och paketdetektering.
    Kan serialiseras direkt in och ut ur config-filer via dictionaries.
    """
    distance_mm: float = -1.0           # Aggregerat avstånd i mm (t.ex. 950.4, -1.0 vid fel)
    timestamp: float = 0.0              # UNIX-tidsstämpel i sekunder (time.time())
    samples: int = 0                    # Antal mätningar som användes för tidsbrusreducering
    sample_method: str = ""             # Tidsfilter: 'median', 'mean' eller 'min'
    zone: str = ""                      # Zon-urval: 'center', 'spot' eller 'all'
    zone_method: str = ""               # Rumsfilter: 'median', 'mean' eller 'min'
    valid_zones: int = 0                # Antal zoner med godkänd hårdvarustatus
    _is_valid: bool = False             # Internt flaggminne för hårdvaru-OK

    def is_valid(self) -> bool:
        """
        Enhetlig metod för att kontrollera giltighet.
        Returnerar True om mätvärdet är godkänt och större än 0 mm.
        """
        return bool(self._is_valid and self.distance_mm > 0.0 and self.valid_zones > 0)

    @classmethod
    def invalid(cls) -> "ToFReading":
        """Skapar en standardiserad ogiltig ToF-mätning utan argument."""
        return cls(
            distance_mm=-1.0,
            timestamp=time.time(),
            samples=0,
            sample_method="",
            zone="",
            zone_method="",
            valid_zones=0,
            _is_valid=False,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Konverterar mätningen till en ren dictionary för config.json."""
        return {
            "distance_mm": round(self.distance_mm, 2) if self.distance_mm > 0 else -1.0,
            "is_valid": self.is_valid(),
            "timestamp": self.timestamp,
            "samples": self.samples,
            "sample_method": self.sample_method,
            "zone": self.zone,
            "zone_method": self.zone_method,
            "valid_zones": self.valid_zones,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToFReading":
        """Skapar en ToFReading från en dictionary (t.ex. inläst från config.json)."""
        if not isinstance(data, dict):
            return cls.invalid()

        try:
            return cls(
                distance_mm=float(data.get("distance_mm", -1.0)),
                timestamp=float(data.get("timestamp", 0.0)),
                samples=int(data.get("samples", 0)),
                sample_method=str(data.get("sample_method", "")),
                zone=str(data.get("zone", "")),
                zone_method=str(data.get("zone_method", "")),
                valid_zones=int(data.get("valid_zones", 0)),
                _is_valid=bool(data.get("is_valid", False)),
            )
        except (ValueError, TypeError):
            return cls.invalid()

    def is_expired(self, max_age_hours: float = 168.0) -> bool:
        """
        Kontrollerar om mätvärdet är äldre än specificerat antal timmar.
        Default: 168.0 timmar (7 dagar).
        """
        if not self.is_valid() or self.timestamp <= 0.0:
            return True
        max_age_seconds = max_age_hours * 3600.0
        return (time.time() - self.timestamp) > max_age_seconds

    def __str__(self) -> str:
        """Kompakt strängrepresentation baserad på to_dict()."""
        return f"ToFReading: {self.to_dict()}"


@dataclass
class BarcodeReading:
    """
    Representerar en scannad streckkod från handscannern (Honeywell MS5145).

    Håller koden, tidsstämpel och en 'is_used'-flagga för att garantera
    att en kod aldrig råkar kopplas till fel paket av misstag.
    """
    code: str = ""                  # Scannad alfanumerisk kod (t.ex. "7312345678901")
    timestamp: float = 0.0          # UNIX-tidsstämpel i sekunder (time.time())
    is_used: bool = False           # True om koden redan förbrukats i en mätning

    def is_valid(self) -> bool:
        """Enhetlig metod: True om koden inte är tom och inte redan förbrukad."""
        return bool(self.code and not self.is_used)

    @classmethod
    def invalid(cls) -> "BarcodeReading":
        """Skapar en standardiserad ogiltig/tom streckkodsläsning (Null Object Pattern)."""
        return cls(
            code="",
            timestamp=0.0,
            is_used=True,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Konverterar till en ren dictionary för loggar och export."""
        return {
            "code": self.code,
            "timestamp": self.timestamp,
            "is_used": self.is_used,
            "is_valid": self.is_valid(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BarcodeReading":
        """Skapar en BarcodeReading från en dictionary."""
        if not isinstance(data, dict):
            return cls.invalid()

        try:
            return cls(
                code=str(data.get("code", "")),
                timestamp=float(data.get("timestamp", 0.0)),
                is_used=bool(data.get("is_used", False)),
            )
        except (ValueError, TypeError):
            return cls.invalid()

    def __str__(self) -> str:
        """Kompakt strängrepresentation baserad på to_dict()."""
        return f"BarcodeReading: {self.to_dict()}"


