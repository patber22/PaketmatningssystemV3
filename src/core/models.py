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
    distance_mm: float = 0.0        # Aggregerat avstånd i mm (t.ex. 950.4)
    is_valid: bool = False          # True om sensorn läste OK och hittade ett plan
    timestamp: float = 0.0          # UNIX-tidsstämpel i sekunder (time.time())
    samples: int = 1                # Antal mätningar som användes för brusreducering
    method: str = "median"          # Aggregeringsmetod: 'median', 'mean' eller 'min'
    zone: str = "center"            # Zon-urval: 'center', 'spot' eller 'all'
    valid_zones: int = 0            # Antal zoner med godkänd hårdvarustatus

    def to_dict(self) -> Dict[str, Any]:
        """Konverterar mätningen till en ren dictionary för config.json."""
        return {
            "distance_mm": round(self.distance_mm, 2),
            "is_valid": self.is_valid,
            "timestamp": self.timestamp,
            "samples": self.samples,
            "method": self.method,
            "zone": self.zone,
            "valid_zones": self.valid_zones,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToFReading":
        """Skapar en ToFReading från en dictionary (t.ex. inläst från config.json)."""
        if not isinstance(data, dict):
            return cls()

        try:
            return cls(
                distance_mm=float(data.get("distance_mm", 0.0)),
                is_valid=bool(data.get("is_valid", False)),
                timestamp=float(data.get("timestamp", 0.0)),
                samples=int(data.get("samples", 1)),
                method=str(data.get("method", "median")),
                zone=str(data.get("zone", "center")),
                valid_zones=int(data.get("valid_zones", 0)),
            )
        except (ValueError, TypeError):
            return cls()

    def is_expired(self, max_age_hours: float = 168.0) -> bool:
        """
        Kontrollerar om mätvärdet är äldre än specificerat antal timmar.
        Default: 168.0 timmar (7 dagar).
        """
        if not self.is_valid or self.timestamp <= 0.0:
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

    @property
    def is_valid(self) -> bool:
        """True om koden inte är tom och inte redan förbrukad."""
        return bool(self.code and not self.is_used)

    def to_dict(self) -> Dict[str, Any]:
        """Konverterar till en ren dictionary för loggar och export."""
        return {
            "code": self.code,
            "timestamp": self.timestamp,
            "is_used": self.is_used,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BarcodeReading":
        """Skapar en BarcodeReading från en dictionary."""
        if not isinstance(data, dict):
            return cls()

        try:
            return cls(
                code=str(data.get("code", "")),
                timestamp=float(data.get("timestamp", 0.0)),
                is_used=bool(data.get("is_used", False)),
            )
        except (ValueError, TypeError):
            return cls()

    def __str__(self) -> str:
        """Kompakt strängrepresentation baserad på to_dict()."""
        return f"BarcodeReading: {self.to_dict()}"


