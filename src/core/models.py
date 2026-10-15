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
        """Snygg strängrepresentation för terminal och loggar."""
        status = "OK" if self.is_valid else "OGILTIG"
        return f"[ToF] {self.distance_mm:.1f} mm ({status}, {self.valid_zones} zoner, metod: {self.method}, zon: {self.zone})"

    def to_string(self) -> str:
        """Alias för __str__ för enkelhet och Java-vana."""
        return str(self)

