"""
Kärndataklasser och domänmodeller för Paketmätningssystem V3.
Håller standardiserade datastrukturer oberoende av hårdvarudrivrutiner.
"""

from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np

import cv2


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


@dataclass
class CameraFrame:
    """
    Representerar en tagen bildruta från en NoIR-kamera (CAM0 eller CAM1).

    Håller bildmatrisen (NumPy ndarray) i RAM samt optisk och tidsmässig metadata.
    """
    image: Optional[Any] = None              # Bildmatrisen (H, W, C) eller (H, W) som t.ex. np.ndarray
    camera_id: str = ""                      # "CAM0" (Höger) eller "CAM1" (Vänster)
    timestamp: float = 0.0                   # UNIX-tidsstämpel i sekunder (time.time())
    resolution: tuple = (0, 0)               # (Bredd, Höjd), t.ex. (1640, 1232)
    exposure_us: int = 0                     # Slutartid i mikrosekunder (t.ex. 25000 för 25 ms)
    gain: float = 0.0                        # Analog/digital förstärkning
    samples: int = 1                         # Antal samplade bildrutor för tidsfilter
    sample_method: str = "none"              # Tidsfilter: 'none', 'median' eller 'mean'
    _is_valid: bool = False                  # Intern flagga för hårdvaru-OK

    def is_valid(self) -> bool:
        """True om bilden finns i RAM, är en giltig matris och flaggats som giltig."""
        if not self._is_valid or self.image is None:
            return False
        if np is not None and isinstance(self.image, np.ndarray):
            return bool(self.image.size > 0)
        return True

    @classmethod
    def invalid(cls) -> "CameraFrame":
        """Null Object Pattern: Skapar en standardiserad tom/ogiltig bildram utan argument."""
        return cls(
            image=None,
            camera_id="",
            timestamp=time.time(),
            resolution=(0, 0),
            exposure_us=0,
            gain=0.0,
            samples=0,
            sample_method="none",
            _is_valid=False,
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialiserar metadata för loggar och JSON.
        Obs: Själva råbilden sparas som shape och dtype istället för megabyte av rådata.
        """
        has_img = self.image is not None
        img_shape = list(self.image.shape) if has_img and hasattr(self.image, "shape") else None
        img_dtype = str(self.image.dtype) if has_img and hasattr(self.image, "dtype") else None
        return {
            "camera_id": self.camera_id,
            "is_valid": self.is_valid(),
            "timestamp": self.timestamp,
            "resolution": list(self.resolution) if isinstance(self.resolution, (tuple, list)) else [0, 0],
            "exposure_us": self.exposure_us,
            "gain": self.gain,
            "samples": self.samples,
            "sample_method": self.sample_method,
            "has_image": bool(has_img),
            "image_shape": img_shape,
            "image_dtype": img_dtype,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], image: Optional[Any] = None) -> "CameraFrame":
        """Skapar en CameraFrame från dictionary och valfri separat bildmatris."""
        if not isinstance(data, dict):
            return cls.invalid()

        try:
            res_raw = data.get("resolution", [0, 0])
            resolution = tuple(res_raw) if isinstance(res_raw, (list, tuple)) else (0, 0)
            is_valid_flag = bool(data.get("is_valid", False))
            return cls(
                image=image,
                camera_id=str(data.get("camera_id", "")),
                timestamp=float(data.get("timestamp", 0.0)),
                resolution=resolution,
                exposure_us=int(data.get("exposure_us", 0)),
                gain=float(data.get("gain", 0.0)),
                samples=int(data.get("samples", 1)),
                sample_method=str(data.get("sample_method", "none")),
                _is_valid=is_valid_flag and (image is not None),
            )
        except (ValueError, TypeError):
            return cls.invalid()

    def to_jpeg(
        self,
        path: Union[str, Path] = Path("."),
        filename: Optional[str] = None
    ) -> Optional[Path]:
        """
        Sparar bildmatrisen som en JPEG-fil på angiven sökväg.

        - Om path är en katalog eller om filnamn saknas:
          Genererar filnamn: '{int(timestamp)}_{camera_id}.jpg'.
        - Om path är en fullständig sökväg som slutar på .jpg/.jpeg används den direkt.

        Hanterar färgkonvertering (RGB till BGR) automatiskt för OpenCV.
        Returnerar Path till den sparade filen, eller None vid ogiltig bild eller fel.
        """
        if not self.is_valid() or self.image is None:
            return None

        try:
            target_path = Path(path)

            # Avgör om target_path är en fil eller katalog
            if target_path.suffix.lower() in [".jpg", ".jpeg"]:
                out_file = target_path
            else:
                target_path.mkdir(parents=True, exist_ok=True)
                if filename is None:
                    cid = f"_{self.camera_id}" if self.camera_id else ""
                    ts = int(self.timestamp) if self.timestamp > 0 else int(time.time())
                    filename = f"{ts}{cid}.jpg"
                out_file = target_path / filename

            # Säkerställ att föräldrakatalogen finns
            out_file.parent.mkdir(parents=True, exist_ok=True)

            # Konvertera färger om det är en 3-kanals RGB-bild (Picamera2 RGB888 -> OpenCV BGR)
            if len(self.image.shape) == 3 and self.image.shape[2] == 3:
                bgr_img = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)
            else:
                bgr_img = self.image

            success = cv2.imwrite(str(out_file), bgr_img)
            return out_file if success else None

        except Exception:
            return None

    def __str__(self) -> str:
        """Kompakt strängrepresentation baserad på to_dict()."""
        return f"CameraFrame: {self.to_dict()}"


@dataclass
class CameraFramePair:
    """
    Representerar ett synkroniserat stereopar från NoIR-kamerorna.

    Garanterar semantisk separation mellan vänster (CAM1) och höger (CAM0) kamera
    så att datorseendealgoritmer aldrig blandar ihop stereobasens geometri.
    """
    left: CameraFrame = field(default_factory=CameraFrame.invalid)
    right: CameraFrame = field(default_factory=CameraFrame.invalid)
    timestamp: float = 0.0

    def is_valid(self) -> bool:
        """True om BÅDA bilderna i paret är giltiga."""
        return bool(self.left.is_valid() and self.right.is_valid())

    @classmethod
    def invalid(cls) -> "CameraFramePair":
        """Null Object Pattern: Skapar ett tomt/ogiltigt stereopar."""
        return cls(
            left=CameraFrame.invalid(),
            right=CameraFrame.invalid(),
            timestamp=time.time(),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Konverterar stereoparets metadata till dictionary för loggar/export."""
        return {
            "is_valid": self.is_valid(),
            "timestamp": self.timestamp,
            "left": self.left.to_dict(),
            "right": self.right.to_dict(),
        }

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        left_image: Optional[Any] = None,
        right_image: Optional[Any] = None
    ) -> "CameraFramePair":
        """Skapar en CameraFramePair från dictionary och valfria bildmatriser."""
        if not isinstance(data, dict):
            return cls.invalid()

        try:
            left_dict = data.get("left", {})
            right_dict = data.get("right", {})
            left_frame = CameraFrame.from_dict(left_dict, image=left_image)
            right_frame = CameraFrame.from_dict(right_dict, image=right_image)
            return cls(
                left=left_frame,
                right=right_frame,
                timestamp=float(data.get("timestamp", 0.0)),
            )
        except (ValueError, TypeError):
            return cls.invalid()

    def to_jpeg(self, path: Union[str, Path] = Path(".")) -> Tuple[Optional[Path], Optional[Path]]:
        """Sparar båda bilderna i paret (t.ex. {ts}_CAM1.jpg och {ts}_CAM0.jpg)."""
        left_path = self.left.to_jpeg(path) if self.left.is_valid() else None
        right_path = self.right.to_jpeg(path) if self.right.is_valid() else None
        return (left_path, right_path)

    def __str__(self) -> str:
        """Kompakt strängrepresentation baserad på to_dict()."""
        return f"CameraFramePair: {self.to_dict()}"


@dataclass
class WeightReading:
    """
    Representerar en avläsning från paketvågen (ESP32-C3 / HX711).

    Håller vikten i gram, stabilitetsflagga, tidsstämpel och råvärde för tara/kalibrering.
    """
    weight_g: float = -1.0              # Uppmätt vikt i gram (t.ex. 1250.0, -1.0 vid fel)
    is_stable: bool = False             # True om vikten har stabiliserats (låg varians över tid)
    timestamp: float = 0.0              # UNIX-tidsstämpel i sekunder (time.time())
    samples: int = 1                    # Antal råmätningar som använts för filtrering
    raw_value: int = 0                  # Obehandlat ADC-råvärde från HX711
    _is_valid: bool = False             # Intern hårdvaruflagg

    def is_valid(self) -> bool:
        """
        Enhetlig metod för att kontrollera giltighet.
        Returnerar True om mätvärdet är godkänt och större än eller lika med 0 gram.
        """
        return bool(self._is_valid and self.weight_g >= 0.0)

    @classmethod
    def invalid(cls) -> "WeightReading":
        """Null Object Pattern: Skapar en standardiserad ogiltig viktavläsning utan argument."""
        return cls(
            weight_g=-1.0,
            is_stable=False,
            timestamp=time.time(),
            samples=0,
            raw_value=0,
            _is_valid=False,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Konverterar viktavläsningen till en ren dictionary för loggar och export."""
        return {
            "weight_g": round(self.weight_g, 1) if self.weight_g >= 0.0 else -1.0,
            "is_stable": self.is_stable,
            "is_valid": self.is_valid(),
            "timestamp": self.timestamp,
            "samples": self.samples,
            "raw_value": self.raw_value,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WeightReading":
        """Skapar en WeightReading från en dictionary."""
        if not isinstance(data, dict):
            return cls.invalid()

        try:
            return cls(
                weight_g=float(data.get("weight_g", -1.0)),
                is_stable=bool(data.get("is_stable", False)),
                timestamp=float(data.get("timestamp", 0.0)),
                samples=int(data.get("samples", 1)),
                raw_value=int(data.get("raw_value", 0)),
                _is_valid=bool(data.get("is_valid", False)),
            )
        except (ValueError, TypeError):
            return cls.invalid()

    def __str__(self) -> str:
        """Kompakt strängrepresentation baserad på to_dict()."""
        return f"WeightReading: {self.to_dict()}"


@dataclass
class PhotometricSweep:
    """
    Representerar ett fotometriskt ljussvep över de fyra IR LED-kanalerna (Kanal 1-4).

    Håller ett CameraFramePair för varje belysningsvinkel (eller enskild CameraFrame i vänster/höger).
    """
    ch1: CameraFramePair = field(default_factory=CameraFramePair.invalid)
    ch2: CameraFramePair = field(default_factory=CameraFramePair.invalid)
    ch3: CameraFramePair = field(default_factory=CameraFramePair.invalid)
    ch4: CameraFramePair = field(default_factory=CameraFramePair.invalid)
    timestamp: float = 0.0

    def is_valid(self) -> bool:
        """True om alla fyra kanalernas bildpar är giltiga."""
        return bool(
            self.ch1.is_valid()
            and self.ch2.is_valid()
            and self.ch3.is_valid()
            and self.ch4.is_valid()
        )

    @classmethod
    def invalid(cls) -> "PhotometricSweep":
        """Null Object Pattern: Skapar ett tomt/ogiltigt fotometriskt svep."""
        return cls(
            ch1=CameraFramePair.invalid(),
            ch2=CameraFramePair.invalid(),
            ch3=CameraFramePair.invalid(),
            ch4=CameraFramePair.invalid(),
            timestamp=time.time(),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Konverterar svepets metadata till dictionary baserat på barnens to_dict()."""
        return {
            "is_valid": self.is_valid(),
            "timestamp": self.timestamp,
            "ch1": self.ch1.to_dict(),
            "ch2": self.ch2.to_dict(),
            "ch3": self.ch3.to_dict(),
            "ch4": self.ch4.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PhotometricSweep":
        """Skapar en PhotometricSweep från metadata-dictionary."""
        if not isinstance(data, dict):
            return cls.invalid()

        try:
            return cls(
                ch1=CameraFramePair.from_dict(data.get("ch1", {})),
                ch2=CameraFramePair.from_dict(data.get("ch2", {})),
                ch3=CameraFramePair.from_dict(data.get("ch3", {})),
                ch4=CameraFramePair.from_dict(data.get("ch4", {})),
                timestamp=float(data.get("timestamp", 0.0)),
            )
        except (ValueError, TypeError):
            return cls.invalid()

    def get_channel(self, channel: int) -> CameraFramePair:
        """Hämtar bildparet för en given kanal (1-4)."""
        mapping = {1: self.ch1, 2: self.ch2, 3: self.ch3, 4: self.ch4}
        if channel not in mapping:
            raise ValueError(f"Ogiltig kanal {channel}. Välj 1, 2, 3 eller 4.")
        return mapping[channel]

    def to_jpeg(self, path: Union[str, Path] = Path(".")) -> Dict[int, Tuple[Optional[Path], Optional[Path]]]:
        """Sparar alla fyra kanalernas bildpar till disk."""
        target_dir = Path(path)
        return {
            1: self.ch1.to_jpeg(target_dir),
            2: self.ch2.to_jpeg(target_dir),
            3: self.ch3.to_jpeg(target_dir),
            4: self.ch4.to_jpeg(target_dir),
        }

    def __str__(self) -> str:
        """Kompakt strängrepresentation baserad på to_dict()."""
        return f"PhotometricSweep: {self.to_dict()}"


@dataclass
class VisionCapture:
    """
    Representerar en komplett optisk datainsamling från mätbädden.

    Innehåller både ett Full Flash-stereopar och ett fotometriskt ljussvep (Kanal 1-4).
    """
    stereo: CameraFramePair = field(default_factory=CameraFramePair.invalid)
    photometric: PhotometricSweep = field(default_factory=PhotometricSweep.invalid)
    timestamp: float = 0.0

    def is_valid(self) -> bool:
        """True om både stereoparet och det fotometriska svepet är giltiga."""
        return bool(self.stereo.is_valid() and self.photometric.is_valid())

    @classmethod
    def invalid(cls) -> "VisionCapture":
        """Null Object Pattern: Skapar en tom/ogiltig komplett optisk fångst."""
        return cls(
            stereo=CameraFramePair.invalid(),
            photometric=PhotometricSweep.invalid(),
            timestamp=time.time(),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Konverterar till dictionary baserat på underliggande to_dict()."""
        return {
            "is_valid": self.is_valid(),
            "timestamp": self.timestamp,
            "stereo": self.stereo.to_dict(),
            "photometric": self.photometric.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VisionCapture":
        """Skapar en VisionCapture från metadata-dictionary."""
        if not isinstance(data, dict):
            return cls.invalid()

        try:
            return cls(
                stereo=CameraFramePair.from_dict(data.get("stereo", {})),
                photometric=PhotometricSweep.from_dict(data.get("photometric", {})),
                timestamp=float(data.get("timestamp", 0.0)),
            )
        except (ValueError, TypeError):
            return cls.invalid()

    def to_jpeg(self, path: Union[str, Path] = Path(".")) -> Dict[str, Any]:
        """Sparar både stereopar och fotometriskt svep till disk."""
        target_dir = Path(path)
        return {
            "stereo": self.stereo.to_jpeg(target_dir),
            "photometric": self.photometric.to_jpeg(target_dir),
        }

    def __str__(self) -> str:
        """Kompakt strängrepresentation baserad på to_dict()."""
        return f"VisionCapture: {self.to_dict()}"





