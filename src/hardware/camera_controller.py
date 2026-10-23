"""
Kontroller och fasad för dubbelkamerariggen (CAM0 Höger & CAM1 Vänster).

Hanterar synkroniserad exponering, upplösning och stereobildtagning
som returnerar semantiskt säkra CameraFramePair-objekt (left=CAM1, right=CAM0).
"""

import time
from typing import Dict, Tuple, Optional
from src.core.config import CAMERAS
from src.core.models import CameraFrame, CameraFramePair
from src.hardware.camera import Camera


class CameraController:
    """
    Fasad som koordinerar och styr båda NoIR-kamerorna som ett integrerat stereopar.
    """

    def __init__(
        self,
        resolution: Tuple[int, int] = (1640, 1232),
        exposure_us: int = 25000,
        gain: float = 4.0,
        verbose: bool = True,
        auto_connect: bool = True,
    ):
        self.resolution: Tuple[int, int] = resolution
        self.exposure_us: int = exposure_us
        self.gain: float = gain
        self.verbose: bool = verbose

        self._cameras: Dict[str, Camera] = {
            cid: Camera(
                camera_id=cid,
                resolution=resolution,
                exposure_us=exposure_us,
                gain=gain,
                verbose=verbose,
                auto_connect=auto_connect,
            )
            for cid in CAMERAS.keys()
        }

    def connect(self) -> bool:
        """
        Initierar anslutningen för båda kamerorna.
        Returnerar True om båda lyckades ansluta.
        """
        results = [cam.connect() for cam in self._cameras.values()]
        return all(results)

    def disconnect(self) -> None:
        """Stoppar och stänger båda kamerornas hårdvaruresurser."""
        for cam in self._cameras.values():
            cam.disconnect()

    @property
    def is_connected(self) -> bool:
        """Returnerar True om båda kamerorna är fysiskt anslutna och aktiva."""
        return all(cam.is_connected for cam in self._cameras.values())

    def set_exposure(self, exposure_us: int, gain: float, flush_frames: int = 3) -> bool:
        """
        Justerar slutartid och förstärkning på båda kamerorna synkront.
        1. Kontrollerar om värdena redan matchar nuvarande (gör ingenting och ingen flush om samma).
        2. Skickar nya kontroller omedelbart till båda kamerorna (ingen väntan mellan dem).
        3. Spolar bildrutor i synkroniserade par (CAM1 sedan CAM0) så att
           båda kamerornas hårdvarurör töms samtidigt i lockstep.
        """
        # Kontrollera om värdena redan matchar (undvik onödig registeroperation och flush)
        if self.exposure_us == exposure_us and abs(self.gain - gain) < 1e-3:
            all_match = all(
                cam.exposure_us == exposure_us and abs(cam.gain - gain) < 1e-3
                for cam in self._cameras.values()
            )
            if all_match:
                return True

        self.exposure_us = exposure_us
        self.gain = gain

        # 1. Skicka kontroller till båda kamerorna omedelbart
        results = [cam.set_exposure(exposure_us, gain) for cam in self._cameras.values()]

        # 2. Spola båda kamerornas hårdvarurör synkront i par
        if all(results) and flush_frames > 0:
            for _ in range(flush_frames):
                for cid in ["CAM1", "CAM0"]:
                    cam = self._cameras.get(cid)
                    if cam is not None and getattr(cam, "is_connected", False):
                        if hasattr(cam, "flush"):
                            cam.flush(1)

        return all(results)

    def set_single_exposure(self, camera_id: str, exposure_us: int, gain: float) -> bool:
        """Justerar exponeringen på en enskild kamera."""
        if camera_id not in self._cameras:
            raise ValueError(f"Ogiltigt camera_id '{camera_id}'. Välj mellan {list(self._cameras.keys())}.")
        return self._cameras[camera_id].set_exposure(exposure_us, gain)

    def capture_single(self, camera_id: str, samples: int = 1, method: str = "median") -> CameraFrame:
        """
        Hämtar en bildruta från en enskild kamera (t.ex. CAM0 vid fotometrisk stereo).
        """
        if camera_id not in self._cameras:
            raise ValueError(f"Ogiltigt camera_id '{camera_id}'. Välj mellan {list(self._cameras.keys())}.")
        return self._cameras[camera_id].capture(samples=samples, method=method)

    def capture_both(self, samples: int = 1, method: str = "median") -> CameraFramePair:
        """
        Hämtar ett synkroniserat bildpar från båda kamerorna.
        Garanterar att left är CAM1 (Vänster) och right är CAM0 (Höger).
        """
        frame_left = self._cameras["CAM1"].capture(samples=samples, method=method)
        frame_right = self._cameras["CAM0"].capture(samples=samples, method=method)
        ts = time.time()
        return CameraFramePair(left=frame_left, right=frame_right, timestamp=ts)

    def __enter__(self) -> "CameraController":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.disconnect()

    def __repr__(self) -> str:
        conn = "Ansluten" if self.is_connected else "Frånkopplad"
        return (f"CameraController({conn}, Res={self.resolution[0]}x{self.resolution[1]}, "
                f"Exp={self.exposure_us}µs, Gain={self.gain})")
