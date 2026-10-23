"""
Hårdvarudrivrutin för en enskild NoIR-kamera (Sony IMX219) på Raspberry Pi CM5.

Styr anslutning, fast exponering/gain och bildtagning till RAM (NumPy ndarray)
via det officiella Picamera2-biblioteket.
"""

import time
from typing import Optional, Tuple, Any
from src.core.config import CAMERAS
from src.core.models import CameraFrame

try:
    from picamera2 import Picamera2
    HAS_PICAM2 = True
except ImportError:
    HAS_PICAM2 = False

import numpy as np


class Camera:
    """
    Representerar en enskild fysisk NoIR-kamera (CAM0 eller CAM1).
    """

    def __init__(
        self,
        camera_id: str,
        resolution: Tuple[int, int] = (1640, 1232),
        exposure_us: int = 25000,
        gain: float = 4.0,
        verbose: bool = True,
        auto_connect: bool = True,
    ):
        """
        Kräver explicit angivande av camera_id ('CAM0' eller 'CAM1').
        """
        if camera_id not in CAMERAS:
            raise ValueError(f"Ogiltigt camera_id '{camera_id}'. Välj mellan {list(CAMERAS.keys())}.")

        self.camera_id: str = camera_id
        self.port: int = CAMERAS[camera_id]["port"]
        self.description: str = CAMERAS[camera_id]["description"]

        self.resolution: Tuple[int, int] = resolution
        self.exposure_us: int = exposure_us
        self.gain: float = gain
        self.verbose: bool = verbose

        self.is_connected: bool = False
        self._picam2: Optional[Any] = None

        if auto_connect:
            self.connect()

    def connect(self) -> bool:
        """
        Initierar kameraströmmen via Picamera2 med fast exponering och gain.
        Returnerar True vid lyckad start, annars False. Kraschar aldrig.
        """
        if self.is_connected:
            return True

        if not HAS_PICAM2:
            if self.verbose:
                print(f"[{self.camera_id}] INFO: 'picamera2' saknas (körs på Windows/utvecklingsmiljö).")
                print(f"[{self.camera_id}] Kameran är inte ansluten till fysisk hårdvara.")
            self.is_connected = False
            return False

        try:
            # Initiera kameran på den specifika CSI-porten
            self._picam2 = Picamera2(camera_num=self.port)

            # Konfigurera kontinuerlig videoström i RAM (RGB888)
            # queue=False och buffer_count=2 garanterar att Picamera2 inte buffrar
            # gamla bildrutor i en FIFO-kö utan alltid levererar den senast exponerade
            # bildrutan direkt från sensorn (kritiskt vid snabb LED-växling).
            config = self._picam2.create_video_configuration(
                main={"size": self.resolution, "format": "RGB888"},
                queue=False,
                buffer_count=2,
            )
            self._picam2.configure(config)

            # Lås manuell exponering och analog gain (stäng av AEC/AGC)
            self._picam2.set_controls({
                "ExposureTime": self.exposure_us,
                "AnalogueGain": self.gain,
                "AeEnable": False,
            })

            # Starta kamerasensorns ström i bakgrunden
            self._picam2.start()
            self.is_connected = True

            if self.verbose:
                print(f"[{self.camera_id}] Ansluten på CSI-port {self.port} ({self.description}) "
                      f"i {self.resolution[0]}x{self.resolution[1]} med exp={self.exposure_us}µs, gain={self.gain}.")
            return True

        except Exception as e:
            if self.verbose:
                print(f"[{self.camera_id}] FEL: Kunde inte initiera kameran på port {self.port}: {e}")
            self.is_connected = False
            self._picam2 = None
            return False

    def disconnect(self) -> None:
        """Stoppar och stänger kamerans hårdvaruresurser."""
        if self._picam2 is not None:
            try:
                self._picam2.stop()
                self._picam2.close()
            except Exception:
                pass
            self._picam2 = None

        self.is_connected = False

    def flush(self, count: int = 1) -> None:
        """Tömmer N bildrutor ur kamerasensorns hårdvarurör."""
        if self.is_connected and self._picam2 is not None:
            try:
                for _ in range(max(0, count)):
                    self._picam2.capture_array("main")
            except Exception:
                pass

    def set_exposure(self, exposure_us: int, gain: float, flush_frames: int = 0) -> bool:
        """
        Uppdaterar slutartid och förstärkning på en aktiv kameraström.
        Om värdena redan stämmer görs ingenting.
        Vid förändring skickas nya kontroller omedelbart.
        Om flush_frames > 0 spolas hårdvaruröret.
        Returnerar True vid lyckad uppdatering, annars False.
        """
        if not self.is_connected or self._picam2 is None:
            self.exposure_us = exposure_us
            self.gain = gain
            return False

        # Undvik onödiga registeroperationer och spolningar om värdena redan matchar
        if self.exposure_us == exposure_us and abs(self.gain - gain) < 1e-3:
            return True

        self.exposure_us = exposure_us
        self.gain = gain

        try:
            self._picam2.set_controls({
                "ExposureTime": self.exposure_us,
                "AnalogueGain": self.gain,
                "AeEnable": False,
            })
            if flush_frames > 0:
                self.flush(flush_frames)
            return True
        except Exception as e:
            if self.verbose:
                print(f"[{self.camera_id}] FEL vid ändring av exponering: {e}")
            return False

    def capture(self, samples: int = 1, method: str = "median") -> CameraFrame:
        """
        Hämtar den färskaste bildrutan direkt ur RAM via Picamera2.
        Vid samples > 1 samlas N bildrutor in och filtreras över tid (standard: median).
        """
        if not self.is_connected or self._picam2 is None:
            if self.verbose:
                print(f"[{self.camera_id}] FEL: Kan inte ta bild, kameran är inte ansluten.")
            return CameraFrame.invalid()

        try:
            samples = max(1, samples)
            frames = [self._picam2.capture_array("main") for _ in range(samples)]

            if samples == 1:
                final_img = frames[0]
            else:
                stack = np.stack(frames, axis=0)
                m = method.lower()
                if m in ["mean", "average"]:
                    final_img = np.mean(stack, axis=0).astype(np.uint8)
                else:
                    final_img = np.median(stack, axis=0).astype(np.uint8)

            return CameraFrame(
                image=final_img,
                camera_id=self.camera_id,
                timestamp=time.time(),
                resolution=self.resolution,
                exposure_us=self.exposure_us,
                gain=self.gain,
                samples=samples,
                sample_method=method if samples > 1 else "none",
                _is_valid=True,
            )

        except Exception as e:
            if self.verbose:
                print(f"[{self.camera_id}] FEL under bildtagning: {e}")
            return CameraFrame.invalid()

    def __enter__(self) -> "Camera":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.disconnect()

    def __repr__(self) -> str:
        conn = "Ansluten" if self.is_connected else "Frånkopplad"
        return (f"Camera({self.camera_id}, Port {self.port}, {conn}, "
                f"{self.resolution[0]}x{self.resolution[1]}, Exp={self.exposure_us}µs, Gain={self.gain})")
