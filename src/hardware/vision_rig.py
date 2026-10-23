"""
Hårdvarusamordnare och fasad för den optiska mätbädden (VisionRig).

Knyter samman CameraController (Dual NoIR NoIR Stereo) och LEDController (4x IR LED)
till en enhetlig fasad som hanterar synkroniserad belysning, exponeringsprofiler och
fotograferingssekvenser för både 3D-stereo och fotometrisk stereo.
"""

import time
from typing import Any, Dict, Optional, Union

from src.core.models import (
    CameraFrame,
    CameraFramePair,
    PhotometricSweep,
    VisionCapture,
)
from src.hardware.camera_controller import CameraController
from src.hardware.led_controller import LEDController


class VisionRig:
    """
    Fasad som koordinerar belysning och kameror för paketmätningssystemet.
    """

    def __init__(
        self,
        camera_controller: Optional[CameraController] = None,
        led_controller: Optional[LEDController] = None,
        stereo_exposure_us: int = 28000,
        stereo_gain: float = 3.0,
        photo_exposure_us: int = 28000,
        photo_gain: float = 3.0,
        settle_time_s: float = 0.05,
        verbose: bool = True,
        auto_connect: bool = True,
    ):
        self.verbose: bool = verbose
        self.settle_time_s: float = settle_time_s

        # Standardprofiler för slutartid och förstärkning
        self.stereo_exposure_us: int = stereo_exposure_us
        self.stereo_gain: float = stereo_gain
        self.photo_exposure_us: int = photo_exposure_us
        self.photo_gain: float = photo_gain

        # Initiera eller injicera kontroller-instanser
        self.cameras: CameraController = camera_controller or CameraController(
            exposure_us=stereo_exposure_us,
            gain=stereo_gain,
            verbose=verbose,
            auto_connect=auto_connect,
        )
        self.leds: LEDController = led_controller or LEDController(
            verbose=verbose,
            auto_connect=auto_connect,
        )

    def connect(self) -> bool:
        """
        Initierar hårdvaruanslutningen för både kameror och LED-kontroller.
        Returnerar True om båda lyckades ansluta.
        """
        cam_ok = self.cameras.connect()
        led_ok = self.leds.connect()
        return bool(cam_ok and led_ok)

    def disconnect(self) -> None:
        """Släcker alla lampor och stänger ner alla kameror säkert."""
        try:
            self.leds.all_off()
        except Exception:
            pass
        self.leds.disconnect()
        self.cameras.disconnect()

    @property
    def is_connected(self) -> bool:
        """Returnerar True om både kameror och LED-kontroller är anslutna."""
        return bool(self.cameras.is_connected and self.leds.is_connected)

    def set_exposure(self, exposure_us: int, gain: float) -> bool:
        """Sätter slutartid och förstärkning direkt på båda kamerorna."""
        return self.cameras.set_exposure(exposure_us, gain)

    def _apply_exposure(self, target_exposure_us: int, target_gain: float) -> bool:
        """
        Intern hjälpmetod: Ändrar kamerornas exponering endast om
        de avviker från önskat målvärde.
        """
        if (
            self.cameras.exposure_us != target_exposure_us
            or abs(self.cameras.gain - target_gain) > 1e-3
        ):
            return self.cameras.set_exposure(target_exposure_us, target_gain)
        return True

    def capture_stereo(
        self,
        exposure_us: Optional[int] = None,
        gain: Optional[float] = None,
        led_mode: str = "all",
        samples: int = 1,
        method: str = "median",
    ) -> CameraFramePair:
        """
        Tar ett synkroniserat stereobildpar (left=CAM1, right=CAM0).

        Args:
            exposure_us: Slutartid i mikrosekunder (None = self.stereo_exposure_us).
            gain: Analog förstärkning (None = self.stereo_gain).
            led_mode: 'all' (tänder alla 4 lampor), eller 'ambient' (släckta lampor).
            samples: Antal sampel för brusfiltrering (skickas till CameraController).
            method: Filtreringsmetod ('median' eller 'mean').
        """
        exp = exposure_us if exposure_us is not None else self.stereo_exposure_us
        gn = gain if gain is not None else self.stereo_gain
        self._apply_exposure(exp, gn)

        if led_mode == "all":
            self.leds.all_on()
            if self.settle_time_s > 0:
                time.sleep(self.settle_time_s)
            pair = self.cameras.capture_both(samples=samples, method=method)
            self.leds.all_off()
            return pair
        elif led_mode == "ambient":
            self.leds.all_off()
            return self.cameras.capture_both(samples=samples, method=method)
        else:
            raise ValueError(f"Ogiltigt led_mode '{led_mode}'. Välj 'all' eller 'ambient'.")

    def capture_photometric_sweep(
        self,
        exposure_us: Optional[int] = None,
        gain: Optional[float] = None,
        samples: int = 1,
        method: str = "median",
    ) -> PhotometricSweep:
        """
        Kör ett fotometriskt ljussvep över de 4 LED-armarna (Kanal 1 -> 2 -> 3 -> 4).
        För varje lampa tas ett stereobildpar och sparas i en PhotometricSweep.

        Args:
            exposure_us: Slutartid i mikrosekunder (None = self.photo_exposure_us).
            gain: Analog förstärkning (None = self.photo_gain).
            samples: Antal sampel per bild.
            method: Filtreringsmetod ('median' eller 'mean').
        """
        exp = exposure_us if exposure_us is not None else self.photo_exposure_us
        gn = gain if gain is not None else self.photo_gain
        self._apply_exposure(exp, gn)

        pairs: Dict[int, CameraFramePair] = {}
        self.leds.all_off()

        try:
            for ch in [1, 2, 3, 4]:
                self.leds.on(ch)
                if self.settle_time_s > 0:
                    time.sleep(self.settle_time_s)

                pairs[ch] = self.cameras.capture_both(
                    samples=samples, method=method
                )
                self.leds.off(ch)

            return PhotometricSweep(
                ch1=pairs[1],
                ch2=pairs[2],
                ch3=pairs[3],
                ch4=pairs[4],
                timestamp=time.time(),
            )
        except Exception as e:
            if self.verbose:
                print(f"[VisionRig] Fel under fotometriskt svep: {e}")
            return PhotometricSweep.invalid()
        finally:
            self.leds.all_off()

    def capture_measurement_cycle(
        self,
        stereo_samples: int = 1,
        photo_samples: int = 1,
        method: str = "median",
        stereo_exposure_us: Optional[int] = None,
        stereo_gain: Optional[float] = None,
        photo_exposure_us: Optional[int] = None,
        photo_gain: Optional[float] = None,
    ) -> VisionCapture:
        """
        Kör en fullständig, optimerad datainsamlingssekvens för en paketmätning:
        1. Full Flash Stereo (12W belysning) med dedikerad stereo-exponering.
        2. Fotometriskt svep (Kanal 1-4) med dedikerad fotometri-exponering.

        Returnerar:
            VisionCapture med 'stereo', 'photometric' och 'timestamp'.
        """
        t_start = time.time()

        # 1. Full Flash Stereopar
        stereo_pair = self.capture_stereo(
            exposure_us=stereo_exposure_us,
            gain=stereo_gain,
            led_mode="all",
            samples=stereo_samples,
            method=method,
        )

        # 2. Fotometriskt svep
        photo_sweep = self.capture_photometric_sweep(
            exposure_us=photo_exposure_us,
            gain=photo_gain,
            samples=photo_samples,
            method=method,
        )

        return VisionCapture(
            stereo=stereo_pair,
            photometric=photo_sweep,
            timestamp=t_start,
        )

    def __enter__(self) -> "VisionRig":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.disconnect()
