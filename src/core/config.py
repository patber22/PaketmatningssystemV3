"""
Central hårdvarukonfiguration och GPIO-pinout för Paketmätningssystem V3.

Alla fysiska stift (Physical Pin 1-40) och BCM GPIO-nummer samlas här
så att mjukvaran och drivrutinerna har en gemensam sanning.
"""

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class PinMapping:
    """Representerar en hårdvaruanslutning på Raspberry Pi 40-pin GPIO list."""
    bcm: int            # GPIO-nummer som används i kod (t.ex. GPIO 5)
    physical_pin: int   # Fysiskt stiftnummer på kretskortet (t.ex. Stift 29)
    description: str    # Beskrivning av signalen och komponenten


# ==============================================================================
# 1. 3W 850nm IR LED STROBOSKOP (YYNMOS-4 MOSFET)
# Styrprincip: Active-LOW (LOW/0V = Tänd, HIGH/3.3V = Släckt)
# Alla PWM1..4 på MOSFET är gemensamt bryggade till 5V (Stift 2 eller 4).
# GND1..4 på MOSFET styrs individuellt av GPIO-pinnarna nedan.
# ==============================================================================

LED_ACTIVE_HIGH: bool = False  # YYNMOS-4 optokopplare styrs Active-LOW

LED_CHANNELS: Dict[int, PinMapping] = {
    # Kanal 1: Sydost / Arm 1 (Testad och verifierad!)
    1: PinMapping(
        bcm=5,
        physical_pin=29,
        description="LED 1 (GND1 på MOSFET, Active-LOW)"
    ),

    # Kanal 2: Sydväst / Arm 2 (Kopplad)
    2: PinMapping(
        bcm=6,
        physical_pin=31,
        description="LED 2 (GND2 på MOSFET, Active-LOW)"
    ),

    # Kanal 3: Nordväst / Arm 3 (Kopplad)
    3: PinMapping(
        bcm=25,
        physical_pin=22,
        description="LED 3 (GND3 på MOSFET, Active-LOW)"
    ),

    # Kanal 4: Nordost / Arm 4 (Kopplad)
    4: PinMapping(
        bcm=24,
        physical_pin=18,
        description="LED 4 (GND4 på MOSFET, Active-LOW)"
    ),
}

# Gemensam matning för MOSFET
MOSFET_PWM_SUPPLY_PIN = 2   # 5V (Fysiskt stift 2 eller 4) bryggad till PWM1..4
MOSFET_DC_POWER_PIN = 4     # 5V (Fysiskt stift 4) till DC+ skruvplint
MOSFET_DC_GROUND_PIN = 30   # GND (Fysiskt stift 30) till DC- skruvplint


# ==============================================================================
# 2. TIME-OF-FLIGHT SENSOR (ST VL53L5CX)
# Protokoll: Linux I2C-1 (/dev/i2c-1), Adress: 0x29
# ==============================================================================

TOF_I2C_BUS = 1             # /dev/i2c-1
TOF_I2C_ADDRESS = 0x29      # 7-bitars hårdvaruadress

TOF_PINS = {
    "SDA": PinMapping(bcm=2, physical_pin=3, description="I2C1 SDA"),
    "SCL": PinMapping(bcm=3, physical_pin=5, description="I2C1 SCL"),
    "VIN": PinMapping(bcm=-1, physical_pin=1, description="3.3V Power"),
    "GND": PinMapping(bcm=-1, physical_pin=25, description="GND"),
}


# ==============================================================================
# 3. NOIR DUAL CAMERAS (Sony IMX219)
# Portar på CM5-IO-BASE-B (22-pin till 15-pin FPC kablar)
# ==============================================================================

CAMERAS = {
    "CAM0": {"port": 0, "description": "NoIR Kamera Vänster (CSI0)"},
    "CAM1": {"port": 1, "description": "NoIR Kamera Höger (CSI1)"},
}


# ==============================================================================
# HJÄLPFUNKTIONER
# ==============================================================================

def get_led_bcm_pin(channel: int) -> int:
    """Hämtar BCM GPIO-nummer för en specifik LED-kanal (1-4)."""
    if channel not in LED_CHANNELS:
        raise ValueError(f"Ogiltig LED-kanal {channel}. Välj 1, 2, 3 eller 4.")
    return LED_CHANNELS[channel].bcm


def get_led_physical_pin(channel: int) -> int:
    """Hämtar fysiskt stiftnummer för en specifik LED-kanal (1-4)."""
    if channel not in LED_CHANNELS:
        raise ValueError(f"Ogiltig LED-kanal {channel}. Välj 1, 2, 3 eller 4.")
    return LED_CHANNELS[channel].physical_pin
