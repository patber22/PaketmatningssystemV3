# Projekt-Färdplan: Paketmätningssystem V3

Detta är ett levande dokument som beskriver utvecklingsordningen, faserna och milstolparna för V3. Vi bockar av moment efter hand och finjusterar stegen under resans gång.

---

## Övergripande Fasstruktur

```
[Fas 1: Mjukvara & Förberedelser]  -->  (Vid skrivbordet medan vi väntar på leveranser)
               │
               ▼
[Fas 2: CAD & 3D-print (Trärigg)]  -->  (Övergripande design anpassad för labb-träriggen)
               │
               ▼
[Fas 3: Mottagning & Säkerhetstest]-->  (ESP32-test av LED/MOSFET utan voltmeter, CM5-montering)
               │
               ▼
[Fas 4: Montering & Kabling]       -->  (Lödning, krympslang, motstånd & anslutning till GPIO)
               │
               ▼
[Fas 5: Systemintegration i Trärigg]-->  (Blixtsekvens, kalibrering & mätcykel < 0.6s)
               │
               ▼
[Fas 6: Slutrigg & Pilotkund]      -->  (1x1x1m aluminiumrigg, installation hos Clean Part AB)
```

---

## Fas 1: Mjukvara & Algoritmer (Vid skrivbordet NU)

Denna fas bygger kärnan i mjukvaran i lugn och ro innan hårdvaran anländer. Varje fil skrivs och testas isolerat.

- [x] **1.1. ToF-sensor drivrutin (`src/hardware/tof.py`)**
  - Ren I2C-drivrutin för ST VL53L5CX (8x8 multizon) över Linux I2C (Buss 1).
  - Tvåstegs brusfiltrering: tidsfilter (`read_matrix` över N samples) och rumsfilter (`read` över zoner som center, inner4x4, etc).
  - Null Object Pattern (`ToFReading.invalid()`, `is_valid()` metod).
  - Full enhetstestsvit (`tests/test_tof.py`) med trådsynkronisering och feltolerans.

- [x] **1.2. Streckkodsläsare drivrutin (`src/hardware/scanner.py`)**
  - Drivrutin för Honeywell Metrologic Eclipse MS5145 via Linux `evdev` (USB HID).
  - Läser koder i bakgrunden utan att terminalen behöver ha aktivt fokus.
  - Konsumentmönster (`peek`/`consume`/`clear`) och Null Object Pattern (`BarcodeReading.invalid()`, `is_valid()` metod).
  - Event-driven callback (`on_scan_callback`) som triggar mätsekvensen.
  - Full enhetstestsvit (`tests/test_scanner.py`) för trådsäkerhet och konsumentlogik.

- [ ] **1.3. Stereomatchning & Ytterkanter (`src/algorithms/stereo_matcher.py`)**
  - OpenCV SGBM-implementation optimerad för Bild 5 (Full Flash stereopar).
  - Beräknar djupkarta och detekterar paketets metriska ytterkanter och hörn i millimeter.
  - Enhetstest med simulerade stereobildpar.

- [ ] **1.4. Fotometrisk Stereo ytnormaler (`src/algorithms/photometric_stereo.py`)**
  - NumPy least-squares-matrislösare som beräknar ytans normalvektorer $\mathbf{N} = [n_x, n_y, n_z]$ ur de 4 riktade stroboskopbilderna (Bild 1–4).
  - Isolerar ytans lutning, buktningar och ojämnheter på kartonglocket, helt oberoende av textur.
  - Enhetstest med syntetiskt belyst sfär och buktande låda.

- [ ] **1.5. Djupfusion (`src/algorithms/depth_fusion.py`)**
  - Kärnfusionsmotorn som kombinerar all tre datakällor:
    1. **ToF:** Absolut referenshöjd.
    2. **Stereo:** Bounding box ytterkanter och avstånd i millimeter.
    3. **Fotometrisk stereo:** Ytnormaler och lockform.
  - SciPy 2D FFT Poisson-rekonstruktion som smälter samman ytnormalerna inuti stereokantens ram.
  - Returnerar slutgiltiga mått ($L \times B \times H$) med millimeterprecision.

---

## Fas 2: CAD & 3D-print (Träriggens fästen)

*Detaljmått tas fram när fysiska komponenter kan mätas på plats med skjutmått.*

- [ ] **2.1. LED-fästen & armar:**
  - 4 hållare för 20mm stjärnkretsarna + 60°-linserna, vinklade ~35° in mot bordet, anpassade för att fästas på den befintliga träriggen.
- [ ] **2.2. Kamerahus & Motljusskydd:**
  - Hållare för 2x Pi Camera v2 NoIR (med träriggens nuvarande kompakta baslinje) samt centralt fäste för ToF-sensorn.
  - Skärmväggar för att förhindra att stroboskopets IR-ljus lyser rakt in i linserna från sidan.
- [ ] **2.3. Elektronikmontering:**
  - Enkel bottenplatta/hållare för Waveshare CM5-kortet och YYNMOS-4 MOSFET-kortet med kylningsavstånd för motstånden.

---

## Fas 3: Mottagning & Säkerhetstest (När paketen anländer)

> [!CAUTION]
> **Säkerhetsverifiering utan voltmeter:**
> 1. **Optokopplarens LED:** YYNMOS-4 har röda indikator-lysdioder som tänds vid 3.3V styrsignal.
> 2. **Mobilkameratest:** 850nm IR-ljus syns tydligt som lila/rosa sken på en mobilkamera.
> 3. **Värmetest:** Korta pulser (100 ms) för att säkerställa att motstånd och LED inte blir heta.

- [ ] **3.1. ESP32-verifiering av MOSFET & 1x IR LED**
  - Koppla 3.3V GPIO från ESP32 till MOSFET-ingång kanal 1.
  - Koppla 5V matning, 6.8 Ω 5W motstånd och en 3W IR LED.
  - Kör blink-skript på ESP32 och verifiera lila ljus i mobilkameran utan värme.
  - Upprepa för alla 4 kanaler.

- [ ] **3.2. CM5-montering på Waveshare-kortet**
  - Montera CM5 och aluminiumkylfläns på Waveshare CM5-IO-BASE-B.
  - Boota via officiella 27W-adaptern och verifiera I2C (`i2cdetect -y 1`).

- [ ] **3.3. Test av NoIR-kamerorna**
  - Koppla in de två v2 NoIR-kamerorna med de nya 22-till-15-pin FPC-kablarna.
  - Verifiera bildtagning på båda CSI-portarna med `rpicam-still`.

---

## Fas 4: Montering & Kabling i Träriggen

- [ ] **4.1. Lödning av LED-satelliter:**
  - Löd 2x0.5 mm² kablar till stjärnkretsarnas pads.
  - Isolera lödpunkterna med 3.2 mm krympslang.
- [ ] **4.2. Kabling till MOSFET och CM5:**
  - Montera de fyra 6.8 Ω-motstånden i serie på plusledarna vid MOSFET-plinten.
  - Anslut MOSFET-styrsignalerna till CM5 GPIO (17, 27, 22, 23).
  - Anslut 5V från CM5 stift 2/4 och GND till stift 6.

---

## Fas 5: Systemintegration & Testbänk (Träriggen)

- [ ] **5.1. Blixtsekvens-controller (`src/hardware/strobe.py`):**
  - Styrkoden som kör 4 riktade blixtar (16.6 ms st) + 1 Full Flash (16.6 ms).
- [ ] **5.2. Stereokalibrering i träriggen:**
  - Kalibrering med ChArUco-bräde under Bild 5 (Full Flash) och generering av `stereo_calibration.json`.
- [ ] **5.3. End-to-End mätcykel i träriggen:**
  - Paket ställs på plattan $\rightarrow$ Scanna streckkod $\rightarrow$ Väg $\rightarrow$ Blixtsekvens $\rightarrow$ Djupfusion $\rightarrow$ Slutresultat presenteras på under 0.6 sekunder.

---

## Fas 6: Uppskalning till Slutrigg & Pilot (Aluminium & Clean Part AB)

- [ ] **6.1. Konstruktion av 1x1x1m aluminiumprofilrigg:**
  - Bygga metallchassi med integrerad matt mörk bottenplatta och bredare kamera-baslinje för stora paket.
- [ ] **6.2. Överföring & omkalibrering:**
  - Flytta över den bevisade elektroniken från träriggen till aluminiumriggen och kalibrera om.
- [ ] **6.3. Pilotinstallation hos Clean Part AB:**
  - Montera över packbord hos Clean Part AB för 3–4 veckors skarp testdrift och referenstagning.
