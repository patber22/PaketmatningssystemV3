# Paketmätningssystem V3

Automatiserad dimensionerings- och vägningsstation för paket och e-handelsgods, optimerad för fristående drift på **Raspberry Pi Compute Module 5 (CM5)**.

Systemet beräknar paketets dimensioner ($L \times B \times H$), volymvikt och verklig vikt, kopplat till paketets streckkod, på **under 0.6 sekunder** genom att kombinera aktiv fotometrisk stereo, stereoseende, Time-of-Flight och digital våg.

---

## 1. Bakgrund & Evolution: Från V1 till V3

```
[V1: Trärigg & Webbkameror] 
         │  (Bevisade konceptet, men USB-kameror och Windows gav osynk och lagg)
         ▼
[V2: R&D & Prototyputforskning] 
         │  (Flytt till CM4/CM5, CSI-kameror, SGBM, ToF, våg & scanner. 
         │   Kinect-laser testades men lades ner p.g.a. ögonsäkerhet.
         │   Passiv stereo visade sig sårbar för varierande rumsljus och släta ytor.)
         ▼
[V3: Aktiv Belysningsfusion & Ren Produktionsarkitektur]
            (Nativ Linux, 4-kanals IR-stroboskop, fotometrisk stereo + stereofusion,
             strikt 4-lagers mjukvaruarkitektur, byggd för verklig lagerdrift.)
```

### V1 — Proof of Concept
* **Hårdvara:** Enkel hembyggd träram, billiga USB-webbkameror, separat Windows-laptop.
* **Lärdomar:** Konceptet med datorseende fungerade, men USB-webbkameror gav osynkade bildrutor, rullande slutare (rolling shutter), varierande vitbalans och opålitliga millimetermått.

### V2 — R&D och Embedded-prototyp
* **Hårdvara:** Raspberry Pi CM4 / CM5, officiella Sony IMX219 CSI-kameror, UART/I2C ToF-sensor, Dymo S100 USB-våg och Honeywell MS5145 handscanner.
* **Lärdomar:** 
  * Dedikerade CSI-kameror gav stabil bildtagning direkt på hårdvarunivå.
  * SGBM (Stereo Semi-Global Block Matching) optimerades ner från 8 sekunder till 0.3 sekunder på CM5.
  * **Problemet med passiv stereo:** Släta vita och bruna kartonger saknar naturlig textur, och varierande belysning (lysrörsflimmer, dagsljus) skapade brus i mätningarna. Experiment med Kinect-strukturerat laserljus lades ner av ögonsäkerhetsskäl (klass 3B IR-laser).
  * **Slutsats från V2:** Belysningen måste kontrolleras aktivt av systemet självt.

### V3 — Produktionsfärdig Ljusfusion & Ren Arkitektur
V3 bygger på ett paradigmskifte: **Vi anpassar inte kameran efter rummets ljus – vi kontrollerar ljuset helt och hållet själva.**

* **Aktiv Fotometrisk Stereo (850nm IR):** 4 riktade 3W IR-lysdioder (Norr, Öst, Söder, Väst) pulsas i 2 ms-blixtar. Genom att analysera hur skuggorna faller från fyra vinklar beräknas paketets lockform, buktningar och ytlutningar ner på sub-millimeternivå.
* **Full Flash Stereo (SGBM):** En femte bildruta tas med alla 4 lysdioder tända samtidigt. Detta skapar ett skuggfritt "studieljus" som gör paketets ytterkanter och hörn knivskarpa mot den mörka mätplattan.
* **ToF-förankring (VL53L5CX):** Höjden förankras mot en central absolut Time-of-Flight-referens.
* **Strikt Linux-arkitektur:** Systemet körs nativt som en Linux-tjänst/CLI på CM5, helt oberoende av skärm eller tangentbord.

---

## 2. Hårdvaruöversikt (V3-stacken)

| Komponent | Modell / Specifikation | Gränssnitt | Funktion |
| :--- | :--- | :--- | :--- |
| **Dator** | Raspberry Pi Compute Module 5 (CM5) | – | Huvudprocessor (4x Cortex-A76 @ 2.4 GHz) |
| **Bärarkort** | Waveshare CM5-IO-BASE-B | – | Strömförsörjning, dubbla CSI, USB, I2C, RTC |
| **Kameror** | 2x Raspberry Pi Camera v2 NoIR (IMX219) | 2x CSI-2 (120 mm baslinje) | Stereopar utan IR-filter (känsliga för 850nm IR) |
| **Belysning** | 4x 3W 850nm IR Power LEDs med 60° linser | PWM / GPIO via MOSFET | Riktat stroboskop för fotometrisk stereo |
| **Drivsteg** | YYNMOS-4 (4-kanals optokopplad MOSFET) | 4x GPIO (3.3V styrning, 5V last) | Blixtsnabb pulsstyrning (< 2 ms) |
| **ToF-sensor** | VL53L5CX (8x8 multizon) | I2C (Buss 1, 400 kHz) | Absolut höjdreferens mot bord och lock |
| **Streckkodsläsare** | Honeywell Metrologic Eclipse MS5145 | USB HID (Linux `evdev`) | Trigger för mätning och paket-ID |
| **Våg** | Dymo S100 (USB Digitalvåg) | USB HID | Automatisk viktavläsning |

---

## 3. Mjukvaruarkitektur (4-Lagersmodellen)

Källkoden i V3 är strikt uppdelad i fyra lager för att separera hårdvarudrivrutiner från matematik och applikationsflöde:

```
src/
├── hardware/           # Lager 1: Hårdvaruabstraktioner (I/O, bussar, protokoll)
│   ├── camera_pair.py  # Dual NoIR CSI-hantering via libcamera/V4L2
│   ├── strobe.py       # 4-kanals MOSFET GPIO-kontroller
│   ├── tof.py          # VL53L5CX I2C-drivrutin
│   ├── scanner.py      # Honeywell MS5145 via evdev
│   └── scale.py        # Dymo S100 USB HID
│
├── algorithms/         # Lager 2: Rena matematiska algoritmer (noll hårdvaruberoende)
│   ├── stereo_matcher.py      # SGBM stereomatchning & djupkarta
│   ├── photometric_stereo.py  # Ytnormaler via NumPy least-squares
│   └── surface_fusion.py      # Poisson FFT höjdfusion (Normaler + Stereodjup)
│
├── core/               # Lager 3: Orkestrering & domänmodeller
│   ├── parcel_station.py      # Mätsekvensen (samlar hårdvara och algoritmer)
│   ├── config.py              # Central kalibrerings- och systemkonfiguration
│   └── models.py              # Dataklasser (MeasurementResult, BoundingBox)
│
└── apps/               # Lager 4: Körbara gränssnitt
    └── cli.py          # Blixtsnabb headless terminalapp för SSH / drift
```

---

## 4. Utvecklingsfilosofi för V3

* **Lugnt och metodiskt tempo:** Varje modul utvecklas och testas enskilt innan den kopplas ihop med nästa.
* **Separata enhetstester för allt:** Alla algoritmer testas med syntetiska data innan hårdvaran kopplas in.
* **Testa utan onödiga risker:** Känsliga komponenter (LEDs, MOSFETs) verifieras isolerat (t.ex. på ESP32) innan de ansluts till Compute Modulen.
* **Målet:** En robust, industriellt pålitlig maskin som kan ställas vid ett riktigt packbord hos en pilotkund (Clean Part AB).
