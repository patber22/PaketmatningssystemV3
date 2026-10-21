# Projekt-Färdplan & Uppgiftslista: Paketmätningssystem V3

Denna lista är projektets centrala minne och kompass. Den är strikt uppdelad i **sekventiella, atomiska steg**.
Regeln är: **Endast EN aktiv puck i taget.** Allt annat vilar i kön så att du slipper ha hela projektet snurrande i huvudet.

---

## 📌 NUVARANDE STATUS & NÄSTA STEG

* **Nivå:** Fas 3 (Drivrutiner & Hårdvarutest på CM5 – Mjukvara)
* **AKTIV PUCK JUST NU:** 🟢 **Steg 3.1 – ToF On-Demand / Burst i mjukvaran**

---

## Detaljerad Steg-för-steg-lista (En sak i taget)

### 🧱 FAS 2: Mekanik & Fysisk Montering på Träriggen (HELT KLAR! 🎉)

- [x] **Steg 2.1: Utskrift & Torrmontering (KLART!)**
  * **Genomfört:** Alla delar utskrivna. `Sensorlock` justerat med extra plats för CSI-kablarna. Chassit provmonterat på träbalken och sitter fint!

- [x] **Steg 2.2: Sensormontering & Kapsling under balken (KLART!)**
  * **Genomfört:** Båda NoIR-kamerorna och ST VL53L5CX ToF monterade och fastskruvade. ToF-kabel dragavlastad med insmält standoff + buntband. Kablar dragna till CM5 och verifierade live (båda kamerorna och ToF på 0x29 svarar och fungerar perfekt!).

- [x] **Steg 2.3: Lödning & Kabeldragning av de 4 LED-armarna (KLART! 🎉)**
  * **Genomfört:** Alla 4x 3W 850nm IR-lampor lödda och monterade i armarna med 60°-linser och lock. YYNMOS-4 MOSFET kopplad med 6.8 Ω seriemotstånd. Live-verifierat på CM5 med sekventiell 5-bildsfotografering på CAM0: Alla 4 individuella riktningsskuggor (SE, SW, NW, NE) samt full 4-vägs kombinationsbelysning fungerar precis enligt plan!

---

### 💻 FAS 3: Drivrutiner & Hårdvarutest på CM5 (Mjukvara)

- [ ] **Steg 3.1: ToF On-Demand / Burst i mjukvaran**
  * **Uppgift:** Uppdatera `src/hardware/tof.py` med `start_ranging()` och `stop_ranging()` så att 940nm-lasern vilar i mörker som standard och bara tar blixtmätningar vid behov.
  * **Definition of Done:** Enhetstester i `tests/test_tof.py` passerar på PC och CM5.

- [ ] **Steg 3.2: Foto-experiment (ToF PÅ vs AV på NoIR-kameran)**
  * **Uppgift:** Köra ett snabbt testskript på CM5 som tar två NoIR-foton på en kartong (ett med ToF-lasern tänd, ett med den släckt).
  * **Definition of Done:** Vi ser bilderna och utvärderar om ToF-mönstret ger textur för stereomatchning.

- [ ] **Steg 3.3: Dual Camera & Stroboskopsekvens (`src/hardware/dual_camera.py`)**
  * **Uppgift:** Skriva drivrutinen som initierar båda NoIR-kamerorna synkront på CAM0 och CAM1, och synkar 4x riktade LED-blixtar via GPIO (Active-LOW) med kameraslutaren.
  * **Definition of Done:** Skript tar 4 riktade bilder + 1 stereopar och sparar till disk på < 250 ms.

- [ ] **Steg 3.4: Dymo S100 Vågdrivrutin (`src/hardware/scale.py`)**
  * **Uppgift:** Läsa USB HID-rådata från Dymo S100 och returnera en ren `WeightReading`.
  * **Definition of Done:** Enhetstest och liveskript läser stabil vikt i gram.

---

### 📐 FAS 4: Kalibrering & Algoritmmotor (Datorseende)

- [ ] **Steg 4.1: Stereokalibrering på träriggen**
  * **Uppgift:** Skriv ut ett ChArUco/schackbräde. Fota med båda kamerorna och kör OpenCV stereo-kalibrering för att spara `stereo_calibration.json`.
  * **Definition of Done:** Genomsnittligt reprojektionsfel under 0.5 pixlar.

- [ ] **Steg 4.2: Tomt bord-kalibrering (Referensplan $Z_{\text{bord}}$)**
  * **Uppgift:** Skapa skript som läser 64 ToF-zoner + fotometrisk normal på tomt bord och sparar ett matematiskt 3D-nollplan.
  * **Definition of Done:** Referensmatrisen sparad och verifierad i millimeter.

- [ ] **Steg 4.3: Mätmotor (2D Kontur + Stereo SGBM)**
  * **Uppgift:** Implementera `src/algorithms/stereo_matcher.py`. Hög upplösning för 2D-konturer ($L \times B$), nedskalad SGBM för planhöjd ($H$).
  * **Definition of Done:** Returnerar färdigt `MeasurementResult` på testpaket med $< 5\text{ mm}$ fel.

---

### 🚀 FAS 5: Systemintegration & Prestandatest (< 0.6s)

- [ ] **Steg 5.1: Huvudpipeline (`src/core/pipeline.py`)**
  * **Uppgift:** Knyta ihop Honeywell-scanner $\rightarrow$ Våg $\rightarrow$ ToF $\rightarrow$ Blixtsekvens $\rightarrow$ Mätmotor.
  * **Definition of Done:** En scannad streckkod triggar full mätning och loggar resultat till JSON/konsol.

- [ ] **Steg 5.2: Tidmätning & Optimering**
  * **Uppgift:** Mäta exakt tidsåtgång per delsteg och säkerställa att hela cykeln landar under 0.6 sekunder på CM5.

---

### 🏭 FAS 6: Slutrigg hos Clean Part AB (Framtiden)
* Byggs i $1.25\text{–}1.30\text{ m}$ aluminiumprofiler för att hantera $80 \times 60 \times 50\text{ cm}$-paket. All mjukvara och elektronik flyttas bara över direkt.
