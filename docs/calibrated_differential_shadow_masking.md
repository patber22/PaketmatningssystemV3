# Calibrated Differential Shadow Magnitude (CDSM)
## Robust Paketmaskning och Konturextrahering via Riktade Skuggfält

Detta dokument sammanfattar och bevarar den metod för 2D-paketmaskning som utvecklades den 25 oktober 2026. Metoden löser paketsegmentering med millimeterprecision på enfärgade, texturlösa ytor och träunderlag **helt utan hårdkodade tröskelvärden, maskininlärning eller antaganden om rektangulära rätblock.**

---

## 1. Fysikalisk Princip och Bakgrund

Ett stort problem vid optisk paketmätning under standardbelysning är att enfärgade paket (t.ex. en vit kartong eller en blå bok) ofta har nästan exakt samma genomsnittliga gråskalei-intensitet som underlaget (OSB-träbord: ~108 ADU, vit bok: ~109 ADU). Träbordets ådring skapar dessutom falska kanter som bryter sönder vanliga kantdetektorer (Canny m.fl.).

Genom att använda riggens fyra 850nm IR-LED-lampor monterade runt centrala kameran (CAM1) i motstående par (Sydost/Nordväst och Sydväst/Nordost) utnyttjar vi skuggornas fysik:
1. **Paketets toppyta:** Belyses direkt av samtliga 4 lampor (aldrig skugga).
2. **Underlaget runt paketet:** Skuggas djupt i exakt den riktning ljuset blockeras av paketets väggar.
3. **Mellanrummet:** Skuggan börjar exakt vid paketets fysiska fot/kant.

---

## 2. Steg-för-steg Algoritm (CDSM)

### Steg 1: Kalibrerad intensitetsnormalisering
De fyra råa bilderna ($I_1, I_2, I_3, I_4$) från mittkameran CAM1 skalas med de kalibrerade lampstyrkorna ($s_1..s_4$) från `config/calibration/data/photometric.json`:
$$I_{k,\text{norm}} = \frac{I_k}{s_k}, \quad k \in \{1, 2, 3, 4\}$$
* **Syfte:** Raderar fysiska styrkeskillnader mellan LED-armarna (upp till 14 % skillnad mellan armarna).

### Steg 2: Motstående differentiella par
Bilder från motstående lampor subtraheras för att isolera skuggorna i respektive axel:
* **Horisontellt skuggfält (X):** $\Delta I_x = |I_{1,\text{norm}} - I_{3,\text{norm}}|$
* **Vertikalt skuggfält (Y):** $\Delta I_y = |I_{2,\text{norm}} - I_{4,\text{norm}}|$
* **Syfte:** Statisk belysning, träbordets textur och neutrala bakgrunder släcks ut.

### Steg 3: Icke-linjär brusundertryckning (Kvadratisk dämpning)
Varje differensbild skalas med sitt maxvärde och kvadreras:
$$S_x = \left(\frac{\Delta I_x}{\max(\Delta I_x)}\right)^2 \times 255$$
$$S_y = \left(\frac{\Delta I_y}{\max(\Delta I_y)}\right)^2 \times 255$$
* **Syfte:** Låga bakgrundsvariationer dämpas kvadratiskt mot noll ($10 \rightarrow 0.4$), medan starka skuggor behålls intakta.

### Steg 4: Vektormagnitud (2D Shadow Field Fusion)
De två vinkelräta skuggfälten kombineras via euklidisk magnitud och skalas till 0–255:
$$M_{\text{raw}} = \sqrt{S_x^2 + S_y^2}$$
$$M = \frac{M_{\text{raw}}}{\max(M_{\text{raw}})} \times 255$$
* **Resultat:** En obruten, lysande "skuggkrage" runt hela paketet med en mörk, sammanhängande ö i mitten (paketets toppyta).

### Steg 5: Adaptiv tröskling via Otsu (Inga hårdkodade siffror)
Otsus algoritm körs på det aktiva bordsområdet (bed ROI):
$$T = \text{Otsu}(M)$$
$$B_{\text{skugga}} = (M > T)$$
* **Syfte:** Otsu hittar automatiskt dalen mellan den mörka ön och den ljusa skuggkragen (t.ex. $T = 63.0$ på blå bok och $T = 64.0$ på liten vit bok). Tröskeln anpassar sig automatiskt till varje enskild bild och belysning.

### Steg 6: Ö-isolering via inverterad flödesfyllning (Island Extraction)
1. En morfologisk stängning (11x11 rektangel) tätar eventuella mikroskopiska glipor i skuggkragen vid hörnen.
2. Flödesfyllning (`cv2.floodFill`) startar från ytterkanten $(0, 0)$ och fyller bordet utanför skuggan.
3. Det isolerade hålet i mitten som färgen inte nådde är själva paketet:
$$\text{Mask}_{\text{paket}} = \neg \text{Flood}$$
* **Syfte:** Skiljer 100 % på det mörka bordet utanför och det mörka paketet inuti skuggringen.

### Steg 7: Konturextrahering (`cv2.RETR_EXTERNAL`)
På den rena paketmasken körs `cv2.findContours(..., cv2.RETR_EXTERNAL)`:
* **Sann fysisk 2D-kontur:** Följer påsens alla veck, flikar, bucklor och runda hörn. Inre veck och text på paketet ignoreras automatiskt av `RETR_EXTERNAL`.

---

## 3. Verifierade Benchmark-Resultat

| Testobjekt | Fysiska mått (GT) | Teoretiska pixlar | CDSM-mätt mask | Otsu-tröskel | Avvikelse (px / mm) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **blue_book** (stor blå bok) | $251 \times 173 \times 72$ mm | $676 \times 456$ px | **$681 \times 464$ px** | **63.0** | +5 px / ~2.0 mm |
| **small_book** (liten vit bok) | $178 \times 110 \times 27$ mm | $437 \times 269$ px | **$438 \times 271$ px** | **64.0** | +1 px / ~0.5 mm |

* **Körtid på Raspberry Pi CM5:** < 15 millisekunder.

---

## 4. Nästa steg för morgondagen

1. **Testa med fler fysiska objekt:**
   - Oregelbundna påsar (poly mailers), vadderade kuvert och kartonger med veck.
2. **Testa med en bula (Bula-simulering):**
   - Placera ett litet objekt ovanpå den stora boken för att simulera en bula på en plan yta.
   - Verifiera att skuggorna/differensen fångar bulans gränser och textur.
3. **Steg 2: Höjdberäkning via skuggor:**
   - Använda skuggstrålarnas geometri från lampvektorerna för att triangulera höjden $H$ och $Z$ för konturpunkterna.
