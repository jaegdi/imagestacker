# ImageStacker

Eine leistungsstarke, GPU-beschleunigte Focus-Stacking-Anwendung zum Kombinieren mehrerer Bilder mit unterschiedlichen Fokuspunkten zu einem einzigen scharfen Bild. Entwickelt mit Rust, OpenCV (OpenCL) und dem Iced GUI-Framework.

**Version 1.0.0**

## Funktionen

- **Automatische Bildausrichtung**: Merkmalserkennung (ORB, SIFT, AKAZE) und ECC-Subpixel-Präzisionsausrichtung
- **Focus Stacking**: 7-stufige Laplace-Pyramide mit Winner-Take-All-Schärfeauswahl (ohne Geisterbilder)
- **Alphakanal-Unterstützung**: Transparente PNG-Unterstützung mit UND-verknüpftem Alpha und Erosion für artefaktfreie Kanten
- **GPU-Beschleunigung**: OpenCL-basierte Verarbeitung über OpenCV UMat (2–6× Beschleunigung)
- **Regionale Schärfeerkennung**: Rasterbasierte Analyse für intelligente Unschärfefilterung
- **Stapelverarbeitung**: Adaptive Speicherverwaltung für große Bildserien (42MP+)
- **Nachbearbeitung**: Rauschreduzierung, Schärfung, Farbkorrektur
- **Moderne Benutzeroberfläche**: Dunkles Design, interne Vorschau, Integration externer Editoren, Auswahlmodi
- **Schärfeanalyse**: Bildweise Schärfe-Zwischenspeicherung mit YAML-Persistenz
- **Kommandozeilenschnittstelle**: Automatisierungsunterstützung mit `--import`-Parameter

## Schnellstart

### GUI-Modus
```bash
imagestacker
```

### Ordner automatisch importieren
```bash
imagestacker --import /pfad/zu/bildern
```

## Arbeitsablauf

1. **Ordner hinzufügen** – Bilder aus einem Verzeichnis importieren
2. **Bilder ausrichten** – Alle Bilder automatisch ausrichten (ORB/SIFT/AKAZE/ECC)
3. **Ausgerichtete stapeln** – Ausgerichtete Bilder auswählen und stapeln
4. **Ergebnis anzeigen** – Vorschau im Final-Bereich

## Stacking-Algorithmus

ImageStacker verwendet eine **7-stufige Laplace-Pyramide** mit **Winner-Take-All**-Pixelauswahl:

1. Jedes Bild wird in eine Laplace-Pyramide zerlegt (7 Stufen)
2. Die Schärfeenergie wird pro Pixel berechnet mittels Laplacian → AbsDiff → GaussianBlur
3. Bei Bildern mit Alphakanal wird die Energie gewichtet: `Energie × (Alpha / 255)`
4. Auf jeder Pyramidenstufe gewinnt der Pixel mit der höchsten Energie (kein Mitteln → keine Geisterbilder)
5. Alphakanäle werden UND-verknüpft über alle Bilder (kleinste gemeinsame opake Fläche)
6. Eine morphologische Erosion von 5px entfernt mögliche Kantenartefakte an Transparenzgrenzen

Der Algorithmus verarbeitet BGR-Kanäle durch die Pyramide unabhängig vom Alphakanal und gewährleistet so saubere transparente Kanten im Endergebnis.

## Konfiguration

### Ausrichtungsmethoden

| Detektor | Geschwindigkeit | Qualität | Geeignet für |
|----------|-----------------|----------|--------------|
| **ORB** | ⚡ Schnell | Gut | Allgemeine Nutzung, Aufnahmen aus der Hand |
| **SIFT** | 🐌 Langsam | Beste | Maximale Ausrichtungsqualität |
| **AKAZE** | ⚖️ Mittel | Gut | Ausgewogener Kompromiss |
| **ECC** | 🔬 Variabel | Ultra/Subpixel | Makro/Stativ, statische Motive |

### Verarbeitungsoptionen

- **Schärfe-Schwellwert**: 10–10000 (Standard: 3000)
- **Schärfe-Raster**: 4×4 bis 16×16 (Standard: 4×4)
- **CLAHE**: Verbessert dunkle Bilder für bessere Ausrichtung
- **Nachbearbeitung**: Rauschreduzierung, Schärfung, Kontrast/Helligkeit/Sättigung

### Externe Anwendungen

Konfigurierbar unter Einstellungen → Vorschau & UI:

- **Externer Bildbetrachter**: Linksklick öffnet Bilder (z. B. eog, geeqie)
- **Externer Editor**: Rechtsklick öffnet zur Bearbeitung (z. B. GIMP, Darktable)
- **Standardfont**: Aus DropDown Feld auswählen. (Standard: DejaVu Sans)

## Systemanforderungen

- **OpenCV** ≥ 4.5 (4.12+ empfohlen für volle OpenCL-Unterstützung)
- **OpenCL**-fähige GPU (optional, aber empfohlen für 2–6× Beschleunigung)
- **GTK3**
- **8-16 GB+ RAM** (16 GB freier Speicher empfohlen für 40MP+-Bilder, also eher 32GB System)
- **Rust** ≥ 1.70 (zum Kompilieren aus dem Quellcode)

## Dokumentation

Ausführliche Nutzungsanweisungen finden Sie unter:
- `/usr/share/doc/imagestacker/USER_MANUAL.md` (nach der Installation)
- Oder `USER_MANUAL.md` im Quellcode-Repository
- `PROJECT_STATUS.md` für technische Architekturdetails

## Installation

### Aus RPM (SUSE/openSUSE)
```bash
sudo zypper install imagestacker-1.0.0-1.rpm
```

### Aus dem Quellcode
```bash
cargo build --release
./target/release/imagestacker
```

## Pakete bauen

### Linux RPM (SUSE Tumbleweed)
```bash
cd packaging/linux
./quick-install.sh     # Ein-Klick-Build & Installation
# oder
./build-rpm.sh         # Nur RPM bauen
```

### macOS
```bash
cd packaging/macos
./build.sh
```

### Windows
```powershell
cd packaging/windows
./build.ps1
```

## Lizenz

Siehe LICENSE-Datei für Details.

## Projekt

- GitHub: https://github.com/jaegdi/imagestacker
- Version: 1.0.0

## Erstellt mit

- **Rust** – Systemprogrammiersprache
- **OpenCV 0.94** (opencv-rust) – Bildverarbeitung mit OpenCL-GPU-Beschleunigung
- **Iced 0.13** – Plattformübergreifendes GUI-Framework
- **Rayon** – Datenparallelismus für Stapelverarbeitung
- **Tokio** – Asynchrone Laufzeitumgebung für I/O-Operationen
