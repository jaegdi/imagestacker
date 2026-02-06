# ImageStacker - Project Status & Continuation Prompt

## Projektübersicht

Rust-basierte Focus-Stacking-Anwendung mit OpenCV für die Verarbeitung von Fotoserien. Verwendet iced GUI-Framework (v0.13) für die Benutzeroberfläche mit **vollständiger GPU-Beschleunigung via OpenCL**.

## Aktueller Stand (6. Februar 2026)

### Kern-Algorithmus: Laplacian Pyramid Focus Stacking

#### Winner-Take-All mit Alpha-Kanal-Behandlung

- **7-Level Laplacian Pyramid**: Dekomposition jedes Bildes in Frequenzbänder
- **Energy-basierte Pixelselektion**: Laplacian → AbsDiff → GaussianBlur(3×3)
- **Alpha-gewichtete Energy**: `weighted_energy = sharpness_energy × (alpha / 255.0)`
- **Winner-Take-All**: Pixel mit höchster Energy gewinnt auf jeder Pyramiden-Ebene (kein Averaging → kein Ghosting)
- **BGR-only Pyramid**: Alpha-Kanal wird separat verarbeitet, nicht durch die Pyramide geschickt
- **AND-kombiniertes Alpha**: Kleinstes gemeinsames opakes Gebiet aller Eingangsbilder
- **5px Morphologische Erosion**: Entfernt potentielle Artefakte an Transparenz-Kanten
- **Ergebnis**: Artefaktfreie transparente Ränder in PNG-Ausgabe

#### Modulare Helper-Funktionen (Refactored)

```text
extract_bgr_and_alpha()    - Trennt BGR von Alpha für unabhängige Verarbeitung
init_alpha()               - Initialisiert fused Alpha vom ersten Bild oder erstellt opakes Alpha
to_grayscale()             - Konvertiert BGRA/BGR/Einkanal → Graustufen
compute_sharpness_energy() - Laplacian → AbsDiff → GaussianBlur Pipeline
fuse_layer_with_alpha()    - Winner-Take-All mit Alpha-gewichtetem Energy-Vergleich
update_fused_alpha()       - AND-Kombination der Alpha-Kanäle über alle Bilder
assemble_final_image()     - Merge BGR+Alpha, Erosion(5px), Konvertierung zu 8-Bit
```

### GPU-Beschleunigung (OpenCL)

#### OpenCL GPU-Integration

- **Vollständig GPU-beschleunigt**: Blur Detection, Feature Extraction, Warping, Stacking
- **Thread-sicheres Design**: Global OpenCL Mutex für sichere parallele Verarbeitung
- **UMat-basiert**: Minimiert CPU↔GPU Transfers durch GPU-native Operationen
- **Performance**: 2-6× Speedup je nach Operation
  - Blur Detection: 1-2s pro 42MP Bild (GPU)
  - Feature Extraction: 6.3× schneller mit ORB vs SIFT
  - Warping: ~1.5s pro 42MP Bild (GPU)
  - Gesamte SIFT Alignment: ~130-140s für 46 Bilder
  - Gesamte ORB Alignment: ~89s für 46 Bilder

#### GPU-Operationen

- **Blur Detection (`sharpness.rs`)**: GPU Gaussian blur, Laplacian, Sobel, Grid-Analyse
- **Feature Extraction (`alignment.rs`)**: GPU Preprocessing (Color conversion, CLAHE, Resizing)
- **Image Warping (`alignment.rs`)**: GPU Affine transformations mit UMat
- **Focus Stacking (`stacking.rs`)**: GPU Laplacian pyramid generation und collapse

#### Memory-Management für GPU

- **Adaptive Batch-Größen für große Bilder (>30MP)**:
  - ORB: 3 Bilder gleichzeitig
  - SIFT: 3 Bilder gleichzeitig (mit reduzierter Feature-Anzahl)
  - AKAZE: 2 Bilder gleichzeitig
- **UMat Reference Handling**: Automatisches Cloning beim Transfer zu Mat
- **Typischer RAM-Verbrauch**: 8-10GB für 46×42MP Bilder
- **Keine OOM-Fehler**: Stabil mit großen Image-Sets

### Alignment-Methoden

#### Feature-basiert (ORB/SIFT/AKAZE)

- **ORB**: Schnell, niedriger Speicherbedarf, 5000 Features
- **SIFT**: Beste Qualität, 2000 Features (optimiert von 3000), 512 bytes/descriptor
- **AKAZE**: Ausgewogen, 3000 Features nach Sortierung
- **Adaptive Batch-Größen**: Basierend auf verfügbarem RAM und Bildgröße

#### ECC Precision Alignment

- **Sub-Pixel Genauigkeit**: Enhanced Correlation Coefficient für höchste Präzision
- **4 Motion-Modelle**: Translation (2-DOF), Euclidean (3-DOF), Affine (6-DOF), Homography (8-DOF)
- **Iterative Optimierung**: Bis zu 30.000 Iterationen für 0.01-0.001px Genauigkeit
- **Perfekt für Makro-Focus-Stacking**: Übertrifft Feature-Detektoren bei statischen Motiven
- **Konfigurierbare Parameter**:
  - `ecc_motion_type`: Transformation model (Standard: Homography)
  - `ecc_max_iterations`: 3000-30000 (Standard: 10000)
  - `ecc_epsilon`: 1e-8 bis 1e-4 Konvergenz-Schwellwert (Standard: 1e-6)
  - `ecc_gauss_filter_size`: 3×3 bis 15×15 Pre-Blur Kernel (Standard: 7×7)
  - `ecc_chunk_size`: 4-24 Bilder für parallele Verarbeitung (Standard: 12)
- **GPU-beschleunigt**: Nutzt OpenCV's optimierte `find_transform_ecc()` via OpenCL
- **Automatisches Sharpness-Ranking**: Wählt mittelscharfes Bild als Referenz (stabil)
- **Datei**: `src/alignment.rs` `align_images_ecc()`

### Schärfe-Analyse

#### Regionale Schärfe-Erkennung

- **Grid-basierte Analyse**: Teilt Bild in NxN Grid (4×4 bis 16×16 konfigurierbar)
- **Max-Regional Scoring**: Bewertet Bilder basierend auf schärfstem Bereich
- **Permissive Threshold**: Q1 - 3.0*IQR statt Q1 - 1.0*IQR
- **GPU-beschleunigt**: UMat-basierte Grid-Analyse

#### Sharpness Caching

- **YAML-basiert**: Schärfe-Werte werden pro Bild in `.yml`-Dateien gecacht
- **Verzeichnis**: `<source_folder>/sharpness/` Unterordner
- **Inhalt**: Globale Schärfe, regionale Schärfe-Werte, Metadaten
- **Vorteil**: Wiederholte Analysen nutzen Cache, erhebliche Zeitersparnis

### Erfolgreich implementierte Features

#### 1. Winner-Take-All Stacking mit Alpha-Behandlung

- **Ghosting-frei**: Keine Mittelung auf irgendeiner Pyramid-Ebene
- **Laplacian Pyramid**: 7 Level Dekomposition mit Energy-basierter Selektion
- **Alpha-Kanal**: AND-Kombination + 5px Erosion für artefaktfreie transparente Ränder
- **BGR-only Pyramid**: Alpha wird separat verarbeitet für saubere Ergebnisse
- **Datei**: `src/stacking.rs`

#### 2. Memory-Optimierungen für SIFT & AKAZE

- **Feature-Limitierung**:
  - SIFT: 2000 Features max (128-dim float Descriptors = 512 bytes/Feature)
  - AKAZE: 3000 Features max (nach Sortierung nach Stärke)
  - ORB: 5000 Features (32-byte binary Descriptors)
- **Adaptive Batch-Größen**: ORB 16, SIFT 4, AKAZE 4 Bilder/Batch
- **Ergebnis**: Läuft stabil mit 42MP Bildern (7952×5304) ohne OOM-Fehler

#### 3. GUI-Hilfesystem

- **Browser-basierte Hilfe**: Konvertiert USER_MANUAL.md zu HTML, öffnet im Browser
- **8 Kategorien**: Getting Started, Panes, Settings, Advanced Processing, Tips, Workflow, GPU, Troubleshooting

#### 4. Responsive Settings-Panel

- **3 organisierte Panes**: Alignment & Detection, Post-Processing, Preview & UI
- **Responsives Layout**: Horizontal (≥1200px) oder vertikal (<1200px)
- **Adaptive Slider-Breiten**: Passen sich an Layout an
- **Reset to Defaults Button**

#### 5. Automatische Verzeichnis-Bereinigung

- Löscht `aligned/` und `bunches/` vor jedem neuen Lauf

#### 6. Smart Button States & Thumbnail-Farbcodierung

- Buttons nur aktiv wenn Voraussetzungen erfüllt
- Blau: Input, Grün: Verarbeitete Bilder

#### 7. Externe Anwendungsintegration

- External Viewer (Linksklick) und Editor (Rechtsklick) konfigurierbar

#### 8. Selektives Stacking

- Wähle einzelne aligned Bilder oder Bunches zum Stacking
- Select All/Deselect All für schnelle Massenauswahl

#### 9. Linux RPM Packaging für SUSE Tumbleweed

- Vollständiges RPM-Paket mit Desktop Integration und Icons
- One-Click Installation via `quick-install.sh`

### Technische Details

#### Wichtige Dateien

| Datei | Zeilen | Beschreibung |
|-------|--------|--------------|
| `src/stacking.rs` | 601 | Laplacian Pyramid Focus Stacking mit 7 Helper-Funktionen |
| `src/alignment.rs` | 2100 | Feature-basierte + ECC Alignment mit GPU |
| `src/sharpness.rs` | 463 | GPU-beschleunigte Schärfe-Analyse |
| `src/sharpness_cache.rs` | — | YAML-basiertes Sharpness-Caching |
| `src/config.rs` | 120 | ProcessingConfig, FeatureDetector, EccMotionType |
| `src/messages.rs` | 113 | GUI Message Enum (~65 Varianten) |
| `src/main.rs` | 105 | Daemon-basierte App mit clap CLI |
| `src/gui/` | 5530 | Modulare GUI (handlers/ + views/) |

#### Dependencies (Cargo.toml)

```toml
[package]
name = "imagestacker"
version = "1.0.0"

[dependencies]
iced = { version = "0.13", features = ["image", "canvas", "tokio"] }
opencv = "0.94"
image = "0.25"
rfd = "0.15"
anyhow = "1.0"
log = "0.4"
env_logger = "0.11"
opener = "0.8.4"
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
serde_yaml = "0.9"
dirs = "5.0"
rayon = "1.10"
tokio = { version = "1.0", features = ["fs", "rt-multi-thread"] }
pulldown-cmark = "0.9"
clap = { version = "4.5", features = ["derive"] }
chrono = "0.4"
```

### Bekannte Limitierungen

1. **Memory-Intensiv bei sehr großen Bildern**: 42MP Bilder benötigen conservative Batch-Größen
2. **SIFT/AKAZE langsamer als ORB**: Höhere Qualität kostet Processing-Zeit
3. **ECC benötigt ähnliche Belichtung**: Funktioniert nicht gut bei stark unterschiedlichen Belichtungen
4. **Stärkere Beschneidung bei Transparenz**: AND-Alpha + Erosion schneidet etwas mehr ab als nötig (bewusster Trade-off für artefaktfreie Ergebnisse)

### Mögliche zukünftige Verbesserungen

1. **Konfigurierbare Pyramid Levels**: User-definierte Tiefe für Stacking
2. **Batch Progress Details**: Zeige aktuellen Batch-Fortschritt
3. **Image Quality Metrics**: Automatische Bewertung der Stack-Qualität
4. **Konfigurierbare Erosion**: Slider für Erosion-Stärke (derzeit fest 5px)
5. **OR-Alpha Option**: Alternative Alpha-Kombination für weniger Beschneidung

### Typische Probleme & Lösungen

#### Problem: Out of Memory bei SIFT/AKAZE

**Lösung**: Automatisch behandelt durch adaptive Batch-Größen. Alternative: ORB verwenden.

#### Problem: Alignment schlägt fehl

**Lösung**: CLAHE aktivieren, SIFT/ECC verwenden, Threshold reduzieren.

#### Problem: Weiße/helle Artefakte an transparenten Rändern

**Lösung**: ✅ Behoben durch AND-Alpha + 5px Erosion + BGR-only Pyramid.

#### Problem: Ghosting in Final Image

**Lösung**: ✅ Behoben durch Winner-Take-All Algorithmus (keine Mittelung).

### Build & Run

```bash
cd /home/dirk/devel/rust/imagestacker
cargo build --release    # Für optimierte Performance (wichtig!)
cargo run --release      # Development & Testing

# Mit Debug-Logging:
RUST_LOG=debug cargo run --release
```

**Performance-Tipp**: Immer `--release` verwenden! Debug-Builds sind 10-100× langsamer.

### Installation (Linux RPM)

```bash
cd /home/dirk/devel/rust/imagestacker/packaging/linux
./quick-install.sh  # One-click build & install
```

## Prompt für Fortsetzung

```text
Ich arbeite am Rust-Projekt ImageStacker (@workspace). Es ist eine Focus-Stacking-Anwendung
mit OpenCV und iced GUI (v0.13).

AKTUELLER STAND (6. Februar 2026):
- Laplacian Pyramid (7 Level) mit Winner-Take-All: Ghosting-frei
- Alpha-Kanal-Behandlung: AND-Alpha + 5px Erosion für artefaktfreie transparente Ränder
- BGR-only Pyramid: Alpha separat verarbeitet
- GPU-Beschleunigung: Vollständig via OpenCL (2-6× Speedup)
- 4 Alignment-Methoden: ORB, SIFT, AKAZE, ECC (Sub-Pixel)
- Sharpness Caching: YAML-basiert pro Bild
- Modularer Stacking-Code: 7 Helper-Funktionen in stacking.rs
- Responsive Settings UI: 3-Pane Layout

STACKING-ALGORITHMUS:
- extract_bgr_and_alpha() → init_alpha() → to_grayscale()
- compute_sharpness_energy() → fuse_layer_with_alpha()
- update_fused_alpha() (AND) → assemble_final_image() (Erosion 5px)

DATEIEN:
- src/stacking.rs (601): Focus Stacking mit 7 Helper-Funktionen
- src/alignment.rs (2100): Feature-basiert + ECC Alignment
- src/sharpness.rs (463): GPU-beschleunigte Schärfe-Analyse
- src/config.rs (120): ProcessingConfig mit ECC-Parametern
- src/gui/ (5530): Modulare GUI (handlers/ + views/)

Lies PROJECT_STATUS.md für vollständige Details.

[Hier deine konkrete Aufgabe einfügen]
```

## Hinweise für AI-Assistenten

- **Immer** `PROJECT_STATUS.md` lesen bei Fortsetzung
- **Winner-Take-All nicht ändern** - verhindert Ghosting
- **Alpha-Behandlung**: AND-Alpha + Erosion beibehalten (45+ Iterationen bis zur Lösung)
- **BGR-only Pyramid**: Alpha NICHT durch die Pyramide schicken
- **Batch-Processing beibehalten** für Memory-Sicherheit
- **Feature-Limits nicht erhöhen** - führt zu OOM bei großen Bildern
- **Detector-spezifische Batch-Größen kritisch** - nicht vereinheitlichen!
- **Daemon API erforderlich** - Multi-Window Support benötigt `iced::daemon()`
- **log::debug! für Details, log::info! nur für High-Level** Messages
- **Vor größeren Änderungen**: Aktuellen Stand testen mit 42MP Bildern

### Code-Review Checkliste

- [ ] Memory-sichere Batch-Verarbeitung verwendet?
- [ ] Feature-Limits für SIFT/AKAZE eingehalten?
- [ ] Alpha-Kanal korrekt behandelt (AND + Erosion)?
- [ ] BGR-only Pyramid ohne Alpha-Kanal?
- [ ] log::debug! für Detail-Logging, log::info! für High-Level?
- [ ] Error-Handling mit `Result<>` und proper Logging?
- [ ] GUI-Updates über Messages statt direkte Änderungen?
- [ ] Settings in `ProcessingConfig` serializable?
- [ ] Dokumentation aktualisiert?

---

**Letztes Update**: 6. Februar 2026
**Version**: 1.0.0
**Status**: ✅ Produktionsreif
**Stacking**: 7-Level Laplacian Pyramid, Winner-Take-All, AND-Alpha + 5px Erosion
**Performance**: GPU-beschleunigt via OpenCL (2-6×), stabil für 42MP Bilder
**GUI**: 5530 Zeilen, modulare Architektur (handlers/ + views/)
**Packaging**: RPM (SUSE), macOS .app, Windows portable ZIP
