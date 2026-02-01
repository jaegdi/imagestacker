# ImageStacker - Project Status & Continuation Prompt

## Projektübersicht
Rust-basierte Focus-Stacking-Anwendung mit OpenCV für die Verarbeitung von Fotoserien. Verwendet iced GUI-Framework (v0.13) für die Benutzeroberfläche.

## Aktueller Stand (31. Januar 2026)

### Erfolgreich implementierte Optimierungen

#### 1. Regionale Schärfe-Erkennung (Regional Sharpness Detection)
- **Grid-basierte Analyse**: Teilt Bild in NxN Grid (4x4 bis 16x16 konfigurierbar)
- **Max-Regional Scoring**: Bewertet Bilder basierend auf schärfstem Bereich
- **Perfekt für Focus-Stacking**: Akzeptiert Bilder auch wenn nur ein Bereich scharf ist
- **Permissive Threshold**: Q1 - 3.0*IQR statt Q1 - 1.0*IQR
- **Konfigurierbar**: Sharpness Grid Size Slider in Settings (4x4 bis 16x16)
- **Funktion**: `compute_regional_sharpness()` in `src/sharpness.rs`

#### 2. Winner-Take-All Stacking Algorithmus
- **Ghosting-frei**: Verwendet Winner-Take-All statt Averaging auf allen Pyramid-Ebenen
- **Laplacian Pyramid**: 7 Level Dekomposition mit Energy-basierter Selektion
- **Basis-Level**: Wählt schärfstes Pixel aus jedem Bild ohne Mittelung
- **Höhere Ebenen**: Konsistente Winner-Take-All Strategie durchgehend
- **Ergebnis**: Kristallklare gestackte Bilder ohne Geisterbilder
- **Datei**: `src/stacking.rs`

#### 3. Memory-Optimierungen für SIFT & AKAZE
- **Feature-Limitierung**:
  - SIFT: 3000 Features max (128-dim float Descriptors = 512 bytes/Feature)
  - AKAZE: 3000 Features max (nach Sortierung nach Stärke)
  - ORB: 5000 Features (32-byte binary Descriptors)
  
- **Adaptive Batch-Größen**:
  - ORB: 16 Bilder/Batch (base size)
  - SIFT: 4 Bilder/Batch (base / 4) - sehr hoher Memory-Bedarf
  - AKAZE: 4 Bilder/Batch (base / 4) - nach Sortierung + Limitierung
  
- **Threshold-Anpassungen**:
  - SIFT: nfeatures=3000 (limitiert Extraktion)
  - AKAZE: threshold=0.003, nOctaves=3, explizite Top-3000 Auswahl
  
- **Ergebnis**: Läuft stabil mit 42MP Bildern (7952x5304) ohne OOM-Fehler
- **Datei**: `src/alignment.rs` Lines 16-125, 325-336

#### 4. Umfangreiches GUI-Hilfesystem
- **Separates Help-Fenster**: Öffnet User Manual in eigenem Fenster (900x800px)
- **Externe Markdown-Datei**: Lädt Hilfetext aus `USER_MANUAL.md`
- **8 Kategorien**: Getting Started, Understanding Panes, Settings, Advanced Processing, Tips, Workflow, Technical Details, Troubleshooting
- **Close Button**: Rotes Button zum Schließen des Fensters
- **Multi-Window Support**: Verwendet iced daemon API für separate Fenster
- **Scrollbar**: Vollständige Dokumentation mit scrollbarem Content
- **Datei**: `USER_MANUAL.md` (Content), `src/gui.rs` render_help_window()

#### 5. Responsive Settings-Panel
- **3 organisierte Panes**:
  1. **Alignment & Detection**: Blur threshold, Sharpness grid, Feature detector, Adaptive batches, CLAHE
  2. **Post-Processing**: Noise reduction, Sharpening, Color correction (Contrast, Brightness, Saturation)
  3. **Preview & UI**: Internal preview toggle, Preview dimensions
  
- **Responsives Layout**:
  - **Horizontal** (Fenster ≥ 1200px): 3 Panes nebeneinander, Detector-Buttons vertikal
  - **Vertikal** (Fenster < 1200px): 3 Panes gestapelt, Detector-Buttons horizontal
  - Window Resize Subscription für automatische Anpassung
  - **Adaptive Slider-Breiten**: Passen sich an Layout an (label/slider/value widths)
  
- **Reset to Defaults Button**: Rotes Button zum Zurücksetzen aller Einstellungen
- **Responsive Slider Values**: In horizontalem Layout optimierte Breiten (120/150/60) verhindern vertikales Text-Stacking
- **Datei**: `src/gui.rs` render_settings_panel(), Lines 1028-1420

#### 6. Automatische Verzeichnis-Bereinigung
- **Aligned Cleanup**: Löscht `aligned/` Ordner vor jedem Alignment-Lauf
- **Bunches Cleanup**: Löscht `bunches/` Ordner vor jedem Stacking-Lauf
- **Verhindert**: Alte/inkonsistente Dateien in Output-Verzeichnissen
- **User-freundlich**: Keine manuellen Cleanup-Schritte erforderlich
- **Datei**: `src/gui.rs` Lines 256-283, 403-417

#### 7. Smart Button States
- **Align Button**: Nur aktiv wenn Bilder importiert
- **Stack Button**: Nur aktiv wenn aligned Bilder vorhanden
- **Visual Feedback**: Deaktivierte Buttons klar erkennbar
- **Verhindert**: Fehlerhafte Operationen durch User
- **Datei**: `src/gui.rs` Lines 793-815

#### 8. Thumbnail-Visualisierung mit Farbcodierung
- **Farbige Borders**:
  - **Blau**: Importierte Original-Bilder (Input)
  - **Grün**: Verarbeitete/Generierte Bilder (Aligned, Bunches, Final)
- **2-Column Grid Layout**: Responsive Thumbnail-Darstellung
- **Click to Preview**: Öffnet Bilder in Modal oder System Viewer
- **Right-Click**: Öffnet Bild in externem Editor (konfigurierbar in Settings)
- **Größenkonsistenz**: 120x90px Thumbnails bleiben konstant bei Resize
- **Datei**: `src/gui.rs` render_pane_with_columns()

#### 9. Externe Anwendungsintegration
- **External Viewer**: Konfigurierbar für Linksklick auf Thumbnails
  - Standard: Internes Preview-Modal
  - Optional: System-Viewer (z.B. eog, geeqie, gwenview)
  - Einstellung: Settings → "External Image Viewer Path"
- **External Editor**: Konfigurierbar für Rechtsklick auf Thumbnails
  - Standard: Kein Editor konfiguriert
  - Optional: Bildbearbeitung (z.B. GIMP, Darktable, Krita)
  - Einstellung: Settings → "External Image Editor Path"
- **Mouse Area Wrapper**: Erkennt Links- und Rechtsklick auf Thumbnails
- **Datei**: `src/gui.rs`, `src/config.rs`, `USER_MANUAL.md`

#### 10. Linux RPM Packaging für SUSE Tumbleweed
- **Vollständiges RPM-Paket**: Installierbare Distribution mit allen Dependencies
- **Desktop Integration**: 
  - Application Menu Entry mit Icon
  - MIME-Type Associations (PNG, JPEG, TIFF)
  - FreeDesktop.org kompatibel mit KDE-spezifischen Erweiterungen
- **Icon Integration**: 
  - Multi-Size Icons: 64x64, 128x128, 256x256 (automatische Generierung)
  - HiDPI Support durch hicolor icon theme
  - Source: `icons/imagestacker_icon.png`
- **Build-System**:
  - Rustup-Detection: Vermeidet Konflikte mit System-Cargo/Rust
  - Automatische Dependency-Installation
  - One-Click Installation via `quick-install.sh`
- **Package Contents**:
  - Binary: `/usr/bin/imagestacker`
  - Desktop File: `/usr/share/applications/imagestacker.desktop`
  - Icons: `/usr/share/pixmaps/` und `/usr/share/icons/hicolor/`
  - Post-Install Scripts: Icon-Cache und Desktop-Database Updates
- **Dateien**: 
  - `packaging/linux/imagestacker.spec` (RPM spec file)
  - `packaging/linux/build-rpm.sh` (Build automation)
  - `packaging/linux/quick-install.sh` (One-click installer)
  - `packaging/linux/Makefile` (Build targets)
  - `packaging/linux/README.md` (Packaging documentation)
  - `packaging/linux/INSTALL.md` (Quick start guide)

### Technische Details

#### Wichtige Dateien
- **`src/alignment.rs`** (792 Zeilen): Feature-basierte Alignment mit Memory-Optimierung
  - `extract_features()`: ORB/SIFT/AKAZE mit Feature-Limitierung
  - `align_images()`: Batch-Processing mit detector-spezifischen Batch-Größen
  - Regional sharpness detection mit konfigurierbarem Grid
  - Permissive threshold für Focus-Stacking
  
- **`src/sharpness.rs`** (195 Zeilen): Schärfe-Analyse
  - `compute_regional_sharpness()`: Grid-basierte Analyse
  - `compute_sharpness()`: Globale Schärfe-Messung
  - Laplacian Variance Methode

- **`src/stacking.rs`** (533 Zeilen): Laplacian Pyramid Focus Stacking
  - `stack_images()`: 7-Level Pyramid mit Winner-Take-All
  - Keine Mittelung auf irgendeiner Ebene (ghosting-frei)
  - Energy-basierte Pixelselektion

- **`src/gui.rs`** (1663 Zeilen): iced GUI mit responsivem Design
  - `render_settings_panel()`: 3-Pane Layout mit responsive Umschaltung
  - `render_help_window()`: Separates Help-Fenster mit Close Button
  - `subscription()`: Auto-Refresh + Window Resize Events
  - Smart button states, thumbnail rendering, preview modal
  - Multi-window support via iced daemon API

- **`src/main.rs`**: Daemon-basierte Anwendung
  - Verwendet `iced::daemon()` statt `application()` für Multi-Window Support
  - Initialisiert Hauptfenster mit 1800x1000 Größe
  - Window title handling für main und help windows

- **`USER_MANUAL.md`**: Komplette User-Dokumentation
  - Markdown-Format für einfache Bearbeitung
  - 8 Hauptsektionen mit detaillierten Anleitungen
  - Wird dynamisch von Help-Fenster geladen

- **`src/config.rs`** (67 Zeilen): Processing Configuration
  - `ProcessingConfig`: Alle verarbeitungsrelevanten Parameter
  - `FeatureDetector` enum: ORB, SIFT, AKAZE
  - Serializable für Settings-Persistenz

- **`src/messages.rs`** (53 Zeilen): GUI Message Enum
  - Settings-Messages (Threshold, Grid Size, Detector, etc.)
  - Processing-Messages (Align, Stack, Progress)
  - UI-Messages (Toggle Settings/Help, CloseHelp, Preview, Scroll)
  - Window Resize Message für responsive Layout

#### Dependencies (Cargo.toml)
```toml
[dependencies]
iced = { version = "0.13", features = ["image", "canvas", "tokio"] }
opencv = { version = "0.92", features = ["default"] }
rayon = "1.10"
anyhow = "1.0"
log = "0.4"
env_logger = "0.11"
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
dirs = "5.0"
```

**Wichtig**: `tokio` Feature ist erforderlich für `iced::time::every()` und Event-Subscriptions

#### Performance-Parameter (Konfigurierbar in GUI)
```rust
// Feature Detector (GUI: 3 Buttons)
FeatureDetector::ORB    // Fast, low memory
FeatureDetector::SIFT   // Best quality, high memory
FeatureDetector::AKAZE  // Balanced

// Sharpness Grid Size (GUI: Slider 4-16)
sharpness_grid_size: 4-16  // Default: 4 (4x4 Grid)

// Blur Threshold (GUI: Slider 10-100)
sharpness_threshold: 10.0-100.0  // Default: 30.0

// Adaptive Batch Sizes (GUI: Checkbox)
use_adaptive_batches: bool  // Auto-adjust based on RAM

// CLAHE (GUI: Checkbox)
use_clahe: bool  // Enhance dark images

// Post-Processing (GUI: Checkboxes + Sliders)
enable_noise_reduction: bool
noise_reduction_strength: 1.0-10.0
enable_sharpening: bool
sharpening_strength: 0.0-5.0
enable_color_correction: bool
contrast_boost: 0.5-3.0
brightness_boost: -100.0-100.0
saturation_boost: 0.0-3.0
```

#### Automatische Batch-Größen-Berechnung
```rust
// In src/system_info.rs & src/alignment.rs
let available_ram_gb = get_available_memory_gb();
let base_batch_size = calculate_batch_size(available_ram_gb);

// Detector-spezifische Skalierung:
ORB:   batch_size = base_batch_size (16)
SIFT:  batch_size = base_batch_size / 4 (4)  // 128-dim float = 512 bytes/desc
AKAZE: batch_size = base_batch_size / 4 (4)  // Limited to 3000 features
```

### Bekannte Limitierungen & Nächste Schritte

#### Aktuelle Limitierungen
1. **Memory-Intensiv bei sehr großen Bildern**: 42MP Bilder benötigen conservative Batch-Größen
2. **SIFT/AKAZE langsamer als ORB**: Höhere Qualität kostet Processing-Zeit
3. **Keine GPU-Beschleunigung**: Läuft komplett auf CPU

#### Mögliche zukünftige Verbesserungen
1. **GPU-Beschleunigung**: OpenCV CUDA für Feature Detection
2. **Multi-Threading für Stacking**: Parallele Pyramid-Ebenen-Verarbeitung
3. **Konfigurierbare Pyramid Levels**: User-definierte Tiefe für Stacking
4. **Batch Progress Details**: Zeige aktuellen Batch-Fortschritt (z.B. "Batch 3/10")
5. **Image Quality Metrics**: Automatische Bewertung der Stack-Qualität

### Typische Probleme & Lösungen

#### Problem: Out of Memory bei SIFT/AKAZE
**Lösung**: 
- Automatisch behandelt durch adaptive Batch-Größen (4 Bilder/Batch)
- Bei Bedarf manuell in `src/alignment.rs` weitere Reduzierung möglich
- Alternative: ORB verwenden (schneller, weniger Memory)

#### Problem: Alignment schlägt fehl
**Lösung**: 
- CLAHE aktivieren in Settings (für dunkle Bilder)
- SIFT Detector verwenden (beste Qualität)
- Sharpness Threshold reduzieren (mehr Bilder akzeptieren)
- Sharpness Grid Size erhöhen (16x16 für feinere Analyse)

#### Problem: Ghosting in Final Image
**Lösung**: 
- ✅ Bereits behoben durch Winner-Take-All Algorithmus
- Sollte nicht mehr auftreten

#### Problem: Zu viele/wenige Bilder gefiltert
**Lösung**: 
- Sharpness Threshold in Settings anpassen (Slider 10-100)
- Sharpness Grid Size anpassen (kleineres Grid = strenger)
- Regional sharpness: Akzeptiert Bilder mit mind. einem scharfen Bereich

#### Problem: Finale Bilder zu dunkel/hell
**Lösung**:
- Color Correction in Settings aktivieren
- Contrast Boost anpassen (0.5-3.0)
- Brightness Boost anpassen (-100 bis +100)
- Saturation Boost anpassen (0.0-3.0)

### Build & Run

```bash
cd /home/dirk/devel/rust/imagestacker
cargo build --release  # Für optimierte Performance (wichtig!)
cargo run --release   # Development & Testing
./target/release/imagestacker  # Direct execution
```

**Performance-Tipp**: Immer `--release` verwenden für reale Verarbeitung! Debug-Builds sind 10-100x langsamer.

### Installation (Linux RPM)

```bash
cd /home/dirk/devel/rust/imagestacker/packaging/linux
./quick-install.sh  # One-click build & install
# Oder manuell:
make clean && make rpm
sudo zypper install --allow-unsigned-rpm -f ~/rpmbuild/RPMS/x86_64/imagestacker-0.1.0-1.x86_64.rpm
```

**Nach Installation**:
- Start aus Terminal: `imagestacker`
- Start aus Application Menu: ImageStacker Icon
- Bei KDE Desktop-Integration Problemen:
  ```bash
  kbuildsycoca5 --noincremental  # Rebuild KDE cache
  ```

**Wichtig**: Alte lokale Desktop-Dateien in `~/.local/share/applications/` können System-Einträge überschreiben und zu Startproblemen führen.

### Testing
Testbilder in: `/home/dirk/devel/rust/imagestacker/testimages/`
- Import-Bilder über GUI "Add Images" oder "Add Folder"
- `aligned/`: Aligned Bilder (automatisch erstellt)
- `bunches/`: Batch-gruppierte Bilder (automatisch erstellt)
- `final/`: Gestackte Ergebnisse (automatisch erstellt)

**Test-Workflow**:
1. Start App: `cargo run --release`
2. "Add Folder" → testimages/ auswählen
3. Settings öffnen → Parameter anpassen
4. "Align Images" klicken
5. Warten bis aligned/ Thumbnails erscheinen
6. "Stack Images" klicken
7. Ergebnis in final/ prüfen

## Prompt für Fortsetzung

Nutze diesen Prompt um an diesem Projekt weiterzuarbeiten:

```text
Ich arbeite am Rust-Projekt ImageStacker (@workspace). Es ist eine Focus-Stacking-Anwendung
mit OpenCV und iced GUI (v0.13).

AKTUELLER STAND (31. Januar 2026):
- Regional Sharpness Detection: Grid-basiert (4x4 bis 16x16), max-regional scoring
- Winner-Take-All Stacking: Ghosting-frei durch keine Mittelung
- Memory-Optimierungen: SIFT/AKAZE mit Feature-Limits und adaptiven Batch-Größen
- Responsive Settings UI: 3-Pane Layout mit automatischer Anpassung an Fenstergröße
- Adaptive Slider-Breiten: Verhindert Text-Wrapping in horizontalem Layout
- Help System: Separates Fenster mit USER_MANUAL.md, Multi-Window Support via daemon API
- Smart Features: Auto-cleanup, smart button states, colored thumbnails
- External Applications: Right-click für Editor, Left-click optional für Viewer
- Linux RPM Packaging: Vollständige SUSE Tumbleweed Integration mit Desktop Entry und Icons

MEMORY HANDLING:
- ORB: 16 Bilder/Batch, 5000 Features
- SIFT: 4 Bilder/Batch, 3000 Features (512 bytes/descriptor)
- AKAZE: 4 Bilder/Batch, 3000 Features (nach Sortierung)
- Läuft stabil mit 42MP Bildern (7952x5304)

PACKAGING:
- RPM Build System mit rustup-Detection
- Desktop Integration: KDE-kompatibel, FreeDesktop.org Standards
- Icon Support: Multi-Size (64x64, 128x128, 256x256), HiDPI-ready
- Installation: Terminal (imagestacker) und Application Menu funktionieren
- Wichtig: Lokale Desktop-Dateien in ~/.local/share/applications/ können System-Einträge überschreiben

DATEIEN:
- src/alignment.rs: Feature extraction mit detector-spezifischen Batch-Größen
- src/sharpness.rs: Regional sharpness detection mit Grid-Analyse
- src/stacking.rs: Winner-Take-All Laplacian Pyramid (7 levels, ghosting-frei)
- src/gui.rs: Responsive 3-pane settings, adaptive slider widths, help window, external apps
- src/main.rs: Daemon API für Multi-Window Support
- src/config.rs: ProcessingConfig mit external_viewer_path und external_editor_path
- src/messages.rs: Alle GUI Messages inkl. WindowResized, CloseHelp, External*PathChanged
- USER_MANUAL.md: Externe User-Dokumentation mit "Image Preview & Editing" Sektion
- packaging/linux/*: Komplettes RPM Build-System für SUSE
- icons/imagestacker_icon.png: Application Icon (multi-size generation)

Lies PROJECT_STATUS.md für vollständige Details zu allen Optimierungen.

[Hier deine konkrete Aufgabe einfügen]
```

## Hinweise für AI-Assistenten

- **Immer** `PROJECT_STATUS.md` lesen bei Fortsetzung
- **Regional Sharpness beibehalten** - essentiell für Focus-Stacking
- **Winner-Take-All nicht ändern** - verhindert Ghosting
- **Batch-Processing beibehalten** für Memory-Sicherheit
- **Feature-Limits nicht erhöhen** - führt zu OOM bei großen Bildern
- **Detector-spezifische Batch-Größen kritisch** - nicht vereinheitlichen!
- **Daemon API erforderlich** - Multi-Window Support benötigt `iced::daemon()`
- **Help Content in USER_MANUAL.md** - nicht im Code hardcoden
- **Adaptive Slider-Breiten wichtig** - verhindert Text-Wrapping in responsivem Layout
- **External Applications konfigurierbar** - viewer/editor Pfade in Settings
- **RPM Packaging**: Icon aus `icons/imagestacker_icon.png`, nicht aus testimages
- **Desktop Integration**: Lokale Desktop-Dateien in `~/.local/share/applications/` haben Vorrang vor System-Dateien
- **Parameter-Änderungen dokumentieren** in diesem File
- **Vor größeren Änderungen**: Aktuellen Stand testen mit 42MP Bildern
- **Nach Optimierungen**: Memory-Usage und Performance-Impact dokumentieren

### Code-Review Checkliste

- [ ] Memory-sichere Batch-Verarbeitung verwendet?
- [ ] Feature-Limits für SIFT/AKAZE eingehalten?
- [ ] Parallele Verarbeitung mit rayon wo sinnvoll?
- [ ] Error-Handling mit `Result<>` und proper Logging?
- [ ] GUI-Updates über Messages statt direkte Änderungen?
- [ ] Multi-Window: Daemon API korrekt verwendet?
- [ ] Help Content: In USER_MANUAL.md statt hardcoded?
- [ ] Responsive UI: Slider-Breiten adaptiv an Layout?
- [ ] External Applications: Viewer/Editor Pfade konfigurierbar?
- [ ] Settings in `ProcessingConfig` serializable?
- [ ] Dokumentation in Kommentaren aktualisiert?
- [ ] RPM Packaging: Icon-Pfade korrekt in spec file?
- [ ] Desktop Integration: Absolute Pfade in .desktop file?

---

**Letztes Update**: 31. Januar 2026
**Status**: ✅ Produktionsreif - Alle kritischen Features implementiert
**Architecture**: Multi-Window Support via iced daemon API (v0.13)
**Performance**: Optimal für 42MP Bilder, stabil, ghosting-frei
**Memory**: Conservative Batch-Größen verhindern OOM bei SIFT/AKAZE
**GUI**: Vollständig responsiv, adaptive Slider-Breiten, separates Help-Fenster, external apps, user-freundlich
**Documentation**: USER_MANUAL.md (externe Markdown-Datei)
**Packaging**: RPM für SUSE Tumbleweed mit vollständiger Desktop-Integration, KDE-kompatibel
