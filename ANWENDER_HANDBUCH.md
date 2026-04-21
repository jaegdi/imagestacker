# 📖 Rust Image Stacker - Anwenderhandbuch

**Version 1.0.0 - Focus Stacking Anwendung**

---

## 🚀 Erste Schritte

### Schnellstart

Der einfachste Weg zu starten ist über **"Add Folder"**:

1. Klicken Sie auf die Schaltfläche 'Ordner hinzufügen' oder starten Sie imagestacker mit dem Parameter --import
   `imagestacker --import <path to directory with images>`
2. Wählen Sie einen Ordner aus, der Ihre Fokus-Stack-Bilder enthält
3. **Automatisches Laden für bestehende Projekte:**
   - Alle Bilder im Ordner → Bereich **Importiert**
   - Vorhandene Bilder im Unterordner `sharpness/` → Bereich **Schärfe**
   - Vorhandene Bilder im Unterordner `aligned/` → Bereich **Ausgerichtet**
   - Vorhandene Bilder im Unterordner `bunches/` → Bereich **Stapelgruppen**
   - Vorhandene Bilder im Unterordner `final/` → Bereich **Finale Bilder**

Dies ermöglicht es Ihnen, die Arbeit an einem früheren Projekt fortzusetzen oder bestehende Ergebnisse zu sichten!

**Bei einem neuen Projekt mit einem Satz Fokus-Bracketing-Bilder können Sie diesen Schritten folgen**:

- Wählen Sie "Ordner importieren" und selektieren Sie im Dateidialog den Ordner mit Ihrem Bildersatz.
- Wählen Sie "Schärfe erkennen", dies berechnet die Schärfe für Teilbereiche jedes Bildes.
- Wählen Sie "Ausrichten", dies richtet alle Bilder geometrisch zueinander aus.
- Wählen Sie "Ausgerichtete stapeln", dies stapelt die ausgerichteten Bilder, indem aus jedem Bild die scharfen Teile ausgewählt werden, und erstellt "Bunches" (Zwischenstapel) sowie ein finales Bild.

### Nutzung über die Kommandozeile

Sie können die Anwendung auch mit einem bereits vorgeladenen Ordner starten:

```bash
# Linux/macOS
./imagestacker --import /path/to/your/images

# Windows
imagestacker.exe --import C:\path\to\your\images

# Short form
./imagestacker -i /path/to/images
```

Die Anwendung wird:

- Das GUI-Fenster öffnen
- Den angegebenen Ordner automatisch laden
- Nach Bildern und vorhandenen Ergebnissen suchen (sharpness/, aligned/, bunches/, final/) und diese laden
- Sofort bereit für die Verarbeitung sein

**Tipp:** Relative Pfade werden automatisch in absolute Pfade umgewandelt.

### Alternative: Einzelne Bilder hinzufügen

- Klicken Sie auf 'Bilder hinzufügen', um spezifische Bilddateien auszuwählen
- Unterstützte Formate: JPG, PNG, TIFF
- Bilder erscheinen im Bereich 'Importiert'

---

## 📂 Die Programmbereiche verstehen

### Importiert (Dunkelgrauer Rand)

- Aus Ihrem Ordner geladene Originalbilder
- Klicken Sie auf ein Vorschaubild für die Vollbildansicht
- Dies sind Ihre Quellbilder für die Ausrichtung

### Schärfe (Sharpness)

- Bilder nach der Schärfeberechnung
- Dies sind YAML-Dateien mit Verknüpfung zum Originalbild und den berechneten Schärfedaten
- Gespeichert unter `<Quellordner>/sharpness/`
- Dargestellt als Originalbild mit einer Textüberlagerung der Schärfedaten

### Ausgerichtet (Aligned - Dunkelgrauer Rand)

- Bilder nach dem Ausrichtungsprozess
- Geometrisch korrigiert, um zueinander zu passen
- Gespeichert unter: `<Quellordner>/aligned/`
- **Selektives Stapeln:** Wählen Sie aus, welche ausgerichteten Bilder gestapelt werden sollen

### Stapelgruppen (Bunches - Dunkelgrauer Rand)

- Zwischenbilder während des Stacking-Prozesses
- Zeigt das schrittweise Stapeln von Bildgruppen
- Gespeichert unter: `<Quellordner>/bunches/`
- **Selektives Stapeln:** Wählen Sie aus, welche Gruppen zum Endergebnis kombiniert werden sollen

### Finale Bilder (Final - Dunkelgrauer Rand)

- Fertige Focus-Stacking-Ergebnisse
- Enthält scharfe Details aus allen Quellbildern
- Gespeichert unter: `<Quellordner>/final/result_0001.png`, `result_0002.png`, etc.
- Jeder Stacking-Vorgang erzeugt ein neues, nummeriertes Ergebnis

---

## 🖼️ Bildvorschau & Bearbeitung

### Interaktionen mit Vorschaubildern

Jedes Vorschaubild in allen Bereichen (Importiert, Ausgerichtet, Stapelgruppen, Final) unterstützt:

- **Linksklick:**

  - Öffnet das Bild in der internen Vorschau (wenn in den Einstellungen aktiviert)
  - Oder öffnet es im konfigurierten externen Betrachter (wenn interne Vorschau deaktiviert)
- **Rechtsklick:**

  - Öffnet das Bild in Ihrem konfigurierten externen Editor
  - Ideal für schnelle Bearbeitungen in GIMP, Darktable, Photoshop, etc.
  - Nutzt den Systemstandard, falls kein Editor konfiguriert ist

### Navigation in der Bildvorschau

Wenn Sie ein Bild in der internen Vorschau betrachten, können Sie durch alle Bilder des aktuellen Bereichs navigieren:

- **Pfeiltasten:**

  - **Pfeil LINKS:** Vorheriges Bild im Bereich anzeigen
  - **Pfeil RECHTS:** Nächstes Bild im Bereich anzeigen
  - **ENTF-Taste:** Bild löschen; im Schärfe-Bereich werden sowohl die YAML-Datei als auch die Original-Bilddatei gelöscht
  - **ESC:** Vorschau schließen
- **Mausrad:**

  - **Scrollen nach oben:** Vorheriges Bild anzeigen
  - **Scrollen nach unten:** Nächstes Bild anzeigen
- **Navigations-Schaltflächen:**

  - **< Zurück:** Zum vorherigen Bild gehen
  - **Weiter >:** Zum nächsten Bild gehen
- **Informationsanzeige:**

  - Zeigt die aktuelle Position: "Bild X von Y"
  - Erinnert an die Tastenkombinationen zur Navigation

**Tipp:** Die Navigation ist endlos – nach dem letzten Bild führt "Weiter" zurück zum ersten Bild und umgekehrt.

### Konfiguration externer Anwendungen

Öffnen Sie **Einstellungen** → Bereich **Vorschau & UI**:

**Externer Bildbetrachter (für Linksklick):**

- Wird verwendet, wenn "Interne Vorschau verwenden" deaktiviert ist
- Geben Sie den vollständigen Pfad zu Ihrem bevorzugten Bildbetrachter ein:
  - **Linux:** `/usr/bin/eog` (Eye of GNOME), `/usr/bin/geeqie`, `/usr/bin/gwenview`
  - **macOS:** `/Applications/Preview.app/Contents/MacOS/Preview`
  - **Windows:** `C:\Program Files\IrfanView\i_view64.exe`
- Leer lassen, um den Standard-Betrachter des Systems zu nutzen

**Externer Bildeditor (für Rechtsklick):**

- Wird zum Bearbeiten von Bildern beim Rechtsklick auf Vorschaubilder verwendet
- Geben Sie den vollständigen Pfad zu Ihrem Bildeditor ein:
  - **Linux:** `/usr/bin/gimp`, `/usr/bin/darktable`, `/usr/bin/krita`
  - **macOS:** `/Applications/GIMP.app/Contents/MacOS/gimp`
  - **Windows:** `C:\Program Files\GIMP\bin\gimp.exe`, `C:\Program Files\Adobe\Photoshop\Photoshop.exe`
- Leer lassen, um den Standard-Betrachter des Systems zu nutzen

**Tipp:** Verwenden Sie absolute Pfade für die beste Kompatibilität. Die Einstellungen werden automatisch gespeichert.

### Einstellungen der internen Vorschau

- **Vorschau Max. Breite:** Maximale Breite des Vorschaufensters (400-2000px)
- **Vorschau Max. Höhe:** Maximale Höhe des Vorschaufensters (300-1500px)
- **Interne Vorschau verwenden:** Schaltet zwischen interner Vorschau und System-Betrachter für Linksklicks um

---

## �📋 Standard Workflow

### Method 1: Complete New Stack

```

1. Add Folder
   └─ Loads all images from folder
   └─ Auto-loads existing aligned/bunches/final if present

2. Calculate Sharpness

3. Align Images
   └─ Click "Align" button
   └─ Imports → Aligned (saved to aligned/ subfolder)
   └─ Choose to reuse existing aligned images or re-align

4. Stack Aligned
   └─ Click "Stack Aligned" button
   └─ Enters selection mode in Aligned pane
   └─ Click thumbnails to select/deselect images (green = selected)
   └─ Use "Select All" or "Deselect All" for convenience
   └─ Click "Stack" button
   └─ Creates bunches → final result

5. View Result
   └─ Final image appears in Final pane
   └─ Click to preview full size
```

### Method 2: Selective Aligned Stacking

```
1. Add Folder (with existing aligned images)

2. Stack Aligned (Selective)
   └─ Click "Stack Aligned" button
   └─ Selection mode activates
   └─ Select only the aligned images you want to stack
   └─ Click "Stack" to create final image
   └─ Useful for: excluding bad images, creating multiple variants
```

### Method 3: Selective Bunch Stacking

```
1. Add Folder (with existing bunches)

2. Stack Bunches (Selective)
   └─ Click "Stack Bunches" button
   └─ Selection mode activates
   └─ Select which bunches to combine
   └─ Click "Stack" to create final image
   └─ Useful for: combining partial stacks, creating variants
```

### Method 4: Resume Previous Work

```
1. Add Folder (contains aligned/, bunches/, final/)
   └─ All previous results load automatically
   └─ Review existing final images
   └─ Continue stacking with different selections
   └─ Each new stack creates result_NNNN.png
```

---

## 🎯 Selection Mode Features

### Aligned Pane Selection

When you click **"Stack Aligned"**, selection mode activates:

- **Click thumbnails** to toggle selection (green border = selected)
- **Select All** button: selects all aligned images
- **Deselect All** button: clears all selections
- **Cancel** button: exits selection mode without stacking
- **Stack (N selected)** button: stacks the selected images

### Bunches Pane Selection

When you click **"Stack Bunches"**, selection mode activates:

- **Click thumbnails** to toggle selection (green border = selected)
- **Select All** button: selects all bunch images
- **Deselect All** button: clears all selections
- **Cancel** button: exits selection mode without stacking
- **Stack (N selected)** button: stacks the selected bunches

### Visual Feedback

- **Selected images:** Green background with bright green border
- **Unselected images:** Dark gray background with subtle border
- **Normal mode:** Blue-gray thumbnails (no selection active)

---

## ⚙️ Settings Explained

### Blur Threshold (10-100)

Minimum sharpness score to keep an image

- Lower = more permissive (keeps more images)
- Higher = stricter (filters out blurry images)
- **Recommended:** 30 for focus stacks

### Sharpness Grid (4x4 to 16x16)

Divides each image into regions for sharpness analysis

- Higher values = more detailed regional analysis
- An image is kept if ANY region is sharp
- 4x4 = fast, good for general use
- 16x16 = slower, better for finding small sharp details
- **Recommended:** 4-8 for most cases

### Auto-adjust Batch Sizes

Automatically calculates optimal batch sizes based on RAM

- Prevents out-of-memory errors
- **Recommended:** Keep enabled

### Use CLAHE

Contrast Limited Adaptive Histogram Equalization

- Enhances dark images for better feature detection
- Useful for underexposed images
- **Recommended:** Enable for challenging lighting

### Feature Detector

- **ORB (Fast):** Quick alignment, good for most cases
- **SIFT (Best Quality):** Slower but more accurate
- **AKAZE (Balanced):** Good compromise
- **ECC (Precision):** Sub-pixel accuracy for macro focus stacking
- **Recommended:** ORB for speed, SIFT for quality, ECC for macro

### ECC Parameters (when ECC detector selected)

**Motion Type:**

- **Translation (2-DOF):** X/Y shifts only - fastest, use for minimal movement
- **Euclidean (3-DOF):** Shifts + rotation - good for tripod shots with slight rotation
- **Affine (6-DOF):** Shifts + rotation + scaling + shear - handles lens distortion
- **Homography (8-DOF):** Full perspective transform - best for macro (default)

**Max Iterations (3000-30000):**

- Higher = more precise but slower
- Default: 10000 (good for most cases)
- Macro shots: 15000-20000 for ultimate precision
- Test shots: 5000-7000 for faster preview

**Epsilon (1e-8 to 1e-4):**

- Convergence threshold (when to stop iterating)
- Lower = more precise (longer processing)
- Default: 1e-6 (balanced)
- Ultra-precision: 1e-7 or 1e-8
- Quick preview: 1e-5

**Gaussian Filter Size (3x3 to 15x15):**

- Pre-blur kernel to reduce noise before alignment
- Must be odd number (3, 5, 7, 9, 11, 13, 15)
- Default: 7x7 (good for macro)
- Noisy images: 9x9 or 11x11
- Clean images: 5x5 or 3x3

**Parallel Chunk Size (4-24):**

- Number of images to align in parallel
- Higher = faster but more RAM
- Default: 12
- Large images (>30MP): 6-8
- Small images (<10MP): 16-20

**When to use ECC:**

- ✅ Macro focus stacking with static subject
- ✅ Tripod-mounted camera with minimal movement
- ✅ Need sub-pixel (0.01-0.001px) accuracy
- ✅ Images have good correlation (similar brightness/contrast)
- ❌ Handheld shots with large motion (use SIFT instead)
- ❌ Scenes with moving elements (use ORB/SIFT)
- ❌ Very different exposures between frames

---

## 🎨 Advanced Processing

### Noise Reduction (1-10)

Reduces sensor noise in final image

- Higher values = stronger noise reduction
- Can blur fine details if too high
- **Recommended:** 3-5 for high ISO images

### Sharpening (0-5)

Enhances edge definition

- Higher values = more sharpening
- Can create halos if too high
- **Recommended:** 1-2 for subtle enhancement

### Color Correction

- **Contrast (0.5-3.0):** Adjusts tonal range
- **Brightness (-100 to 100):** Lightens or darkens
- **Saturation (0-3.0):** Adjusts color intensity
- **Recommended:** Start with defaults (1.0, 0, 1.0)

---

## 💡 Focus Stacking Tips

### Best Practices:

✓ Use a tripod for stable shots
✓ Keep consistent lighting across all images
✓ Use manual focus, incrementally changing focus distance
✓ Overlap focus zones between images
✓ Shoot in RAW and convert to TIFF for best quality
✓ Use aperture that balances depth and diffraction (f/8-f/11)

### Common Issues:

- **Ghosting:** Reduce to 2-3 overlapping images per zone
- **Halos:** Lower sharpening strength or disable
- **Blurry regions:** Increase sharpness grid size
- **Missing images:** Lower blur threshold
- **Out of memory:** Enable auto-adjust batch sizes

---

## 🎮 GPU Acceleration

### Overview

Imagestacker uses **OpenCL GPU acceleration** for significant performance improvements:

- **Blur detection:** GPU-accelerated sharpness computation using Gaussian blur, Laplacian, and Sobel operators
- **Feature extraction:** GPU preprocessing with color conversion, CLAHE, and image resizing
- **Image warping:** GPU-accelerated affine transformations
- **Focus stacking:** GPU Laplacian pyramid generation and collapse

### Performance

GPU acceleration provides **2-6x speedup** depending on the operation:

- Blur detection: 1-2 seconds per 42MP image
- Warping: ~1.5 seconds per 42MP image
- Overall SIFT alignment: ~2 minutes for 46 images
- Overall ORB alignment: ~1.5 minutes for 46 images

### Memory Management

The application automatically manages memory for large images:

- **Adaptive batch sizing** based on image dimensions
- For 42MP images: processes 2-3 images simultaneously
- Typical RAM usage: 8-10GB for 46×42MP images
- GPU operations are thread-safe with automatic serialization

### Optimization Tips

- **Feature Detector Choice:**

  - **ORB:** 6x faster than SIFT, excellent for most cases
  - **SIFT:** Best quality but slower, optimized to 2000 features
  - **AKAZE:** Good balance but CPU-only feature detection
- **GPU Utilization:**

  - Most visible during warping phase (continuous GPU work)
  - Feature extraction alternates between GPU preprocessing and CPU feature detection
  - This is optimal - CPU and GPU work in parallel
- **System Requirements:**

  - OpenCL-capable GPU (most modern GPUs)
  - 8GB+ RAM recommended for 40MP+ images
  - 16GB+ RAM for best performance with large image sets

---

## � Transparency & Alpha Channel Handling

### How It Works

When stacking images with transparent areas (e.g., aligned PNGs with black/transparent borders), ImageStacker handles the alpha channel separately from the color data:

1. **BGR-only Pyramid**: Only the color channels (BGR) go through the Laplacian pyramid. The alpha channel is never blended through the pyramid, preventing edge artifacts.
2. **Alpha-weighted Energy**: When selecting the "winner" pixel at each pyramid level, transparent pixels are suppressed: `energy × (alpha / 255)`. This prevents transparent or border regions from incorrectly winning over sharp, opaque content.
3. **AND-combined Alpha**: The final alpha channel is the intersection (AND) of all input alpha channels. Only pixels that are opaque in **all** input images remain opaque in the result.
4. **Erosion (5px)**: A 5-pixel morphological erosion is applied to the final alpha mask, removing any residual artifacts at the transition between opaque and transparent regions.

### What This Means for Your Results

- **Clean transparent edges**: No white lines, bright halos, or blending artifacts at transparency borders
- **Slightly more cropping**: Because AND-alpha uses the smallest common opaque area plus erosion, the result may be slightly more cropped than any single input image
- **Perfect for aligned images**: Aligned images typically have small transparent borders from geometric correction — these are handled cleanly

### Tips

- If the final image appears more cropped than expected, this is normal behavior from AND-alpha + erosion
- The trade-off (slightly more cropping vs. artifact-free edges) is intentional
- For images without transparency (e.g., JPEG sources), the alpha handling has no effect — all pixels are treated as fully opaque

---

## �💡 Usage Tips

### Creating Multiple Variants

- Stack different selections of aligned images for comparison
- Each stack creates a new numbered result (result_0001.png, result_0002.png, etc.)
- Experiment with including/excluding edge images

### Iterative Workflow

- Add Folder loads all previous work automatically
- Review existing results before creating new stacks
- Aligned and bunches folders persist between sessions
- No need to re-align when experimenting with different selections

### Quality Control

- Preview images by clicking thumbnails
- Use selection mode to exclude bad images
- Check aligned images for proper registration
- Review bunches to see progressive stacking

---

## 🔬 Technical Details

### Alignment Algorithm:

- Feature detection using ORB/SIFT/AKAZE
- Feature matching with BFMatcher/FLANN
- Rigid affine transformation (rotation + translation + scale)
- RANSAC outlier rejection
- High-quality LANCZOS4 interpolation

### Stacking Algorithm:

- Laplacian pyramid decomposition (7 levels)
- Winner-take-all selection at each pyramid level (no averaging → no ghosting)
- Sharpness energy: Laplacian → AbsDiff → GaussianBlur(3×3)
- Alpha-weighted energy: transparent pixels suppressed during selection
- BGR channels processed through pyramid separately from alpha
- AND-combined alpha channel across all input images
- 5px morphological erosion for clean transparency edges

### Sharpness Detection:

- Regional analysis using configurable grid (4x4 to 16x16)
- GPU-accelerated computation using UMat for OpenCL
- Combines Laplacian variance, Tenengrad, Modified Laplacian
- Image kept if ANY region exceeds threshold
- Perfect for focus stacks with varying sharp regions
- Parallel processing with mutex-serialized GPU operations

---

## 🔧 Troubleshooting

### Images not aligning properly:

- Try different feature detector (SIFT for best quality)
- Enable CLAHE for dark images
- Ensure images have sufficient overlap
- Check for camera movement between shots

### Some images skipped during alignment:

- Lower the blur threshold
- Increase sharpness grid size
- Check if those images are genuinely blurry
- Review console output for sharpness scores

### Ghosting in final image:

- Alignment may be imperfect - try SIFT detector
- Ensure camera was stable during capture
- Re-capture with better camera stability

### Out of memory errors:

- Enable 'Auto-adjust batch sizes'
- Close other applications to free RAM
- Process fewer images at once
- Resize images before stacking if very large

### Application crashes or freezes:

- Check console output for error messages
- Ensure sufficient disk space for output
- Verify image files are not corrupted
- Try with smaller test set first

---

*For more information, see the project documentation or report issues on GitHub.*
