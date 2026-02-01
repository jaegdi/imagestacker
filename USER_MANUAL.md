# 📖 Rust Image Stacker - User Manual

**Version 0.1.0 - Focus Stacking Application**

---

## 🚀 Getting Started

### Quick Start: Add Folder
The easiest way to start is using **Add Folder**:
1. Click 'Add Folder' button
2. Select a folder containing your focus stack images
3. **Automatic Loading:**
   - All images in the folder → **Imported** pane
   - Existing `aligned/` subfolder images → **Aligned** pane
   - Existing `bunches/` subfolder images → **Bunches** pane
   - Existing `final/` subfolder images → **Final** pane

This allows you to resume work on a previous project or review existing results!

### Command Line Usage
You can also start the application with a folder pre-loaded:

```bash
# Linux/macOS
./imagestacker --import /path/to/your/images

# Windows
imagestacker.exe --import C:\path\to\your\images

# Short form
./imagestacker -i /path/to/images
```

The application will:
- Open the GUI window
- Automatically load the specified folder
- Scan for images and existing results (aligned/, bunches/, final/)
- Ready to start processing immediately

**Tip:** Relative paths are automatically converted to absolute paths.

### Alternative: Add Individual Images
- Click 'Add Images' to select specific image files
- Supported formats: JPG, PNG, TIFF
- Images appear in the 'Imported' pane

---

## 📂 Understanding the Panes

### Imported (Dark Gray Border)
- Original images loaded from your folder
- Click any thumbnail to preview in full size
- These are your source images for alignment

### Aligned (Dark Gray Border)
- Images after alignment process
- Geometrically corrected to match each other
- Saved to: `<source_folder>/aligned/`
- **Selective Stacking:** Choose which aligned images to stack

### Bunches (Dark Gray Border)
- Intermediate images during stacking process
- Shows progressive stacking of image groups
- Saved to: `<source_folder>/bunches/`
- **Selective Stacking:** Choose which bunches to stack into final

### Final (Dark Gray Border)
- Completed focus-stacked images
- Contains sharp details from all source images
- Saved to: `<source_folder>/final/result_0001.png`, `result_0002.png`, etc.
- Each stacking operation creates a new numbered result

---

## �️ Image Preview & Editing

### Thumbnail Interactions
Each thumbnail in all panes (Imported, Aligned, Bunches, Final) supports:

- **Left-click:**
  - Opens image in internal preview (if enabled in Settings)
  - Or opens in configured external viewer (if internal preview disabled)
  
- **Right-click:**
  - Opens image in your configured external editor
  - Perfect for quick edits in GIMP, Darktable, Photoshop, etc.
  - Falls back to system default if no editor configured

### Image Preview Navigation
When viewing an image in the internal preview, you can navigate through all images in the current pane:

- **Arrow Keys:**
  - **LEFT Arrow:** Show previous image in pane
  - **RIGHT Arrow:** Show next image in pane
  - **ESC:** Close preview
  
- **Mouse Wheel:**
  - **Scroll Up:** Show previous image
  - **Scroll Down:** Show next image
  
- **Navigation Buttons:**
  - **< Previous:** Go to previous image
  - **Next >:** Go to next image
  
- **Info Display:**
  - Shows current position: "Image X of Y"
  - Reminds you of navigation shortcuts

**Tip:** Navigation wraps around - from the last image, "Next" goes back to the first image, and vice versa.

### Configuring External Applications
Open **Settings** → **Preview & UI** section:

**External Image Viewer (for left-click):**
- Used when "Use Internal Preview" is disabled
- Enter the full path to your preferred image viewer:
  - **Linux:** `/usr/bin/eog` (Eye of GNOME), `/usr/bin/geeqie`, `/usr/bin/gwenview`
  - **macOS:** `/Applications/Preview.app/Contents/MacOS/Preview`
  - **Windows:** `C:\Program Files\IrfanView\i_view64.exe`
- Leave empty to use system default viewer

**External Image Editor (for right-click):**
- Used for editing images when right-clicking thumbnails
- Enter the full path to your image editor:
  - **Linux:** `/usr/bin/gimp`, `/usr/bin/darktable`, `/usr/bin/krita`
  - **macOS:** `/Applications/GIMP.app/Contents/MacOS/gimp`
  - **Windows:** `C:\Program Files\GIMP\bin\gimp.exe`, `C:\Program Files\Adobe\Photoshop\Photoshop.exe`
- Leave empty to use system default viewer

**Tip:** Use absolute paths for best compatibility. Settings are saved automatically.

### Internal Preview Settings
- **Preview Max Width:** Maximum width for preview window (400-2000px)
- **Preview Max Height:** Maximum height for preview window (300-1500px)
- **Use Internal Preview:** Toggle internal preview vs system viewer for left-click

---

## �📋 Standard Workflow

### Method 1: Complete New Stack
```
1. Add Folder
   └─ Loads all images from folder
   └─ Auto-loads existing aligned/bunches/final if present

2. Align Images
   └─ Click "Align" button
   └─ Imports → Aligned (saved to aligned/ subfolder)
   └─ Choose to reuse existing aligned images or re-align

3. Stack Aligned
   └─ Click "Stack Aligned" button
   └─ Enters selection mode in Aligned pane
   └─ Click thumbnails to select/deselect images (green = selected)
   └─ Use "Select All" or "Deselect All" for convenience
   └─ Click "Stack" button
   └─ Creates bunches → final result

4. View Result
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
- **Recommended:** ORB for speed, SIFT for quality

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

## 💡 Usage Tips

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
- Winner-take-all selection at each pyramid level
- Sharpness measured using Laplacian variance
- Energy-based masking for region selection
- Gaussian blur for smooth energy maps

### Sharpness Detection:
- Regional analysis using configurable grid
- Combines Laplacian variance, Tenengrad, Modified Laplacian
- Image kept if ANY region exceeds threshold
- Perfect for focus stacks with varying sharp regions

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
