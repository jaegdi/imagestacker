# ImageStacker

A powerful GPU-accelerated focus stacking application for combining multiple images with different focus points into a single sharp image. Built with Rust, OpenCV (OpenCL), and the Iced GUI framework.

**Version 1.0.0**

## Features

- **Automatic Image Alignment**: Feature detection (ORB, SIFT, AKAZE) and ECC sub-pixel precision alignment
- **Focus Stacking**: 7-level Laplacian pyramid with winner-take-all sharpness selection (ghosting-free)
- **Alpha Channel Handling**: Transparent PNG support with AND-combined alpha and erosion for artifact-free edges
- **GPU Acceleration**: OpenCL-based processing via OpenCV UMat (2-6× speedup)
- **Regional Sharpness Detection**: Grid-based analysis for intelligent blur filtering
- **Batch Processing**: Adaptive memory management for large image sets (42MP+)
- **Post-Processing**: Noise reduction, sharpening, color correction
- **Modern GUI**: Dark theme, internal preview, external editor integration, selection modes
- **Sharpness Analysis**: Per-image sharpness caching with YAML persistence
- **Command-Line Interface**: Automation support with `--import` parameter

## Quick Start

### GUI Mode
```bash
imagestacker
```

### Auto-Import Folder
```bash
imagestacker --import /path/to/images
```

## Workflow

1. **Add Folder** – Import images from a directory
2. **Align Images** – Automatically align all images (ORB/SIFT/AKAZE/ECC)
3. **Stack Aligned** – Select and stack aligned images
4. **View Result** – Preview in Final pane

## Stacking Algorithm

ImageStacker uses a **7-level Laplacian pyramid** with **winner-take-all** pixel selection:

1. Each image is decomposed into a Laplacian pyramid (7 levels)
2. Sharpness energy is computed per pixel using Laplacian → AbsDiff → GaussianBlur
3. For images with alpha channels, energy is weighted: `energy × (alpha / 255)`
4. At each pyramid level, the pixel with the highest energy wins (no averaging → no ghosting)
5. Alpha channels are AND-combined across all images (smallest common opaque area)
6. A 5px morphological erosion removes potential edge artifacts at transparency borders

The algorithm processes BGR channels through the pyramid independently from alpha, ensuring clean transparent edges in the final output.

## Configuration

### Alignment Methods

| Detector | Speed | Quality | Best For |
|----------|-------|---------|----------|
| **ORB** | ⚡ Fast | Good | General use, handheld shots |
| **SIFT** | 🐌 Slow | Best | Maximum alignment quality |
| **AKAZE** | ⚖️ Medium | Good | Balanced compromise |
| **ECC** | 🔬 Variable | Sub-pixel | Macro/tripod, static subjects |

### Processing Options

- **Sharpness Threshold**: 10–100 (default: 30)
- **Sharpness Grid**: 4×4 to 16×16 (default: 4×4)
- **CLAHE**: Enhances dark images for better alignment
- **Post-Processing**: Noise reduction, sharpening, contrast/brightness/saturation

### External Applications

Configure in Settings → Preview & UI:

- **External Viewer**: Left-click opens images (e.g., eog, geeqie)
- **External Editor**: Right-click opens for editing (e.g., GIMP, Darktable)

## System Requirements

- **OpenCV** ≥ 4.5 (4.12+ recommended for full OpenCL support)
- **OpenCL**-capable GPU (optional but recommended for 2-6× speedup)
- **GTK3**
- **8 GB+ RAM** (16 GB recommended for 40MP+ images)
- **Rust** ≥ 1.70 (for building from source)

## Documentation

For detailed usage instructions, see:
- `/usr/share/doc/imagestacker/USER_MANUAL.md` (after installation)
- Or view `USER_MANUAL.md` in the source repository
- `PROJECT_STATUS.md` for technical architecture details

## Installation

### From RPM (SUSE/openSUSE)
```bash
sudo zypper install imagestacker-1.0.0-1.rpm
```

### From Source
```bash
cargo build --release
./target/release/imagestacker
```

## Building Packages

### Linux RPM (SUSE Tumbleweed)
```bash
cd packaging/linux
./quick-install.sh     # One-click build & install
# or
./build-rpm.sh         # Build RPM only
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

## License

See LICENSE file for details.

## Project

- GitHub: https://github.com/jaegdi/imagestacker
- Version: 1.0.0

## Built With

- **Rust** – Systems programming language
- **OpenCV 0.94** (opencv-rust) – Image processing with OpenCL GPU acceleration
- **Iced 0.13** – Cross-platform GUI framework
- **Rayon** – Data parallelism for batch processing
- **Tokio** – Async runtime for I/O operations
