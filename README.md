# ImageStacker

A powerful focus stacking application for combining multiple images with different focus points into a single sharp image.

## Features

- **Automatic Image Alignment**: Uses advanced feature detection (ORB, SIFT, AKAZE) to align images
- **Focus Stacking**: Laplacian pyramid decomposition for optimal sharpness selection
- **Regional Sharpness Detection**: Configurable grid-based analysis (4x4 to 16x16)
- **Batch Processing**: Adaptive memory management for large image sets
- **Post-Processing**: Noise reduction, sharpening, color correction
- **Modern GUI**: Internal preview, external editor integration, selection modes
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

1. **Add Folder** - Import images from a directory
2. **Align Images** - Automatically align all images
3. **Stack Aligned** - Select and stack aligned images
4. **View Result** - Preview in Final pane

## Configuration

### External Applications

Configure in Settings → Preview & UI:

- **External Viewer**: Left-click opens images (e.g., eog, geeqie)
- **External Editor**: Right-click opens for editing (e.g., GIMP, Darktable)

### Processing Options

- **Sharpness Threshold**: 10-100 (default: 30)
- **Sharpness Grid**: 4x4 to 16x16 (default: 4x4)
- **Feature Detector**: ORB (fast), SIFT (quality), AKAZE (balanced)
- **CLAHE**: Enhances dark images for better alignment

## System Requirements

- OpenCV >= 4.5
- GTK3
- Rust >= 1.70 (for building from source)

## Documentation

For detailed usage instructions, see:
- `/usr/share/doc/imagestacker/USER_MANUAL.md` (after installation)
- Or view `USER_MANUAL.md` in the source repository

## Installation

### From RPM (SUSE/openSUSE)
```bash
sudo zypper install imagestacker-0.1.0-1.rpm
```

### From Source
```bash
cargo build --release
./target/release/imagestacker
```

## Building RPM Package

```bash
cd packaging/linux
./build-rpm.sh
```

## License

See LICENSE file for details.

## Project

- GitHub: https://github.com/jaegdi/imagestacker
- Version: 0.1.0

## Credits

Built with:
- Rust programming language
- OpenCV for image processing
- Iced GUI framework
