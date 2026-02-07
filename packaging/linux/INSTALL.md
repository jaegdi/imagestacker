# ImageStacker RPM Package for SUSE Tumbleweed

## Quick Start

The fastest way to build and install:

```bash
cd packaging/linux
./quick-install.sh
```

This will install all dependencies, build the RPM, and install ImageStacker.

## Manual Build

### Option 1: Using the build script

```bash
cd packaging/linux
./build-rpm.sh
```

### Option 2: Using make

```bash
cd packaging/linux
make rpm          # Build RPM
make install      # Build and install
make clean        # Clean artifacts
```

## What You Get

After installation, ImageStacker provides:

### Application
- **Binary:** `/usr/bin/imagestacker`
- Launch from terminal or application menu

### Desktop Integration
- Application menu entry in Graphics/Photography categories
- Icon and desktop file for GUI launchers
- Right-click context menu support for images (configurable)

### Documentation
- User manual: `/usr/share/doc/imagestacker/USER_MANUAL.md`
- View with: `cat /usr/share/doc/imagestacker/USER_MANUAL.md`

## Features

- **Focus Stacking:** Combine images with different focus points
- **Alignment:** Automatic alignment using ORB, SIFT, AKAZE, or ECC (sub-pixel precision)
- **GPU Acceleration:** OpenCL-based processing for 2-6× speedup
- **Alpha Handling:** Artifact-free transparent edges with AND-alpha + erosion
- **GUI:** Modern dark-themed interface with preview and editing capabilities
- **Selection Modes:** Choose specific images to stack
- **External Tools:** Configure GIMP, Darktable, or other editors
- **CLI Support:** Automate with `--import` parameter
- **Post-Processing:** Noise reduction, sharpening, color correction
- **Sharpness Caching:** YAML-based per-image sharpness caching

## System Requirements

### Runtime
- SUSE Tumbleweed (or compatible openSUSE)
- OpenCV >= 4.5
- GTK3

### Build
- Rust >= 1.70
- cargo
- gcc/g++
- cmake
- pkg-config
- opencv-devel
- gtk3-devel
- ImageMagick (for icon resizing)

## Usage Examples

### Launch GUI
```bash
imagestacker
```

### Auto-import folder
```bash
imagestacker --import ~/Pictures/focus-stack
```

### Configure external editor
1. Open ImageStacker
2. Click Settings button
3. Navigate to "Preview & UI" section
4. Enter path to GIMP: `/usr/bin/gimp`
5. Enter path to viewer: `/usr/bin/eog`

## Package Management

### Check if installed
```bash
rpm -q imagestacker
```

### List package files
```bash
rpm -ql imagestacker
```

### Get package info
```bash
rpm -qi imagestacker
```

### Uninstall
```bash
sudo zypper remove imagestacker
```

## Troubleshooting

### Build fails with "opencv not found"
```bash
sudo zypper install opencv-devel
pkg-config --modversion opencv4
```

### Application won't start
Check dependencies:
```bash
ldd /usr/bin/imagestacker
```

### Missing menu entry
Update desktop database:
```bash
sudo update-desktop-database
```

## Files Created

The package creates these files on your system:

```
/usr/bin/imagestacker                              # Main executable
/usr/share/applications/imagestacker.desktop      # Desktop entry
/usr/share/pixmaps/imagestacker.png              # Application icon
/usr/share/doc/imagestacker/USER_MANUAL.md       # Documentation
```

## Development

### Rebuild after code changes
```bash
cd packaging/linux
make clean
make rpm
```

### Test without installing
```bash
cargo run --release
```

### Create custom RPM
Edit `imagestacker.spec` and rebuild:
```bash
cd ~/rpmbuild/SPECS
rpmbuild -ba imagestacker.spec
```

## Support

- Report issues on GitHub
- Check USER_MANUAL.md for usage help
- Consult SUSE Tumbleweed docs for system issues

---

**Version:** 0.1.0  
**Platform:** SUSE Tumbleweed (openSUSE)  
**Package Format:** RPM  
**License:** MIT (check LICENSE file)
