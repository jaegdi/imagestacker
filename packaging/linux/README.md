# ImageStacker Linux Packaging

This directory contains packaging files for creating Linux distribution packages.

## Building RPM Package for SUSE Tumbleweed

### Prerequisites

Make sure you have the following installed:

```bash
sudo zypper install rpm-build rpmdevtools cargo rust gcc gcc-c++ cmake pkg-config opencv-devel gtk3-devel ImageMagick
```

### Build Steps

1. Navigate to the packaging directory:
   ```bash
   cd packaging/linux
   ```

2. Run the build script:
   ```bash
   ./build-rpm.sh
   ```

3. The script will:
   - Install required build dependencies
   - Set up RPM build environment (`~/rpmbuild`)
   - Create a source tarball
   - Build the RPM package

### Installation

After successful build, install the package:

```bash
# Install with zypper (recommended - handles dependencies)
sudo zypper install ~/rpmbuild/RPMS/x86_64/imagestacker-1.0.0-1.*.rpm

# Or install with rpm directly
sudo rpm -ivh ~/rpmbuild/RPMS/x86_64/imagestacker-1.0.0-1.*.rpm
```

### What Gets Installed

- **Binary:** `/usr/bin/imagestacker`
- **Desktop File:** `/usr/share/applications/imagestacker.desktop`
- **Icon:** `/usr/share/pixmaps/imagestacker.png`
- **Documentation:** `/usr/share/doc/imagestacker/USER_MANUAL.md`

### Desktop Integration

After installation, ImageStacker will appear in your application menu under:
- Graphics → ImageStacker
- Photography → ImageStacker

You can also launch it from the terminal:
```bash
imagestacker
```

Or with auto-import:
```bash
imagestacker --import /path/to/images
```

### Uninstallation

To remove the package:

```bash
sudo zypper remove imagestacker
# or
sudo rpm -e imagestacker
```

## Package Contents

### Files Included

- `imagestacker.spec` - RPM spec file with build instructions and metadata
- `build-rpm.sh` - Automated build script for SUSE Tumbleweed
- `README.md` - This file

### Dependencies

**Build Dependencies:**
- cargo, rust >= 1.70
- gcc, gcc-c++, cmake, pkg-config
- opencv-devel >= 4.5
- gtk3-devel
- ImageMagick (for icon generation)

**Runtime Dependencies:**
- opencv >= 4.5
- gtk3

## Customization

### Changing Version

Edit the version in two places:
1. `imagestacker.spec`: Change `Version: 1.0.0`
2. `build-rpm.sh`: Change `VERSION="1.0.0"`

### Adding Icon

The package uses `icons/imagestacker_icon.png` as the application icon.
During the build process, this image is automatically resized to multiple icon sizes:
- 64x64 for standard displays (`/usr/share/pixmaps/imagestacker.png`)
- 128x128 for HiDPI displays
- 256x256 for high-resolution displays

To use a different icon, replace `icons/imagestacker_icon.png` in your
source tree before building, or modify the spec file icon paths.

### Package Metadata

Edit `imagestacker.spec` to customize:
- Package description
- License information
- URL and source locations
- Dependencies
- Changelog entries

## Troubleshooting

### Build Fails with Missing Dependencies

Install missing packages manually:
```bash
sudo zypper search <package-name>
sudo zypper install <package-name>
```

### OpenCV Not Found

Ensure opencv-devel is installed:
```bash
sudo zypper install opencv-devel
pkg-config --modversion opencv4
```

### Rust Version Too Old

Update Rust:
```bash
rustup update stable
```

Or install newer Rust from zypper:
```bash
sudo zypper install rust cargo
```

### Permission Denied

Make sure the build script is executable:
```bash
chmod +x build-rpm.sh
```

## Advanced Usage

### Building Source RPM Only

```bash
cd ~/rpmbuild/SPECS
rpmbuild -bs imagestacker.spec
```

### Building from Source RPM

```bash
rpmbuild --rebuild ~/rpmbuild/SRPMS/imagestacker-0.1.0-1.src.rpm
```

### Installing in Custom Location

Modify the `%{_prefix}` macro in the spec file or use:
```bash
rpm --prefix=/opt/imagestacker -ivh imagestacker-1.0.0-1.rpm
```

## Support

For issues with:
- **Package building:** Check this README and the build script
- **Application bugs:** Report on GitHub issues
- **SUSE-specific problems:** Consult SUSE Tumbleweed documentation

## License

See the main LICENSE file in the project root.
