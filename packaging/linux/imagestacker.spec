Name:           imagestacker
Version:        1.0.0
Release:        1%{?dist}
Summary:        Focus stacking application for combining multiple images

License:        MIT
URL:            https://github.com/jaegdi/imagestacker
Source0:        %{name}-%{version}.tar.gz

# Note: rust and cargo are expected to be installed via rustup
# BuildRequires:  cargo
# BuildRequires:  rust >= 1.70
BuildRequires:  gcc
BuildRequires:  gcc-c++
BuildRequires:  cmake
BuildRequires:  pkgconfig
BuildRequires:  pkgconfig(opencv4)
BuildRequires:  gtk3-devel
BuildRequires:  ImageMagick

Requires:       opencv >= 4.5
Requires:       gtk3

%description
ImageStacker is a powerful focus stacking application that combines multiple
images with different focus points into a single sharp image. Features include:
- Automatic image alignment using feature detection (ORB, SIFT, AKAZE)
- Advanced focus stacking with Laplacian pyramid decomposition
- Regional sharpness detection with configurable grid
- Batch processing with adaptive memory management
- Post-processing: noise reduction, sharpening, color correction
- GUI with internal preview and external editor integration
- Command-line interface for automation

%prep
%setup -q

%build
# Build release version
cargo build --release

%install
# Create directories
install -d %{buildroot}%{_bindir}
install -d %{buildroot}%{_datadir}/applications
install -d %{buildroot}%{_datadir}/pixmaps
install -d %{buildroot}%{_datadir}/doc/%{name}

# Install binary
install -m 755 target/release/imagestacker %{buildroot}%{_bindir}/imagestacker

# Install desktop file
cat > %{buildroot}%{_datadir}/applications/imagestacker.desktop <<EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=ImageStacker
GenericName=Focus Stacking Tool
Comment=Combine multiple images with different focus points
Exec=/usr/bin/imagestacker %U
Icon=imagestacker
Categories=Graphics;Photography;2DGraphics;
Terminal=false
StartupNotify=true
Keywords=focus;stacking;photography;macro;image;HDR;
MimeType=image/png;image/jpeg;image/tiff;image/x-portable-pixmap;
X-KDE-SubstituteUID=false
X-KDE-StartupNotify=true
EOF

# Install icon from icons/imagestacker_icon.png
# Resize to standard icon sizes for better desktop integration
convert icons/imagestacker_icon.png -resize 64x64 %{buildroot}%{_datadir}/pixmaps/imagestacker.png
# Also create higher resolution icons for HiDPI displays
install -d %{buildroot}%{_datadir}/icons/hicolor/128x128/apps
install -d %{buildroot}%{_datadir}/icons/hicolor/256x256/apps
convert icons/imagestacker_icon.png -resize 128x128 %{buildroot}%{_datadir}/icons/hicolor/128x128/apps/imagestacker.png
convert icons/imagestacker_icon.png -resize 256x256 %{buildroot}%{_datadir}/icons/hicolor/256x256/apps/imagestacker.png

# Install documentation
install -m 644 USER_MANUAL.md %{buildroot}%{_datadir}/doc/%{name}/USER_MANUAL.md
if [ -f README.md ]; then
    install -m 644 README.md %{buildroot}%{_datadir}/doc/%{name}/README.md
fi

%files
%{_bindir}/imagestacker
%{_datadir}/applications/imagestacker.desktop
%{_datadir}/pixmaps/imagestacker.png
%{_datadir}/icons/hicolor/128x128/apps/imagestacker.png
%{_datadir}/icons/hicolor/256x256/apps/imagestacker.png
%{_datadir}/doc/%{name}/USER_MANUAL.md
%{_datadir}/doc/%{name}/README.md

%post
# Update icon cache after installation
if [ -x /usr/bin/gtk-update-icon-cache ]; then
    /usr/bin/gtk-update-icon-cache -f -t %{_datadir}/icons/hicolor 2>/dev/null || :
fi
if [ -x /usr/bin/update-desktop-database ]; then
    /usr/bin/update-desktop-database %{_datadir}/applications 2>/dev/null || :
fi

%postun
# Update icon cache after uninstallation
if [ -x /usr/bin/gtk-update-icon-cache ]; then
    /usr/bin/gtk-update-icon-cache -f -t %{_datadir}/icons/hicolor 2>/dev/null || :
fi
if [ -x /usr/bin/update-desktop-database ]; then
    /usr/bin/update-desktop-database %{_datadir}/applications 2>/dev/null || :
fi

%changelog
* Sat Feb 01 2025 ImageStacker Team <imagestacker@example.com> - 1.0.0-1
- Version 1.0.0 stable release
- Full GPU acceleration with OpenCL for image processing
- GPU-accelerated blur detection and sharpness analysis
- GPU-accelerated image preprocessing (CLAHE, resize, color conversion)
- GPU-accelerated warping and Laplacian pyramid stacking
- Smart GPU/CPU fallback with automatic detection
- Adaptive batch processing for memory management (42MP images)
- Optimized SIFT feature detection (2000 features)
- Thread-safe OpenCL operations with global mutex
- Regional sharpness detection with configurable grid
- Focus stacking with multiple alignment methods (ORB, SIFT, AKAZE)
- GUI with internal preview and external editor support
- Advanced post-processing options
- Comprehensive user manual and documentation

* Thu Jan 30 2025 ImageStacker Team <imagestacker@example.com> - 0.1.0-1
- Initial RPM release
- Focus stacking with multiple alignment methods
- GUI with internal preview and external editor support
- Configurable external viewer and editor
- Command-line interface with --import parameter
- Selection modes for aligned images and bunches
- Advanced post-processing options
