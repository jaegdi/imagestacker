#!/bin/bash
# Build RPM package for ImageStacker on SUSE Tumbleweed
# Usage: ./build-rpm.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VERSION="1.0.0"
PACKAGE_NAME="imagestacker"

echo "================================================"
echo "Building RPM package for ImageStacker v${VERSION}"
echo "================================================"
echo ""

# Check if we're on SUSE
if ! command -v zypper &> /dev/null; then
    echo "Warning: zypper not found. This script is designed for SUSE systems."
    echo "Continuing anyway..."
fi

# Install build dependencies
echo "Installing build dependencies..."
echo "You may need to enter your password for sudo:"

# Check if rustup is installed
if command -v rustup &> /dev/null; then
    echo "rustup detected - skipping system cargo/rust installation"
    RUST_PACKAGES=""
else
    echo "rustup not found - will install system cargo/rust"
    RUST_PACKAGES="cargo rust"
fi

sudo zypper install -y \
    rpm-build \
    rpmdevtools \
    $RUST_PACKAGES \
    gcc \
    gcc-c++ \
    cmake \
    pkg-config \
    opencv-devel \
    gtk3-devel \
    ImageMagick || {
    echo "Warning: Some dependencies might not be installed."
    echo "Please install them manually if build fails."
}

# Verify rust/cargo are available
if ! command -v cargo &> /dev/null; then
    echo "ERROR: cargo not found!"
    echo "Please install Rust via:"
    echo "  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
    exit 1
fi

echo "Using Rust: $(rustc --version)"
echo "Using Cargo: $(cargo --version)"

echo ""
echo "Setting up RPM build environment..."

# Create RPM build directory structure
RPMBUILD_DIR="$HOME/rpmbuild"
mkdir -p "$RPMBUILD_DIR"/{BUILD,RPMS,SOURCES,SPECS,SRPMS}

# Clean previous builds
rm -f "$RPMBUILD_DIR/SOURCES/${PACKAGE_NAME}-${VERSION}.tar.gz"
rm -f "$RPMBUILD_DIR/SPECS/imagestacker.spec"

echo "Creating source tarball..."
cd "$PROJECT_ROOT"

# Create a clean source tarball
TEMP_DIR=$(mktemp -d)
SOURCE_DIR="$TEMP_DIR/${PACKAGE_NAME}-${VERSION}"
mkdir -p "$SOURCE_DIR"

# Copy source files
rsync -a \
    --exclude 'target/' \
    --exclude '.git/' \
    --exclude 'testimages/' \
    --exclude 'testimages-*/' \
    --exclude '*.rpm' \
    --exclude '*.deb' \
    --exclude 'rpmbuild/' \
    "$PROJECT_ROOT/" "$SOURCE_DIR/"

# Ensure the icon file is included
mkdir -p "$SOURCE_DIR/icons"
cp "$PROJECT_ROOT/icons/imagestacker_icon.png" "$SOURCE_DIR/icons/" 2>/dev/null || {
    echo "Warning: Icon file not found at icons/imagestacker_icon.png"
    echo "Creating a placeholder icon..."
    # Create a simple placeholder PNG using ImageMagick
    convert -size 256x256 xc:blue -pointsize 72 -fill white -gravity center \
        -annotate +0+0 "IS" "$SOURCE_DIR/icons/imagestacker_icon.png"
}

# Create tarball
cd "$TEMP_DIR"
tar czf "${PACKAGE_NAME}-${VERSION}.tar.gz" "${PACKAGE_NAME}-${VERSION}"
mv "${PACKAGE_NAME}-${VERSION}.tar.gz" "$RPMBUILD_DIR/SOURCES/"

# Cleanup temp dir
rm -rf "$TEMP_DIR"

# Copy spec file
cp "$SCRIPT_DIR/imagestacker.spec" "$RPMBUILD_DIR/SPECS/"

echo ""
echo "Building RPM package..."
cd "$RPMBUILD_DIR/SPECS"
rpmbuild -ba imagestacker.spec

echo ""
echo "================================================"
echo "Build completed successfully!"
echo "================================================"
echo ""
echo "RPM packages created:"
ls -lh "$RPMBUILD_DIR/RPMS/"*/*.rpm 2>/dev/null || true
ls -lh "$RPMBUILD_DIR/SRPMS/"*.rpm 2>/dev/null || true

echo ""
echo "To install the package, run:"
echo "  sudo zypper install $RPMBUILD_DIR/RPMS/x86_64/${PACKAGE_NAME}-${VERSION}-1.*.rpm"
echo ""
echo "Or to install with automatic dependency resolution:"
echo "  sudo rpm -ivh $RPMBUILD_DIR/RPMS/x86_64/${PACKAGE_NAME}-${VERSION}-1.*.rpm"
echo ""
