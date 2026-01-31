#!/bin/bash
# Quick install script for ImageStacker on SUSE Tumbleweed
# This script downloads dependencies, builds, and installs ImageStacker

set -e

echo "================================================"
echo "ImageStacker Quick Install for SUSE Tumbleweed"
echo "================================================"
echo ""

# Check for root/sudo
if [ "$EUID" -ne 0 ]; then 
    echo "This script needs sudo privileges to install dependencies."
    echo "You will be prompted for your password."
    echo ""
fi

# Install dependencies
echo "Step 1: Installing build dependencies..."
sudo zypper install -y \
    rpm-build \
    rpmdevtools \
    cargo \
    rust \
    gcc \
    gcc-c++ \
    cmake \
    pkg-config \
    opencv-devel \
    gtk3-devel \
    ImageMagick

echo ""
echo "Step 2: Building RPM package..."
cd "$(dirname "$0")"
./build-rpm.sh

echo ""
echo "Step 3: Installing ImageStacker..."
RPM_FILE=$(ls -t ~/rpmbuild/RPMS/x86_64/imagestacker-*.rpm 2>/dev/null | head -1)

if [ -z "$RPM_FILE" ]; then
    echo "Error: RPM package not found!"
    exit 1
fi

sudo zypper install -y "$RPM_FILE"

echo ""
echo "================================================"
echo "Installation Complete!"
echo "================================================"
echo ""
echo "ImageStacker has been installed successfully."
echo ""
echo "To launch:"
echo "  1. Find 'ImageStacker' in your application menu (Graphics/Photography)"
echo "  2. Or run from terminal: imagestacker"
echo "  3. With auto-import: imagestacker --import /path/to/images"
echo ""
echo "Documentation: /usr/share/doc/imagestacker/USER_MANUAL.md"
echo ""
