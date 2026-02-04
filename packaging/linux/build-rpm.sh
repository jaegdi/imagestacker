#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

VERSION="1.0.0"
PACKAGE_NAME="imagestacker"

echo "================================================"
echo "Building RPM package for ImageStacker v${VERSION}"
echo "================================================"

RPMBUILD_TEMP_DIR=$(mktemp -d)
# Use local sources dir, no sudo needed
RPMBUILD_DIR="/$RPMBUILD_TEMP_DIR/rpmbuild"
mkdir -p "$RPMBUILD_DIR/{BUILD,RPMS,SOURCES,SPECS,SRPMS}"

# Define the final archive path
TAR_GZ_NAME="${PACKAGE_NAME}-${VERSION}.tar.gz"
TAR_PATH="$RPMBUILD_DIR/SOURCES/$TAR_GZ_NAME"

# Output info
echo "Creating source tarball: $TAR_PATH ...."
cd "$PROJECT_ROOT"

# Step 1: Create a clean copy of the source tree in temp dir
TEMP_DIR=$(mktemp -d)
SOURCE_DIR="$TEMP_DIR/${PACKAGE_NAME}-${VERSION}"
mkdir -p "$SOURCE_DIR"

# Copy all files except known irrelevant ones
rsync -a \
    --exclude='target/' \
    --exclude='.git/' \
    --exclude='testimages/' \
    --exclude='testimages-*/' \
    --exclude='*.rpm' \
    --exclude='*.deb' \
    --exclude='rpmbuild/' \
    --exclude='*.log' \
    --exclude='.DS_Store' \
    --exclude='*~' \
    "$PROJECT_ROOT/" "$SOURCE_DIR/"

# Ensure icon exists
ICON_PATH="$SOURCE_DIR/icons/imagestacker_icon.png"
if [ ! -f "$ICON_PATH" ]; then
    echo "Creating placeholder icon..."
    convert -size 256x256 xc:blue -pointsize 72 -fill white -gravity center \
        -annotate +0+0 "IS" "$ICON_PATH"
fi

# Step 2: Create the tarball **from the correct path**
cd "$TEMP_DIR"
tar czf "$TAR_PATH" "${PACKAGE_NAME}-${VERSION}"

# Step 3: Verify the tarball exists
if [ ! -f "$TAR_PATH" ]; then
    echo "ERROR: Tarball not created! Path: $TAR_PATH"
    exit 1
fi

echo "✅ Source tarball created successfully: $TAR_PATH"

# Step 4: Copy the .spec file to SPECS
SPEC_SOURCE="$SCRIPT_DIR/imagestacker.spec"
SPEC_DEST="$RPMBUILD_DIR/SPECS/${PACKAGE_NAME}.spec"

cp "$SPEC_SOURCE" "$SPEC_DEST"
echo "Copied spec file to: $SPEC_DEST"

# Step 5: Build RPM using rpmbuild (without sudo)
cd "$RPMBUILD_DIR/SPECS"
echo "Building RPM package..."
rpmbuild -v -bb \
    --define "_sourcedir $RPMBUILD_DIR/SOURCES" \
    --define "_builddir $RPMBUILD_DIR/BUILD" \
    --buildroot="/tmp/imagestacker-build-root" \
    "$PACKAGE_NAME.spec"

echo ""
echo "================================================"
echo "Build completed successfully!"
echo "================================================"

ls -lh "$RPMBUILD_DIR/RPMS/"*/*.rpm 2>/dev/null || echo "No RPMs found"
ls -lh "$RPMBUILD_DIR/SRPMS/"*.rpm 2>/dev/null || echo "No SRPMs found"