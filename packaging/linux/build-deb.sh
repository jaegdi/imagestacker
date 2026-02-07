#!/bin/bash
#
# Build a Debian/Ubuntu .deb package for ImageStacker.
#
# Usage:  ./build-deb.sh
#
# The resulting .deb is placed in ../../dist/deb/
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

VERSION="1.0.0"
PACKAGE_NAME="imagestacker"
ARCH="amd64"
DEB_NAME="${PACKAGE_NAME}_${VERSION}-1_${ARCH}"

echo "================================================"
echo "Building DEB package for ImageStacker v${VERSION}"
echo "================================================"

# ── 1. Build release binary (skip if already built) ──────────────────
BINARY="$PROJECT_ROOT/target/release/imagestacker"
if [ ! -x "$BINARY" ]; then
    echo "Building release binary..."
    cd "$PROJECT_ROOT"
    cargo build --release
fi
echo "✅ Binary: $BINARY"

# ── 2. Create staging tree ───────────────────────────────────────────
STAGING=$(mktemp -d)
PKG_ROOT="$STAGING/$DEB_NAME"
trap 'rm -rf "$STAGING"' EXIT

mkdir -p "$PKG_ROOT/DEBIAN"
mkdir -p "$PKG_ROOT/usr/bin"
mkdir -p "$PKG_ROOT/usr/share/applications"
mkdir -p "$PKG_ROOT/usr/share/pixmaps"
mkdir -p "$PKG_ROOT/usr/share/icons/hicolor/128x128/apps"
mkdir -p "$PKG_ROOT/usr/share/icons/hicolor/256x256/apps"
mkdir -p "$PKG_ROOT/usr/share/doc/${PACKAGE_NAME}"

# ── 3. Install files ────────────────────────────────────────────────
install -m 755 "$BINARY" "$PKG_ROOT/usr/bin/imagestacker"

# Desktop entry
cat > "$PKG_ROOT/usr/share/applications/imagestacker.desktop" <<EOF
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
EOF

# Icons – use ImageMagick (convert/magick) if available, otherwise just copy
ICON_SRC="$PROJECT_ROOT/icons/imagestacker_icon.png"
if [ -f "$ICON_SRC" ]; then
    CONVERT_CMD=""
    if command -v magick &>/dev/null; then
        CONVERT_CMD="magick"
    elif command -v convert &>/dev/null; then
        CONVERT_CMD="convert"
    fi

    if [ -n "$CONVERT_CMD" ]; then
        $CONVERT_CMD "$ICON_SRC" -resize 64x64   "$PKG_ROOT/usr/share/pixmaps/imagestacker.png"
        $CONVERT_CMD "$ICON_SRC" -resize 128x128  "$PKG_ROOT/usr/share/icons/hicolor/128x128/apps/imagestacker.png"
        $CONVERT_CMD "$ICON_SRC" -resize 256x256  "$PKG_ROOT/usr/share/icons/hicolor/256x256/apps/imagestacker.png"
    else
        echo "⚠  ImageMagick not found – copying icon as-is"
        cp "$ICON_SRC" "$PKG_ROOT/usr/share/pixmaps/imagestacker.png"
        cp "$ICON_SRC" "$PKG_ROOT/usr/share/icons/hicolor/128x128/apps/imagestacker.png"
        cp "$ICON_SRC" "$PKG_ROOT/usr/share/icons/hicolor/256x256/apps/imagestacker.png"
    fi
else
    echo "⚠  Icon not found at $ICON_SRC – skipping"
fi

# Documentation
[ -f "$PROJECT_ROOT/USER_MANUAL.md" ] && install -m 644 "$PROJECT_ROOT/USER_MANUAL.md" "$PKG_ROOT/usr/share/doc/${PACKAGE_NAME}/"
[ -f "$PROJECT_ROOT/README.md" ]      && install -m 644 "$PROJECT_ROOT/README.md"      "$PKG_ROOT/usr/share/doc/${PACKAGE_NAME}/"

# ── 4. Determine installed size (KiB) ───────────────────────────────
INSTALLED_SIZE=$(du -sk "$PKG_ROOT" | awk '{print $1}')

# ── 5. DEBIAN/control ───────────────────────────────────────────────
cat > "$PKG_ROOT/DEBIAN/control" <<EOF
Package: ${PACKAGE_NAME}
Version: ${VERSION}-1
Section: graphics
Priority: optional
Architecture: ${ARCH}
Depends: libopencv-dev (>= 4.5) | libopencv-core4t64, libgtk-3-0t64 | libgtk-3-0, fontconfig
Installed-Size: ${INSTALLED_SIZE}
Maintainer: ImageStacker Team <imagestacker@example.com>
Homepage: https://github.com/jaegdi/imagestacker
Description: Focus stacking application for combining multiple images
 ImageStacker is a powerful focus stacking application that combines
 multiple images with different focus points into a single sharp image.
 .
 Features include automatic image alignment (ORB, SIFT, AKAZE),
 Laplacian pyramid focus stacking, regional sharpness detection,
 GPU acceleration via OpenCL, batch processing, and a modern GUI.
EOF

# ── 6. Post-install / post-remove scripts ────────────────────────────
cat > "$PKG_ROOT/DEBIAN/postinst" <<'EOF'
#!/bin/sh
set -e
if command -v gtk-update-icon-cache >/dev/null 2>&1; then
    gtk-update-icon-cache -f -t /usr/share/icons/hicolor 2>/dev/null || true
fi
if command -v update-desktop-database >/dev/null 2>&1; then
    update-desktop-database /usr/share/applications 2>/dev/null || true
fi
EOF
chmod 755 "$PKG_ROOT/DEBIAN/postinst"

cat > "$PKG_ROOT/DEBIAN/postrm" <<'EOF'
#!/bin/sh
set -e
if command -v gtk-update-icon-cache >/dev/null 2>&1; then
    gtk-update-icon-cache -f -t /usr/share/icons/hicolor 2>/dev/null || true
fi
if command -v update-desktop-database >/dev/null 2>&1; then
    update-desktop-database /usr/share/applications 2>/dev/null || true
fi
EOF
chmod 755 "$PKG_ROOT/DEBIAN/postrm"

# ── 7. Build the .deb ───────────────────────────────────────────────
echo "Building .deb package..."
dpkg-deb --root-owner-group --build "$PKG_ROOT" "$STAGING/${DEB_NAME}.deb"

# ── 8. Copy to dist/ ────────────────────────────────────────────────
OUT_DIR="$PROJECT_ROOT/dist/deb"
mkdir -p "$OUT_DIR"
cp -v "$STAGING/${DEB_NAME}.deb" "$OUT_DIR/"

echo ""
echo "================================================"
echo "DEB package built successfully!"
echo "================================================"
ls -lh "$OUT_DIR/${DEB_NAME}.deb"
