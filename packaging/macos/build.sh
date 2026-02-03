#!/usr/bin/env bash
# ImageStacker macOS Apple Silicon build script
# Creates a minimal .app bundle and a zipped distribution.
#
# Usage:
#   ./packaging/macos/build.sh
#
# Requirements:
# - Run on macOS 13+ (recommended)
# - Xcode Command Line Tools installed
# - Rust toolchain via rustup
#
# Notes:
# - This builds for the host architecture. On Apple Silicon that's aarch64 (arm64).
# - OpenCV is a native dependency; ensure your OpenCV linkage strategy is set up
#   for macOS (brew opencv or custom build) so the binary can run.
# - Code signing / notarization is NOT done here.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
APP_NAME="ImageStacker"
BUNDLE_ID="com.jaegdi.imagestacker"
VERSION="$(grep -E '^version\s*=\s*"' "$PROJECT_ROOT/Cargo.toml" | head -1 | sed -E 's/.*"([^"]+)".*/\1/')"

DIST_ROOT="$PROJECT_ROOT/dist/macos"
APP_DIR="$DIST_ROOT/${APP_NAME}.app"
CONTENTS_DIR="$APP_DIR/Contents"
MACOS_DIR="$CONTENTS_DIR/MacOS"
RESOURCES_DIR="$CONTENTS_DIR/Resources"

echo "Building ImageStacker (macOS release)..."
cd "$PROJECT_ROOT"
cargo build --release

rm -rf "$DIST_ROOT"
mkdir -p "$MACOS_DIR" "$RESOURCES_DIR"

# Copy binary
cp "$PROJECT_ROOT/target/release/imagestacker" "$MACOS_DIR/imagestacker"
chmod +x "$MACOS_DIR/imagestacker"

# Copy docs
if [[ -f "$PROJECT_ROOT/USER_MANUAL.md" ]]; then
  cp "$PROJECT_ROOT/USER_MANUAL.md" "$RESOURCES_DIR/USER_MANUAL.md"
fi
if [[ -f "$PROJECT_ROOT/README.md" ]]; then
  cp "$PROJECT_ROOT/README.md" "$RESOURCES_DIR/README.md"
fi

# Copy icon (PNG). Converting to .icns is optional; we keep PNG for now.
if [[ -f "$PROJECT_ROOT/icons/imagestacker_icon.png" ]]; then
  cp "$PROJECT_ROOT/icons/imagestacker_icon.png" "$RESOURCES_DIR/imagestacker_icon.png"
fi

# Info.plist
cat > "$CONTENTS_DIR/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key>
  <string>${APP_NAME}</string>
  <key>CFBundleDisplayName</key>
  <string>${APP_NAME}</string>
  <key>CFBundleIdentifier</key>
  <string>${BUNDLE_ID}</string>
  <key>CFBundleVersion</key>
  <string>${VERSION}</string>
  <key>CFBundleShortVersionString</key>
  <string>${VERSION}</string>
  <key>CFBundleExecutable</key>
  <string>imagestacker</string>
  <key>CFBundlePackageType</key>
  <string>APPL</string>
  <key>LSMinimumSystemVersion</key>
  <string>12.0</string>
</dict>
</plist>
EOF

# Zip it
ZIP_PATH="$PROJECT_ROOT/dist/imagestacker-macos-arm64.zip"
rm -f "$ZIP_PATH"
cd "$PROJECT_ROOT/dist"
/usr/bin/zip -qry "${ZIP_PATH}" "macos/${APP_NAME}.app"

echo "Created: $ZIP_PATH"
