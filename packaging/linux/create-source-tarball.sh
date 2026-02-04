#!/bin/bash

set -euo pipefail

echo "Creating source tarball for ImageStacker..."
SOURCE_DIR="./"
TAR_NAME="imagestacker-1.0.0.tar.gz"

# Remove old tarball if exists
rm -f "$TAR_NAME"

# Create the source archive (excluding .git, build artifacts)
tar --exclude='*.git' \
    --exclude='target/' \
    --exclude='build/' \
    --exclude='*.log' \
    --exclude='.gitignore' \
    -czf "$TAR_NAME" \
    -C "$(dirname "$SOURCE_DIR")" "$(basename "$SOURCE_DIR")"

echo "Source tarball created: $TAR_NAME"
ls -l "$TAR_NAME"
```