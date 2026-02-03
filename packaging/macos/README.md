# macOS builds (Apple Silicon / arm64)

This directory contains helper scripts to produce a **portable** macOS `.app` bundle on Apple Silicon.

## What you get

- `dist/macos/ImageStacker.app`
- `dist/imagestacker-macos-arm64.zip`

The `.app` is a minimal bundle with the compiled binary and documentation copied into `Contents/Resources/`.

## Build on macOS

Prerequisites:

- Xcode Command Line Tools (`xcode-select --install`)
- Rust toolchain via rustup
- OpenCV available per your `opencv` crate setup (often via Homebrew)

Run:

```bash
./packaging/macos/build.sh
```

## Notes (important)

- This script **does not** code-sign or notarize the app.
- If your binary depends on shared libraries (e.g. OpenCV), you may need to bundle them or ensure they’re available on the target system.
- If you want a DMG installer or a signed/notarized release, we can add that next.
