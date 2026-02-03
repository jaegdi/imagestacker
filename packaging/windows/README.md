# Windows builds (portable ZIP)

This project is primarily packaged as an RPM on Linux today, but you can also produce a Windows portable build.

## What you get

- `imagestacker.exe`
- `USER_MANUAL.md` + `README.md` (copied into `doc/`)
- optional `imagestacker_icon.png`

The output is a ZIP archive under `dist/`.

## Build on Windows

Prerequisites:

- Rust toolchain via rustup
- OpenCV development/runtime on Windows as required by the `opencv` crate configuration

Run in PowerShell:

```powershell
./packaging/windows/build.ps1
```

Output:

- `dist/imagestacker-windows-x86_64.zip`

## Notes about OpenCV

The Rust `opencv` crate typically links against native OpenCV libraries. Depending on your setup, your `imagestacker.exe` may require OpenCV DLLs available on `PATH` or placed next to the executable.
