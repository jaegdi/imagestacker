use anyhow::Result;
use opencv::prelude::*;
use opencv::imgcodecs;
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// Supported file extensions
// ---------------------------------------------------------------------------

/// Standard raster image file extensions (lowercase).
pub const STANDARD_IMAGE_EXT: &[&str] = &["jpg", "jpeg", "png", "tif", "tiff"];

/// RAW camera file extensions (lowercase).
pub const RAW_IMAGE_EXT: &[&str] = &[
    "arw",  // Sony Alpha RAW
    "cr2",  // Canon RAW 2
    "cr3",  // Canon RAW 3
    "nef",  // Nikon Electronic Format
    "dng",  // Digital Negative (Adobe)
    "orf",  // Olympus RAW Format
    "raf",  // Fujifilm RAW Format
    "rw2",  // Panasonic RAW
    "pef",  // Pentax Electronic Format
    "srw",  // Samsung RAW
    "3fr",  // Hasselblad RAW
    "mef",  // Mamiya Electronic Format
    "mrw",  // Minolta/Konica-Minolta RAW
    "x3f",  // Sigma RAW
    "dcr",  // Kodak DC RAW
    "kdc",  // Kodak Digital Camera RAW
    "erf",  // Epson RAW Format
    "nrw",  // Nikon RAW compact
    "raw",  // Generic RAW
    "sr2",  // Sony RAW 2
    "rwl",  // Leica RAW
];

/// All file extensions useful for a combined filter dialog.
pub const ALL_IMAGE_EXT: &[&str] = &[
    "jpg", "jpeg", "png", "tif", "tiff",
    "arw", "cr2", "cr3", "nef", "dng", "orf", "raf", "rw2", "pef", "srw",
    "3fr", "mef", "mrw", "x3f", "dcr", "kdc", "erf", "nrw", "raw", "sr2", "rwl",
];

/// Returns `true` if `ext` (case-insensitive) is a standard raster format.
pub fn is_standard_image_ext(ext: &str) -> bool {
    let lower = ext.to_lowercase();
    STANDARD_IMAGE_EXT.iter().any(|&s| s == lower)
}

/// Returns `true` if `ext` (case-insensitive) is a known RAW camera format.
pub fn is_raw_ext(ext: &str) -> bool {
    let lower = ext.to_lowercase();
    RAW_IMAGE_EXT.iter().any(|&s| s == lower)
}

/// Returns `true` if `ext` (case-insensitive) is any supported image type.
pub fn is_any_image_ext(ext: &str) -> bool {
    is_standard_image_ext(ext) || is_raw_ext(ext)
}

// ---------------------------------------------------------------------------
// RAW decoding via rawloader (pure-Rust fallback)
// ---------------------------------------------------------------------------

/// Decode a RAW camera file using rawloader and return a BGR 8-bit OpenCV Mat.
///
/// Uses simple 2×2 block-average demosaicing which produces half-resolution
/// output but works correctly for all common Bayer (RGGB/BGGR/GRBG/GBRG)
/// and most non-standard colour filter arrays.
fn load_raw_via_rawloader(path: &Path) -> Result<opencv::core::Mat> {
    let path_str = path
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("Invalid UTF-8 in RAW path"))?;

    let rawfile = rawloader::decode_file(path_str)
        .map_err(|e| anyhow::anyhow!("RAW decode failed for {}: {}", path.display(), e))?;

    let width = rawfile.width;
    let height = rawfile.height;

    let data_f32: Vec<f32> = match &rawfile.data {
        rawloader::RawImageData::Integer(v) => v.iter().map(|&x| x as f32).collect(),
        rawloader::RawImageData::Float(v) => v.clone(),
    };

    if data_f32.len() < width * height {
        return Err(anyhow::anyhow!(
            "RAW data too short: {} < {} for {}×{}",
            data_f32.len(), width * height, width, height
        ));
    }

    // Per-channel black/white levels (rawloader stores them in RGBE order: 0=R, 1=G, 2=B, 3=E)
    let bl = [
        rawfile.blacklevels[0] as f32, // R
        rawfile.blacklevels[1] as f32, // G
        rawfile.blacklevels[2] as f32, // B
    ];
    let wl = [
        rawfile.whitelevels[0] as f32, // R
        rawfile.whitelevels[1] as f32, // G
        rawfile.whitelevels[2] as f32, // B
    ];

    // White balance coefficients (RGBE order); normalise by the green multiplier
    // so that a neutral scene renders as neutral grey.
    let wb = rawfile.wb_coeffs;
    let g_ref = if wb[1] > 0.0 { wb[1] } else { 1.0_f32 };
    let wb_mult = [
        if wb[0] > 0.0 { wb[0] / g_ref } else { 1.0_f32 }, // R
        1.0_f32,                                              // G (reference)
        if wb[2] > 0.0 { wb[2] / g_ref } else { 1.0_f32 }, // B
    ];

    let cfa = &rawfile.cfa;
    let out_w = (width / 2).max(1);
    let out_h = (height / 2).max(1);
    let mut bgr = vec![0u8; out_w * out_h * 3];

    for y in 0..out_h {
        for x in 0..out_w {
            let mut ch_sum = [0f32; 3];
            let mut ch_cnt = [0u32; 3];

            for dy in 0..2usize {
                for dx in 0..2usize {
                    let row = y * 2 + dy;
                    let col = x * 2 + dx;
                    if row >= height || col >= width {
                        continue;
                    }
                    // clamp to 0-2 for R/G/B (handles X-Trans and other exotic patterns)
                    let ch = cfa.color_at(row, col).min(2);
                    let val = data_f32[row * width + col];
                    let ch_scale = (wl[ch] - bl[ch]).max(1.0);
                    let norm = ((val - bl[ch]) / ch_scale).clamp(0.0, 1.0);
                    ch_sum[ch] += norm;
                    ch_cnt[ch] += 1;
                }
            }

            let r = (if ch_cnt[0] > 0 { ch_sum[0] / ch_cnt[0] as f32 } else { 0.0 } * wb_mult[0]).clamp(0.0, 1.0);
            let g = (if ch_cnt[1] > 0 { ch_sum[1] / ch_cnt[1] as f32 } else { 0.0 } * wb_mult[1]).clamp(0.0, 1.0);
            let b = (if ch_cnt[2] > 0 { ch_sum[2] / ch_cnt[2] as f32 } else { 0.0 } * wb_mult[2]).clamp(0.0, 1.0);

            let idx = (y * out_w + x) * 3;
            bgr[idx]     = (b * 255.0).round() as u8; // B
            bgr[idx + 1] = (g * 255.0).round() as u8; // G
            bgr[idx + 2] = (r * 255.0).round() as u8; // R
        }
    }

    let mut mat = unsafe {
        opencv::core::Mat::new_rows_cols(
            out_h as i32, out_w as i32, opencv::core::CV_8UC3,
        )?
    };
    mat.data_bytes_mut()?.copy_from_slice(&bgr);

    log::info!("RAW demosaic: {}×{} → {}×{} BGR8", width, height, out_w, out_h);
    Ok(mat)
}

// ---------------------------------------------------------------------------
// Public image loading
// ---------------------------------------------------------------------------

/// Load an image from disk with timing and logging.
///
/// Supports all standard OpenCV formats (JPEG, PNG, TIFF, …) plus RAW camera
/// files via rawloader fallback when OpenCV returns an empty result.
/// 16-bit images are automatically down-converted to 8-bit.
pub fn load_image(path: &PathBuf) -> Result<opencv::core::Mat> {
    let start = std::time::Instant::now();
    let filename = path.file_name().unwrap_or_default().to_string_lossy();
    log::info!("Loading image: {}", path.display());

    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");

    // Bypass OpenCV for RAW camera files so libtiff does not emit noisy
    // decoder errors for formats like Sony ARW before we can fall back.
    if is_raw_ext(ext) {
        let img = load_raw_via_rawloader(path.as_path())?;
        let elapsed = start.elapsed();
        log::info!(
            "Loaded {} in {:?} – {}×{}, {} ch",
            filename, elapsed, img.cols(), img.rows(), img.channels()
        );
        return Ok(img);
    }

    // Use IMREAD_UNCHANGED to preserve alpha channel (transparency)
    let img = imgcodecs::imread(path.to_str().unwrap(), imgcodecs::IMREAD_UNCHANGED)?;

    // If OpenCV returned an empty result, try rawloader for RAW files
    let img = if img.empty() {
        if is_raw_ext(ext) {
            log::info!("OpenCV returned empty – falling back to rawloader for: {}", filename);
            load_raw_via_rawloader(path.as_path())?
        } else {
            return Err(anyhow::anyhow!("Failed to load image (empty): {}", path.display()));
        }
    } else {
        // Convert 16-bit to 8-bit if necessary
        if img.depth() == opencv::core::CV_16U || img.depth() == opencv::core::CV_16S {
            let mut img_8 = opencv::core::Mat::default();
            img.convert_to(&mut img_8, opencv::core::CV_8U, 1.0 / 256.0, 0.0)?;
            log::debug!("Converted 16-bit to 8-bit for {}", filename);
            img_8
        } else {
            img
        }
    };

    let elapsed = start.elapsed();
    log::info!(
        "Loaded {} in {:?} – {}×{}, {} ch",
        filename, elapsed, img.cols(), img.rows(), img.channels()
    );

    Ok(img)
}
