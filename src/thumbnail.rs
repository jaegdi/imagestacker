use std::path::{Path, PathBuf};
use opencv::core;
use opencv::imgcodecs;
use opencv::imgproc;
use opencv::prelude::{MatTraitConst, UMatTraitConst, MatTraitConstManual};

/// Maximum dimension for thumbnails (longest side)
const THUMB_MAX_DIM: f64 = 800.0;

/// Name of the hidden thumbnail cache directory
const THUMB_CACHE_DIR: &str = ".thumbnails";

/// Get the path where a cached thumbnail would be stored.
///
/// Layout:
///   - Source: `/path/to/images/A7R03436.jpg`
///     Cache:  `/path/to/images/.thumbnails/A7R03436.jpg.png`
///   - Source: `/path/to/images/aligned/0042.png`
///     Cache:  `/path/to/images/aligned/.thumbnails/0042.png.png`
fn thumbnail_cache_path(source: &Path) -> Option<PathBuf> {
    let parent = source.parent()?;
    let filename = source.file_name()?;
    // Append .png so we always write PNG regardless of source format
    let cache_name = format!("{}.png", filename.to_string_lossy());
    Some(parent.join(THUMB_CACHE_DIR).join(cache_name))
}

/// Check if a cached thumbnail is still valid (exists and newer than source).
fn is_cache_valid(source: &Path, cache: &Path) -> bool {
    if !cache.exists() {
        return false;
    }
    // Compare modification times: cache must be newer than source
    let src_modified = match std::fs::metadata(source).and_then(|m| m.modified()) {
        Ok(t) => t,
        Err(_) => return false,
    };
    let cache_modified = match std::fs::metadata(cache).and_then(|m| m.modified()) {
        Ok(t) => t,
        Err(_) => return false,
    };
    cache_modified >= src_modified
}

/// Load a cached thumbnail from disk and return an iced image Handle.
fn load_cached_thumbnail(cache_path: &Path) -> anyhow::Result<iced::widget::image::Handle> {
    // Read the small PNG directly — no GPU needed, very fast
    let img = imgcodecs::imread(cache_path.to_str().unwrap(), imgcodecs::IMREAD_UNCHANGED)?;
    if img.empty() {
        return Err(anyhow::anyhow!("Cached thumbnail is empty"));
    }
    mat_to_rgba_handle(&img)
}

/// Save an already-resized Mat (BGR or BGRA) to the thumbnail cache as PNG.
fn save_to_cache(cache_path: &Path, mat: &core::Mat) -> anyhow::Result<()> {
    if let Some(parent) = cache_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    // Save as PNG with compression level 6 (good balance of speed vs size)
    let params = opencv::core::Vector::<i32>::from_iter([
        imgcodecs::IMWRITE_PNG_COMPRESSION, 6,
    ]);
    imgcodecs::imwrite(cache_path.to_str().unwrap(), mat, &params)?;
    Ok(())
}

/// Convert a Mat (any channel format) to an iced RGBA Handle.
fn mat_to_rgba_handle(img: &core::Mat) -> anyhow::Result<iced::widget::image::Handle> {
    let channels = img.channels();
    let depth = img.depth();

    // Convert 16-bit to 8-bit if necessary
    let img_8bit = if depth == core::CV_16U || depth == core::CV_16S {
        let mut img_8 = core::Mat::default();
        img.convert_to(&mut img_8, core::CV_8U, 1.0 / 256.0, 0.0)?;
        img_8
    } else {
        img.clone()
    };

    // Convert to RGBA for iced
    let img_rgba = match channels {
        1 => {
            // Grayscale → RGBA
            let mut rgba = core::Mat::default();
            imgproc::cvt_color(&img_8bit, &mut rgba, imgproc::COLOR_GRAY2RGBA, 0, core::AlgorithmHint::ALGO_HINT_DEFAULT)?;
            rgba
        }
        3 => {
            let mut rgba = core::Mat::default();
            imgproc::cvt_color(&img_8bit, &mut rgba, imgproc::COLOR_BGR2RGBA, 0, core::AlgorithmHint::ALGO_HINT_DEFAULT)?;
            rgba
        }
        4 => {
            let mut rgba = core::Mat::default();
            imgproc::cvt_color(&img_8bit, &mut rgba, imgproc::COLOR_BGRA2RGBA, 0, core::AlgorithmHint::ALGO_HINT_DEFAULT)?;
            rgba
        }
        _ => return Err(anyhow::anyhow!("Unsupported image format: {} channels", channels)),
    };

    let size = img_rgba.size()?;
    let mut pixels = vec![0u8; (img_rgba.total() * img_rgba.elem_size()?) as usize];
    let data = img_rgba.data_bytes()?;
    pixels.copy_from_slice(data);

    Ok(iced::widget::image::Handle::from_rgba(
        size.width as u32,
        size.height as u32,
        pixels,
    ))
}

/// Generate a thumbnail for an image, using disk cache when available.
///
/// 1. Check if a valid cached thumbnail exists on disk → load it (fast path)
/// 2. Otherwise, load the full image, resize with GPU, save cache, return handle
pub fn generate_thumbnail(path: &PathBuf) -> anyhow::Result<iced::widget::image::Handle> {
    // Fast path: try disk cache first
    if let Some(cache_path) = thumbnail_cache_path(path) {
        if is_cache_valid(path, &cache_path) {
            match load_cached_thumbnail(&cache_path) {
                Ok(handle) => return Ok(handle),
                Err(e) => {
                    log::debug!("Cached thumbnail invalid, regenerating: {}", e);
                    // Fall through to regenerate
                }
            }
        }
    }

    // Slow path: load full image and resize
    let img = imgcodecs::imread(path.to_str().unwrap(), imgcodecs::IMREAD_UNCHANGED)?;
    if img.empty() {
        return Err(anyhow::anyhow!("Failed to load image for thumbnail"));
    }

    let channels = img.channels();
    let depth = img.depth();

    // Convert 16-bit to 8-bit if necessary
    let img_8bit = if depth == core::CV_16U || depth == core::CV_16S {
        let mut img_8 = core::Mat::default();
        img.convert_to(&mut img_8, core::CV_8U, 1.0 / 256.0, 0.0)?;
        img_8
    } else {
        img
    };

    // Convert to a standard format for resizing (BGR or BGRA)
    let img_for_resize = if channels == 4 {
        img_8bit.clone() // already BGRA
    } else if channels == 3 {
        img_8bit.clone() // already BGR
    } else {
        // Grayscale → BGR for consistent handling
        let mut bgr = core::Mat::default();
        imgproc::cvt_color(&img_8bit, &mut bgr, imgproc::COLOR_GRAY2BGR, 0, core::AlgorithmHint::ALGO_HINT_DEFAULT)?;
        bgr
    };

    let size = img_for_resize.size()?;
    let scale = (THUMB_MAX_DIM / size.width as f64).min(THUMB_MAX_DIM / size.height as f64);
    let new_size = core::Size::new(
        (size.width as f64 * scale) as i32,
        (size.height as f64 * scale) as i32,
    );

    // Resize using UMat for GPU acceleration
    let img_umat = img_for_resize.get_umat(
        core::AccessFlag::ACCESS_READ,
        core::UMatUsageFlags::USAGE_DEFAULT,
    )?;
    let mut small_umat = core::UMat::new(core::UMatUsageFlags::USAGE_DEFAULT);
    imgproc::resize(
        &img_umat,
        &mut small_umat,
        new_size,
        0.0,
        0.0,
        imgproc::INTER_AREA,
    )?;

    // Deep-copy from GPU into a standalone Mat (avoids UMat lifetime issue)
    let mut small_mat = core::Mat::default();
    small_umat.copy_to(&mut small_mat)?;

    // Save to disk cache (BGR/BGRA PNG — small file)
    if let Some(cache_path) = thumbnail_cache_path(path) {
        if let Err(e) = save_to_cache(&cache_path, &small_mat) {
            log::debug!("Failed to cache thumbnail for {}: {}", path.display(), e);
        }
    }

    // Convert to RGBA Handle for iced
    mat_to_rgba_handle(&small_mat)
}

/// Name of the hidden preview cache directory (screen-sized previews)
const PREVIEW_CACHE_DIR: &str = ".previews";

/// Get the path where a cached screen-preview would be stored.
fn preview_cache_path(source: &Path, max_dim: u32) -> Option<PathBuf> {
    let parent = source.parent()?;
    let stem = source.file_stem()?;
    // Include max dimension in filename so different screen sizes don't collide
    let cache_name = format!("{}_{}.jpg", stem.to_string_lossy(), max_dim);
    Some(parent.join(PREVIEW_CACHE_DIR).join(cache_name))
}

/// Check if a file extension indicates JPEG format.
fn is_jpeg(path: &Path) -> bool {
    match path.extension().and_then(|e| e.to_str()) {
        Some(ext) => matches!(ext.to_lowercase().as_str(), "jpg" | "jpeg"),
        None => false,
    }
}

/// Load an image scaled to fit the given screen dimensions.
///
/// Optimization strategies:
/// 1. **Disk cache**: Screen-sized previews are cached as small JPEGs (~100KB)
///    for instant loading on subsequent views.
/// 2. **JPEG DCT scaling**: For JPEG source files, `IMREAD_REDUCED_COLOR_4/8`
///    decodes at 1/4 or 1/8 resolution during the DCT stage, skipping most work.
///    A 43MP JPEG loads in ~100ms instead of ~800ms.
/// 3. **GPU resize**: Any remaining resize uses OpenCL UMat for GPU acceleration.
/// 4. **Minimal memory**: Only screen-sized pixel data (~2MP RGBA ≈ 8MB) is
///    produced instead of full resolution (~43MP RGBA ≈ 160MB).
pub fn load_preview_for_screen(path: &Path, screen_width: f32, screen_height: f32) -> anyhow::Result<iced::widget::image::Handle> {
    let start = std::time::Instant::now();
    let max_dim = screen_width.max(screen_height) as u32;
    
    // Fast path: check disk cache
    if let Some(cache_path) = preview_cache_path(path, max_dim) {
        if is_cache_valid(path, &cache_path) {
            match load_cached_thumbnail(&cache_path) {
                Ok(handle) => {
                    let elapsed = start.elapsed();
                    log::info!("Preview from cache: {} in {:.0}ms",
                        path.file_name().unwrap_or_default().to_string_lossy(),
                        elapsed.as_millis());
                    return Ok(handle);
                }
                Err(e) => {
                    log::debug!("Preview cache invalid, regenerating: {}", e);
                }
            }
        }
    }

    let t_cache_check = start.elapsed();

    // For JPEG files, use reduced-resolution decoding (DCT scaling)
    // This is the main speedup: decodes at 1/8 or 1/4 resolution natively
    let (img, decode_reduction) = if is_jpeg(path) {
        // Try 1/8 first (fastest) — for 43MP images gives ~1MP
        let img8 = imgcodecs::imread(path.to_str().unwrap(), imgcodecs::IMREAD_REDUCED_COLOR_8)?;
        if img8.empty() {
            return Err(anyhow::anyhow!("Failed to load image"));
        }
        let s8 = img8.size()?;
        
        // Check if 1/8 resolution is big enough for the screen
        let scale_at_8 = (screen_width / s8.width as f32).min(screen_height / s8.height as f32);
        if scale_at_8 <= 1.0 {
            // 1/8 is big enough — use it (fastest path!)
            (img8, 8.0_f32)
        } else {
            // 1/8 is too small, try 1/4
            let img4 = imgcodecs::imread(path.to_str().unwrap(), imgcodecs::IMREAD_REDUCED_COLOR_4)?;
            if !img4.empty() {
                let s4 = img4.size()?;
                let scale_at_4 = (screen_width / s4.width as f32).min(screen_height / s4.height as f32);
                if scale_at_4 <= 1.0 {
                    (img4, 4.0)
                } else {
                    // 1/4 still too small, try 1/2
                    let img2 = imgcodecs::imread(path.to_str().unwrap(), imgcodecs::IMREAD_REDUCED_COLOR_2)?;
                    if !img2.empty() {
                        (img2, 2.0)
                    } else {
                        (img4, 4.0) // fallback
                    }
                }
            } else {
                (img8, 8.0) // fallback
            }
        }
    } else {
        // Non-JPEG (PNG, TIFF): must load full resolution
        let img = imgcodecs::imread(path.to_str().unwrap(), imgcodecs::IMREAD_UNCHANGED)?;
        if img.empty() {
            return Err(anyhow::anyhow!("Failed to load image"));
        }
        (img, 1.0)
    };

    let t_decode = start.elapsed();

    let size = img.size()?;
    let img_w = size.width as f32;
    let img_h = size.height as f32;

    // Scale down to fit screen (the image may still be larger than screen after DCT reduction)
    let scale = (screen_width / img_w).min(screen_height / img_h).min(1.0);
    
    let result = if scale < 0.95 {
        // Needs further resize
        let new_size = core::Size::new(
            (img_w * scale) as i32,
            (img_h * scale) as i32,
        );

        // Convert 16-bit to 8-bit if necessary
        let img_8bit = if img.depth() == core::CV_16U || img.depth() == core::CV_16S {
            let mut img_8 = core::Mat::default();
            img.convert_to(&mut img_8, core::CV_8U, 1.0 / 256.0, 0.0)?;
            img_8
        } else {
            img
        };

        // Use GPU for resize
        let img_umat = img_8bit.get_umat(
            core::AccessFlag::ACCESS_READ,
            core::UMatUsageFlags::USAGE_DEFAULT,
        )?;
        let mut small_umat = core::UMat::new(core::UMatUsageFlags::USAGE_DEFAULT);
        imgproc::resize(
            &img_umat,
            &mut small_umat,
            new_size,
            0.0,
            0.0,
            imgproc::INTER_AREA,
        )?;

        let mut result_mat = core::Mat::default();
        small_umat.copy_to(&mut result_mat)?;
        result_mat
    } else {
        // Already close to screen size — use as-is
        if img.depth() == core::CV_16U || img.depth() == core::CV_16S {
            let mut img_8 = core::Mat::default();
            img.convert_to(&mut img_8, core::CV_8U, 1.0 / 256.0, 0.0)?;
            img_8
        } else {
            img
        }
    };

    let t_resize = start.elapsed();

    // Save screen-sized preview to disk cache (as JPEG for speed)
    if let Some(cache_path) = preview_cache_path(path, max_dim) {
        if let Some(parent) = cache_path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let params = opencv::core::Vector::<i32>::from_iter([
            imgcodecs::IMWRITE_JPEG_QUALITY, 92,
        ]);
        if let Err(e) = imgcodecs::imwrite(cache_path.to_str().unwrap(), &result, &params) {
            log::debug!("Failed to cache preview: {}", e);
        }
    }

    let t_cache_save = start.elapsed();

    let result_size = result.size()?;
    let handle = mat_to_rgba_handle(&result)?;

    let t_total = start.elapsed();

    let orig_w = (img_w * decode_reduction) as i32;
    let orig_h = (img_h * decode_reduction) as i32;
    log::info!("Preview loaded: {}x{} →(1/{:.0})→ {}x{} →(resize)→ {}x{} in {:.0}ms \
        [cache:{:.0}ms decode:{:.0}ms resize:{:.0}ms save:{:.0}ms rgba:{:.0}ms]",
        orig_w, orig_h,
        decode_reduction,
        size.width, size.height,
        result_size.width, result_size.height,
        t_total.as_millis(),
        t_cache_check.as_millis(),
        (t_decode - t_cache_check).as_millis(),
        (t_resize - t_decode).as_millis(),
        (t_cache_save - t_resize).as_millis(),
        (t_total - t_cache_save).as_millis(),
    );

    Ok(handle)
}

/// Clear the thumbnail cache directory for a given image directory.
///
/// Call this when the user does a full re-import or wants to force regeneration.
#[allow(dead_code)]
pub fn clear_thumbnail_cache(image_dir: &Path) {
    let cache_dir = image_dir.join(THUMB_CACHE_DIR);
    if cache_dir.exists() {
        if let Err(e) = std::fs::remove_dir_all(&cache_dir) {
            log::warn!("Failed to clear thumbnail cache at {}: {}", cache_dir.display(), e);
        } else {
            log::info!("Cleared thumbnail cache: {}", cache_dir.display());
        }
    }
}

/// Delete all cache derivatives (thumbnail + preview) for a given source image.
///
/// This removes:
/// - `.thumbnails/<filename>.png` — the 800px thumbnail
/// - `.previews/<stem>_*.jpg` — all screen-sized preview variants
pub fn delete_cache_for_image(source: &Path) {
    // Delete thumbnail cache
    if let Some(thumb_cache) = thumbnail_cache_path(source) {
        if thumb_cache.exists() {
            if let Err(e) = std::fs::remove_file(&thumb_cache) {
                log::warn!("Failed to delete thumbnail cache {}: {}", thumb_cache.display(), e);
            } else {
                log::debug!("Deleted thumbnail cache: {}", thumb_cache.display());
            }
        }
    }

    // Delete all preview cache variants (may have multiple screen sizes)
    if let Some(parent) = source.parent() {
        let preview_dir = parent.join(PREVIEW_CACHE_DIR);
        if preview_dir.exists() {
            if let Some(stem) = source.file_stem() {
                let prefix = format!("{}_", stem.to_string_lossy());
                if let Ok(entries) = std::fs::read_dir(&preview_dir) {
                    for entry in entries.flatten() {
                        let name = entry.file_name();
                        if name.to_string_lossy().starts_with(&prefix) {
                            if let Err(e) = std::fs::remove_file(entry.path()) {
                                log::warn!("Failed to delete preview cache {}: {}", entry.path().display(), e);
                            } else {
                                log::debug!("Deleted preview cache: {}", entry.path().display());
                            }
                        }
                    }
                }
            }
        }
    }
}