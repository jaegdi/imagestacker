use anyhow::Result;
use opencv::prelude::*;
use opencv::imgcodecs;
use std::path::PathBuf;

/// Load an image from disk with timing and logging
pub fn load_image(path: &PathBuf) -> Result<Mat> {
    let start = std::time::Instant::now();
    let filename = path.file_name().unwrap_or_default().to_string_lossy();
    println!("Loading image: {}", filename);
    log::info!("Loading image: {}", path.display());

    // Use IMREAD_UNCHANGED to preserve alpha channel (transparency)
    let img = imgcodecs::imread(path.to_str().unwrap(), imgcodecs::IMREAD_UNCHANGED)?;

    // Convert 16-bit to 8-bit if necessary (some PNG files are 16-bit)
    let img = if img.depth() == opencv::core::CV_16U || img.depth() == opencv::core::CV_16S {
        let mut img_8 = Mat::default();
        img.convert_to(&mut img_8, opencv::core::CV_8U, 1.0/256.0, 0.0)?;
        log::debug!("Converted 16-bit to 8-bit for {}", filename);
        img_8
    } else {
        img
    };

    let elapsed = start.elapsed();
    log::info!(
        "Loaded {} in {:?} - Size: {}x{}, Channels: {}",
        filename, elapsed, img.cols(), img.rows(), img.channels()
    );
    println!(
        "  ✓ Loaded in {:?} - Size: {}x{}, Channels: {}",
        elapsed, img.cols(), img.rows(), img.channels()
    );

    Ok(img)
}