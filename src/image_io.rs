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

    let img = imgcodecs::imread(path.to_str().unwrap(), imgcodecs::IMREAD_COLOR)?;

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