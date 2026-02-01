use std::path::PathBuf;
use opencv::core;
use opencv::imgcodecs;
use opencv::imgproc;
use opencv::prelude::{MatTraitConst, UMatTraitConst, MatTraitConstManual};

pub fn generate_thumbnail(path: &PathBuf) -> anyhow::Result<iced::widget::image::Handle> {
    // Load with IMREAD_UNCHANGED to preserve alpha channel (transparency)
    let img = imgcodecs::imread(path.to_str().unwrap(), imgcodecs::IMREAD_UNCHANGED)?;

    if img.empty() {
        return Err(anyhow::anyhow!("Failed to load image for thumbnail"));
    }

    let size = img.size()?;
    let max_dim = 800.0; // Increased from 200 for better preview quality
    let scale = (max_dim / size.width as f64).min(max_dim / size.height as f64);
    let new_size = core::Size::new(
        (size.width as f64 * scale) as i32,
        (size.height as f64 * scale) as i32,
    );

    // Use UMat for GPU-accelerated resizing and color conversion
    let img_umat = img.get_umat(
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

    // Convert to RGBA, handling both BGR and BGRA inputs
    let mut rgba_umat = core::UMat::new(core::UMatUsageFlags::USAGE_DEFAULT);
    let rgba_mat = small_umat.get_mat(core::AccessFlag::ACCESS_READ)?;
    
    if rgba_mat.channels() == 3 {
        // BGR to RGBA
        imgproc::cvt_color(
            &small_umat,
            &mut rgba_umat,
            imgproc::COLOR_BGR2RGBA,
            0,
            core::AlgorithmHint::ALGO_HINT_DEFAULT,
        )?;
    } else if rgba_mat.channels() == 4 {
        // BGRA to RGBA
        imgproc::cvt_color(
            &small_umat,
            &mut rgba_umat,
            imgproc::COLOR_BGRA2RGBA,
            0,
            core::AlgorithmHint::ALGO_HINT_DEFAULT,
        )?;
    } else {
        return Err(anyhow::anyhow!("Unsupported image format"));
    }

    // Get raw pixels from GPU
    let rgba_mat = rgba_umat.get_mat(core::AccessFlag::ACCESS_READ)?;
    let mut pixels = vec![0u8; (rgba_mat.total() * rgba_mat.elem_size()?) as usize];
    let data = rgba_mat.data_bytes()?;
    pixels.copy_from_slice(data);

    Ok(iced::widget::image::Handle::from_rgba(
        new_size.width as u32,
        new_size.height as u32,
        pixels,
    ))
}