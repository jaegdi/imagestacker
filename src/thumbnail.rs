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
    
    let channels = img.channels();
    let depth = img.depth();
    
    // Convert 16-bit to 8-bit if necessary
    let img_8bit = if depth == opencv::core::CV_16U || depth == opencv::core::CV_16S {
        let mut img_8 = core::Mat::default();
        img.convert_to(&mut img_8, opencv::core::CV_8U, 1.0/256.0, 0.0)?;
        img_8
    } else {
        img
    };
    
    // Convert to RGBA
    let img_rgba = if channels == 3 {
        let mut rgba = core::Mat::default();
        imgproc::cvt_color(&img_8bit, &mut rgba, imgproc::COLOR_BGR2RGBA, 0, core::AlgorithmHint::ALGO_HINT_DEFAULT)?;
        rgba
    } else if channels == 4 {
        let mut rgba = core::Mat::default();
        imgproc::cvt_color(&img_8bit, &mut rgba, imgproc::COLOR_BGRA2RGBA, 0, core::AlgorithmHint::ALGO_HINT_DEFAULT)?;
        rgba
    } else {
        return Err(anyhow::anyhow!("Unsupported image format: {} channels", channels));
    };

    let size = img_rgba.size()?;
    let max_dim = 800.0;
    let scale = (max_dim / size.width as f64).min(max_dim / size.height as f64);
    let new_size = core::Size::new(
        (size.width as f64 * scale) as i32,
        (size.height as f64 * scale) as i32,
    );

    // Resize using UMat for GPU acceleration
    let img_rgba_umat = img_rgba.get_umat(
        core::AccessFlag::ACCESS_READ,
        core::UMatUsageFlags::USAGE_DEFAULT,
    )?;
    let mut small_umat = core::UMat::new(core::UMatUsageFlags::USAGE_DEFAULT);

    imgproc::resize(
        &img_rgba_umat,
        &mut small_umat,
        new_size,
        0.0,
        0.0,
        imgproc::INTER_AREA,
    )?;

    // Get raw pixels from GPU
    let rgba_mat = small_umat.get_mat(core::AccessFlag::ACCESS_READ)?;
    let mut pixels = vec![0u8; (rgba_mat.total() * rgba_mat.elem_size()?) as usize];
    let data = rgba_mat.data_bytes()?;
    pixels.copy_from_slice(data);

    Ok(iced::widget::image::Handle::from_rgba(
        new_size.width as u32,
        new_size.height as u32,
        pixels,
    ))
}