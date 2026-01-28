use anyhow::Result;
use opencv::prelude::*;
use opencv::{core, imgproc};

/// Compute image sharpness using multiple methods for robust blur detection
/// Returns a normalized sharpness score (0.0 = very blurry, higher = sharper)
/// Combines Laplacian variance, Tenengrad (Sobel gradient), and Modified Laplacian
pub fn compute_sharpness(img: &Mat) -> Result<f64> {
    let mut gray = Mat::default();
    if img.channels() == 3 {
        imgproc::cvt_color(
            img,
            &mut gray,
            imgproc::COLOR_BGR2GRAY,
            0,
            core::AlgorithmHint::ALGO_HINT_DEFAULT,
        )?;
    } else {
        gray = img.clone();
    }

    // Convert to float32 for better precision
    let mut gray_float = Mat::default();
    gray.convert_to(&mut gray_float, core::CV_32F, 1.0, 0.0)?;

    // Method 1: Laplacian Variance (with Gaussian blur to reduce noise sensitivity)
    let mut blurred = Mat::default();
    imgproc::gaussian_blur(
        &gray_float,
        &mut blurred,
        core::Size::new(3, 3),
        0.0,
        0.0,
        core::BORDER_DEFAULT,
        core::AlgorithmHint::ALGO_HINT_DEFAULT,
    )?;

    let mut laplacian = Mat::default();
    imgproc::laplacian(
        &blurred,
        &mut laplacian,
        core::CV_32F,
        5, // Larger kernel for better detection
        1.0,
        0.0,
        core::BORDER_DEFAULT,
    )?;

    let mut lap_mean = Mat::default();
    let mut lap_stddev = Mat::default();
    core::mean_std_dev(&laplacian, &mut lap_mean, &mut lap_stddev, &core::Mat::default())?;
    let laplacian_var = lap_stddev.at_2d::<f64>(0, 0)? * lap_stddev.at_2d::<f64>(0, 0)?;

    // Method 2: Tenengrad (Sobel gradient magnitude)
    let mut sobel_x = Mat::default();
    let mut sobel_y = Mat::default();
    imgproc::sobel(
        &gray_float,
        &mut sobel_x,
        core::CV_32F,
        1,
        0,
        3,
        1.0,
        0.0,
        core::BORDER_DEFAULT,
    )?;
    imgproc::sobel(
        &gray_float,
        &mut sobel_y,
        core::CV_32F,
        0,
        1,
        3,
        1.0,
        0.0,
        core::BORDER_DEFAULT,
    )?;

    // Compute gradient magnitude squared
    let mut sobel_x_sq = Mat::default();
    let mut sobel_y_sq = Mat::default();
    core::multiply(&sobel_x, &sobel_x, &mut sobel_x_sq, 1.0, -1)?;
    core::multiply(&sobel_y, &sobel_y, &mut sobel_y_sq, 1.0, -1)?;

    let mut gradient_mag_sq = Mat::default();
    core::add(&sobel_x_sq, &sobel_y_sq, &mut gradient_mag_sq, &core::Mat::default(), -1)?;

    // Calculate mean of gradient magnitude squared (Tenengrad)
    let tenengrad = core::mean(&gradient_mag_sq, &core::Mat::default())?[0];

    // Method 3: Modified Laplacian (sum of absolute Sobel derivatives)
    let mut abs_sobel_x = Mat::default();
    let mut abs_sobel_y = Mat::default();
    core::convert_scale_abs(&sobel_x, &mut abs_sobel_x, 1.0, 0.0)?;
    core::convert_scale_abs(&sobel_y, &mut abs_sobel_y, 1.0, 0.0)?;

    let mut mod_laplacian = Mat::default();
    core::add(&abs_sobel_x, &abs_sobel_y, &mut mod_laplacian, &core::Mat::default(), -1)?;
    let mod_lap_mean = core::mean(&mod_laplacian, &core::Mat::default())?[0];

    // Combine metrics with weights
    // Normalize by image dimensions to make threshold more consistent
    let pixel_count = (gray.rows() * gray.cols()) as f64;
    let size_factor = (pixel_count / 1_000_000.0).sqrt(); // Normalize to ~1MP image

    let combined_score = (
        laplacian_var * 0.4 +      // Laplacian variance is good but noise-sensitive
        tenengrad * 0.3 +          // Tenengrad is robust
        mod_lap_mean * 0.3         // Modified Laplacian is also robust
    ) / size_factor.max(0.5);      // Prevent division issues with very small images

    // Log individual components for debugging
    log::debug!(
        "Sharpness components: lap_var={:.2}, tenengrad={:.2}, mod_lap={:.2}, size_factor={:.2}, combined={:.2}",
        laplacian_var, tenengrad, mod_lap_mean, size_factor, combined_score
    );

    Ok(combined_score)
}