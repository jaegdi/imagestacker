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

/// Compute regional sharpness by dividing image into grid and checking if ANY region is sharp
/// This is better for focus stacking where images may be mostly blurry but have sharp regions
/// Returns (max_regional_sharpness, global_sharpness, sharp_region_count)
pub fn compute_regional_sharpness(img: &Mat, grid_size: i32) -> Result<(f64, f64, usize)> {
    let height = img.rows();
    let width = img.cols();
    
    // Ensure grid doesn't create too-small regions
    let effective_grid = grid_size.max(2).min(height / 100).min(width / 100);
    
    let region_height = height / effective_grid;
    let region_width = width / effective_grid;
    
    if region_height < 50 || region_width < 50 {
        // Image too small for regional analysis, fallback to global
        let global_sharpness = compute_sharpness(img)?;
        return Ok((global_sharpness, global_sharpness, 1));
    }
    
    let mut regional_scores = Vec::new();
    
    // Analyze each region
    for row in 0..effective_grid {
        for col in 0..effective_grid {
            let y = row * region_height;
            let x = col * region_width;
            
            // Clamp to image bounds
            let h = region_height.min(height - y);
            let w = region_width.min(width - x);
            
            if h > 0 && w > 0 {
                let roi = core::Rect::new(x, y, w, h);
                if let Ok(region) = Mat::roi(img, roi) {
                    // Convert BoxedRef to Mat by cloning
                    if let Ok(region_mat) = region.try_clone() {
                        if let Ok(sharpness) = compute_sharpness(&region_mat) {
                            regional_scores.push(sharpness);
                        }
                    }
                }
            }
        }
    }
    
    if regional_scores.is_empty() {
        let global_sharpness = compute_sharpness(img)?;
        return Ok((global_sharpness, global_sharpness, 0));
    }
    
    // Find max regional sharpness (best sharp region)
    let max_sharpness = regional_scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    
    // Compute global sharpness for comparison
    let global_sharpness = compute_sharpness(img)?;
    
    // Count how many regions are "sharp" (above 70% of max)
    let sharp_threshold = max_sharpness * 0.7;
    let sharp_region_count = regional_scores.iter()
        .filter(|&&s| s >= sharp_threshold)
        .count();
    
    log::debug!(
        "Regional analysis: max={:.2}, global={:.2}, sharp_regions={}/{}, grid={}x{}",
        max_sharpness, global_sharpness, sharp_region_count, regional_scores.len(),
        effective_grid, effective_grid
    );
    
    Ok((max_sharpness, global_sharpness, sharp_region_count))
}