use anyhow::Result;
use opencv::prelude::*;
use opencv::{core, imgproc};

/// Smart wrapper for sharpness computation - tries GPU first, falls back to CPU
/// 
/// This is the recommended function to use. It automatically:
/// - Attempts GPU-accelerated computation using UMat (OpenCL)
/// - Falls back to CPU computation if GPU conversion fails
/// - Returns the same results regardless of which path is taken
/// 
/// # Arguments
/// * `img` - Input image (Mat)
/// 
/// # Returns
/// * Sharpness score (0.0 = very blurry, higher = sharper)
pub fn compute_sharpness_auto(img: &Mat) -> Result<f64> {
    // Try GPU first
    match img.get_umat(core::AccessFlag::ACCESS_READ, core::UMatUsageFlags::USAGE_DEFAULT) {
        Ok(img_umat) => {
            // GPU path
            compute_sharpness_umat(&img_umat)
        }
        Err(_) => {
            // Fallback to CPU if GPU conversion fails
            log::debug!("GPU conversion failed, using CPU for sharpness computation");
            compute_sharpness(img)
        }
    }
}

/// Smart wrapper for regional sharpness - tries GPU first, falls back to CPU
/// 
/// This is the recommended function to use. It automatically:
/// - Attempts GPU-accelerated regional analysis using UMat (OpenCL)
/// - Falls back to CPU computation if GPU conversion fails
/// - Returns the same results regardless of which path is taken
/// 
/// # Arguments
/// * `img` - Input image (Mat)
/// * `grid_size` - Grid dimension (e.g., 10 for 10x10 grid)
/// 
/// # Returns
/// * Tuple of (max_regional_sharpness, global_sharpness, sharp_region_count)
pub fn compute_regional_sharpness_auto(img: &Mat, grid_size: i32) -> Result<(f64, f64, usize)> {
    // Try GPU first
    match img.get_umat(core::AccessFlag::ACCESS_READ, core::UMatUsageFlags::USAGE_DEFAULT) {
        Ok(img_umat) => {
            // GPU path
            compute_regional_sharpness_umat(&img_umat, grid_size)
        }
        Err(_) => {
            // Fallback to CPU if GPU conversion fails
            log::debug!("GPU conversion failed, using CPU for regional sharpness computation");
            compute_regional_sharpness(img, grid_size)
        }
    }
}

/// CPU-only: Compute image sharpness using multiple methods for robust blur detection
/// 
/// **Note**: Use `compute_sharpness_auto()` instead for automatic GPU acceleration.
/// This function is kept for CPU fallback when GPU is not available.
/// 
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

/// CPU-only: Compute regional sharpness by dividing image into grid
/// 
/// **Note**: Use `compute_regional_sharpness_auto()` instead for automatic GPU acceleration.
/// This function is kept for CPU fallback when GPU is not available.
/// 
/// This is better for focus stacking where images may be mostly blurry but have sharp regions.
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

/// GPU-accelerated: Compute image sharpness using UMat for OpenCL GPU operations
/// 
/// **Note**: This is called automatically by `compute_sharpness_auto()`.
/// Direct use is only needed when you already have a UMat from previous GPU operations.
/// 
/// Uses GPU for all operations: color conversion, Gaussian blur, Laplacian, Sobel.
/// Returns a normalized sharpness score (0.0 = very blurry, higher = sharper)
pub fn compute_sharpness_umat(img_umat: &core::UMat) -> Result<f64> {
    let gray_umat = if img_umat.channels() == 3 {
        let mut gray = core::UMat::new_def();
        imgproc::cvt_color(
            img_umat,
            &mut gray,
            imgproc::COLOR_BGR2GRAY,
            0,
            core::AlgorithmHint::ALGO_HINT_DEFAULT,
        )?;
        gray
    } else {
        img_umat.clone()
    };

    // Convert to float32 for better precision
    let mut gray_float = core::UMat::new_def();
    gray_umat.convert_to(&mut gray_float, core::CV_32F, 1.0, 0.0)?;

    // Method 1: Laplacian Variance (with Gaussian blur to reduce noise sensitivity)
    let mut blurred = core::UMat::new_def();
    imgproc::gaussian_blur(
        &gray_float,
        &mut blurred,
        core::Size::new(3, 3),
        0.0,
        0.0,
        core::BORDER_DEFAULT,
        core::AlgorithmHint::ALGO_HINT_DEFAULT,
    )?;

    let mut laplacian = core::UMat::new_def();
    imgproc::laplacian(
        &blurred,
        &mut laplacian,
        core::CV_32F,
        5, // Larger kernel for better detection
        1.0,
        0.0,
        core::BORDER_DEFAULT,
    )?;

    // Get stats from UMat (requires downloading to CPU)
    let laplacian_mat = laplacian.get_mat(core::AccessFlag::ACCESS_READ)?;
    let mut lap_mean = Mat::default();
    let mut lap_stddev = Mat::default();
    core::mean_std_dev(&laplacian_mat, &mut lap_mean, &mut lap_stddev, &core::Mat::default())?;
    let laplacian_var = lap_stddev.at_2d::<f64>(0, 0)? * lap_stddev.at_2d::<f64>(0, 0)?;

    // Method 2: Tenengrad (Sobel gradient magnitude)
    let mut sobel_x = core::UMat::new_def();
    let mut sobel_y = core::UMat::new_def();
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
    let mut sobel_x_sq = core::UMat::new_def();
    let mut sobel_y_sq = core::UMat::new_def();
    core::multiply(&sobel_x, &sobel_x, &mut sobel_x_sq, 1.0, -1)?;
    core::multiply(&sobel_y, &sobel_y, &mut sobel_y_sq, 1.0, -1)?;

    let mut gradient_mag_sq = core::UMat::new_def();
    core::add(&sobel_x_sq, &sobel_y_sq, &mut gradient_mag_sq, &core::UMat::new_def(), -1)?;

    // Calculate mean (requires downloading to CPU)
    let gradient_mat = gradient_mag_sq.get_mat(core::AccessFlag::ACCESS_READ)?;
    let tenengrad = core::mean(&gradient_mat, &core::Mat::default())?[0];

    // Method 3: Modified Laplacian (sum of absolute Sobel derivatives)
    let mut abs_sobel_x = core::UMat::new_def();
    let mut abs_sobel_y = core::UMat::new_def();
    core::convert_scale_abs(&sobel_x, &mut abs_sobel_x, 1.0, 0.0)?;
    core::convert_scale_abs(&sobel_y, &mut abs_sobel_y, 1.0, 0.0)?;

    let mut mod_laplacian = core::UMat::new_def();
    core::add(&abs_sobel_x, &abs_sobel_y, &mut mod_laplacian, &core::UMat::new_def(), -1)?;
    
    let mod_lap_mat = mod_laplacian.get_mat(core::AccessFlag::ACCESS_READ)?;
    let mod_lap_mean = core::mean(&mod_lap_mat, &core::Mat::default())?[0];

    // Combine metrics with weights
    let pixel_count = (gray_umat.rows() * gray_umat.cols()) as f64;
    let size_factor = (pixel_count / 1_000_000.0).sqrt();

    let combined_score = (
        laplacian_var * 0.4 +
        tenengrad * 0.3 +
        mod_lap_mean * 0.3
    ) / size_factor.max(0.5);

    log::debug!(
        "GPU Sharpness: lap_var={:.2}, tenengrad={:.2}, mod_lap={:.2}, combined={:.2}",
        laplacian_var, tenengrad, mod_lap_mean, combined_score
    );

    Ok(combined_score)
}

/// GPU-accelerated: Regional sharpness computation using UMat for OpenCL GPU operations
/// 
/// **Note**: This is called automatically by `compute_regional_sharpness_auto()`.
/// Direct use is only needed when you already have a UMat from previous GPU operations.
/// 
/// Divides image into grid, computes GPU-accelerated sharpness for each region.
/// Returns (max_regional_sharpness, global_sharpness, sharp_region_count)
pub fn compute_regional_sharpness_umat(img_umat: &core::UMat, grid_size: i32) -> Result<(f64, f64, usize)> {
    let height = img_umat.rows();
    let width = img_umat.cols();
    
    let effective_grid = grid_size.max(2).min(height / 100).min(width / 100);
    let region_height = height / effective_grid;
    let region_width = width / effective_grid;
    
    if region_height < 50 || region_width < 50 {
        let global_sharpness = compute_sharpness_umat(img_umat)?;
        return Ok((global_sharpness, global_sharpness, 1));
    }
    
    let mut regional_scores = Vec::new();
    
    // Analyze each region
    for row in 0..effective_grid {
        for col in 0..effective_grid {
            let y = row * region_height;
            let x = col * region_width;
            let h = region_height.min(height - y);
            let w = region_width.min(width - x);
            
            if h > 0 && w > 0 {
                let roi = core::Rect::new(x, y, w, h);
                // Note: UMat::roi creates a view, we need to clone it
                if let Ok(region_umat) = core::UMat::roi(img_umat, roi) {
                    if let Ok(region_clone) = region_umat.try_clone() {
                        if let Ok(sharpness) = compute_sharpness_umat(&region_clone) {
                            regional_scores.push(sharpness);
                        }
                    }
                }
            }
        }
    }
    
    if regional_scores.is_empty() {
        let global_sharpness = compute_sharpness_umat(img_umat)?;
        return Ok((global_sharpness, global_sharpness, 0));
    }
    
    let max_sharpness = regional_scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let global_sharpness = compute_sharpness_umat(img_umat)?;
    let sharp_threshold = max_sharpness * 0.7;
    let sharp_region_count = regional_scores.iter()
        .filter(|&&s| s >= sharp_threshold)
        .count();
    
    log::debug!(
        "GPU Regional analysis: max={:.2}, global={:.2}, sharp_regions={}/{}, grid={}x{}",
        max_sharpness, global_sharpness, sharp_region_count, regional_scores.len(),
        effective_grid, effective_grid
    );
    
    Ok((max_sharpness, global_sharpness, sharp_region_count))
}
