use anyhow::Result;
use opencv::prelude::*;
use opencv::{calib3d, core, features2d, imgproc};
use rayon::prelude::*;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

/// Progress callback: (message, percentage)
pub type ProgressCallback = Arc<Mutex<dyn FnMut(String, f32) + Send>>;

/// Feature detector type
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FeatureDetectorType {
    ORB,
    SIFT,
    AKAZE,
}

/// Processing configuration
#[derive(Debug, Clone)]
pub struct ProcessingConfig {
    pub sharpness_threshold: f32,
    pub use_clahe: bool,
    pub feature_detector: FeatureDetectorType,
    pub sharpness_batch_size: usize,
    pub feature_batch_size: usize,
    pub warp_batch_size: usize,
    pub stacking_batch_size: usize,
}

impl Default for ProcessingConfig {
    fn default() -> Self {
        Self {
            sharpness_threshold: 30.0,
            use_clahe: true,
            feature_detector: FeatureDetectorType::ORB,
            sharpness_batch_size: 8,
            feature_batch_size: 16,
            warp_batch_size: 16,
            stacking_batch_size: 12,
        }
    }
}

pub fn load_image(path: &PathBuf) -> Result<Mat> {
    use opencv::imgcodecs;
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

/// Compute image sharpness using multiple methods for robust blur detection
/// Returns a normalized sharpness score (0.0 = very blurry, higher = sharper)
/// Combines Laplacian variance, Tenengrad (Sobel gradient), and Modified Laplacian
fn compute_sharpness(img: &Mat) -> Result<f64> {
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

/// Extract features from an image using the specified detector
fn extract_features(
    img: &Mat,
    detector_type: FeatureDetectorType,
) -> Result<(opencv::core::Vector<core::KeyPoint>, core::Mat)> {
    let mut keypoints = opencv::core::Vector::new();
    let mut descriptors = core::Mat::default();

    match detector_type {
        FeatureDetectorType::ORB => {
            let mut orb = features2d::ORB::create(
                3000, // Increased from 1500 for better feature detection in difficult images
                1.2,  // scaleFactor
                8,    // nlevels - multi-scale detection
                15,   // edgeThreshold - reduced from 31 for more edge features
                0,    // firstLevel
                2,    // WTA_K
                features2d::ORB_ScoreType::HARRIS_SCORE,
                31, // patchSize
                10, // fastThreshold - reduced from 20 for more sensitive detection
            )?;
            orb.detect_and_compute(
                img,
                &core::Mat::default(),
                &mut keypoints,
                &mut descriptors,
                false,
            )?;
        }
        FeatureDetectorType::SIFT => {
            // SIFT: Scale-Invariant Feature Transform - best quality, slower
            let mut sift = features2d::SIFT::create(
                0,     // nfeatures (0 = unlimited)
                3,     // nOctaveLayers
                0.04,  // contrastThreshold
                10.0,  // edgeThreshold
                1.6,   // sigma
                false, // enable_precise_upscale
            )?;
            sift.detect_and_compute(
                img,
                &core::Mat::default(),
                &mut keypoints,
                &mut descriptors,
                false,
            )?;
        }
        FeatureDetectorType::AKAZE => {
            // AKAZE: Accelerated-KAZE - good balance of speed and quality
            let mut akaze = features2d::AKAZE::create(
                features2d::AKAZE_DescriptorType::DESCRIPTOR_MLDB,
                0,      // descriptor_size
                3,      // descriptor_channels
                0.001,  // threshold
                4,      // nOctaves
                4,      // nOctaveLayers
                features2d::KAZE_DiffusivityType::DIFF_PM_G2,
                512,    // max_points
            )?;
            akaze.detect_and_compute(
                img,
                &core::Mat::default(),
                &mut keypoints,
                &mut descriptors,
                false,
            )?;
        }
    }

    Ok((keypoints, descriptors))
}

pub fn align_images(
    image_paths: &[PathBuf], 
    output_dir: &std::path::Path,
    config: &ProcessingConfig,
    progress_cb: Option<ProgressCallback>,
) -> Result<core::Rect> {
    let report_progress = |msg: &str, pct: f32| {
        if let Some(ref cb) = progress_cb {
            if let Ok(mut cb_lock) = cb.lock() {
                cb_lock(msg.to_string(), pct);
            }
        }
    };

    report_progress("Starting alignment...", 0.0);

    if image_paths.len() < 2 {
        let first_img = load_image(&image_paths[0])?;
        return Ok(core::Rect::new(0, 0, first_img.cols(), first_img.rows()));
    }

    let aligned_dir = output_dir.join("aligned");
    std::fs::create_dir_all(&aligned_dir)?;

    // Filter images by sharpness
    log::info!("Checking image sharpness for {} images...", image_paths.len());
    println!("\n=== BLUR DETECTION STARTING ===");
    println!("Analyzing {} images for sharpness (parallel batches)...", image_paths.len());
    
    report_progress("Analyzing image sharpness...", 5.0);

    // Process images in batches to avoid excessive memory usage
    let batch_size = config.sharpness_batch_size;
    let mut sharpness_scores: Vec<(usize, PathBuf, f64)> = Vec::new();
    
    let total_batches = (image_paths.len() + batch_size - 1) / batch_size;
    for (batch_idx, batch) in image_paths.chunks(batch_size).enumerate() {
        let batch_start = batch_idx * batch_size;
        println!("  Processing batch {}/{} ({} images)...", 
                 batch_idx + 1, total_batches, batch.len());
        
        let progress_pct = 5.0 + (batch_idx as f32 / total_batches as f32) * 15.0;
        report_progress(&format!("Blur detection: batch {}/{}", batch_idx + 1, total_batches), progress_pct);

        let batch_results: Vec<(usize, PathBuf, f64)> = batch
            .par_iter()
            .enumerate()
            .filter_map(|(batch_idx, path)| {
                let idx = batch_start + batch_idx;
                let filename = path.file_name().unwrap_or_default().to_string_lossy();
                match load_image(path) {
                    Ok(img) => match compute_sharpness(&img) {
                        Ok(sharpness) => {
                            log::info!("Image {} ({}): sharpness = {:.2}", idx, filename, sharpness);
                            println!("    [{}] {}: sharpness = {:.2}", idx, filename, sharpness);
                            Some((idx, path.clone(), sharpness))
                        }
                        Err(e) => {
                            log::warn!("Failed to compute sharpness for {}: {}", filename, e);
                            println!("    [{}] {}: ERROR computing sharpness: {}", idx, filename, e);
                            None
                        }
                    },
                    Err(e) => {
                        log::warn!("Failed to load image {}: {}", filename, e);
                        println!("    [{}] {}: ERROR loading: {}", idx, filename, e);
                        None
                    }
                }
            })
            .collect();
        
        sharpness_scores.extend(batch_results);
    }

    // Calculate statistics for adaptive thresholding
    let scores: Vec<f64> = sharpness_scores.iter().map(|(_, _, s)| *s).collect();
    let mean_sharpness: f64 = scores.iter().sum::<f64>() / scores.len() as f64;
    
    let variance: f64 = scores.iter()
        .map(|s| (s - mean_sharpness).powi(2))
        .sum::<f64>() / scores.len() as f64;
    let stddev = variance.sqrt();
    
    // Sort scores to find median and quartiles
    let mut sorted_scores = scores.clone();
    sorted_scores.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = sorted_scores[sorted_scores.len() / 2];
    let q1 = sorted_scores[sorted_scores.len() / 4];
    let q3 = sorted_scores[(3 * sorted_scores.len()) / 4];
    
    // Adaptive threshold: Use multiple criteria to detect outliers
    // 1. Absolute threshold from config
    let absolute_threshold = config.sharpness_threshold as f64;
    // 2. Statistical threshold (mean - 1.0 * stddev) - more aggressive than before
    let statistical_threshold = (mean_sharpness - 1.0 * stddev).max(absolute_threshold);
    // 3. Quartile-based threshold (Q1 - 1.0 * IQR) for robust outlier detection - more aggressive
    let iqr = q3 - q1;
    let quartile_threshold = (q1 - 1.0 * iqr).max(absolute_threshold);
    
    // Use the most conservative (highest) threshold
    let dynamic_threshold = absolute_threshold.max(statistical_threshold).max(quartile_threshold);
    
    let stats_msg = format!(
        "Sharpness statistics:\n  Mean: {:.2}\n  Median: {:.2}\n  StdDev: {:.2}\n  Q1: {:.2}\n  Q3: {:.2}\n  IQR: {:.2}\n  Threshold: {:.2}",
        mean_sharpness, median, stddev, q1, q3, iqr, dynamic_threshold
    );
    log::info!("{}", stats_msg);
    println!("\n{}", stats_msg);

    let mut sharp_image_paths = Vec::new();
    let mut skipped_count = 0;

    println!("\n=== FILTERING RESULTS ===");
    for (idx, path, sharpness) in sharpness_scores {
        let filename = path.file_name().unwrap_or_default().to_string_lossy();
        if sharpness >= dynamic_threshold {
            sharp_image_paths.push((idx, path.clone()));
            let msg = format!("✓ Image {} ({}): {:.2} - INCLUDED", idx, filename, sharpness);
            log::info!("{}", msg);
            println!("{}", msg);
        } else {
            skipped_count += 1;
            let msg = format!("✗ Image {} ({}): {:.2} - SKIPPED (below {:.2})", 
                idx, filename, sharpness, dynamic_threshold);
            log::warn!("{}", msg);
            println!("{}", msg);
        }
    }

    let summary = format!(
        "\n=== FILTERING SUMMARY ===\n  Total images: {}\n  Sharp images: {}\n  Blurry images skipped: {}\n========================\n",
        image_paths.len(),
        sharp_image_paths.len(),
        skipped_count
    );
    log::info!("{}", summary);
    println!("{}", summary);

    if sharp_image_paths.is_empty() {
        let err_msg = "ERROR: No sharp images found for alignment! All images appear too blurry.";
        log::error!("{}", err_msg);
        println!("{}", err_msg);
        return Err(anyhow::anyhow!(err_msg));
    }

    if sharp_image_paths.len() < 2 {
        let first_img = load_image(&sharp_image_paths[0].1)?;
        log::warn!("Only one sharp image found, skipping alignment");
        println!("⚠ Only one sharp image found, skipping alignment");
        return Ok(core::Rect::new(0, 0, first_img.cols(), first_img.rows()));
    }

    // Save reference image (first sharp image)
    println!("\n=== STARTING ALIGNMENT ===");
    println!("Using {} sharp images for alignment", sharp_image_paths.len());
    println!("Reference image: {}", sharp_image_paths[0].1.file_name().unwrap_or_default().to_string_lossy());
    
    let ref_img = load_image(&sharp_image_paths[0].1)?;
    opencv::imgcodecs::imwrite(
        aligned_dir
            .join(format!("{:04}.png", sharp_image_paths[0].0))
            .to_str()
            .unwrap(),
        &ref_img,
        &opencv::core::Vector::new(),
    )?;

    let mut common_mask = Mat::new_rows_cols_with_default(
        ref_img.rows(),
        ref_img.cols(),
        core::CV_8U,
        core::Scalar::all(255.0),
    )?;

    let start_total = std::time::Instant::now();

    // 1. Batched Parallel Feature Extraction and Pairwise Matching
    log::info!(
        "Processing {} sharp image pairs in batches...",
        sharp_image_paths.len() - 1
    );
    println!("\nExtracting features and matching {} image pairs...", sharp_image_paths.len() - 1);
    println!("Using {} feature detector", match config.feature_detector {
        FeatureDetectorType::ORB => "ORB (Fast)",
        FeatureDetectorType::SIFT => "SIFT (Best Quality)",
        FeatureDetectorType::AKAZE => "AKAZE (Balanced)",
    });
    
    report_progress("Feature extraction starting...", 20.0);

    let start_matching = std::time::Instant::now();
    use rayon::prelude::*;

    // Alignment scale: less downsampling to preserve small sharp areas
    const ALIGNMENT_SCALE: f64 = 0.7; // Increased from 0.5 to preserve more detail
    let feature_batch_size = config.feature_batch_size;
    let mut pairwise_transforms = Vec::new();

    let total_feature_batches = (sharp_image_paths.len() + feature_batch_size) / feature_batch_size;
    let mut feature_batch_count = 0;

    // Process sharp images in batches for memory efficiency with parallel processing within batches
    for batch_start in (0..sharp_image_paths.len()).step_by(feature_batch_size) {
        let batch_end = (batch_start + feature_batch_size + 1).min(sharp_image_paths.len());
        let batch_paths: Vec<&PathBuf> = sharp_image_paths[batch_start..batch_end]
            .iter()
            .map(|(_, path)| path)
            .collect();

        feature_batch_count += 1;
        let progress_pct = 20.0 + (feature_batch_count as f32 / total_feature_batches as f32) * 30.0;
        report_progress(&format!("Feature extraction: batch {}/{}", feature_batch_count, total_feature_batches), progress_pct);

        log::info!(
            "Extracting features for batch {}-{} of {} (batch size: {})",
            batch_start,
            batch_end - 1,
            sharp_image_paths.len() - 1,
            batch_paths.len()
        );

        let detector_type = config.feature_detector;
        let use_clahe = config.use_clahe;

        // Extract features for this batch in parallel
        let batch_features: Vec<Result<(Vec<core::KeyPoint>, core::Mat, f64)>> = batch_paths
            .par_iter()
            .map(|&path| {
                let img = load_image(path)?;

                // Convert to grayscale for preprocessing
                let mut gray = Mat::default();
                if img.channels() == 3 {
                    imgproc::cvt_color(
                        &img,
                        &mut gray,
                        imgproc::COLOR_BGR2GRAY,
                        0,
                        core::AlgorithmHint::ALGO_HINT_DEFAULT,
                    )?;
                } else {
                    gray = img.clone();
                }

                // Apply CLAHE if enabled (dramatically improves feature detection in dark images)
                let preprocessed = if use_clahe {
                    let mut clahe = imgproc::create_clahe(2.0, core::Size::new(8, 8))?;
                    let mut enhanced = Mat::default();
                    clahe.apply(&gray, &mut enhanced)?;
                    enhanced
                } else {
                    gray
                };

                // Downsample image for faster feature detection
                let mut small_img = Mat::default();
                let scale = ALIGNMENT_SCALE;
                imgproc::resize(
                    &preprocessed,
                    &mut small_img,
                    core::Size::default(),
                    scale,
                    scale,
                    imgproc::INTER_AREA,
                )?;

                // Extract features using configured detector
                let (keypoints, descriptors) = extract_features(&small_img, detector_type)?;
                
                Ok((keypoints.to_vec(), descriptors, scale))
            })
            .collect();

        // Convert to valid features
        let mut valid_batch_features = Vec::new();
        for f in batch_features {
            valid_batch_features.push(f?);
        }

        // Compute pairwise transforms for consecutive pairs in this batch
        for i in 0..valid_batch_features.len() - 1 {
            let (ref prev_keypoints, ref prev_descriptors, prev_scale) = valid_batch_features[i];
            let (ref curr_keypoints, ref curr_descriptors, curr_scale) =
                valid_batch_features[i + 1];

            let t_step_2x3 = if curr_descriptors.empty() || prev_descriptors.empty() {
                Mat::default()
            } else {
                // Choose matcher based on feature detector type
                let norm_type = match config.feature_detector {
                    FeatureDetectorType::ORB => core::NORM_HAMMING,      // Binary descriptors
                    FeatureDetectorType::SIFT => core::NORM_L2,          // Float descriptors
                    FeatureDetectorType::AKAZE => core::NORM_HAMMING,    // Binary descriptors
                };

                let mut matcher = features2d::BFMatcher::create(norm_type, false)?;

                // Add training descriptors first
                let mut train_descriptors = opencv::core::Vector::<core::Mat>::new();
                train_descriptors.push(prev_descriptors.clone());
                matcher.add(&train_descriptors)?;

                // Then match
                let mut matches = opencv::core::Vector::<core::DMatch>::new();
                matcher.match_(curr_descriptors, &mut matches, &core::Mat::default())?;

                let mut matches_vec = matches.to_vec();
                matches_vec.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());

                // Use top 40% of matches (increased from 30% for more robust alignment)
                // This is especially important for dark images with fewer good features
                let count = (matches_vec.len() as f32 * 0.4) as usize;
                let count = count.max(15).min(matches_vec.len()); // Increased min from 10 to 15
                let good_matches = &matches_vec[..count];

                let mut src_pts = opencv::core::Vector::<core::Point2f>::new();
                let mut dst_pts = opencv::core::Vector::<core::Point2f>::new();

                // Scale keypoints back to original image coordinates
                for m in good_matches {
                    let curr_pt = curr_keypoints[m.query_idx as usize].pt();
                    let prev_pt = prev_keypoints[m.train_idx as usize].pt();
                    src_pts.push(core::Point2f::new(
                        curr_pt.x / curr_scale as f32,
                        curr_pt.y / curr_scale as f32,
                    ));
                    dst_pts.push(core::Point2f::new(
                        prev_pt.x / prev_scale as f32,
                        prev_pt.y / prev_scale as f32,
                    ));
                }

                if src_pts.len() >= 4 {
                    let mut inliers = Mat::default();
                    // Improved RANSAC parameters for challenging alignments
                    calib3d::estimate_affine_partial_2d(
                        &src_pts,
                        &dst_pts,
                        &mut inliers,
                        calib3d::RANSAC,
                        5.0,  // Increased from 3.0 for more tolerance with small features
                        5000, // Increased from 2000 for more thorough search
                        0.995, // Increased confidence from 0.99
                        10,
                    )?
                } else {
                    Mat::default()
                }
            };

            pairwise_transforms.push(t_step_2x3);
        }

        // Batch features are dropped here, freeing memory before next batch
    }

    log::info!(
        "Batched parallel matching took {:?}",
        start_matching.elapsed()
    );

    // 2. Sequential Transformation Accumulation
    log::info!("Accumulating transformations...");
    let mut total_transforms = Vec::new();
    let mut t_total = Mat::eye(3, 3, core::CV_64F)?.to_mat()?;
    total_transforms.push(t_total.clone());

    for t_step_2x3 in pairwise_transforms {
        if !t_step_2x3.empty() {
            let mut t_step_3x3 = Mat::eye(3, 3, core::CV_64F)?.to_mat()?;
            for row in 0..2 {
                for col in 0..3 {
                    *t_step_3x3.at_2d_mut::<f64>(row, col)? = *t_step_2x3.at_2d::<f64>(row, col)?;
                }
            }

            let mut next_t_total = Mat::default();
            core::gemm(
                &t_total,
                &t_step_3x3,
                1.0,
                &Mat::default(),
                0.0,
                &mut next_t_total,
                0,
            )?;
            t_total = next_t_total;
        }
        total_transforms.push(t_total.clone());
    }

    // 3. Batched Parallel Warping
    log::info!("Warping images in parallel batches...");
    println!("\nWarping {} sharp images...", sharp_image_paths.len());
    let start_warping = std::time::Instant::now();
    let ref_size = ref_img.size()?;

    let warp_batch_size = config.warp_batch_size;
    let total_sharp_images = sharp_image_paths.len(); // Use sharp images count, not all images
    
    report_progress("Warping images...", 50.0);

    let total_warp_batches = (total_sharp_images + warp_batch_size - 1) / warp_batch_size;
    let mut warp_batch_count = 0;

    // Note: We iterate over indices in sharp_image_paths (0 to total_sharp_images-1)
    // Each element contains (original_index, path) so we can save with the correct filename

    for batch_start in (0..total_sharp_images).step_by(warp_batch_size) {
        let batch_end = (batch_start + warp_batch_size).min(total_sharp_images);
        
        warp_batch_count += 1;
        let progress_pct = 50.0 + (warp_batch_count as f32 / total_warp_batches as f32) * 30.0;
        report_progress(&format!("Warping images: batch {}/{}", warp_batch_count, total_warp_batches), progress_pct);

        log::info!(
            "Warping batch {}-{} of {} in parallel",
            batch_start,
            batch_end - 1,
            total_sharp_images - 1
        );
        println!("  Warping batch {}-{} of {}...", batch_start, batch_end - 1, total_sharp_images - 1);

        // Process this batch in parallel
        let batch_results: Vec<Result<()>> = (batch_start..batch_end)
            .into_par_iter()
            .map(|i| {
                let (orig_idx, path) = &sharp_image_paths[i];
                let img = load_image(path)?;
                let mut warped = Mat::default();
                let mut t_warp_2x3 = Mat::zeros(2, 3, core::CV_64F)?.to_mat()?;

                for row in 0..2 {
                    for col in 0..3 {
                        *t_warp_2x3.at_2d_mut::<f64>(row, col)? =
                            *total_transforms[i].at_2d::<f64>(row, col)?;
                    }
                }

                imgproc::warp_affine(
                    &img,
                    &mut warped,
                    &t_warp_2x3,
                    ref_size,
                    imgproc::INTER_LINEAR,
                    core::BORDER_CONSTANT,
                    core::Scalar::default(),
                )?;

                // Save aligned image with original index
                let aligned_path = aligned_dir.join(format!("{:04}.png", orig_idx));
                opencv::imgcodecs::imwrite(
                    aligned_path.to_str().unwrap(),
                    &warped,
                    &opencv::core::Vector::new(),
                )?;

                Ok(())
            })
            .collect();

        // Check for errors
        for res in batch_results {
            res?;
        }
        // Batch is complete, all images in batch are now released
    }

    log::info!("Warping took {:?}", start_warping.elapsed());
    println!("✓ Warping complete in {:?}", start_warping.elapsed());

    // 4. Update Common Mask (Parallel)
    log::info!("Updating common mask...");
    println!("\nComputing common mask area...");
    let start_mask = std::time::Instant::now();

    // Compute all masks in parallel (only for sharp images)
    let masks: Vec<Result<Mat>> = sharp_image_paths
        .par_iter()
        .map(|(orig_idx, _)| {
            let aligned_path = aligned_dir.join(format!("{:04}.png", orig_idx));
            let img = opencv::imgcodecs::imread(
                aligned_path.to_str().unwrap(),
                opencv::imgcodecs::IMREAD_COLOR,
            )?;
            let mut gray = Mat::default();
            imgproc::cvt_color(
                &img,
                &mut gray,
                imgproc::COLOR_BGR2GRAY,
                0,
                core::AlgorithmHint::ALGO_HINT_DEFAULT,
            )?;
            let mut mask = Mat::default();
            imgproc::threshold(&gray, &mut mask, 1.0, 255.0, imgproc::THRESH_BINARY)?;
            Ok(mask)
        })
        .collect();

    // Combine masks sequentially
    for mask_result in masks {
        let mask = mask_result?;
        let mut new_common = Mat::default();
        core::bitwise_and(&common_mask, &mask, &mut new_common, &core::Mat::default())?;
        common_mask = new_common;
    }

    log::info!("Mask computation took {:?}", start_mask.elapsed());
    println!("✓ Mask computation complete in {:?}", start_mask.elapsed());

    log::info!("Total alignment took {:?}", start_total.elapsed());
    log::info!("Alignment complete");
    println!("\n=== ALIGNMENT COMPLETE ===");
    println!("Total time: {:?}", start_total.elapsed());
    println!("========================\n");

    // Find bounding box of common area
    let mut non_zero = Mat::default();
    core::find_non_zero(&common_mask, &mut non_zero)?;
    let crop_rect = imgproc::bounding_rect(&non_zero)?;
    log::info!("Common area crop rect: {:?}", crop_rect);

    Ok(crop_rect)
}

pub fn stack_images(
    image_paths: &[PathBuf],
    output_dir: &Path,
    crop_rect: Option<core::Rect>,
    config: &ProcessingConfig,
    progress_cb: Option<ProgressCallback>,
) -> Result<Mat> {
    let report_progress = |msg: &str, pct: f32| {
        if let Some(ref cb) = progress_cb {
            if let Ok(mut cb_lock) = cb.lock() {
                cb_lock(msg.to_string(), pct);
            }
        }
    };

    report_progress("Starting stacking...", 0.0);

    log::info!("Stacking {} images", image_paths.len());
    let mut reversed_paths: Vec<PathBuf> = image_paths.iter().cloned().collect();
    reversed_paths.reverse();
    
    let result = stack_recursive(&reversed_paths, output_dir, 0, config, progress_cb.clone())?;

    report_progress("Saving final result...", 95.0);

    let final_dir = output_dir.join("final");
    std::fs::create_dir_all(&final_dir)?;

    let mut final_path = final_dir.join("result.png");
    let mut counter = 1;
    while final_path.exists() {
        final_path = final_dir.join(format!("result_{}.png", counter));
        counter += 1;
    }

    log::info!("Saving final result to {}", final_path.display());

    let result = if let Some(rect) = crop_rect {
        log::info!("Cropping final result to {:?}", rect);
        let roi = Mat::roi(&result, rect)?;
        let mut cropped = Mat::default();
        roi.copy_to(&mut cropped)?;
        cropped
    } else {
        result.clone()
    };

    opencv::imgcodecs::imwrite(
        final_path.to_str().unwrap(),
        &result,
        &opencv::core::Vector::new(),
    )?;

    Ok(result)
}

fn stack_recursive(
    image_paths: &[PathBuf], 
    output_dir: &Path, 
    level: usize,
    config: &ProcessingConfig,
    progress_cb: Option<ProgressCallback>,
) -> Result<Mat> {
    if image_paths.is_empty() {
        return Err(anyhow::anyhow!("No images to stack"));
    }
    if image_paths.len() == 1 {
        return load_image(&image_paths[0]);
    }

    let batch_size = config.stacking_batch_size;
    const OVERLAP: usize = 2;

    if image_paths.len() <= batch_size {
        println!("Loading {} images for direct stacking...", image_paths.len());
        
        if let Some(ref cb) = progress_cb {
            if let Ok(mut cb_lock) = cb.lock() {
                let pct = 10.0 + (level as f32 * 10.0).min(80.0);
                cb_lock(format!("Stacking {} images (level {})...", image_paths.len(), level), pct);
            }
        }

        // Load images in parallel for better performance
        let images: Vec<Result<Mat>> = image_paths
            .par_iter()
            .map(|path| load_image(path))
            .collect();
        
        let mut valid_images = Vec::new();
        for (idx, img_result) in images.into_iter().enumerate() {
            match img_result {
                Ok(img) => valid_images.push(img),
                Err(e) => log::warn!("Failed to load image {}: {}", idx, e),
            }
        }
        
        if valid_images.is_empty() {
            return Err(anyhow::anyhow!("No valid images to stack"));
        }
        
        return stack_images_direct(&valid_images);
    }

    let bunches_dir = output_dir.join("bunches");
    std::fs::create_dir_all(&bunches_dir)?;

    let mut intermediate_files = Vec::new();
    let step = batch_size - OVERLAP;
    let mut i = 0;
    let mut batch_idx = 0;
    let mut overlapping_images: Vec<Mat> = Vec::new();
    
    // Calculate total batches for progress reporting
    let total_batches = ((image_paths.len() as f32 - OVERLAP as f32) / step as f32).ceil() as usize;

    while i < image_paths.len() {
        let end = (i + batch_size).min(image_paths.len());
        let batch_paths = &image_paths[i..end];

        log::info!(
            "Level {}: Stacking batch {} (images {} to {})",
            level,
            batch_idx,
            i,
            end - 1
        );
        println!("  Level {}: Batch {} (images {}-{})...", level, batch_idx, i, end - 1);
        
        // Report progress for this batch
        if let Some(ref cb) = progress_cb {
            if let Ok(mut cb_lock) = cb.lock() {
                let batch_progress = (batch_idx as f32 / total_batches as f32) * 70.0; // 0-70%
                let pct = 10.0 + batch_progress;
                cb_lock(format!("Stacking batch {}/{} (level {})...", batch_idx + 1, total_batches, level), pct);
            }
        }

        let mut batch_images = overlapping_images;
        // Only load images that are not already in memory from the previous batch
        let start_load = if batch_idx == 0 { 0 } else { OVERLAP };
        
        // Parallel load of new images
        let new_images: Vec<Result<Mat>> = batch_paths[start_load..]
            .par_iter()
            .map(|path| load_image(path))
            .collect();
        
        for img_result in new_images {
            batch_images.push(img_result?);
        }

        let result = stack_images_direct(&batch_images)?;

        let filename = format!("L{}_B{:04}.png", level, batch_idx);
        let path = bunches_dir.join(&filename);

        opencv::imgcodecs::imwrite(
            path.to_str().unwrap(),
            &result,
            &opencv::core::Vector::new(),
        )?;
        println!("    ✓ Saved {}", filename);

        intermediate_files.push(path);
        batch_idx += 1;

        if end == image_paths.len() {
            break;
        }

        // Keep only the last OVERLAP images for the next batch
        overlapping_images = batch_images.drain(batch_images.len() - OVERLAP..).collect();
        // batch_images is now empty (or contains what's left after drain) and will be dropped

        i += step;
    }

    // Report progress before recursive stacking
    if let Some(ref cb) = progress_cb {
        if let Ok(mut cb_lock) = cb.lock() {
            let pct = 80.0 + (level as f32 * 5.0).min(15.0);
            cb_lock(format!("Combining {} bunches (level {})...", intermediate_files.len(), level + 1), pct);
        }
    }

    // Recursively stack the intermediate results
    stack_recursive(&intermediate_files, output_dir, level + 1, config, progress_cb)
}

fn stack_images_direct(images: &[Mat]) -> Result<Mat> {
    if images.is_empty() {
        return Err(anyhow::anyhow!("No images to stack"));
    }
    if images.len() == 1 {
        return Ok(images[0].clone());
    }

    let levels = 6; // Increased pyramid levels for better detail
    let mut fused_pyramid: Vec<core::UMat> = Vec::new();
    let mut max_energies: Vec<core::UMat> = Vec::new();

    for (idx, img) in images.iter().enumerate() {
        log::info!("Processing image {}/{} for stacking", idx + 1, images.len());
        let mut float_img = core::UMat::new(core::UMatUsageFlags::USAGE_DEFAULT);
        img.get_umat(
            core::AccessFlag::ACCESS_READ,
            core::UMatUsageFlags::USAGE_DEFAULT,
        )?
        .convert_to(&mut float_img, core::CV_32F, 1.0, 0.0)?;

        let current_pyramid = generate_laplacian_pyramid(&float_img, levels)?;

        if idx == 0 {
            // Initialize fused pyramid with the first image's pyramid
            fused_pyramid = current_pyramid.clone();

            // Initialize max energies for Laplacian levels
            for l in 0..levels as usize {
                let layer = &current_pyramid[l];

                // Compute initial energy using Laplacian for better focus detection
                let mut gray = core::UMat::new(core::UMatUsageFlags::USAGE_DEFAULT);
                if layer.channels() == 3 {
                    imgproc::cvt_color(
                        layer,
                        &mut gray,
                        imgproc::COLOR_BGR2GRAY,
                        0,
                        core::AlgorithmHint::ALGO_HINT_DEFAULT,
                    )?;
                } else {
                    gray = layer.clone();
                }

                let mut laplacian = core::UMat::new(core::UMatUsageFlags::USAGE_DEFAULT);
                imgproc::laplacian(
                    &gray,
                    &mut laplacian,
                    core::CV_32F,
                    3,
                    1.0,
                    0.0,
                    core::BORDER_DEFAULT,
                )?;

                let mut energy = core::UMat::new(core::UMatUsageFlags::USAGE_DEFAULT);
                core::absdiff(&laplacian, &core::Scalar::all(0.0), &mut energy)?;

                let mut blurred_energy = core::UMat::new(core::UMatUsageFlags::USAGE_DEFAULT);
                imgproc::gaussian_blur(
                    &energy,
                    &mut blurred_energy,
                    core::Size::new(5, 5),
                    0.0,
                    0.0,
                    core::BORDER_DEFAULT,
                    core::AlgorithmHint::ALGO_HINT_DEFAULT,
                )?;

                max_energies.push(blurred_energy);
            }

            // For the base level (Gaussian), we'll use it for averaging later
            let base_idx = levels as usize;
            let mut float_base = core::UMat::new(core::UMatUsageFlags::USAGE_DEFAULT);
            fused_pyramid[base_idx].convert_to(&mut float_base, core::CV_32F, 1.0, 0.0)?;
            fused_pyramid[base_idx] = float_base;
        } else {
            // Fuse with current image
            for l in 0..levels as usize {
                let layer = &current_pyramid[l];

                // Compute energy using Laplacian
                let mut gray = core::UMat::new(core::UMatUsageFlags::USAGE_DEFAULT);
                if layer.channels() == 3 {
                    imgproc::cvt_color(
                        layer,
                        &mut gray,
                        imgproc::COLOR_BGR2GRAY,
                        0,
                        core::AlgorithmHint::ALGO_HINT_DEFAULT,
                    )?;
                } else {
                    gray = layer.clone();
                }

                let mut laplacian = core::UMat::new(core::UMatUsageFlags::USAGE_DEFAULT);
                imgproc::laplacian(
                    &gray,
                    &mut laplacian,
                    core::CV_32F,
                    3,
                    1.0,
                    0.0,
                    core::BORDER_DEFAULT,
                )?;

                let mut energy = core::UMat::new(core::UMatUsageFlags::USAGE_DEFAULT);
                core::absdiff(&laplacian, &core::Scalar::all(0.0), &mut energy)?;

                let mut blurred_energy = core::UMat::new(core::UMatUsageFlags::USAGE_DEFAULT);
                imgproc::gaussian_blur(
                    &energy,
                    &mut blurred_energy,
                    core::Size::new(5, 5),
                    0.0,
                    0.0,
                    core::BORDER_DEFAULT,
                    core::AlgorithmHint::ALGO_HINT_DEFAULT,
                )?;

                // Update fused layer where energy is higher
                let mut mask = core::UMat::new(core::UMatUsageFlags::USAGE_DEFAULT);
                core::compare(&blurred_energy, &max_energies[l], &mut mask, core::CMP_GT)?;

                layer.copy_to_masked(&mut fused_pyramid[l], &mask)?;
                blurred_energy.copy_to_masked(&mut max_energies[l], &mask)?;
            }

            // Accumulate base level for averaging
            let base_idx = levels as usize;
            let mut float_base = core::UMat::new(core::UMatUsageFlags::USAGE_DEFAULT);
            current_pyramid[base_idx].convert_to(&mut float_base, core::CV_32F, 1.0, 0.0)?;

            let mut next_fused_base = core::UMat::new(core::UMatUsageFlags::USAGE_DEFAULT);
            core::add(
                &fused_pyramid[base_idx],
                &float_base,
                &mut next_fused_base,
                &core::UMat::new(core::UMatUsageFlags::USAGE_DEFAULT),
                -1,
            )?;
            fused_pyramid[base_idx] = next_fused_base;
        }
    }

    log::info!("Collapsing pyramid...");
    // Finalize base level averaging
    let base_idx = levels as usize;
    let mut final_base = core::UMat::new(core::UMatUsageFlags::USAGE_DEFAULT);
    fused_pyramid[base_idx].convert_to(&mut final_base, -1, 1.0 / images.len() as f64, 0.0)?;
    fused_pyramid[base_idx] = final_base;

    // 3. Collapse Pyramid
    let result_umat = collapse_pyramid(&fused_pyramid)?;

    let mut final_img_umat = core::UMat::new(core::UMatUsageFlags::USAGE_DEFAULT);
    result_umat.convert_to(&mut final_img_umat, core::CV_8U, 1.0, 0.0)?;

    let mut final_img = Mat::default();
    final_img_umat
        .get_mat(core::AccessFlag::ACCESS_READ)?
        .copy_to(&mut final_img)?;
    log::info!("Stacking batch complete");
    Ok(final_img)
}

fn generate_laplacian_pyramid(img: &core::UMat, levels: i32) -> Result<Vec<core::UMat>> {
    let mut current = img.clone();
    let mut pyramid = Vec::new();

    for _ in 0..levels {
        let mut down = core::UMat::new(core::UMatUsageFlags::USAGE_DEFAULT);
        imgproc::pyr_down(
            &current,
            &mut down,
            core::Size::default(),
            core::BORDER_DEFAULT,
        )?;

        let mut up = core::UMat::new(core::UMatUsageFlags::USAGE_DEFAULT);
        imgproc::pyr_up(&down, &mut up, current.size()?, core::BORDER_DEFAULT)?;

        // Resize up if needed (due to odd dimensions)
        if up.size()? != current.size()? {
            let mut resized = core::UMat::new(core::UMatUsageFlags::USAGE_DEFAULT);
            imgproc::resize(
                &up,
                &mut resized,
                current.size()?,
                0.0,
                0.0,
                imgproc::INTER_LINEAR,
            )?;
            up = resized;
        }

        let mut lap = core::UMat::new(core::UMatUsageFlags::USAGE_DEFAULT);
        core::subtract(
            &current,
            &up,
            &mut lap,
            &core::UMat::new(core::UMatUsageFlags::USAGE_DEFAULT),
            -1,
        )?;
        pyramid.push(lap);

        current = down;
    }
    pyramid.push(current); // Last level is Gaussian
    Ok(pyramid)
}

fn collapse_pyramid(pyramid: &[core::UMat]) -> Result<core::UMat> {
    let mut current = pyramid.last().unwrap().clone();

    for i in (0..pyramid.len() - 1).rev() {
        let mut up: core::UMat = core::UMat::new(core::UMatUsageFlags::USAGE_DEFAULT);
        imgproc::pyr_up(&current, &mut up, pyramid[i].size()?, core::BORDER_DEFAULT)?;

        if up.size()? != pyramid[i].size()? {
            let mut resized: core::UMat = core::UMat::new(core::UMatUsageFlags::USAGE_DEFAULT);
            imgproc::resize(
                &up,
                &mut resized,
                pyramid[i].size()?,
                0.0,
                0.0,
                imgproc::INTER_LINEAR,
            )?;
            up = resized;
        }

        let mut next: core::UMat = core::UMat::new(core::UMatUsageFlags::USAGE_DEFAULT);
        core::add(
            &up,
            &pyramid[i],
            &mut next,
            &core::UMat::new(core::UMatUsageFlags::USAGE_DEFAULT),
            -1,
        )?;
        current = next;
    }
    Ok(current)
}

#[cfg(test)]
mod tests {
    use super::*;
    use opencv::core::{Scalar, Size};

    #[test]
    fn test_stack_images_synthetic() -> Result<()> {
        // Create two images: one focused on left, one on right
        let mut img1 = Mat::new_rows_cols_with_default(100, 100, core::CV_8UC3, Scalar::all(0.0))?;
        let mut img2 = Mat::new_rows_cols_with_default(100, 100, core::CV_8UC3, Scalar::all(0.0))?;

        // Draw white circle on left of img1 (focused)
        imgproc::circle(
            &mut img1,
            core::Point::new(25, 50),
            20,
            Scalar::all(255.0),
            -1,
            imgproc::LINE_8,
            0,
        )?;
        // Blur right side of img1
        imgproc::gaussian_blur(
            &img1.clone(),
            &mut img1,
            Size::new(15, 15),
            0.0,
            0.0,
            core::BORDER_DEFAULT,
            core::AlgorithmHint::ALGO_HINT_DEFAULT,
        )?;
        // Redraw sharp circle on left
        imgproc::circle(
            &mut img1,
            core::Point::new(25, 50),
            20,
            Scalar::all(255.0),
            -1,
            imgproc::LINE_8,
            0,
        )?;

        // Draw white circle on right of img2 (focused)
        imgproc::circle(
            &mut img2,
            core::Point::new(75, 50),
            20,
            Scalar::all(255.0),
            -1,
            imgproc::LINE_8,
            0,
        )?;
        // Blur left side of img2
        imgproc::gaussian_blur(
            &img2.clone(),
            &mut img2,
            Size::new(15, 15),
            0.0,
            0.0,
            core::BORDER_DEFAULT,
            core::AlgorithmHint::ALGO_HINT_DEFAULT,
        )?;
        // Redraw sharp circle on right
        imgproc::circle(
            &mut img2,
            core::Point::new(75, 50),
            20,
            Scalar::all(255.0),
            -1,
            imgproc::LINE_8,
            0,
        )?;

        let temp_dir = std::env::temp_dir().join("imagestacker_test_synthetic");
        std::fs::create_dir_all(&temp_dir)?;

        let mut paths = Vec::new();
        let p1 = temp_dir.join("img1.png");
        let p2 = temp_dir.join("img2.png");
        opencv::imgcodecs::imwrite(p1.to_str().unwrap(), &img1, &opencv::core::Vector::new())?;
        opencv::imgcodecs::imwrite(p2.to_str().unwrap(), &img2, &opencv::core::Vector::new())?;
        paths.push(p1);
        paths.push(p2);

        let result = stack_images(&paths, &temp_dir, None)?;

        assert!(!result.empty());
        assert_eq!(result.size()?, Size::new(100, 100));

        // Check that both circles are present (roughly)
        // Center left (25, 50) should be bright
        let p1 = result.at_2d::<core::Vec3b>(50, 25)?;
        assert!(p1[0] > 100);

        // Center right (75, 50) should be bright
        let p2 = result.at_2d::<core::Vec3b>(50, 75)?;
        assert!(p2[0] > 100);

        Ok(())
    }
    #[test]
    fn test_stack_images_batched() -> Result<()> {
        // Create 12 images (batch size 10, overlap 2 -> 2 batches)
        let mut images = Vec::new();
        for _ in 0..12 {
            let mut img =
                Mat::new_rows_cols_with_default(100, 100, core::CV_8UC3, Scalar::all(0.0))?;

            // Draw a circle
            imgproc::circle(
                &mut img,
                core::Point::new(50, 50),
                20,
                Scalar::all(255.0),
                -1,
                imgproc::LINE_8,
                0,
            )?;

            images.push(img);
        }

        let temp_dir = std::env::temp_dir().join("imagestacker_test_batched");
        std::fs::create_dir_all(&temp_dir)?;

        let mut paths = Vec::new();
        for (idx, img) in images.iter().enumerate() {
            let p = temp_dir.join(format!("img_{:04}.png", idx));
            opencv::imgcodecs::imwrite(p.to_str().unwrap(), img, &opencv::core::Vector::new())?;
            paths.push(p);
        }

        let result = stack_images(&paths, &temp_dir, None)?;

        assert!(!result.empty());
        assert_eq!(result.size()?, Size::new(100, 100));

        // Check center brightness
        let p = result.at_2d::<core::Vec3b>(50, 50)?;
        assert!(p[0] > 200);

        Ok(())
    }
}
