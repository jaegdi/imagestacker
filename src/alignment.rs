use anyhow::Result;
use opencv::prelude::*;
use opencv::{calib3d, core, features2d, imgproc};
use rayon::prelude::*;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicBool, Ordering};

use crate::config::{FeatureDetector, ProcessingConfig};
use crate::image_io::load_image;
use crate::sharpness;

/// Progress callback: (message, percentage)
pub type ProgressCallback = Arc<Mutex<dyn FnMut(String, f32) + Send>>;

/// Extract features from an image using the specified detector
pub fn extract_features(
    img: &Mat,
    detector_type: FeatureDetector,
) -> Result<(opencv::core::Vector<core::KeyPoint>, core::Mat)> {
    let mut keypoints = opencv::core::Vector::new();
    let mut descriptors = core::Mat::default();

    match detector_type {
        FeatureDetector::ORB => {
            let mut orb = features2d::ORB::create(
                5000, // Increased from 3000 for more features and better alignment
                1.2,  // scaleFactor
                8,    // nlevels - multi-scale detection
                10,   // edgeThreshold - reduced from 15 for more edge features
                0,    // firstLevel
                2,    // WTA_K
                features2d::ORB_ScoreType::HARRIS_SCORE,
                31, // patchSize
                5,  // fastThreshold - reduced from 10 for more sensitive detection
            )?;
            orb.detect_and_compute(
                img,
                &core::Mat::default(),
                &mut keypoints,
                &mut descriptors,
                false,
            )?;
        }
        FeatureDetector::SIFT => {
            // SIFT: Scale-Invariant Feature Transform - best quality, slower
            // Limit to 3000 features to prevent memory issues on very large images
            let mut sift = features2d::SIFT::create(
                3000,  // nfeatures - limit to 3000 to prevent out-of-memory (128-dim float descriptors = 512 bytes each)
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
        FeatureDetector::AKAZE => {
            // AKAZE: Accelerated-KAZE - good balance of speed and quality
            // Very conservative settings for large images to prevent OOM
            let mut akaze = features2d::AKAZE::create(
                features2d::AKAZE_DescriptorType::DESCRIPTOR_MLDB,
                0,      // descriptor_size
                3,      // descriptor_channels
                0.003,  // threshold - increased to reduce number of keypoints
                3,      // nOctaves - reduced from 4 to save memory
                4,      // nOctaveLayers
                features2d::KAZE_DiffusivityType::DIFF_PM_G2,
                64,     // max_points - reduced from 128 to prevent OOM on very large images
            )?;
            akaze.detect_and_compute(
                img,
                &core::Mat::default(),
                &mut keypoints,
                &mut descriptors,
                false,
            )?;
            
            // AKAZE can detect many more keypoints than we need, especially on large images
            // Limit to top 3000 keypoints by response to prevent memory issues
            if keypoints.len() > 3000 {
                use opencv::core::KeyPointTraitConst;
                
                // Sort keypoints by response (strength) in descending order
                let mut kp_vec: Vec<_> = keypoints.to_vec();
                kp_vec.sort_by(|a, b| b.response().partial_cmp(&a.response()).unwrap_or(std::cmp::Ordering::Equal));
                
                // Keep only top 3000
                kp_vec.truncate(3000);
                
                // Rebuild keypoints and descriptors
                keypoints = opencv::core::Vector::from_iter(kp_vec.iter().cloned());
                
                // Extract corresponding descriptor rows
                let mut new_descriptors = core::Mat::default();
                for (i, _) in kp_vec.iter().enumerate() {
                    let row = descriptors.row(i as i32)?;
                    if new_descriptors.empty() {
                        new_descriptors = row.try_clone()?;
                    } else {
                        let mut combined = core::Mat::default();
                        core::vconcat2(&new_descriptors, &row, &mut combined)?;
                        new_descriptors = combined;
                    }
                }
                descriptors = new_descriptors;
            }
        }
    }

    Ok((keypoints, descriptors))
}

pub fn align_images(
    image_paths: &[PathBuf],
    output_dir: &std::path::Path,
    config: &ProcessingConfig,
    progress_cb: Option<ProgressCallback>,
    cancel_flag: Option<Arc<AtomicBool>>,
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

    // Filter images by sharpness using regional analysis
    log::info!("Checking image sharpness for {} images (regional analysis)...", image_paths.len());
    println!("\n=== BLUR DETECTION STARTING (Regional Analysis) ===");
    println!("Analyzing {} images for sharpness in {}x{} grid regions (parallel batches)...", 
             image_paths.len(), config.sharpness_grid_size, config.sharpness_grid_size);
    println!("Strategy: Accept image if ANY region is sharp (good for focus stacking)");

    report_progress("Analyzing image sharpness...", 5.0);

    // Process images in batches to avoid excessive memory usage
    let batch_size = config.batch_config.sharpness_batch_size;
    let mut sharpness_scores: Vec<(usize, PathBuf, f64)> = Vec::new();

    let total_batches = (image_paths.len() + batch_size - 1) / batch_size;
    for (batch_idx, batch) in image_paths.chunks(batch_size).enumerate() {
        // Check for cancellation
        if let Some(ref flag) = cancel_flag {
            if flag.load(Ordering::Relaxed) {
                log::info!("✋ Alignment cancelled by user during sharpness detection (batch {})", batch_idx);
                return Err(anyhow::anyhow!("Operation cancelled by user"));
            }
        }
        
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
                let filename = path.file_name().unwrap_or_default().to_string_lossy().to_string();
                match load_image(path) {
                    Ok(img) => {
                        // Use configured grid size for regional analysis
                        match sharpness::compute_regional_sharpness(&img, config.sharpness_grid_size) {
                            Ok((max_regional, global, sharp_count)) => {
                                log::info!(
                                    "Image {} ({}): max_regional={:.2}, global={:.2}, sharp_regions={}",
                                    idx, filename, max_regional, global, sharp_count
                                );
                                println!(
                                    "    [{}] {}: max_region={:.2}, global={:.2}, sharp_areas={}",
                                    idx, filename, max_regional, global, sharp_count
                                );
                                // Use max regional sharpness (best sharp region)
                                Some((idx, path.clone(), max_regional))
                            }
                            Err(e) => {
                                log::warn!("Failed to compute sharpness for {}: {}", filename, e);
                                println!("    [{}] {}: ERROR computing sharpness: {}", idx, filename, e);
                                None
                            }
                        }
                    }
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

    // For focus stacking, use an adaptive threshold
    // Each image contributes sharp details from its focus plane, even if overall blurry
    // But we still need to filter out truly blurry/defocused images
    let absolute_threshold = config.sharpness_threshold as f64;
    let iqr = q3 - q1;
    
    // Use Q1 - IQR_multiplier * IQR for outlier detection
    // Standard value is 1.5 (standard outlier detection)
    // Higher values (e.g., 3.0) are more permissive
    let iqr_multiplier = config.sharpness_iqr_multiplier as f64;
    let outlier_threshold = q1 - iqr_multiplier * iqr;
    let dynamic_threshold = outlier_threshold.max(absolute_threshold);
    
    let stats_msg = format!(
        "Sharpness statistics:\n  Mean: {:.2}\n  Median: {:.2}\n  StdDev: {:.2}\n  Q1: {:.2}\n  Q3: {:.2}\n  IQR: {:.2}\n  Min: {:.2}\n  Max: {:.2}\n  Outlier threshold: {:.2} (Q1 - {:.1}*IQR = {:.2} - {:.1}*{:.2})\n  Absolute threshold: {:.2}\n  Final threshold: {:.2} ({})",
        mean_sharpness, median, stddev, q1, q3, iqr,
        sorted_scores[0], sorted_scores[sorted_scores.len() - 1],
        outlier_threshold, iqr_multiplier, q1, iqr_multiplier, iqr,
        absolute_threshold,
        dynamic_threshold,
        if dynamic_threshold == absolute_threshold { "using ABSOLUTE" } else { "using OUTLIER" }
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

    let common_mask = Mat::new_rows_cols_with_default(
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
        FeatureDetector::ORB => "ORB (Fast)",
        FeatureDetector::SIFT => "SIFT (Best Quality)",
        FeatureDetector::AKAZE => "AKAZE (Balanced)",
    });

    report_progress("Feature extraction starting...", 20.0);

    let start_matching = std::time::Instant::now();

    // Alignment scale: higher resolution for more accurate alignment
    const ALIGNMENT_SCALE: f64 = 0.85; // Increased from 0.7 to reduce ghosting
    
    // Adjust batch size based on feature detector type to prevent out-of-memory
    // SIFT and AKAZE use much more memory per image than ORB
    // For very large images (>40MP), both need very small batches
    let base_feature_batch_size = config.batch_config.feature_batch_size;
    let feature_batch_size = match config.feature_detector {
        FeatureDetector::ORB => base_feature_batch_size,
        FeatureDetector::SIFT => (base_feature_batch_size / 4).max(3), // SIFT uses ~4x memory (128-dim float descriptors)
        FeatureDetector::AKAZE => (base_feature_batch_size / 4).max(3), // AKAZE uses ~4x memory with very large images
    };
    
    log::info!(
        "Using {} detector with batch size {} (base: {})",
        match config.feature_detector {
            FeatureDetector::ORB => "ORB",
            FeatureDetector::SIFT => "SIFT",
            FeatureDetector::AKAZE => "AKAZE",
        },
        feature_batch_size,
        base_feature_batch_size
    );
    
    let mut pairwise_transforms = Vec::new();
    let mut pairwise_image_indices = Vec::new(); // Track which image each transform corresponds to
    let mut last_batch_features: Option<(Vec<core::KeyPoint>, core::Mat, f64)> = None;

    let total_feature_batches = (sharp_image_paths.len() + feature_batch_size) / feature_batch_size;
    let mut feature_batch_count = 0;

    // Process sharp images in batches for memory efficiency with parallel processing within batches
    for batch_start in (0..sharp_image_paths.len()).step_by(feature_batch_size) {
        // Check for cancellation
        if let Some(ref flag) = cancel_flag {
            if flag.load(Ordering::Relaxed) {
                log::info!("Alignment cancelled by user");
                return Err(anyhow::anyhow!("Operation cancelled by user"));
            }
        }
        
        let is_first_batch = batch_start == 0;
        
        // First batch loads batch_size+1 images to create overlap
        // Non-first batches load batch_size images starting from batch_start+1
        // (because batch_start is the overlapping image from previous batch)
        let load_start = if is_first_batch { batch_start } else { batch_start + 1 };
        
        // Stop if we've already processed all images
        if load_start >= sharp_image_paths.len() {
            break;
        }
        
        let load_count = if is_first_batch { 
            feature_batch_size + 1 
        } else { 
            feature_batch_size 
        };
        let batch_end = (load_start + load_count).min(sharp_image_paths.len());
        
        log::info!(
            "DEBUG: batch_start={}, is_first_batch={}, load_start={}, load_count={}, batch_end={}",
            batch_start, is_first_batch, load_start, load_count, batch_end
        );
        
        let batch_paths: Vec<&PathBuf> = sharp_image_paths[load_start..batch_end]
            .iter()
            .map(|(_, path)| path)
            .collect();

        feature_batch_count += 1;
        let progress_pct = 20.0 + (feature_batch_count as f32 / total_feature_batches as f32) * 30.0;
        report_progress(&format!("Feature extraction: batch {}/{}", feature_batch_count, total_feature_batches), progress_pct);

        log::info!(
            "Extracting features for batch {}-{} of {} (batch size: {})",
            load_start,
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
        
        // For non-first batches, prepend the last image's features from previous batch
        // to maintain feature consistency across batch boundaries
        if let Some(prev_features) = last_batch_features.take() {
            valid_batch_features.push(prev_features);
        }
        
        for f in batch_features {
            valid_batch_features.push(f?);
        }
        
        // Save the last image's features for the next batch (if not the last batch)
        if batch_end < sharp_image_paths.len() {
            last_batch_features = Some(valid_batch_features.last().unwrap().clone());
        }

        // Compute pairwise transforms for consecutive pairs in this batch
        for i in 0..valid_batch_features.len() - 1 {
            // Check for cancellation
            if let Some(ref flag) = cancel_flag {
                if flag.load(Ordering::Relaxed) {
                    log::info!("Alignment cancelled by user during pairwise matching");
                    return Err(anyhow::anyhow!("Operation cancelled by user"));
                }
            }
            
            let (ref prev_keypoints, ref prev_descriptors, prev_scale) = valid_batch_features[i];
            let (ref curr_keypoints, ref curr_descriptors, curr_scale) =
                valid_batch_features[i + 1];

            // Calculate actual sharp image indices for logging
            let actual_prev_idx = if i == 0 && !is_first_batch {
                // First feature in non-first batch is the saved feature from previous batch
                load_start - 1
            } else if is_first_batch {
                i
            } else {
                load_start + i - 1
            };
            let actual_curr_idx = if is_first_batch {
                i + 1
            } else {
                load_start + i
            };

            let t_step_2x3 = if curr_descriptors.empty() || prev_descriptors.empty() {
                Mat::default()
            } else {
                // Choose matcher based on feature detector type
                let norm_type = match config.feature_detector {
                    FeatureDetector::ORB => core::NORM_HAMMING,      // Binary descriptors
                    FeatureDetector::SIFT => core::NORM_L2,          // Float descriptors
                    FeatureDetector::AKAZE => core::NORM_HAMMING,    // Binary descriptors
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

                // Use top 50% of matches (increased from 40% for better alignment precision)
                // More matches = more accurate transform estimation
                let count = (matches_vec.len() as f32 * 0.5) as usize;
                let count = count.max(20).min(matches_vec.len()); // Increased min from 15 to 20
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
                    // Use estimateAffinePartial2D for rigid transform (rotation + translation + uniform scale)
                    // This is more robust than homography for image alignment
                    let mut inliers = core::Mat::default();
                    let transform = calib3d::estimate_affine_partial_2d(
                        &src_pts,
                        &dst_pts,
                        &mut inliers,
                        calib3d::RANSAC,
                        1.5, // ransacReprojThreshold - reduced from 3.0 for tighter alignment
                        5000, // maxIters - increased from 2000 for better convergence
                        0.995, // confidence - increased from 0.99 for more reliable estimates
                        20, // refineIters - increased from 10 for subpixel accuracy
                    )?;

                    if !transform.empty() {
                        let mut t_step_2x3_f32 = Mat::default();
                        transform.convert_to(
                            &mut t_step_2x3_f32,
                            core::CV_32F,
                            1.0,
                            0.0,
                        )?;
                        t_step_2x3_f32
                    } else {
                        Mat::default()
                    }
                } else {
                    Mat::default()
                }
            };

            if !t_step_2x3.empty() {
                // Validate the transform: check if it's degenerate
                // A valid affine transform should have non-zero determinant
                // For 2x3 affine: [[a, b, tx], [c, d, ty]], determinant = a*d - b*c
                let t_data = t_step_2x3.data_typed::<f32>()?;
                let a = t_data[0];
                let b = t_data[1];
                let c = t_data[3];
                let d = t_data[4];
                let determinant = a * d - b * c;
                
                // Check if transform is valid (determinant should be close to 1.0 for rigid/similarity transforms)
                // Allow some tolerance but reject degenerate transforms (det near 0)
                if determinant.abs() > 0.1 && determinant.abs() < 10.0 {
                    let pair_idx = pairwise_transforms.len();
                    log::info!("  Pairwise[{}]: sharp_img[{}] -> sharp_img[{}] (det={:.4})", 
                        pair_idx, actual_curr_idx, actual_prev_idx, determinant);
                    pairwise_transforms.push(t_step_2x3);
                    pairwise_image_indices.push(actual_curr_idx);
                } else {
                    log::warn!("  REJECTED (degenerate): sharp_img[{}] -> sharp_img[{}] (det={:.6}, a={:.4}, b={:.4}, c={:.4}, d={:.4})", 
                        actual_curr_idx, actual_prev_idx, determinant, a, b, c, d);
                }
            } else {
                log::warn!("  FAILED: sharp_img[{}] -> sharp_img[{}]", actual_curr_idx, actual_prev_idx);
            }
        }
    }

    // 2. Accumulate Transforms to Reference Frame
    log::info!("Accumulating {} pairwise transforms to reference frame", pairwise_transforms.len());
    println!("\nAccumulating transforms to reference frame...");

    let mut accumulated_transforms = Vec::new();
    
    // Create a map from image index to transform
    let mut transform_map: std::collections::HashMap<usize, Mat> = std::collections::HashMap::new();
    for (transform, img_idx) in pairwise_transforms.iter().zip(pairwise_image_indices.iter()) {
        transform_map.insert(*img_idx, transform.clone());
    }
    
    // Start with identity transform (3x3 for proper composition)
    let mut current_transform_3x3 = Mat::eye(3, 3, core::CV_32F)?.to_mat()?;

    // Process images 1 to N (image 0 is reference)
    for img_idx in 1..sharp_image_paths.len() {
        // Check for cancellation
        if let Some(ref flag) = cancel_flag {
            if flag.load(Ordering::Relaxed) {
                log::info!("Alignment cancelled by user during transform accumulation");
                return Err(anyhow::anyhow!("Operation cancelled by user"));
            }
        }
        
        // Check if we have a pairwise transform for this image
        if let Some(t_step) = transform_map.get(&img_idx) {
            // We have a transform: compose it with the previous accumulated transform
            // Convert 2x3 affine to 3x3 homogeneous matrix
            let t_data = t_step.data_typed::<f32>()?;
            let t_3x3_data = [
                t_data[0], t_data[1], t_data[2],
                t_data[3], t_data[4], t_data[5],
                0.0, 0.0, 1.0,
            ];
            let t_step_3x3 = Mat::from_slice_2d::<f32>(&t_3x3_data.chunks(3).collect::<Vec<_>>())?;

            // Compose transforms: accumulated[i] = accumulated[i-1] * transform[i->i-1]
            // This builds the chain: image[i] -> image[i-1] -> ... -> reference
            // Matrix multiplication order: the transform applied FIRST goes on the RIGHT
            let mut new_transform_3x3 = Mat::default();
            let no_array = Mat::default();
            core::gemm(
                &current_transform_3x3,
                &t_step_3x3,
                1.0,
                &no_array,
                0.0,
                &mut new_transform_3x3,
                0,
            )?;

            // DEBUG: Log composition
            if img_idx <= 10 {
                let curr_data = current_transform_3x3.data_typed::<f32>()?;
                let step_data = t_step_3x3.data_typed::<f32>()?;
                let new_data_3x3 = new_transform_3x3.data_typed::<f32>()?;
                log::info!("  Compose[img={}]: prev=[[{:.4},{:.4},{:.2}],[{:.4},{:.4},{:.2}]] * step=[[{:.4},{:.4},{:.2}],[{:.4},{:.4},{:.2}]] = [[{:.4},{:.4},{:.2}],[{:.4},{:.4},{:.2}]]",
                    img_idx,
                    curr_data[0], curr_data[1], curr_data[2],
                    curr_data[3], curr_data[4], curr_data[5],
                    step_data[0], step_data[1], step_data[2],
                    step_data[3], step_data[4], step_data[5],
                    new_data_3x3[0], new_data_3x3[1], new_data_3x3[2],
                    new_data_3x3[3], new_data_3x3[4], new_data_3x3[5],
                );
            }

            current_transform_3x3 = new_transform_3x3;
        } else {
            // No pairwise transform found: reset to identity
            // This treats the image as if it's already at reference position (no alignment needed)
            log::warn!("  No transform for image {}, using identity (treating as already aligned to reference)", img_idx);
            current_transform_3x3 = Mat::eye(3, 3, core::CV_32F)?.to_mat()?;
        }
        
        // Convert current 3x3 to 2x3 for storage
        let current_data = current_transform_3x3.data_typed::<f32>()?;
        let affine_2x3_data = [
            current_data[0], current_data[1], current_data[2],
            current_data[3], current_data[4], current_data[5],
        ];
        let affine_2x3 = Mat::from_slice_2d::<f32>(&affine_2x3_data.chunks(3).collect::<Vec<_>>())?;
        
        accumulated_transforms.push(affine_2x3);
    }

    // 3. Parallel Warp and Crop with Batched Processing
    log::info!("Warping {} images with accumulated transforms", accumulated_transforms.len());
    println!("Warping {} images to reference frame...", accumulated_transforms.len());

    report_progress("Warping images to reference frame...", 50.0);

    let warp_batch_size = config.batch_config.warp_batch_size;
    let total_warp_batches = (accumulated_transforms.len() + warp_batch_size - 1) / warp_batch_size;

    for (batch_idx, batch) in accumulated_transforms.chunks(warp_batch_size).enumerate() {
        // Check for cancellation
        if let Some(ref flag) = cancel_flag {
            if flag.load(Ordering::Relaxed) {
                log::info!("Alignment cancelled by user during warping");
                return Err(anyhow::anyhow!("Operation cancelled by user"));
            }
        }
        
        let batch_start = batch_idx * warp_batch_size;
        let progress_pct = 50.0 + (batch_idx as f32 / total_warp_batches as f32) * 40.0;
        report_progress(&format!("Warping: batch {}/{}", batch_idx + 1, total_warp_batches), progress_pct);

        let warp_results: Vec<Result<()>> = (batch_start..batch_start + batch.len())
            .into_par_iter()
            .map(|i| {
                let img_idx = i + 1; // +1 because we skip the reference image
                let img_path = &sharp_image_paths[img_idx].1;
                let transform = &accumulated_transforms[i];
                
                // Debug: log which image is being warped with which transform
                let orig_idx = sharp_image_paths[img_idx].0;
                
                // Log transform matrix for debugging
                let t_data = transform.data_typed::<f32>()?;
                let is_near_identity = (t_data[0] - 1.0).abs() < 0.01 && 
                                      (t_data[1]).abs() < 0.01 &&
                                      (t_data[2]).abs() < 1.0 &&
                                      (t_data[3]).abs() < 0.01 &&
                                      (t_data[4] - 1.0).abs() < 0.01 &&
                                      (t_data[5]).abs() < 1.0;
                                      
                log::info!("  Warp {:04}.png (sharp[{}]): transform[{}] = [[{:.4}, {:.4}, {:.2}], [{:.4}, {:.4}, {:.2}]] {}",
                    orig_idx, img_idx, i,
                    t_data[0], t_data[1], t_data[2],
                    t_data[3], t_data[4], t_data[5],
                    if is_near_identity { "⚠ NEAR-IDENTITY" } else { "" }
                );

                let img = load_image(img_path)?;

                // Warp image to reference frame
                // Use INTER_LANCZOS4 for highest quality interpolation
                
                // Convert to BGRA if needed (to support transparent borders)
                let img_with_alpha = if img.channels() == 3 {
                    let mut bgra = Mat::default();
                    imgproc::cvt_color(&img, &mut bgra, imgproc::COLOR_BGR2BGRA, 0, core::AlgorithmHint::ALGO_HINT_DEFAULT)?;
                    bgra
                } else {
                    img.clone()
                };
                
                let mut warped = Mat::default();
                imgproc::warp_affine(
                    &img_with_alpha,
                    &mut warped,
                    transform,
                    core::Size::new(ref_img.cols(), ref_img.rows()),
                    imgproc::INTER_LANCZOS4, // Best quality interpolation for alignment
                    core::BORDER_CONSTANT,
                    core::Scalar::new(0.0, 0.0, 0.0, 0.0), // Transparent black borders (BGRA)
                )?;

                // Update common mask (intersection of all valid pixels)
                let mut img_mask = Mat::new_rows_cols_with_default(
                    warped.rows(),
                    warped.cols(),
                    core::CV_8U,
                    core::Scalar::all(255.0),
                )?;

                // For images with alpha channel or transparency, use that as mask
                if img.channels() == 4 {
                    let mut alpha = Mat::default();
                    core::extract_channel(&img, &mut alpha, 3)?;
                    img_mask = alpha;
                }

                // Warp the mask too
                let mut warped_mask = Mat::default();
                imgproc::warp_affine(
                    &img_mask,
                    &mut warped_mask,
                    transform,
                    core::Size::new(ref_img.cols(), ref_img.rows()),
                    imgproc::INTER_NEAREST,
                    core::BORDER_CONSTANT,
                    core::Scalar::all(0.0),
                )?;

                // Set alpha channel from warped mask (make borders transparent)
                if warped.channels() == 4 {
                    // Split the BGRA channels
                    let mut channels = opencv::core::Vector::<Mat>::new();
                    core::split(&warped, &mut channels)?;
                    
                    // Replace the alpha channel (index 3) with the warped mask
                    // This makes borders fully transparent (alpha=0) and valid areas opaque (alpha=255)
                    channels.set(3, warped_mask)?;
                    
                    // Merge the channels back
                    core::merge(&channels, &mut warped)?;
                }

                // Update common mask with intersection
                // TODO: Fix parallel mask intersection
                // core::bitwise_and(&common_mask, &warped_mask, &mut common_mask, &core::Mat::default())?;

                // Save aligned image with alpha channel
                let output_path = aligned_dir.join(format!("{:04}.png", sharp_image_paths[img_idx].0));
                
                // Ensure PNG saves with alpha channel
                let mut params = opencv::core::Vector::new();
                params.push(opencv::imgcodecs::IMWRITE_PNG_COMPRESSION);
                params.push(3); // Compression level
                
                opencv::imgcodecs::imwrite(
                    output_path.to_str().unwrap(),
                    &warped,
                    &params,
                )?;

                Ok(())
            })
            .collect();

        // Check for errors in batch
        for result in warp_results {
            result?;
        }
    }

    // 4. Find Common Crop Rectangle
    log::info!("Finding common crop rectangle from mask");
    println!("\nFinding optimal crop rectangle...");

    report_progress("Finding optimal crop area...", 90.0);

    let crop_rect;
    let mut min_row = ref_img.rows();
    let mut max_row = 0;
    let mut min_col = ref_img.cols();
    let mut max_col = 0;

    // Find bounding box of common valid area
    for row in 0..common_mask.rows() {
        // Check for cancellation every 100 rows
        if row % 100 == 0 {
            if let Some(ref flag) = cancel_flag {
                if flag.load(Ordering::Relaxed) {
                    log::info!("Alignment cancelled by user during crop calculation");
                    return Err(anyhow::anyhow!("Operation cancelled by user"));
                }
            }
        }
        
        for col in 0..common_mask.cols() {
            let pixel = common_mask.at_2d::<u8>(row, col)?;
            if *pixel > 0 {
                min_row = min_row.min(row);
                max_row = max_row.max(row);
                min_col = min_col.min(col);
                max_col = max_col.max(col);
            }
        }
    }

    if max_row > min_row && max_col > min_col {
        crop_rect = core::Rect::new(min_col, min_row, max_col - min_col + 1, max_row - min_row + 1);
        log::info!("Common crop rectangle: {}x{} at ({}, {})",
                  crop_rect.width, crop_rect.height, crop_rect.x, crop_rect.y);
        println!("Crop rectangle: {}x{} at ({}, {})",
                crop_rect.width, crop_rect.height, crop_rect.x, crop_rect.y);
    } else {
        log::warn!("No common area found, using full image dimensions");
        crop_rect = core::Rect::new(0, 0, ref_img.cols(), ref_img.rows());
    }

    let elapsed_total = start_total.elapsed();
    let elapsed_matching = start_matching.elapsed();

    log::info!(
        "Alignment completed in {:?} (matching: {:?})",
        elapsed_total, elapsed_matching
    );
    println!("\n✓ Alignment completed in {:?}", elapsed_total);
    println!("  Matching time: {:?}", elapsed_matching);

    report_progress("Alignment completed!", 100.0);

    Ok(crop_rect)
}