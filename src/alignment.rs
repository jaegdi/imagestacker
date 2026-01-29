use anyhow::Result;
use opencv::prelude::*;
use opencv::{calib3d, core, features2d, imgproc};
use rayon::prelude::*;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use crate::config::{FeatureDetector, ProcessingConfig};
use crate::image_io::load_image;
use crate::sharpness::compute_sharpness;

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
        FeatureDetector::AKAZE => {
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
    let batch_size = config.batch_config.sharpness_batch_size;
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
    let feature_batch_size = config.batch_config.feature_batch_size;
    let mut pairwise_transforms = Vec::new();
    let mut last_batch_features: Option<(Vec<core::KeyPoint>, core::Mat, f64)> = None;

    let total_feature_batches = (sharp_image_paths.len() + feature_batch_size) / feature_batch_size;
    let mut feature_batch_count = 0;

    // Process sharp images in batches for memory efficiency with parallel processing within batches
    for batch_start in (0..sharp_image_paths.len()).step_by(feature_batch_size) {
        let is_first_batch = batch_start == 0;
        
        // First batch loads batch_size+1 images to create overlap
        // Non-first batches load batch_size images starting from batch_start+1
        // (because batch_start is the overlapping image from previous batch)
        let load_start = if is_first_batch { batch_start } else { batch_start + 1 };
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
                let pair_idx = pairwise_transforms.len();
                log::info!("  Pairwise[{}]: sharp_img[{}] -> sharp_img[{}]", 
                    pair_idx, actual_curr_idx, actual_prev_idx);
                pairwise_transforms.push(t_step_2x3);
            } else {
                log::warn!("  FAILED: sharp_img[{}] -> sharp_img[{}]", actual_curr_idx, actual_prev_idx);
            }
        }
    }

    // 2. Accumulate Transforms to Reference Frame
    log::info!("Accumulating {} pairwise transforms to reference frame", pairwise_transforms.len());
    println!("\nAccumulating transforms to reference frame...");

    let mut accumulated_transforms = Vec::new();
    
    // Start with identity transform (3x3 for proper composition)
    let mut current_transform_3x3 = Mat::eye(3, 3, core::CV_32F)?.to_mat()?;

    for (_i, t_step) in pairwise_transforms.iter().enumerate() {
        // Convert 2x3 affine to 3x3 homogeneous matrix
        let t_data = t_step.data_typed::<f32>()?;
        let t_3x3_data = [
            t_data[0], t_data[1], t_data[2],
            t_data[3], t_data[4], t_data[5],
            0.0, 0.0, 1.0,
        ];
        let t_step_3x3 = Mat::from_slice_2d::<f32>(&t_3x3_data.chunks(3).collect::<Vec<_>>())?;

        // Compose transforms: new = current * step  
        // Each accumulated transform maps image[i+1] to reference
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
        if _i < 3 {
            let curr_data = current_transform_3x3.data_typed::<f32>()?;
            let step_data = t_step_3x3.data_typed::<f32>()?;
            let new_data_3x3 = new_transform_3x3.data_typed::<f32>()?;
            log::info!("  Compose[{}]: current=[[{:.4},{:.4},{:.2}],[{:.4},{:.4},{:.2}]] * step=[[{:.4},{:.4},{:.2}],[{:.4},{:.4},{:.2}]] = [[{:.4},{:.4},{:.2}],[{:.4},{:.4},{:.2}]]",
                _i,
                curr_data[0], curr_data[1], curr_data[2],
                curr_data[3], curr_data[4], curr_data[5],
                step_data[0], step_data[1], step_data[2],
                step_data[3], step_data[4], step_data[5],
                new_data_3x3[0], new_data_3x3[1], new_data_3x3[2],
                new_data_3x3[3], new_data_3x3[4], new_data_3x3[5],
            );
        }

        // Convert back to 2x3 for storage
        let new_data = new_transform_3x3.data_typed::<f32>()?;
        let affine_2x3_data = [
            new_data[0], new_data[1], new_data[2],
            new_data[3], new_data[4], new_data[5],
        ];
        let affine_2x3 = Mat::from_slice_2d::<f32>(&affine_2x3_data.chunks(3).collect::<Vec<_>>())?;
        
        accumulated_transforms.push(affine_2x3);
        current_transform_3x3 = new_transform_3x3;
    }

    // 3. Parallel Warp and Crop with Batched Processing
    log::info!("Warping {} images with accumulated transforms", accumulated_transforms.len());
    println!("Warping {} images to reference frame...", accumulated_transforms.len());

    report_progress("Warping images to reference frame...", 50.0);

    let warp_batch_size = config.batch_config.warp_batch_size;
    let total_warp_batches = (accumulated_transforms.len() + warp_batch_size - 1) / warp_batch_size;

    for (batch_idx, batch) in accumulated_transforms.chunks(warp_batch_size).enumerate() {
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
                let mut warped = Mat::default();
                imgproc::warp_affine(
                    &img,
                    &mut warped,
                    transform,
                    core::Size::new(ref_img.cols(), ref_img.rows()),
                    imgproc::INTER_LANCZOS4, // Best quality interpolation for alignment
                    core::BORDER_CONSTANT,
                    core::Scalar::all(0.0),
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

                // Update common mask with intersection
                // TODO: Fix parallel mask intersection
                // core::bitwise_and(&common_mask, &warped_mask, &mut common_mask, &core::Mat::default())?;

                // Save aligned image
                let output_path = aligned_dir.join(format!("{:04}.png", sharp_image_paths[img_idx].0));
                opencv::imgcodecs::imwrite(
                    output_path.to_str().unwrap(),
                    &warped,
                    &opencv::core::Vector::new(),
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