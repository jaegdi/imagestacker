use anyhow::Result;
use opencv::prelude::*;
use opencv::{calib3d, core, features2d, imgproc, video};
use rayon::prelude::*;
use std::path::PathBuf;
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use crate::config::{EccMotionType, FeatureDetector, ProcessingConfig};
use crate::image_io::load_image;
use crate::sharpness;
use crate::sharpness_cache::SharpnessInfo;

/// Progress callback: (message, percentage)
pub type ProgressCallback = Arc<Mutex<dyn FnMut(String, f32) + Send>>;

// ---------------------------------------------------------------------------
// GPU concurrency limiter (counting semaphore)
// ---------------------------------------------------------------------------
// Instead of a binary mutex (1 thread at a time) or unlimited parallelism
// (GPU OOM on large images), we allow N concurrent GPU operations.
//
// Controlled by IMAGESTACKER_GPU_CONCURRENCY env var:
//   0  = unlimited (no guard, fastest but may OOM on large images)
//   1  = fully serialized (safest, equivalent to old OPENCL_MUTEX)
//   2+ = bounded parallelism (default: 2, good balance)
//
// The old IMAGESTACKER_OPENCL_MUTEX=1 env var is still honored as a
// shortcut for GPU_CONCURRENCY=1.

/// A counting semaphore for bounding concurrent GPU operations.
pub struct GpuSemaphore {
    state: Mutex<usize>,   // current number of available permits
    condvar: Condvar,
    max_permits: usize,    // 0 means unlimited (no-op)
}

/// RAII guard that releases a GPU permit when dropped.
pub struct GpuPermit<'a> {
    sem: &'a GpuSemaphore,
}

impl GpuSemaphore {
    fn new(max_permits: usize) -> Self {
        Self {
            state: Mutex::new(max_permits),
            condvar: Condvar::new(),
            max_permits,
        }
    }

    /// Acquire a permit. Blocks until one is available.
    /// Returns `None` if concurrency is unlimited (max_permits == 0).
    pub fn acquire(&self) -> Option<GpuPermit<'_>> {
        if self.max_permits == 0 {
            return None; // unlimited — no guard needed
        }
        let mut permits = self.state.lock().unwrap();
        while *permits == 0 {
            permits = self.condvar.wait(permits).unwrap();
        }
        *permits -= 1;
        Some(GpuPermit { sem: self })
    }
}

impl Drop for GpuPermit<'_> {
    fn drop(&mut self) {
        let mut permits = self.sem.state.lock().unwrap();
        *permits += 1;
        self.sem.condvar.notify_one();
    }
}

/// Global GPU semaphore, initialized once from environment variables.
static GPU_SEMAPHORE: OnceLock<GpuSemaphore> = OnceLock::new();

/// Get the global GPU concurrency semaphore.
///
/// Priority (first match wins):
///   1. `IMAGESTACKER_OPENCL_MUTEX=1`  → concurrency = 1 (full serialization)
///   2. `IMAGESTACKER_GPU_CONCURRENCY` → concurrency = env value
///   3. Settings file (`gpu_concurrency`) → concurrency = saved value
///   4. default                        → concurrency = 2
pub fn gpu_semaphore() -> &'static GpuSemaphore {
    GPU_SEMAPHORE.get_or_init(|| {
        let concurrency = if std::env::var("IMAGESTACKER_OPENCL_MUTEX").unwrap_or_default() == "1" {
            log::info!("GPU concurrency: 1 (forced by IMAGESTACKER_OPENCL_MUTEX=1)");
            1
        } else if let Ok(val) = std::env::var("IMAGESTACKER_GPU_CONCURRENCY") {
            let c = val.parse::<usize>().unwrap_or(2);
            log::info!("GPU concurrency: {} (from IMAGESTACKER_GPU_CONCURRENCY env var)", c);
            c
        } else {
            // Read from saved settings
            let config = crate::settings::load_settings();
            log::info!("GPU concurrency: {} (from settings)", config.gpu_concurrency);
            config.gpu_concurrency
        };
        GpuSemaphore::new(concurrency)
    })
}

/// Try to load cached sharpness value from YAML file
/// Returns Some(max_regional_sharpness) if cache exists and is valid, None otherwise
fn load_cached_sharpness(image_path: &PathBuf, output_dir: &std::path::Path) -> Option<f64> {
    let sharpness_dir = output_dir.join("sharpness");
    let yaml_name = SharpnessInfo::yaml_filename_for_image(image_path);
    let yaml_path = sharpness_dir.join(yaml_name);
    
    if yaml_path.exists() {
        match SharpnessInfo::load_from_file(&yaml_path) {
            Ok(info) => {
                log::info!("   ✓ Using cached sharpness for {}: {:.2}", 
                          image_path.file_name().unwrap_or_default().to_string_lossy(),
                          info.max_regional_sharpness);
                Some(info.max_regional_sharpness)
            }
            Err(e) => {
                log::warn!("   ⚠ Failed to load cached sharpness for {}: {}", 
                          image_path.file_name().unwrap_or_default().to_string_lossy(), e);
                None
            }
        }
    } else {
        None
    }
}

/// Extract features from an image using the specified detector
pub fn extract_features(
    img: &Mat,
    detector_type: FeatureDetector,
) -> Result<(opencv::core::Vector<core::KeyPoint>, core::Mat)> {
    let mut keypoints = opencv::core::Vector::new();
    let mut descriptors = core::Mat::default();

    match detector_type {
        FeatureDetector::ECC => {
            // ECC doesn't use feature detection - it works on full image correlation
            // Return empty keypoints and descriptors
            return Ok((keypoints, descriptors));
        }
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
            // Optimized to 2000 features for better speed/quality balance
            let mut sift = features2d::SIFT::create(
                2000,  // nfeatures - reduced from 3000 for ~30% speed improvement
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

/// Convert EccMotionType to OpenCV video::MOTION_* constant
fn ecc_motion_to_opencv(motion: EccMotionType) -> i32 {
    match motion {
        EccMotionType::Translation => video::MOTION_TRANSLATION,
        EccMotionType::Euclidean => video::MOTION_EUCLIDEAN,
        EccMotionType::Affine => video::MOTION_AFFINE,
        EccMotionType::Homography => video::MOTION_HOMOGRAPHY,
    }
}

/// Compute ECC-based transformation matrix between two grayscale images
#[allow(dead_code)]
fn compute_ecc_transform(
    reference: &Mat,
    target: &Mat,
    config: &ProcessingConfig,
) -> Result<Mat> {
    // 1. Convert to grayscale if needed (can run in parallel - OpenCV 4.x is thread-safe for basic ops)
    let ref_gray = if reference.channels() == 1 {
        reference.clone()
    } else {
        let mut gray = Mat::default();
        crate::opencv_compat::cvt_color(reference, &mut gray, imgproc::COLOR_BGR2GRAY, 0)?;
        gray
    };
    
    let tgt_gray = if target.channels() == 1 {
        target.clone()
    } else {
        let mut gray = Mat::default();
        crate::opencv_compat::cvt_color(target, &mut gray, imgproc::COLOR_BGR2GRAY, 0)?;
        gray
    };
    
    // 2. Apply Gaussian blur to reduce noise (can run in parallel)
    let mut ref_blurred = Mat::default();
    let mut tgt_blurred = Mat::default();
    
    let kernel_size = core::Size::new(config.ecc_gauss_filter_size, config.ecc_gauss_filter_size);
    crate::opencv_compat::gaussian_blur(&ref_gray, &mut ref_blurred, kernel_size, 0.0, 0.0, core::BORDER_DEFAULT)?;
    crate::opencv_compat::gaussian_blur(&tgt_gray, &mut tgt_blurred, kernel_size, 0.0, 0.0, core::BORDER_DEFAULT)?;
    
    compute_ecc_transform_internal(&ref_blurred, &tgt_blurred, config)
}

/// Optimized version that accepts pre-processed reference image (grayscale + blurred)
#[allow(dead_code)]
fn compute_ecc_transform_with_ref(
    ref_blurred: &Mat,
    target: &Mat,
    config: &ProcessingConfig,
) -> Result<Mat> {
    // 1. Convert target to grayscale if needed
    let tgt_gray = if target.channels() == 1 {
        target.clone()
    } else {
        let mut gray = Mat::default();
        crate::opencv_compat::cvt_color(target, &mut gray, imgproc::COLOR_BGR2GRAY, 0)?;
        gray
    };
    
    // 2. Apply Gaussian blur to target
    let mut tgt_blurred = Mat::default();
    let kernel_size = core::Size::new(config.ecc_gauss_filter_size, config.ecc_gauss_filter_size);
    crate::opencv_compat::gaussian_blur(&tgt_gray, &mut tgt_blurred, kernel_size, 0.0, 0.0, core::BORDER_DEFAULT)?;
    
    compute_ecc_transform_internal(ref_blurred, &tgt_blurred, config)
}

/// Internal ECC transform computation (common code)
/// Hybrid alignment: Use keypoint matching for initial coarse alignment, 
/// then refine with ECC for subpixel precision
/// This reduces computation time by 40-50% while maintaining quality
fn compute_hybrid_ecc_transform(
    ref_img: &Mat,
    tgt_img: &Mat,
    ref_blurred: &Mat,
    tgt_blurred: &Mat,
    config: &ProcessingConfig,
) -> Result<Mat> {
    // Step 1: Keypoint-based coarse alignment
    // Use SIFT for better precision than AKAZE (worth the extra time for quality)
    let ref_keypoints: opencv::core::Vector<core::KeyPoint>;
    let ref_descriptors: core::Mat;
    let tgt_keypoints: opencv::core::Vector<core::KeyPoint>;
    let tgt_descriptors: core::Mat;
    
    {
        // Feature extraction runs on CPU Mat — no GPU mutex needed
        (ref_keypoints, ref_descriptors) = extract_features(ref_img, FeatureDetector::SIFT)?;
        (tgt_keypoints, tgt_descriptors) = extract_features(tgt_img, FeatureDetector::SIFT)?;
    }
    
    // Match keypoints
    let initial_warp = if !ref_descriptors.empty() && !tgt_descriptors.empty() && ref_keypoints.len() >= 10 && tgt_keypoints.len() >= 10 {
        let mut matcher = features2d::BFMatcher::create(core::NORM_L2, false)?;
        let mut train_descriptors = opencv::core::Vector::<core::Mat>::new();
        train_descriptors.push(ref_descriptors.clone());
        matcher.add(&train_descriptors)?;
        
        let mut matches = opencv::core::Vector::<core::DMatch>::new();
        matcher.match_(&tgt_descriptors, &mut matches, &core::Mat::default())?;
        
        let mut matches_vec = matches.to_vec();
        matches_vec.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
        
        // Use top 30% of matches for more robust initial estimate (stricter filtering)
        let count = ((matches_vec.len() as f32 * 0.3) as usize).max(10).min(matches_vec.len());
        let good_matches = &matches_vec[..count];
        
        let mut src_pts = opencv::core::Vector::<core::Point2f>::new();
        let mut dst_pts = opencv::core::Vector::<core::Point2f>::new();
        
        for m in good_matches {
            src_pts.push(tgt_keypoints.get(m.query_idx as usize)?.pt());
            dst_pts.push(ref_keypoints.get(m.train_idx as usize)?.pt());
        }
        
        if src_pts.len() >= 4 {
            // Compute initial transform based on motion type
            let initial_transform = if config.ecc_motion_type == EccMotionType::Homography {
                // For homography, use findHomography
                let mut inliers = core::Mat::default();
                let h = calib3d::find_homography(
                    &src_pts,
                    &dst_pts,
                    &mut inliers,
                    calib3d::RANSAC,
                    3.0,
                )?;
                
                if !h.empty() {
                    let mut h_f32 = Mat::default();
                    h.convert_to(&mut h_f32, core::CV_32F, 1.0, 0.0)?;
                    Some(h_f32)
                } else {
                    None
                }
            } else {
                // For affine/euclidean/translation, use estimateAffinePartial2D
                let mut inliers = core::Mat::default();
                let transform = calib3d::estimate_affine_partial_2d(
                    &src_pts,
                    &dst_pts,
                    &mut inliers,
                    calib3d::RANSAC,
                    2.0,
                    2000,
                    0.995,
                    10,
                )?;
                
                if !transform.empty() {
                    let mut t_f32 = Mat::default();
                    transform.convert_to(&mut t_f32, core::CV_32F, 1.0, 0.0)?;
                    
                    // Convert 2x3 affine to 3x3 for homography motion type if needed
                    if config.ecc_motion_type == EccMotionType::Homography {
                        let mut h_3x3 = Mat::eye(3, 3, core::CV_32F)?.to_mat()?;
                        let mut roi = h_3x3.rowscols_mut(core::Range::new(0, 2)?, core::Range::new(0, 3)?)?;
                        t_f32.copy_to(&mut roi)?;
                        Some(h_3x3)
                    } else {
                        Some(t_f32)
                    }
                } else {
                    None
                }
            };
            
            initial_transform
        } else {
            None
        }
    } else {
        None
    };
    
    // Step 2: Refine with ECC for subpixel precision
    let motion_type = ecc_motion_to_opencv(config.ecc_motion_type);
    let rows = if motion_type == video::MOTION_HOMOGRAPHY { 3 } else { 2 };
    
    // Reduce ECC iterations since we start from a good initial estimate
    // This is where the 60-70% speedup comes from
    let has_initial = initial_warp.is_some();
    
    // Use keypoint result as initialization if available, otherwise identity
    let mut warp_matrix = if let Some(init) = initial_warp {
        log::debug!("   Using keypoint-based initialization for ECC refinement");
        init
    } else {
        log::debug!("   No keypoint match found, using identity initialization");
        Mat::eye(rows, 3, core::CV_32F)?.to_mat()?
    };
    
    let reduced_iterations = if has_initial {
        // With good initialization, we still need substantial ECC refinement
        // 50% of iterations provides good balance between speed and quality
        (config.ecc_max_iterations as f32 * 0.5) as i32
    } else {
        config.ecc_max_iterations
    };
    
    let criteria = core::TermCriteria {
        typ: core::TermCriteria_COUNT + core::TermCriteria_EPS,
        max_count: reduced_iterations,
        epsilon: config.ecc_epsilon,
    };
    
    // Run ECC refinement — GPU concurrency bounded by gpu_semaphore()
    let _gpu_permit = gpu_semaphore().acquire();
    
    find_transform_ecc_with_timeout(
        tgt_blurred,
        ref_blurred,
        &mut warp_matrix,
        motion_type,
        criteria,
        config.ecc_timeout_seconds,
        "hybrid",
    )?;
    
    Ok(warp_matrix)
}

fn compute_ecc_transform_internal(
    ref_blurred: &Mat,
    tgt_blurred: &Mat,
    config: &ProcessingConfig,
) -> Result<Mat> {
    // 3. Initialize warp matrix based on motion type
    let motion_type = ecc_motion_to_opencv(config.ecc_motion_type);
    let rows = if motion_type == video::MOTION_HOMOGRAPHY { 3 } else { 2 };
    let mut warp_matrix = Mat::eye(rows, 3, core::CV_32F)?.to_mat()?;
    
    // 4. Define termination criteria
    let criteria = core::TermCriteria {
        typ: core::TermCriteria_COUNT + core::TermCriteria_EPS,
        max_count: config.ecc_max_iterations,
        epsilon: config.ecc_epsilon,
    };
    
    // 5. Compute ECC transformation
    // GPU concurrency bounded by gpu_semaphore()
    let _gpu_permit = gpu_semaphore().acquire();
    
    find_transform_ecc_with_timeout(
        tgt_blurred,
        ref_blurred,
        &mut warp_matrix,
        motion_type,
        criteria,
        config.ecc_timeout_seconds,
        "standard",
    )?;
    drop(_gpu_permit);
    
    Ok(warp_matrix)
}

/// Run find_transform_ecc with a timeout to prevent infinite hangs.
/// OpenCV's find_transform_ecc is a blocking C++ call that cannot be interrupted.
/// On certain images, it can run for many minutes without converging or failing.
/// This wrapper spawns a dedicated thread and waits with a timeout.
/// If the timeout is reached, the ECC computation is abandoned (the thread will
/// eventually finish on its own, but we don't wait for it).
fn find_transform_ecc_with_timeout(
    tgt_blurred: &Mat,
    ref_blurred: &Mat,
    warp_matrix: &mut Mat,
    motion_type: i32,
    criteria: core::TermCriteria,
    timeout_seconds: u64,
    filename: &str,
) -> Result<()> {
    // Clone the data for the thread (Mat is refcounted, clone is cheap for headers but we need deep copies)
    let tgt_clone = tgt_blurred.clone();
    let ref_clone = ref_blurred.clone();
    let mut warp_clone = warp_matrix.clone();
    
    let (tx, rx) = std::sync::mpsc::channel();
    let fname = filename.to_string();
    
    std::thread::spawn(move || {
        let result = video::find_transform_ecc(
            &tgt_clone,
            &ref_clone,
            &mut warp_clone,
            motion_type,
            criteria,
            &Mat::default(),
            5,
        );
        // Send result back (may fail if receiver dropped due to timeout, that's ok)
        let _ = tx.send((result, warp_clone));
    });
    
    match rx.recv_timeout(Duration::from_secs(timeout_seconds)) {
        Ok((warp_result, result_matrix)) => {
            warp_result?;
            // Copy the result back into the caller's warp_matrix
            result_matrix.copy_to(warp_matrix)?;
            Ok(())
        }
        Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
            log::warn!("   ⏱️  ECC timed out after {}s for {} (thread abandoned)", timeout_seconds, fname);
            Err(anyhow::anyhow!("ECC timeout after {}s - do not converge", timeout_seconds))
        }
        Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
            log::error!("   ECC thread panicked for {}", fname);
            Err(anyhow::anyhow!("ECC thread panicked"))
        }
    }
}

/// Check if a transformation matrix is reasonable (not too distorted)
/// Returns true if the transformation is acceptable, false if it's too extreme
fn is_transform_reasonable(warp_matrix: &Mat, motion_type: EccMotionType, config: &ProcessingConfig) -> Result<bool> {
    if motion_type == EccMotionType::Homography {
        // For homography (3x3), check determinant and scale factors
        let m00 = *warp_matrix.at_2d::<f32>(0, 0)?;
        let m01 = *warp_matrix.at_2d::<f32>(0, 1)?;
        let m10 = *warp_matrix.at_2d::<f32>(1, 0)?;
        let m11 = *warp_matrix.at_2d::<f32>(1, 1)?;
        let m02 = *warp_matrix.at_2d::<f32>(0, 2)?;
        let m12 = *warp_matrix.at_2d::<f32>(1, 2)?;
        
        // Calculate scale factors
        let scale_x = (m00 * m00 + m10 * m10).sqrt();
        let scale_y = (m01 * m01 + m11 * m11).sqrt();
        
        // Calculate translation magnitude
        let translation = (m02 * m02 + m12 * m12).sqrt();
        
        // Calculate determinant
        let det = m00 * m11 - m01 * m10;
        
        // Check against config limits
        let min_scale = 1.0 / config.max_transform_scale;
        let max_scale = config.max_transform_scale;
        let max_translation = config.max_transform_translation;
        let max_det = config.max_transform_determinant;
        let min_det = 1.0 / config.max_transform_determinant;
        
        if scale_x < min_scale || scale_x > max_scale || scale_y < min_scale || scale_y > max_scale {
            log::warn!("   ⚠️  Transform rejected: extreme scaling (x: {:.3}, y: {:.3}, limits: {:.3}-{:.3})", 
                scale_x, scale_y, min_scale, max_scale);
            return Ok(false);
        }
        
        if translation > max_translation {
            log::warn!("   ⚠️  Transform rejected: excessive translation ({:.1} pixels, limit: {:.1})", 
                translation, max_translation);
            return Ok(false);
        }
        
        if det.abs() < min_det || det.abs() > max_det {
            log::warn!("   ⚠️  Transform rejected: extreme determinant ({:.3}, limits: {:.3}-{:.3})", 
                det, min_det, max_det);
            return Ok(false);
        }
    } else {
        // For affine (2x3), similar checks
        let m00 = *warp_matrix.at_2d::<f32>(0, 0)?;
        let m01 = *warp_matrix.at_2d::<f32>(0, 1)?;
        let m10 = *warp_matrix.at_2d::<f32>(1, 0)?;
        let m11 = *warp_matrix.at_2d::<f32>(1, 1)?;
        let m02 = *warp_matrix.at_2d::<f32>(0, 2)?;
        let m12 = *warp_matrix.at_2d::<f32>(1, 2)?;
        
        let scale_x = (m00 * m00 + m10 * m10).sqrt();
        let scale_y = (m01 * m01 + m11 * m11).sqrt();
        let translation = (m02 * m02 + m12 * m12).sqrt();
        let det = m00 * m11 - m01 * m10;
        
        let min_scale = 1.0 / config.max_transform_scale;
        let max_scale = config.max_transform_scale;
        let max_translation = config.max_transform_translation;
        let max_det = config.max_transform_determinant;
        let min_det = 1.0 / config.max_transform_determinant;
        
        if scale_x < min_scale || scale_x > max_scale || scale_y < min_scale || scale_y > max_scale {
            log::warn!("   ⚠️  Transform rejected: extreme scaling (x: {:.3}, y: {:.3}, limits: {:.3}-{:.3})", 
                scale_x, scale_y, min_scale, max_scale);
            return Ok(false);
        }
        
        if translation > max_translation {
            log::warn!("   ⚠️  Transform rejected: excessive translation ({:.1} pixels, limit: {:.1})", 
                translation, max_translation);
            return Ok(false);
        }
        
        if det.abs() < min_det || det.abs() > max_det {
            log::warn!("   ⚠️  Transform rejected: extreme determinant ({:.3}, limits: {:.3}-{:.3})", 
                det, min_det, max_det);
            return Ok(false);
        }
    }
    
    Ok(true)
}

/// Fallback to feature-based alignment when ECC fails
/// Uses ORB for robust feature matching (faster than SIFT, not used in hybrid mode)
fn compute_feature_based_transform(
    ref_gray: &Mat,
    tgt_gray: &Mat,
    motion_type: EccMotionType,
) -> Result<Mat> {
    log::info!("   🔄 Attempting feature-based alignment fallback (ORB)...");
    
    // Extract ORB features (fast and robust)
    let mut orb = features2d::ORB::create(
        5000, // More features for better accuracy
        1.2,  // scaleFactor
        8,    // nlevels
        10,   // edgeThreshold
        0,    // firstLevel
        2,    // WTA_K
        features2d::ORB_ScoreType::HARRIS_SCORE,
        31,   // patchSize
        5,    // fastThreshold
    )?;
    
    let mut ref_keypoints = opencv::core::Vector::new();
    let mut ref_descriptors = Mat::default();
    orb.detect_and_compute(ref_gray, &Mat::default(), &mut ref_keypoints, &mut ref_descriptors, false)?;
    
    let mut tgt_keypoints = opencv::core::Vector::new();
    let mut tgt_descriptors = Mat::default();
    orb.detect_and_compute(tgt_gray, &Mat::default(), &mut tgt_keypoints, &mut tgt_descriptors, false)?;
    
    if ref_keypoints.len() < 10 || tgt_keypoints.len() < 10 {
        log::warn!("   ⚠️  Not enough features found (ref: {}, tgt: {})", ref_keypoints.len(), tgt_keypoints.len());
        return Err(anyhow::anyhow!("Insufficient features for matching"));
    }
    
    // Match features using BFMatcher with Hamming distance (for ORB)
    let matcher = features2d::BFMatcher::create(core::NORM_HAMMING, true)?;
    let mut matches = opencv::core::Vector::new();
    matcher.train_match(&tgt_descriptors, &ref_descriptors, &mut matches, &Mat::default())?;
    
    if matches.len() < 10 {
        log::warn!("   ⚠️  Not enough matches found: {}", matches.len());
        return Err(anyhow::anyhow!("Insufficient matches"));
    }
    
    // Extract matching points
    let mut ref_points = opencv::core::Vector::<core::Point2f>::new();
    let mut tgt_points = opencv::core::Vector::<core::Point2f>::new();
    
    for m in matches.iter() {
        ref_points.push(ref_keypoints.get(m.train_idx as usize)?.pt());
        tgt_points.push(tgt_keypoints.get(m.query_idx as usize)?.pt());
    }
    
    log::info!("   ✓ Found {} feature matches", matches.len());
    
    // Compute transformation
    let warp_matrix = if motion_type == EccMotionType::Homography {
        // Find homography with RANSAC
        let mut mask = Mat::default();
        let h = calib3d::find_homography(
            &tgt_points,
            &ref_points,
            &mut mask,
            calib3d::RANSAC,
            3.0, // ransacReprojThreshold
        )?;
        
        if h.empty() {
            return Err(anyhow::anyhow!("Failed to compute homography"));
        }
        
        // Convert to f32
        let mut h_f32 = Mat::default();
        h.convert_to(&mut h_f32, core::CV_32F, 1.0, 0.0)?;
        h_f32
    } else {
        // Estimate affine transform
        let mut inliers = Mat::default();
        let affine = calib3d::estimate_affine_2d(
            &tgt_points,
            &ref_points,
            &mut inliers,
            calib3d::RANSAC,
            3.0, // ransacReprojThreshold
            2000, // maxIters
            0.99, // confidence
            10, // refineIters
        )?;
        
        if affine.empty() {
            return Err(anyhow::anyhow!("Failed to estimate affine transform"));
        }
        
        // Convert to f32
        let mut affine_f32 = Mat::default();
        affine.convert_to(&mut affine_f32, core::CV_32F, 1.0, 0.0)?;
        affine_f32
    };
    
    log::info!("   ✓ Feature-based alignment successful");
    Ok(warp_matrix)
}

/// Align images using ECC method (for macro/precision focus stacking)
fn align_images_ecc(
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

    report_progress("Starting ECC alignment...", 0.0);
    log::info!("🔬 ECC Alignment starting with {} images", image_paths.len());
    log::info!("   Motion: {:?}, Iterations: {}, Epsilon: {:.1e}, Kernel: {}x{}",
               config.ecc_motion_type, config.ecc_max_iterations, 
               config.ecc_epsilon, config.ecc_gauss_filter_size, config.ecc_gauss_filter_size);
    log::info!("   Hybrid mode: {} (keypoint init + ECC refinement)", 
               if config.ecc_use_hybrid { "ENABLED (60-70% faster)" } else { "DISABLED" });
    
    // Check OpenCL/GPU availability
    let opencl_available = opencv::core::use_opencl().unwrap_or(false);
    log::info!("   OpenCL available: {}", opencl_available);
    if opencl_available {
        if let Ok(device_name) = opencv::core::Device::get_default() {
            log::info!("   OpenCL device: {}", device_name.name().unwrap_or_else(|_| "Unknown".to_string()));
        }
    } else {
        log::warn!("   ⚠️  OpenCL not available - using CPU only");
        log::warn!("   Consider installing OpenCL drivers for GPU acceleration");
    }

    if image_paths.len() < 2 {
        let first_img = load_image(&image_paths[0])?;
        return Ok(core::Rect::new(0, 0, first_img.cols(), first_img.rows()));
    }

    let aligned_dir = output_dir.join("aligned");
    std::fs::create_dir_all(&aligned_dir)?;

    // Step 1: Load sharpness scores WITHOUT loading images into memory
    report_progress("Computing sharpness scores...", 5.0);
    log::info!("Step 1: Computing sharpness for {} images (memory-efficient mode)", image_paths.len());
    
    // Only store paths and sharpness scores, NOT the images
    let mut image_metadata: Vec<(usize, PathBuf, f64)> = Vec::new();
    let mut rejected_count = 0;
    
    for (idx, path) in image_paths.iter().enumerate() {
        // Check cancellation
        if let Some(ref flag) = cancel_flag {
            if flag.load(Ordering::Relaxed) {
                log::info!("✋ ECC alignment cancelled by user");
                return Err(anyhow::anyhow!("Operation cancelled by user"));
            }
        }
        
        // Try to load cached sharpness first
        let max_regional = if let Some(cached) = load_cached_sharpness(path, output_dir) {
            cached
        } else {
            // Need to load image temporarily to compute sharpness
            let img = load_image(path)?;
            log::info!("   Computing sharpness for {}", 
                      path.file_name().unwrap_or_default().to_string_lossy());
            let _gpu_permit = gpu_semaphore().acquire();
            let (max_regional, _, _) = sharpness::compute_regional_sharpness_auto(&img, config.sharpness_grid_size)?;
            // Image is dropped here, freeing memory
            max_regional
        };
        
        // Apply sharpness threshold filter
        if max_regional < config.sharpness_threshold as f64 {
            let filename = path.file_name().unwrap_or_default().to_string_lossy();
            log::info!("   ✗ Rejected {} (sharpness {:.2} < threshold {:.2})", 
                      filename, max_regional, config.sharpness_threshold);
            rejected_count += 1;
            continue; // Skip this image
        }
        
        // Only store metadata, not the image
        image_metadata.push((idx, path.clone(), max_regional));
        
        let progress = 5.0 + (idx as f32 / image_paths.len() as f32) * 10.0;
        report_progress(&format!("Analyzing images: {}/{}", idx + 1, image_paths.len()), progress);
    }
    
    // Report filtering results
    if rejected_count > 0 {
        log::info!("📊 Filtered out {} blurry images (threshold: {:.2})", rejected_count, config.sharpness_threshold);
        log::info!("   Proceeding with {} sharp images", image_metadata.len());
    }
    
    // Check if we have enough images left
    if image_metadata.len() < 2 {
        return Err(anyhow::anyhow!(
            "Not enough sharp images after filtering (need at least 2, have {}). Lower the sharpness threshold in settings.",
            image_metadata.len()
        ));
    }
    
    // Step 2: Sort by sharpness - use middle image as reference (most stable)
    image_metadata.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
    let ref_idx = image_metadata.len() / 2;  // Middle sharpness image
    let reference_path = &image_metadata[ref_idx].1;
    let reference_orig_idx = image_metadata[ref_idx].0;
    
    log::info!("Using image {} as reference (sharpness: {:.2})", 
               reference_path.display(), image_metadata[ref_idx].2);
    
    // Load reference image (only this one image in memory)
    let reference_img = load_image(reference_path)?;
    
    // Save reference image (or skip if it already exists in resume mode)
    let ref_filename = reference_path.file_name().unwrap().to_string_lossy();
    let ref_output = aligned_dir.join(format!("aligned_{:04}_{}", reference_orig_idx, ref_filename));
    
    if !ref_output.exists() {
        opencv::imgcodecs::imwrite(ref_output.to_str().unwrap(), &reference_img, &core::Vector::new())?;
        log::info!("   ✓ Saved reference image: {}", ref_output.display());
    } else {
        log::info!("   ⏩ Skipping reference image (already exists): {}", ref_output.display());
    }
    
    report_progress("Aligning images with ECC...", 15.0);
    
    // Pre-compute reference image processing (done once, reused for all alignments)
    let ref_gray = if reference_img.channels() == 1 {
        reference_img.clone()
    } else {
        let mut gray = Mat::default();
        crate::opencv_compat::cvt_color(&reference_img, &mut gray, imgproc::COLOR_BGR2GRAY, 0)?;
        gray
    };
    
    let mut ref_blurred = Mat::default();
    let kernel_size = core::Size::new(config.ecc_gauss_filter_size, config.ecc_gauss_filter_size);
    crate::opencv_compat::gaussian_blur(&ref_gray, &mut ref_blurred, kernel_size, 0.0, 0.0, core::BORDER_DEFAULT)?;
    
    log::info!("✓ Reference image preprocessed (will be reused for all {} alignments)", image_metadata.len() - 1);
    
    // Keep reference images for hybrid mode (keypoint extraction needs original)
    // Clone ref_gray for hybrid mode, then drop reference_img to save memory
    let ref_gray_for_hybrid = if config.ecc_use_hybrid {
        Some(Arc::new(ref_gray.clone()))
    } else {
        None
    };
    
    let ref_size = reference_img.size()?;
    drop(reference_img);
    
    // Wrap ref_blurred in Arc for sharing across threads
    let ref_blurred = Arc::new(ref_blurred);
    
    // Step 3: Process images in batches to limit memory usage
    // Use config value, but allow environment variable override for testing
    let batch_size = std::env::var("IMAGESTACKER_ECC_BATCH_SIZE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(config.ecc_batch_size);
    
    log::info!("📦 Processing in batches of {} images to limit memory usage", batch_size);
    log::info!("   Configure via Settings dialog or IMAGESTACKER_ECC_BATCH_SIZE environment variable");
    
    // Step 3: Process images in batches (excluding reference)
    // Each batch loads only batch_size images into memory
    // This allows parallel execution of:
    // - Image warping (warp_perspective/warp_affine)
    // - File I/O operations
    // Only the iterative ECC optimization itself is serialized due to potential OpenCL race conditions
    let total_to_align = image_metadata.len() - 1;
    let aligned_count = Arc::new(Mutex::new(0_usize));
    
    log::info!("🚀 Starting batch ECC alignment of {} images (batch_size={})", total_to_align, batch_size);
    log::info!("   Using Rayon parallel processing with {} threads per batch", rayon::current_num_threads());
    log::info!("   GPU concurrency limit: {} (set IMAGESTACKER_GPU_CONCURRENCY to change)", 
        std::env::var("IMAGESTACKER_GPU_CONCURRENCY").unwrap_or_else(|_| "2".to_string()));
    log::info!("   Per-image ECC timeout: {}s (configurable in Settings)", config.ecc_timeout_seconds);
    
    // Calculate common bounding box for all aligned images
    let bbox = Arc::new(Mutex::new(core::Rect::new(0, 0, ref_size.width, ref_size.height)));
    
    // Process images in batches
    let batches: Vec<_> = image_metadata.iter()
        .enumerate()
        .filter(|(list_idx, _)| *list_idx != ref_idx) // Skip reference
        .collect::<Vec<_>>()
        .chunks(batch_size)
        .map(|chunk| chunk.to_vec())
        .collect();
    
    let total_batches = batches.len();
    log::info!("📦 Processing {} batches", total_batches);
    
    for (batch_num, batch) in batches.iter().enumerate() {
        log::info!("📦 Batch {}/{}: Processing {} images", batch_num + 1, total_batches, batch.len());
        
        // Check cancellation
        if let Some(ref flag) = cancel_flag {
            if flag.load(Ordering::Relaxed) {
                log::info!("✋ ECC alignment cancelled by user");
                return Err(anyhow::anyhow!("Operation cancelled by user"));
            }
        }
        
        // Process this batch in parallel
        let results: Vec<_> = batch.par_iter().map(|(_list_idx, (orig_idx, path, _sharpness))| {
            // Check cancellation
            if let Some(ref flag) = cancel_flag {
                if flag.load(Ordering::Relaxed) {
                    return Err(anyhow::anyhow!("Operation cancelled by user"));
                }
            }
            
            let thread_id = rayon::current_thread_index().unwrap_or(0);
            let filename = path.file_name().unwrap().to_string_lossy();
            
            // Check if output already exists (for resume functionality)
            let output_path = aligned_dir.join(format!("aligned_{:04}_{}", orig_idx, filename));
            if output_path.exists() {
                log::info!("   [Thread {}] ⏩ Skipping (already aligned): {}", thread_id, filename);
                // Update progress counter
                let mut count = aligned_count.lock().unwrap();
                *count += 1;
                drop(count);
                return Ok(());
            }
            
            log::info!("   [Thread {}] Loading: {}", thread_id, filename);
            
            // Load image (only for this batch)
            let img = load_image(path)?;
            
            // Step 1: Preprocess target image (runs in parallel)
            let tgt_gray = if img.channels() == 1 {
                img.clone()
            } else {
                let mut gray = Mat::default();
                crate::opencv_compat::cvt_color(&img, &mut gray, imgproc::COLOR_BGR2GRAY, 0)?;
                gray
            };
            
            let mut tgt_blurred = Mat::default();
            let kernel_size = core::Size::new(config.ecc_gauss_filter_size, config.ecc_gauss_filter_size);
            crate::opencv_compat::gaussian_blur(&tgt_gray, &mut tgt_blurred, kernel_size, 0.0, 0.0, core::BORDER_DEFAULT)?;
            
            // Step 2: Compute ECC transform (with optional hybrid acceleration)
            log::info!("   [Thread {}] Computing ECC for: {}", thread_id, filename);
            
            // Try to compute transform, but handle convergence failures gracefully
            let warp_matrix_result = if config.ecc_use_hybrid {
                // Hybrid mode: keypoint init + ECC refinement (60-70% faster)
                if let Some(ref ref_gray_img) = ref_gray_for_hybrid {
                    compute_hybrid_ecc_transform(ref_gray_img, &tgt_gray, &ref_blurred, &tgt_blurred, config)
                } else {
                    // Fallback to standard ECC if ref_gray not available
                    compute_ecc_transform_internal(&ref_blurred, &tgt_blurred, config)
                }
            } else {
                // Standard ECC (no keypoint initialization)
                compute_ecc_transform_internal(&ref_blurred, &tgt_blurred, config)
            };
            
            // Handle ECC convergence failures gracefully - use identity transform or feature fallback
            let mut warp_matrix = match warp_matrix_result {
                Ok(matrix) => matrix,
                Err(e) => {
                    let error_msg = e.to_string();
                    if error_msg.contains("do not converge") || error_msg.contains("StsNoConv") {
                        log::warn!("   [Thread {}] ⚠️  ECC did not converge for {}, trying feature-based fallback", thread_id, filename);
                        
                        // Try feature-based alignment as fallback
                        if let Some(ref ref_gray_img) = ref_gray_for_hybrid {
                            match compute_feature_based_transform(ref_gray_img, &tgt_gray, config.ecc_motion_type) {
                                Ok(feature_matrix) => {
                                    log::info!("   [Thread {}] ✓ Feature-based fallback successful for {}", thread_id, filename);
                                    feature_matrix
                                }
                                Err(feature_err) => {
                                    log::warn!("   [Thread {}] ⚠️  Feature fallback also failed: {}, using identity", thread_id, feature_err);
                                    // Last resort: identity transform
                                    if config.ecc_motion_type == EccMotionType::Homography {
                                        Mat::eye(3, 3, core::CV_32F)?.to_mat()?
                                    } else {
                                        let identity_data = [1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0];
                                        Mat::from_slice_2d::<f32>(&identity_data.chunks(3).collect::<Vec<_>>())?
                                    }
                                }
                            }
                        } else {
                            // No ref_gray available, use identity
                            log::warn!("   [Thread {}]    Using identity transform (no ref_gray for feature fallback)", thread_id);
                            if config.ecc_motion_type == EccMotionType::Homography {
                                Mat::eye(3, 3, core::CV_32F)?.to_mat()?
                            } else {
                                let identity_data = [1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0];
                                Mat::from_slice_2d::<f32>(&identity_data.chunks(3).collect::<Vec<_>>())?
                            }
                        }
                    } else {
                        // Other errors are still fatal
                        return Err(e);
                    }
                }
            };
            
            // Check if the transformation is reasonable
            match is_transform_reasonable(&warp_matrix, config.ecc_motion_type, config) {
                Ok(true) => {
                    // Transform is good, use it
                    log::debug!("   [Thread {}] ✓ Transform validation passed", thread_id);
                }
                Ok(false) => {
                    // Transform is too distorted, try feature-based fallback
                    log::warn!("   [Thread {}] ⚠️  ECC transform too distorted for {}, trying feature-based fallback", thread_id, filename);
                    
                    if let Some(ref ref_gray_img) = ref_gray_for_hybrid {
                        match compute_feature_based_transform(ref_gray_img, &tgt_gray, config.ecc_motion_type) {
                            Ok(feature_matrix) => {
                                // Check if feature-based transform is better
                                if is_transform_reasonable(&feature_matrix, config.ecc_motion_type, config).unwrap_or(false) {
                                    log::info!("   [Thread {}] ✓ Using feature-based transform instead", thread_id);
                                    warp_matrix = feature_matrix;
                                } else {
                                    log::warn!("   [Thread {}] ⚠️  Feature-based also distorted, using identity", thread_id);
                                    warp_matrix = if config.ecc_motion_type == EccMotionType::Homography {
                                        Mat::eye(3, 3, core::CV_32F)?.to_mat()?
                                    } else {
                                        let identity_data = [1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0];
                                        Mat::from_slice_2d::<f32>(&identity_data.chunks(3).collect::<Vec<_>>())?
                                    };
                                }
                            }
                            Err(e) => {
                                log::warn!("   [Thread {}] ⚠️  Feature fallback failed: {}, using identity", thread_id, e);
                                warp_matrix = if config.ecc_motion_type == EccMotionType::Homography {
                                    Mat::eye(3, 3, core::CV_32F)?.to_mat()?
                                } else {
                                    let identity_data = [1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0];
                                    Mat::from_slice_2d::<f32>(&identity_data.chunks(3).collect::<Vec<_>>())?
                                };
                            }
                        }
                    } else {
                        log::warn!("   [Thread {}] ⚠️  No ref_gray for fallback, using identity", thread_id);
                        warp_matrix = if config.ecc_motion_type == EccMotionType::Homography {
                            Mat::eye(3, 3, core::CV_32F)?.to_mat()?
                        } else {
                            let identity_data = [1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0];
                            Mat::from_slice_2d::<f32>(&identity_data.chunks(3).collect::<Vec<_>>())?
                        };
                    }
                }
                Err(e) => {
                    log::warn!("   [Thread {}] ⚠️  Transform validation error: {}", thread_id, e);
                }
            }
            
            // Step 3: Apply transformation (runs in parallel)
            log::info!("   [Thread {}] Warping: {}", thread_id, filename);
            let mut aligned = Mat::default();
            
            if config.ecc_motion_type == EccMotionType::Homography {
                // Use perspective warp for homography
                imgproc::warp_perspective(
                    &img,
                    &mut aligned,
                    &warp_matrix,
                    ref_size,
                    imgproc::INTER_LINEAR,
                    core::BORDER_CONSTANT,
                    core::Scalar::default(),
                )?;
            } else {
                // Use affine warp for other motion types
                imgproc::warp_affine(
                    &img,
                    &mut aligned,
                    &warp_matrix,
                    ref_size,
                    imgproc::INTER_LINEAR,
                    core::BORDER_CONSTANT,
                    core::Scalar::default(),
                )?;
            }
            
            // Image is dropped here, freeing memory for this thread
            drop(img);
            
            // Save aligned image
            log::info!("   [Thread {}] Saving: {}", thread_id, filename);
            let output_path = aligned_dir.join(format!("aligned_{:04}_{}", orig_idx, filename));
            opencv::imgcodecs::imwrite(output_path.to_str().unwrap(), &aligned, &core::Vector::new())?;
            
            // Update progress
            let mut count = aligned_count.lock().unwrap();
            *count += 1;
            let progress = 15.0 + (*count as f32 / total_to_align as f32) * 80.0;
            let current_count = *count;
            drop(count);
            
            report_progress(&format!("ECC aligned: {}/{}", current_count, total_to_align), progress);
            log::info!("   [Thread {}] ✓ Completed: {}", thread_id, filename);
            
            Ok(())
        }).collect();
        
        log::info!("📦 Batch {}/{}: All threads completed, checking for errors...", batch_num + 1, total_batches);
        
        // Check for any errors in this batch
        for (idx, result) in results.iter().enumerate() {
            if let Err(e) = result {
                log::error!("📦 Batch {}/{}: Thread {} failed with error: {}", batch_num + 1, total_batches, idx, e);
                return Err(anyhow::anyhow!("Batch {} failed: {}", batch_num + 1, e));
            }
        }
        
        log::info!("📦 Batch {}/{} completed successfully", batch_num + 1, total_batches);
    }
    
    report_progress("ECC alignment complete", 100.0);
    log::info!("✅ ECC alignment completed successfully - {} images aligned", total_to_align);
    
    let final_bbox = bbox.lock().unwrap().clone();
    Ok(final_bbox)
}

pub fn align_images(
    image_paths: &[PathBuf],
    output_dir: &std::path::Path,
    config: &ProcessingConfig,
    progress_cb: Option<ProgressCallback>,
    cancel_flag: Option<Arc<AtomicBool>>,
) -> Result<core::Rect> {
    // Route to ECC-based alignment if selected
    if config.feature_detector == FeatureDetector::ECC {
        return align_images_ecc(image_paths, output_dir, config, progress_cb, cancel_flag);
    }
    
    // Log OpenCL state at function entry
    let opencl_at_start = opencv::core::use_opencl().unwrap_or(false);
    log::info!("🔍 align_images() called - OpenCL enabled at entry: {}", opencl_at_start);
    if opencl_at_start {
        log::info!("🚀 Using hybrid parallel processing: Rayon threads + GPU (mutex-serialized OpenCL)");
        log::info!("   • Multiple images processed in parallel (Rayon)");
        log::info!("   • GPU operations serialized via mutex for thread safety");
    } else {
        log::info!("⚙️  Using parallel CPU processing (Rayon only, no GPU)");
    }
    
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

        // Use parallel processing with OpenCL enabled
        // Modern OpenCV (4.x) handles OpenCL thread safety internally
        let batch_results: Vec<(usize, PathBuf, f64)> = batch
            .par_iter()
            .enumerate()
            .filter_map(|(batch_idx, path)| {
                // Check for cancellation inside parallel work
                if let Some(ref flag) = cancel_flag {
                    if flag.load(Ordering::Relaxed) {
                        return None; // Stop processing this item
                    }
                }
                
                let idx = batch_start + batch_idx;
                let filename = path.file_name().unwrap_or_default().to_string_lossy().to_string();
                
                // Try to load cached sharpness first
                if let Some(cached_sharpness) = load_cached_sharpness(path, output_dir) {
                    // Apply threshold even for cached values
                    if cached_sharpness < config.sharpness_threshold as f64 {
                        log::info!(
                            "Image {} ({}): max_regional={:.2} (cached) - SKIPPED (below threshold {:.2})",
                            idx, filename, cached_sharpness, config.sharpness_threshold
                        );
                        println!(
                            "    [{}] {}: max_region={:.2} (cached) - SKIPPED",
                            idx, filename, cached_sharpness
                        );
                        return None;
                    }
                    
                    log::info!(
                        "Image {} ({}): max_regional={:.2} (cached)",
                        idx, filename, cached_sharpness
                    );
                    println!(
                        "    [{}] {}: max_region={:.2} (cached)",
                        idx, filename, cached_sharpness
                    );
                    return Some((idx, path.clone(), cached_sharpness));
                }
                
                // If not cached, compute sharpness
                match load_image(path) {
                    Ok(img) => {
                        // Use configured grid size for regional analysis
                        // GPU concurrency bounded by gpu_semaphore()
                        let _gpu_permit = gpu_semaphore().acquire();
                        let sharpness_result = sharpness::compute_regional_sharpness_auto(&img, config.sharpness_grid_size);
                        
                        match sharpness_result {
                            Ok((max_regional, global, sharp_count)) => {
                                // Apply threshold to computed values too
                                if max_regional < config.sharpness_threshold as f64 {
                                    log::info!(
                                        "Image {} ({}): max_regional={:.2}, global={:.2}, sharp_regions={} - SKIPPED (below threshold {:.2})",
                                        idx, filename, max_regional, global, sharp_count, config.sharpness_threshold
                                    );
                                    println!(
                                        "    [{}] {}: max_region={:.2}, global={:.2}, sharp_areas={} - SKIPPED",
                                        idx, filename, max_regional, global, sharp_count
                                    );
                                    return None;
                                }
                                
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
        
        // Check if cancellation occurred during work
        if let Some(ref flag) = cancel_flag {
            if flag.load(Ordering::Relaxed) {
                log::info!("✋ Alignment cancelled by user during sharpness detection");
                return Err(anyhow::anyhow!("Operation cancelled by user"));
            }
        }
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
        FeatureDetector::ECC => "ECC (Precision)",
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
    
    // Calculate image size to adjust batch size for very large images
    // GPU operations create multiple copies, so we need much smaller batches for large images
    let first_img = load_image(&sharp_image_paths[0].1)?;
    let megapixels = (first_img.rows() * first_img.cols()) as f64 / 1_000_000.0;
    let is_very_large = megapixels > 30.0; // >30MP images need special handling
    
    let feature_batch_size = if is_very_large {
        // For very large images (>30MP), use minimal batches to prevent OOM
        // With reduced SIFT features (2000), we can safely process 3 images at once
        match config.feature_detector {
            FeatureDetector::ECC => 1,                               // ECC: serial (parallelizes internally)
            FeatureDetector::ORB => 3.min(base_feature_batch_size),     // ORB: max 3 images
            FeatureDetector::SIFT => 3,                                  // SIFT: max 3 images (increased from 2)
            FeatureDetector::AKAZE => 2,                                 // AKAZE: max 2 images
        }
    } else {
        match config.feature_detector {
            FeatureDetector::ECC => config.ecc_chunk_size,          // ECC: use configured chunk size
            FeatureDetector::ORB => base_feature_batch_size,
            FeatureDetector::SIFT => (base_feature_batch_size / 4).max(3),
            FeatureDetector::AKAZE => (base_feature_batch_size / 4).max(3),
        }
    };
    
    log::info!(
        "Using {} detector with batch size {} (base: {}, image size: {:.1}MP)",
        match config.feature_detector {
            FeatureDetector::ECC => "ECC",
            FeatureDetector::ORB => "ORB",
            FeatureDetector::SIFT => "SIFT",
            FeatureDetector::AKAZE => "AKAZE",
        },
        feature_batch_size,
        base_feature_batch_size,
        megapixels
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

        // Use parallel processing with OpenCL enabled
        // Modern OpenCV (4.x) handles OpenCL thread safety internally
        let batch_features: Vec<Result<(Vec<core::KeyPoint>, core::Mat, f64)>> = batch_paths
            .par_iter()
            .map(|&path| {
                // Check for cancellation inside parallel work
                if let Some(ref flag) = cancel_flag {
                    if flag.load(Ordering::Relaxed) {
                        return Err(anyhow::anyhow!("Operation cancelled by user"));
                    }
                }
                
                let img = load_image(path)?;

                // GPU preprocessing: minimize CPU↔GPU transfers by doing all GPU work in one go
                // GPU concurrency bounded by gpu_semaphore()
                let gpu_start = std::time::Instant::now();
                let (preprocessed, scale) = {
                    let _gpu_permit = gpu_semaphore().acquire();
                    let lock_acquired = std::time::Instant::now();
                    
                    // Upload to GPU once
                    let img_umat = img.get_umat(core::AccessFlag::ACCESS_READ, core::UMatUsageFlags::USAGE_DEFAULT)?;
                    let upload_done = std::time::Instant::now();
                    
                    // All GPU operations on UMat (no transfers)
                    let gray_umat = if img.channels() == 4 {
                        // BGRA to Gray
                        let mut gray = core::UMat::new_def();
                        crate::opencv_compat::cvt_color(
                            &img_umat,
                            &mut gray,
                            imgproc::COLOR_BGRA2GRAY,
                            0,
                        )?;
                        gray
                    } else if img.channels() == 3 {
                        // BGR to Gray
                        let mut gray = core::UMat::new_def();
                        crate::opencv_compat::cvt_color(
                            &img_umat,
                            &mut gray,
                            imgproc::COLOR_BGR2GRAY,
                            0,
                        )?;
                        gray
                    } else {
                        // Already grayscale
                        img_umat
                    };

                    let preprocessed_umat = if use_clahe {
                        let mut clahe = imgproc::create_clahe(2.0, core::Size::new(8, 8))?;
                        let mut enhanced = core::UMat::new_def();
                        clahe.apply(&gray_umat, &mut enhanced)?;
                        enhanced
                    } else {
                        gray_umat
                    };

                    let mut small_umat = core::UMat::new_def();
                    let scale = ALIGNMENT_SCALE;
                    imgproc::resize(
                        &preprocessed_umat,
                        &mut small_umat,
                        core::Size::default(),
                        scale,
                        scale,
                        imgproc::INTER_AREA,
                    )?;
                    let gpu_ops_done = std::time::Instant::now();
                    
                    // Deep-copy from GPU into a standalone Mat (avoids UMat lifetime issue)
                    let mut small_img = core::Mat::default();
                    small_umat.copy_to(&mut small_img)?;
                    let download_done = std::time::Instant::now();
                    
                    log::debug!(
                        "GPU preprocessing: wait={:.1}ms, upload={:.1}ms, ops={:.1}ms, download={:.1}ms",
                        (lock_acquired - gpu_start).as_secs_f64() * 1000.0,
                        (upload_done - lock_acquired).as_secs_f64() * 1000.0,
                        (gpu_ops_done - upload_done).as_secs_f64() * 1000.0,
                        (download_done - gpu_ops_done).as_secs_f64() * 1000.0
                    );
                    
                    Ok::<_, anyhow::Error>((small_img, scale))
                }?;

                // Feature detection runs on CPU (not GPU accelerated) - can run in parallel
                let (keypoints, descriptors) = extract_features(&preprocessed, detector_type)?;

                Ok((keypoints.to_vec(), descriptors, scale))
            })
            .collect();

        // Check if cancellation occurred during work
        if let Some(ref flag) = cancel_flag {
            if flag.load(Ordering::Relaxed) {
                log::info!("✋ Alignment cancelled by user during feature extraction");
                return Err(anyhow::anyhow!("Operation cancelled by user"));
            }
        }

        log::info!("✓ Feature extraction batch complete - processing {} feature sets", batch_features.len());

        // Convert to valid features
        let mut valid_batch_features = Vec::new();
        
        // For non-first batches, prepend the last image's features from previous batch
        // to maintain feature consistency across batch boundaries
        if let Some(prev_features) = last_batch_features.take() {
            valid_batch_features.push(prev_features);
        }
        
        log::info!("Converting {} feature results to valid features...", batch_features.len());
        for (idx, f) in batch_features.into_iter().enumerate() {
            match f {
                Ok(features) => {
                    log::debug!("  Feature set {}: {} keypoints", idx, features.0.len());
                    valid_batch_features.push(features);
                }
                Err(e) => {
                    log::error!("  Feature set {} failed: {}", idx, e);
                    return Err(e);
                }
            }
        }
        
        log::info!("✓ Converted to {} valid feature sets", valid_batch_features.len());
        
        // Save the last image's features for the next batch (if not the last batch)
        if batch_end < sharp_image_paths.len() {
            last_batch_features = Some(valid_batch_features.last().unwrap().clone());
        }

        log::info!("Starting pairwise matching for {} consecutive pairs...", valid_batch_features.len() - 1);

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
            // valid_batch_features[0] in non-first batches is the saved feature from previous batch (at load_start-1)
            // valid_batch_features[1..] in non-first batches are new features (at load_start onwards)
            // valid_batch_features[0..] in first batch are all at indices 0 onwards
            let actual_prev_idx = if i == 0 && !is_first_batch {
                // First feature in non-first batch is the saved feature from previous batch
                load_start - 1
            } else if is_first_batch {
                i
            } else {
                // i-th feature (i >= 1) maps to (load_start + i - 1)
                load_start + i - 1
            };
            let actual_curr_idx = if is_first_batch {
                i + 1
            } else {
                // (i+1)-th feature maps to (load_start + i)
                // For i=0: load_start + 0 = load_start (first new image)
                // For i=1: load_start + 1 (second new image)
                load_start + i
            };

            let t_step_2x3 = if curr_descriptors.empty() || prev_descriptors.empty() {
                Mat::default()
            } else {
                // Choose matcher based on feature detector type
                let norm_type = match config.feature_detector {
                    FeatureDetector::ECC => core::NORM_L2,           // ECC uses full images, not descriptors
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
                // Validate the transform using configurable limits
                // This checks scale, translation, and determinant against config thresholds
                match is_transform_reasonable(&t_step_2x3, EccMotionType::Affine, config) {
                    Ok(true) => {
                        let pair_idx = pairwise_transforms.len();
                        // Log transform details for debugging
                        let t_data = t_step_2x3.data_typed::<f32>()?;
                        let a = t_data[0];
                        let b = t_data[1];
                        let c = t_data[3];
                        let d = t_data[4];
                        let determinant = a * d - b * c;
                        
                        log::info!("  Pairwise[{}]: sharp_img[{}] -> sharp_img[{}] (det={:.4})", 
                            pair_idx, actual_curr_idx, actual_prev_idx, determinant);
                        pairwise_transforms.push(t_step_2x3);
                        pairwise_image_indices.push(actual_curr_idx);
                    }
                    Ok(false) => {
                        // Transform rejected by validation (logged in is_transform_reasonable)
                        log::warn!("  REJECTED: sharp_img[{}] -> sharp_img[{}] - transform validation failed", 
                            actual_curr_idx, actual_prev_idx);
                    }
                    Err(e) => {
                        log::warn!("  ERROR validating transform for sharp_img[{}] -> sharp_img[{}]: {}", 
                            actual_curr_idx, actual_prev_idx, e);
                    }
                }
            } else {
                log::warn!("  FAILED: sharp_img[{}] -> sharp_img[{}]", actual_curr_idx, actual_prev_idx);
            }
        }
        
        log::info!("✓ Pairwise matching batch complete - found {} transforms so far", pairwise_transforms.len());
    }

    log::info!("✓ Feature extraction and matching complete");
    log::info!("Found {} pairwise transforms for {} sharp images", pairwise_transforms.len(), sharp_image_paths.len());
    println!("\n✓ Found {} pairwise transforms", pairwise_transforms.len());

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

        // Use parallel processing with OpenCL enabled
        // Modern OpenCV (4.x) handles OpenCL thread safety internally
        let warp_results: Vec<Result<()>> = (batch_start..batch_start + batch.len())
            .into_par_iter()
            .map(|i| {
                // Check for cancellation inside parallel work
                if let Some(ref flag) = cancel_flag {
                    if flag.load(Ordering::Relaxed) {
                        return Err(anyhow::anyhow!("Operation cancelled by user"));
                    }
                }
                
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

                // Warp operations - GPU concurrency bounded by gpu_semaphore()
                let (warped, output_path) = {
                    let _gpu_permit = gpu_semaphore().acquire();
                    
                    // Convert to BGRA if needed (to support transparent borders)
                    let img_with_alpha = if img.channels() == 3 {
                        let mut bgra = Mat::default();
                        crate::opencv_compat::cvt_color(&img, &mut bgra, imgproc::COLOR_BGR2BGRA, 0)?;
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

                    let output_path = aligned_dir.join(format!("{:04}.png", sharp_image_paths[img_idx].0));
                    Ok::<_, anyhow::Error>((warped, output_path))
                }?;

                // Save aligned image with alpha channel (I/O can be parallel)
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

        // Check if cancellation occurred during work
        if let Some(ref flag) = cancel_flag {
            if flag.load(Ordering::Relaxed) {
                log::info!("✋ Alignment cancelled by user during warping");
                return Err(anyhow::anyhow!("Operation cancelled by user"));
            }
        }

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

    // Log OpenCL state at function exit
    let opencl_at_end = opencv::core::use_opencl().unwrap_or(false);
    log::info!("🔍 align_images() exiting - OpenCL enabled at exit: {}", opencl_at_end);
    if !opencl_at_end && opencl_at_start {
        log::warn!("⚠️  OpenCL was disabled during alignment and not restored!");
    }

    Ok(crop_rect)
}