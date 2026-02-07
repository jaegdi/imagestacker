use anyhow::Result;
use opencv::prelude::*;
use opencv::{core, imgproc};
use rayon::prelude::*;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicBool, Ordering};

use crate::config::ProcessingConfig;
use crate::image_io::load_image;
use crate::post_processing::apply_advanced_processing;

/// Progress callback: (message, percentage)
pub type ProgressCallback = Arc<Mutex<dyn FnMut(String, f32) + Send>>;

pub fn stack_images(
    image_paths: &[PathBuf],
    output_dir: &Path,
    crop_rect: Option<core::Rect>,
    config: &ProcessingConfig,
    progress_cb: Option<ProgressCallback>,
    cancel_flag: Option<Arc<AtomicBool>>,
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

    let result = stack_recursive(&reversed_paths, output_dir, 0, config, progress_cb.clone(), cancel_flag.clone())?;

    report_progress("Saving final result...", 95.0);

    let final_dir = output_dir.join("final");
    std::fs::create_dir_all(&final_dir)?;

    let mut final_path = final_dir.join("result_0001.png");
    let mut counter = 1;
    while final_path.exists() {
        counter += 1;
        final_path = final_dir.join(format!("result_{:04}.png", counter));
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

    // Apply advanced processing
    let result = apply_advanced_processing(result, config)?;

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
    cancel_flag: Option<Arc<AtomicBool>>,
) -> Result<Mat> {
    // Check for cancellation at the start of each recursive call
    if let Some(ref flag) = cancel_flag {
        if flag.load(Ordering::Relaxed) {
            log::info!("Stacking cancelled by user");
            return Err(anyhow::anyhow!("Operation cancelled by user"));
        }
    }
    
    if image_paths.is_empty() {
        return Err(anyhow::anyhow!("No images to stack"));
    }
    if image_paths.len() == 1 {
        return load_image(&image_paths[0]);
    }

    let batch_size = if config.auto_bunch_size {
        config.batch_config.stacking_batch_size
    } else {
        config.stacking_bunch_size
    };
    const OVERLAP: usize = 2;

    if image_paths.len() <= batch_size {
        println!("Loading {} images for direct stacking...", image_paths.len());

        if let Some(ref cb) = progress_cb {
            if let Ok(mut cb_lock) = cb.lock() {
                let pct = 10.0 + (level as f32 * 10.0).min(80.0);
                cb_lock(format!("Stacking {} images (level {})...", image_paths.len(), level), pct);
            }
        }

        // Process images one at a time to minimize GPU memory usage.
        // Each 43MP BGRA float32 image uses ~672MB of VRAM, so loading all
        // images simultaneously would easily exceed GPU memory.
        return stack_images_direct_from_paths(image_paths);
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
        // Check for cancellation
        if let Some(ref flag) = cancel_flag {
            if flag.load(Ordering::Relaxed) {
                log::info!("Stacking cancelled by user during batch processing");
                return Err(anyhow::anyhow!("Operation cancelled by user"));
            }
        }
        
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
    stack_recursive(&intermediate_files, output_dir, level + 1, config, progress_cb, cancel_flag)
}

fn stack_images_direct(images: &[Mat]) -> Result<Mat> {
    if images.is_empty() {
        return Err(anyhow::anyhow!("No images to stack"));
    }
    if images.len() == 1 {
        return Ok(images[0].clone());
    }

    let levels = 7;
    
    let use_gpu = opencv::core::use_opencl().unwrap_or(false);
    log::info!("stack_images_direct: {} images, OpenCL={}", images.len(), use_gpu);
    
    let mut fused_pyramid: Vec<core::UMat> = Vec::new();
    let mut max_energies: Vec<core::UMat> = Vec::new();
    let mut fused_alpha = core::UMat::new(core::UMatUsageFlags::USAGE_DEFAULT);

    for (idx, img) in images.iter().enumerate() {
        log::debug!("Processing image {}/{}", idx + 1, images.len());
        
        process_image_into_pyramid(
            img, idx, levels,
            &mut fused_pyramid, &mut max_energies, &mut fused_alpha,
        )?;
    }

    log::debug!("Collapsing pyramid...");
    let result_bgr = collapse_pyramid(&fused_pyramid)?;
    let final_img = assemble_final_image(&result_bgr, &fused_alpha)?;
    
    log::debug!("Stacking batch complete");
    Ok(final_img)
}

/// Memory-efficient variant: loads images one at a time from paths.
/// Each image is loaded, processed into the pyramid, then dropped before the next is loaded.
/// This prevents GPU VRAM exhaustion with large images (43MP BGRA float32 = ~672MB each).
fn stack_images_direct_from_paths(image_paths: &[PathBuf]) -> Result<Mat> {
    if image_paths.is_empty() {
        return Err(anyhow::anyhow!("No images to stack"));
    }
    if image_paths.len() == 1 {
        return Ok(load_image(&image_paths[0])?);
    }

    let levels = 7;
    
    let use_gpu = opencv::core::use_opencl().unwrap_or(false);
    log::info!("stack_images_direct_from_paths: {} images, OpenCL={}", image_paths.len(), use_gpu);
    
    let mut fused_pyramid: Vec<core::UMat> = Vec::new();
    let mut max_energies: Vec<core::UMat> = Vec::new();
    let mut fused_alpha = core::UMat::new(core::UMatUsageFlags::USAGE_DEFAULT);

    for (idx, path) in image_paths.iter().enumerate() {
        let filename = path.file_name().unwrap_or_default().to_string_lossy();
        log::debug!("Processing image {}/{}: {}", idx + 1, image_paths.len(), filename);
        
        // Load image — it will be dropped at the end of this iteration
        let img = match load_image(path) {
            Ok(img) => img,
            Err(e) => {
                log::warn!("Failed to load image {}: {}", filename, e);
                continue;
            }
        };
        
        process_image_into_pyramid(
            &img, idx, levels,
            &mut fused_pyramid, &mut max_energies, &mut fused_alpha,
        )?;
        // `img` is dropped here, freeing CPU memory before loading the next image
    }

    if fused_pyramid.is_empty() {
        return Err(anyhow::anyhow!("No valid images to stack"));
    }

    log::debug!("Collapsing pyramid...");
    let result_bgr = collapse_pyramid(&fused_pyramid)?;
    let final_img = assemble_final_image(&result_bgr, &fused_alpha)?;
    
    log::debug!("Stacking batch complete");
    Ok(final_img)
}

/// Process a single image into the fused Laplacian pyramid.
/// After this function returns, the caller can drop the source image to free memory.
/// Only the pyramid contributions (much smaller than the full image) are retained.
fn process_image_into_pyramid(
    img: &Mat,
    idx: usize,
    levels: i32,
    fused_pyramid: &mut Vec<core::UMat>,
    max_energies: &mut Vec<core::UMat>,
    fused_alpha: &mut core::UMat,
) -> Result<()> {
        
        // Ensure all images have 4 channels (BGRA)
        let img_normalized = if img.channels() == 3 {
            log::debug!("Converting 3-channel BGR to 4-channel BGRA");
            let mut bgra = Mat::default();
            crate::opencv_compat::cvt_color(img, &mut bgra, imgproc::COLOR_BGR2BGRA, 0)?;
            bgra
        } else {
            img.clone()
        };
        
        // Upload to GPU (UMat) and convert to float
        let mut float_img = core::UMat::new(core::UMatUsageFlags::USAGE_DEFAULT);
        img_normalized.get_umat(
            core::AccessFlag::ACCESS_READ,
            core::UMatUsageFlags::USAGE_DEFAULT,
        )?.convert_to(&mut float_img, core::CV_32F, 1.0, 0.0)?;

        // Separate BGR and alpha channels
        // Alpha is handled independently to avoid corruption by Laplacian operations
        let (pyramid_input, original_alpha) = extract_bgr_and_alpha(&float_img)?;

        let current_pyramid = generate_laplacian_pyramid(&pyramid_input, levels)?;

        if idx == 0 {
            *fused_pyramid = current_pyramid.clone();
            *fused_alpha = init_alpha(&original_alpha, pyramid_input.rows(), pyramid_input.cols())?;

            // Initialize max energies for all levels (Laplacian + base)
            for l in 0..=levels as usize {
                let energy = compute_sharpness_energy(&current_pyramid[l])?;
                max_energies.push(energy);
            }
            
            // Ensure base level is float
            let base_idx = levels as usize;
            let mut float_base = core::UMat::new(core::UMatUsageFlags::USAGE_DEFAULT);
            fused_pyramid[base_idx].convert_to(&mut float_base, core::CV_32F, 1.0, 0.0)?;
            fused_pyramid[base_idx] = float_base;
        } else {
            // Fuse Laplacian levels using winner-take-all based on sharpness energy
            for l in 0..levels as usize {
                let layer = &current_pyramid[l];
                let energy = compute_sharpness_energy(layer)?;
                
                let (fused_layer, fused_energy) = fuse_layer_with_alpha(
                    layer, &energy, &fused_pyramid[l], &max_energies[l], &original_alpha,
                )?;
                fused_pyramid[l] = fused_layer;
                max_energies[l] = fused_energy;
            }

            // Fuse base level (Gaussian)
            let base_idx = levels as usize;
            let base_layer = &current_pyramid[base_idx];
            let base_energy = compute_sharpness_energy(base_layer)?;
            
            let mut float_base_layer = core::UMat::new(core::UMatUsageFlags::USAGE_DEFAULT);
            base_layer.convert_to(&mut float_base_layer, core::CV_32F, 1.0, 0.0)?;
            
            let (fused_base, fused_base_energy) = fuse_layer_with_alpha(
                &float_base_layer, &base_energy, &fused_pyramid[base_idx], &max_energies[base_idx],
                &original_alpha,
            )?;
            fused_pyramid[base_idx] = fused_base;
            max_energies[base_idx] = fused_base_energy;

            // Update fused_alpha: AND-combine so only pixels opaque in ALL images remain opaque
            update_fused_alpha(fused_alpha, &original_alpha)?;
        }
    
    Ok(())
}

/// Extract BGR channels and alpha from a BGRA float UMat.
/// Returns (BGR 3-channel UMat, Option<alpha 1-channel UMat>).
fn extract_bgr_and_alpha(float_img: &core::UMat) -> Result<(core::UMat, Option<core::UMat>)> {
    if float_img.channels() == 4 {
        let mut channels = opencv::core::Vector::<core::UMat>::new();
        core::split(float_img, &mut channels)?;
        
        let alpha = channels.get(3)?;
        
        let mut bgr_channels = opencv::core::Vector::<core::UMat>::new();
        bgr_channels.push(channels.get(0)?);
        bgr_channels.push(channels.get(1)?);
        bgr_channels.push(channels.get(2)?);
        
        let mut bgr_img = core::UMat::new(core::UMatUsageFlags::USAGE_DEFAULT);
        core::merge(&bgr_channels, &mut bgr_img)?;
        
        Ok((bgr_img, Some(alpha)))
    } else {
        Ok((float_img.clone(), None))
    }
}

/// Initialize fused alpha from the first image's alpha, or create opaque alpha.
fn init_alpha(original_alpha: &Option<core::UMat>, rows: i32, cols: i32) -> Result<core::UMat> {
    if let Some(ref alpha) = original_alpha {
        Ok(alpha.clone())
    } else {
        Ok(core::UMat::new_rows_cols_with_default(
            rows, cols, core::CV_32F,
            core::Scalar::all(255.0),
            core::UMatUsageFlags::USAGE_DEFAULT,
        )?)
    }
}

/// Convert a UMat to single-channel grayscale, handling BGRA, BGR, and single-channel input.
fn to_grayscale(img: &core::UMat) -> Result<core::UMat> {
    let mut gray = core::UMat::new(core::UMatUsageFlags::USAGE_DEFAULT);
    match img.channels() {
        4 => crate::opencv_compat::cvt_color(img, &mut gray, imgproc::COLOR_BGRA2GRAY, 0)?,
        3 => crate::opencv_compat::cvt_color(img, &mut gray, imgproc::COLOR_BGR2GRAY, 0)?,
        _ => gray = img.clone(),
    }
    Ok(gray)
}

/// Compute sharpness energy for a pyramid layer using Laplacian + Gaussian blur.
fn compute_sharpness_energy(layer: &core::UMat) -> Result<core::UMat> {
    let gray = to_grayscale(layer)?;

    let mut laplacian = core::UMat::new(core::UMatUsageFlags::USAGE_DEFAULT);
    imgproc::laplacian(&gray, &mut laplacian, core::CV_32F, 5, 1.0, 0.0, core::BORDER_DEFAULT)?;

    let mut energy = core::UMat::new(core::UMatUsageFlags::USAGE_DEFAULT);
    core::absdiff(&laplacian, &core::Scalar::all(0.0), &mut energy)?;

    let mut blurred = core::UMat::new(core::UMatUsageFlags::USAGE_DEFAULT);
    crate::opencv_compat::gaussian_blur(&energy, &mut blurred, core::Size::new(3, 3), 0.0, 0.0,
        core::BORDER_DEFAULT)?;

    Ok(blurred)
}

/// Fuse a new layer into the fused pyramid using winner-take-all based on sharpness energy.
/// If alpha is present, energy is weighted by alpha to prevent transparent regions from winning.
/// Returns (updated fused layer, updated max energy).
fn fuse_layer_with_alpha(
    layer: &core::UMat,
    energy: &core::UMat,
    fused_layer: &core::UMat,
    max_energy: &core::UMat,
    original_alpha: &Option<core::UMat>,
) -> Result<(core::UMat, core::UMat)> {
    let mut new_fused = fused_layer.clone();
    let mut new_max_energy = max_energy.clone();

    if let Some(ref orig_alpha) = original_alpha {
        // Resize alpha to match layer size
        let mut alpha_resized = core::UMat::new(core::UMatUsageFlags::USAGE_DEFAULT);
        imgproc::resize(orig_alpha, &mut alpha_resized, layer.size()?, 0.0, 0.0,
            imgproc::INTER_LINEAR)?;
        
        // Weight energy by alpha [0,1] so transparent regions have zero energy
        let mut alpha_weight = core::UMat::new(core::UMatUsageFlags::USAGE_DEFAULT);
        alpha_resized.convert_to(&mut alpha_weight, core::CV_32F, 1.0 / 255.0, 0.0)?;
        
        let mut weighted_energy = core::UMat::new(core::UMatUsageFlags::USAGE_DEFAULT);
        core::multiply(energy, &alpha_weight, &mut weighted_energy, 1.0, -1)?;
        
        // Winner-take-all: copy where this image has higher weighted energy
        let mut mask = core::UMat::new(core::UMatUsageFlags::USAGE_DEFAULT);
        core::compare(&weighted_energy, max_energy, &mut mask, core::CMP_GT)?;
        
        layer.copy_to_masked(&mut new_fused, &mask)?;
        weighted_energy.copy_to_masked(&mut new_max_energy, &mask)?;
    } else {
        // No alpha: simple energy comparison
        let mut mask = core::UMat::new(core::UMatUsageFlags::USAGE_DEFAULT);
        core::compare(energy, max_energy, &mut mask, core::CMP_GT)?;
        
        layer.copy_to_masked(&mut new_fused, &mask)?;
        energy.copy_to_masked(&mut new_max_energy, &mask)?;
    }

    Ok((new_fused, new_max_energy))
}

/// Update fused alpha by AND-combining with a new image's alpha.
/// Only pixels opaque in ALL images remain opaque, ensuring the transparent border
/// is large enough to hide pyramid artifacts at edges.
fn update_fused_alpha(fused_alpha: &mut core::UMat, original_alpha: &Option<core::UMat>) -> Result<()> {
    if let Some(ref orig_alpha) = original_alpha {
        let mut orig_mask = core::UMat::new(core::UMatUsageFlags::USAGE_DEFAULT);
        core::compare(orig_alpha, &core::Scalar::all(0.0), &mut orig_mask, core::CMP_GT)?;
        
        let mut fused_mask = core::UMat::new(core::UMatUsageFlags::USAGE_DEFAULT);
        core::compare(fused_alpha as &core::UMat, &core::Scalar::all(0.0), &mut fused_mask, core::CMP_GT)?;
        
        let mut combined = core::UMat::new(core::UMatUsageFlags::USAGE_DEFAULT);
        core::bitwise_and(&orig_mask, &fused_mask, &mut combined, &core::no_array())?;
        
        combined.convert_to(fused_alpha, core::CV_32F, 1.0, 0.0)?;
    }
    Ok(())
}

/// Assemble the final BGRA image from collapsed BGR pyramid and fused alpha.
/// Erodes alpha slightly to hide pyramid artifacts at edges.
fn assemble_final_image(result_bgr: &core::UMat, fused_alpha: &core::UMat) -> Result<Mat> {
    let mut final_img_umat = result_bgr.clone();
    
    log::debug!("Assembling final image: BGR {}x{}, Alpha {}x{} (empty={})",
        result_bgr.cols(), result_bgr.rows(),
        fused_alpha.cols(), fused_alpha.rows(), fused_alpha.empty());

    if final_img_umat.channels() == 3 && !fused_alpha.empty() {
        let mut channels = opencv::core::Vector::<core::UMat>::new();
        core::split(&final_img_umat, &mut channels)?;

        // Resize alpha if needed
        let mut alpha = if fused_alpha.rows() != final_img_umat.rows() || fused_alpha.cols() != final_img_umat.cols() {
            let mut resized = core::UMat::new(core::UMatUsageFlags::USAGE_DEFAULT);
            let target_size = core::Size::new(final_img_umat.cols(), final_img_umat.rows());
            imgproc::resize(fused_alpha, &mut resized, target_size, 0.0, 0.0, imgproc::INTER_LINEAR)?;
            log::debug!("Resized fused_alpha from {}x{} to {}x{}", fused_alpha.cols(), fused_alpha.rows(), resized.cols(), resized.rows());
            resized
        } else {
            fused_alpha.clone()
        };

        // Erode alpha to push opaque boundary inward, hiding pyramid artifacts at edges
        let erode_size = 5;
        let kernel = imgproc::get_structuring_element(
            imgproc::MORPH_ELLIPSE,
            core::Size::new(erode_size * 2 + 1, erode_size * 2 + 1),
            core::Point::new(-1, -1),
        )?;
        let mut alpha_8u = core::UMat::new(core::UMatUsageFlags::USAGE_DEFAULT);
        alpha.convert_to(&mut alpha_8u, core::CV_8U, 1.0, 0.0)?;
        let mut alpha_eroded = core::UMat::new(core::UMatUsageFlags::USAGE_DEFAULT);
        imgproc::erode(&alpha_8u, &mut alpha_eroded, &kernel,
            core::Point::new(-1, -1), 1, core::BORDER_CONSTANT,
            imgproc::morphology_default_border_value()?)?;
        alpha_eroded.convert_to(&mut alpha, core::CV_32F, 1.0, 0.0)?;

        channels.push(alpha);
        core::merge(&channels, &mut final_img_umat)?;
    } else if final_img_umat.channels() == 3 {
        // No alpha tracked: add fully opaque alpha
        let mut channels = opencv::core::Vector::<core::UMat>::new();
        core::split(&final_img_umat, &mut channels)?;
        let opaque = core::UMat::new_rows_cols_with_default(
            final_img_umat.rows(), final_img_umat.cols(), core::CV_32F,
            core::Scalar::all(255.0), core::UMatUsageFlags::USAGE_DEFAULT)?;
        channels.push(opaque);
        core::merge(&channels, &mut final_img_umat)?;
    }
    
    // Convert to 8-bit and download from GPU
    let mut final_8u = core::UMat::new(core::UMatUsageFlags::USAGE_DEFAULT);
    final_img_umat.convert_to(&mut final_8u, core::CV_8U, 1.0, 0.0)?;
    let mut final_img = Mat::default();
    final_8u.get_mat(core::AccessFlag::ACCESS_READ)?.copy_to(&mut final_img)?;
    
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
        let mut up = core::UMat::new(core::UMatUsageFlags::USAGE_DEFAULT);
        imgproc::pyr_up(&current, &mut up, pyramid[i].size()?, core::BORDER_DEFAULT)?;

        if up.size()? != pyramid[i].size()? {
            let mut resized = core::UMat::new(core::UMatUsageFlags::USAGE_DEFAULT);
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

        let mut next = core::UMat::new(core::UMatUsageFlags::USAGE_DEFAULT);
        core::add(
            &up,
            &pyramid[i],
            &mut next,
            &core::UMat::new(core::UMatUsageFlags::USAGE_DEFAULT),
            -1,
        )?;
        
        // CRITICAL: Clip values to [0, 255] to prevent bright artifacts
        // When Laplacian details from different images are added, values can exceed valid range
        let mut clipped = core::UMat::new(core::UMatUsageFlags::USAGE_DEFAULT);
        core::min(&next, &core::Scalar::all(255.0), &mut clipped)?;
        let mut clipped2 = core::UMat::new(core::UMatUsageFlags::USAGE_DEFAULT);
        core::max(&clipped, &core::Scalar::all(0.0), &mut clipped2)?;
        
        current = clipped2;
    }
    Ok(current)
}