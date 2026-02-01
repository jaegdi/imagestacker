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

    let batch_size = config.batch_config.stacking_batch_size;
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

    let levels = 7; // Increased from 6 for finer detail preservation
    let mut fused_pyramid: Vec<core::UMat> = Vec::new();
    let mut max_energies: Vec<core::UMat> = Vec::new();

    for (idx, img) in images.iter().enumerate() {
        log::info!("Processing image {}/{} for stacking", idx + 1, images.len());
        
        // Ensure all images have 4 channels (BGRA) for consistent processing
        let img_normalized = if img.channels() == 3 {
            log::info!("Converting 3-channel BGR image to 4-channel BGRA");
            let mut bgra = Mat::default();
            imgproc::cvt_color(
                img,
                &mut bgra,
                imgproc::COLOR_BGR2BGRA,
                0,
                core::AlgorithmHint::ALGO_HINT_DEFAULT,
            )?;
            log::info!("Converted to {} channels", bgra.channels());
            bgra
        } else {
            img.clone()
        };
        
        let mut float_img = core::UMat::new(core::UMatUsageFlags::USAGE_DEFAULT);
        img_normalized.get_umat(
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
                if layer.channels() == 4 {
                    // BGRA to Gray
                    imgproc::cvt_color(
                        layer,
                        &mut gray,
                        imgproc::COLOR_BGRA2GRAY,
                        0,
                        core::AlgorithmHint::ALGO_HINT_DEFAULT,
                    )?;
                } else if layer.channels() == 3 {
                    // BGR to Gray
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
                    5, // Increased from 3 for better sharpness detection
                    1.0,
                    0.0,
                    core::BORDER_DEFAULT,
                )?;

                let mut energy = core::UMat::new(core::UMatUsageFlags::USAGE_DEFAULT);
                core::absdiff(&laplacian, &core::Scalar::all(0.0), &mut energy)?;

                // Apply smaller blur to preserve sharp regions better
                let mut blurred_energy = core::UMat::new(core::UMatUsageFlags::USAGE_DEFAULT);
                imgproc::gaussian_blur(
                    &energy,
                    &mut blurred_energy,
                    core::Size::new(3, 3), // Reduced from 5x5 to preserve local sharpness
                    0.0,
                    0.0,
                    core::BORDER_DEFAULT,
                    core::AlgorithmHint::ALGO_HINT_DEFAULT,
                )?;

                max_energies.push(blurred_energy);
            }

            // For the base level (Gaussian), also use winner-take-all based on sharpness
            let base_idx = levels as usize;
            let base_layer = &current_pyramid[base_idx];
            
            // Compute energy for base level
            let mut gray = core::UMat::new(core::UMatUsageFlags::USAGE_DEFAULT);
            if base_layer.channels() == 4 {
                imgproc::cvt_color(
                    base_layer,
                    &mut gray,
                    imgproc::COLOR_BGRA2GRAY,
                    0,
                    core::AlgorithmHint::ALGO_HINT_DEFAULT,
                )?;
            } else if base_layer.channels() == 3 {
                imgproc::cvt_color(
                    base_layer,
                    &mut gray,
                    imgproc::COLOR_BGR2GRAY,
                    0,
                    core::AlgorithmHint::ALGO_HINT_DEFAULT,
                )?;
            } else {
                gray = base_layer.clone();
            }

            let mut laplacian = core::UMat::new(core::UMatUsageFlags::USAGE_DEFAULT);
            imgproc::laplacian(
                &gray,
                &mut laplacian,
                core::CV_32F,
                5,
                1.0,
                0.0,
                core::BORDER_DEFAULT,
            )?;

            let mut base_energy = core::UMat::new(core::UMatUsageFlags::USAGE_DEFAULT);
            core::absdiff(&laplacian, &core::Scalar::all(0.0), &mut base_energy)?;

            let mut blurred_base_energy = core::UMat::new(core::UMatUsageFlags::USAGE_DEFAULT);
            imgproc::gaussian_blur(
                &base_energy,
                &mut blurred_base_energy,
                core::Size::new(3, 3),
                0.0,
                0.0,
                core::BORDER_DEFAULT,
                core::AlgorithmHint::ALGO_HINT_DEFAULT,
            )?;

            max_energies.push(blurred_base_energy);
            
            let mut float_base = core::UMat::new(core::UMatUsageFlags::USAGE_DEFAULT);
            fused_pyramid[base_idx].convert_to(&mut float_base, core::CV_32F, 1.0, 0.0)?;
            fused_pyramid[base_idx] = float_base;
        } else {
            // Fuse with current image
            for l in 0..levels as usize {
                let layer = &current_pyramid[l];

                // Compute energy using Laplacian
                let mut gray = core::UMat::new(core::UMatUsageFlags::USAGE_DEFAULT);
                if layer.channels() == 4 {
                    imgproc::cvt_color(
                        layer,
                        &mut gray,
                        imgproc::COLOR_BGRA2GRAY,
                        0,
                        core::AlgorithmHint::ALGO_HINT_DEFAULT,
                    )?;
                } else if layer.channels() == 3 {
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
                    5, // Increased from 3 for better sharpness detection
                    1.0,
                    0.0,
                    core::BORDER_DEFAULT,
                )?;

                let mut energy = core::UMat::new(core::UMatUsageFlags::USAGE_DEFAULT);
                core::absdiff(&laplacian, &core::Scalar::all(0.0), &mut energy)?;

                // Apply smaller blur to preserve sharp regions better
                let mut blurred_energy = core::UMat::new(core::UMatUsageFlags::USAGE_DEFAULT);
                imgproc::gaussian_blur(
                    &energy,
                    &mut blurred_energy,
                    core::Size::new(3, 3), // Reduced from 5x5 to preserve local sharpness
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

            // Also use winner-take-all for base level instead of averaging
            let base_idx = levels as usize;
            let base_layer = &current_pyramid[base_idx];
            
            // Compute energy for base level
            let mut gray = core::UMat::new(core::UMatUsageFlags::USAGE_DEFAULT);
            if base_layer.channels() == 4 {
                imgproc::cvt_color(
                    base_layer,
                    &mut gray,
                    imgproc::COLOR_BGRA2GRAY,
                    0,
                    core::AlgorithmHint::ALGO_HINT_DEFAULT,
                )?;
            } else if base_layer.channels() == 3 {
                imgproc::cvt_color(
                    base_layer,
                    &mut gray,
                    imgproc::COLOR_BGR2GRAY,
                    0,
                    core::AlgorithmHint::ALGO_HINT_DEFAULT,
                )?;
            } else {
                gray = base_layer.clone();
            }

            let mut laplacian = core::UMat::new(core::UMatUsageFlags::USAGE_DEFAULT);
            imgproc::laplacian(
                &gray,
                &mut laplacian,
                core::CV_32F,
                5,
                1.0,
                0.0,
                core::BORDER_DEFAULT,
            )?;

            let mut base_energy = core::UMat::new(core::UMatUsageFlags::USAGE_DEFAULT);
            core::absdiff(&laplacian, &core::Scalar::all(0.0), &mut base_energy)?;

            let mut blurred_base_energy = core::UMat::new(core::UMatUsageFlags::USAGE_DEFAULT);
            imgproc::gaussian_blur(
                &base_energy,
                &mut blurred_base_energy,
                core::Size::new(3, 3),
                0.0,
                0.0,
                core::BORDER_DEFAULT,
                core::AlgorithmHint::ALGO_HINT_DEFAULT,
            )?;

            // Update base level where energy is higher
            let mut base_mask = core::UMat::new(core::UMatUsageFlags::USAGE_DEFAULT);
            core::compare(&blurred_base_energy, &max_energies[base_idx], &mut base_mask, core::CMP_GT)?;

            // Convert base layer to float before copying
            let mut float_base_layer = core::UMat::new(core::UMatUsageFlags::USAGE_DEFAULT);
            base_layer.convert_to(&mut float_base_layer, core::CV_32F, 1.0, 0.0)?;
            
            float_base_layer.copy_to_masked(&mut fused_pyramid[base_idx], &base_mask)?;
            blurred_base_energy.copy_to_masked(&mut max_energies[base_idx], &base_mask)?;
        }
    }

    log::info!("Collapsing pyramid...");
    // Base level now uses winner-take-all, no averaging needed

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