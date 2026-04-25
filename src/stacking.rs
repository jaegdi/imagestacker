use anyhow::Result;
use opencv::prelude::*;
use opencv::{core, imgproc};
use rayon::prelude::*;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

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

    // Disable OpenCL for the stacking pipeline to prevent CL_INVALID_COMMAND_QUEUE crashes.
    // The Laplacian pyramid stacking creates many temporary UMat buffers across multiple batches,
    // which can exhaust GPU memory and corrupt the OpenCL command queue (especially on GPUs with
    // limited VRAM). With OpenCL disabled, UMat operations transparently fall back to CPU.
    // CPU stacking is still fast and avoids the fatal OpenCL driver crashes.
    let opencl_was_enabled = crate::opencv_compat::use_opencl();
    if opencl_was_enabled {
        log::info!("Temporarily disabling OpenCL for stacking to prevent GPU memory exhaustion");
        crate::opencv_compat::set_use_opencl(false);
    }

    let result = stack_recursive(&reversed_paths, output_dir, 0, config, progress_cb.clone(), cancel_flag.clone());

    // Restore OpenCL state
    if opencl_was_enabled {
        log::info!("Re-enabling OpenCL after stacking");
        crate::opencv_compat::set_use_opencl(true);
    }
    
    let result = result?;

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

    let bunches_dir = Arc::new(output_dir.join("bunches"));
    std::fs::create_dir_all(bunches_dir.as_ref())?;

    // Pre-compute all chunk path lists (including overlaps) so batches can run in parallel.
    // Overlap images are simply re-loaded from disk in the adjacent chunk instead of being
    // carried in memory — a negligible I/O trade-off for full multi-core utilisation.
    let step = batch_size - OVERLAP;
    let mut chunks: Vec<(usize, Vec<PathBuf>)> = Vec::new();
    {
        let mut i = 0;
        let mut batch_idx = 0;
        while i < image_paths.len() {
            let end = (i + batch_size).min(image_paths.len());
            chunks.push((batch_idx, image_paths[i..end].to_vec()));
            batch_idx += 1;
            if end == image_paths.len() {
                break;
            }
            i += step;
        }
    }
    let total_batches = chunks.len();

    log::info!(
        "Level {}: Processing {} batches in parallel ({} images, batch_size={}, overlap={})",
        level, total_batches, image_paths.len(), batch_size, OVERLAP
    );

    let completed = Arc::new(AtomicUsize::new(0));

    // Process all chunks in parallel — rayon uses all available CPU cores.
    let results: Vec<(usize, Result<PathBuf, String>)> = chunks
        .into_par_iter()
        .map(|(idx, chunk_paths)| {
            // Check cancellation before starting this batch
            if cancel_flag
                .as_ref()
                .map(|f| f.load(Ordering::Relaxed))
                .unwrap_or(false)
            {
                return (idx, Err("Cancelled by user".to_string()));
            }

            log::info!("Level {}: Starting batch {} ({} images)", level, idx, chunk_paths.len());
            println!("  Level {}: Batch {} ({} images)...", level, idx, chunk_paths.len());

            let result = stack_images_direct_from_paths(&chunk_paths)
                .and_then(|mat| {
                    let filename = format!("L{}_B{:04}.png", level, idx);
                    let path = bunches_dir.join(&filename);
                    opencv::imgcodecs::imwrite(
                        path.to_str().unwrap_or_default(),
                        &mat,
                        &opencv::core::Vector::new(),
                    )?;
                    log::info!("  ✓ Level {}: Saved batch {} → {}", level, idx, filename);
                    println!("    ✓ Saved {}", filename);
                    Ok(path)
                })
                .map_err(|e| e.to_string());

            // Update shared progress counter (lock-free)
            let done = completed.fetch_add(1, Ordering::Relaxed) + 1;
            if let Some(ref cb) = progress_cb {
                if let Ok(mut cb_lock) = cb.lock() {
                    let pct = 10.0 + (done as f32 / total_batches as f32) * 70.0;
                    cb_lock(
                        format!("Stacking: {}/{} batches done (level {})", done, total_batches, level),
                        pct,
                    );
                }
            }

            (idx, result)
        })
        .collect();

    // Collect results in original order (rayon may deliver out-of-order)
    let mut indexed: Vec<(usize, PathBuf)> = Vec::with_capacity(results.len());
    for (idx, result) in results {
        match result {
            Ok(path) => indexed.push((idx, path)),
            Err(e) if e.contains("Cancelled") => {
                log::info!("Stacking cancelled by user during batch processing");
                return Err(anyhow::anyhow!("Operation cancelled by user"));
            }
            Err(e) => {
                return Err(anyhow::anyhow!("Batch {} failed: {}", idx, e));
            }
        }
    }
    indexed.sort_by_key(|(idx, _)| *idx);
    let intermediate_files: Vec<PathBuf> = indexed.into_iter().map(|(_, p)| p).collect();

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

#[allow(dead_code)]
fn stack_images_direct(images: &[Mat]) -> Result<Mat> {
    if images.is_empty() {
        return Err(anyhow::anyhow!("No images to stack"));
    }
    if images.len() == 1 {
        return Ok(images[0].clone());
    }

    let levels = 7;
    
    let use_gpu = crate::opencv_compat::use_opencl();
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
        
        // Flush OpenCL queue between images to prevent CL_INVALID_COMMAND_QUEUE errors.
        // Without this, async GPU operations can accumulate and exhaust the command queue,
        // especially with large batches or limited GPU memory.
        if use_gpu {
            crate::opencv_compat::finish();
        }
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
    
    let use_gpu = crate::opencv_compat::use_opencl();
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
        
        // Flush OpenCL queue between images to prevent CL_INVALID_COMMAND_QUEUE errors
        if use_gpu {
            crate::opencv_compat::finish();
        }
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
            *fused_alpha = init_alpha(&original_alpha, pyramid_input.rows(), pyramid_input.cols())?;

            // Parallel: compute initial max energies for all pyramid levels simultaneously.
            // Consume current_pyramid into owned items — UMat is Send, so rayon can assign
            // each level to a separate thread without requiring Sync.
            let n_levels = levels as usize + 1;
            let base_idx  = levels as usize;

            let init_results: Vec<Result<(usize, core::UMat, core::UMat)>> = current_pyramid
                .into_par_iter()
                .enumerate()
                .map(|(l, layer)| {
                    let energy = compute_sharpness_energy(&layer)?;
                    Ok((l, layer, energy))
                })
                .collect();

            *fused_pyramid = vec![core::UMat::new(core::UMatUsageFlags::USAGE_DEFAULT); n_levels];
            *max_energies  = vec![core::UMat::new(core::UMatUsageFlags::USAGE_DEFAULT); n_levels];

            for result in init_results {
                let (l, layer, energy) = result?;
                (*fused_pyramid)[l] = layer;
                (*max_energies)[l]  = energy;
            }

            // Convert base level to float
            let mut float_base = core::UMat::new(core::UMatUsageFlags::USAGE_DEFAULT);
            (*fused_pyramid)[base_idx].convert_to(&mut float_base, core::CV_32F, 1.0, 0.0)?;
            (*fused_pyramid)[base_idx] = float_base;
        } else {
            let n_levels = levels as usize + 1;
            let base_idx  = levels as usize;

            // Pre-resize alpha to every pyramid level size (sequential, cheap).
            // This avoids sharing the alpha UMat across rayon threads (UMat is not Sync).
            let alphas_per_level: Vec<Option<core::UMat>> = if let Some(ref alpha) = original_alpha {
                (0..n_levels)
                    .map(|l| {
                        let layer_size = current_pyramid[l].size().ok()?;
                        let mut resized = core::UMat::new(core::UMatUsageFlags::USAGE_DEFAULT);
                        imgproc::resize(alpha, &mut resized, layer_size, 0.0, 0.0,
                            imgproc::INTER_LINEAR).ok()?;
                        Some(resized)
                    })
                    .collect()
            } else {
                vec![None; n_levels]
            };

            // Convert base level to float before consuming current_pyramid
            let mut cur_base_float = core::UMat::new(core::UMatUsageFlags::USAGE_DEFAULT);
            current_pyramid[base_idx].convert_to(&mut cur_base_float, core::CV_32F, 1.0, 0.0)?;

            // Consume all three Vecs to build owned tuples for parallel processing.
            // into_par_iter() transfers ownership per item — no Sync needed on UMat.
            let mut cur_levels = current_pyramid;
            cur_levels[base_idx] = cur_base_float;
            let fused_levels: Vec<core::UMat> = std::mem::take(fused_pyramid);
            let max_e_levels:  Vec<core::UMat> = std::mem::take(max_energies);

            let level_tuples: Vec<(usize, core::UMat, core::UMat, core::UMat, Option<core::UMat>)> =
                cur_levels
                    .into_iter()
                    .zip(fused_levels)
                    .zip(max_e_levels)
                    .zip(alphas_per_level)
                    .enumerate()
                    .map(|(l, (((cur, fused), max_e), alpha))| (l, cur, fused, max_e, alpha))
                    .collect();

            // Parallel: compute sharpness energy + fuse each of the 8 pyramid levels.
            // On an 8-core machine this computes all levels simultaneously instead of serially.
            let level_results: Vec<Result<(usize, core::UMat, core::UMat)>> = level_tuples
                .into_par_iter()
                .map(|(l, cur_layer, fused_layer, max_energy, level_alpha)| {
                    let energy = compute_sharpness_energy(&cur_layer)?;
                    let (new_fused, new_energy) = fuse_layer_with_preresized_alpha(
                        &cur_layer, &energy, &fused_layer, &max_energy, &level_alpha,
                    )?;
                    Ok((l, new_fused, new_energy))
                })
                .collect();

            // Reconstruct fused_pyramid and max_energies from parallel results
            *fused_pyramid = vec![core::UMat::new(core::UMatUsageFlags::USAGE_DEFAULT); n_levels];
            *max_energies  = vec![core::UMat::new(core::UMatUsageFlags::USAGE_DEFAULT); n_levels];

            for result in level_results {
                let (l, new_fused, new_energy) = result?;
                (*fused_pyramid)[l] = new_fused;
                (*max_energies)[l]  = new_energy;
            }

            // AND-combine alpha: only pixels opaque in ALL images remain opaque
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

/// Variant of `fuse_layer_with_alpha` that accepts an already-resized alpha channel.
/// Used in the parallel pyramid-level processing path to avoid sharing the original
/// alpha UMat across rayon threads (UMat is Send but not Sync).
fn fuse_layer_with_preresized_alpha(
    layer: &core::UMat,
    energy: &core::UMat,
    fused_layer: &core::UMat,
    max_energy: &core::UMat,
    level_alpha: &Option<core::UMat>, // already resized to `layer`'s dimensions
) -> Result<(core::UMat, core::UMat)> {
    let mut new_fused = fused_layer.clone();
    let mut new_max_energy = max_energy.clone();

    if let Some(ref alpha_resized) = level_alpha {
        // Normalize alpha from [0,255] float to [0,1]
        let mut alpha_weight = core::UMat::new(core::UMatUsageFlags::USAGE_DEFAULT);
        alpha_resized.convert_to(&mut alpha_weight, core::CV_32F, 1.0 / 255.0, 0.0)?;

        let mut weighted_energy = core::UMat::new(core::UMatUsageFlags::USAGE_DEFAULT);
        core::multiply(energy, &alpha_weight, &mut weighted_energy, 1.0, -1)?;

        let mut mask = core::UMat::new(core::UMatUsageFlags::USAGE_DEFAULT);
        core::compare(&weighted_energy, max_energy, &mut mask, core::CMP_GT)?;

        layer.copy_to_masked(&mut new_fused, &mask)?;
        weighted_energy.copy_to_masked(&mut new_max_energy, &mask)?;
    } else {
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
    // Flush OpenCL queue before GPU→CPU transfer to ensure all operations are complete
    crate::opencv_compat::finish();
    let mut final_8u = core::UMat::new(core::UMatUsageFlags::USAGE_DEFAULT);
    final_img_umat.convert_to(&mut final_8u, core::CV_8U, 1.0, 0.0)?;
    crate::opencv_compat::finish();
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