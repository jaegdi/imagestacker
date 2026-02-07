//! Sharpness detection handlers
//!
//! Handles separate sharpness detection workflow with caching to YAML files

use crate::gui::state::ImageStacker;
use crate::messages::Message;
use crate::sharpness_cache::{prepare_sharpness_cache_dir, SharpnessInfo};
use iced::Task;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

impl ImageStacker {
    /// Handle DetectSharpness - start sharpness detection workflow
    pub fn handle_detect_sharpness(&mut self) -> Task<Message> {
        if self.images.is_empty() {
            self.status = "No images to analyze".to_string();
            return Task::none();
        }

        if self.is_processing {
            self.status = "Already processing...".to_string();
            return Task::none();
        }

        self.is_processing = true;
        self.status = "Detecting sharpness...".to_string();
        self.progress_value = 0.0;

        let images = self.images.clone();
        let output_dir = images[0]
            .parent()
            .unwrap_or(std::path::Path::new("."))
            .to_path_buf();
        let config = self.config.clone();
        let cancel_flag = self.cancel_flag.clone();
        
        // Reset cancel flag
        cancel_flag.store(false, Ordering::Relaxed);

        Task::perform(
            async move {
                detect_sharpness_task(images, output_dir, config, cancel_flag).await
            },
            |result| Message::SharpnessDetectionDone(result),
        )
    }

    /// Handle SharpnessDetectionDone - process results of sharpness detection
    pub fn handle_sharpness_detection_done(&mut self, result: Result<Vec<PathBuf>, String>) -> Task<Message> {
        self.is_processing = false;
        self.progress_value = 0.0;
        self.progress_message.clear();

        match result {
            Ok(yaml_paths) => {
                self.sharpness_images = yaml_paths.clone();
                self.status = format!("Sharpness detection complete: {} images analyzed", yaml_paths.len());
                
                // Trigger refresh to show sharpness pane
                Task::done(Message::RefreshPanes)
            }
            Err(e) => {
                self.status = format!("Sharpness detection failed: {}", e);
                log::error!("Sharpness detection error: {}", e);
                Task::none()
            }
        }
    }
}

/// Async task to detect sharpness for all images and save to YAML
async fn detect_sharpness_task(
    images: Vec<PathBuf>,
    output_dir: PathBuf,
    config: crate::config::ProcessingConfig,
    cancel_flag: Arc<AtomicBool>,
) -> Result<Vec<PathBuf>, String> {
    use rayon::prelude::*;
    
    // Prepare sharpness cache directory
    let sharpness_dir = prepare_sharpness_cache_dir(&output_dir)
        .map_err(|e| format!("Failed to prepare sharpness directory: {}", e))?;

    log::info!("🔍 Starting sharpness detection for {} images", images.len());
    log::info!("   Grid size: {}x{}", config.sharpness_grid_size, config.sharpness_grid_size);
    log::info!("   Output: {}", sharpness_dir.display());

    // Process images in parallel
    let results: Vec<Result<PathBuf, String>> = images
        .par_iter()
        .enumerate()
        .map(|(idx, image_path)| {
            // Check for cancellation
            if cancel_flag.load(Ordering::Relaxed) {
                return Err("Cancelled by user".to_string());
            }

            // Load image
            let img = opencv::imgcodecs::imread(
                image_path.to_str().ok_or_else(|| "Invalid path".to_string())?,
                opencv::imgcodecs::IMREAD_COLOR,
            ).map_err(|e| format!("Failed to load {}: {}", image_path.display(), e))?;

            // Compute sharpness with OpenCL mutex for thread safety
            use std::sync::Mutex;
            static OPENCL_MUTEX: Mutex<()> = Mutex::new(());
            let _lock = OPENCL_MUTEX.lock().unwrap();
            let (max_regional, global_sharpness, sharp_region_count) = 
                crate::sharpness::compute_regional_sharpness_auto(&img, config.sharpness_grid_size as i32)
                    .map_err(|e| format!("Failed to compute sharpness for {}: {}", image_path.display(), e))?;
            drop(_lock);

            // Create sharpness info
            let image_filename = image_path.file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string();
            
            use opencv::core::MatTraitConst;
            let image_size = (img.cols(), img.rows());
            
            let info = SharpnessInfo::new(
                image_filename.clone(),
                max_regional,
                global_sharpness,
                sharp_region_count as f64,  // Convert usize to f64 for storage
                config.sharpness_grid_size as usize,
                image_size,
            );

            // Save to YAML file
            let yaml_filename = SharpnessInfo::yaml_filename_for_image(image_path);
            let yaml_path = sharpness_dir.join(&yaml_filename);
            
            info.save_to_file(&yaml_path)
                .map_err(|e| format!("Failed to save YAML for {}: {}", image_path.display(), e))?;

            log::debug!(
                "[{}/{}] {} -> max: {:.2}, global: {:.2}, sharp_regions: {}",
                idx + 1,
                images.len(),
                image_filename,
                max_regional,
                global_sharpness,
                sharp_region_count
            );

            Ok(yaml_path)
        })
        .collect();

    // Check if cancelled
    if cancel_flag.load(Ordering::Relaxed) {
        return Err("Operation cancelled by user".to_string());
    }

    // Collect successful results
    let mut yaml_paths = Vec::new();
    for result in results {
        match result {
            Ok(path) => yaml_paths.push(path),
            Err(e) => {
                log::warn!("Sharpness detection error: {}", e);
                // Continue with other images
            }
        }
    }

    if yaml_paths.is_empty() {
        return Err("No images could be analyzed".to_string());
    }

    log::info!("✅ Sharpness detection complete: {}/{} images", yaml_paths.len(), images.len());
    
    Ok(yaml_paths)
}
