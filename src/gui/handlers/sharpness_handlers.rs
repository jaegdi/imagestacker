//! Sharpness detection handlers
//!
//! Handles separate sharpness detection workflow with caching to YAML files

use crate::gui::state::ImageStacker;
use crate::messages::Message;
use crate::sharpness_cache::{prepare_sharpness_cache_dir, SharpnessInfo};
use iced::Task;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

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
        self.progress_message = "Starting sharpness detection...".to_string();

        let images = self.images.clone();
        let output_dir = images[0]
            .parent()
            .unwrap_or(std::path::Path::new("."))
            .to_path_buf();
        let config = self.config.clone();
        let cancel_flag = self.cancel_flag.clone();
        
        // Reset cancel flag
        cancel_flag.store(false, Ordering::Relaxed);

        Task::run(
            iced::stream::channel(100, move |mut sender| async move {
                std::thread::spawn(move || {
                    use rayon::prelude::*;

                    // Prepare sharpness cache directory
                    let sharpness_dir = match prepare_sharpness_cache_dir(&output_dir) {
                        Ok(dir) => dir,
                        Err(e) => {
                            let _ = sender.try_send(Message::SharpnessDetectionDone(
                                Err(format!("Failed to prepare sharpness directory: {}", e))
                            ));
                            return;
                        }
                    };

                    let total = images.len();
                    log::info!("🔍 Starting sharpness detection for {} images", total);
                    log::info!("   Grid size: {}x{}", config.sharpness_grid_size, config.sharpness_grid_size);
                    log::info!("   Output: {}", sharpness_dir.display());

                    // Atomic counter for progress tracking across parallel threads
                    let completed = Arc::new(AtomicUsize::new(0));
                    let progress_sender = sender.clone();

                    // Process images in parallel
                    let results: Vec<Result<PathBuf, String>> = images
                        .par_iter()
                        .map(|image_path| {
                            // Check for cancellation
                            if cancel_flag.load(Ordering::Relaxed) {
                                return Err("Cancelled by user".to_string());
                            }

                            let image_filename = image_path.file_name()
                                .unwrap_or_default()
                                .to_string_lossy()
                                .to_string();

                            // Load image
                            let img = opencv::imgcodecs::imread(
                                image_path.to_str().ok_or_else(|| "Invalid path".to_string())?,
                                opencv::imgcodecs::IMREAD_COLOR,
                            ).map_err(|e| format!("Failed to load {}: {}", image_path.display(), e))?;

                            // Compute sharpness - GPU concurrency bounded by gpu_semaphore()
                            let _gpu_permit = crate::alignment::gpu_semaphore().acquire();
                            let (max_regional, global_sharpness, sharp_region_count) = 
                                crate::sharpness::compute_regional_sharpness_auto(&img, config.sharpness_grid_size as i32)
                                    .map_err(|e| format!("Failed to compute sharpness for {}: {}", image_path.display(), e))?;
                            drop(_gpu_permit);

                            // Create sharpness info
                            use opencv::core::MatTraitConst;
                            let image_size = (img.cols(), img.rows());
                            
                            let info = SharpnessInfo::new(
                                image_filename.clone(),
                                max_regional,
                                global_sharpness,
                                sharp_region_count as f64,
                                config.sharpness_grid_size as usize,
                                image_size,
                            );

                            // Save to YAML file
                            let yaml_filename = SharpnessInfo::yaml_filename_for_image(image_path);
                            let yaml_path = sharpness_dir.join(&yaml_filename);
                            
                            info.save_to_file(&yaml_path)
                                .map_err(|e| format!("Failed to save YAML for {}: {}", image_path.display(), e))?;

                            // Update progress
                            let done = completed.fetch_add(1, Ordering::Relaxed) + 1;
                            let pct = (done as f32 / total as f32) * 100.0;
                            log::debug!(
                                "[{}/{}] {} -> max: {:.2}, global: {:.2}, sharp_regions: {}",
                                done, total, image_filename, max_regional, global_sharpness, sharp_region_count
                            );
                            
                            // Send progress update (non-blocking, ignore if channel is full)
                            let msg = format!("Sharpness: {}/{} ({})", done, total, image_filename);
                            let _ = progress_sender.clone().try_send(Message::ProgressUpdate(msg, pct));

                            Ok(yaml_path)
                        })
                        .collect();

                    // Check if cancelled
                    if cancel_flag.load(Ordering::Relaxed) {
                        let _ = sender.try_send(Message::SharpnessDetectionDone(
                            Err("Operation cancelled by user".to_string())
                        ));
                        return;
                    }

                    // Collect successful results
                    let mut yaml_paths = Vec::new();
                    for result in results {
                        match result {
                            Ok(path) => yaml_paths.push(path),
                            Err(e) => {
                                log::warn!("Sharpness detection error: {}", e);
                            }
                        }
                    }

                    if yaml_paths.is_empty() {
                        let _ = sender.try_send(Message::SharpnessDetectionDone(
                            Err("No images could be analyzed".to_string())
                        ));
                        return;
                    }

                    log::info!("✅ Sharpness detection complete: {}/{} images", yaml_paths.len(), total);
                    let _ = sender.try_send(Message::SharpnessDetectionDone(Ok(yaml_paths)));
                });
            }),
            |msg| msg,
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
