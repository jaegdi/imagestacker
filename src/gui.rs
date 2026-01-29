use iced::widget::{
    button, checkbox, column, container, image as iced_image, row, scrollable, slider, text, text_input,
};
use iced::Length;
use iced::{Element, Task, Theme};
use opencv::prelude::*;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};
use std::time::Duration;

use crate::alignment;
use crate::config::{FeatureDetector, ProcessingConfig};
use crate::messages::Message;
use crate::settings::{load_settings, save_settings};
use crate::stacking;
use crate::thumbnail;
use crate::system_info;

pub struct ImageStacker {
    images: Vec<PathBuf>,
    aligned_images: Vec<PathBuf>,
    bunch_images: Vec<PathBuf>,
    final_images: Vec<PathBuf>,
    thumbnail_cache: Arc<RwLock<HashMap<PathBuf, iced::widget::image::Handle>>>,
    status: String,
    preview_handle: Option<iced::widget::image::Handle>,
    result_mat: Option<Mat>,
    crop_rect: Option<opencv::core::Rect>,
    is_processing: bool,
    // New: Configuration and progress
    config: ProcessingConfig,
    show_settings: bool,
    progress_message: String,
    progress_value: f32,
    // Image preview
    preview_image_path: Option<PathBuf>,
    preview_loading: bool,
    preview_is_thumbnail: bool,
    // Scroll position tracking
    imported_scroll_offset: f32,
    aligned_scroll_offset: f32,
    bunches_scroll_offset: f32,
    final_scroll_offset: f32,
}

impl Default for ImageStacker {
    fn default() -> Self {
        Self {
            images: Vec::new(),
            aligned_images: Vec::new(),
            bunch_images: Vec::new(),
            final_images: Vec::new(),
            thumbnail_cache: Arc::new(RwLock::new(HashMap::new())),
            status: "Ready".to_string(),
            preview_handle: None,
            result_mat: None,
            crop_rect: None,
            is_processing: false,
            config: load_settings(),
            show_settings: false,
            progress_message: String::new(),
            progress_value: 0.0,
            preview_image_path: None,
            preview_loading: false,
            preview_is_thumbnail: false,
            imported_scroll_offset: 0.0,
            aligned_scroll_offset: 0.0,
            bunches_scroll_offset: 0.0,
            final_scroll_offset: 0.0,
        }
    }
}

impl ImageStacker {
    fn create_processing_config(&self) -> ProcessingConfig {
        self.config.clone()
    }

    pub fn update(&mut self, message: Message) -> Task<Message> {
        let task = match message {
            Message::AddImages => Task::perform(
                async {
                    let files = rfd::AsyncFileDialog::new()
                        .add_filter("Images", &["jpg", "jpeg", "png", "tif", "tiff"])
                        .pick_files()
                        .await;

                    if let Some(files) = files {
                        let paths = files.into_iter().map(|f| f.path().to_path_buf()).collect();
                        Message::ImagesSelected(paths)
                    } else {
                        Message::None
                    }
                },
                |msg| msg,
            ),
            Message::AddFolder => Task::perform(
                async {
                    let folder = rfd::AsyncFileDialog::new().pick_folder().await;

                    if let Some(folder) = folder {
                        let path = folder.path();
                        let mut paths = Vec::new();
                        if let Ok(entries) = std::fs::read_dir(path) {
                            for entry in entries.flatten() {
                                let p = entry.path();
                                if p.is_file() {
                                    if let Some(ext) = p.extension().and_then(|e| e.to_str()) {
                                        let ext = ext.to_lowercase();
                                        if ["jpg", "jpeg", "png", "tif", "tiff"]
                                            .contains(&ext.as_str())
                                        {
                                            paths.push(p);
                                        }
                                    }
                                }
                            }
                        }
                        paths.sort();
                        Message::ImagesSelected(paths)
                    } else {
                        Message::None
                    }
                },
                |msg| msg,
            ),
            Message::ImagesSelected(paths) => {
                // Clear all existing image lists to start a new project
                self.images.clear();
                self.aligned_images.clear();
                self.bunch_images.clear();
                self.final_images.clear();
                self.preview_handle = None;
                self.result_mat = None;
                self.crop_rect = None;

                // Clear thumbnail cache
                {
                    let mut cache = self.thumbnail_cache.write().unwrap();
                    cache.clear();
                }

                // Add new images
                self.images.extend(paths.clone());
                self.status = format!("Loaded {} images", self.images.len());

                let cache = self.thumbnail_cache.clone();
                // Generate thumbnails in parallel using rayon
                Task::run(
                    iced::stream::channel(100, move |sender| async move {
                        std::thread::spawn(move || {
                            use rayon::prelude::*;
                            let sender = Arc::new(std::sync::Mutex::new(sender));
                            
                            paths.par_iter().for_each(|path: &PathBuf| {
                                if let Ok(handle) = thumbnail::generate_thumbnail(path) {
                                    let mut locked = cache.write().unwrap();
                                    locked.insert(path.clone(), handle.clone());
                                    drop(locked);
                                    if let Ok(mut sender_lock) = sender.lock() {
                                        let _ = sender_lock.try_send(Message::ThumbnailUpdated(path.clone(), handle));
                                    }
                                }
                            });
                        });
                    }),
                    |msg| msg,
                )
            }
            Message::ThumbnailUpdated(path, _handle) => {
                log::trace!("Thumbnail updated for {}", path.display());
                Task::none()
            }
            Message::InternalPathsScanned(aligned, bunches, final_imgs, paths_to_process) => {
                self.aligned_images = aligned;
                self.bunch_images = bunches;
                self.final_images = final_imgs;

                if paths_to_process.is_empty() {
                    return Task::none();
                }

                let cache = self.thumbnail_cache.clone();
                // Generate thumbnails in parallel using rayon
                Task::run(
                    iced::stream::channel(100, move |sender| async move {
                        std::thread::spawn(move || {
                            use rayon::prelude::*;
                            let sender = Arc::new(std::sync::Mutex::new(sender));
                            
                            paths_to_process.par_iter().for_each(|path: &PathBuf| {
                                if let Ok(handle) = thumbnail::generate_thumbnail(path) {
                                    let mut locked = cache.write().unwrap();
                                    locked.insert(path.clone(), handle.clone());
                                    drop(locked);
                                    if let Ok(mut sender_lock) = sender.lock() {
                                        let _ = sender_lock.try_send(Message::ThumbnailUpdated(path.clone(), handle));
                                    }
                                }
                            });
                        });
                    }),
                    |msg| msg,
                )
            }
            Message::AlignImages => {
                if let Some(first_path) = self.images.first() {
                    let output_dir = first_path
                        .parent()
                        .unwrap_or(std::path::Path::new("."))
                        .to_path_buf();
                    let aligned_dir = output_dir.join("aligned");

                    if aligned_dir.exists() && aligned_dir.is_dir() {
                        // Check if it contains images
                        let has_images = std::fs::read_dir(&aligned_dir)
                            .map(|entries| {
                                entries.flatten().any(|e| {
                                    let p = e.path();
                                    p.is_file()
                                        && p.extension()
                                            .and_then(|ext| ext.to_str())
                                            .map(|ext| {
                                                ["jpg", "jpeg", "png", "tif", "tiff"]
                                                    .contains(&ext.to_lowercase().as_str())
                                            })
                                            .unwrap_or(false)
                                })
                            })
                            .unwrap_or(false);

                        if has_images {
                            return Task::perform(
                                async move {
                                    let confirmed = rfd::AsyncMessageDialog::new()
                                        .set_title("Reuse Aligned Images?")
                                        .set_description("Aligned images already exist. Use them instead of re-aligning?")
                                        .set_buttons(rfd::MessageButtons::YesNo)
                                        .show()
                                        .await;
                                    Message::AlignImagesConfirmed(
                                        confirmed == rfd::MessageDialogResult::Yes,
                                    )
                                },
                                |msg| msg,
                            );
                        }
                    }
                    Task::done(Message::AlignImagesConfirmed(false))
                } else {
                    self.status = "No images loaded".to_string();
                    Task::none()
                }
            }
            Message::AlignImagesConfirmed(reuse) => {
                if let Some(first_path) = self.images.first() {
                    let output_dir = first_path
                        .parent()
                        .unwrap_or(std::path::Path::new("."))
                        .to_path_buf();

                    if reuse {
                        log::info!(
                            "User confirmed reuse of existing aligned images in {}",
                            output_dir.display()
                        );
                        self.status = "Using existing aligned images".to_string();
                        Task::done(Message::RefreshPanes)
                    } else {
                        // Clean up existing aligned images before starting new alignment
                        let aligned_dir = output_dir.join("aligned");
                        if aligned_dir.exists() {
                            log::info!("Cleaning up existing aligned images in {}", aligned_dir.display());
                            if let Err(e) = std::fs::remove_dir_all(&aligned_dir) {
                                log::warn!("Failed to remove aligned directory: {}", e);
                            }
                            // Recreate the directory
                            if let Err(e) = std::fs::create_dir_all(&aligned_dir) {
                                log::warn!("Failed to recreate aligned directory: {}", e);
                            }
                        }
                        
                        self.status = "Aligning images...".to_string();
                        self.progress_message = "Starting alignment...".to_string();
                        self.progress_value = 0.0;
                        let images_paths = self.images.clone();
                        let proc_config = self.create_processing_config();
                        self.is_processing = true;  // Mark as processing

                        Task::run(
                            iced::stream::channel(100, move |mut sender| async move {
                                // Run processing in a separate thread
                                std::thread::spawn(move || {
                                    let progress_cb = std::sync::Arc::new(std::sync::Mutex::new(
                                        {
                                            let mut sender_clone = sender.clone();
                                            move |msg: String, pct: f32| {
                                                let _ = sender_clone.try_send(Message::ProgressUpdate(msg, pct));
                                            }
                                        }
                                    ));

                                    let result = alignment::align_images(
                                        &images_paths, 
                                        &output_dir,
                                        &proc_config,
                                        Some(progress_cb),
                                    );
                                    
                                    // Send final result
                                    match result {
                                        Ok(rect) => {
                                            let _ = sender.try_send(Message::AlignmentDone(Ok(rect)));
                                        }
                                        Err(e) => {
                                            let _ = sender.try_send(Message::AlignmentDone(Err(e.to_string())));
                                        }
                                    }
                                });
                            }),
                            |msg| msg,
                        )
                    }
                } else {
                    Task::none()
                }
            }
            Message::AlignmentDone(result) => {
                self.is_processing = false;  // Processing complete
                match result {
                    Ok(rect) => {
                        self.status = "Aligned".to_string();
                        self.crop_rect = Some(rect);
                    }
                    Err(e) => self.status = format!("Alignment failed: {}", e),
                }
                Task::done(Message::RefreshPanes)
            }
            Message::StackImages => {
                // Determine which images to stack and get output directory
                // First, try to find aligned images in the aligned directory
                let (images_to_stack, output_dir) = if !self.images.is_empty() {
                    let first_path = &self.images[0];
                    let base_dir = first_path
                        .parent()
                        .unwrap_or(std::path::Path::new("."));
                    let aligned_dir = base_dir.join("aligned");
                    
                    // Check if aligned directory exists and has images
                    if aligned_dir.exists() {
                        if let Ok(entries) = std::fs::read_dir(&aligned_dir) {
                            let mut aligned_paths: Vec<PathBuf> = entries
                                .flatten()
                                .map(|e| e.path())
                                .filter(|p| {
                                    p.is_file()
                                        && p.extension()
                                            .and_then(|ext| ext.to_str())
                                            .map(|ext| {
                                                ["jpg", "jpeg", "png", "tif", "tiff"]
                                                    .contains(&ext.to_lowercase().as_str())
                                            })
                                            .unwrap_or(false)
                                })
                                .collect();
                            
                            if !aligned_paths.is_empty() {
                                aligned_paths.sort();
                                log::info!("Using {} aligned images from {}", aligned_paths.len(), aligned_dir.display());
                                (aligned_paths, base_dir.to_path_buf())
                            } else if !self.aligned_images.is_empty() {
                                (self.aligned_images.clone(), base_dir.to_path_buf())
                            } else {
                                (self.images.clone(), base_dir.to_path_buf())
                            }
                        } else if !self.aligned_images.is_empty() {
                            let first_aligned = &self.aligned_images[0];
                            let out_dir = first_aligned
                                .parent()
                                .and_then(|p| p.parent())
                                .unwrap_or(std::path::Path::new("."))
                                .to_path_buf();
                            (self.aligned_images.clone(), out_dir)
                        } else {
                            (self.images.clone(), base_dir.to_path_buf())
                        }
                    } else if !self.aligned_images.is_empty() {
                        let first_aligned = &self.aligned_images[0];
                        let out_dir = first_aligned
                            .parent()
                            .and_then(|p| p.parent())
                            .unwrap_or(std::path::Path::new("."))
                            .to_path_buf();
                        (self.aligned_images.clone(), out_dir)
                    } else {
                        (self.images.clone(), base_dir.to_path_buf())
                    }
                } else {
                    // No images at all
                    self.status = "No images loaded".to_string();
                    return Task::none();
                };

                // Clean up existing bunches images before starting new stacking
                let bunches_dir = output_dir.join("bunches");
                if bunches_dir.exists() {
                    log::info!("Cleaning up existing bunches images in {}", bunches_dir.display());
                    if let Err(e) = std::fs::remove_dir_all(&bunches_dir) {
                        log::warn!("Failed to remove bunches directory: {}", e);
                    }
                    // Recreate the directory
                    if let Err(e) = std::fs::create_dir_all(&bunches_dir) {
                        log::warn!("Failed to recreate bunches directory: {}", e);
                    }
                }

                self.status = "Stacking images...".to_string();
                self.progress_message = "Starting stacking...".to_string();
                self.progress_value = 0.0;
                
                let crop_rect = self.crop_rect;
                let proc_config = self.create_processing_config();
                self.is_processing = true;  // Mark as processing
                
                Task::run(
                    iced::stream::channel(100, move |mut sender| async move {
                        // Run processing in a separate thread
                        std::thread::spawn(move || {
                            let progress_cb = std::sync::Arc::new(std::sync::Mutex::new(
                                {
                                    let mut sender_clone = sender.clone();
                                    move |msg: String, pct: f32| {
                                        let _ = sender_clone.try_send(Message::ProgressUpdate(msg, pct));
                                    }
                                }
                            ));

                            let result = stacking::stack_images(
                                &images_to_stack, 
                                &output_dir, 
                                crop_rect,
                                &proc_config,
                                Some(progress_cb),
                            );
                            
                            // Send final result
                            match result {
                                Ok(res) => {
                                    // Convert to PNG bytes for display
                                    let mut buf = opencv::core::Vector::new();
                                    if opencv::imgcodecs::imencode(
                                        ".png",
                                        &res,
                                        &mut buf,
                                        &opencv::core::Vector::new(),
                                    )
                                    .is_ok()
                                    {
                                        let _ = sender.try_send(Message::StackingDone(Ok((buf.to_vec(), res))));
                                    } else {
                                        let _ = sender.try_send(Message::StackingDone(Err(
                                            "Failed to encode image".to_string()
                                        )));
                                    }
                                }
                                Err(e) => {
                                    let _ = sender.try_send(Message::StackingDone(Err(e.to_string())));
                                }
                            }
                        });
                    }),
                    |msg| msg,
                )
            }
            Message::StackingDone(result) => {
                self.is_processing = false;  // Processing complete
                match result {
                    Ok((bytes, mat)) => {
                        self.status = "Stacking complete".to_string();
                        self.preview_handle = Some(iced::widget::image::Handle::from_bytes(bytes));
                        self.result_mat = Some(mat);
                    }
                    Err(e) => self.status = format!("Stacking failed: {}", e),
                }
                Task::done(Message::RefreshPanes)
            }
            Message::OpenImage(path) => {
                let _ = opener::open(path);
                Task::none()
            }
            Message::ShowImagePreview(path) => {
                self.preview_image_path = Some(path.clone());
                self.preview_loading = true;
                self.preview_is_thumbnail = true; // Start with thumbnail
                // Load a scaled-down preview image asynchronously
                Task::perform(
                    async move {
                        // Load and scale down the image for faster preview
                        match thumbnail::generate_thumbnail(&path) {
                            Ok(thumb_handle) => (path, thumb_handle, true), // true = is thumbnail
                            Err(_) => {
                                // Fallback: load full image if thumbnail fails
                                match tokio::fs::read(&path).await {
                                    Ok(bytes) => {
                                        let handle = iced::widget::image::Handle::from_bytes(bytes);
                                        (path, handle, false) // false = full image
                                    }
                                    Err(_) => {
                                        let handle = iced::widget::image::Handle::from_path(&path);
                                        (path, handle, false)
                                    }
                                }
                            }
                        }
                    },
                    |(path, handle, is_thumbnail)| Message::ImagePreviewLoaded(path, handle, is_thumbnail),
                )
            }
            Message::ImagePreviewLoaded(path, handle, is_thumbnail) => {
                // Only update if this is still the current preview path
                if self.preview_image_path.as_ref() == Some(&path) {
                    self.preview_handle = Some(handle);
                    self.preview_loading = false;
                    self.preview_is_thumbnail = is_thumbnail;
                }
                Task::none()
            }
            Message::LoadFullImage(path) => {
                self.preview_loading = true;
                // Load the full-resolution image
                Task::perform(
                    async move {
                        match tokio::fs::read(&path).await {
                            Ok(bytes) => {
                                let handle = iced::widget::image::Handle::from_bytes(bytes);
                                (path, handle, false) // false = full image
                            }
                            Err(_) => {
                                let handle = iced::widget::image::Handle::from_path(&path);
                                (path, handle, false)
                            }
                        }
                    },
                    |(path, handle, is_thumbnail)| Message::ImagePreviewLoaded(path, handle, is_thumbnail),
                )
            },
            Message::CloseImagePreview => {
                self.preview_image_path = None;
                self.preview_handle = None;
                self.preview_loading = false;
                self.preview_is_thumbnail = false;
                // Restore scroll positions after closing preview
                Task::batch(vec![
                    iced::widget::scrollable::scroll_to(
                        iced::widget::scrollable::Id::new("imported"),
                        iced::widget::scrollable::AbsoluteOffset { x: 0.0, y: self.imported_scroll_offset },
                    ),
                    iced::widget::scrollable::scroll_to(
                        iced::widget::scrollable::Id::new("aligned"),
                        iced::widget::scrollable::AbsoluteOffset { x: 0.0, y: self.aligned_scroll_offset },
                    ),
                    iced::widget::scrollable::scroll_to(
                        iced::widget::scrollable::Id::new("bunches"),
                        iced::widget::scrollable::AbsoluteOffset { x: 0.0, y: self.bunches_scroll_offset },
                    ),
                    iced::widget::scrollable::scroll_to(
                        iced::widget::scrollable::Id::new("final"),
                        iced::widget::scrollable::AbsoluteOffset { x: 0.0, y: self.final_scroll_offset },
                    ),
                ])
            }
            // Scroll position tracking
            Message::ImportedScrollChanged(offset) => {
                self.imported_scroll_offset = offset;
                Task::none()
            }
            Message::AlignedScrollChanged(offset) => {
                self.aligned_scroll_offset = offset;
                Task::none()
            }
            Message::BunchesScrollChanged(offset) => {
                self.bunches_scroll_offset = offset;
                Task::none()
            }
            Message::FinalScrollChanged(offset) => {
                self.final_scroll_offset = offset;
                Task::none()
            }
            Message::RefreshPanes => {
                if let Some(first_path) = self.images.first() {
                    let base_dir = first_path
                        .parent()
                        .unwrap_or(std::path::Path::new("."))
                        .to_path_buf();
                    let cache = self.thumbnail_cache.clone();

                    return Task::perform(
                        async move {
                            let scan_dir = |dir_name: &str| -> Vec<PathBuf> {
                                let dir_path = base_dir.join(dir_name);
                                let mut paths = Vec::new();
                                if let Ok(entries) = std::fs::read_dir(dir_path) {
                                    for entry in entries.flatten() {
                                        let p = entry.path();
                                        if p.is_file() {
                                            if let Some(ext) =
                                                p.extension().and_then(|e| e.to_str())
                                            {
                                                let ext = ext.to_lowercase();
                                                if ["jpg", "jpeg", "png", "tif", "tiff"]
                                                    .contains(&ext.as_str())
                                                {
                                                    paths.push(p);
                                                }
                                            }
                                        }
                                    }
                                }
                                paths.sort();
                                paths
                            };

                            let aligned = scan_dir("aligned");
                            let bunches = scan_dir("bunches");
                            let final_imgs = scan_dir("final");

                            let mut all_new_paths = Vec::new();
                            all_new_paths.extend(aligned.clone());
                            all_new_paths.extend(bunches.clone());
                            all_new_paths.extend(final_imgs.clone());

                            let cache_locked = cache.read().unwrap();
                            let paths_to_process: Vec<_> = all_new_paths
                                .into_iter()
                                .filter(|p| !cache_locked.contains_key(p))
                                .collect();
                            drop(cache_locked);

                            // We can't easily return multiple messages from one Task::perform without a stream
                            // but we can return the scanned paths and then trigger another message.
                            (aligned, bunches, final_imgs, paths_to_process)
                        },
                        |(aligned, bunches, final_imgs, paths_to_process)| {
                            // This is a bit tricky in iced 0.13 without streams for incremental updates
                            // but we can at least update the paths and then start another task for thumbnails.
                            Message::InternalPathsScanned(
                                aligned,
                                bunches,
                                final_imgs,
                                paths_to_process,
                            )
                        },
                    );
                }
                Task::none()
            }
            Message::AutoRefreshTick => {
                // Auto-refresh file lists when processing is active
                if self.is_processing {
                    Task::done(Message::RefreshPanes)
                } else {
                    Task::none()
                }
            }
            Message::ToggleSettings => {
                self.show_settings = !self.show_settings;
                Task::none()
            }
            Message::SharpnessThresholdChanged(value) => {
                self.config.sharpness_threshold = value;
                let _ = save_settings(&self.config);
                Task::none()
            }
            Message::SharpnessGridSizeChanged(value) => {
                self.config.sharpness_grid_size = value as i32;
                let _ = save_settings(&self.config);
                Task::none()
            }
            Message::UseAdaptiveBatchSizes(enabled) => {
                self.config.use_adaptive_batches = enabled;
                if enabled {
                    // Recalculate batch sizes based on system RAM
                    let available_gb = system_info::get_available_memory_gb();
                    // Estimate average image size (assume 24MP RGB image ~72MB)
                    let avg_size_mb = 72.0;
                    self.config.batch_config = 
                        system_info::BatchSizeConfig::calculate_optimal(available_gb, avg_size_mb);
                }
                let _ = save_settings(&self.config);
                Task::none()
            }
            Message::UseCLAHE(enabled) => {
                self.config.use_clahe = enabled;
                let _ = save_settings(&self.config);
                Task::none()
            }
            Message::FeatureDetectorChanged(detector) => {
                self.config.feature_detector = detector;
                let _ = save_settings(&self.config);
                Task::none()
            }
            Message::ProgressUpdate(msg, value) => {
                self.progress_message = msg;
                self.progress_value = value;
                Task::none()
            }
            // Advanced processing message handlers
            Message::EnableNoiseReduction(enabled) => {
                self.config.enable_noise_reduction = enabled;
                let _ = save_settings(&self.config);
                Task::none()
            }
            Message::NoiseReductionStrengthChanged(value) => {
                self.config.noise_reduction_strength = value;
                let _ = save_settings(&self.config);
                Task::none()
            }
            Message::EnableSharpening(enabled) => {
                self.config.enable_sharpening = enabled;
                let _ = save_settings(&self.config);
                Task::none()
            }
            Message::SharpeningStrengthChanged(value) => {
                self.config.sharpening_strength = value;
                let _ = save_settings(&self.config);
                Task::none()
            }
            Message::EnableColorCorrection(enabled) => {
                self.config.enable_color_correction = enabled;
                let _ = save_settings(&self.config);
                Task::none()
            }
            Message::ContrastBoostChanged(value) => {
                self.config.contrast_boost = value;
                let _ = save_settings(&self.config);
                Task::none()
            }
            Message::BrightnessBoostChanged(value) => {
                self.config.brightness_boost = value;
                let _ = save_settings(&self.config);
                Task::none()
            }
            Message::SaturationBoostChanged(value) => {
                self.config.saturation_boost = value;
                let _ = save_settings(&self.config);
                Task::none()
            }
            // Preview settings
            Message::UseInternalPreview(enabled) => {
                self.config.use_internal_preview = enabled;
                let _ = save_settings(&self.config);
                Task::none()
            }
            Message::PreviewMaxWidthChanged(width) => {
                self.config.preview_max_width = width;
                let _ = save_settings(&self.config);
                Task::none()
            }
            Message::PreviewMaxHeightChanged(height) => {
                self.config.preview_max_height = height;
                let _ = save_settings(&self.config);
                Task::none()
            }
            Message::Exit => {
                std::process::exit(0);
            }
            Message::None => Task::none(),
        };
        task
    }
    
    pub fn subscription(&self) -> iced::Subscription<Message> {
        // Only refresh when processing is active
        if self.is_processing {
            iced::time::every(Duration::from_secs(2)).map(|_| Message::AutoRefreshTick)
        } else {
            iced::Subscription::none()
        }
    }

    pub fn view(&self) -> Element<'_, Message> {
        self.render_image_preview()
    }

    fn render_main_view(&self) -> Element<'_, Message> {
        // Align button: enabled when images are imported and not processing
        let align_button = if !self.images.is_empty() && !self.is_processing {
            button("Align").on_press(Message::AlignImages)
        } else {
            button("Align")
                .style(|theme, _status| button::Style {
                    background: Some(iced::Background::Color(iced::Color::from_rgb(0.3, 0.3, 0.3))),
                    text_color: iced::Color::from_rgb(0.5, 0.5, 0.5),
                    ..button::secondary(theme, button::Status::Disabled)
                })
        };

        // Stack button: enabled when aligned images exist and not processing
        let stack_button = if !self.aligned_images.is_empty() && !self.is_processing {
            button("Stack").on_press(Message::StackImages)
        } else {
            button("Stack")
                .style(|theme, _status| button::Style {
                    background: Some(iced::Background::Color(iced::Color::from_rgb(0.3, 0.3, 0.3))),
                    text_color: iced::Color::from_rgb(0.5, 0.5, 0.5),
                    ..button::secondary(theme, button::Status::Disabled)
                })
        };

        let buttons = row![
            button("Add Images").on_press(Message::AddImages),
            button("Add Folder").on_press(Message::AddFolder),
            align_button,
            stack_button,
            button(if self.show_settings { "Hide Settings" } else { "Settings" })
                .on_press(Message::ToggleSettings),
            button("Exit").on_press(Message::Exit),
        ]
        .spacing(10)
        .padding(10);

        let mut main_column = column![buttons];

        // Settings panel
        if self.show_settings {
            let settings_panel = self.render_settings_panel();
            main_column = main_column.push(settings_panel);
        }

        // Progress bar (only visible when processing)
        if self.is_processing {
            let progress_bar = container(
                column![
                    text(if self.progress_message.is_empty() {
                        "Processing..."
                    } else {
                        &self.progress_message
                    }).size(14),
                    iced::widget::progress_bar(0.0..=100.0, self.progress_value)
                        .width(Length::Fill)
                ]
                .spacing(5)
            )
            .padding(10)
            .width(Length::Fill)
            .style(|_| container::Style::default()
                .background(iced::Color::from_rgb(0.15, 0.25, 0.35)));
            
            main_column = main_column.push(progress_bar);
        }

        let panes = row![
            self.render_pane("Imported", &self.images),
            self.render_pane("Aligned", &self.aligned_images),
            self.render_pane("Bunches", &self.bunch_images),
            self.render_pane("Final", &self.final_images),
        ]
        .spacing(10)
        .padding(10)
        .height(Length::Fill);

        main_column = main_column
            .push(panes)
            .push(
                container(
                    text_input("", &self.status)
                        .size(16)
                        .style(|_theme, _status| text_input::Style {
                            background: iced::Background::Color(iced::Color::from_rgb(0.1, 0.1, 0.1)),
                            border: iced::Border {
                                color: iced::Color::from_rgb(0.6, 0.6, 0.6),
                                width: 1.0,
                                radius: 4.0.into(),
                            },
                            icon: iced::Color::from_rgb(0.8, 0.8, 0.8),
                            placeholder: iced::Color::from_rgb(0.5, 0.5, 0.5),
                            value: iced::Color::from_rgb(0.9, 0.9, 0.9),
                            selection: iced::Color::from_rgb(0.5, 0.7, 1.0),
                        })
                )
                    .padding(5)
                    .width(Length::Fill)
                    .style(|_| container::Style::default()
                        .background(iced::Color::from_rgb(0.1, 0.1, 0.1)))
            );

        main_column.into()
    }

    fn render_image_preview(&self) -> Element<'_, Message> {
        if let Some(path) = &self.preview_image_path {
            // Dark overlay background
            let background = container(
                button("")
                    .on_press(Message::CloseImagePreview)
                    .style(|_theme, _status| button::Style {
                        background: Some(iced::Background::Color(iced::Color::from_rgba(0.0, 0.0, 0.0, 0.7))),
                        ..button::Style::default()
                    })
                    .width(Length::Fill)
                    .height(Length::Fill)
            )
            .width(Length::Fill)
            .height(Length::Fill);

            // Create the image content element
            let image_content: Element<'_, Message> = if self.preview_loading {
                // Show loading indicator
                container(text("Loading image...").size(18))
                    .width(Length::Fill)
                    .height(Length::Fill)
                    .center_x(Length::Fill)
                    .center_y(Length::Fill)
                    .into()
            } else if let Some(handle) = &self.preview_handle {
                // Show the loaded image
                container(
                    iced_image(handle.clone())
                        .width(Length::Fixed(self.config.preview_max_width - 40.0)) // Account for padding
                        .height(Length::Fixed(self.config.preview_max_height - 140.0)) // Account for header/footer/padding
                        .content_fit(iced::ContentFit::Contain)
                )
                .width(Length::Fixed(self.config.preview_max_width - 40.0))
                .height(Length::Fixed(self.config.preview_max_height - 140.0))
                .center_x(Length::Fill)
                .center_y(Length::Fill)
                .into()
            } else {
                // Fallback if no handle
                container(text("Failed to load image").size(16))
                    .width(Length::Fill)
                    .height(Length::Fill)
                    .center_x(Length::Fill)
                    .center_y(Length::Fill)
                    .into()
            };

            // Preview content
            let preview_content = container(
                column![
                    // Header with filename and close button
                    row![
                        text(path.file_name().unwrap_or_default().to_string_lossy())
                            .size(16)
                            .width(Length::Fill),
                        button("✕")
                            .on_press(Message::CloseImagePreview)
                            .style(button::secondary)
                    ]
                    .spacing(10)
                    .align_y(iced::Alignment::Center),
                    // Full-size image content
                    image_content,
                    // Footer with buttons
                    if self.preview_is_thumbnail && !self.preview_loading {
                        // Show full resolution button when displaying thumbnail
                        row![
                            button("Load Full Resolution")
                                .on_press(Message::LoadFullImage(path.clone()))
                                .style(button::primary),
                            button("Open in External Viewer")
                                .on_press(Message::OpenImage(path.clone()))
                                .style(button::secondary),
                            button("Close")
                                .on_press(Message::CloseImagePreview)
                                .style(button::secondary)
                        ]
                        .spacing(10)
                    } else {
                        // Normal buttons
                        row![
                            button("Open in External Viewer")
                                .on_press(Message::OpenImage(path.clone()))
                                .style(button::primary),
                            button("Close")
                                .on_press(Message::CloseImagePreview)
                                .style(button::secondary)
                        ]
                        .spacing(10)
                    }
                ]
                .spacing(10)
                .padding(20)
            )
            .width(Length::Fixed(self.config.preview_max_width))
            .height(Length::Fixed(self.config.preview_max_height))
            .style(|_| container::Style::default()
                .background(iced::Background::Color(iced::Color::from_rgb(0.15, 0.15, 0.15)))
                .border(iced::Border::default()
                    .width(2.0)
                    .color(iced::Color::from_rgb(0.3, 0.3, 0.3))));

            // Stack the preview on top of the background
            iced::widget::stack![
                background,
                container(preview_content)
                    .width(Length::Fill)
                    .height(Length::Fill)
                    .center_x(Length::Fill)
                    .center_y(Length::Fill)
            ].into()
        } else {
            // No preview, show main view
            self.render_main_view()
        }
    }

    fn render_settings_panel(&self) -> Element<'_, Message> {
        let sharpness_slider = row![
            text("Blur Threshold:"),
            slider(10.0..=100.0, self.config.sharpness_threshold, Message::SharpnessThresholdChanged)
                .width(200),
            text(format!("{:.1}", self.config.sharpness_threshold)),
        ]
        .spacing(10)
        .align_y(iced::Alignment::Center);

        let grid_size_slider = row![
            text("Sharpness Grid:"),
            slider(4.0..=16.0, self.config.sharpness_grid_size as f32, Message::SharpnessGridSizeChanged)
                .step(1.0)
                .width(200),
            text(format!("{}x{}", self.config.sharpness_grid_size, self.config.sharpness_grid_size)),
        ]
        .spacing(10)
        .align_y(iced::Alignment::Center);

        let adaptive_batch_checkbox = checkbox(
            "Auto-adjust batch sizes (RAM-based)",
            self.config.use_adaptive_batches
        )
        .on_toggle(Message::UseAdaptiveBatchSizes);

        let clahe_checkbox = checkbox(
            "Use CLAHE (enhances dark images)",
            self.config.use_clahe
        )
        .on_toggle(Message::UseCLAHE);

        let feature_detector_label = text("Feature Detector:");
        
        let orb_selected = self.config.feature_detector == FeatureDetector::ORB;
        let orb_button = button(
            text(if orb_selected { 
                "✓ ORB (Fast)" 
            } else { 
                "  ORB (Fast)" 
            })
        )
        .style(move |theme, status| {
            if orb_selected {
                button::Style {
                    background: Some(iced::Background::Color(iced::Color::from_rgb(0.3, 0.5, 0.3))),
                    text_color: iced::Color::WHITE,
                    border: iced::Border {
                        color: iced::Color::from_rgb(0.4, 0.7, 0.4),
                        width: 2.0,
                        radius: 4.0.into(),
                    },
                    ..Default::default()
                }
            } else {
                button::secondary(theme, status)
            }
        })
        .on_press(Message::FeatureDetectorChanged(FeatureDetector::ORB));

        let sift_selected = self.config.feature_detector == FeatureDetector::SIFT;
        let sift_button = button(
            text(if sift_selected { 
                "✓ SIFT (Best)" 
            } else { 
                "  SIFT (Best)" 
            })
        )
        .style(move |theme, status| {
            if sift_selected {
                button::Style {
                    background: Some(iced::Background::Color(iced::Color::from_rgb(0.3, 0.5, 0.3))),
                    text_color: iced::Color::WHITE,
                    border: iced::Border {
                        color: iced::Color::from_rgb(0.4, 0.7, 0.4),
                        width: 2.0,
                        radius: 4.0.into(),
                    },
                    ..Default::default()
                }
            } else {
                button::secondary(theme, status)
            }
        })
        .on_press(Message::FeatureDetectorChanged(FeatureDetector::SIFT));

        let akaze_selected = self.config.feature_detector == FeatureDetector::AKAZE;
        let akaze_button = button(
            text(if akaze_selected { 
                "✓ AKAZE (Balanced)" 
            } else { 
                "  AKAZE (Balanced)" 
            })
        )
        .style(move |theme, status| {
            if akaze_selected {
                button::Style {
                    background: Some(iced::Background::Color(iced::Color::from_rgb(0.3, 0.5, 0.3))),
                    text_color: iced::Color::WHITE,
                    border: iced::Border {
                        color: iced::Color::from_rgb(0.4, 0.7, 0.4),
                        width: 2.0,
                        radius: 4.0.into(),
                    },
                    ..Default::default()
                }
            } else {
                button::secondary(theme, status)
            }
        })
        .on_press(Message::FeatureDetectorChanged(FeatureDetector::AKAZE));

        let feature_row = row![
            feature_detector_label,
            orb_button,
            sift_button,
            akaze_button,
        ]
        .spacing(10);

        let batch_info = if self.config.use_adaptive_batches {
            text(format!(
                "Batch sizes: Sharp={}, Features={}, Warp={}, Stack={}",
                self.config.batch_config.sharpness_batch_size,
                self.config.batch_config.feature_batch_size,
                self.config.batch_config.warp_batch_size,
                self.config.batch_config.stacking_batch_size,
            )).size(12)
        } else {
            text("Using default batch sizes").size(12)
        };

        // Advanced processing controls
        let advanced_section = column![
            text("Advanced Processing").size(16).style(|_| text::Style { color: Some(iced::Color::from_rgb(0.8, 0.8, 1.0)) }),
            
            checkbox("Enable Noise Reduction", self.config.enable_noise_reduction)
                .on_toggle(Message::EnableNoiseReduction),
            
            row![
                text("Noise Strength:"),
                slider(1.0..=10.0, self.config.noise_reduction_strength, Message::NoiseReductionStrengthChanged)
                    .width(150),
                text(format!("{:.1}", self.config.noise_reduction_strength)),
            ]
            .spacing(10)
            .align_y(iced::Alignment::Center),
            
            checkbox("Enable Sharpening", self.config.enable_sharpening)
                .on_toggle(Message::EnableSharpening),
            
            row![
                text("Sharpen Strength:"),
                slider(0.0..=5.0, self.config.sharpening_strength, Message::SharpeningStrengthChanged)
                    .width(150),
                text(format!("{:.1}", self.config.sharpening_strength)),
            ]
            .spacing(10)
            .align_y(iced::Alignment::Center),
            
            checkbox("Enable Color Correction", self.config.enable_color_correction)
                .on_toggle(Message::EnableColorCorrection),
            
            row![
                text("Contrast:"),
                slider(0.5..=3.0, self.config.contrast_boost, Message::ContrastBoostChanged)
                    .width(120),
                text(format!("{:.1}", self.config.contrast_boost)),
            ]
            .spacing(10)
            .align_y(iced::Alignment::Center),
            
            row![
                text("Brightness:"),
                slider(-100.0..=100.0, self.config.brightness_boost, Message::BrightnessBoostChanged)
                    .width(120),
                text(format!("{:.0}", self.config.brightness_boost)),
            ]
            .spacing(10)
            .align_y(iced::Alignment::Center),
            
            row![
                text("Saturation:"),
                slider(0.0..=3.0, self.config.saturation_boost, Message::SaturationBoostChanged)
                    .width(120),
                text(format!("{:.1}", self.config.saturation_boost)),
            ]
            .spacing(10)
            .align_y(iced::Alignment::Center),
        ]
        .spacing(8);

        // Preview settings
        let preview_section = column![
            text("Preview Settings").size(16).style(|_| text::Style { color: Some(iced::Color::from_rgb(0.8, 0.8, 1.0)) }),
            
            checkbox("Use Internal Preview (modal overlay)", self.config.use_internal_preview)
                .on_toggle(Message::UseInternalPreview),
            
            text("When disabled, clicking thumbnails opens images in your system's default viewer").size(10)
                .style(|_| text::Style { color: Some(iced::Color::from_rgb(0.6, 0.6, 0.6)) }),
            
            row![
                text("Preview Max Width:"),
                slider(400.0..=2000.0, self.config.preview_max_width, Message::PreviewMaxWidthChanged)
                    .width(150),
                text(format!("{:.0}px", self.config.preview_max_width)),
            ]
            .spacing(10)
            .align_y(iced::Alignment::Center),
            
            row![
                text("Preview Max Height:"),
                slider(300.0..=1500.0, self.config.preview_max_height, Message::PreviewMaxHeightChanged)
                    .width(150),
                text(format!("{:.0}px", self.config.preview_max_height)),
            ]
            .spacing(10)
            .align_y(iced::Alignment::Center),
        ]
        .spacing(8);

        container(
            column![
                text("Processing Settings").size(18),
                sharpness_slider,
                grid_size_slider,
                adaptive_batch_checkbox,
                clahe_checkbox,
                feature_row,
                batch_info,
                advanced_section,
                preview_section,
            ]
            .spacing(10)
        )
        .padding(15)
        .width(Length::Fill)
        .style(|_| container::Style::default()
            .background(iced::Color::from_rgb(0.2, 0.2, 0.25)))
        .into()
    }

    fn render_pane<'a>(&self, title: &'a str, images: &'a [PathBuf]) -> Element<'a, Message> {
        // Fixed column count to prevent thumbnail squeezing
        // With 4 panes, 2 columns works well for window widths 1400px+
        // For narrower windows, thumbnails stay fixed size and pane gets horizontal scroll if needed
        let thumbs_per_row = 2;
        self.render_pane_with_columns(title, images, thumbs_per_row)
    }

    fn render_pane_with_columns<'a>(&self, title: &'a str, images: &'a [PathBuf], thumbs_per_row: usize) -> Element<'a, Message> {
        // Create scroll message closure
        let scroll_message = match title {
            "Imported" => |offset: f32| Message::ImportedScrollChanged(offset),
            "Aligned" => |offset: f32| Message::AlignedScrollChanged(offset),
            "Bunches" => |offset: f32| Message::BunchesScrollChanged(offset),
            "Final" => |offset: f32| Message::FinalScrollChanged(offset),
            _ => |_: f32| Message::None,
        };

        // Constants for thumbnail sizing
        const THUMB_WIDTH: f32 = 120.0;
        const THUMB_HEIGHT: f32 = 90.0;
        const THUMB_SPACING: f32 = 8.0;
        
        // Use fixed number of columns passed as parameter
        // This prevents thumbnails from being squeezed when window resizes
        
        // Group images into rows based on fixed column count
        let mut rows_vec: Vec<Element<Message>> = Vec::new();
        
        for chunk in images.chunks(thumbs_per_row) {
            let mut row_elements: Vec<Element<Message>> = Vec::new();
            
            for path in chunk {
                let path_clone = path.clone();
                let cache = self.thumbnail_cache.read().unwrap();
                let handle = cache.get(path).cloned();

                let image_widget: Element<Message> = if let Some(h) = handle {
                    iced_image(h)
                        .width(Length::Fixed(THUMB_WIDTH))
                        .height(Length::Fixed(THUMB_HEIGHT))
                        .content_fit(iced::ContentFit::ScaleDown)
                        .into()
                } else {
                    container(text("Loading...").size(10))
                        .width(Length::Fixed(THUMB_WIDTH))
                        .height(Length::Fixed(THUMB_HEIGHT))
                        .center_x(Length::Fill)
                        .center_y(Length::Fill)
                        .style(|_| {
                            container::Style::default().background(iced::Color::from_rgb(0.2, 0.2, 0.2))
                        })
                        .into()
                };

                let thumbnail_element = button(
                    column![
                        image_widget,
                        container(text(path.file_name().unwrap_or_default().to_string_lossy()).size(9))
                            .width(Length::Fixed(THUMB_WIDTH))
                            .center_x(Length::Fill)
                    ]
                    .align_x(iced::Alignment::Center),
                )
                .on_press(if self.config.use_internal_preview {
                    Message::ShowImagePreview(path_clone)
                } else {
                    Message::OpenImage(path_clone)
                })
                .style(button::secondary)
                .width(Length::Fixed(THUMB_WIDTH))
                .into();
                
                row_elements.push(thumbnail_element);
            }
            
            // Create a row with the thumbnails
            let thumbnail_row = row(row_elements)
                .spacing(THUMB_SPACING)
                .align_y(iced::Alignment::Start)
                .into();
            
            rows_vec.push(thumbnail_row);
        }
        
        let content = column(rows_vec)
            .spacing(THUMB_SPACING)
            .align_x(iced::Alignment::Center)
            .height(Length::Shrink);

        // Create scrollable ID based on pane title
        let scrollable_id = match title {
            "Imported" => Some(iced::widget::scrollable::Id::new("imported")),
            "Aligned" => Some(iced::widget::scrollable::Id::new("aligned")),
            "Bunches" => Some(iced::widget::scrollable::Id::new("bunches")),
            "Final" => Some(iced::widget::scrollable::Id::new("final")),
            _ => None,
        };

        let mut scrollable_widget = scrollable(content)
            .width(Length::Fill)
            .on_scroll(move |viewport| {
                // Calculate scroll offset from viewport
                let offset = viewport.absolute_offset().y;
                scroll_message(offset)
            });

        // Set ID if available to preserve scroll position
        if let Some(id) = scrollable_id {
            scrollable_widget = scrollable_widget.id(id);
        }

        container(
            column![
                text(title)
                    .size(18)
                    .width(Length::Fill)
                    .align_x(iced::Alignment::Center),
                container(scrollable_widget)
                    .width(Length::Fill)
                    .height(Length::Fill)
                    .style(|_| container::Style {
                        background: Some(iced::Background::Color(iced::Color::TRANSPARENT)),
                        border: iced::Border::default(),
                        text_color: Some(iced::Color::TRANSPARENT),
                        shadow: iced::Shadow::default(),
                    })
            ]
            .spacing(10),
        )
        .width(Length::FillPortion(1))
        .height(Length::Fill)
        .padding(5)
        .style(|_| {
            container::Style::default().border(
                iced::Border::default()
                    .width(1.0)
                    .color(iced::Color::from_rgb(0.3, 0.3, 0.3)),
            )
        })
        .into()
    }

    pub fn theme(&self) -> Theme {
        Theme::Dark
    }


}


