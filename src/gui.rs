use iced::widget::{
    button, checkbox, column, container, horizontal_space, image as iced_image, mouse_area, row, scrollable, slider, text, text_input,
};
use iced::{Element, Length, Task, Theme};
use iced::window;
use opencv::prelude::*;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, RwLock, Mutex};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;
use std::fs;

use crate::alignment;
use crate::config::{FeatureDetector, ProcessingConfig};
use crate::messages::Message;
use crate::settings::{load_settings, save_settings};
use crate::stacking;
use crate::thumbnail;
use crate::system_info;

// ============================================================================
// LOG CAPTURE
// ============================================================================
use std::sync::OnceLock;

static LOG_BUFFER: OnceLock<Mutex<Vec<String>>> = OnceLock::new();

pub fn get_log_buffer() -> &'static Mutex<Vec<String>> {
    LOG_BUFFER.get_or_init(|| Mutex::new(Vec::new()))
}

pub fn append_log(message: String) {
    if let Ok(mut buffer) = get_log_buffer().lock() {
        buffer.push(message);
        // Keep only last 1000 messages to avoid memory issues
        if buffer.len() > 1000 {
            buffer.drain(0..100);
        }
    }
}

pub fn get_logs() -> Vec<String> {
    if let Ok(buffer) = get_log_buffer().lock() {
        buffer.clone()
    } else {
        vec!["Failed to access log buffer".to_string()]
    }
}

// ============================================================================
// STATE MANAGEMENT
// ============================================================================
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
    help_window_id: Option<window::Id>,
    log_window_id: Option<window::Id>,
    progress_message: String,
    progress_value: f32,
    // Image preview
    preview_image_path: Option<PathBuf>,
    preview_loading: bool,
    preview_is_thumbnail: bool,
    preview_current_pane: Vec<PathBuf>, // List of images in current pane for navigation
    preview_navigation_throttle: bool, // Throttle navigation to prevent too fast switching
    // Scroll position tracking
    imported_scroll_offset: f32,
    aligned_scroll_offset: f32,
    bunches_scroll_offset: f32,
    final_scroll_offset: f32,
    // Window size for responsive layout
    window_width: f32,
    // Selection modes
    aligned_selection_mode: bool,
    selected_aligned: Vec<PathBuf>,
    bunch_selection_mode: bool,
    selected_bunches: Vec<PathBuf>,
    // Cancellation flag for background tasks
    cancel_flag: Arc<AtomicBool>,
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
            help_window_id: None,
            log_window_id: None,
            progress_message: String::new(),
            progress_value: 0.0,
            preview_image_path: None,
            preview_loading: false,
            preview_is_thumbnail: false,
            preview_current_pane: Vec::new(),
            preview_navigation_throttle: false,
            imported_scroll_offset: 0.0,
            aligned_scroll_offset: 0.0,
            bunches_scroll_offset: 0.0,
            final_scroll_offset: 0.0,
            window_width: 1400.0, // Default reasonable width
            aligned_selection_mode: false,
            selected_aligned: Vec::new(),
            bunch_selection_mode: false,
            selected_bunches: Vec::new(),
            cancel_flag: Arc::new(AtomicBool::new(false)),
        }
    }
}

// ============================================================================
// MESSAGE HANDLING (UPDATE FUNCTION)
// ============================================================================

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
                        let path = folder.path().to_path_buf();
                        Message::LoadFolder(path)
                    } else {
                        Message::None
                    }
                },
                |msg| msg,
            ),
            Message::LoadFolder(path) => Task::perform(
                async move {
                    let mut paths = Vec::new();
                    if let Ok(entries) = std::fs::read_dir(&path) {
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
                        
                    // Also scan for aligned, bunches, and final images
                    let mut aligned_paths = Vec::new();
                    let mut bunch_paths = Vec::new();
                    let mut final_paths = Vec::new();
                    
                    let aligned_dir = path.join("aligned");
                    if aligned_dir.exists() {
                        if let Ok(entries) = std::fs::read_dir(&aligned_dir) {
                            for entry in entries.flatten() {
                                let p = entry.path();
                                if p.is_file() {
                                    if let Some(ext) = p.extension().and_then(|e| e.to_str()) {
                                        let ext = ext.to_lowercase();
                                        if ["jpg", "jpeg", "png", "tif", "tiff"]
                                            .contains(&ext.as_str())
                                        {
                                            aligned_paths.push(p);
                                        }
                                    }
                                }
                            }
                        }
                    }
                    aligned_paths.sort();
                    
                    let bunches_dir = path.join("bunches");
                    if bunches_dir.exists() {
                        if let Ok(entries) = std::fs::read_dir(&bunches_dir) {
                            for entry in entries.flatten() {
                                let p = entry.path();
                                if p.is_file() {
                                    if let Some(ext) = p.extension().and_then(|e| e.to_str()) {
                                        let ext = ext.to_lowercase();
                                        if ["jpg", "jpeg", "png", "tif", "tiff"]
                                            .contains(&ext.as_str())
                                        {
                                            bunch_paths.push(p);
                                        }
                                    }
                                }
                            }
                        }
                    }
                    bunch_paths.sort();
                    
                    let final_dir = path.join("final");
                    if final_dir.exists() {
                        if let Ok(entries) = std::fs::read_dir(&final_dir) {
                            for entry in entries.flatten() {
                                let p = entry.path();
                                if p.is_file() {
                                    if let Some(ext) = p.extension().and_then(|e| e.to_str()) {
                                        let ext = ext.to_lowercase();
                                        if ["jpg", "jpeg", "png", "tif", "tiff"]
                                            .contains(&ext.as_str())
                                        {
                                            final_paths.push(p);
                                        }
                                    }
                                }
                            }
                        }
                    }
                    final_paths.sort();
                    
                    // Create a combined message with all paths
                    // We'll use InternalPathsScanned to set aligned/bunches/final
                    // and then set the main images
                    if !aligned_paths.is_empty() || !bunch_paths.is_empty() || !final_paths.is_empty() {
                        // We have existing processed images
                        Message::InternalPathsScanned(aligned_paths, bunch_paths, final_paths, paths)
                    } else {
                        // No existing processed images, just load the main images
                        Message::ImagesSelected(paths)
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
                // Determine if this is an initial load or a refresh
                let is_initial_load = !paths_to_process.is_empty() && self.images.is_empty();
                let is_imported_refresh = !paths_to_process.is_empty() && !self.images.is_empty();
                
                if is_initial_load || is_imported_refresh {
                    // Initial load or imported refresh: update imported images list
                    if is_imported_refresh {
                        // Clear all panes when doing an import refresh (like starting new project)
                        self.images.clear();
                        self.aligned_images.clear();
                        self.bunch_images.clear();
                        self.final_images.clear();
                        
                        // Clear entire thumbnail cache for import refresh
                        let mut cache = self.thumbnail_cache.write().unwrap();
                        cache.clear();
                    }
                    
                    self.images.extend(paths_to_process.clone());
                    
                    if is_initial_load {
                        self.aligned_images = aligned.clone();
                        self.bunch_images = bunches.clone();
                        self.final_images = final_imgs.clone();
                        
                        // Update status
                        self.status = format!(
                            "Loaded {} images ({} aligned, {} bunches, {} final)",
                            self.images.len(),
                            aligned.len(),
                            bunches.len(),
                            final_imgs.len()
                        );
                        
                        // Clear thumbnail cache only on initial load
                        let mut cache = self.thumbnail_cache.write().unwrap();
                        cache.clear();
                    } else {
                        // For imported refresh, reload all subdirectories too
                        self.aligned_images = aligned.clone();
                        self.bunch_images = bunches.clone();
                        self.final_images = final_imgs.clone();
                        
                        self.status = format!(
                            "Refreshed {} images ({} aligned, {} bunches, {} final)",
                            self.images.len(),
                            aligned.len(),
                            bunches.len(),
                            final_imgs.len()
                        );
                    }
                }
                
                // Handle refresh of aligned, bunches, or final (when paths_to_process is empty)
                if paths_to_process.is_empty() {
                    // Clear and replace the specific pane being refreshed
                    if !aligned.is_empty() {
                        self.aligned_images.clear();
                        self.aligned_images.extend(aligned.clone());
                    }
                    if !bunches.is_empty() {
                        self.bunch_images.clear();
                        self.bunch_images.extend(bunches.clone());
                    }
                    if !final_imgs.is_empty() {
                        self.final_images.clear();
                        self.final_images.extend(final_imgs.clone());
                    }
                }

                // Collect all paths that need thumbnails (only those not already cached)
                let mut all_paths = Vec::new();
                
                if is_initial_load || is_imported_refresh {
                    // Initial load or refresh: process paths_to_process
                    all_paths.extend(paths_to_process.clone());
                }
                all_paths.extend(aligned.clone());
                all_paths.extend(bunches.clone());
                all_paths.extend(final_imgs.clone());

                // Filter out paths that already have thumbnails (for refresh operations)
                let cache_locked = self.thumbnail_cache.read().unwrap();
                let paths_needing_thumbnails: Vec<_> = all_paths
                    .into_iter()
                    .filter(|p| !cache_locked.contains_key(p))
                    .collect();
                drop(cache_locked);

                if paths_needing_thumbnails.is_empty() {
                    return Task::none();
                }

                let cache = self.thumbnail_cache.clone();
                // Generate thumbnails in parallel using rayon
                Task::run(
                    iced::stream::channel(100, move |sender| async move {
                        std::thread::spawn(move || {
                            use rayon::prelude::*;
                            let sender = Arc::new(std::sync::Mutex::new(sender));
                            
                            paths_needing_thumbnails.par_iter().for_each(|path: &PathBuf| {
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
                        // Clean up existing aligned and bunches images before starting new alignment
                        let aligned_dir = output_dir.join("aligned");
                        let bunches_dir = output_dir.join("bunches");
                        
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
                        
                        // Clear aligned and bunches panes when starting alignment
                        self.aligned_images.clear();
                        self.bunch_images.clear();
                        
                        self.status = "Aligning images...".to_string();
                        self.progress_message = "Starting alignment...".to_string();
                        self.progress_value = 0.0;
                        
                        // Reset cancel flag for new operation
                        self.cancel_flag.store(false, Ordering::Relaxed);
                        
                        let images_paths = self.images.clone();
                        let proc_config = self.create_processing_config();
                        let cancel_flag = self.cancel_flag.clone();
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
                                        Some(cancel_flag),
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
                    Err(e) => {
                        // Check if this was a user cancellation
                        if e.contains("cancelled by user") {
                            self.status = "Alignment cancelled by user".to_string();
                        } else {
                            self.status = format!("Alignment failed: {}", e);
                        }
                    }
                }
                Task::done(Message::RefreshPanes)
            }
            Message::StackImages => {
                // Enter aligned selection mode
                self.aligned_selection_mode = true;
                self.selected_aligned.clear();
                self.status = "Select aligned images to stack, then click Stack button below aligned pane".to_string();
                Task::none()
            }
            Message::CancelAlignedSelection => {
                self.aligned_selection_mode = false;
                self.selected_aligned.clear();
                self.status = "Aligned selection cancelled".to_string();
                Task::none()
            }
            Message::ToggleAlignedImage(path) => {
                if let Some(pos) = self.selected_aligned.iter().position(|p| p == &path) {
                    self.selected_aligned.remove(pos);
                } else {
                    self.selected_aligned.push(path);
                }
                Task::none()
            }
            Message::SelectAllAligned => {
                self.selected_aligned = self.aligned_images.clone();
                Task::none()
            }
            Message::DeselectAllAligned => {
                self.selected_aligned.clear();
                Task::none()
            }
            Message::StackSelectedAligned => {
                if self.selected_aligned.is_empty() {
                    self.status = "No aligned images selected".to_string();
                    self.aligned_selection_mode = false;
                    return Task::none();
                }

                // Exit selection mode
                self.aligned_selection_mode = false;

                // Determine output directory from first aligned image
                let output_dir = if let Some(first_aligned) = self.selected_aligned.first() {
                    first_aligned
                        .parent()
                        .and_then(|p| p.parent())
                        .unwrap_or(std::path::Path::new("."))
                        .to_path_buf()
                } else {
                    self.status = "No aligned images selected".to_string();
                    return Task::none();
                };

                let images_to_stack = self.selected_aligned.clone();
                self.selected_aligned.clear(); // Clear selection

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

                self.status = format!("Stacking {} selected aligned images...", images_to_stack.len());
                self.progress_message = "Starting stacking of selected aligned images...".to_string();
                self.progress_value = 0.0;
                
                // Reset cancel flag for new operation
                self.cancel_flag.store(false, Ordering::Relaxed);
                
                let crop_rect = self.crop_rect;
                let proc_config = self.create_processing_config();
                let cancel_flag = self.cancel_flag.clone();
                self.is_processing = true;
                
                Task::run(
                    iced::stream::channel(100, move |mut sender| async move {
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
                                Some(cancel_flag),
                            );
                            
                            match result {
                                Ok(res) => {
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
            Message::StackBunches => {
                // Enter bunch selection mode
                self.bunch_selection_mode = true;
                self.selected_bunches.clear();
                self.status = "Select bunches to stack, then click Stack button below bunches pane".to_string();
                Task::none()
            }
            Message::CancelBunchSelection => {
                self.bunch_selection_mode = false;
                self.selected_bunches.clear();
                self.status = "Bunch selection cancelled".to_string();
                Task::none()
            }
            Message::ToggleBunchImage(path) => {
                if let Some(pos) = self.selected_bunches.iter().position(|p| p == &path) {
                    self.selected_bunches.remove(pos);
                } else {
                    self.selected_bunches.push(path);
                }
                Task::none()
            }
            Message::SelectAllBunches => {
                self.selected_bunches = self.bunch_images.clone();
                Task::none()
            }
            Message::DeselectAllBunches => {
                self.selected_bunches.clear();
                Task::none()
            }
            Message::StackSelectedBunches => {
                if self.selected_bunches.is_empty() {
                    self.status = "No bunches selected".to_string();
                    self.bunch_selection_mode = false;
                    return Task::none();
                }

                // Exit selection mode
                self.bunch_selection_mode = false;

                // Determine output directory from first bunch image
                let output_dir = if let Some(first_bunch) = self.selected_bunches.first() {
                    first_bunch
                        .parent()
                        .and_then(|p| p.parent())
                        .unwrap_or(std::path::Path::new("."))
                        .to_path_buf()
                } else {
                    self.status = "No bunches selected".to_string();
                    return Task::none();
                };

                let images_to_stack = self.selected_bunches.clone();
                self.selected_bunches.clear(); // Clear selection

                self.status = format!("Stacking {} selected bunches...", images_to_stack.len());
                self.progress_message = "Starting stacking of selected bunches...".to_string();
                self.progress_value = 0.0;
                
                // Reset cancel flag for new operation
                self.cancel_flag.store(false, Ordering::Relaxed);
                
                let crop_rect = self.crop_rect;
                let proc_config = self.create_processing_config();
                let cancel_flag = self.cancel_flag.clone();
                self.is_processing = true;
                
                Task::run(
                    iced::stream::channel(100, move |mut sender| async move {
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
                                Some(cancel_flag),
                            );
                            
                            match result {
                                Ok(res) => {
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
                    Err(e) => {
                        // Check if this was a user cancellation
                        if e.contains("cancelled by user") {
                            self.status = "Stacking cancelled by user".to_string();
                        } else {
                            self.status = format!("Stacking failed: {}", e);
                        }
                    }
                }
                Task::done(Message::RefreshPanes)
            }
            Message::OpenImage(path) => {
                let viewer_path = self.config.external_viewer_path.clone();
                if viewer_path.is_empty() {
                    // Use system default
                    let _ = opener::open(path);
                } else {
                    // Use configured external viewer
                    let _ = std::process::Command::new(&viewer_path)
                        .arg(&path)
                        .spawn();
                }
                Task::none()
            }
            Message::OpenImageWithExternalEditor(path) => {
                let editor_path = self.config.external_editor_path.clone();
                if editor_path.is_empty() {
                    // Use system default
                    let _ = opener::open(path);
                } else {
                    // Use configured external editor
                    let _ = std::process::Command::new(&editor_path)
                        .arg(&path)
                        .spawn();
                }
                Task::none()
            }
            Message::ShowImagePreview(path, pane_images) => {
                self.preview_image_path = Some(path.clone());
                self.preview_current_pane = pane_images;
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
                // If we're processing, ESC should cancel the operation instead
                if self.is_processing {
                    self.is_processing = false;
                    // Signal background task to cancel
                    self.cancel_flag.store(true, Ordering::Relaxed);
                    // Keep progress bar visible showing cancellation in progress
                    self.progress_message = "Cancelling... (stopping background task)".to_string();
                    self.status = "Processing cancelled - stopping background task".to_string();
                    log::warn!("User pressed ESC - cancel flag set to TRUE, waiting for background task to stop");
                    return Task::none();
                }
                
                self.preview_image_path = None;
                self.preview_handle = None;
                self.preview_loading = false;
                self.preview_is_thumbnail = false;
                self.preview_current_pane.clear();
                self.preview_navigation_throttle = false;
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
            Message::NextImageInPreview => {
                // Only navigate if preview is open and not throttled
                if self.preview_image_path.is_none() || self.preview_current_pane.is_empty() || self.preview_navigation_throttle {
                    return Task::none();
                }
                
                // Set throttle flag
                self.preview_navigation_throttle = true;
                
                if let Some(current_path) = &self.preview_image_path {
                    if let Some(current_index) = self.preview_current_pane.iter().position(|p| p == current_path) {
                        let next_index = (current_index + 1) % self.preview_current_pane.len();
                        if let Some(next_path) = self.preview_current_pane.get(next_index) {
                            let next_path = next_path.clone();
                            let pane_images = self.preview_current_pane.clone();
                            
                            // Schedule throttle reset after 50ms
                            let reset_task = Task::perform(
                                async {
                                    tokio::time::sleep(Duration::from_millis(50)).await;
                                },
                                |_| Message::NavigationThrottleReset
                            );
                            
                            return Task::batch(vec![
                                self.update(Message::ShowImagePreview(next_path, pane_images)),
                                reset_task
                            ]);
                        }
                    }
                }
                Task::none()
            }
            Message::PreviousImageInPreview => {
                // Only navigate if preview is open and not throttled
                if self.preview_image_path.is_none() || self.preview_current_pane.is_empty() || self.preview_navigation_throttle {
                    return Task::none();
                }
                
                // Set throttle flag
                self.preview_navigation_throttle = true;
                
                if let Some(current_path) = &self.preview_image_path {
                    if let Some(current_index) = self.preview_current_pane.iter().position(|p| p == current_path) {
                        let prev_index = if current_index == 0 {
                            self.preview_current_pane.len() - 1
                        } else {
                            current_index - 1
                        };
                        if let Some(prev_path) = self.preview_current_pane.get(prev_index) {
                            let prev_path = prev_path.clone();
                            let pane_images = self.preview_current_pane.clone();
                            
                            // Schedule throttle reset after 50ms
                            let reset_task = Task::perform(
                                async {
                                    tokio::time::sleep(Duration::from_millis(50)).await;
                                },
                                |_| Message::NavigationThrottleReset
                            );
                            
                            return Task::batch(vec![
                                self.update(Message::ShowImagePreview(prev_path, pane_images)),
                                reset_task
                            ]);
                        }
                    }
                }
                Task::none()
            }
            Message::NavigationThrottleReset => {
                self.preview_navigation_throttle = false;
                Task::none()
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

                            // Return scanned paths - don't pass paths_to_process as it would
                            // trigger import behavior. Let InternalPathsScanned determine which
                            // thumbnails need to be generated.
                            (aligned, bunches, final_imgs, Vec::<PathBuf>::new())
                        },
                        |(aligned, bunches, final_imgs, _paths_to_process)| {
                            // Pass empty vec for paths_to_process to indicate this is a
                            // subdirectory refresh, not an import operation
                            Message::InternalPathsScanned(
                                aligned,
                                bunches,
                                final_imgs,
                                vec![],
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
            Message::RefreshImportedPane => {
                // Refresh imported pane by rescanning the base directory
                if let Some(first_path) = self.images.first() {
                    let base_dir = first_path.parent().unwrap_or(std::path::Path::new(".")).to_path_buf();
                    
                    // Clear thumbnails for imported images
                    {
                        let mut cache = self.thumbnail_cache.write().unwrap();
                        for path in &self.images {
                            cache.remove(path);
                        }
                    }
                    
                    return Task::perform(
                        async move {
                            let mut paths = Vec::new();
                            if let Ok(entries) = std::fs::read_dir(&base_dir) {
                                for entry in entries.flatten() {
                                    let p = entry.path();
                                    if p.is_file() {
                                        if let Some(ext) = p.extension().and_then(|e| e.to_str()) {
                                            let ext = ext.to_lowercase();
                                            if ["jpg", "jpeg", "png", "tif", "tiff"].contains(&ext.as_str()) {
                                                paths.push(p);
                                            }
                                        }
                                    }
                                }
                            }
                            paths.sort();
                            Message::InternalPathsScanned(vec![], vec![], vec![], paths)
                        },
                        |msg| msg,
                    );
                }
                Task::none()
            }
            Message::RefreshAlignedPane => {
                // Refresh aligned pane by rescanning the aligned directory
                if let Some(first_path) = self.images.first() {
                    let base_dir = first_path.parent().unwrap_or(std::path::Path::new(".")).to_path_buf();
                    
                    // Clear thumbnails for aligned images
                    {
                        let mut cache = self.thumbnail_cache.write().unwrap();
                        for path in &self.aligned_images {
                            cache.remove(path);
                        }
                    }
                    
                    return Task::perform(
                        async move {
                            let aligned_dir = base_dir.join("aligned");
                            let mut paths = Vec::new();
                            if let Ok(entries) = std::fs::read_dir(&aligned_dir) {
                                for entry in entries.flatten() {
                                    let p = entry.path();
                                    if p.is_file() {
                                        if let Some(ext) = p.extension().and_then(|e| e.to_str()) {
                                            let ext = ext.to_lowercase();
                                            if ["jpg", "jpeg", "png", "tif", "tiff"].contains(&ext.as_str()) {
                                                paths.push(p);
                                            }
                                        }
                                    }
                                }
                            }
                            paths.sort();
                            Message::InternalPathsScanned(paths, vec![], vec![], vec![])
                        },
                        |msg| msg,
                    );
                }
                Task::none()
            }
            Message::RefreshBunchesPane => {
                // Refresh bunches pane by rescanning the bunches directory
                if let Some(first_path) = self.images.first() {
                    let base_dir = first_path.parent().unwrap_or(std::path::Path::new(".")).to_path_buf();
                    
                    // Clear thumbnails for bunch images
                    {
                        let mut cache = self.thumbnail_cache.write().unwrap();
                        for path in &self.bunch_images {
                            cache.remove(path);
                        }
                    }
                    
                    return Task::perform(
                        async move {
                            let bunches_dir = base_dir.join("bunches");
                            let mut paths = Vec::new();
                            if let Ok(entries) = std::fs::read_dir(&bunches_dir) {
                                for entry in entries.flatten() {
                                    let p = entry.path();
                                    if p.is_file() {
                                        if let Some(ext) = p.extension().and_then(|e| e.to_str()) {
                                            let ext = ext.to_lowercase();
                                            if ["jpg", "jpeg", "png", "tif", "tiff"].contains(&ext.as_str()) {
                                                paths.push(p);
                                            }
                                        }
                                    }
                                }
                            }
                            paths.sort();
                            Message::InternalPathsScanned(vec![], paths, vec![], vec![])
                        },
                        |msg| msg,
                    );
                }
                Task::none()
            }
            Message::RefreshFinalPane => {
                // Refresh final pane by rescanning the final directory
                if let Some(first_path) = self.images.first() {
                    let base_dir = first_path.parent().unwrap_or(std::path::Path::new(".")).to_path_buf();
                    
                    // Clear thumbnails for final images
                    {
                        let mut cache = self.thumbnail_cache.write().unwrap();
                        for path in &self.final_images {
                            cache.remove(path);
                        }
                    }
                    
                    return Task::perform(
                        async move {
                            let final_dir = base_dir.join("final");
                            let mut paths = Vec::new();
                            if let Ok(entries) = std::fs::read_dir(&final_dir) {
                                for entry in entries.flatten() {
                                    let p = entry.path();
                                    if p.is_file() {
                                        if let Some(ext) = p.extension().and_then(|e| e.to_str()) {
                                            let ext = ext.to_lowercase();
                                            if ["jpg", "jpeg", "png", "tif", "tiff"].contains(&ext.as_str()) {
                                                paths.push(p);
                                            }
                                        }
                                    }
                                }
                            }
                            paths.sort();
                            Message::InternalPathsScanned(vec![], vec![], paths, vec![])
                        },
                        |msg| msg,
                    );
                }
                Task::none()
            }
            Message::ToggleSettings => {
                self.show_settings = !self.show_settings;
                Task::none()
            }
            Message::ToggleHelp => {
                // Convert markdown to HTML and open in browser
                Task::perform(
                    async {
                        // Read markdown file
                        let markdown = match fs::read_to_string("USER_MANUAL.md") {
                            Ok(content) => content,
                            Err(e) => {
                                log::error!("Failed to read USER_MANUAL.md: {}", e);
                                return Err(format!("Failed to read USER_MANUAL.md: {}", e));
                            }
                        };

                        // Convert markdown to HTML
                        use pulldown_cmark::{Parser, html};
                        let parser = Parser::new(&markdown);
                        let mut html_output = String::new();
                        html::push_html(&mut html_output, parser);

                        // Create a complete HTML document with CSS styling
                        let full_html = format!(
                            r#"<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Image Stacker - User Manual</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            line-height: 1.6;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            background-color: #1e1e1e;
            color: #d4d4d4;
        }}
        h1 {{ color: #4ec9f0; border-bottom: 2px solid #4ec9f0; padding-bottom: 10px; }}
        h2 {{ color: #66ccff; margin-top: 30px; }}
        h3 {{ color: #88ddff; margin-top: 20px; }}
        code {{
            background-color: #2d2d2d;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
        }}
        pre {{
            background-color: #2d2d2d;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
        }}
        pre code {{
            background-color: transparent;
            padding: 0;
        }}
        a {{ color: #4ec9f0; text-decoration: none; }}
        a:hover {{ text-decoration: underline; }}
        ul, ol {{ padding-left: 30px; }}
        li {{ margin: 8px 0; }}
        hr {{ border: none; border-top: 1px solid #444; margin: 30px 0; }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #444;
            padding: 10px;
            text-align: left;
        }}
        th {{ background-color: #2d2d2d; }}
    </style>
</head>
<body>
{}
</body>
</html>"#,
                            html_output
                        );

                        // Write to temporary HTML file
                        let temp_dir = std::env::temp_dir();
                        let html_path = temp_dir.join("imagestacker_manual.html");
                        
                        if let Err(e) = fs::write(&html_path, full_html) {
                            log::error!("Failed to write HTML file: {}", e);
                            return Err(format!("Failed to write HTML file: {}", e));
                        }

                        // Open in default browser
                        if let Err(e) = opener::open(&html_path) {
                            log::error!("Failed to open browser: {}", e);
                            return Err(format!("Failed to open browser: {}", e));
                        }

                        Ok(())
                    },
                    |result| {
                        if let Err(e) = result {
                            log::error!("Help display error: {}", e);
                        }
                        Message::None
                    }
                )
            }
            Message::CloseHelp => {
                if let Some(id) = self.help_window_id.take() {
                    window::close::<Message>(id)
                } else {
                    Task::none()
                }
            }
            Message::ToggleLog => {
                if self.log_window_id.is_some() {
                    // Log window already open, do nothing
                    Task::none()
                } else {
                    // Create and open log window
                    let (id, open) = window::open(window::Settings {
                        size: iced::Size::new(900.0, 600.0),
                        position: window::Position::Centered,
                        exit_on_close_request: false,
                        ..Default::default()
                    });
                    self.log_window_id = Some(id);
                    open.map(|_| Message::None)
                }
            }
            Message::CloseLog => {
                if let Some(id) = self.log_window_id.take() {
                    window::close::<Message>(id)
                } else {
                    Task::none()
                }
            }
            Message::ResetToDefaults => {
                self.config = ProcessingConfig::default();
                let _ = save_settings(&self.config);
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
            Message::SharpnessIqrMultiplierChanged(value) => {
                self.config.sharpness_iqr_multiplier = value;
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
            Message::ExternalViewerPathChanged(path) => {
                self.config.external_viewer_path = path;
                let _ = save_settings(&self.config);
                Task::none()
            }
            Message::ExternalEditorPathChanged(path) => {
                self.config.external_editor_path = path;
                let _ = save_settings(&self.config);
                Task::none()
            }
            Message::WindowResized(width, _height) => {
                self.window_width = width;
                Task::none()
            }
            Message::Exit => {
                std::process::exit(0);
            }
            Message::None => Task::none(),
        };
        task
    }
    
    // ========================================================================
    // EVENT SUBSCRIPTIONS
    // ========================================================================
    
    pub fn subscription(&self) -> iced::Subscription<Message> {
        let refresh = if self.is_processing {
            iced::time::every(Duration::from_secs(2)).map(|_| Message::AutoRefreshTick)
        } else {
            iced::Subscription::none()
        };
        
        // Add window resize and close subscription
        let window_events = iced::event::listen_with(|event, _status, _id| {
            match event {
                iced::Event::Window(iced::window::Event::Resized(size)) => {
                    Some(Message::WindowResized(size.width, size.height))
                }
                iced::Event::Window(iced::window::Event::CloseRequested) => {
                    // When help window is closed, send CloseHelp message
                    Some(Message::CloseHelp)
                }
                _ => None,
            }
        });
        
        // Keyboard events for image preview navigation
        let keyboard_events = iced::event::listen_with(move |event, _status, _id| {
            if let iced::Event::Keyboard(keyboard_event) = event {
                match keyboard_event {
                    iced::keyboard::Event::KeyPressed { key, .. } => {
                        match key {
                            iced::keyboard::Key::Named(iced::keyboard::key::Named::ArrowRight) => {
                                Some(Message::NextImageInPreview)
                            }
                            iced::keyboard::Key::Named(iced::keyboard::key::Named::ArrowLeft) => {
                                Some(Message::PreviousImageInPreview)
                            }
                            iced::keyboard::Key::Named(iced::keyboard::key::Named::Escape) => {
                                // ESC can either close preview or cancel processing
                                // The handler will decide based on is_processing state
                                Some(Message::CloseImagePreview)
                            }
                            _ => None,
                        }
                    }
                    _ => None,
                }
            } else {
                None
            }
        });
        
        // Mouse wheel events for image preview navigation
        let mouse_events = iced::event::listen_with(move |event, _status, _id| {
            if let iced::Event::Mouse(mouse_event) = event {
                match mouse_event {
                    iced::mouse::Event::WheelScrolled { delta } => {
                        match delta {
                            iced::mouse::ScrollDelta::Lines { y, .. } => {
                                if y > 0.0 {
                                    Some(Message::PreviousImageInPreview)
                                } else if y < 0.0 {
                                    Some(Message::NextImageInPreview)
                                } else {
                                    None
                                }
                            }
                            iced::mouse::ScrollDelta::Pixels { y, .. } => {
                                if y > 0.0 {
                                    Some(Message::PreviousImageInPreview)
                                } else if y < 0.0 {
                                    Some(Message::NextImageInPreview)
                                } else {
                                    None
                                }
                            }
                        }
                    }
                    _ => None,
                }
            } else {
                None
            }
        });
        
        iced::Subscription::batch(vec![refresh, window_events, keyboard_events, mouse_events])
    }

    // ========================================================================
    // VIEW RENDERING
    // ========================================================================

    pub fn view(&self, window: window::Id) -> Element<'_, Message> {
        if Some(window) == self.help_window_id {
            self.render_help_window()
        } else if Some(window) == self.log_window_id {
            self.render_log_window()
        } else {
            self.render_image_preview()
        }
    }

    // ------------------------------------------------------------------------
    // Main View
    // ------------------------------------------------------------------------

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
        let stack_aligned_button = if !self.aligned_images.is_empty() && !self.is_processing {
            button("Stack Aligned").on_press(Message::StackImages)
        } else {
            button("Stack Aligned")
                .style(|theme, _status| button::Style {
                    background: Some(iced::Background::Color(iced::Color::from_rgb(0.3, 0.3, 0.3))),
                    text_color: iced::Color::from_rgb(0.5, 0.5, 0.5),
                    ..button::secondary(theme, button::Status::Disabled)
                })
        };

        // Stack Bunches button: enabled when bunch images exist and not processing
        let stack_bunches_button = if !self.bunch_images.is_empty() && !self.is_processing {
            button("Stack Bunches").on_press(Message::StackBunches)
        } else {
            button("Stack Bunches")
                .style(|theme, _status| button::Style {
                    background: Some(iced::Background::Color(iced::Color::from_rgb(0.3, 0.3, 0.3))),
                    text_color: iced::Color::from_rgb(0.5, 0.5, 0.5),
                    ..button::secondary(theme, button::Status::Disabled)
                })
        };

        // Group buttons in frames
        let import_group = container(
            column![
                row![
                    button("Add Images").on_press(Message::AddImages),
                    button("Add Folder").on_press(Message::AddFolder),
                ]
                .spacing(10)
            ]
        )
        .padding(8)
        .style(|_| container::Style::default()
            .border(iced::Border::default()
                .width(1.0)
                .color(iced::Color::from_rgb(0.4, 0.4, 0.4))));

        let align_group = container(
            column![
                align_button
            ]
        )
        .padding(8)
        .style(|_| container::Style::default()
            .border(iced::Border::default()
                .width(1.0)
                .color(iced::Color::from_rgb(0.4, 0.4, 0.4))));

        let stack_group = container(
            column![
                row![
                    stack_aligned_button,
                    stack_bunches_button,
                ]
                .spacing(10)
            ]
        )
        .padding(8)
        .style(|_| container::Style::default()
            .border(iced::Border::default()
                .width(1.0)
                .color(iced::Color::from_rgb(0.4, 0.4, 0.4))));

        let tools_group = container(
            column![
                row![
                    button(if self.show_settings { "Hide Settings" } else { "Settings" })
                        .on_press(Message::ToggleSettings),
                    button("Help")
                        .on_press(Message::ToggleHelp),
                    button("Show Log")
                        .on_press(Message::ToggleLog),
                ]
                .spacing(10)
            ]
        )
        .padding(8)
        .style(|_| container::Style::default()
            .border(iced::Border::default()
                .width(1.0)
                .color(iced::Color::from_rgb(0.4, 0.4, 0.4))));

        let buttons = row![
            import_group,
            align_group,
            stack_group,
            tools_group,
            horizontal_space(),
            button("Exit").on_press(Message::Exit),
        ]
        .spacing(10)
        .padding(10)
        .align_y(iced::Alignment::Center);

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
                        .width(Length::Fill),
                    text("Press ESC to cancel")
                        .size(12)
                        .style(|_| text::Style {
                            color: Some(iced::Color::from_rgb(1.0, 0.8, 0.0))
                        })
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
            self.render_aligned_pane(),
            self.render_bunches_pane(),
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

    // ------------------------------------------------------------------------
    // Image Preview Modal
    // ------------------------------------------------------------------------

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
                        button("X")
                            .on_press(Message::CloseImagePreview)
                            .style(button::secondary)
                    ]
                    .spacing(10)
                    .align_y(iced::Alignment::Center),
                    // Navigation info
                    if !self.preview_current_pane.is_empty() {
                        let current_index = self.preview_current_pane.iter().position(|p| Some(p) == self.preview_image_path.as_ref()).unwrap_or(0);
                        text(format!("Image {} of {} (Use < | > arrow keys or mouse wheel to navigate)", 
                            current_index + 1, 
                            self.preview_current_pane.len()))
                            .size(12)
                            .style(|_theme| text::Style { color: Some(iced::Color::from_rgb(0.7, 0.7, 0.7)) })
                    } else {
                        text("").size(12)
                    },
                    // Full-size image content
                    image_content,
                    // Footer with buttons
                    if self.preview_is_thumbnail && !self.preview_loading {
                        // Show full resolution button when displaying thumbnail
                        row![
                            button("< Previous")
                                .on_press(Message::PreviousImageInPreview)
                                .style(button::secondary),
                            button("Load Full Resolution")
                                .on_press(Message::LoadFullImage(path.clone()))
                                .style(button::primary),
                            button("Open in External Viewer")
                                .on_press(Message::OpenImage(path.clone()))
                                .style(button::secondary),
                            button("Next >")
                                .on_press(Message::NextImageInPreview)
                                .style(button::secondary),
                            button("Close")
                                .on_press(Message::CloseImagePreview)
                                .style(button::secondary)
                        ]
                        .spacing(10)
                    } else {
                        // Normal buttons
                        row![
                            button("< Previous")
                                .on_press(Message::PreviousImageInPreview)
                                .style(button::secondary),
                            button("Open in External Viewer")
                                .on_press(Message::OpenImage(path.clone()))
                                .style(button::primary),
                            button("Next >")
                                .on_press(Message::NextImageInPreview)
                                .style(button::secondary),
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

    // ------------------------------------------------------------------------
    // Settings Panel
    // ------------------------------------------------------------------------

    fn render_settings_panel(&self) -> Element<'_, Message> {
        // Determine layout based on window width
        // Minimum width per pane: 380px, so need at least 1200px for horizontal (3*380 + spacing + padding)
        let use_horizontal_layout = self.window_width >= 1200.0;
        
        // ============== PANE 1: ALIGNMENT & DETECTION ==============
        // Adjust slider widths based on layout to prevent value stacking
        let (label_width, slider_width, value_width) = if use_horizontal_layout {
            (120, 150, 60)  // Narrower for horizontal pane layout
        } else {
            (150, 200, 50)  // Original widths for vertical layout
        };
        
        let sharpness_slider = row![
            text("Blur Threshold:").width(label_width),
            slider(10.0..=10000.0, self.config.sharpness_threshold, Message::SharpnessThresholdChanged)
                .step(10.0)
                .width(slider_width),
            text(format!("{:.0}", self.config.sharpness_threshold)).width(value_width),
        ]
        .spacing(10)
        .align_y(iced::Alignment::Center);

        let grid_size_slider = row![
            text("Sharpness Grid:").width(label_width),
            slider(4.0..=16.0, self.config.sharpness_grid_size as f32, Message::SharpnessGridSizeChanged)
                .step(1.0)
                .width(slider_width),
            text(format!("{}x{}", self.config.sharpness_grid_size, self.config.sharpness_grid_size)).width(value_width),
        ]
        .spacing(10)
        .align_y(iced::Alignment::Center);

        let iqr_multiplier_slider = row![
            text("Blur Filter (IQR):").width(label_width),
            slider(0.5..=5.0, self.config.sharpness_iqr_multiplier, Message::SharpnessIqrMultiplierChanged)
                .step(0.1)
                .width(slider_width),
            text(format!("{:.1} {}", 
                self.config.sharpness_iqr_multiplier,
                if self.config.sharpness_iqr_multiplier <= 1.0 { "(strict)" }
                else if self.config.sharpness_iqr_multiplier <= 2.0 { "(normal)" }
                else if self.config.sharpness_iqr_multiplier <= 3.0 { "(relaxed)" }
                else { "(very permissive)" }
            )).width(value_width),
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

        // Feature detector layout: vertical when horizontal pane layout, horizontal when vertical pane layout
        let feature_layout: Element<'_, Message> = if use_horizontal_layout {
            // Horizontal layout -> stack detector buttons vertically
            column![
                text("Feature Detector:").size(14),
                orb_button.width(Length::Fixed(160.0)),
                sift_button.width(Length::Fixed(160.0)),
                akaze_button.width(Length::Fixed(160.0)),
            ]
            .spacing(5)
            .into()
        } else {
            // Vertical layout -> stack detector buttons horizontally
            row![
                feature_detector_label,
                orb_button,
                sift_button,
                akaze_button,
            ]
            .spacing(10)
            .into()
        };

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

        let alignment_pane = container(
            column![
                text("Alignment & Detection").size(16).style(|_| text::Style { 
                    color: Some(iced::Color::from_rgb(0.8, 0.8, 1.0)) 
                }),
                sharpness_slider,
                grid_size_slider,
                iqr_multiplier_slider,
                adaptive_batch_checkbox,
                batch_info,
                clahe_checkbox,
                feature_layout,
            ]
            .spacing(10)
        )
        .padding(10)
        .style(|_theme| {
            container::Style {
                background: Some(iced::Background::Color(iced::Color::from_rgba(0.1, 0.1, 0.15, 0.3))),
                border: iced::Border {
                    color: iced::Color::from_rgb(0.3, 0.3, 0.4),
                    width: 1.0,
                    radius: 8.0.into(),
                },
                ..Default::default()
            }
        })
        .width(Length::Fill);

        // ============== PANE 2: POST-PROCESSING ==============
        let noise_section = column![
            checkbox("Enable Noise Reduction", self.config.enable_noise_reduction)
                .on_toggle(Message::EnableNoiseReduction),
            
            row![
                text("Noise Strength:").width(label_width),
                slider(1.0..=10.0, self.config.noise_reduction_strength, Message::NoiseReductionStrengthChanged)
                    .width(slider_width),
                text(format!("{:.1}", self.config.noise_reduction_strength)).width(value_width),
            ]
            .spacing(10)
            .align_y(iced::Alignment::Center),
        ]
        .spacing(5);

        let sharpen_section = column![
            checkbox("Enable Sharpening", self.config.enable_sharpening)
                .on_toggle(Message::EnableSharpening),
            
            row![
                text("Sharpen Strength:").width(label_width),
                slider(0.0..=5.0, self.config.sharpening_strength, Message::SharpeningStrengthChanged)
                    .width(slider_width),
                text(format!("{:.1}", self.config.sharpening_strength)).width(value_width),
            ]
            .spacing(10)
            .align_y(iced::Alignment::Center),
        ]
        .spacing(5);

        let color_section = column![
            checkbox("Enable Color Correction", self.config.enable_color_correction)
                .on_toggle(Message::EnableColorCorrection),
            
            row![
                text("Contrast:").width(label_width),
                slider(0.5..=3.0, self.config.contrast_boost, Message::ContrastBoostChanged)
                    .width(slider_width),
                text(format!("{:.1}", self.config.contrast_boost)).width(value_width),
            ]
            .spacing(10)
            .align_y(iced::Alignment::Center),
            
            row![
                text("Brightness:").width(label_width),
                slider(-100.0..=100.0, self.config.brightness_boost, Message::BrightnessBoostChanged)
                    .width(slider_width),
                text(format!("{:.0}", self.config.brightness_boost)).width(value_width),
            ]
            .spacing(10)
            .align_y(iced::Alignment::Center),
            
            row![
                text("Saturation:").width(label_width),
                slider(0.0..=3.0, self.config.saturation_boost, Message::SaturationBoostChanged)
                    .width(slider_width),
                text(format!("{:.1}", self.config.saturation_boost)).width(value_width),
            ]
            .spacing(10)
            .align_y(iced::Alignment::Center),
        ]
        .spacing(5);

        let postprocessing_pane = container(
            column![
                text("Post-Processing").size(16).style(|_| text::Style { 
                    color: Some(iced::Color::from_rgb(0.8, 0.8, 1.0)) 
                }),
                noise_section,
                sharpen_section,
                color_section,
            ]
            .spacing(10)
        )
        .padding(10)
        .style(|_theme| {
            container::Style {
                background: Some(iced::Background::Color(iced::Color::from_rgba(0.1, 0.1, 0.15, 0.3))),
                border: iced::Border {
                    color: iced::Color::from_rgb(0.3, 0.3, 0.4),
                    width: 1.0,
                    radius: 8.0.into(),
                },
                ..Default::default()
            }
        })
        .width(Length::Fill);

        // ============== PANE 3: PREVIEW & UI ==============
        let preview_pane = container(
            column![
                text("Preview & UI").size(16).style(|_| text::Style { 
                    color: Some(iced::Color::from_rgb(0.8, 0.8, 1.0)) 
                }),
                
                checkbox("Use Internal Preview (modal overlay)", self.config.use_internal_preview)
                    .on_toggle(Message::UseInternalPreview),
                
                text("When disabled, left-click opens in external viewer (configurable below)").size(10)
                    .style(|_| text::Style { color: Some(iced::Color::from_rgb(0.6, 0.6, 0.6)) }),
                
                row![
                    text("Preview Max Width:").width(label_width),
                    slider(400.0..=2000.0, self.config.preview_max_width, Message::PreviewMaxWidthChanged)
                        .width(slider_width),
                    text(format!("{:.0}px", self.config.preview_max_width)).width(value_width),
                ]
                .spacing(10)
                .align_y(iced::Alignment::Center),
                
                row![
                    text("Preview Max Height:").width(label_width),
                    slider(300.0..=1500.0, self.config.preview_max_height, Message::PreviewMaxHeightChanged)
                        .width(slider_width),
                    text(format!("{:.0}px", self.config.preview_max_height)).width(value_width),
                ]
                .spacing(10)
                .align_y(iced::Alignment::Center),
                
                container(column![]).height(10),  // Spacer
                
                text("External Image Viewer (for left-click when internal preview disabled)").size(12)
                    .style(|_| text::Style { color: Some(iced::Color::from_rgb(0.7, 0.7, 0.9)) }),
                
                text_input(
                    "Path to viewer (e.g., /usr/bin/eog, /usr/bin/geeqie)...",
                    &self.config.external_viewer_path
                )
                    .on_input(Message::ExternalViewerPathChanged)
                    .padding(5),
                
                text("Leave empty to use system default viewer. Used when 'Use Internal Preview' is disabled.")
                    .size(10)
                    .style(|_| text::Style { color: Some(iced::Color::from_rgb(0.6, 0.6, 0.6)) }),
                
                container(column![]).height(10),  // Spacer
                
                text("External Image Editor (for right-click)").size(12)
                    .style(|_| text::Style { color: Some(iced::Color::from_rgb(0.7, 0.7, 0.9)) }),
                
                text_input(
                    "Path to editor (e.g., /usr/bin/gimp, darktable, photoshop)...",
                    &self.config.external_editor_path
                )
                    .on_input(Message::ExternalEditorPathChanged)
                    .padding(5),
                
                text("Leave empty to use system default viewer. Right-click any thumbnail to open with editor.")
                    .size(10)
                    .style(|_| text::Style { color: Some(iced::Color::from_rgb(0.6, 0.6, 0.6)) }),
            ]
            .spacing(10)
        )
        .padding(10)
        .style(|_theme| {
            container::Style {
                background: Some(iced::Background::Color(iced::Color::from_rgba(0.1, 0.1, 0.15, 0.3))),
                border: iced::Border {
                    color: iced::Color::from_rgb(0.3, 0.3, 0.4),
                    width: 1.0,
                    radius: 8.0.into(),
                },
                ..Default::default()
            }
        })
        .width(Length::Fill);

        // ============== RESET BUTTON ==============
        let reset_button = button(
            row![
                text("🔄").size(16),
                text("Reset to Defaults"),
            ]
            .spacing(5)
            .align_y(iced::Alignment::Center)
        )
        .style(|theme, status| {
            button::Style {
                background: Some(iced::Background::Color(iced::Color::from_rgb(0.6, 0.3, 0.3))),
                text_color: iced::Color::WHITE,
                border: iced::Border {
                    color: iced::Color::from_rgb(0.8, 0.4, 0.4),
                    width: 1.0,
                    radius: 6.0.into(),
                },
                ..button::primary(theme, status)
            }
        })
        .on_press(Message::ResetToDefaults);

        // ============== RESPONSIVE LAYOUT ==============
        // Use horizontal layout when window is wide enough (>= 1200px for 3 panes at 380px each)
        // Otherwise, stack vertically for better readability
        const MIN_PANE_WIDTH: f32 = 380.0;
        
        let panes_layout: Element<'_, Message> = if use_horizontal_layout {
            // Horizontal layout for wide screens - panes side by side
            row![
                container(alignment_pane).width(Length::Fixed(MIN_PANE_WIDTH)),
                container(postprocessing_pane).width(Length::Fixed(MIN_PANE_WIDTH)),
                container(preview_pane).width(Length::Fixed(MIN_PANE_WIDTH)),
            ]
            .spacing(15)
            .into()
        } else {
            // Vertical layout for narrow screens - panes stacked
            column![
                alignment_pane,
                postprocessing_pane,
                preview_pane,
            ]
            .spacing(15)
            .into()
        };

        container(
            column![
                text("Processing Settings").size(20).style(|_| text::Style { 
                    color: Some(iced::Color::from_rgb(0.9, 0.9, 1.0)) 
                }),
                panes_layout,
                reset_button,
            ]
            .spacing(15)
        )
        .padding(15)
        .width(Length::Fill)
        .style(|_| container::Style::default()
            .background(iced::Color::from_rgb(0.2, 0.2, 0.25)))
        .into()
    }

    // ------------------------------------------------------------------------
    // Help Window
    // ------------------------------------------------------------------------

    fn render_help_window(&self) -> Element<'_, Message> {
        // Load help text from markdown file
        let help_markdown = fs::read_to_string("USER_MANUAL.md")
            .unwrap_or_else(|_| "# Error\n\nCould not load USER_MANUAL.md file.".to_string());
        
        // Parse markdown and create formatted text elements
        let mut elements: Vec<Element<Message>> = Vec::new();
        
        for line in help_markdown.lines() {
            let trimmed = line.trim();
            
            if trimmed.is_empty() {
                // Empty line - add small spacing
                elements.push(container(text("")).height(Length::Fixed(8.0)).into());
            } else if trimmed.starts_with("# ") {
                // H1 heading
                elements.push(
                    text(trimmed.trim_start_matches("# ").to_string())
                        .size(24)
                        .style(|_| text::Style { 
                            color: Some(iced::Color::from_rgb(0.3, 0.7, 1.0)) 
                        })
                        .into()
                );
            } else if trimmed.starts_with("## ") {
                // H2 heading
                elements.push(
                    text(trimmed.trim_start_matches("## ").to_string())
                        .size(20)
                        .style(|_| text::Style { 
                            color: Some(iced::Color::from_rgb(0.4, 0.8, 1.0)) 
                        })
                        .into()
                );
            } else if trimmed.starts_with("### ") {
                // H3 heading
                elements.push(
                    text(trimmed.trim_start_matches("### ").to_string())
                        .size(16)
                        .style(|_| text::Style { 
                            color: Some(iced::Color::from_rgb(0.5, 0.85, 1.0)) 
                        })
                        .into()
                );
            } else if trimmed.starts_with("---") || trimmed.starts_with("===") {
                // Horizontal rule
                elements.push(
                    container(text(""))
                        .width(Length::Fill)
                        .height(Length::Fixed(1.0))
                        .style(|_| container::Style::default()
                            .background(iced::Color::from_rgb(0.5, 0.5, 0.5)))
                        .into()
                );
            } else if trimmed.starts_with("- ") || trimmed.starts_with("* ") {
                // Bullet list
                let content = trimmed.trim_start_matches("- ").trim_start_matches("* ").to_string();
                elements.push(
                    container(
                        text(format!("  • {}", content))
                            .size(13)
                            .style(|_| text::Style { 
                                color: Some(iced::Color::from_rgb(0.9, 0.9, 0.9)) 
                            })
                    )
                    .padding(iced::Padding::new(0.0).left(10.0))
                    .into()
                );
            } else if trimmed.starts_with("✓ ") {
                // Checkmark list
                elements.push(
                    container(
                        text(trimmed.to_string())
                            .size(13)
                            .style(|_| text::Style { 
                                color: Some(iced::Color::from_rgb(0.4, 1.0, 0.4)) 
                            })
                    )
                    .padding(iced::Padding::new(0.0).left(10.0))
                    .into()
                );
            } else if trimmed.starts_with("```") {
                // Code block marker - skip
                continue;
            } else if trimmed.chars().next().map_or(false, |c| c.is_numeric()) 
                      && trimmed.chars().nth(1) == Some('.') {
                // Numbered list
                elements.push(
                    container(
                        text(format!("  {}", trimmed))
                            .size(13)
                            .style(|_| text::Style { 
                                color: Some(iced::Color::from_rgb(0.9, 0.9, 0.9)) 
                            })
                    )
                    .padding(iced::Padding::new(0.0).left(10.0))
                    .into()
                );
            } else if trimmed.starts_with("**") && trimmed.ends_with("**") {
                // Bold text as a paragraph
                let content = trimmed.trim_start_matches("**").trim_end_matches("**").to_string();
                elements.push(
                    text(content)
                        .size(14)
                        .style(|_| text::Style { 
                            color: Some(iced::Color::from_rgb(1.0, 1.0, 1.0)) 
                        })
                        .into()
                );
            } else {
                // Regular paragraph
                elements.push(
                    text(trimmed.to_string())
                        .size(13)
                        .style(|_| text::Style { 
                            color: Some(iced::Color::from_rgb(0.85, 0.85, 0.85)) 
                        })
                        .into()
                );
            }
        }
        
        let help_content = scrollable(
            column(elements)
                .spacing(4)
                .padding(20)
        )
        .height(Length::Fill);

        let close_button = container(
            button("Close")
                .on_press(Message::CloseHelp)
                .padding(10)
                .style(|_theme, _status| button::Style {
                    background: Some(iced::Background::Color(iced::Color::from_rgb(0.8, 0.3, 0.3))),
                    text_color: iced::Color::WHITE,
                    border: iced::Border {
                        radius: 4.0.into(),
                        ..Default::default()
                    },
                    ..button::Style::default()
                })
        )
        .padding(15)
        .width(Length::Fill)
        .center_x(Length::Fill);

        container(
            column![
                help_content,
                close_button,
            ]
            .spacing(0)
        )
        .padding(0)
        .width(Length::Fill)
        .height(Length::Fill)
        .style(|_| container::Style::default()
            .background(iced::Color::from_rgb(0.15, 0.15, 0.2)))
        .into()
    }

    fn render_log_window(&self) -> Element<'_, Message> {
        // Get captured log messages
        let logs = get_logs();
        
        let log_content = if logs.is_empty() {
            "No log messages yet.\n\nLog messages will appear here when the application performs operations.".to_string()
        } else {
            logs.join("\n")
        };
        
        let log_text = scrollable(
            container(
                text(log_content)
                    .size(12)
                    .font(iced::Font::MONOSPACE)
                    .style(|_| text::Style { 
                        color: Some(iced::Color::from_rgb(0.9, 0.9, 0.9)) 
                    })
            )
            .padding(20)
            .width(Length::Fill)
        )
        .height(Length::Fill);

        let close_button = container(
            button("Close")
                .on_press(Message::CloseLog)
                .padding(10)
                .style(|_theme, _status| button::Style {
                    background: Some(iced::Background::Color(iced::Color::from_rgb(0.8, 0.3, 0.3))),
                    text_color: iced::Color::WHITE,
                    border: iced::Border {
                        radius: 4.0.into(),
                        ..Default::default()
                    },
                    ..button::Style::default()
                })
        )
        .padding(15)
        .width(Length::Fill)
        .center_x(Length::Fill);

        container(
            column![
                log_text,
                close_button,
            ]
            .spacing(0)
        )
        .padding(0)
        .width(Length::Fill)
        .height(Length::Fill)
        .style(|_| container::Style::default()
            .background(iced::Color::from_rgb(0.1, 0.1, 0.15)))
        .into()
    }

    // ========================================================================
    // PANE RENDERING (Aligned, Bunches, Final)
    // ========================================================================
    
    // ------------------------------------------------------------------------
    // Aligned Pane (with selection support)
    // ------------------------------------------------------------------------

    fn render_aligned_pane(&self) -> Element<'_, Message> {
        const THUMB_WIDTH: f32 = 120.0;
        const THUMB_HEIGHT: f32 = 90.0;
        const THUMB_SPACING: f32 = 8.0;
        let thumbs_per_row = 2;

        let mut rows_vec: Vec<Element<Message>> = Vec::new();
        
        for chunk in self.aligned_images.chunks(thumbs_per_row) {
            let mut row_elements: Vec<Element<Message>> = Vec::new();
            
            for path in chunk {
                let path_clone = path.clone();
                let path_for_right_click = path.clone();
                let is_selected = self.selected_aligned.contains(path);
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

                let thumbnail_content = column![
                    image_widget,
                    container(text(path.file_name().unwrap_or_default().to_string_lossy()).size(9))
                        .width(Length::Fixed(THUMB_WIDTH))
                        .center_x(Length::Fill)
                ]
                .align_x(iced::Alignment::Center);

                let thumbnail_element = if self.aligned_selection_mode {
                    // In selection mode: make clickable for selection
                    let btn = button(thumbnail_content)
                        .on_press(Message::ToggleAlignedImage(path_clone))
                        .width(Length::Fixed(THUMB_WIDTH));
                    
                    // Apply different style if selected - green background and border
                    if is_selected {
                        container(
                            btn.style(|_theme, _status| button::Style {
                                background: Some(iced::Background::Color(iced::Color::from_rgb(0.2, 0.5, 0.2))),
                                text_color: iced::Color::WHITE,
                                border: iced::Border {
                                    color: iced::Color::from_rgb(0.3, 0.9, 0.3),
                                    width: 4.0,
                                    radius: 4.0.into(),
                                },
                                ..button::Style::default()
                            })
                        )
                        .style(|_| container::Style::default()
                            .background(iced::Color::from_rgb(0.1, 0.3, 0.1)))
                        .padding(2)
                        .into()
                    } else {
                        // Unselected - darker background
                        btn.style(|_theme, _status| button::Style {
                            background: Some(iced::Background::Color(iced::Color::from_rgb(0.15, 0.15, 0.2))),
                            text_color: iced::Color::WHITE,
                            border: iced::Border {
                                color: iced::Color::from_rgb(0.4, 0.4, 0.5),
                                width: 2.0,
                                radius: 4.0.into(),
                            },
                            ..button::Style::default()
                        })
                        .into()
                    }
                } else {
                    // Normal mode: preview on click
                    button(thumbnail_content)
                        .on_press(if self.config.use_internal_preview {
                            Message::ShowImagePreview(path_clone, self.aligned_images.clone())
                        } else {
                            Message::OpenImage(path_clone)
                        })
                        .style(button::secondary)
                        .width(Length::Fixed(THUMB_WIDTH))
                        .into()
                };
                
                // Wrap in mouse_area to detect right-clicks (only in normal mode)
                let final_element = if !self.aligned_selection_mode {
                    mouse_area(thumbnail_element)
                        .on_right_press(Message::OpenImageWithExternalEditor(path_for_right_click))
                        .into()
                } else {
                    thumbnail_element
                };
                
                row_elements.push(final_element);
            }
            
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

        let scrollable_widget = scrollable(content)
            .width(Length::Fill)
            .on_scroll(move |viewport| {
                let offset = viewport.absolute_offset().y;
                Message::AlignedScrollChanged(offset)
            })
            .id(iced::widget::scrollable::Id::new("aligned"));

        // Create pane header with title, count, and refresh button
        let pane_header = row![
            column![
                text("Aligned")
                    .size(18)
                    .align_x(iced::Alignment::Center),
                text(format!("{} images", self.aligned_images.len()))
                    .size(12)
                    .style(|_theme| text::Style {
                        color: Some(iced::Color::from_rgb(0.7, 0.7, 0.7))
                    })
                    .align_x(iced::Alignment::Center)
            ]
            .width(Length::Fill)
            .align_x(iced::Alignment::Center),
            button("Refresh")
                .on_press(Message::RefreshAlignedPane)
                .padding(4)
                .style(|theme, status| button::Style {
                    background: Some(iced::Background::Color(iced::Color::from_rgb(0.2, 0.7, 0.2))),
                    text_color: iced::Color::WHITE,
                    border: iced::Border {
                        radius: 4.0.into(),
                        ..Default::default()
                    },
                    ..button::secondary(theme, status)
                })
        ]
        .spacing(5)
        .align_y(iced::Alignment::Center);

        // Add Cancel and Stack buttons at bottom if in selection mode
        let mut pane_content = column![
            pane_header,
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
        .spacing(10);

        if self.aligned_selection_mode {
            pane_content = pane_content
                .push(
                    row![
                        button("Select All")
                            .on_press(Message::SelectAllAligned)
                            .style(button::secondary),
                        button("Deselect All")
                            .on_press(Message::DeselectAllAligned)
                            .style(button::secondary),
                    ]
                    .spacing(10)
                    .padding(5)
                    .width(Length::Fill)
                )
                .push(
                    row![
                        button("Cancel")
                            .on_press(Message::CancelAlignedSelection)
                            .style(|theme, status| button::Style {
                                background: Some(iced::Background::Color(iced::Color::from_rgb(0.8, 0.3, 0.3))),
                                text_color: iced::Color::WHITE,
                                ..button::secondary(theme, status)
                            }),
                        button(text(format!("Stack ({} selected)", self.selected_aligned.len())))
                            .on_press(Message::StackSelectedAligned)
                            .style(|theme, status| button::Style {
                                background: Some(iced::Background::Color(iced::Color::from_rgb(0.2, 0.7, 0.2))),
                                text_color: iced::Color::WHITE,
                                ..button::secondary(theme, status)
                            }),
                    ]
                    .spacing(10)
                    .padding(5)
                    .width(Length::Fill)
                );
        }

        container(pane_content)
            .width(Length::Fill)
            .height(Length::Fill)
            .padding(10)
            .style(|_| container::Style::default()
                .background(iced::Color::from_rgb(0.15, 0.15, 0.2))
                .border(iced::Border {
                    color: iced::Color::from_rgb(0.5, 0.5, 0.6),
                    width: 2.0,
                    radius: 4.0.into(),
                }))
            .into()
    }

    // ------------------------------------------------------------------------
    // Bunches Pane (with selection support)
    // ------------------------------------------------------------------------

    fn render_bunches_pane(&self) -> Element<'_, Message> {

        const THUMB_WIDTH: f32 = 120.0;
        const THUMB_HEIGHT: f32 = 90.0;
        const THUMB_SPACING: f32 = 8.0;
        let thumbs_per_row = 2;

        let mut rows_vec: Vec<Element<Message>> = Vec::new();
        
        for chunk in self.bunch_images.chunks(thumbs_per_row) {
            let mut row_elements: Vec<Element<Message>> = Vec::new();
            
            for path in chunk {
                let path_clone = path.clone();
                let path_for_right_click = path.clone();
                let is_selected = self.selected_bunches.contains(path);
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

                let thumbnail_content = column![
                    image_widget,
                    container(text(path.file_name().unwrap_or_default().to_string_lossy()).size(9))
                        .width(Length::Fixed(THUMB_WIDTH))
                        .center_x(Length::Fill)
                ]
                .align_x(iced::Alignment::Center);

                let thumbnail_element = if self.bunch_selection_mode {
                    // In selection mode: make clickable for selection
                    let btn = button(thumbnail_content)
                        .on_press(Message::ToggleBunchImage(path_clone))
                        .width(Length::Fixed(THUMB_WIDTH));
                    
                    // Apply different style if selected - green background and border
                    if is_selected {
                        container(
                            btn.style(|_theme, _status| button::Style {
                                background: Some(iced::Background::Color(iced::Color::from_rgb(0.2, 0.5, 0.2))),
                                text_color: iced::Color::WHITE,
                                border: iced::Border {
                                    color: iced::Color::from_rgb(0.3, 0.9, 0.3),
                                    width: 4.0,
                                    radius: 4.0.into(),
                                },
                                ..button::Style::default()
                            })
                        )
                        .style(|_| container::Style::default()
                            .background(iced::Color::from_rgb(0.1, 0.3, 0.1)))
                        .padding(2)
                        .into()
                    } else {
                        // Unselected - darker background
                        btn.style(|_theme, _status| button::Style {
                            background: Some(iced::Background::Color(iced::Color::from_rgb(0.15, 0.15, 0.2))),
                            text_color: iced::Color::WHITE,
                            border: iced::Border {
                                color: iced::Color::from_rgb(0.4, 0.4, 0.5),
                                width: 2.0,
                                radius: 4.0.into(),
                            },
                            ..button::Style::default()
                        })
                        .into()
                    }
                } else {
                    // Normal mode: preview on click
                    button(thumbnail_content)
                        .on_press(if self.config.use_internal_preview {
                            Message::ShowImagePreview(path_clone, self.bunch_images.clone())
                        } else {
                            Message::OpenImage(path_clone)
                        })
                        .style(button::secondary)
                        .width(Length::Fixed(THUMB_WIDTH))
                        .into()
                };
                
                // Wrap in mouse_area to detect right-clicks (only in normal mode)
                let final_element = if !self.bunch_selection_mode {
                    mouse_area(thumbnail_element)
                        .on_right_press(Message::OpenImageWithExternalEditor(path_for_right_click))
                        .into()
                } else {
                    thumbnail_element
                };
                
                row_elements.push(final_element);
            }
            
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

        let scrollable_widget = scrollable(content)
            .width(Length::Fill)
            .on_scroll(move |viewport| {
                let offset = viewport.absolute_offset().y;
                Message::BunchesScrollChanged(offset)
            })
            .id(iced::widget::scrollable::Id::new("bunches"));

        // Create pane header with title, count, and refresh button
        let pane_header = row![
            column![
                text("Bunches")
                    .size(18)
                    .align_x(iced::Alignment::Center),
                text(format!("{} images", self.bunch_images.len()))
                    .size(12)
                    .style(|_theme| text::Style {
                        color: Some(iced::Color::from_rgb(0.7, 0.7, 0.7))
                    })
                    .align_x(iced::Alignment::Center)
            ]
            .width(Length::Fill)
            .align_x(iced::Alignment::Center),
            button("Refresh")
                .on_press(Message::RefreshBunchesPane)
                .padding(4)
                .style(|theme, status| button::Style {
                    background: Some(iced::Background::Color(iced::Color::from_rgb(0.2, 0.7, 0.2))),
                    text_color: iced::Color::WHITE,
                    border: iced::Border {
                        radius: 4.0.into(),
                        ..Default::default()
                    },
                    ..button::secondary(theme, status)
                })
        ]
        .spacing(5)
        .align_y(iced::Alignment::Center);

        // Add Cancel and Stack buttons at bottom if in selection mode
        let mut pane_content = column![
            pane_header,
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
        .spacing(10);

        if self.bunch_selection_mode {
            pane_content = pane_content
                .push(
                    row![
                        button("Select All")
                            .on_press(Message::SelectAllBunches)
                            .style(button::secondary),
                        button("Deselect All")
                            .on_press(Message::DeselectAllBunches)
                            .style(button::secondary),
                    ]
                    .spacing(10)
                    .padding(5)
                    .width(Length::Fill)
                )
                .push(
                    row![
                        button("Cancel")
                            .on_press(Message::CancelBunchSelection)
                            .style(|theme, status| button::Style {
                                background: Some(iced::Background::Color(iced::Color::from_rgb(0.8, 0.3, 0.3))),
                                text_color: iced::Color::WHITE,
                                ..button::secondary(theme, status)
                            }),
                        button(text(format!("Stack ({} selected)", self.selected_bunches.len())))
                            .on_press(Message::StackSelectedBunches)
                            .style(|theme, status| button::Style {
                                background: Some(iced::Background::Color(iced::Color::from_rgb(0.2, 0.7, 0.2))),
                                text_color: iced::Color::WHITE,
                                ..button::secondary(theme, status)
                            }),
                    ]
                    .spacing(10)
                    .padding(5)
                    .width(Length::Fill)
                );
        }

        container(pane_content)
            .width(Length::Fill)
            .height(Length::Fill)
            .padding(10)
            .style(|_| container::Style::default()
                .background(iced::Color::from_rgb(0.15, 0.15, 0.2))
                .border(iced::Border {
                    color: iced::Color::from_rgb(0.5, 0.5, 0.6),
                    width: 2.0,
                    radius: 4.0.into(),
                }))
            .into()
    }

    // ------------------------------------------------------------------------
    // Generic Pane Rendering (Imported, Final)
    // ------------------------------------------------------------------------

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

                let path_for_left_click = path_clone.clone();
                let path_for_right_click = path_clone.clone();
                
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
                    Message::ShowImagePreview(path_for_left_click, images.to_vec())
                } else {
                    Message::OpenImage(path_for_left_click)
                })
                .style(button::secondary)
                .width(Length::Fixed(THUMB_WIDTH));
                
                // Wrap in mouse_area to detect right-clicks
                let thumbnail_with_mouse = mouse_area(thumbnail_element)
                    .on_right_press(Message::OpenImageWithExternalEditor(path_for_right_click))
                    .into();
                
                row_elements.push(thumbnail_with_mouse);
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

        // Determine refresh message based on pane title
        let refresh_message = match title {
            "Imported" => Message::RefreshImportedPane,
            "Aligned" => Message::RefreshAlignedPane,
            "Bunches" => Message::RefreshBunchesPane,
            "Final" => Message::RefreshFinalPane,
            _ => Message::None,
        };

        // Create pane header with title, count, and refresh button
        let pane_header = row![
            column![
                text(title)
                    .size(18)
                    .align_x(iced::Alignment::Center),
                text(format!("{} images", images.len()))
                    .size(12)
                    .style(|_theme| text::Style {
                        color: Some(iced::Color::from_rgb(0.7, 0.7, 0.7))
                    })
                    .align_x(iced::Alignment::Center)
            ]
            .width(Length::Fill)
            .align_x(iced::Alignment::Center),
            button("Refresh")
                .on_press(refresh_message)
                .padding(4)
                .style(|theme, status| button::Style {
                    background: Some(iced::Background::Color(iced::Color::from_rgb(0.2, 0.7, 0.2))),
                    text_color: iced::Color::WHITE,
                    border: iced::Border {
                        radius: 4.0.into(),
                        ..Default::default()
                    },
                    ..button::secondary(theme, status)
                })
        ]
        .spacing(5)
        .align_y(iced::Alignment::Center);

        container(
            column![
                pane_header,
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

    // ========================================================================
    // WINDOW CONFIGURATION
    // ========================================================================

    pub fn theme(&self, _window: window::Id) -> Theme {
        Theme::Dark
    }

    pub fn title(&self, window: window::Id) -> String {
        if Some(window) == self.help_window_id {
            "Rust Image Stacker - User Manual".to_string()
        } else {
            "Rust Image Stacker".to_string()
        }
    }
}


