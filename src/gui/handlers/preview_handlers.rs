//! Image preview and navigation handlers
//!
//! Handlers for:
//! - ShowImagePreview, ImagePreviewLoaded, LoadFullImage
//! - CloseImagePreview, NextImageInPreview, PreviousImageInPreview
//! - NavigationThrottleReset
//! - Scroll position tracking

use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::time::Duration;

use iced::Task;

use crate::messages::Message;
use crate::thumbnail;

use crate::gui::state::ImageStacker;

impl ImageStacker {
    /// Handle ShowImagePreview - display image preview
    pub fn handle_show_image_preview(&mut self, path: PathBuf, pane_images: Vec<PathBuf>) -> Task<Message> {
        self.preview_image_path = Some(path.clone());
        self.preview_current_pane = pane_images;
        self.preview_loading = true;
        self.preview_is_thumbnail = true; // Start with thumbnail
        
        // Try to load sharpness info if available
        if let Some(first_img) = self.images.first() {
            let sharpness_dir = first_img.parent()
                .unwrap_or(std::path::Path::new("."))
                .join("sharpness");
            
            let yaml_name = path.file_stem()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string() + ".yml";
            let yaml_path = sharpness_dir.join(yaml_name);
            
            self.preview_sharpness_info = crate::sharpness_cache::SharpnessInfo::load_from_file(&yaml_path).ok();
        } else {
            self.preview_sharpness_info = None;
        }
        
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

    /// Handle ImagePreviewLoaded - update preview state when image is loaded
    pub fn handle_image_preview_loaded(
        &mut self,
        path: PathBuf,
        handle: iced::widget::image::Handle,
        is_thumbnail: bool,
    ) -> Task<Message> {
        // Only update if this is still the current preview path
        if self.preview_image_path.as_ref() == Some(&path) {
            self.preview_handle = Some(handle);
            self.preview_loading = false;
            self.preview_is_thumbnail = is_thumbnail;
        }
        Task::none()
    }

    /// Handle LoadFullImage - load full-resolution image (screen-optimized)
    pub fn handle_load_full_image(&mut self, path: PathBuf) -> Task<Message> {
        self.preview_loading = true;
        let screen_w = self.window_width;
        let screen_h = self.window_height;
        // Load the image scaled to screen size via OpenCV (GPU-accelerated)
        Task::perform(
            async move {
                let p = path.clone();
                match tokio::task::spawn_blocking(move || {
                    thumbnail::load_preview_for_screen(&p, screen_w, screen_h)
                }).await {
                    Ok(Ok(handle)) => (path, handle, false),
                    _ => {
                        // Fallback: let iced load from path
                        let handle = iced::widget::image::Handle::from_path(&path);
                        (path, handle, false)
                    }
                }
            },
            |(path, handle, is_thumbnail)| Message::ImagePreviewLoaded(path, handle, is_thumbnail),
        )
    }

    /// Handle CloseImagePreview - close preview or cancel processing
    pub fn handle_close_image_preview(&mut self) -> Task<Message> {
        // If a preview is open, close it first (don't cancel background processing)
        if self.preview_image_path.is_some() {
            self.preview_image_path = None;
            self.preview_handle = None;
            self.preview_loading = false;
            self.preview_is_thumbnail = false;
            self.preview_sharpness_info = None;
            self.preview_current_pane.clear();
            self.preview_navigation_throttle = false;
            // Restore scroll positions after closing preview
            return Task::batch(vec![
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
            ]);
        }
        
        // If no preview is open but we're processing, ESC should cancel the operation
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
        
        Task::none()
    }

    /// Handle NextImageInPreview - navigate to next image
    pub fn handle_next_image_in_preview(&mut self) -> Task<Message> {
        // Only navigate if preview is open and not throttled
        if self.preview_image_path.is_none() || self.preview_current_pane.is_empty() || self.preview_navigation_throttle {
            return Task::none();
        }
        
        // Set throttle flag
        self.preview_navigation_throttle = true;
        
        // Remember if we're in full resolution mode before navigating
        let was_full_resolution = !self.preview_is_thumbnail;
        
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
                    
                    let preview_task = if was_full_resolution {
                        // Full resolution mode: keep the current image visible while loading
                        // the next one in the background. No flicker.
                        self.navigate_full_resolution(next_path)
                    } else {
                        self.handle_show_image_preview(next_path, pane_images)
                    };
                    
                    return Task::batch(vec![
                        preview_task,
                        reset_task
                    ]);
                }
            }
        }
        Task::none()
    }

    /// Handle PreviousImageInPreview - navigate to previous image
    pub fn handle_previous_image_in_preview(&mut self) -> Task<Message> {
        // Only navigate if preview is open and not throttled
        if self.preview_image_path.is_none() || self.preview_current_pane.is_empty() || self.preview_navigation_throttle {
            return Task::none();
        }
        
        // Set throttle flag
        self.preview_navigation_throttle = true;
        
        // Remember if we're in full resolution mode before navigating
        let was_full_resolution = !self.preview_is_thumbnail;
        
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
                    
                    let preview_task = if was_full_resolution {
                        // Full resolution mode: keep the current image visible while loading
                        // the next one in the background. No flicker.
                        self.navigate_full_resolution(prev_path)
                    } else {
                        self.handle_show_image_preview(prev_path, pane_images)
                    };
                    
                    return Task::batch(vec![
                        preview_task,
                        reset_task
                    ]);
                }
            }
        }
        Task::none()
    }

    /// Navigate to a new image while staying in full resolution mode.
    /// Updates the path and sharpness info immediately, but keeps the current image
    /// displayed until the new full-resolution image has finished loading.
    /// This avoids the flicker of closing/reopening the preview window.
    fn navigate_full_resolution(&mut self, new_path: PathBuf) -> Task<Message> {
        // Update path immediately so the title/info bar shows the new filename
        self.preview_image_path = Some(new_path.clone());
        // Stay in full resolution mode — do NOT reset preview_is_thumbnail
        // Do NOT clear preview_handle — keep showing the old image until new one arrives
        
        // Update sharpness info for the new image
        if let Some(first_img) = self.images.first() {
            let sharpness_dir = first_img.parent()
                .unwrap_or(std::path::Path::new("."))
                .join("sharpness");
            
            let yaml_name = new_path.file_stem()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string() + ".yml";
            let yaml_path = sharpness_dir.join(yaml_name);
            
            self.preview_sharpness_info = crate::sharpness_cache::SharpnessInfo::load_from_file(&yaml_path).ok();
        }
        
        // Load the screen-optimized image in the background.
        // When it arrives via ImagePreviewLoaded, the handle will be swapped seamlessly.
        let path = new_path.clone();
        let screen_w = self.window_width;
        let screen_h = self.window_height;
        Task::perform(
            async move {
                let p = path.clone();
                match tokio::task::spawn_blocking(move || {
                    thumbnail::load_preview_for_screen(&p, screen_w, screen_h)
                }).await {
                    Ok(Ok(handle)) => (path, handle, false),
                    _ => {
                        let handle = iced::widget::image::Handle::from_path(&path);
                        (path, handle, false)
                    }
                }
            },
            |(path, handle, is_thumbnail)| Message::ImagePreviewLoaded(path, handle, is_thumbnail),
        )
    }

    /// Handle NavigationThrottleReset
    pub fn handle_navigation_throttle_reset(&mut self) -> Task<Message> {
        self.preview_navigation_throttle = false;
        Task::none()
    }

    /// Handle ImportedScrollChanged
    pub fn handle_imported_scroll_changed(&mut self, offset: f32) -> Task<Message> {
        self.imported_scroll_offset = offset;
        Task::none()
    }

    /// Handle SharpnessScrollChanged
    pub fn handle_sharpness_scroll_changed(&mut self, offset: f32) -> Task<Message> {
        self.sharpness_scroll_offset = offset;
        Task::none()
    }

    /// Handle AlignedScrollChanged
    pub fn handle_aligned_scroll_changed(&mut self, offset: f32) -> Task<Message> {
        self.aligned_scroll_offset = offset;
        Task::none()
    }

    /// Handle BunchesScrollChanged
    pub fn handle_bunches_scroll_changed(&mut self, offset: f32) -> Task<Message> {
        self.bunches_scroll_offset = offset;
        Task::none()
    }

    /// Handle FinalScrollChanged
    pub fn handle_final_scroll_changed(&mut self, offset: f32) -> Task<Message> {
        self.final_scroll_offset = offset;
        Task::none()
    }

    /// Handle DeletePreviewImage - delete the currently previewed image and all its cache derivatives.
    ///
    /// Deletes:
    /// - The source image file itself
    /// - Thumbnail cache (`.thumbnails/<name>.png`)
    /// - Preview cache (`.previews/<stem>_*.jpg`)
    /// - Sharpness YAML (if exists)
    /// - Entry from in-memory thumbnail cache
    /// - Entry from all image lists
    ///
    /// After deletion, navigates to the next image or closes preview if none remain.
    pub fn handle_delete_preview_image(&mut self) -> Task<Message> {
        // Only act if a preview is open
        let path = match self.preview_image_path.clone() {
            Some(p) => p,
            None => return Task::none(),
        };

        log::info!("Deleting image and all derivatives: {}", path.display());

        // 1. Delete cache derivatives (thumbnail + preview caches)
        thumbnail::delete_cache_for_image(&path);

        // 2. Delete sharpness YAML if it exists
        if let Some(first_img) = self.images.first() {
            let sharpness_dir = first_img.parent()
                .unwrap_or(std::path::Path::new("."))
                .join("sharpness");
            let yaml_name = path.file_stem()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string() + ".yml";
            let yaml_path = sharpness_dir.join(&yaml_name);
            if yaml_path.exists() {
                if let Err(e) = std::fs::remove_file(&yaml_path) {
                    log::warn!("Failed to delete sharpness YAML {}: {}", yaml_path.display(), e);
                } else {
                    log::info!("Deleted sharpness YAML: {}", yaml_path.display());
                }
            }
        }

        // 3. Delete the source image file
        if let Err(e) = std::fs::remove_file(&path) {
            log::error!("Failed to delete image file {}: {}", path.display(), e);
            self.status = format!("Error deleting {}: {}", path.file_name().unwrap_or_default().to_string_lossy(), e);
            return Task::none();
        }
        log::info!("Deleted image file: {}", path.display());

        // 4. Remove from in-memory thumbnail cache
        if let Ok(mut cache) = self.thumbnail_cache.write() {
            cache.remove(&path);
        }

        // 5. Remove from all image lists
        self.images.retain(|p| p != &path);
        self.sharpness_images.retain(|p| {
            // Sharpness images are YAML files — match by stem
            let yaml_stem = p.file_stem().unwrap_or_default().to_string_lossy();
            let img_stem = path.file_stem().unwrap_or_default().to_string_lossy();
            yaml_stem != img_stem
        });
        self.aligned_images.retain(|p| p != &path);
        self.bunch_images.retain(|p| p != &path);
        self.final_images.retain(|p| p != &path);

        // 6. Remove from selection lists
        self.selected_imported.retain(|p| p != &path);
        self.selected_sharpness.retain(|p| p != &path);
        self.selected_aligned.retain(|p| p != &path);
        self.selected_bunches.retain(|p| p != &path);

        // 7. Determine next image in the pane and navigate or close
        let current_pane = &self.preview_current_pane;
        let current_index = current_pane.iter().position(|p| p == &path);

        // Remove from the pane navigation list
        self.preview_current_pane.retain(|p| p != &path);

        let filename = path.file_name().unwrap_or_default().to_string_lossy().to_string();

        if self.preview_current_pane.is_empty() {
            // No more images in this pane — close preview
            self.status = format!("Deleted: {}", filename);
            self.handle_close_image_preview()
        } else {
            // Navigate to the next image (or last if we were at the end)
            let next_index = match current_index {
                Some(idx) => idx.min(self.preview_current_pane.len() - 1),
                None => 0,
            };
            let next_path = self.preview_current_pane[next_index].clone();
            let pane_images = self.preview_current_pane.clone();
            let was_full_resolution = !self.preview_is_thumbnail;

            self.status = format!("Deleted: {}", filename);

            if was_full_resolution {
                self.navigate_full_resolution(next_path)
            } else {
                self.handle_show_image_preview(next_path, pane_images)
            }
        }
    }
}
