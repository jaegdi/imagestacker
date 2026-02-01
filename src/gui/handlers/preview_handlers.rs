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

    /// Handle LoadFullImage - load full-resolution image
    pub fn handle_load_full_image(&mut self, path: PathBuf) -> Task<Message> {
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
    }

    /// Handle CloseImagePreview - close preview or cancel processing
    pub fn handle_close_image_preview(&mut self) -> Task<Message> {
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

    /// Handle NextImageInPreview - navigate to next image
    pub fn handle_next_image_in_preview(&mut self) -> Task<Message> {
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
                        self.handle_show_image_preview(next_path, pane_images),
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
                        self.handle_show_image_preview(prev_path, pane_images),
                        reset_task
                    ]);
                }
            }
        }
        Task::none()
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
}
