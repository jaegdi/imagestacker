//! Stacking and selection handlers
//!
//! Handlers for:
//! - StackImages, StackSelectedAligned, StackBunches, StackSelectedBunches, StackingDone
//! - Selection mode toggles and operations
//! - OpenImage, OpenImageWithExternalEditor

use std::path::PathBuf;
use std::sync::atomic::Ordering;

use iced::Task;
use opencv::prelude::VectorToVec;

use crate::messages::Message;
use crate::stacking;

use crate::gui::state::ImageStacker;

impl ImageStacker {
    /// Handle StackImages - enter aligned selection mode
    pub fn handle_stack_images(&mut self) -> Task<Message> {
        // Enter aligned selection mode
        self.aligned_selection_mode = true;
        self.selected_aligned.clear();
        self.status = "Select aligned images to stack, then click Stack button below aligned pane".to_string();
        Task::none()
    }

    /// Handle CancelAlignedSelection
    pub fn handle_cancel_aligned_selection(&mut self) -> Task<Message> {
        self.aligned_selection_mode = false;
        self.selected_aligned.clear();
        self.status = "Aligned selection cancelled".to_string();
        Task::none()
    }

    /// Handle ToggleAlignedImage
    pub fn handle_toggle_aligned_image(&mut self, path: PathBuf) -> Task<Message> {
        if let Some(pos) = self.selected_aligned.iter().position(|p| p == &path) {
            self.selected_aligned.remove(pos);
        } else {
            self.selected_aligned.push(path);
        }
        Task::none()
    }

    /// Handle SelectAllAligned
    pub fn handle_select_all_aligned(&mut self) -> Task<Message> {
        self.selected_aligned = self.aligned_images.clone();
        Task::none()
    }

    /// Handle DeselectAllAligned
    pub fn handle_deselect_all_aligned(&mut self) -> Task<Message> {
        self.selected_aligned.clear();
        Task::none()
    }

    /// Handle StackSelectedAligned - stack selected aligned images
    pub fn handle_stack_selected_aligned(&mut self) -> Task<Message> {
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

    /// Handle StackBunches - enter bunch selection mode
    pub fn handle_stack_bunches(&mut self) -> Task<Message> {
        // Enter bunch selection mode
        self.bunch_selection_mode = true;
        self.selected_bunches.clear();
        self.status = "Select bunches to stack, then click Stack button below bunches pane".to_string();
        Task::none()
    }

    /// Handle CancelBunchSelection
    pub fn handle_cancel_bunch_selection(&mut self) -> Task<Message> {
        self.bunch_selection_mode = false;
        self.selected_bunches.clear();
        self.status = "Bunch selection cancelled".to_string();
        Task::none()
    }

    /// Handle ToggleBunchImage
    pub fn handle_toggle_bunch_image(&mut self, path: PathBuf) -> Task<Message> {
        if let Some(pos) = self.selected_bunches.iter().position(|p| p == &path) {
            self.selected_bunches.remove(pos);
        } else {
            self.selected_bunches.push(path);
        }
        Task::none()
    }

    /// Handle SelectAllBunches
    pub fn handle_select_all_bunches(&mut self) -> Task<Message> {
        self.selected_bunches = self.bunch_images.clone();
        Task::none()
    }

    /// Handle DeselectAllBunches
    pub fn handle_deselect_all_bunches(&mut self) -> Task<Message> {
        self.selected_bunches.clear();
        Task::none()
    }

    /// Handle StackSelectedBunches - stack selected bunches
    pub fn handle_stack_selected_bunches(&mut self) -> Task<Message> {
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

    /// Handle StackingDone - process stacking result
    pub fn handle_stacking_done(&mut self, result: Result<(Vec<u8>, opencv::core::Mat), String>) -> Task<Message> {
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

    /// Handle OpenImage - open image in external viewer
    pub fn handle_open_image(&mut self, path: PathBuf) -> Task<Message> {
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

    /// Handle OpenImageWithExternalEditor
    pub fn handle_open_image_with_external_editor(&mut self, path: PathBuf) -> Task<Message> {
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
}
