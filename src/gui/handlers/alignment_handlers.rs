//! Alignment processing handlers
//!
//! Handlers for:
//! - AlignImages, AlignImagesConfirmed, AlignmentDone

use std::sync::atomic::Ordering;

use iced::Task;

use crate::alignment;
use crate::messages::{AlignmentMode, Message};

use crate::gui::state::ImageStacker;

impl ImageStacker {
    /// Handle AlignImages - check for existing aligned images and prompt user
    pub fn handle_align_images(&mut self) -> Task<Message> {
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
                    // Count how many aligned images exist
                    let aligned_count = std::fs::read_dir(&aligned_dir)
                        .map(|entries| entries.flatten().filter(|e| {
                            let p = e.path();
                            p.is_file() && p.extension()
                                .and_then(|ext| ext.to_str())
                                .map(|ext| ["jpg", "jpeg", "png", "tif", "tiff"]
                                    .contains(&ext.to_lowercase().as_str()))
                                .unwrap_or(false)
                        }).count())
                        .unwrap_or(0);
                    
                    return Task::perform(
                        async move {
                            // Dialog with 2 custom buttons: Resume or Start Fresh
                            let choice = rfd::AsyncMessageDialog::new()
                                .set_title("Aligned Images Found")
                                .set_description(&format!(
                                    "Found {} aligned images. What would you like to do?\n\n\
                                    • Resume: Continue from where it stopped (align only missing images)\n\
                                    • Start Fresh: Delete existing and re-align all images",
                                    aligned_count
                                ))
                                .set_buttons(rfd::MessageButtons::OkCancelCustom("Resume".to_string(), "Start Fresh".to_string()))
                                .show()
                                .await;
                            
                            log::info!("Dialog result: {:?}", choice);
                            
                            let mode = match choice {
                                rfd::MessageDialogResult::Ok => {
                                    log::info!("User selected: Ok (Resume)");
                                    Some(AlignmentMode::Resume)
                                }
                                rfd::MessageDialogResult::Cancel => {
                                    log::info!("User selected: Cancel (Start Fresh)");
                                    Some(AlignmentMode::StartFresh)
                                }
                                rfd::MessageDialogResult::Custom(ref text) => {
                                    log::info!("User selected: Custom({})", text);
                                    // Check which button was clicked
                                    if text == "Resume" {
                                        Some(AlignmentMode::Resume)
                                    } else {
                                        Some(AlignmentMode::StartFresh)
                                    }
                                }
                                _ => {
                                    log::info!("Dialog was closed (ESCAPE or X)");
                                    None  // Dialog was closed with ESCAPE or X button
                                }
                            };
                            
                            Message::AlignImagesConfirmed(mode)
                        },
                        |msg| msg,
                    );
                }
            }
            Task::done(Message::AlignImagesConfirmed(Some(AlignmentMode::StartFresh)))
        } else {
            self.status = "No images loaded".to_string();
            Task::none()
        }
    }

    /// Handle AlignImagesConfirmed - start alignment in the selected mode
    pub fn handle_align_images_confirmed(&mut self, mode: Option<AlignmentMode>) -> Task<Message> {
        // If mode is None, the dialog was cancelled
        let mode = match mode {
            Some(m) => m,
            None => {
                log::info!("Alignment dialog was cancelled by user");
                self.status = "Alignment cancelled".to_string();
                return Task::none();
            }
        };
        
        if let Some(first_path) = self.images.first() {
            let output_dir = first_path
                .parent()
                .unwrap_or(std::path::Path::new("."))
                .to_path_buf();

            // Determine which images to process based on mode
            let images_to_process = match mode {
                AlignmentMode::StartFresh => {
                    // Clean up existing aligned and bunches images before starting new alignment
                    let aligned_dir = output_dir.join("aligned");
                    let bunches_dir = output_dir.join("bunches");
                    
                    if aligned_dir.exists() {
                        log::info!("Starting fresh: Cleaning up existing aligned images in {}", aligned_dir.display());
                        if let Err(e) = std::fs::remove_dir_all(&aligned_dir) {
                            log::warn!("Failed to remove aligned directory: {}", e);
                        }
                        // Recreate the directory
                        if let Err(e) = std::fs::create_dir_all(&aligned_dir) {
                            log::warn!("Failed to recreate aligned directory: {}", e);
                        }
                    }
                    
                    if bunches_dir.exists() {
                        log::info!("Starting fresh: Cleaning up existing bunches images in {}", bunches_dir.display());
                        if let Err(e) = std::fs::remove_dir_all(&bunches_dir) {
                            log::warn!("Failed to remove bunches directory: {}", e);
                        }
                        // Recreate the directory
                        if let Err(e) = std::fs::create_dir_all(&bunches_dir) {
                            log::warn!("Failed to recreate bunches directory: {}", e);
                        }
                    }
                    
                    // Clear aligned and bunches panes
                    self.aligned_images.clear();
                    self.bunch_images.clear();
                    
                    // Process all images
                    self.images.clone()
                }
                
                AlignmentMode::Resume => {
                    log::info!("Resume mode: Will skip already aligned images during processing");
                    // In resume mode, we pass ALL images to the alignment function
                    // The function will check if output files already exist and skip them
                    // This preserves the original indices in the filenames
                    self.images.clone()
                }
            };
            
            self.status = "Aligning images...".to_string();
            self.progress_message = "Starting alignment...".to_string();
            self.progress_value = 0.0;
            
            // Reset cancel flag for new operation
            self.cancel_flag.store(false, Ordering::Relaxed);
            
            let proc_config = self.create_processing_config();
            let cancel_flag = self.cancel_flag.clone();
            self.is_processing = true;  // Mark as processing

            Task::run(
                iced::stream::channel(1000, move |mut sender| async move {
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
                            &images_to_process, 
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
        } else {
            Task::none()
        }
    }

    /// Handle AlignmentDone - process alignment result
    pub fn handle_alignment_done(&mut self, result: Result<opencv::core::Rect, String>) -> Task<Message> {
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
}
