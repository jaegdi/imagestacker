//! File loading and thumbnail handling
//!
//! Handlers for:
//! - AddImages, AddFolder, LoadFolder
//! - ImagesSelected, ThumbnailUpdated, InternalPathsScanned

use std::path::PathBuf;
use std::sync::Arc;

use iced::Task;

use crate::messages::Message;
use crate::thumbnail;

use crate::gui::state::ImageStacker;

impl ImageStacker {
    /// Handle AddImages - open file dialog to select images
    pub fn handle_add_images(&mut self) -> Task<Message> {
        Task::perform(
            async {
                let files = rfd::AsyncFileDialog::new()
                    .add_filter("Images", &["jpg", "jpeg", "png", "tif", "tiff", "JPG", "JPEG", "PNG", "TIF", "TIFF"])
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
        )
    }

    /// Handle AddFolder - open folder dialog
    pub fn handle_add_folder(&mut self) -> Task<Message> {
        Task::perform(
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
        )
    }

    /// Handle LoadFolder - scan a folder for images and subdirectories
    pub fn handle_load_folder(&mut self, path: PathBuf) -> Task<Message> {
        Task::perform(
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
                
                // Check for sharpness YAML files
                let mut sharpness_paths = Vec::new();
                let sharpness_dir = path.join("sharpness");
                if sharpness_dir.exists() {
                    if let Ok(entries) = std::fs::read_dir(&sharpness_dir) {
                        for entry in entries.flatten() {
                            let path = entry.path();
                            if path.is_file() {
                                if let Some(ext) = path.extension() {
                                    if ext == "yml" || ext == "yaml" {
                                        sharpness_paths.push(path);
                                    }
                                }
                            }
                        }
                    }
                }
                sharpness_paths.sort();
                
                // Create a combined message with all paths
                if !sharpness_paths.is_empty() || !aligned_paths.is_empty() || !bunch_paths.is_empty() || !final_paths.is_empty() {
                    // We have existing processed images
                    Message::InternalPathsScanned(paths, sharpness_paths, aligned_paths, bunch_paths, final_paths)
                } else {
                    // No existing processed images, just load the main images
                    Message::ImagesSelected(paths)
                }
            },
            |msg| msg,
        )
    }

    /// Handle ImagesSelected - load selected images and generate thumbnails
    pub fn handle_images_selected(&mut self, paths: Vec<PathBuf>) -> Task<Message> {
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

    /// Handle ThumbnailUpdated - log thumbnail update
    pub fn handle_thumbnail_updated(&mut self, path: PathBuf) -> Task<Message> {
        log::trace!("Thumbnail updated for {}", path.display());
        Task::none()
    }

    /// Handle InternalPathsScanned - process scanned paths from folder loading
    pub fn handle_internal_paths_scanned(
        &mut self,
        paths_to_process: Vec<PathBuf>,  // imported
        sharpness: Vec<PathBuf>,
        aligned: Vec<PathBuf>,
        bunches: Vec<PathBuf>,
        final_imgs: Vec<PathBuf>,
    ) -> Task<Message> {
        // Determine if this is an initial load or a refresh
        let is_initial_load = !paths_to_process.is_empty() && self.images.is_empty();
        let is_imported_refresh = !paths_to_process.is_empty() && !self.images.is_empty();
        
        if is_initial_load || is_imported_refresh {
            // Initial load or imported refresh: update imported images list
            if is_imported_refresh {
                // Clear all panes when doing an import refresh (like starting new project)
                self.images.clear();
                self.sharpness_images.clear();
                self.aligned_images.clear();
                self.bunch_images.clear();
                self.final_images.clear();
                
                // Clear entire thumbnail cache for import refresh
                let mut cache = self.thumbnail_cache.write().unwrap();
                cache.clear();
            }
            
            self.images.extend(paths_to_process.clone());
            
            if is_initial_load {
                self.sharpness_images = sharpness.clone();
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
        
        // Handle refresh of individual panes (when paths_to_process is empty)
        // Update the pane based on which vector has content
        if paths_to_process.is_empty() {
            let has_sharpness = !sharpness.is_empty();
            let has_aligned = !aligned.is_empty();
            let has_bunches = !bunches.is_empty();
            let has_final = !final_imgs.is_empty();
            
            // Count how many panes have data
            let active_count = [has_sharpness, has_aligned, has_bunches, has_final]
                .iter()
                .filter(|&&x| x)
                .count();
            
            // Only update if exactly one pane has data (single pane refresh)
            if active_count == 1 {
                if has_sharpness {
                    log::debug!("Refreshing sharpness pane with {} YAML files", sharpness.len());
                    self.sharpness_images.clear();
                    self.sharpness_images.extend(sharpness.clone());
                } else if has_aligned {
                    log::debug!("Refreshing aligned pane with {} images", aligned.len());
                    self.aligned_images.clear();
                    self.aligned_images.extend(aligned.clone());
                } else if has_bunches {
                    log::debug!("Refreshing bunches pane with {} images", bunches.len());
                    self.bunch_images.clear();
                    self.bunch_images.extend(bunches.clone());
                } else if has_final {
                    log::debug!("Refreshing final pane with {} images", final_imgs.len());
                    self.final_images.clear();
                    self.final_images.extend(final_imgs.clone());
                }
            } else if active_count > 1 {
                // Multiple panes have data - this can happen when file watcher triggers
                // multiple events simultaneously (e.g., during batch alignment)
                // Update all panes that have data
                log::debug!("Multi-pane refresh: {} panes active (sharpness:{}, aligned:{}, bunches:{}, final:{})",
                           active_count, sharpness.len(), aligned.len(), bunches.len(), final_imgs.len());
                
                if has_sharpness {
                    self.sharpness_images.clear();
                    self.sharpness_images.extend(sharpness.clone());
                }
                if has_aligned {
                    self.aligned_images.clear();
                    self.aligned_images.extend(aligned.clone());
                }
                if has_bunches {
                    self.bunch_images.clear();
                    self.bunch_images.extend(bunches.clone());
                }
                if has_final {
                    self.final_images.clear();
                    self.final_images.extend(final_imgs.clone());
                }
            }
        }

        // Collect all paths that need thumbnails (only those not already cached)
        let mut all_paths = Vec::new();
        
        if is_initial_load || is_imported_refresh {
            // Initial load or refresh: process paths_to_process
            all_paths.extend(paths_to_process.clone());
        }
        
        // For sharpness refresh, we need to find the corresponding image files
        // and regenerate their thumbnails (already removed from cache above)
        if !sharpness.is_empty() && aligned.is_empty() && bunches.is_empty() && final_imgs.is_empty() {
            // This is a sharpness-only refresh
            // Find the corresponding image files for each YAML file
            for yaml_path in &sharpness {
                if let Some(yaml_stem) = yaml_path.file_stem() {
                    let yaml_stem_str = yaml_stem.to_string_lossy();
                    // Find matching image in self.images
                    for img_path in &self.images {
                        if let Some(img_stem) = img_path.file_stem() {
                            if img_stem.to_string_lossy() == yaml_stem_str {
                                all_paths.push(img_path.clone());
                                break;
                            }
                        }
                    }
                }
            }
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
}
