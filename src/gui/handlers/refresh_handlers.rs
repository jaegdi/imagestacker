//! Pane refresh handlers
//!
//! Handlers for:
//! - RefreshPanes, AutoRefreshTick
//! - RefreshImportedPane, RefreshAlignedPane, RefreshBunchesPane, RefreshFinalPane

use std::path::PathBuf;

use iced::Task;

use crate::messages::Message;

use crate::gui::state::ImageStacker;

impl ImageStacker {
    /// Handle RefreshPanes - refresh all panes by rescanning directories
    pub fn handle_refresh_panes(&mut self) -> Task<Message> {
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

    /// Handle AutoRefreshTick - auto-refresh when processing is active
    pub fn handle_auto_refresh_tick(&mut self) -> Task<Message> {
        // Auto-refresh file lists when processing is active
        if self.is_processing {
            Task::done(Message::RefreshPanes)
        } else {
            Task::none()
        }
    }

    /// Handle RefreshImportedPane - refresh imported pane by rescanning base directory
    pub fn handle_refresh_imported_pane(&mut self) -> Task<Message> {
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

    /// Handle RefreshAlignedPane - refresh aligned pane by rescanning aligned directory
    pub fn handle_refresh_aligned_pane(&mut self) -> Task<Message> {
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

    /// Handle RefreshBunchesPane - refresh bunches pane by rescanning bunches directory
    pub fn handle_refresh_bunches_pane(&mut self) -> Task<Message> {
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

    /// Handle RefreshFinalPane - refresh final pane by rescanning final directory
    pub fn handle_refresh_final_pane(&mut self) -> Task<Message> {
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
}
