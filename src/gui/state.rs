//! Application state for the ImageStacker GUI
//!
//! This module defines the main `ImageStacker` struct that holds all
//! application state including loaded images, configuration, and UI state.

use iced::window;
use opencv::prelude::*;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};
use std::sync::atomic::AtomicBool;

use crate::config::ProcessingConfig;
use crate::settings::load_settings;

/// Main application state
pub struct ImageStacker {
    // Image collections
    pub(crate) images: Vec<PathBuf>,
    pub(crate) aligned_images: Vec<PathBuf>,
    pub(crate) bunch_images: Vec<PathBuf>,
    pub(crate) final_images: Vec<PathBuf>,
    
    // Caching
    pub(crate) thumbnail_cache: Arc<RwLock<HashMap<PathBuf, iced::widget::image::Handle>>>,
    
    // Status and results
    pub(crate) status: String,
    pub(crate) preview_handle: Option<iced::widget::image::Handle>,
    pub(crate) result_mat: Option<Mat>,
    pub(crate) crop_rect: Option<opencv::core::Rect>,
    pub(crate) is_processing: bool,
    
    // Configuration
    pub(crate) config: ProcessingConfig,
    pub(crate) show_settings: bool,
    
    // Window management
    pub(crate) help_window_id: Option<window::Id>,
    pub(crate) log_window_id: Option<window::Id>,
    
    // Progress tracking
    pub(crate) progress_message: String,
    pub(crate) progress_value: f32,
    
    // Image preview state
    pub(crate) preview_image_path: Option<PathBuf>,
    pub(crate) preview_loading: bool,
    pub(crate) preview_is_thumbnail: bool,
    pub(crate) preview_current_pane: Vec<PathBuf>,
    pub(crate) preview_navigation_throttle: bool,
    
    // Scroll position tracking
    pub(crate) imported_scroll_offset: f32,
    pub(crate) aligned_scroll_offset: f32,
    pub(crate) bunches_scroll_offset: f32,
    pub(crate) final_scroll_offset: f32,
    
    // Window sizing
    pub(crate) window_width: f32,
    
    // Selection modes
    pub(crate) aligned_selection_mode: bool,
    pub(crate) selected_aligned: Vec<PathBuf>,
    pub(crate) bunch_selection_mode: bool,
    pub(crate) selected_bunches: Vec<PathBuf>,
    
    // Cancellation flag for background tasks
    pub(crate) cancel_flag: Arc<AtomicBool>,
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
            window_width: 1400.0,
            aligned_selection_mode: false,
            selected_aligned: Vec::new(),
            bunch_selection_mode: false,
            selected_bunches: Vec::new(),
            cancel_flag: Arc::new(AtomicBool::new(false)),
        }
    }
}

impl ImageStacker {
    /// Create a ProcessingConfig from the current state
    pub fn create_processing_config(&self) -> ProcessingConfig {
        self.config.clone()
    }
}
