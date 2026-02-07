//! Application state for the ImageStacker GUI
//!
//! This module defines the main `ImageStacker` struct that holds all
//! application state including loaded images, configuration, and UI state.

use iced::widget::pane_grid;
use iced::window;
use opencv::prelude::*;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};
use std::sync::atomic::AtomicBool;

use crate::config::ProcessingConfig;
use crate::settings::load_settings;

/// Identifies which pipeline stage a pane displays
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PaneId {
    Imported,
    Sharpness,
    Aligned,
    Bunches,
    Final,
}

impl std::fmt::Display for PaneId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PaneId::Imported => write!(f, "Imported"),
            PaneId::Sharpness => write!(f, "Sharpness"),
            PaneId::Aligned => write!(f, "Aligned"),
            PaneId::Bunches => write!(f, "Bunches"),
            PaneId::Final => write!(f, "Final"),
        }
    }
}

/// Main application state
pub struct ImageStacker {
    // Image collections
    pub(crate) images: Vec<PathBuf>,
    pub(crate) sharpness_images: Vec<PathBuf>,  // YAML files with sharpness data
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
    pub(crate) preview_sharpness_info: Option<crate::sharpness_cache::SharpnessInfo>,
    pub(crate) preview_current_pane: Vec<PathBuf>,
    pub(crate) preview_navigation_throttle: bool,
    
    // Scroll position tracking
    pub(crate) imported_scroll_offset: f32,
    pub(crate) sharpness_scroll_offset: f32,
    pub(crate) aligned_scroll_offset: f32,
    pub(crate) bunches_scroll_offset: f32,
    pub(crate) final_scroll_offset: f32,
    
    // Window sizing
    pub(crate) window_width: f32,
    pub(crate) window_height: f32,
    
    // Selection modes
    pub(crate) imported_selection_mode: bool,
    pub(crate) selected_imported: Vec<PathBuf>,
    pub(crate) sharpness_selection_mode: bool,
    pub(crate) selected_sharpness: Vec<PathBuf>,
    pub(crate) aligned_selection_mode: bool,
    pub(crate) selected_aligned: Vec<PathBuf>,
    pub(crate) bunch_selection_mode: bool,
    pub(crate) selected_bunches: Vec<PathBuf>,
    
    // Cancellation flag for background tasks
    pub(crate) cancel_flag: Arc<AtomicBool>,

    // PaneGrid state for resizable panels
    pub(crate) pane_state: pane_grid::State<PaneId>,
    
    // Available system fonts (populated at startup)
    pub(crate) available_fonts: Vec<String>,
}

impl Default for ImageStacker {
    fn default() -> Self {
        Self {
            images: Vec::new(),
            sharpness_images: Vec::new(),
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
            preview_sharpness_info: None,
            preview_current_pane: Vec::new(),
            preview_navigation_throttle: false,
            imported_scroll_offset: 0.0,
            sharpness_scroll_offset: 0.0,
            aligned_scroll_offset: 0.0,
            bunches_scroll_offset: 0.0,
            final_scroll_offset: 0.0,
            window_width: 1400.0,
            window_height: 900.0,
            imported_selection_mode: false,
            selected_imported: Vec::new(),
            sharpness_selection_mode: false,
            selected_sharpness: Vec::new(),
            aligned_selection_mode: false,
            selected_aligned: Vec::new(),
            bunch_selection_mode: false,
            selected_bunches: Vec::new(),
            cancel_flag: Arc::new(AtomicBool::new(false)),
            pane_state: Self::create_pane_grid(),
            available_fonts: Self::discover_system_fonts(),
        }
    }
}

impl ImageStacker {
    /// Create the initial pane_grid layout: 5 panes in a horizontal row, equally sized
    fn create_pane_grid() -> pane_grid::State<PaneId> {
        use pane_grid::{Axis, Configuration};

        // Build a balanced binary tree of vertical splits for 5 equal panes:
        //   Split(V, 0.4, Split(V, 0.5, Imported, Sharpness), Split(V, 0.333, Aligned, Split(V, 0.5, Bunches, Final)))
        // Ratios: first split 2/5=0.4, left side 1/2=0.5, right side 1/3=0.333, innermost 1/2=0.5
        let config = Configuration::Split {
            axis: Axis::Vertical,
            ratio: 0.4,
            a: Box::new(Configuration::Split {
                axis: Axis::Vertical,
                ratio: 0.5,
                a: Box::new(Configuration::Pane(PaneId::Imported)),
                b: Box::new(Configuration::Pane(PaneId::Sharpness)),
            }),
            b: Box::new(Configuration::Split {
                axis: Axis::Vertical,
                ratio: 0.333,
                a: Box::new(Configuration::Pane(PaneId::Aligned)),
                b: Box::new(Configuration::Split {
                    axis: Axis::Vertical,
                    ratio: 0.5,
                    a: Box::new(Configuration::Pane(PaneId::Bunches)),
                    b: Box::new(Configuration::Pane(PaneId::Final)),
                }),
            }),
        };

        pane_grid::State::with_configuration(config)
    }

    /// Create a ProcessingConfig from the current state
    pub fn create_processing_config(&self) -> ProcessingConfig {
        self.config.clone()
    }

    /// Discover installed system fonts.
    /// Uses platform-specific methods:
    /// - Linux: `fc-list` (fontconfig)
    /// - macOS: `system_profiler SPFontsDataType` or `fc-list` if available
    /// - Windows: PowerShell font enumeration
    fn discover_system_fonts() -> Vec<String> {
        use std::collections::BTreeSet;

        let mut fonts = BTreeSet::new();

        if cfg!(target_os = "linux") {
            Self::discover_fonts_fclist(&mut fonts);
        } else if cfg!(target_os = "macos") {
            // Try fc-list first (available via Homebrew), fall back to system_profiler
            if !Self::discover_fonts_fclist(&mut fonts) {
                Self::discover_fonts_macos(&mut fonts);
            }
        } else if cfg!(target_os = "windows") {
            Self::discover_fonts_windows(&mut fonts);
        }

        // Always ensure DejaVu Sans is in the list as our default
        fonts.insert("DejaVu Sans".to_string());

        let result: Vec<String> = fonts.into_iter().collect();
        log::info!("Discovered {} system fonts", result.len());
        result
    }

    /// Discover fonts using fc-list (Linux, and macOS with Homebrew fontconfig).
    /// Returns true if fonts were found.
    fn discover_fonts_fclist(fonts: &mut std::collections::BTreeSet<String>) -> bool {
        // fc-list :scalable=true family  — lists only scalable (non-bitmap) font families
        match std::process::Command::new("fc-list")
            .args([":scalable=true", "family"])
            .output()
        {
            Ok(output) if output.status.success() => {
                let stdout = String::from_utf8_lossy(&output.stdout);
                for line in stdout.lines() {
                    // fc-list outputs lines like "DejaVu Sans" or "Noto Sans,Noto Sans Thin"
                    // Take the first family name (before any comma)
                    let family = line.split(',').next().unwrap_or("").trim();
                    // Skip empty lines and TeX/math fonts that aren't useful for UI
                    if !family.is_empty()
                        && !family.contains('\\')
                        && !family.contains(".pcf")
                        && !family.starts_with("TeX")
                        && !family.starts_with("bgu")
                        && !family.starts_with("txb")
                    {
                        fonts.insert(family.to_string());
                    }
                }
                !fonts.is_empty()
            }
            Ok(_) => {
                log::warn!("fc-list returned non-zero exit code");
                false
            }
            Err(e) => {
                log::debug!("fc-list not available: {}", e);
                false
            }
        }
    }

    /// Discover fonts on macOS using system_profiler.
    #[allow(dead_code)]
    fn discover_fonts_macos(fonts: &mut std::collections::BTreeSet<String>) {
        // Use the macOS-native font list via CoreText through a small Python snippet,
        // which is more reliable than parsing system_profiler output
        let script = r#"
import subprocess, re
out = subprocess.check_output(["system_profiler", "SPFontsDataType"], text=True)
for line in out.splitlines():
    line = line.strip()
    if line.startswith("Family:"):
        family = line[len("Family:"):].strip()
        if family:
            print(family)
"#;
        match std::process::Command::new("python3")
            .args(["-c", script])
            .output()
        {
            Ok(output) if output.status.success() => {
                let stdout = String::from_utf8_lossy(&output.stdout);
                for line in stdout.lines() {
                    let family = line.trim();
                    if !family.is_empty() {
                        fonts.insert(family.to_string());
                    }
                }
            }
            _ => {
                log::warn!("Could not enumerate macOS fonts, using fallback list");
                for f in ["Helvetica", "Arial", "Times New Roman", "San Francisco", "Menlo"] {
                    fonts.insert(f.to_string());
                }
            }
        }
    }

    /// Discover fonts on Windows using PowerShell.
    #[allow(dead_code)]
    fn discover_fonts_windows(fonts: &mut std::collections::BTreeSet<String>) {
        // Use .NET's InstalledFontCollection via PowerShell
        let script = r#"
Add-Type -AssemblyName System.Drawing
$fc = New-Object System.Drawing.Text.InstalledFontCollection
$fc.Families | ForEach-Object { $_.Name }
"#;
        match std::process::Command::new("powershell")
            .args(["-NoProfile", "-NonInteractive", "-Command", script])
            .output()
        {
            Ok(output) if output.status.success() => {
                let stdout = String::from_utf8_lossy(&output.stdout);
                for line in stdout.lines() {
                    let family = line.trim();
                    if !family.is_empty() {
                        fonts.insert(family.to_string());
                    }
                }
            }
            _ => {
                log::warn!("Could not enumerate Windows fonts, using fallback list");
                for f in ["Arial", "Segoe UI", "Times New Roman", "Calibri", "Verdana"] {
                    fonts.insert(f.to_string());
                }
            }
        }
    }
}
