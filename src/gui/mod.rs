//! GUI module for the ImageStacker application
//!
//! This module provides the complete graphical user interface for the
//! image stacking application using the Iced framework.
//!
//! ## Module Structure
//!
//! - `log_capture` - Thread-safe log buffer for GUI display
//! - `state` - Application state (ImageStacker struct)
//! - `update` - Message handling dispatch
//! - `handlers/` - Message handler implementations
//!   - `file_handlers` - File loading and thumbnail operations
//!   - `alignment_handlers` - Alignment processing
//!   - `stacking_handlers` - Stacking and selection operations
//!   - `preview_handlers` - Image preview and navigation
//!   - `refresh_handlers` - Pane refresh operations
//!   - `settings_handlers` - Settings changes
//!   - `window_handlers` - Window management
//! - `subscriptions` - Event subscriptions (keyboard, mouse, window)
//! - `views/` - UI rendering functions
//!   - `main_view` - Main application view and image preview
//!   - `settings` - Settings panel
//!   - `panes` - Image pane rendering
//!   - `windows` - Help and Log window rendering

pub mod log_capture;
pub mod state;
pub mod theme;
mod update;
mod handlers;
mod subscriptions;
pub mod views;

// Re-export the log capture functions
pub use log_capture::append_log;

// Re-export the main GUI types
pub use state::ImageStacker;

use iced::{Color, Theme, window};
use iced::theme::Palette;

impl ImageStacker {
    /// Custom dark theme for the application
    pub fn theme(&self, _window: window::Id) -> Theme {
        let palette = Palette {
            background: Color::from_rgb(0.10, 0.10, 0.13),
            text: Color::from_rgb(0.92, 0.92, 0.95),
            primary: Color::from_rgb(0.30, 0.52, 0.80),
            success: Color::from_rgb(0.25, 0.62, 0.35),
            danger: Color::from_rgb(0.75, 0.28, 0.28),
        };
        Theme::custom("ImageStacker".to_string(), palette)
    }

    /// Window title
    pub fn title(&self, window: window::Id) -> String {
        if Some(window) == self.help_window_id {
            "Rust Image Stacker - User Manual".to_string()
        } else if Some(window) == self.log_window_id {
            "Rust Image Stacker - Log".to_string()
        } else {
            "Rust Image Stacker".to_string()
        }
    }
}
