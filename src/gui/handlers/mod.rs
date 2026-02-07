//! Message handlers split into logical submodules
//!
//! This module organizes the message handlers from the main update function
//! into smaller, focused modules for better maintainability.

pub mod file_handlers;
pub mod alignment_handlers;
pub mod stacking_handlers;
pub mod sharpness_handlers;
pub mod preview_handlers;
pub mod refresh_handlers;
pub mod settings_handlers;
pub mod window_handlers;
