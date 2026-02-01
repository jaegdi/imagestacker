//! Log capture for GUI display
//!
//! This module provides a thread-safe log buffer that captures log messages
//! for display in the GUI log window.

use std::sync::{Mutex, OnceLock};

static LOG_BUFFER: OnceLock<Mutex<Vec<String>>> = OnceLock::new();

/// Get a reference to the log buffer
pub fn get_log_buffer() -> &'static Mutex<Vec<String>> {
    LOG_BUFFER.get_or_init(|| Mutex::new(Vec::new()))
}

/// Append a log message to the buffer
pub fn append_log(message: String) {
    if let Ok(mut buffer) = get_log_buffer().lock() {
        buffer.push(message);
        // Keep only last 1000 messages to avoid memory issues
        if buffer.len() > 1000 {
            buffer.drain(0..100);
        }
    }
}

/// Get all log messages
pub fn get_logs() -> Vec<String> {
    if let Ok(buffer) = get_log_buffer().lock() {
        buffer.clone()
    } else {
        vec!["Failed to access log buffer".to_string()]
    }
}

/// Clear all log messages
#[allow(dead_code)]
pub fn clear_logs() {
    if let Ok(mut buffer) = get_log_buffer().lock() {
        buffer.clear();
    }
}
