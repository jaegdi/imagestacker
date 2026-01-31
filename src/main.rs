mod alignment;
mod config;
mod gui;
mod image_io;
mod logger;
mod messages;
mod post_processing;
mod settings;
mod sharpness;
mod stacking;
mod system_info;
mod thumbnail;

use gui::ImageStacker;
use iced::{daemon, window};
use clap::Parser;
use std::path::PathBuf;

/// Rust Image Stacker - Focus Stacking Application
#[derive(Parser, Debug)]
#[command(name = "imagestacker")]
#[command(about = "A focus stacking application for combining multiple images", long_about = None)]
struct Args {
    /// Directory containing images to import automatically
    #[arg(short, long, value_name = "DIR")]
    import: Option<PathBuf>,
}

pub fn main() -> iced::Result {
    std::env::set_var("WGPU_VALIDATION", "0");
    if std::env::var("RUST_LOG").is_err() {
        std::env::set_var("RUST_LOG", "imagestacker=info,wgpu_hal=off,wgpu_core=off");
    }
    // std::env::set_var("WGPU_BACKEND", "gl");
    logger::DualLogger::init();
    
    // Parse command line arguments
    let args = Args::parse();
    
    // Calculate initial window size as 2/3 of typical screen width
    // Common screen resolutions: 1920x1080, 2560x1440, 3840x2160
    // We'll use 2/3 of 1920 = 1280 as a safe default for most displays
    // Height maintains a good aspect ratio for the 4-pane layout
    let window_width = 1280.0;  // 2/3 of 1920px (Full HD)
    let window_height = 900.0;   // Maintains good aspect ratio
    
    daemon(ImageStacker::title, ImageStacker::update, ImageStacker::view)
        .subscription(ImageStacker::subscription)
        .theme(ImageStacker::theme)
        .run_with(move || {
            let (_id, open) = window::open(window::Settings {
                size: iced::Size::new(window_width, window_height),
                resizable: true,
                ..Default::default()
            });
            
            // Create ImageStacker with optional import directory
            let app = ImageStacker::default();
            
            // Determine initial task - combine window open with optional folder load
            let initial_task = if let Some(import_dir) = args.import {
                // Convert to absolute path
                let abs_path = if import_dir.is_absolute() {
                    import_dir
                } else {
                    std::env::current_dir()
                        .unwrap_or_else(|_| PathBuf::from("."))
                        .join(&import_dir)
                };
                
                if abs_path.is_dir() {
                    log::info!("Auto-importing directory from CLI: {:?}", abs_path);
                    // Combine window open task with folder load
                    iced::Task::batch(vec![
                        open.map(|_| messages::Message::None),
                        iced::Task::done(messages::Message::LoadFolder(abs_path))
                    ])
                } else {
                    log::warn!("Provided import path is not a directory: {:?}", abs_path);
                    open.map(|_| messages::Message::None)
                }
            } else {
                open.map(|_| messages::Message::None)
            };
            
            (app, initial_task)
        })
}
