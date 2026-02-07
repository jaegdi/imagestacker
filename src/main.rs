mod alignment;
mod config;
mod gui;
mod image_io;
mod logger;
mod messages;
mod post_processing;
mod settings;
mod sharpness;
mod sharpness_cache;
mod stacking;
mod system_info;
mod thumbnail;

use gui::ImageStacker;
use iced::{daemon, window, Font};
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
    
    // Try to enable OpenCL for GPU acceleration in OpenCV
    // If it fails or causes issues, OpenCV will fall back to CPU
    match opencv::core::set_use_opencl(true) {
        Ok(_) => {
            match opencv::core::use_opencl() {
                Ok(true) => log::info!("OpenCV GPU acceleration (OpenCL) enabled"),
                Ok(false) => log::info!("OpenCV using CPU (OpenCL not available)"),
                Err(e) => log::warn!("Could not check OpenCL status: {}", e),
            }
        }
        Err(e) => {
            log::warn!("Could not enable OpenCL: {}, using CPU", e);
        }
    }
    
    // Parse command line arguments
    let args = Args::parse();
    
    // Load config to get font setting (must happen before daemon starts)
    let saved_config = settings::load_settings();
    let font_name: &'static str = Box::leak(saved_config.default_font.clone().into_boxed_str());
    
    // Calculate initial window size to comfortably fit all 5 panes
    // Each pane needs ~280-300px width + spacing
    // 5 panes * 300px + spacing = ~1600px minimum
    // Height maintains a good aspect ratio for the 4-pane layout
    let window_width = 1600.0;  // Wide enough for all 5 panes without scrolling
    let window_height = 900.0;   // Maintains good aspect ratio
    
    daemon(ImageStacker::title, ImageStacker::update, ImageStacker::view)
        .subscription(ImageStacker::subscription)
        .theme(ImageStacker::theme)
        .default_font(Font {
            family: iced::font::Family::Name(font_name),
            ..Font::DEFAULT
        })
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
