mod alignment;
mod config;
mod gui;
mod image_io;
mod messages;
mod post_processing;
mod processing;
mod settings;
mod sharpness;
mod stacking;
mod system_info;
mod thumbnail;

use gui::ImageStacker;
use iced::application;

pub fn main() -> iced::Result {
    std::env::set_var("WGPU_VALIDATION", "0");
    if std::env::var("RUST_LOG").is_err() {
        std::env::set_var("RUST_LOG", "imagestacker=info,wgpu_hal=off,wgpu_core=off");
    }
    // std::env::set_var("WGPU_BACKEND", "gl");
    env_logger::init();
    application(
        "Rust Image Stacker",
        ImageStacker::update,
        ImageStacker::view,
    )
    .subscription(ImageStacker::subscription)
    .theme(ImageStacker::theme)
    .window(iced::window::Settings {
        size: iced::Size::new(1800.0, 1000.0),
        resizable: true,
        ..Default::default()
    })
    .run()
}
