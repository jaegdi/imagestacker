mod gui;
mod processing;
mod system_info;

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
    .run()
}
