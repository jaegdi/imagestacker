use crate::processing;
use iced::widget::{
    button, column, container, image as iced_image, row, scrollable, text, text_input,
};
use iced::Length;
use iced::{Element, Task, Theme};
use opencv::prelude::*;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

#[derive(Debug, Clone)]
pub enum Message {
    AddImages,
    AddFolder,
    ImagesSelected(Vec<PathBuf>),
    ThumbnailUpdated(PathBuf, iced::widget::image::Handle),
    InternalPathsScanned(Vec<PathBuf>, Vec<PathBuf>, Vec<PathBuf>, Vec<PathBuf>),
    AlignImages,
    AlignImagesConfirmed(bool),
    AlignmentDone(Result<opencv::core::Rect, String>),
    StackImages,
    StackingDone(Result<(Vec<u8>, Mat), String>),
    SaveImage,
    OpenImage(PathBuf),
    RefreshPanes,
    Exit,
    None,
}

pub struct ImageStacker {
    images: Vec<PathBuf>,
    aligned_images: Vec<PathBuf>,
    bunch_images: Vec<PathBuf>,
    final_images: Vec<PathBuf>,
    thumbnail_cache: Arc<Mutex<HashMap<PathBuf, iced::widget::image::Handle>>>,
    status: String,
    preview_handle: Option<iced::widget::image::Handle>,
    result_mat: Option<Mat>,
    crop_rect: Option<opencv::core::Rect>,
}

impl Default for ImageStacker {
    fn default() -> Self {
        Self {
            images: Vec::new(),
            aligned_images: Vec::new(),
            bunch_images: Vec::new(),
            final_images: Vec::new(),
            thumbnail_cache: Arc::new(Mutex::new(HashMap::new())),
            status: "Ready".to_string(),
            preview_handle: None,
            result_mat: None,
            crop_rect: None,
        }
    }
}

impl ImageStacker {
    pub fn update(&mut self, message: Message) -> Task<Message> {
        let task = match message {
            Message::AddImages => Task::perform(
                async {
                    let files = rfd::AsyncFileDialog::new()
                        .add_filter("Images", &["jpg", "jpeg", "png", "tif", "tiff"])
                        .pick_files()
                        .await;

                    if let Some(files) = files {
                        let paths = files.into_iter().map(|f| f.path().to_path_buf()).collect();
                        Message::ImagesSelected(paths)
                    } else {
                        Message::None
                    }
                },
                |msg| msg,
            ),
            Message::AddFolder => Task::perform(
                async {
                    let folder = rfd::AsyncFileDialog::new().pick_folder().await;

                    if let Some(folder) = folder {
                        let path = folder.path();
                        let mut paths = Vec::new();
                        if let Ok(entries) = std::fs::read_dir(path) {
                            for entry in entries.flatten() {
                                let p = entry.path();
                                if p.is_file() {
                                    if let Some(ext) = p.extension().and_then(|e| e.to_str()) {
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
                        Message::ImagesSelected(paths)
                    } else {
                        Message::None
                    }
                },
                |msg| msg,
            ),
            Message::ImagesSelected(paths) => {
                self.images.extend(paths.clone());
                self.status = format!("Loaded {} images", self.images.len());

                let cache = self.thumbnail_cache.clone();
                // Use rayon to generate thumbnails in parallel but still return individual messages
                Task::batch(paths.into_iter().map(|path| {
                    let cache = cache.clone();
                    Task::perform(
                        async move {
                            if let Ok(handle) = generate_thumbnail(&path) {
                                let mut locked = cache.lock().unwrap();
                                locked.insert(path.clone(), handle.clone());
                                Message::ThumbnailUpdated(path, handle)
                            } else {
                                Message::None
                            }
                        },
                        |msg| msg,
                    )
                }))
            }
            Message::ThumbnailUpdated(path, _handle) => {
                log::trace!("Thumbnail updated for {}", path.display());
                Task::none()
            }
            Message::InternalPathsScanned(aligned, bunches, final_imgs, paths_to_process) => {
                self.aligned_images = aligned;
                self.bunch_images = bunches;
                self.final_images = final_imgs;

                let cache = self.thumbnail_cache.clone();
                Task::batch(paths_to_process.into_iter().map(|path| {
                    let cache = cache.clone();
                    Task::perform(
                        async move {
                            if let Ok(handle) = generate_thumbnail(&path) {
                                let mut locked = cache.lock().unwrap();
                                locked.insert(path.clone(), handle.clone());
                                Message::ThumbnailUpdated(path, handle)
                            } else {
                                Message::None
                            }
                        },
                        |msg| msg,
                    )
                }))
            }
            Message::AlignImages => {
                if let Some(first_path) = self.images.first() {
                    let output_dir = first_path
                        .parent()
                        .unwrap_or(std::path::Path::new("."))
                        .to_path_buf();
                    let aligned_dir = output_dir.join("aligned");

                    if aligned_dir.exists() && aligned_dir.is_dir() {
                        // Check if it contains images
                        let has_images = std::fs::read_dir(&aligned_dir)
                            .map(|mut entries| {
                                entries.any(|e| e.is_ok() && e.unwrap().path().is_file())
                            })
                            .unwrap_or(false);

                        if has_images {
                            return Task::perform(
                                async move {
                                    let confirmed = rfd::AsyncMessageDialog::new()
                                        .set_title("Reuse Aligned Images?")
                                        .set_description("Aligned images already exist. Use them instead of re-aligning?")
                                        .set_buttons(rfd::MessageButtons::YesNo)
                                        .show()
                                        .await;
                                    Message::AlignImagesConfirmed(
                                        confirmed == rfd::MessageDialogResult::Yes,
                                    )
                                },
                                |msg| msg,
                            );
                        }
                    }
                    Task::done(Message::AlignImagesConfirmed(false))
                } else {
                    self.status = "No images loaded".to_string();
                    Task::none()
                }
            }
            Message::AlignImagesConfirmed(reuse) => {
                if let Some(first_path) = self.images.first() {
                    let output_dir = first_path
                        .parent()
                        .unwrap_or(std::path::Path::new("."))
                        .to_path_buf();

                    if reuse {
                        log::info!(
                            "User confirmed reuse of existing aligned images in {}",
                            output_dir.display()
                        );
                        self.status = "Using existing aligned images".to_string();
                        Task::done(Message::RefreshPanes)
                    } else {
                        self.status =
                            format!("Aligning images (saving to {})...", output_dir.display());
                        let images_paths = self.images.clone();

                        Task::perform(
                            async move {
                                // Load images on demand
                                let mut mats = Vec::new();
                                for path in &images_paths {
                                    if let Ok(mat) = processing::load_image(path) {
                                        mats.push(mat);
                                    }
                                }

                                match processing::align_images(&mut mats, &output_dir) {
                                    Ok(rect) => Message::AlignmentDone(Ok(rect)),
                                    Err(e) => Message::AlignmentDone(Err(e.to_string())),
                                }
                            },
                            |msg| msg,
                        )
                    }
                } else {
                    Task::none()
                }
            }
            Message::AlignmentDone(result) => {
                match result {
                    Ok(rect) => {
                        self.status = "Aligned".to_string();
                        self.crop_rect = Some(rect);
                    }
                    Err(e) => self.status = format!("Alignment failed: {}", e),
                }
                Task::done(Message::RefreshPanes)
            }
            Message::StackImages => {
                if let Some(first_path) = self.images.first() {
                    let output_dir = first_path
                        .parent()
                        .unwrap_or(std::path::Path::new("."))
                        .to_path_buf();
                    self.status =
                        format!("Stacking images (saving to {})...", output_dir.display());
                    let images_paths = if self.aligned_images.is_empty() {
                        self.images.clone()
                    } else {
                        self.aligned_images.clone()
                    };

                    let crop_rect = self.crop_rect;
                    Task::perform(
                        async move {
                            match processing::stack_images(&images_paths, &output_dir, crop_rect) {
                                Ok(res) => {
                                    // Convert to PNG bytes for display
                                    let mut buf = opencv::core::Vector::new();
                                    if opencv::imgcodecs::imencode(
                                        ".png",
                                        &res,
                                        &mut buf,
                                        &opencv::core::Vector::new(),
                                    )
                                    .is_ok()
                                    {
                                        Message::StackingDone(Ok((buf.to_vec(), res)))
                                    } else {
                                        Message::StackingDone(Err(
                                            "Failed to encode image".to_string()
                                        ))
                                    }
                                }
                                Err(e) => Message::StackingDone(Err(e.to_string())),
                            }
                        },
                        |msg| msg,
                    )
                } else {
                    self.status = "No images loaded".to_string();
                    Task::none()
                }
            }
            Message::StackingDone(result) => {
                match result {
                    Ok((bytes, mat)) => {
                        self.status = "Stacking complete".to_string();
                        self.preview_handle = Some(iced::widget::image::Handle::from_bytes(bytes));
                        self.result_mat = Some(mat);
                    }
                    Err(e) => self.status = format!("Stacking failed: {}", e),
                }
                Task::done(Message::RefreshPanes)
            }
            Message::SaveImage => {
                if let Some(mat) = &self.result_mat {
                    let mat = mat.clone();
                    Task::perform(
                        async move {
                            let file = rfd::AsyncFileDialog::new()
                                .add_filter("PNG", &["png"])
                                .add_filter("JPEG", &["jpg", "jpeg"])
                                .save_file()
                                .await;

                            if let Some(file) = file {
                                let path = file.path();
                                if opencv::imgcodecs::imwrite(
                                    path.to_str().unwrap(),
                                    &mat,
                                    &opencv::core::Vector::new(),
                                )
                                .is_ok()
                                {
                                    Message::RefreshPanes
                                } else {
                                    Message::None
                                }
                            } else {
                                Message::None
                            }
                        },
                        |msg| msg,
                    )
                } else {
                    self.status = "No image to save".to_string();
                    Task::none()
                }
            }
            Message::OpenImage(path) => {
                let _ = opener::open(path);
                Task::none()
            }
            Message::RefreshPanes => {
                if let Some(first_path) = self.images.first() {
                    let base_dir = first_path
                        .parent()
                        .unwrap_or(std::path::Path::new("."))
                        .to_path_buf();
                    let cache = self.thumbnail_cache.clone();

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

                            let mut all_new_paths = Vec::new();
                            all_new_paths.extend(aligned.clone());
                            all_new_paths.extend(bunches.clone());
                            all_new_paths.extend(final_imgs.clone());

                            let cache_locked = cache.lock().unwrap();
                            let paths_to_process: Vec<_> = all_new_paths
                                .into_iter()
                                .filter(|p| !cache_locked.contains_key(p))
                                .collect();
                            drop(cache_locked);

                            // We can't easily return multiple messages from one Task::perform without a stream
                            // but we can return the scanned paths and then trigger another message.
                            (aligned, bunches, final_imgs, paths_to_process)
                        },
                        |(aligned, bunches, final_imgs, paths_to_process)| {
                            // This is a bit tricky in iced 0.13 without streams for incremental updates
                            // but we can at least update the paths and then start another task for thumbnails.
                            Message::InternalPathsScanned(
                                aligned,
                                bunches,
                                final_imgs,
                                paths_to_process,
                            )
                        },
                    );
                }
                Task::none()
            }
            Message::Exit => {
                std::process::exit(0);
            }
            Message::None => Task::none(),
        };
        task
    }

    pub fn view(&self) -> Element<'_, Message> {
        let buttons = row![
            button("Add Images").on_press(Message::AddImages),
            button("Add Folder").on_press(Message::AddFolder),
            button("Align").on_press(Message::AlignImages),
            button("Stack").on_press(Message::StackImages),
            button("Save").on_press(Message::SaveImage),
            button("Exit").on_press(Message::Exit),
        ]
        .spacing(10)
        .padding(10);

        let panes = row![
            self.render_pane("Imported", &self.images),
            self.render_pane("Aligned", &self.aligned_images),
            self.render_pane("Bunches", &self.bunch_images),
            self.render_pane("Final", &self.final_images),
        ]
        .spacing(10)
        .padding(10)
        .height(Length::Fill);

        column![
            buttons,
            panes,
            container(text_input("", &self.status).size(14))
                .padding(5)
                .width(Length::Fill)
                .style(|_| container::Style::default()
                    .background(iced::Color::from_rgb(0.1, 0.1, 0.1)))
        ]
        .into()
    }

    fn render_pane<'a>(&self, title: &'a str, images: &'a [PathBuf]) -> Element<'a, Message> {
        let cache = self.thumbnail_cache.lock().unwrap();
        let content = column(images.iter().map(|path| {
            let path_clone = path.clone();
            let handle = cache.get(path).cloned();

            let image_widget: Element<Message> = if let Some(h) = handle {
                iced_image(h)
                    .width(100)
                    .height(100)
                    .content_fit(iced::ContentFit::Cover)
                    .into()
            } else {
                container(text("Loading...").size(10))
                    .width(100)
                    .height(100)
                    .center_x(Length::Fill)
                    .center_y(Length::Fill)
                    .style(|_| {
                        container::Style::default().background(iced::Color::from_rgb(0.2, 0.2, 0.2))
                    })
                    .into()
            };

            button(
                column![
                    image_widget,
                    text(path.file_name().unwrap_or_default().to_string_lossy()).size(10)
                ]
                .align_x(iced::Alignment::Center),
            )
            .on_press(Message::OpenImage(path_clone))
            .style(button::secondary)
            .into()
        }))
        .spacing(10)
        .height(Length::Shrink)
        .align_x(iced::Alignment::Center);

        container(
            column![
                text(title)
                    .size(18)
                    .width(Length::Fill)
                    .align_x(iced::Alignment::Center),
                scrollable(content).height(Length::Fill)
            ]
            .spacing(10),
        )
        .width(Length::FillPortion(1))
        .height(Length::Fill)
        .padding(5)
        .style(|_| {
            container::Style::default().border(
                iced::Border::default()
                    .width(1.0)
                    .color(iced::Color::from_rgb(0.3, 0.3, 0.3)),
            )
        })
        .into()
    }

    pub fn theme(&self) -> Theme {
        Theme::Dark
    }
}

fn generate_thumbnail(path: &PathBuf) -> anyhow::Result<iced::widget::image::Handle> {
    use opencv::core;
    use opencv::imgcodecs;
    use opencv::imgproc;

    let img = imgcodecs::imread(path.to_str().unwrap(), imgcodecs::IMREAD_COLOR)?;

    if img.empty() {
        return Err(anyhow::anyhow!("Failed to load image for thumbnail"));
    }

    let size = img.size()?;
    let max_dim = 200.0;
    let scale = (max_dim / size.width as f64).min(max_dim / size.height as f64);
    let new_size = core::Size::new(
        (size.width as f64 * scale) as i32,
        (size.height as f64 * scale) as i32,
    );

    // Use UMat for GPU-accelerated resizing and color conversion
    let img_umat = img.get_umat(
        core::AccessFlag::ACCESS_READ,
        core::UMatUsageFlags::USAGE_DEFAULT,
    )?;
    let mut small_umat = core::UMat::new(core::UMatUsageFlags::USAGE_DEFAULT);

    imgproc::resize(
        &img_umat,
        &mut small_umat,
        new_size,
        0.0,
        0.0,
        imgproc::INTER_AREA,
    )?;

    let mut rgba_umat = core::UMat::new(core::UMatUsageFlags::USAGE_DEFAULT);
    imgproc::cvt_color(
        &small_umat,
        &mut rgba_umat,
        imgproc::COLOR_BGR2RGBA,
        0,
        core::AlgorithmHint::ALGO_HINT_DEFAULT,
    )?;

    // Get raw pixels from GPU
    let rgba_mat = rgba_umat.get_mat(core::AccessFlag::ACCESS_READ)?;
    let mut pixels = vec![0u8; (rgba_mat.total() * rgba_mat.elem_size()?) as usize];
    let data = rgba_mat.data_bytes()?;
    pixels.copy_from_slice(data);

    Ok(iced::widget::image::Handle::from_rgba(
        new_size.width as u32,
        new_size.height as u32,
        pixels,
    ))
}
