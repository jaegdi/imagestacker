use crate::processing;
use iced::widget::{
    button, column, container, image as iced_image, row, scrollable, text, text_input,
};
use iced::Length;
use iced::{Element, Task, Theme};
use opencv::prelude::*;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

#[derive(Debug, Clone)]
pub enum Message {
    AddImages,
    ImagesSelected(Vec<PathBuf>),
    AlignImages,
    AlignmentDone(Result<String, String>),
    StackImages,
    StackingDone(Result<(Vec<u8>, Mat), String>),
    SaveImage,
    None,
}

pub struct ImageStacker {
    images: Vec<PathBuf>,
    loaded_images: Arc<Mutex<Vec<Mat>>>,
    status: String,
    preview_handle: Option<iced::widget::image::Handle>,
    result_mat: Option<Mat>,
}

impl Default for ImageStacker {
    fn default() -> Self {
        Self {
            images: Vec::new(),
            loaded_images: Arc::new(Mutex::new(Vec::new())),
            status: "Ready".to_string(),
            preview_handle: None,
            result_mat: None,
        }
    }
}

impl ImageStacker {
    pub fn update(&mut self, message: Message) -> Task<Message> {
        match message {
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
            Message::ImagesSelected(paths) => {
                self.images.extend(paths.clone());
                self.status = format!("Loaded {} images", self.images.len());

                // Load images into memory
                let mut mats = Vec::new();
                for path in &paths {
                    if let Ok(mat) = processing::load_image(path) {
                        mats.push(mat);
                    }
                }
                if let Ok(mut locked) = self.loaded_images.lock() {
                    locked.extend(mats);
                }

                Task::none()
            }
            Message::AlignImages => {
                if let Some(first_path) = self.images.first() {
                    let output_dir = first_path
                        .parent()
                        .unwrap_or(std::path::Path::new("."))
                        .to_path_buf();
                    self.status =
                        format!("Aligning images (saving to {})...", output_dir.display());
                    let images_clone = self.loaded_images.clone();

                    Task::perform(
                        async move {
                            let mut locked = images_clone.lock().unwrap();
                            match processing::align_images(&mut locked, &output_dir) {
                                Ok(_) => Message::AlignmentDone(Ok("Aligned".to_string())),
                                Err(e) => Message::AlignmentDone(Err(e.to_string())),
                            }
                        },
                        |msg| msg,
                    )
                } else {
                    self.status = "No images loaded".to_string();
                    Task::none()
                }
            }
            Message::AlignmentDone(result) => {
                match result {
                    Ok(msg) => self.status = msg,
                    Err(e) => self.status = format!("Alignment failed: {}", e),
                }
                Task::none()
            }
            Message::StackImages => {
                if let Some(first_path) = self.images.first() {
                    let output_dir = first_path
                        .parent()
                        .unwrap_or(std::path::Path::new("."))
                        .to_path_buf();
                    self.status =
                        format!("Stacking images (saving to {})...", output_dir.display());
                    let images_clone = self.loaded_images.clone();

                    Task::perform(
                        async move {
                            let locked = images_clone.lock().unwrap();
                            match processing::stack_images(&locked, &output_dir) {
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
                Task::none()
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
                                    Message::None // Could add a Saved message
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
            Message::None => Task::none(),
        }
    }

    pub fn view(&self) -> Element<'_, Message> {
        let sidebar = column![
            button("Add Images").on_press(Message::AddImages),
            button("Align").on_press(Message::AlignImages),
            button("Stack").on_press(Message::StackImages),
            button("Save").on_press(Message::SaveImage),
            scrollable(
                column(self.images.iter().map(|path| {
                    text(path.file_name().unwrap_or_default().to_string_lossy()).into()
                }))
                .spacing(5)
            )
        ]
        .spacing(10)
        .padding(10)
        .width(Length::FillPortion(1));

        let content_element: Element<'_, Message> = if let Some(handle) = &self.preview_handle {
            iced_image(handle.clone())
                .width(Length::Fill)
                .height(Length::Fill)
                .content_fit(iced::ContentFit::Contain)
                .into()
        } else {
            text("Image Preview Area").into()
        };

        let content = container(content_element)
            .width(Length::FillPortion(4))
            .height(Length::Fill)
            .center_x(Length::Fill)
            .center_y(Length::Fill);

        column![
            row![sidebar, content].height(Length::Fill),
            container(text_input("", &self.status).size(14))
                .padding(5)
                .width(Length::Fill)
                .style(|_| container::Style::default()
                    .background(iced::Color::from_rgb(0.1, 0.1, 0.1)))
        ]
        .into()
    }

    pub fn theme(&self) -> Theme {
        Theme::Dark
    }
}
