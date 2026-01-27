use crate::processing;
use crate::system_info;
use iced::widget::{
    button, checkbox, column, container, image as iced_image, row, scrollable, slider, text, text_input,
};
use iced::Length;
use iced::{Element, Task, Theme};
use opencv::prelude::*;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};
use std::time::Duration;

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
    AutoRefreshTick,
    // New: Configuration messages
    ToggleSettings,
    SharpnessThresholdChanged(f32),
    UseAdaptiveBatchSizes(bool),
    UseCLAHE(bool),
    FeatureDetectorChanged(FeatureDetector),
    ProgressUpdate(String, f32),
    Exit,
    None,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FeatureDetector {
    ORB,
    SIFT,
    AKAZE,
}

impl std::fmt::Display for FeatureDetector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FeatureDetector::ORB => write!(f, "ORB (Fast)"),
            FeatureDetector::SIFT => write!(f, "SIFT (Best Quality)"),
            FeatureDetector::AKAZE => write!(f, "AKAZE (Balanced)"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ProcessingConfig {
    pub sharpness_threshold: f32,
    pub use_adaptive_batches: bool,
    pub use_clahe: bool,
    pub feature_detector: FeatureDetector,
    pub batch_config: system_info::BatchSizeConfig,
}

impl Default for ProcessingConfig {
    fn default() -> Self {
        Self {
            sharpness_threshold: 30.0,
            use_adaptive_batches: true,
            use_clahe: true,
            feature_detector: FeatureDetector::ORB,
            batch_config: system_info::BatchSizeConfig::default_config(),
        }
    }
}

pub struct ImageStacker {
    images: Vec<PathBuf>,
    aligned_images: Vec<PathBuf>,
    bunch_images: Vec<PathBuf>,
    final_images: Vec<PathBuf>,
    thumbnail_cache: Arc<RwLock<HashMap<PathBuf, iced::widget::image::Handle>>>,
    status: String,
    preview_handle: Option<iced::widget::image::Handle>,
    result_mat: Option<Mat>,
    crop_rect: Option<opencv::core::Rect>,
    is_processing: bool,
    // New: Configuration and progress
    config: ProcessingConfig,
    show_settings: bool,
    progress_message: String,
    progress_value: f32,
}

impl Default for ImageStacker {
    fn default() -> Self {
        Self {
            images: Vec::new(),
            aligned_images: Vec::new(),
            bunch_images: Vec::new(),
            final_images: Vec::new(),
            thumbnail_cache: Arc::new(RwLock::new(HashMap::new())),
            status: "Ready".to_string(),
            preview_handle: None,
            result_mat: None,
            crop_rect: None,
            is_processing: false,
            config: ProcessingConfig::default(),
            show_settings: false,
            progress_message: String::new(),
            progress_value: 0.0,
        }
    }
}

impl ImageStacker {
    fn create_processing_config(&self) -> processing::ProcessingConfig {
        processing::ProcessingConfig {
            sharpness_threshold: self.config.sharpness_threshold,
            use_clahe: self.config.use_clahe,
            feature_detector: match self.config.feature_detector {
                FeatureDetector::ORB => processing::FeatureDetectorType::ORB,
                FeatureDetector::SIFT => processing::FeatureDetectorType::SIFT,
                FeatureDetector::AKAZE => processing::FeatureDetectorType::AKAZE,
            },
            sharpness_batch_size: self.config.batch_config.sharpness_batch_size,
            feature_batch_size: self.config.batch_config.feature_batch_size,
            warp_batch_size: self.config.batch_config.warp_batch_size,
            stacking_batch_size: self.config.batch_config.stacking_batch_size,
        }
    }

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
                // Clear all existing image lists to start a new project
                self.images.clear();
                self.aligned_images.clear();
                self.bunch_images.clear();
                self.final_images.clear();
                self.preview_handle = None;
                self.result_mat = None;
                self.crop_rect = None;

                // Clear thumbnail cache
                {
                    let mut cache = self.thumbnail_cache.write().unwrap();
                    cache.clear();
                }

                // Add new images
                self.images.extend(paths.clone());
                self.status = format!("Loaded {} images", self.images.len());

                let cache = self.thumbnail_cache.clone();
                // Generate thumbnails in parallel using rayon
                Task::run(
                    iced::stream::channel(100, move |sender| async move {
                        std::thread::spawn(move || {
                            use rayon::prelude::*;
                            let sender = Arc::new(std::sync::Mutex::new(sender));
                            
                            paths.par_iter().for_each(|path| {
                                if let Ok(handle) = generate_thumbnail(path) {
                                    let mut locked = cache.write().unwrap();
                                    locked.insert(path.clone(), handle.clone());
                                    drop(locked);
                                    if let Ok(mut sender_lock) = sender.lock() {
                                        let _ = sender_lock.try_send(Message::ThumbnailUpdated(path.clone(), handle));
                                    }
                                }
                            });
                        });
                    }),
                    |msg| msg,
                )
            }
            Message::ThumbnailUpdated(path, _handle) => {
                log::trace!("Thumbnail updated for {}", path.display());
                Task::none()
            }
            Message::InternalPathsScanned(aligned, bunches, final_imgs, paths_to_process) => {
                self.aligned_images = aligned;
                self.bunch_images = bunches;
                self.final_images = final_imgs;

                if paths_to_process.is_empty() {
                    return Task::none();
                }

                let cache = self.thumbnail_cache.clone();
                // Generate thumbnails in parallel using rayon
                Task::run(
                    iced::stream::channel(100, move |sender| async move {
                        std::thread::spawn(move || {
                            use rayon::prelude::*;
                            let sender = Arc::new(std::sync::Mutex::new(sender));
                            
                            paths_to_process.par_iter().for_each(|path| {
                                if let Ok(handle) = generate_thumbnail(path) {
                                    let mut locked = cache.write().unwrap();
                                    locked.insert(path.clone(), handle.clone());
                                    drop(locked);
                                    if let Ok(mut sender_lock) = sender.lock() {
                                        let _ = sender_lock.try_send(Message::ThumbnailUpdated(path.clone(), handle));
                                    }
                                }
                            });
                        });
                    }),
                    |msg| msg,
                )
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
                            .map(|entries| {
                                entries.flatten().any(|e| {
                                    let p = e.path();
                                    p.is_file()
                                        && p.extension()
                                            .and_then(|ext| ext.to_str())
                                            .map(|ext| {
                                                ["jpg", "jpeg", "png", "tif", "tiff"]
                                                    .contains(&ext.to_lowercase().as_str())
                                            })
                                            .unwrap_or(false)
                                })
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
                        self.status = "Aligning images...".to_string();
                        self.progress_message = "Starting alignment...".to_string();
                        self.progress_value = 0.0;
                        let images_paths = self.images.clone();
                        let proc_config = self.create_processing_config();
                        self.is_processing = true;  // Mark as processing

                        Task::run(
                            iced::stream::channel(100, move |mut sender| async move {
                                // Run processing in a separate thread
                                std::thread::spawn(move || {
                                    let progress_cb = std::sync::Arc::new(std::sync::Mutex::new(
                                        {
                                            let mut sender_clone = sender.clone();
                                            move |msg: String, pct: f32| {
                                                let _ = sender_clone.try_send(Message::ProgressUpdate(msg, pct));
                                            }
                                        }
                                    ));

                                    let result = processing::align_images(
                                        &images_paths, 
                                        &output_dir,
                                        &proc_config,
                                        Some(progress_cb),
                                    );
                                    
                                    // Send final result
                                    match result {
                                        Ok(rect) => {
                                            let _ = sender.try_send(Message::AlignmentDone(Ok(rect)));
                                        }
                                        Err(e) => {
                                            let _ = sender.try_send(Message::AlignmentDone(Err(e.to_string())));
                                        }
                                    }
                                });
                            }),
                            |msg| msg,
                        )
                    }
                } else {
                    Task::none()
                }
            }
            Message::AlignmentDone(result) => {
                self.is_processing = false;  // Processing complete
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
                // Determine which images to stack and get output directory
                let (images_to_stack, output_dir) = if !self.aligned_images.is_empty() {
                    // Use aligned images if available
                    let first_path = &self.aligned_images[0];
                    // Go up one level from aligned directory to the base directory
                    let out_dir = first_path
                        .parent()  // aligned/
                        .and_then(|p| p.parent())  // base/
                        .unwrap_or(std::path::Path::new("."))
                        .to_path_buf();
                    (self.aligned_images.clone(), out_dir)
                } else if !self.images.is_empty() {
                    // Fall back to original images
                    let first_path = &self.images[0];
                    let out_dir = first_path
                        .parent()
                        .unwrap_or(std::path::Path::new("."))
                        .to_path_buf();
                    (self.images.clone(), out_dir)
                } else {
                    // No images at all
                    self.status = "No images loaded".to_string();
                    return Task::none();
                };

                self.status = "Stacking images...".to_string();
                self.progress_message = "Starting stacking...".to_string();
                self.progress_value = 0.0;
                
                let crop_rect = self.crop_rect;
                let proc_config = self.create_processing_config();
                self.is_processing = true;  // Mark as processing
                
                Task::run(
                    iced::stream::channel(100, move |mut sender| async move {
                        // Run processing in a separate thread
                        std::thread::spawn(move || {
                            let progress_cb = std::sync::Arc::new(std::sync::Mutex::new(
                                {
                                    let mut sender_clone = sender.clone();
                                    move |msg: String, pct: f32| {
                                        let _ = sender_clone.try_send(Message::ProgressUpdate(msg, pct));
                                    }
                                }
                            ));

                            let result = processing::stack_images(
                                &images_to_stack, 
                                &output_dir, 
                                crop_rect,
                                &proc_config,
                                Some(progress_cb),
                            );
                            
                            // Send final result
                            match result {
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
                                        let _ = sender.try_send(Message::StackingDone(Ok((buf.to_vec(), res))));
                                    } else {
                                        let _ = sender.try_send(Message::StackingDone(Err(
                                            "Failed to encode image".to_string()
                                        )));
                                    }
                                }
                                Err(e) => {
                                    let _ = sender.try_send(Message::StackingDone(Err(e.to_string())));
                                }
                            }
                        });
                    }),
                    |msg| msg,
                )
            }
            Message::StackingDone(result) => {
                self.is_processing = false;  // Processing complete
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

                            let cache_locked = cache.read().unwrap();
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
            Message::AutoRefreshTick => {
                // Auto-refresh file lists when processing is active
                if self.is_processing {
                    Task::done(Message::RefreshPanes)
                } else {
                    Task::none()
                }
            }
            Message::ToggleSettings => {
                self.show_settings = !self.show_settings;
                Task::none()
            }
            Message::SharpnessThresholdChanged(value) => {
                self.config.sharpness_threshold = value;
                Task::none()
            }
            Message::UseAdaptiveBatchSizes(enabled) => {
                self.config.use_adaptive_batches = enabled;
                if enabled {
                    // Recalculate batch sizes based on system RAM
                    let available_gb = system_info::get_available_memory_gb();
                    // Estimate average image size (assume 24MP RGB image ~72MB)
                    let avg_size_mb = 72.0;
                    self.config.batch_config = 
                        system_info::BatchSizeConfig::calculate_optimal(available_gb, avg_size_mb);
                }
                Task::none()
            }
            Message::UseCLAHE(enabled) => {
                self.config.use_clahe = enabled;
                Task::none()
            }
            Message::FeatureDetectorChanged(detector) => {
                self.config.feature_detector = detector;
                Task::none()
            }
            Message::ProgressUpdate(msg, value) => {
                self.progress_message = msg;
                self.progress_value = value;
                Task::none()
            }
            Message::Exit => {
                std::process::exit(0);
            }
            Message::None => Task::none(),
        };
        task
    }
    
    pub fn subscription(&self) -> iced::Subscription<Message> {
        // Only refresh when processing is active
        if self.is_processing {
            iced::time::every(Duration::from_secs(2)).map(|_| Message::AutoRefreshTick)
        } else {
            iced::Subscription::none()
        }
    }

    pub fn view(&self) -> Element<'_, Message> {
        let buttons = row![
            button("Add Images").on_press(Message::AddImages),
            button("Add Folder").on_press(Message::AddFolder),
            button("Align").on_press(Message::AlignImages),
            button("Stack").on_press(Message::StackImages),
            button("Save").on_press(Message::SaveImage),
            button(if self.show_settings { "Hide Settings" } else { "Settings" })
                .on_press(Message::ToggleSettings),
            button("Exit").on_press(Message::Exit),
        ]
        .spacing(10)
        .padding(10);

        let mut main_column = column![buttons];

        // Settings panel
        if self.show_settings {
            let settings_panel = self.render_settings_panel();
            main_column = main_column.push(settings_panel);
        }

        // Progress bar (only visible when processing)
        if self.is_processing {
            let progress_bar = container(
                column![
                    text(if self.progress_message.is_empty() {
                        "Processing..."
                    } else {
                        &self.progress_message
                    }).size(14),
                    iced::widget::progress_bar(0.0..=100.0, self.progress_value)
                        .width(Length::Fill)
                ]
                .spacing(5)
            )
            .padding(10)
            .width(Length::Fill)
            .style(|_| container::Style::default()
                .background(iced::Color::from_rgb(0.15, 0.25, 0.35)));
            
            main_column = main_column.push(progress_bar);
        }

        let panes = row![
            self.render_pane("Imported", &self.images),
            self.render_pane("Aligned", &self.aligned_images),
            self.render_pane("Bunches", &self.bunch_images),
            self.render_pane("Final", &self.final_images),
        ]
        .spacing(10)
        .padding(10)
        .height(Length::Fill);

        main_column = main_column
            .push(panes)
            .push(
                container(text_input("", &self.status).size(14))
                    .padding(5)
                    .width(Length::Fill)
                    .style(|_| container::Style::default()
                        .background(iced::Color::from_rgb(0.1, 0.1, 0.1)))
            );

        main_column.into()
    }

    fn render_settings_panel(&self) -> Element<'_, Message> {
        let sharpness_slider = row![
            text("Blur Threshold:"),
            slider(10.0..=100.0, self.config.sharpness_threshold, Message::SharpnessThresholdChanged)
                .width(200),
            text(format!("{:.1}", self.config.sharpness_threshold)),
        ]
        .spacing(10)
        .align_y(iced::Alignment::Center);

        let adaptive_batch_checkbox = checkbox(
            "Auto-adjust batch sizes (RAM-based)",
            self.config.use_adaptive_batches
        )
        .on_toggle(Message::UseAdaptiveBatchSizes);

        let clahe_checkbox = checkbox(
            "Use CLAHE (enhances dark images)",
            self.config.use_clahe
        )
        .on_toggle(Message::UseCLAHE);

        let feature_detector_label = text("Feature Detector:");
        
        let orb_button = button(
            text(if self.config.feature_detector == FeatureDetector::ORB { 
                "● ORB (Fast)" 
            } else { 
                "○ ORB (Fast)" 
            })
        )
        .on_press(Message::FeatureDetectorChanged(FeatureDetector::ORB));

        let sift_button = button(
            text(if self.config.feature_detector == FeatureDetector::SIFT { 
                "● SIFT (Best)" 
            } else { 
                "○ SIFT (Best)" 
            })
        )
        .on_press(Message::FeatureDetectorChanged(FeatureDetector::SIFT));

        let akaze_button = button(
            text(if self.config.feature_detector == FeatureDetector::AKAZE { 
                "● AKAZE (Balanced)" 
            } else { 
                "○ AKAZE (Balanced)" 
            })
        )
        .on_press(Message::FeatureDetectorChanged(FeatureDetector::AKAZE));

        let feature_row = row![
            feature_detector_label,
            orb_button,
            sift_button,
            akaze_button,
        ]
        .spacing(10);

        let batch_info = if self.config.use_adaptive_batches {
            text(format!(
                "Batch sizes: Sharp={}, Features={}, Warp={}, Stack={}",
                self.config.batch_config.sharpness_batch_size,
                self.config.batch_config.feature_batch_size,
                self.config.batch_config.warp_batch_size,
                self.config.batch_config.stacking_batch_size,
            )).size(12)
        } else {
            text("Using default batch sizes").size(12)
        };

        container(
            column![
                text("Processing Settings").size(18),
                sharpness_slider,
                adaptive_batch_checkbox,
                clahe_checkbox,
                feature_row,
                batch_info,
            ]
            .spacing(10)
        )
        .padding(15)
        .width(Length::Fill)
        .style(|_| container::Style::default()
            .background(iced::Color::from_rgb(0.2, 0.2, 0.25)))
        .into()
    }

    fn render_pane<'a>(&self, title: &'a str, images: &'a [PathBuf]) -> Element<'a, Message> {
        let cache = self.thumbnail_cache.read().unwrap();
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
