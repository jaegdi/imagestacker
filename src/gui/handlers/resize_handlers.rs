//! Image resize handlers
//!
//! Resize imported images into a `resized/` subfolder while preserving
//! the aspect ratio. The target width is specified either as an absolute
//! pixel value ("2400") or as a percentage of the original width ("50%").
//! Values <= 0 are ignored.
//!
//! Once resized images exist, `Detect Sharpness` and `Align` operations
//! use them instead of the original high-resolution images.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::Ordering;

use iced::Task;
use rayon::prelude::*;

use crate::config::ResizeAlgorithm;
use crate::image_io::{is_any_image_ext, is_raw_ext, load_image};
use crate::messages::Message;
use crate::gui::state::ImageStacker;

// ---------------------------------------------------------------------------
// Public helper – also used in sharpness/alignment handlers
// ---------------------------------------------------------------------------

/// Parse a resize-width string to the target pixel width.
///
/// - `"2400"` → 2400 px absolute
/// - `"50%"`  → 50 % of `original_width`
/// - empty / non-positive → `None` (disabled)
pub fn parse_resize_target(input: &str, original_width: u32) -> Option<u32> {
    let s = input.trim();
    if s.is_empty() {
        return None;
    }
    if let Some(pct_str) = s.strip_suffix('%') {
        let pct: f64 = pct_str.trim().parse().ok()?;
        if pct <= 0.0 {
            return None;
        }
        let w = ((original_width as f64) * pct / 100.0).round() as u32;
        if w == 0 {
            return None;
        }
        Some(w)
    } else {
        let w: i64 = s.parse().ok()?;
        if w <= 0 {
            return None;
        }
        Some(w as u32)
    }
}

// ---------------------------------------------------------------------------
// Directory scanner helper (shared with file_handlers via pub)
// ---------------------------------------------------------------------------

pub fn scan_image_dir(dir: &Path) -> Vec<PathBuf> {
    let mut paths = Vec::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let p = entry.path();
            if p.is_file() {
                if let Some(ext) = p.extension().and_then(|e| e.to_str()) {
                    if is_any_image_ext(ext) {
                        paths.push(p);
                    }
                }
            }
        }
    }
    paths.sort();
    paths
}

// ---------------------------------------------------------------------------
// ImageStacker handlers
// ---------------------------------------------------------------------------

impl ImageStacker {
    fn resized_output_path(output_dir: &Path, image_path: &Path) -> Result<PathBuf, String> {
        if let Some(ext) = image_path.extension().and_then(|e| e.to_str()) {
            if is_raw_ext(ext) {
                let stem = image_path
                    .file_stem()
                    .ok_or_else(|| format!("No filename stem for {}", image_path.display()))?;
                return Ok(output_dir.join(format!("{}.png", stem.to_string_lossy())));
            }
        }

        let out_name = image_path
            .file_name()
            .ok_or_else(|| format!("No filename for {}", image_path.display()))?;
        Ok(output_dir.join(out_name))
    }

    /// Returns `true` when `resize_width` contains a syntactically valid
    /// positive value (pixel count or percentage).
    pub fn has_valid_resize_config(&self) -> bool {
        let s = self.config.resize_width.trim();
        if s.is_empty() {
            return false;
        }
        if let Some(pct_str) = s.strip_suffix('%') {
            if let Ok(pct) = pct_str.trim().parse::<f64>() {
                return pct > 0.0;
            }
        } else if let Ok(w) = s.parse::<i64>() {
            return w > 0;
        }
        false
    }

    /// Start resizing all imported images into `<source_dir>/resized/`.
    pub fn handle_resize_images(&mut self) -> Task<Message> {
        if self.images.is_empty() {
            self.status = "No images to resize".to_string();
            return Task::none();
        }
        if !self.has_valid_resize_config() {
            self.status = "Invalid resize width – enter pixel count or percentage".to_string();
            return Task::none();
        }
        if self.is_processing {
            self.status = "Already processing…".to_string();
            return Task::none();
        }

        self.is_processing = true;
        self.status = "Resizing images…".to_string();
        self.progress_value = 0.0;
        self.progress_message = "Starting resize…".to_string();

        let images = self.images.clone();
        let resize_width_str = self.config.resize_width.clone();
        let algorithm = self.config.resize_algorithm;
        let cancel_flag = self.cancel_flag.clone();
        cancel_flag.store(false, Ordering::Relaxed);

        let output_dir = images[0]
            .parent()
            .unwrap_or(std::path::Path::new("."))
            .to_path_buf()
            .join("resized");

        Task::run(
            iced::stream::channel(100, move |mut sender| async move {
                std::thread::spawn(move || {
                    use std::sync::atomic::AtomicUsize;

                    if let Err(e) = std::fs::create_dir_all(&output_dir) {
                        let _ = sender.try_send(Message::ResizeDone(Err(
                            format!("Failed to create resized dir: {}", e),
                        )));
                        return;
                    }

                    let interp_flag = match algorithm {
                        ResizeAlgorithm::Fast => opencv::imgproc::INTER_LINEAR,
                        ResizeAlgorithm::HighQuality => opencv::imgproc::INTER_LANCZOS4,
                    };

                    let total = images.len();
                    let completed = Arc::new(AtomicUsize::new(0));
                    let progress_sender = sender.clone();

                    let results: Vec<Result<(), String>> = images
                        .par_iter()
                        .map(|image_path| {
                            if cancel_flag.load(Ordering::Relaxed) {
                                return Err("Cancelled".to_string());
                            }

                            let img = load_image(image_path)
                                .map_err(|e| format!("Load failed {}: {}", image_path.display(), e))?;

                            use opencv::core::MatTraitConst;
                            let orig_w = img.cols() as u32;
                            let orig_h = img.rows() as u32;

                            if orig_w == 0 || orig_h == 0 {
                                return Err(format!("Load failed {}: empty image", image_path.display()));
                            }

                            let target_w =
                                parse_resize_target(&resize_width_str, orig_w)
                                    .ok_or_else(|| "Invalid resize width".to_string())?;

                            let scale = target_w as f64 / orig_w as f64;
                            let target_h = ((orig_h as f64) * scale).round() as i32;

                            let mut resized = opencv::core::Mat::default();
                            opencv::imgproc::resize(
                                &img,
                                &mut resized,
                                opencv::core::Size::new(target_w as i32, target_h),
                                0.0,
                                0.0,
                                interp_flag,
                            )
                            .map_err(|e| format!("Resize failed: {}", e))?;

                            let out_path = Self::resized_output_path(&output_dir, image_path)?;
                            opencv::imgcodecs::imwrite(
                                out_path.to_str().ok_or("Invalid output path")?,
                                &resized,
                                &opencv::core::Vector::new(),
                            )
                            .map_err(|e| format!("Save failed: {}", e))?;

                            let done = completed.fetch_add(1, Ordering::Relaxed) + 1;
                            let pct = (done as f32 / total as f32) * 100.0;
                            let msg = format!("Resize: {}/{}", done, total);
                            let _ = progress_sender
                                .clone()
                                .try_send(Message::ProgressUpdate(msg, pct));

                            Ok(())
                        })
                        .collect();

                    let errors: Vec<String> = results
                        .into_iter()
                        .filter_map(|r| r.err())
                        .filter(|e| e != "Cancelled")
                        .collect();

                    if errors.is_empty() {
                        let _ = sender.try_send(Message::ResizeDone(Ok(total)));
                    } else {
                        let _ = sender.try_send(Message::ResizeDone(Err(format!(
                            "{} error(s): {}",
                            errors.len(),
                            errors[0]
                        ))));
                    }
                });
            }),
            |msg| msg,
        )
    }

    /// Handle the result of the resize operation.
    pub fn handle_resize_done(&mut self, result: Result<usize, String>) -> Task<Message> {
        self.is_processing = false;
        match result {
            Ok(count) => {
                self.status = format!("Resize done: {} images in resized/", count);
                // Update resized_images list
                if let Some(first) = self.images.first() {
                    let resized_dir = first
                        .parent()
                        .unwrap_or(std::path::Path::new("."))
                        .to_path_buf()
                        .join("resized");
                    self.resized_images = scan_image_dir(&resized_dir);
                    log::info!(
                        "Resized images available: {} files in {}",
                        self.resized_images.len(),
                        resized_dir.display()
                    );
                }
            }
            Err(e) => {
                if e.contains("Cancelled") {
                    self.status = "Resize cancelled by user".to_string();
                } else {
                    self.status = format!("Resize failed: {}", e);
                }
            }
        }
        Task::none()
    }

    /// Handle resize width text-input change.
    pub fn handle_resize_width_changed(&mut self, value: String) -> Task<Message> {
        self.config.resize_width = value;
        let _ = crate::settings::save_settings(&self.config);
        Task::none()
    }

    /// Handle resize algorithm selection change.
    pub fn handle_resize_algorithm_changed(&mut self, algo: ResizeAlgorithm) -> Task<Message> {
        self.config.resize_algorithm = algo;
        let _ = crate::settings::save_settings(&self.config);
        Task::none()
    }
}
