//! Settings panel rendering
//!
//! This module contains the settings panel UI with all configuration options.

use iced::widget::{
    button, checkbox, column, container, horizontal_rule, horizontal_space, mouse_area, pick_list, row, scrollable, slider, text, text_input,
};
use iced::{Element, Length};
use iced::mouse::ScrollDelta;

use crate::config::{EccMotionType, FeatureDetector};
use crate::messages::Message;
use crate::gui::theme;
use super::super::state::ImageStacker;

/// Helper function to create a slider with mouse wheel support for fine-grained adjustments
fn scrollable_slider<'a>(
    range: std::ops::RangeInclusive<f32>,
    value: f32,
    on_change: impl Fn(f32) -> Message + 'a + Clone,
    step: f32,
    width: impl Into<Length>,
) -> Element<'a, Message> {
    let min = *range.start();
    let max = *range.end();
    let on_change_clone = on_change.clone();
    
    mouse_area(
        slider(range, value, on_change)
            .step(step)
            .width(width)
    )
    .on_scroll(move |delta| {
        let delta_y = match delta {
            ScrollDelta::Lines { y, .. } => y * step,
            ScrollDelta::Pixels { y, .. } => y * step * 0.1,
        };
        let new_value = (value + delta_y).clamp(min, max);
        on_change_clone(new_value)
    })
    .into()
}

impl ImageStacker {
    pub(crate) fn render_settings_panel(&self) -> Element<'_, Message> {
        // ============== PANE 1: ALIGNMENT & DETECTION ==============
        // Compact widths for horizontal 4-pane layout
        let (label_width, slider_width, value_width) = (120, 150, 60);
        
        let sharpness_slider = row![
            text("Blur Threshold:").width(label_width),
            scrollable_slider(10.0..=10000.0, self.config.sharpness_threshold, Message::SharpnessThresholdChanged, 10.0, slider_width),
            text(format!("{:.0}", self.config.sharpness_threshold)).width(value_width),
        ]
        .spacing(10)
        .align_y(iced::Alignment::Center);

        let grid_size_slider = row![
            text("Sharpness Grid:").width(label_width),
            scrollable_slider(4.0..=16.0, self.config.sharpness_grid_size as f32, Message::SharpnessGridSizeChanged, 1.0, slider_width),
            text(format!("{}x{}", self.config.sharpness_grid_size, self.config.sharpness_grid_size)).width(value_width),
        ]
        .spacing(10)
        .align_y(iced::Alignment::Center);

        let iqr_multiplier_slider = row![
            text("Blur Filter (IQR):").width(label_width),
            scrollable_slider(0.5..=5.0, self.config.sharpness_iqr_multiplier, Message::SharpnessIqrMultiplierChanged, 0.1, slider_width),
            text(format!("{:.1} {}", 
                self.config.sharpness_iqr_multiplier,
                if self.config.sharpness_iqr_multiplier <= 1.0 { "(strict)" }
                else if self.config.sharpness_iqr_multiplier <= 2.0 { "(normal)" }
                else if self.config.sharpness_iqr_multiplier <= 3.0 { "(relaxed)" }
                else { "(very permissive)" }
            )).width(value_width),
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

        let orb_selected = self.config.feature_detector == FeatureDetector::ORB;
        let orb_button = button(
            text(if orb_selected { 
                "✓ ORB (Fast)" 
            } else { 
                "  ORB (Fast)" 
            })
        )
        .style(theme::ecc_toggle_button(orb_selected))
        .on_press(Message::FeatureDetectorChanged(FeatureDetector::ORB));

        let akaze_selected = self.config.feature_detector == FeatureDetector::AKAZE;
        let akaze_button = button(
            text(if akaze_selected { 
                "✓ AKAZE (Balanced)" 
            } else { 
                "  AKAZE (Balanced)" 
            })
        )
        .style(theme::ecc_toggle_button(akaze_selected))
        .on_press(Message::FeatureDetectorChanged(FeatureDetector::AKAZE));

        let sift_selected = self.config.feature_detector == FeatureDetector::SIFT;
        let sift_button = button(
            text(if sift_selected { 
                "✓ SIFT (Best)" 
            } else { 
                "  SIFT (Best)" 
            })    
        )    
        .style(theme::ecc_toggle_button(sift_selected))
        .on_press(Message::FeatureDetectorChanged(FeatureDetector::SIFT));

        let ecc_selected = self.config.feature_detector == FeatureDetector::ECC;
        let ecc_button = button(
            text(if ecc_selected {
                "✓ ECC (ULTRA sub-pixel-precise)"
            } else {
                "  ECC (ULTRA sub-pixel-precise)"
            })
        )
        .style(theme::ecc_toggle_button(ecc_selected))
        .on_press(Message::FeatureDetectorChanged(FeatureDetector::ECC));

        // Feature detector layout: stack buttons vertically for compact horizontal panes
        let feature_layout: Element<'_, Message> = column![
                text("Feature Detector:").size(16),
                orb_button.width(Length::Fixed(300.0)),
                akaze_button.width(Length::Fixed(300.0)),
                sift_button.width(Length::Fixed(300.0)),
                ecc_button.width(Length::Fixed(300.0)),
            ]
            .spacing(5)
            .into();

        // ECC-specific parameters — built only when ECC is selected, displayed in a separate pane
        let ecc_params_ui: Option<Element<'_, Message>> = if self.config.feature_detector == FeatureDetector::ECC {
            // Motion Type buttons
            let translation_selected = self.config.ecc_motion_type == EccMotionType::Translation;
            let euclidean_selected = self.config.ecc_motion_type == EccMotionType::Euclidean;
            let affine_selected = self.config.ecc_motion_type == EccMotionType::Affine;
            let homography_selected = self.config.ecc_motion_type == EccMotionType::Homography;
            
            // Parameters sliders
            let iterations_slider = column![
                text(format!("Max Iterations: {}", self.config.ecc_max_iterations)).size(12),
                scrollable_slider(3000.0..=30000.0, self.config.ecc_max_iterations as f32, Message::EccMaxIterationsChanged, 1000.0, Length::Fill)
            ].spacing(3);

            // Epsilon uses logarithmic scale: slider shows exponent, actual value is 10^x
            let epsilon_exponent = self.config.ecc_epsilon.log10();
            let epsilon_slider = column![
                text(format!("Epsilon: {:.1e} (convergence threshold)", self.config.ecc_epsilon)).size(12),
                scrollable_slider(-8.0..=-4.0, epsilon_exponent as f32, Message::EccEpsilonChanged, 0.5, Length::Fill)
            ].spacing(3);

            let filter_slider = column![
                text(format!("Gaussian Filter Size: {}x{}", self.config.ecc_gauss_filter_size, self.config.ecc_gauss_filter_size)).size(12),
                scrollable_slider(3.0..=15.0, self.config.ecc_gauss_filter_size as f32, Message::EccGaussFilterSizeChanged, 2.0, Length::Fill)  // Ensures odd values
            ].spacing(3);

            let chunk_slider = column![
                text(format!("Parallel Chunk Size: {} images", self.config.ecc_chunk_size)).size(12),
                scrollable_slider(4.0..=24.0, self.config.ecc_chunk_size as f32, Message::EccChunkSizeChanged, 2.0, Length::Fill)
            ].spacing(3);

            let batch_slider = column![
                text(format!("Batch Size: {} images (controls memory usage)", self.config.ecc_batch_size)).size(12),
                scrollable_slider(2.0..=16.0, self.config.ecc_batch_size as f32, Message::EccBatchSizeChanged, 1.0, Length::Fill),
                text("Lower = less memory, slower | Higher = more memory, faster").size(10)
                    .style(|t| theme::muted_text(t))
            ].spacing(3);

            let timeout_slider = column![
                text(format!("ECC Timeout: {}s per image", self.config.ecc_timeout_seconds)).size(12),
                scrollable_slider(10.0..=300.0, self.config.ecc_timeout_seconds as f32, Message::EccTimeoutChanged, 5.0, Length::Fill),
                text("Prevents hangs on difficult images. Falls back to feature-based alignment on timeout.").size(10)
                    .style(|t| theme::muted_text(t))
            ].spacing(3);

            let hybrid_checkbox = column![
                checkbox("Use Hybrid Mode (40-50% faster)", self.config.ecc_use_hybrid)
                    .on_toggle(Message::EccUseHybridChanged),
                text("SIFT initialization + ECC refinement for speed with quality").size(10)
                    .style(|t| theme::muted_text(t))
            ].spacing(3);

            Some(column![
                text("ECC Parameters").size(16).style(|t| theme::heading_text(t)),
                text("Motion Type:").size(12),
                row![
                    button(text("Translation (2-DOF)").size(12))
                        .style(theme::ecc_toggle_button(translation_selected))
                        .on_press(Message::EccMotionTypeChanged(EccMotionType::Translation)),
                    button(text("Euclidean (3-DOF)").size(12))
                        .style(theme::ecc_toggle_button(euclidean_selected))
                        .on_press(Message::EccMotionTypeChanged(EccMotionType::Euclidean)),
                ].spacing(5),
                row![
                    button(text("Affine (6-DOF)").size(12))
                        .style(theme::ecc_toggle_button(affine_selected))
                        .on_press(Message::EccMotionTypeChanged(EccMotionType::Affine)),
                    button(text("Homography (8-DOF)").size(12))
                        .style(theme::ecc_toggle_button(homography_selected))
                        .on_press(Message::EccMotionTypeChanged(EccMotionType::Homography)),
                ].spacing(5),
                iterations_slider,
                epsilon_slider,
                filter_slider,
                chunk_slider,
                batch_slider,
                timeout_slider,
                hybrid_checkbox,
            ]
            .spacing(8)
            .into())
        } else {
            None
        };

        // Transform validation sliders - shown for ALL alignment methods
        let transform_validation_ui = column![
            horizontal_rule(1),
            text("Transform Validation (All Alignment Methods):").size(13).style(|t| theme::section_label(t)),
            text("Reject distorted transformations - applies to ORB, SIFT, AKAZE, and ECC").size(10)
                .style(|t| theme::muted_text(t)),
            column![
                text(format!("Max Transform Scale: {:.2}x", self.config.max_transform_scale)).size(12),
                scrollable_slider(1.1..=3.0, self.config.max_transform_scale, Message::MaxTransformScaleChanged, 0.1, Length::Fill),
                text("Maximum allowed scale deviation (reject if exceeded)").size(10)
                    .style(|t| theme::muted_text(t))
            ].spacing(3),
            column![
                text(format!("Max Translation: {:.0} pixels", self.config.max_transform_translation)).size(12),
                scrollable_slider(100.0..=1000.0, self.config.max_transform_translation, Message::MaxTransformTranslationChanged, 50.0, Length::Fill),
                text("Maximum allowed translation distance (reject if exceeded)").size(10)
                    .style(|t| theme::muted_text(t))
            ].spacing(3),
            column![
                text(format!("Max Determinant: {:.1}x", self.config.max_transform_determinant)).size(12),
                scrollable_slider(1.5..=5.0, self.config.max_transform_determinant, Message::MaxTransformDeterminantChanged, 0.5, Length::Fill),
                text("Maximum determinant deviation (skew/distortion threshold)").size(10)
                    .style(|t| theme::muted_text(t))
            ].spacing(3),
        ]
        .spacing(8)
        .padding(10);

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

        let alignment_pane = container(
            column![
                text("Alignment & Detection").size(16).style(|t| theme::heading_text(t)),
                sharpness_slider,
                grid_size_slider,
                iqr_multiplier_slider,
                adaptive_batch_checkbox,
                batch_info,
                clahe_checkbox,
                feature_layout,
                transform_validation_ui,
            ]
            .spacing(10)
        )
        .padding(10)
        .style(|t| theme::settings_section(t))
        .width(Length::Fill);

        // ============== PANE 2: POST-PROCESSING ==============
        let noise_section = column![
            checkbox("Enable Noise Reduction", self.config.enable_noise_reduction)
                .on_toggle(Message::EnableNoiseReduction),
            
            row![
                text("Noise Strength:").width(label_width),
                scrollable_slider(1.0..=10.0, self.config.noise_reduction_strength, Message::NoiseReductionStrengthChanged, 0.1, slider_width),
                text(format!("{:.1}", self.config.noise_reduction_strength)).width(value_width),
            ]
            .spacing(10)
            .align_y(iced::Alignment::Center),
        ]
        .spacing(5);

        let sharpen_section = column![
            checkbox("Enable Sharpening", self.config.enable_sharpening)
                .on_toggle(Message::EnableSharpening),
            
            row![
                text("Sharpen Strength:").width(label_width),
                scrollable_slider(0.0..=5.0, self.config.sharpening_strength, Message::SharpeningStrengthChanged, 0.1, slider_width),
                text(format!("{:.1}", self.config.sharpening_strength)).width(value_width),
            ]
            .spacing(10)
            .align_y(iced::Alignment::Center),
        ]
        .spacing(5);

        let color_section = column![
            checkbox("Enable Color Correction", self.config.enable_color_correction)
                .on_toggle(Message::EnableColorCorrection),
            
            row![
                text("Contrast:").width(label_width),
                scrollable_slider(0.5..=3.0, self.config.contrast_boost, Message::ContrastBoostChanged, 0.1, slider_width),
                text(format!("{:.1}", self.config.contrast_boost)).width(value_width),
            ]
            .spacing(10)
            .align_y(iced::Alignment::Center),
            
            row![
                text("Brightness:").width(label_width),
                scrollable_slider(-100.0..=100.0, self.config.brightness_boost, Message::BrightnessBoostChanged, 1.0, slider_width),
                text(format!("{:.0}", self.config.brightness_boost)).width(value_width),
            ]
            .spacing(10)
            .align_y(iced::Alignment::Center),
            
            row![
                text("Saturation:").width(label_width),
                scrollable_slider(0.0..=3.0, self.config.saturation_boost, Message::SaturationBoostChanged, 0.1, slider_width),
                text(format!("{:.1}", self.config.saturation_boost)).width(value_width),
            ]
            .spacing(10)
            .align_y(iced::Alignment::Center),
        ]
        .spacing(5);

        let postprocessing_pane = container(
            column![
                text("Post-Processing").size(16).style(|t| theme::heading_text(t)),
                noise_section,
                sharpen_section,
                color_section,
            ]
            .spacing(10)
        )
        .padding(10)
        .style(|t| theme::settings_section(t))
        .width(Length::Fill);

        // ============== PANE 3: PREVIEW & UI ==============
        let preview_pane = container(
            column![
                text("Preview & UI").size(16).style(|t| theme::heading_text(t)),
                
                checkbox("Use Internal Preview (modal overlay)", self.config.use_internal_preview)
                    .on_toggle(Message::UseInternalPreview),
                
                text("When disabled, left-click opens in external viewer (configurable below)").size(10)
                    .style(|t| theme::muted_text(t)),
                
                row![
                    text("Preview Max Width:").width(label_width),
                    scrollable_slider(400.0..=2000.0, self.config.preview_max_width, Message::PreviewMaxWidthChanged, 10.0, slider_width),
                    text(format!("{:.0}px", self.config.preview_max_width)).width(value_width),
                ]
                .spacing(10)
                .align_y(iced::Alignment::Center),
                
                row![
                    text("Preview Max Height:").width(label_width),
                    scrollable_slider(300.0..=1500.0, self.config.preview_max_height, Message::PreviewMaxHeightChanged, 10.0, slider_width),
                    text(format!("{:.0}px", self.config.preview_max_height)).width(value_width),
                ]
                .spacing(10)
                .align_y(iced::Alignment::Center),
                
                container(column![]).height(10),  // Spacer
                
                text("External Image Viewer (for left-click when internal preview disabled)").size(12)
                    .style(|t| theme::section_label(t)),
                
                text_input(
                    "Path to viewer (e.g., /usr/bin/eog, /usr/bin/geeqie)...",
                    &self.config.external_viewer_path
                )
                    .on_input(Message::ExternalViewerPathChanged)
                    .padding(5),
                
                text("Leave empty to use system default viewer. Used when 'Use Internal Preview' is disabled.")
                    .size(10)
                    .style(|t| theme::muted_text(t)),
                
                container(column![]).height(10),  // Spacer
                
                text("External Image Editor (for right-click)").size(12)
                    .style(|t| theme::section_label(t)),
                
                text_input(
                    "Path to editor (e.g., /usr/bin/gimp, darktable, photoshop)...",
                    &self.config.external_editor_path
                )
                    .on_input(Message::ExternalEditorPathChanged)
                    .padding(5),
                
                text("Leave empty to use system default viewer. Right-click any thumbnail to open with editor.")
                    .size(10)
                    .style(|t| theme::muted_text(t)),
                
                container(column![]).height(10),  // Spacer
                
                text("Application Font").size(12)
                    .style(|t| theme::section_label(t)),
                
                pick_list(
                    self.available_fonts.as_slice(),
                    Some(self.config.default_font.clone()),
                    Message::DefaultFontChanged,
                )
                .width(Length::Fill)
                .padding(5),
                
                text("⚠ Font changes take effect on next app restart.")
                    .size(10)
                    .style(|t| theme::muted_text(t)),
            ]
            .spacing(10)
        )
        .padding(10)
        .style(|t| theme::settings_section(t))
        .width(Length::Fill);

        // ============== PANE 4: GPU / PERFORMANCE ==============
        let gpu_concurrency_desc = match self.config.gpu_concurrency {
            0 => "unlimited (fastest, may OOM on large images)",
            1 => "serialized (safest, slowest)",
            2 => "2 concurrent (default, good balance)",
            n => if n <= 4 { "moderate parallelism" } else { "high parallelism (needs lots of VRAM)" },
        };

        let gpu_pane = container(
            column![
                text("GPU / Performance").size(16).style(|t| theme::heading_text(t)),

                text("GPU Concurrency").size(13).style(|t| theme::section_label(t)),
                text("Max simultaneous GPU operations. Lower = less VRAM usage, higher = faster.").size(10)
                    .style(|t| theme::muted_text(t)),
                row![
                    text("GPU Concurrency:").width(label_width),
                    scrollable_slider(0.0..=8.0, self.config.gpu_concurrency as f32, Message::GpuConcurrencyChanged, 1.0, slider_width),
                    text(format!("{}", self.config.gpu_concurrency)).width(value_width),
                ]
                .spacing(10)
                .align_y(iced::Alignment::Center),
                text(gpu_concurrency_desc).size(11)
                    .style(|t| theme::success_text(t)),

                container(column![]).height(10),  // Spacer
                
                text("⚠ Changes take effect on next app restart").size(11)
                    .style(|t| theme::warning_text(t)),
                
                horizontal_rule(1),

                // Stacking Bunch Size
                text("Stacking Bunch Size").size(13).style(|t| theme::section_label(t)),
                text("Number of images per bunch during recursive stacking. More images per bunch = fewer levels, but higher memory usage.").size(10)
                    .style(|t| theme::muted_text(t)),
                checkbox(
                    format!("Auto (RAM-based: {})", self.config.batch_config.stacking_batch_size),
                    self.config.auto_bunch_size,
                )
                .on_toggle(Message::AutoBunchSizeChanged),
                if !self.config.auto_bunch_size {
                    column![
                        row![
                            text("Bunch Size:").width(label_width),
                            scrollable_slider(4.0..=64.0, self.config.stacking_bunch_size as f32, Message::BunchSizeChanged, 1.0, slider_width),
                            text(format!("{}", self.config.stacking_bunch_size)).width(value_width),
                        ]
                        .spacing(10)
                        .align_y(iced::Alignment::Center),
                        text(format!("Active: {} images per bunch", self.config.stacking_bunch_size)).size(11)
                            .style(|t| theme::success_text(t)),
                    ].spacing(4)
                } else {
                    column![
                        text(format!("Active: {} images per bunch (auto)", self.config.batch_config.stacking_batch_size)).size(11)
                            .style(|t| theme::success_text(t)),
                    ]
                },

                horizontal_rule(1),
                
                text("Environment Variable Overrides").size(13).style(|t| theme::section_label(t)),
                text("Env vars override settings (for testing/debugging):").size(10)
                    .style(|t| theme::muted_text(t)),
                
                column![
                    text("IMAGESTACKER_GPU_CONCURRENCY=N").size(11)
                        .style(|t| theme::env_var_text(t)),
                    text("  Override GPU concurrency (0=unlimited, 1=serialized)").size(10)
                        .style(|t| theme::muted_text(t)),
                ].spacing(2),
                
                column![
                    text("IMAGESTACKER_OPENCL_MUTEX=1").size(11)
                        .style(|t| theme::env_var_text(t)),
                    text("  Force GPU concurrency to 1 (fully serialized)").size(10)
                        .style(|t| theme::muted_text(t)),
                ].spacing(2),
                
                column![
                    text("IMAGESTACKER_ECC_BATCH_SIZE=N").size(11)
                        .style(|t| theme::env_var_text(t)),
                    text("  Override ECC batch size (overrides 'Batch Size' setting)").size(10)
                        .style(|t| theme::muted_text(t)),
                ].spacing(2),
            ]
            .spacing(8)
        )
        .padding(10)
        .style(|t| theme::settings_section(t))
        .width(Length::Fill);

        // ============== RESET BUTTON ==============
        let reset_button = button(
            text("↺ Reset to Defaults")
        )
        .style(theme::reset_button)
        .on_press(Message::ResetToDefaults);

        // ============== RESPONSIVE LAYOUT ==============
        // Build the panes layout
        let panes_layout: Element<'_, Message> = if let Some(ecc_content) = ecc_params_ui {
            // ECC pane as a separate styled section below Post-Processing
            let ecc_pane = container(ecc_content)
                .padding(10)
                .style(|t| theme::settings_section(t))
                .width(Length::Fill);

            // Stack Post-Processing and ECC vertically in column 2
            let col2 = column![
                postprocessing_pane,
                ecc_pane,
            ]
            .spacing(10);

            row![
                container(alignment_pane).width(Length::FillPortion(1)),
                container(col2).width(Length::FillPortion(1)),
                container(preview_pane).width(Length::FillPortion(1)),
                container(gpu_pane).width(Length::FillPortion(1)),
            ]
            .spacing(10)
            .align_y(iced::Alignment::Start)
            .into()
        } else {
            // No ECC — simple single row
            row![
                container(alignment_pane).width(Length::FillPortion(1)),
                container(postprocessing_pane).width(Length::FillPortion(1)),
                container(preview_pane).width(Length::FillPortion(1)),
                container(gpu_pane).width(Length::FillPortion(1)),
            ]
            .spacing(10)
            .align_y(iced::Alignment::Start)
            .into()
        };

        // Title row with heading on the left and reset button on the right
        let title_row = row![
            text("Processing Settings").size(20).style(|t| theme::heading_text(t)),
            horizontal_space(),
            reset_button,
        ]
        .spacing(10)
        .align_y(iced::Alignment::Center);

        // Use up to 4/5 of the window height for settings.
        // The scrollable activates only when content exceeds this limit.
        let max_settings_height = (self.window_height * 4.0 / 5.0) as f32;

        container(
            scrollable(
                column![
                    title_row,
                    panes_layout,
                ]
                .spacing(10)
                .width(Length::Fill)
            )
        )
        .padding(10)
        .width(Length::Fill)
        .max_height(max_settings_height)
        .style(|t| theme::settings_panel(t))
        .into()
    }
}
