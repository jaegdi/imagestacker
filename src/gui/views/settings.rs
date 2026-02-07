//! Settings panel rendering
//!
//! This module contains the settings panel UI with all configuration options.

use iced::widget::{
    button, checkbox, column, container, horizontal_rule, mouse_area, row, scrollable, slider, text, text_input,
};
use iced::{Element, Length};
use iced::mouse::ScrollDelta;

use crate::config::{EccMotionType, FeatureDetector};
use crate::messages::Message;
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
        // Determine layout based on window width
        // Minimum width per pane: 380px, so need at least 1200px for horizontal (3*380 + spacing + padding)
        let use_horizontal_layout = self.window_width >= 1200.0;
        
        // ============== PANE 1: ALIGNMENT & DETECTION ==============
        // Adjust slider widths based on layout to prevent value stacking
        let (label_width, slider_width, value_width) = if use_horizontal_layout {
            (120, 150, 60)  // Narrower for horizontal pane layout
        } else {
            (150, 200, 50)  // Original widths for vertical layout
        };
        
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

        let feature_detector_label = text("Feature Detector:");
        
        let orb_selected = self.config.feature_detector == FeatureDetector::ORB;
        let orb_button = button(
            text(if orb_selected { 
                "✓ ORB (Fast)" 
            } else { 
                "  ORB (Fast)" 
            })
        )
        .style(move |theme, status| {
            if orb_selected {
                button::Style {
                    background: Some(iced::Background::Color(iced::Color::from_rgb(0.3, 0.5, 0.3))),
                    text_color: iced::Color::WHITE,
                    border: iced::Border {
                        color: iced::Color::from_rgb(0.4, 0.7, 0.4),
                        width: 2.0,
                        radius: 4.0.into(),
                    },
                    ..Default::default()
                }
            } else {
                button::secondary(theme, status)
            }
        })
        .on_press(Message::FeatureDetectorChanged(FeatureDetector::ORB));

        let sift_selected = self.config.feature_detector == FeatureDetector::SIFT;
        let sift_button = button(
            text(if sift_selected { 
                "✓ SIFT (Best)" 
            } else { 
                "  SIFT (Best)" 
            })
        )
        .style(move |theme, status| {
            if sift_selected {
                button::Style {
                    background: Some(iced::Background::Color(iced::Color::from_rgb(0.3, 0.5, 0.3))),
                    text_color: iced::Color::WHITE,
                    border: iced::Border {
                        color: iced::Color::from_rgb(0.4, 0.7, 0.4),
                        width: 2.0,
                        radius: 4.0.into(),
                    },
                    ..Default::default()
                }
            } else {
                button::secondary(theme, status)
            }
        })
        .on_press(Message::FeatureDetectorChanged(FeatureDetector::SIFT));

        let akaze_selected = self.config.feature_detector == FeatureDetector::AKAZE;
        let akaze_button = button(
            text(if akaze_selected { 
                "✓ AKAZE (Balanced)" 
            } else { 
                "  AKAZE (Balanced)" 
            })
        )
        .style(move |theme, status| {
            if akaze_selected {
                button::Style {
                    background: Some(iced::Background::Color(iced::Color::from_rgb(0.3, 0.5, 0.3))),
                    text_color: iced::Color::WHITE,
                    border: iced::Border {
                        color: iced::Color::from_rgb(0.4, 0.7, 0.4),
                        width: 2.0,
                        radius: 4.0.into(),
                    },
                    ..Default::default()
                }
            } else {
                button::secondary(theme, status)
            }
        })
        .on_press(Message::FeatureDetectorChanged(FeatureDetector::AKAZE));

        let ecc_button = button(text("ECC").size(14))
        .style(move |theme, status| {
            if self.config.feature_detector == FeatureDetector::ECC {
                button::Style {
                    background: Some(iced::Background::Color(iced::Color::from_rgb(0.3, 0.4, 0.6))),
                    text_color: iced::Color::WHITE,
                    border: iced::Border {
                        color: iced::Color::from_rgb(0.4, 0.5, 0.8),
                        width: 2.0,
                        radius: 4.0.into(),
                    },
                    ..Default::default()
                }
            } else {
                button::secondary(theme, status)
            }
        })
        .on_press(Message::FeatureDetectorChanged(FeatureDetector::ECC));

        // Feature detector layout: vertical when horizontal pane layout, horizontal when vertical pane layout
        let feature_layout: Element<'_, Message> = if use_horizontal_layout {
            // Horizontal layout -> stack detector buttons vertically
            column![
                text("Feature Detector:").size(14),
                orb_button.width(Length::Fixed(160.0)),
                sift_button.width(Length::Fixed(160.0)),
                akaze_button.width(Length::Fixed(160.0)),
                ecc_button.width(Length::Fixed(160.0)),
            ]
            .spacing(5)
            .into()
        } else {
            // Vertical layout -> stack detector buttons horizontally
            row![
                feature_detector_label,
                orb_button,
                sift_button,
                akaze_button,
                ecc_button,
            ]
            .spacing(10)
            .into()
        };

        // ECC-specific parameters (conditionally shown)
        let ecc_params_ui: Element<'_, Message> = if self.config.feature_detector == FeatureDetector::ECC {
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
                    .style(|_| text::Style {
                        color: Some(iced::Color::from_rgb(0.6, 0.6, 0.6))
                    })
            ].spacing(3);

            let timeout_slider = column![
                text(format!("ECC Timeout: {}s per image", self.config.ecc_timeout_seconds)).size(12),
                scrollable_slider(10.0..=300.0, self.config.ecc_timeout_seconds as f32, Message::EccTimeoutChanged, 5.0, Length::Fill),
                text("Prevents hangs on difficult images. Falls back to feature-based alignment on timeout.").size(10)
                    .style(|_| text::Style {
                        color: Some(iced::Color::from_rgb(0.6, 0.6, 0.6))
                    })
            ].spacing(3);

            let hybrid_checkbox = column![
                checkbox("Use Hybrid Mode (40-50% faster)", self.config.ecc_use_hybrid)
                    .on_toggle(Message::EccUseHybridChanged),
                text("SIFT initialization + ECC refinement for speed with quality").size(10)
                    .style(|_| text::Style {
                        color: Some(iced::Color::from_rgb(0.6, 0.6, 0.6))
                    })
            ].spacing(3);

            column![
                text("ECC Parameters:").size(13).style(|_| text::Style {
                    color: Some(iced::Color::from_rgb(0.7, 0.8, 1.0))
                }),
                text("Motion Type:").size(12),
                row![
                    button(text("Translation (2-DOF)").size(12))
                        .style(move |theme, status| {
                            if translation_selected {
                                button::Style {
                                    background: Some(iced::Background::Color(iced::Color::from_rgb(0.3, 0.5, 0.6))),
                                    text_color: iced::Color::WHITE,
                                    border: iced::Border {
                                        color: iced::Color::from_rgb(0.4, 0.7, 0.8),
                                        width: 2.0,
                                        radius: 3.0.into(),
                                    },
                                    ..Default::default()
                                }
                            } else {
                                button::secondary(theme, status)
                            }
                        })
                        .on_press(Message::EccMotionTypeChanged(EccMotionType::Translation)),
                    button(text("Euclidean (3-DOF)").size(12))
                        .style(move |theme, status| {
                            if euclidean_selected {
                                button::Style {
                                    background: Some(iced::Background::Color(iced::Color::from_rgb(0.3, 0.5, 0.6))),
                                    text_color: iced::Color::WHITE,
                                    border: iced::Border {
                                        color: iced::Color::from_rgb(0.4, 0.7, 0.8),
                                        width: 2.0,
                                        radius: 3.0.into(),
                                    },
                                    ..Default::default()
                                }
                            } else {
                                button::secondary(theme, status)
                            }
                        })
                        .on_press(Message::EccMotionTypeChanged(EccMotionType::Euclidean)),
                ].spacing(5),
                row![
                    button(text("Affine (6-DOF)").size(12))
                        .style(move |theme, status| {
                            if affine_selected {
                                button::Style {
                                    background: Some(iced::Background::Color(iced::Color::from_rgb(0.3, 0.5, 0.6))),
                                    text_color: iced::Color::WHITE,
                                    border: iced::Border {
                                        color: iced::Color::from_rgb(0.4, 0.7, 0.8),
                                        width: 2.0,
                                        radius: 3.0.into(),
                                    },
                                    ..Default::default()
                                }
                            } else {
                                button::secondary(theme, status)
                            }
                        })
                        .on_press(Message::EccMotionTypeChanged(EccMotionType::Affine)),
                    button(text("Homography (8-DOF)").size(12))
                        .style(move |theme, status| {
                            if homography_selected {
                                button::Style {
                                    background: Some(iced::Background::Color(iced::Color::from_rgb(0.3, 0.5, 0.6))),
                                    text_color: iced::Color::WHITE,
                                    border: iced::Border {
                                        color: iced::Color::from_rgb(0.4, 0.7, 0.8),
                                        width: 2.0,
                                        radius: 3.0.into(),
                                    },
                                    ..Default::default()
                                }
                            } else {
                                button::secondary(theme, status)
                            }
                        })
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
            .padding(10)
            .into()
        } else {
            // Empty column when ECC not selected
            column![].into()
        };

        // Transform validation sliders - shown for ALL alignment methods
        let transform_validation_ui = column![
            horizontal_rule(1),
            text("Transform Validation (All Alignment Methods):").size(13).style(|_| text::Style {
                color: Some(iced::Color::from_rgb(0.7, 0.8, 1.0))
            }),
            text("Reject distorted transformations - applies to ORB, SIFT, AKAZE, and ECC").size(10)
                .style(|_| text::Style {
                    color: Some(iced::Color::from_rgb(0.6, 0.6, 0.6))
                }),
            column![
                text(format!("Max Transform Scale: {:.2}x", self.config.max_transform_scale)).size(12),
                scrollable_slider(1.1..=3.0, self.config.max_transform_scale, Message::MaxTransformScaleChanged, 0.1, Length::Fill),
                text("Maximum allowed scale deviation (reject if exceeded)").size(10)
                    .style(|_| text::Style {
                        color: Some(iced::Color::from_rgb(0.6, 0.6, 0.6))
                    })
            ].spacing(3),
            column![
                text(format!("Max Translation: {:.0} pixels", self.config.max_transform_translation)).size(12),
                scrollable_slider(100.0..=1000.0, self.config.max_transform_translation, Message::MaxTransformTranslationChanged, 50.0, Length::Fill),
                text("Maximum allowed translation distance (reject if exceeded)").size(10)
                    .style(|_| text::Style {
                        color: Some(iced::Color::from_rgb(0.6, 0.6, 0.6))
                    })
            ].spacing(3),
            column![
                text(format!("Max Determinant: {:.1}x", self.config.max_transform_determinant)).size(12),
                scrollable_slider(1.5..=5.0, self.config.max_transform_determinant, Message::MaxTransformDeterminantChanged, 0.5, Length::Fill),
                text("Maximum determinant deviation (skew/distortion threshold)").size(10)
                    .style(|_| text::Style {
                        color: Some(iced::Color::from_rgb(0.6, 0.6, 0.6))
                    })
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
                text("Alignment & Detection").size(16).style(|_| text::Style { 
                    color: Some(iced::Color::from_rgb(0.8, 0.8, 1.0)) 
                }),
                sharpness_slider,
                grid_size_slider,
                iqr_multiplier_slider,
                adaptive_batch_checkbox,
                batch_info,
                clahe_checkbox,
                feature_layout,
                ecc_params_ui,
                transform_validation_ui,
            ]
            .spacing(10)
        )
        .padding(10)
        .style(|_theme| {
            container::Style {
                background: Some(iced::Background::Color(iced::Color::from_rgba(0.1, 0.1, 0.15, 0.3))),
                border: iced::Border {
                    color: iced::Color::from_rgb(0.3, 0.3, 0.4),
                    width: 1.0,
                    radius: 8.0.into(),
                },
                ..Default::default()
            }
        })
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
                text("Post-Processing").size(16).style(|_| text::Style { 
                    color: Some(iced::Color::from_rgb(0.8, 0.8, 1.0)) 
                }),
                noise_section,
                sharpen_section,
                color_section,
            ]
            .spacing(10)
        )
        .padding(10)
        .style(|_theme| {
            container::Style {
                background: Some(iced::Background::Color(iced::Color::from_rgba(0.1, 0.1, 0.15, 0.3))),
                border: iced::Border {
                    color: iced::Color::from_rgb(0.3, 0.3, 0.4),
                    width: 1.0,
                    radius: 8.0.into(),
                },
                ..Default::default()
            }
        })
        .width(Length::Fill);

        // ============== PANE 3: PREVIEW & UI ==============
        let preview_pane = container(
            column![
                text("Preview & UI").size(16).style(|_| text::Style { 
                    color: Some(iced::Color::from_rgb(0.8, 0.8, 1.0)) 
                }),
                
                checkbox("Use Internal Preview (modal overlay)", self.config.use_internal_preview)
                    .on_toggle(Message::UseInternalPreview),
                
                text("When disabled, left-click opens in external viewer (configurable below)").size(10)
                    .style(|_| text::Style { color: Some(iced::Color::from_rgb(0.6, 0.6, 0.6)) }),
                
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
                    .style(|_| text::Style { color: Some(iced::Color::from_rgb(0.7, 0.7, 0.9)) }),
                
                text_input(
                    "Path to viewer (e.g., /usr/bin/eog, /usr/bin/geeqie)...",
                    &self.config.external_viewer_path
                )
                    .on_input(Message::ExternalViewerPathChanged)
                    .padding(5),
                
                text("Leave empty to use system default viewer. Used when 'Use Internal Preview' is disabled.")
                    .size(10)
                    .style(|_| text::Style { color: Some(iced::Color::from_rgb(0.6, 0.6, 0.6)) }),
                
                container(column![]).height(10),  // Spacer
                
                text("External Image Editor (for right-click)").size(12)
                    .style(|_| text::Style { color: Some(iced::Color::from_rgb(0.7, 0.7, 0.9)) }),
                
                text_input(
                    "Path to editor (e.g., /usr/bin/gimp, darktable, photoshop)...",
                    &self.config.external_editor_path
                )
                    .on_input(Message::ExternalEditorPathChanged)
                    .padding(5),
                
                text("Leave empty to use system default viewer. Right-click any thumbnail to open with editor.")
                    .size(10)
                    .style(|_| text::Style { color: Some(iced::Color::from_rgb(0.6, 0.6, 0.6)) }),
            ]
            .spacing(10)
        )
        .padding(10)
        .style(|_theme| {
            container::Style {
                background: Some(iced::Background::Color(iced::Color::from_rgba(0.1, 0.1, 0.15, 0.3))),
                border: iced::Border {
                    color: iced::Color::from_rgb(0.3, 0.3, 0.4),
                    width: 1.0,
                    radius: 8.0.into(),
                },
                ..Default::default()
            }
        })
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
                text("GPU / Performance").size(16).style(|_| text::Style { 
                    color: Some(iced::Color::from_rgb(0.8, 0.8, 1.0)) 
                }),

                text("GPU Concurrency").size(13).style(|_| text::Style {
                    color: Some(iced::Color::from_rgb(0.7, 0.8, 1.0))
                }),
                text("Max simultaneous GPU operations. Lower = less VRAM usage, higher = faster.").size(10)
                    .style(|_| text::Style {
                        color: Some(iced::Color::from_rgb(0.6, 0.6, 0.6))
                    }),
                row![
                    text("GPU Concurrency:").width(label_width),
                    scrollable_slider(0.0..=8.0, self.config.gpu_concurrency as f32, Message::GpuConcurrencyChanged, 1.0, slider_width),
                    text(format!("{}", self.config.gpu_concurrency)).width(value_width),
                ]
                .spacing(10)
                .align_y(iced::Alignment::Center),
                text(gpu_concurrency_desc).size(11)
                    .style(|_| text::Style {
                        color: Some(iced::Color::from_rgb(0.5, 0.7, 0.5))
                    }),

                container(column![]).height(10),  // Spacer
                
                text("⚠ Changes take effect on next app restart").size(11)
                    .style(|_| text::Style {
                        color: Some(iced::Color::from_rgb(0.8, 0.6, 0.3))
                    }),
                
                horizontal_rule(1),

                // Stacking Bunch Size
                text("Stacking Bunch Size").size(13).style(|_| text::Style {
                    color: Some(iced::Color::from_rgb(0.7, 0.8, 1.0))
                }),
                text("Number of images per bunch during recursive stacking. More images per bunch = fewer levels, but higher memory usage.").size(10)
                    .style(|_| text::Style {
                        color: Some(iced::Color::from_rgb(0.6, 0.6, 0.6))
                    }),
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
                            .style(|_| text::Style {
                                color: Some(iced::Color::from_rgb(0.5, 0.7, 0.5))
                            }),
                    ].spacing(4)
                } else {
                    column![
                        text(format!("Active: {} images per bunch (auto)", self.config.batch_config.stacking_batch_size)).size(11)
                            .style(|_| text::Style {
                                color: Some(iced::Color::from_rgb(0.5, 0.7, 0.5))
                            }),
                    ]
                },

                horizontal_rule(1),
                
                text("Environment Variable Overrides").size(13).style(|_| text::Style {
                    color: Some(iced::Color::from_rgb(0.7, 0.8, 1.0))
                }),
                text("Env vars override settings (for testing/debugging):").size(10)
                    .style(|_| text::Style {
                        color: Some(iced::Color::from_rgb(0.6, 0.6, 0.6))
                    }),
                
                column![
                    text("IMAGESTACKER_GPU_CONCURRENCY=N").size(11)
                        .style(|_| text::Style {
                            color: Some(iced::Color::from_rgb(0.6, 0.8, 0.6))
                        }),
                    text("  Override GPU concurrency (0=unlimited, 1=serialized)").size(10)
                        .style(|_| text::Style {
                            color: Some(iced::Color::from_rgb(0.5, 0.5, 0.5))
                        }),
                ].spacing(2),
                
                column![
                    text("IMAGESTACKER_OPENCL_MUTEX=1").size(11)
                        .style(|_| text::Style {
                            color: Some(iced::Color::from_rgb(0.6, 0.8, 0.6))
                        }),
                    text("  Force GPU concurrency to 1 (fully serialized)").size(10)
                        .style(|_| text::Style {
                            color: Some(iced::Color::from_rgb(0.5, 0.5, 0.5))
                        }),
                ].spacing(2),
                
                column![
                    text("IMAGESTACKER_ECC_BATCH_SIZE=N").size(11)
                        .style(|_| text::Style {
                            color: Some(iced::Color::from_rgb(0.6, 0.8, 0.6))
                        }),
                    text("  Override ECC batch size (overrides 'Batch Size' setting)").size(10)
                        .style(|_| text::Style {
                            color: Some(iced::Color::from_rgb(0.5, 0.5, 0.5))
                        }),
                ].spacing(2),
            ]
            .spacing(8)
        )
        .padding(10)
        .style(|_theme| {
            container::Style {
                background: Some(iced::Background::Color(iced::Color::from_rgba(0.1, 0.1, 0.15, 0.3))),
                border: iced::Border {
                    color: iced::Color::from_rgb(0.3, 0.3, 0.4),
                    width: 1.0,
                    radius: 8.0.into(),
                },
                ..Default::default()
            }
        })
        .width(Length::Fill);

        // ============== RESET BUTTON ==============
        let reset_button = button(
            row![
                text("🔄").size(16),
                text("Reset to Defaults"),
            ]
            .spacing(5)
            .align_y(iced::Alignment::Center)
        )
        .style(|theme, status| {
            button::Style {
                background: Some(iced::Background::Color(iced::Color::from_rgb(0.6, 0.3, 0.3))),
                text_color: iced::Color::WHITE,
                border: iced::Border {
                    color: iced::Color::from_rgb(0.8, 0.4, 0.4),
                    width: 1.0,
                    radius: 6.0.into(),
                },
                ..button::primary(theme, status)
            }
        })
        .on_press(Message::ResetToDefaults);

        // ============== RESPONSIVE LAYOUT ==============
        // Use horizontal layout when window is wide enough
        // Wide: 2x2 grid (2 panes per row at 380px each + spacing)
        // Narrow: vertical stack
        const MIN_PANE_WIDTH: f32 = 380.0;
        
        let panes_layout: Element<'_, Message> = if use_horizontal_layout {
            // 2x2 grid for wide screens
            column![
                row![
                    container(alignment_pane).width(Length::Fixed(MIN_PANE_WIDTH)),
                    container(postprocessing_pane).width(Length::Fixed(MIN_PANE_WIDTH)),
                    container(preview_pane).width(Length::Fixed(MIN_PANE_WIDTH)),
                ]
                .spacing(15),
                row![
                    container(gpu_pane).width(Length::Fixed(MIN_PANE_WIDTH)),
                ]
                .spacing(15),
            ]
            .spacing(15)
            .into()
        } else {
            // Vertical layout for narrow screens - panes stacked
            column![
                alignment_pane,
                postprocessing_pane,
                preview_pane,
                gpu_pane,
            ]
            .spacing(15)
            .into()
        };

        container(
            scrollable(
                column![
                    text("Processing Settings").size(20).style(|_| text::Style { 
                        color: Some(iced::Color::from_rgb(0.9, 0.9, 1.0)) 
                    }),
                    panes_layout,
                    reset_button,
                ]
                .spacing(15)
            )
            .height(Length::Fill)
        )
        .padding(15)
        .width(Length::Fill)
        .height(Length::Fill)
        .style(|_| container::Style::default()
            .background(iced::Color::from_rgb(0.2, 0.2, 0.25)))
        .into()
    }
}
