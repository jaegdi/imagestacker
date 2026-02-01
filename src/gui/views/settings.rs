//! Settings panel rendering
//!
//! This module contains the settings panel UI with all configuration options.

use iced::widget::{
    button, checkbox, column, container, row, slider, text, text_input,
};
use iced::{Element, Length};

use crate::config::FeatureDetector;
use crate::messages::Message;
use super::super::state::ImageStacker;

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
            slider(10.0..=10000.0, self.config.sharpness_threshold, Message::SharpnessThresholdChanged)
                .step(10.0)
                .width(slider_width),
            text(format!("{:.0}", self.config.sharpness_threshold)).width(value_width),
        ]
        .spacing(10)
        .align_y(iced::Alignment::Center);

        let grid_size_slider = row![
            text("Sharpness Grid:").width(label_width),
            slider(4.0..=16.0, self.config.sharpness_grid_size as f32, Message::SharpnessGridSizeChanged)
                .step(1.0)
                .width(slider_width),
            text(format!("{}x{}", self.config.sharpness_grid_size, self.config.sharpness_grid_size)).width(value_width),
        ]
        .spacing(10)
        .align_y(iced::Alignment::Center);

        let iqr_multiplier_slider = row![
            text("Blur Filter (IQR):").width(label_width),
            slider(0.5..=5.0, self.config.sharpness_iqr_multiplier, Message::SharpnessIqrMultiplierChanged)
                .step(0.1)
                .width(slider_width),
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

        // Feature detector layout: vertical when horizontal pane layout, horizontal when vertical pane layout
        let feature_layout: Element<'_, Message> = if use_horizontal_layout {
            // Horizontal layout -> stack detector buttons vertically
            column![
                text("Feature Detector:").size(14),
                orb_button.width(Length::Fixed(160.0)),
                sift_button.width(Length::Fixed(160.0)),
                akaze_button.width(Length::Fixed(160.0)),
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
            ]
            .spacing(10)
            .into()
        };

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
                slider(1.0..=10.0, self.config.noise_reduction_strength, Message::NoiseReductionStrengthChanged)
                    .width(slider_width),
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
                slider(0.0..=5.0, self.config.sharpening_strength, Message::SharpeningStrengthChanged)
                    .width(slider_width),
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
                slider(0.5..=3.0, self.config.contrast_boost, Message::ContrastBoostChanged)
                    .width(slider_width),
                text(format!("{:.1}", self.config.contrast_boost)).width(value_width),
            ]
            .spacing(10)
            .align_y(iced::Alignment::Center),
            
            row![
                text("Brightness:").width(label_width),
                slider(-100.0..=100.0, self.config.brightness_boost, Message::BrightnessBoostChanged)
                    .width(slider_width),
                text(format!("{:.0}", self.config.brightness_boost)).width(value_width),
            ]
            .spacing(10)
            .align_y(iced::Alignment::Center),
            
            row![
                text("Saturation:").width(label_width),
                slider(0.0..=3.0, self.config.saturation_boost, Message::SaturationBoostChanged)
                    .width(slider_width),
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
                    slider(400.0..=2000.0, self.config.preview_max_width, Message::PreviewMaxWidthChanged)
                        .width(slider_width),
                    text(format!("{:.0}px", self.config.preview_max_width)).width(value_width),
                ]
                .spacing(10)
                .align_y(iced::Alignment::Center),
                
                row![
                    text("Preview Max Height:").width(label_width),
                    slider(300.0..=1500.0, self.config.preview_max_height, Message::PreviewMaxHeightChanged)
                        .width(slider_width),
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
        // Use horizontal layout when window is wide enough (>= 1200px for 3 panes at 380px each)
        // Otherwise, stack vertically for better readability
        const MIN_PANE_WIDTH: f32 = 380.0;
        
        let panes_layout: Element<'_, Message> = if use_horizontal_layout {
            // Horizontal layout for wide screens - panes side by side
            row![
                container(alignment_pane).width(Length::Fixed(MIN_PANE_WIDTH)),
                container(postprocessing_pane).width(Length::Fixed(MIN_PANE_WIDTH)),
                container(preview_pane).width(Length::Fixed(MIN_PANE_WIDTH)),
            ]
            .spacing(15)
            .into()
        } else {
            // Vertical layout for narrow screens - panes stacked
            column![
                alignment_pane,
                postprocessing_pane,
                preview_pane,
            ]
            .spacing(15)
            .into()
        };

        container(
            column![
                text("Processing Settings").size(20).style(|_| text::Style { 
                    color: Some(iced::Color::from_rgb(0.9, 0.9, 1.0)) 
                }),
                panes_layout,
                reset_button,
            ]
            .spacing(15)
        )
        .padding(15)
        .width(Length::Fill)
        .style(|_| container::Style::default()
            .background(iced::Color::from_rgb(0.2, 0.2, 0.25)))
        .into()
    }
}
