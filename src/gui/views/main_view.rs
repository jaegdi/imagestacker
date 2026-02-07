//! Main view rendering
//!
//! This module contains the main application view and image preview modal.

use iced::widget::{
    button, column, container, horizontal_space, image as iced_image, row, text,
    pane_grid,
};
use iced::{Element, Length};
use iced::window;

use crate::messages::Message;
use crate::gui::state::PaneId;
use crate::gui::theme;
use super::super::state::ImageStacker;

impl ImageStacker {
    pub fn view(&self, window: window::Id) -> Element<'_, Message> {
        if Some(window) == self.help_window_id {
            self.render_help_window()
        } else if Some(window) == self.log_window_id {
            self.render_log_window()
        } else {
            self.render_image_preview()
        }
    }

    // ------------------------------------------------------------------------
    // Main View
    // ------------------------------------------------------------------------

    fn render_main_view(&self) -> Element<'_, Message> {
        // Align button: enabled when images are imported and not processing
        let align_button = if !self.images.is_empty() && !self.is_processing {
            button("Align").on_press(Message::AlignImages)
                .width(Length::Fixed(120.0))
        } else {
            button("Align")
                .style(theme::button_disabled)
                .width(Length::Fixed(120.0))
        };

        // Stack button: enabled when aligned images exist and not processing
        let stack_aligned_button = if !self.aligned_images.is_empty() && !self.is_processing {
            button("Stack Aligned").on_press(Message::StackImages)
                .style(theme::stack_aligned_button)
        } else {
            button("Stack Aligned")
                .style(theme::button_disabled)
        };

        // Stack Imported button: enabled when images are imported and not processing
        let stack_imported_button = if !self.images.is_empty() && !self.is_processing {
            button("Stack Imported").on_press(Message::StackImported)
                .style(theme::stack_imported_button)
        } else {
            button("Stack Imported")
                .style(theme::button_disabled)
        };

        // Stack Sharpness button: enabled when sharpness images exist and not processing
        let stack_sharpness_button = if !self.sharpness_images.is_empty() && !self.is_processing {
            button("Stack Sharpness").on_press(Message::StackSharpness)
                .style(theme::stack_sharpness_button)
        } else {
            button("Stack Sharpness")
                .style(theme::button_disabled)
        };

        // Stack Bunches button: enabled when bunch images exist and not processing
        let stack_bunches_button = if !self.bunch_images.is_empty() && !self.is_processing {
            button("Stack Bunches").on_press(Message::StackBunches)
                .style(theme::stack_bunches_button)
        } else {
            button("Stack Bunches")
                .style(theme::button_disabled)
        };

        // Group buttons in frames
        let import_group = container(
            column![
                row![
                    button("Add Images").on_press(Message::AddImages),
                    button("Add Folder").on_press(Message::AddFolder),
                ]
                .spacing(10)
            ]
        )
        .padding(8)
        .style(|t| theme::button_group(t));

        // Sharpness button: enabled when images are imported and not processing
        let sharpness_button = if !self.images.is_empty() && !self.is_processing {
            button("Detect Sharpness").on_press(Message::DetectSharpness)
        } else {
            button("Detect Sharpness")
                .style(theme::button_disabled)
        };

        let sharpness_group = container(
            column![
                sharpness_button
            ]
        )
        .padding(8)
        .style(|t| theme::button_group(t));

        let align_group = container(
            column![
                align_button
            ]
        )
        .padding(8)
        .style(|t| theme::button_group(t));

        let stack_group = container(
            column![
                row![
                    stack_imported_button,
                    stack_sharpness_button,
                    stack_aligned_button,
                    stack_bunches_button,
                ]
                .spacing(10)
            ]
        )
        .padding(8)
        .style(|t| theme::button_group(t));

        let tools_group = container(
            column![
                row![
                    {
                        let settings_btn = button(if self.show_settings { "Hide Settings" } else { "Settings" })
                            .on_press(Message::ToggleSettings);
                        if self.show_settings {
                            settings_btn.style(theme::settings_active_button)
                        } else {
                            settings_btn
                        }
                    },
                    button("Help")
                        .on_press(Message::ToggleHelp),
                    button("Show Log")
                        .on_press(Message::ToggleLog),
                ]
                .spacing(10)
            ]
        )
        .padding(8)
        .style(|t| theme::button_group(t));

        let buttons = row![
            import_group,
            sharpness_group,
            align_group,
            stack_group,
            horizontal_space(),
            tools_group,
            button("Exit").on_press(Message::Exit),
        ]
        .spacing(10)
        .padding(10)
        .align_y(iced::Alignment::Center);

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
                        .width(Length::Fill),
                    text("Press ESC to cancel")
                        .size(12)
                        .style(|t| theme::warning_text(t))
                ]
                .spacing(5)
            )
            .padding(10)
            .width(Length::Fill)
            .style(|t| theme::progress_container(t));
            
            main_column = main_column.push(progress_bar);
        }

        let panes = pane_grid::PaneGrid::new(&self.pane_state, |_pane, pane_id, _is_maximized| {
                let content: Element<'_, Message> = match pane_id {
                    PaneId::Imported  => self.render_imported_pane(),
                    PaneId::Sharpness => self.render_sharpness_pane(),
                    PaneId::Aligned   => self.render_aligned_pane(),
                    PaneId::Bunches   => self.render_bunches_pane(),
                    PaneId::Final     => self.render_pane("Final", &self.final_images),
                };
                pane_grid::Content::new(content)
            })
            .width(Length::Fill)
            .height(Length::Fill)
            .spacing(4)
            .on_resize(6, |event| Message::PaneResized(event))
            .on_drag(|event| Message::PaneDragged(event))
            .style(|t| theme::pane_grid_style(t));

        main_column = main_column
            .push(panes)
            .push(
                container(
                    row![
                        text(&self.status)
                            .size(13)
                            .width(Length::Fill),
                        text(format!(
                            "Imp: {}  Sharp: {}  Align: {}  Bunch: {}  Final: {}",
                            self.images.len(),
                            self.sharpness_images.len(),
                            self.aligned_images.len(),
                            self.bunch_images.len(),
                            self.final_images.len()
                        ))
                        .size(12)
                        .style(|t| theme::secondary_text(t)),
                    ]
                    .spacing(20)
                    .align_y(iced::Alignment::Center)
                    .padding(iced::Padding::from([4, 10]))
                )
                .width(Length::Fill)
                .style(|t| theme::status_bar(t))
            );

        main_column.into()
    }

    // ------------------------------------------------------------------------
    // Image Preview Modal
    // ------------------------------------------------------------------------

    fn render_image_preview(&self) -> Element<'_, Message> {
        if let Some(path) = &self.preview_image_path {
            // Dark overlay background
            let background = container(
                button("")
                    .on_press(Message::CloseImagePreview)
                    .style(|_theme, _status| button::Style {
                        background: Some(iced::Background::Color(iced::Color::from_rgba(0.0, 0.0, 0.0, 0.7))),
                        ..button::Style::default()
                    })
                    .width(Length::Fill)
                    .height(Length::Fill)
            )
            .width(Length::Fill)
            .height(Length::Fill);

            // Create the image content element
            let image_content: Element<'_, Message> = if self.preview_loading {
                // Show loading indicator
                container(text("Loading image...").size(18))
                    .width(Length::Fill)
                    .height(Length::Fill)
                    .center_x(Length::Fill)
                    .center_y(Length::Fill)
                    .into()
            } else if let Some(handle) = &self.preview_handle {
                // Use full window size for full resolution, configured size for thumbnails
                let (preview_width, preview_height) = if self.preview_is_thumbnail {
                    (self.config.preview_max_width, self.config.preview_max_height)
                } else {
                    // Full resolution: use 95% of window size to leave some margin
                    (self.window_width * 0.95, self.window_height * 0.95)
                };
                
                // Create base image widget
                let base_image = iced_image(handle.clone())
                    .width(Length::Fixed(preview_width - 40.0))
                    .height(Length::Fixed(preview_height - 140.0))
                    .content_fit(iced::ContentFit::Contain);
                
                // Add sharpness overlay if data is available
                let image_with_overlay = if let Some(info) = &self.preview_sharpness_info {
                    let overlay_text = column![
                        text("Sharpness Analysis").size(16).style(|t| theme::white_text(t)),
                        text("").size(4),
                        text(format!("Max Regional: {:.2}", info.max_regional_sharpness)).size(13).style(|t| theme::white_text(t)),
                        text(format!("Global: {:.2}", info.global_sharpness)).size(13).style(|t| theme::white_text(t)),
                        text(format!("Sharp Regions: {:.1}", info.sharp_region_count)).size(13).style(|t| theme::white_text(t)),
                        text(format!("Grid: {}x{}", info.grid_size, info.grid_size)).size(13).style(|t| theme::white_text(t)),
                        text(format!("Size: {}x{}", info.image_size.0, info.image_size.1)).size(13).style(|t| theme::white_text(t)),
                    ]
                    .spacing(3)
                    .padding(8);
                    
                    let overlay_container = container(overlay_text)
                        .style(|_theme: &iced::Theme| {
                            container::Style {
                                background: Some(iced::Background::Color(iced::Color::from_rgba(0.0, 0.0, 0.0, 0.75))),
                                border: iced::Border {
                                    width: 1.0,
                                    color: iced::Color::from_rgba(1.0, 1.0, 1.0, 0.2),
                                    radius: 6.0.into(),
                                },
                                ..Default::default()
                            }
                        });
                    
                    iced::widget::stack![
                        base_image,
                        container(overlay_container)
                            .width(Length::Fill)
                            .height(Length::Fill)
                            .align_x(iced::alignment::Horizontal::Left)
                            .align_y(iced::alignment::Vertical::Top)
                            .padding(12)
                    ]
                } else {
                    iced::widget::stack![base_image]
                };
                
                // Show the loaded image with overlay
                container(image_with_overlay)
                    .width(Length::Fixed(preview_width - 40.0))
                    .height(Length::Fixed(preview_height - 140.0))
                    .center_x(Length::Fill)
                    .center_y(Length::Fill)
                    .into()
            } else {
                // Fallback if no handle
                container(text("Failed to load image").size(16))
                    .width(Length::Fill)
                    .height(Length::Fill)
                    .center_x(Length::Fill)
                    .center_y(Length::Fill)
                    .into()
            };

            // Preview content
            let preview_content = container(
                column![
                    // Header with filename and close button
                    row![
                        text(path.file_name().unwrap_or_default().to_string_lossy())
                            .size(16)
                            .width(Length::Fill),
                        button("X")
                            .on_press(Message::CloseImagePreview)
                            .style(button::secondary)
                    ]
                    .spacing(10)
                    .align_y(iced::Alignment::Center),
                    // Navigation info
                    if !self.preview_current_pane.is_empty() {
                        let current_index = self.preview_current_pane.iter().position(|p| Some(p) == self.preview_image_path.as_ref()).unwrap_or(0);
                        text(format!("Image {} of {} (Use < | > arrow keys or mouse wheel to navigate)", 
                            current_index + 1, 
                            self.preview_current_pane.len()))
                            .size(12)
                            .style(|t| theme::secondary_text(t))
                    } else {
                        text("").size(12)
                    },
                    // Full-size image content
                    image_content,
                    // Footer with buttons
                    if self.preview_is_thumbnail && !self.preview_loading {
                        // Show full resolution button when displaying thumbnail
                        row![
                            button("< Previous")
                                .on_press(Message::PreviousImageInPreview)
                                .style(button::secondary),
                            button("Load Full Resolution")
                                .on_press(Message::LoadFullImage(path.clone()))
                                .style(button::primary),
                            button("Open in External Viewer")
                                .on_press(Message::OpenImage(path.clone()))
                                .style(button::secondary),
                            button("Next >")
                                .on_press(Message::NextImageInPreview)
                                .style(button::secondary),
                            button("Close")
                                .on_press(Message::CloseImagePreview)
                                .style(button::secondary)
                        ]
                        .spacing(10)
                    } else {
                        // Normal buttons
                        row![
                            button("< Previous")
                                .on_press(Message::PreviousImageInPreview)
                                .style(button::secondary),
                            button("Open in External Viewer")
                                .on_press(Message::OpenImage(path.clone()))
                                .style(button::primary),
                            button("Next >")
                                .on_press(Message::NextImageInPreview)
                                .style(button::secondary),
                            button("Close")
                                .on_press(Message::CloseImagePreview)
                                .style(button::secondary)
                        ]
                        .spacing(10)
                    }
                ]
                .spacing(10)
                .padding(20)
            );
            
            // Use full window size for full resolution, configured size for thumbnails
            let (preview_width, preview_height) = if self.preview_is_thumbnail {
                (self.config.preview_max_width, self.config.preview_max_height)
            } else {
                // Full resolution: use 95% of window size
                (self.window_width * 0.95, self.window_height * 0.95)
            };
            
            let preview_container = container(preview_content)
                .width(Length::Fixed(preview_width))
                .height(Length::Fixed(preview_height))
                .style(|t| theme::preview_overlay(t));

            // Stack the preview on top of the background
            iced::widget::stack![
                background,
                container(preview_container)
                    .width(Length::Fill)
                    .height(Length::Fill)
                    .center_x(Length::Fill)
                    .center_y(Length::Fill)
            ].into()
        } else {
            // No preview, show main view
            self.render_main_view()
        }
    }
}