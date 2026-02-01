//! Main view rendering
//!
//! This module contains the main application view and image preview modal.

use iced::widget::{
    button, column, container, horizontal_space, image as iced_image, row, text, text_input,
};
use iced::{Element, Length};
use iced::window;

use crate::messages::Message;
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
        } else {
            button("Align")
                .style(|theme, _status| button::Style {
                    background: Some(iced::Background::Color(iced::Color::from_rgb(0.3, 0.3, 0.3))),
                    text_color: iced::Color::from_rgb(0.5, 0.5, 0.5),
                    ..button::secondary(theme, button::Status::Disabled)
                })
        };

        // Stack button: enabled when aligned images exist and not processing
        let stack_aligned_button = if !self.aligned_images.is_empty() && !self.is_processing {
            button("Stack Aligned").on_press(Message::StackImages)
        } else {
            button("Stack Aligned")
                .style(|theme, _status| button::Style {
                    background: Some(iced::Background::Color(iced::Color::from_rgb(0.3, 0.3, 0.3))),
                    text_color: iced::Color::from_rgb(0.5, 0.5, 0.5),
                    ..button::secondary(theme, button::Status::Disabled)
                })
        };

        // Stack Bunches button: enabled when bunch images exist and not processing
        let stack_bunches_button = if !self.bunch_images.is_empty() && !self.is_processing {
            button("Stack Bunches").on_press(Message::StackBunches)
        } else {
            button("Stack Bunches")
                .style(|theme, _status| button::Style {
                    background: Some(iced::Background::Color(iced::Color::from_rgb(0.3, 0.3, 0.3))),
                    text_color: iced::Color::from_rgb(0.5, 0.5, 0.5),
                    ..button::secondary(theme, button::Status::Disabled)
                })
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
        .style(|_| container::Style::default()
            .border(iced::Border::default()
                .width(1.0)
                .color(iced::Color::from_rgb(0.4, 0.4, 0.4))));

        let align_group = container(
            column![
                align_button
            ]
        )
        .padding(8)
        .style(|_| container::Style::default()
            .border(iced::Border::default()
                .width(1.0)
                .color(iced::Color::from_rgb(0.4, 0.4, 0.4))));

        let stack_group = container(
            column![
                row![
                    stack_aligned_button,
                    stack_bunches_button,
                ]
                .spacing(10)
            ]
        )
        .padding(8)
        .style(|_| container::Style::default()
            .border(iced::Border::default()
                .width(1.0)
                .color(iced::Color::from_rgb(0.4, 0.4, 0.4))));

        let tools_group = container(
            column![
                row![
                    button(if self.show_settings { "Hide Settings" } else { "Settings" })
                        .on_press(Message::ToggleSettings),
                    button("Help")
                        .on_press(Message::ToggleHelp),
                    button("Show Log")
                        .on_press(Message::ToggleLog),
                ]
                .spacing(10)
            ]
        )
        .padding(8)
        .style(|_| container::Style::default()
            .border(iced::Border::default()
                .width(1.0)
                .color(iced::Color::from_rgb(0.4, 0.4, 0.4))));

        let buttons = row![
            import_group,
            align_group,
            stack_group,
            tools_group,
            horizontal_space(),
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
                        .style(|_| text::Style {
                            color: Some(iced::Color::from_rgb(1.0, 0.8, 0.0))
                        })
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
            self.render_aligned_pane(),
            self.render_bunches_pane(),
            self.render_pane("Final", &self.final_images),
        ]
        .spacing(10)
        .padding(10)
        .height(Length::Fill);

        main_column = main_column
            .push(panes)
            .push(
                container(
                    text_input("", &self.status)
                        .size(16)
                        .style(|_theme, _status| text_input::Style {
                            background: iced::Background::Color(iced::Color::from_rgb(0.1, 0.1, 0.1)),
                            border: iced::Border {
                                color: iced::Color::from_rgb(0.6, 0.6, 0.6),
                                width: 1.0,
                                radius: 4.0.into(),
                            },
                            icon: iced::Color::from_rgb(0.8, 0.8, 0.8),
                            placeholder: iced::Color::from_rgb(0.5, 0.5, 0.5),
                            value: iced::Color::from_rgb(0.9, 0.9, 0.9),
                            selection: iced::Color::from_rgb(0.5, 0.7, 1.0),
                        })
                )
                    .padding(5)
                    .width(Length::Fill)
                    .style(|_| container::Style::default()
                        .background(iced::Color::from_rgb(0.1, 0.1, 0.1)))
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
                // Show the loaded image
                container(
                    iced_image(handle.clone())
                        .width(Length::Fixed(self.config.preview_max_width - 40.0)) // Account for padding
                        .height(Length::Fixed(self.config.preview_max_height - 140.0)) // Account for header/footer/padding
                        .content_fit(iced::ContentFit::Contain)
                )
                .width(Length::Fixed(self.config.preview_max_width - 40.0))
                .height(Length::Fixed(self.config.preview_max_height - 140.0))
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
                            .style(|_theme| text::Style { color: Some(iced::Color::from_rgb(0.7, 0.7, 0.7)) })
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
            )
            .width(Length::Fixed(self.config.preview_max_width))
            .height(Length::Fixed(self.config.preview_max_height))
            .style(|_| container::Style::default()
                .background(iced::Background::Color(iced::Color::from_rgb(0.15, 0.15, 0.15)))
                .border(iced::Border::default()
                    .width(2.0)
                    .color(iced::Color::from_rgb(0.3, 0.3, 0.3))));

            // Stack the preview on top of the background
            iced::widget::stack![
                background,
                container(preview_content)
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