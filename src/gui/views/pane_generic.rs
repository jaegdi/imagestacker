//! Generic pane rendering for Imported and Final panes
//!
//! This module provides generic pane rendering functions for panes
//! that don't require selection mode (Imported and Final).

use iced::widget::{
    button, column, container, image as iced_image, mouse_area, row, scrollable, text,
};
use iced::{Element, Length};
use std::path::PathBuf;

use crate::messages::Message;
use crate::gui::state::ImageStacker;
use crate::gui::theme;

impl ImageStacker {
    /// Render a generic pane (for Imported and Final panes)
    pub(crate) fn render_pane<'a>(&self, title: &'a str, images: &'a [PathBuf]) -> Element<'a, Message> {
        // Fixed column count to prevent thumbnail squeezing
        // With 4 panes, 2 columns works well for window widths 1400px+
        // For narrower windows, thumbnails stay fixed size and pane gets horizontal scroll if needed
        let thumbs_per_row = 2;
        self.render_pane_with_columns(title, images, thumbs_per_row)
    }

    /// Render a pane with a specified number of columns
    fn render_pane_with_columns<'a>(&self, title: &'a str, images: &'a [PathBuf], thumbs_per_row: usize) -> Element<'a, Message> {
        // Create scroll message closure
        let scroll_message = match title {
            "Imported" => |offset: f32| Message::ImportedScrollChanged(offset),
            "Sharpness" => |offset: f32| Message::SharpnessScrollChanged(offset),
            "Aligned" => |offset: f32| Message::AlignedScrollChanged(offset),
            "Bunches" => |offset: f32| Message::BunchesScrollChanged(offset),
            "Final" => |offset: f32| Message::FinalScrollChanged(offset),
            _ => |_: f32| Message::None,
        };

        // Constants for thumbnail sizing
        const THUMB_WIDTH: f32 = 120.0;
        const THUMB_HEIGHT: f32 = 90.0;
        const THUMB_SPACING: f32 = 8.0;
        
        // Use fixed number of columns passed as parameter
        // This prevents thumbnails from being squeezed when window resizes
        
        // Group images into rows based on fixed column count
        let mut rows_vec: Vec<Element<Message>> = Vec::new();
        
        for chunk in images.chunks(thumbs_per_row) {
            let mut row_elements: Vec<Element<Message>> = Vec::new();
            
            for path in chunk {
                let path_clone = path.clone();
                let cache = self.thumbnail_cache.read().unwrap();
                let handle = cache.get(path).cloned();

                let image_widget: Element<Message> = if let Some(h) = handle {
                    iced_image(h)
                        .width(Length::Fixed(THUMB_WIDTH))
                        .height(Length::Fixed(THUMB_HEIGHT))
                        .content_fit(iced::ContentFit::ScaleDown)
                        .into()
                } else {
                    container(text("Loading...").size(10))
                        .width(Length::Fixed(THUMB_WIDTH))
                        .height(Length::Fixed(THUMB_HEIGHT))
                        .center_x(Length::Fill)
                        .center_y(Length::Fill)
                        .style(|_| {
                            container::Style::default().background(theme::bg::PLACEHOLDER)
                        })
                        .into()
                };

                let path_for_left_click = path_clone.clone();
                let path_for_right_click = path_clone.clone();
                
                let thumbnail_element = button(
                    column![
                        image_widget,
                        container(text(path.file_name().unwrap_or_default().to_string_lossy()).size(9))
                            .width(Length::Fixed(THUMB_WIDTH))
                            .center_x(Length::Fill)
                    ]
                    .align_x(iced::Alignment::Center),
                )
                .on_press(if self.config.use_internal_preview {
                    Message::ShowImagePreview(path_for_left_click, images.to_vec())
                } else {
                    Message::OpenImage(path_for_left_click)
                })
                .style(button::secondary)
                .width(Length::Fixed(THUMB_WIDTH));
                
                // Wrap in mouse_area to detect right-clicks
                let thumbnail_with_mouse = mouse_area(thumbnail_element)
                    .on_right_press(Message::OpenImageWithExternalEditor(path_for_right_click))
                    .into();
                
                row_elements.push(thumbnail_with_mouse);
            }
            
            // Create a row with the thumbnails
            let thumbnail_row = row(row_elements)
                .spacing(THUMB_SPACING)
                .align_y(iced::Alignment::Start)
                .into();
            
            rows_vec.push(thumbnail_row);
        }
        
        let content = column(rows_vec)
            .spacing(THUMB_SPACING)
            .align_x(iced::Alignment::Center)
            .height(Length::Shrink);

        // Create scrollable ID based on pane title
        let scrollable_id = match title {
            "Imported" => Some(iced::widget::scrollable::Id::new("imported")),
            "Sharpness" => Some(iced::widget::scrollable::Id::new("sharpness")),
            "Aligned" => Some(iced::widget::scrollable::Id::new("aligned")),
            "Bunches" => Some(iced::widget::scrollable::Id::new("bunches")),
            "Final" => Some(iced::widget::scrollable::Id::new("final")),
            _ => None,
        };

        let mut scrollable_widget = scrollable(content)
            .width(Length::Fill)
            .on_scroll(move |viewport| {
                // Calculate scroll offset from viewport
                let offset = viewport.absolute_offset().y;
                scroll_message(offset)
            });

        // Set ID if available to preserve scroll position
        if let Some(id) = scrollable_id {
            scrollable_widget = scrollable_widget.id(id);
        }

        // Determine refresh message based on pane title
        let refresh_message = match title {
            "Imported" => Message::RefreshImportedPane,
            "Sharpness" => Message::RefreshSharpnessPane,
            "Aligned" => Message::RefreshAlignedPane,
            "Bunches" => Message::RefreshBunchesPane,
            "Final" => Message::RefreshFinalPane,
            _ => Message::None,
        };

        // Create pane header with title, count, and refresh button
        let pane_header = row![
            column![
                text(title)
                    .size(18)
                    .align_x(iced::Alignment::Center),
                text(format!("{} images", images.len()))
                    .size(12)
                    .style(|t| theme::secondary_text(t))
                    .align_x(iced::Alignment::Center)
            ]
            .width(Length::Fill)
            .align_x(iced::Alignment::Center),
            button("Refresh")
                .on_press(refresh_message)
                .padding(4)
                .style(theme::refresh_button)
        ]
        .spacing(5)
        .align_y(iced::Alignment::Center);

        container(
            column![
                pane_header,
                container(scrollable_widget)
                    .width(Length::Fill)
                    .height(Length::Fill)
                    .style(|t| theme::scrollable_inner(t))
            ]
            .spacing(10),
        )
        .width(Length::FillPortion(1))
        .height(Length::Fill)
        .padding(5)
        .style(|_| {
            theme::pane_container(theme::pane::FINAL_BG, theme::pane::FINAL_BORDER)
        })
        .into()
    }
}
