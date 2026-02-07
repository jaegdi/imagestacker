//! Imported pane rendering with selection mode support
//!
//! This module handles rendering the imported images pane with:
//! - Selection mode for choosing images to stack
//! - Thumbnail grid with click and right-click support
//! - Select All/Deselect All/Cancel/Stack buttons when in selection mode

use iced::widget::{
    button, column, container, image as iced_image, mouse_area, row, scrollable, text,
};
use iced::{Element, Length};

use crate::messages::Message;
use crate::gui::state::ImageStacker;
use crate::gui::theme;

impl ImageStacker {
    pub(crate) fn render_imported_pane(&self) -> Element<'_, Message> {
        const THUMB_WIDTH: f32 = 120.0;
        const THUMB_HEIGHT: f32 = 90.0;
        const THUMB_SPACING: f32 = 8.0;
        let thumbs_per_row = 2;

        let mut rows_vec: Vec<Element<Message>> = Vec::new();
        
        for chunk in self.images.chunks(thumbs_per_row) {
            let mut row_elements: Vec<Element<Message>> = Vec::new();
            
            for path in chunk {
                let path_clone = path.clone();
                let path_for_right_click = path.clone();
                let is_selected = self.selected_imported.contains(path);
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

                let thumbnail_content = column![
                    image_widget,
                    container(text(path.file_name().unwrap_or_default().to_string_lossy()).size(9))
                        .width(Length::Fixed(THUMB_WIDTH))
                        .center_x(Length::Fill)
                ]
                .align_x(iced::Alignment::Center);

                let thumbnail_element = if self.imported_selection_mode {
                    // In selection mode: make clickable for selection
                    let btn = button(thumbnail_content)
                        .on_press(Message::ToggleImportedImage(path_clone))
                        .width(Length::Fixed(THUMB_WIDTH));
                    
                    // Apply different style if selected - green background and border
                    if is_selected {
                        container(
                            btn.style(theme::thumb_selected)
                        )
                        .style(|t| theme::thumb_selected_container(t))
                        .padding(2)
                        .into()
                    } else {
                        // Unselected - darker background
                        btn.style(theme::thumb_unselected)
                        .into()
                    }
                } else {
                    // Normal mode: preview on click
                    button(thumbnail_content)
                        .on_press(if self.config.use_internal_preview {
                            Message::ShowImagePreview(path_clone, self.images.clone())
                        } else {
                            Message::OpenImage(path_clone)
                        })
                        .style(button::secondary)
                        .width(Length::Fixed(THUMB_WIDTH))
                        .into()
                };
                
                // Wrap in mouse_area to detect right-clicks (only in normal mode)
                let final_element = if !self.imported_selection_mode {
                    mouse_area(thumbnail_element)
                        .on_right_press(Message::OpenImageWithExternalEditor(path_for_right_click))
                        .into()
                } else {
                    thumbnail_element
                };
                
                row_elements.push(final_element);
            }
            
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

        let scrollable_widget = scrollable(content)
            .width(Length::Fill)
            .on_scroll(move |viewport| {
                let offset = viewport.absolute_offset().y;
                Message::ImportedScrollChanged(offset)
            })
            .id(iced::widget::scrollable::Id::new("imported"));

        // Create pane header with title, count, and refresh button
        let pane_header = row![
            column![
                text("Imported")
                    .size(18)
                    .align_x(iced::Alignment::Center),
                text(format!("{} images", self.images.len()))
                    .size(12)
                    .style(|t| theme::secondary_text(t))
                    .align_x(iced::Alignment::Center)
            ]
            .width(Length::Fill)
            .align_x(iced::Alignment::Center),
            button("Refresh")
                .on_press(Message::RefreshImportedPane)
                .padding(4)
                .style(theme::refresh_button)
        ]
        .spacing(5)
        .align_y(iced::Alignment::Center);

        // Add Cancel and Stack buttons at bottom if in selection mode
        let mut pane_content = column![
            pane_header,
            container(scrollable_widget)
                .width(Length::Fill)
                .height(Length::Fill)
                .style(|t| theme::scrollable_inner(t))
        ]
        .spacing(10);

        if self.imported_selection_mode {
            pane_content = pane_content
                .push(
                    row![
                        button("Select All")
                            .on_press(Message::SelectAllImported)
                            .style(button::secondary),
                        button("Deselect All")
                            .on_press(Message::DeselectAllImported)
                            .style(button::secondary),
                    ]
                    .spacing(10)
                    .padding(5)
                    .width(Length::Fill)
                )
                .push(
                    row![
                        button("Cancel")
                            .on_press(Message::CancelImportedSelection)
                            .style(theme::danger_button),
                        button(text(format!("Stack ({} selected)", self.selected_imported.len())))
                            .on_press(Message::StackSelectedImported)
                            .style(theme::success_button),
                    ]
                    .spacing(10)
                    .padding(5)
                    .width(Length::Fill)
                );
        }

        container(pane_content)
            .width(Length::Fill)
            .height(Length::Fill)
            .padding(10)
            .style(|_| theme::pane_container(theme::pane::IMPORTED_BG, theme::pane::IMPORTED_BORDER))
            .into()
    }
}
