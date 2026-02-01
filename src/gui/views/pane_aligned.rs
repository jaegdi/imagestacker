//! Aligned pane rendering with selection mode support
//!
//! This module handles rendering the aligned images pane with:
//! - Selection mode for choosing images to stack
//! - Thumbnail grid with click and right-click support
//! - Select All/Deselect All/Cancel/Stack buttons when in selection mode

use iced::widget::{
    button, column, container, image as iced_image, mouse_area, row, scrollable, text,
};
use iced::{Element, Length};

use crate::messages::Message;
use crate::gui::state::ImageStacker;

impl ImageStacker {
    pub(crate) fn render_aligned_pane(&self) -> Element<'_, Message> {
        const THUMB_WIDTH: f32 = 120.0;
        const THUMB_HEIGHT: f32 = 90.0;
        const THUMB_SPACING: f32 = 8.0;
        let thumbs_per_row = 2;

        let mut rows_vec: Vec<Element<Message>> = Vec::new();
        
        for chunk in self.aligned_images.chunks(thumbs_per_row) {
            let mut row_elements: Vec<Element<Message>> = Vec::new();
            
            for path in chunk {
                let path_clone = path.clone();
                let path_for_right_click = path.clone();
                let is_selected = self.selected_aligned.contains(path);
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
                            container::Style::default().background(iced::Color::from_rgb(0.2, 0.2, 0.2))
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

                let thumbnail_element = if self.aligned_selection_mode {
                    // In selection mode: make clickable for selection
                    let btn = button(thumbnail_content)
                        .on_press(Message::ToggleAlignedImage(path_clone))
                        .width(Length::Fixed(THUMB_WIDTH));
                    
                    // Apply different style if selected - green background and border
                    if is_selected {
                        container(
                            btn.style(|_theme, _status| button::Style {
                                background: Some(iced::Background::Color(iced::Color::from_rgb(0.2, 0.5, 0.2))),
                                text_color: iced::Color::WHITE,
                                border: iced::Border {
                                    color: iced::Color::from_rgb(0.3, 0.9, 0.3),
                                    width: 4.0,
                                    radius: 4.0.into(),
                                },
                                ..button::Style::default()
                            })
                        )
                        .style(|_| container::Style::default()
                            .background(iced::Color::from_rgb(0.1, 0.3, 0.1)))
                        .padding(2)
                        .into()
                    } else {
                        // Unselected - darker background
                        btn.style(|_theme, _status| button::Style {
                            background: Some(iced::Background::Color(iced::Color::from_rgb(0.15, 0.15, 0.2))),
                            text_color: iced::Color::WHITE,
                            border: iced::Border {
                                color: iced::Color::from_rgb(0.4, 0.4, 0.5),
                                width: 2.0,
                                radius: 4.0.into(),
                            },
                            ..button::Style::default()
                        })
                        .into()
                    }
                } else {
                    // Normal mode: preview on click
                    button(thumbnail_content)
                        .on_press(if self.config.use_internal_preview {
                            Message::ShowImagePreview(path_clone, self.aligned_images.clone())
                        } else {
                            Message::OpenImage(path_clone)
                        })
                        .style(button::secondary)
                        .width(Length::Fixed(THUMB_WIDTH))
                        .into()
                };
                
                // Wrap in mouse_area to detect right-clicks (only in normal mode)
                let final_element = if !self.aligned_selection_mode {
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
                Message::AlignedScrollChanged(offset)
            })
            .id(iced::widget::scrollable::Id::new("aligned"));

        // Create pane header with title, count, and refresh button
        let pane_header = row![
            column![
                text("Aligned")
                    .size(18)
                    .align_x(iced::Alignment::Center),
                text(format!("{} images", self.aligned_images.len()))
                    .size(12)
                    .style(|_theme| text::Style {
                        color: Some(iced::Color::from_rgb(0.7, 0.7, 0.7))
                    })
                    .align_x(iced::Alignment::Center)
            ]
            .width(Length::Fill)
            .align_x(iced::Alignment::Center),
            button("Refresh")
                .on_press(Message::RefreshAlignedPane)
                .padding(4)
                .style(|theme, status| button::Style {
                    background: Some(iced::Background::Color(iced::Color::from_rgb(0.2, 0.7, 0.2))),
                    text_color: iced::Color::WHITE,
                    border: iced::Border {
                        radius: 4.0.into(),
                        ..Default::default()
                    },
                    ..button::secondary(theme, status)
                })
        ]
        .spacing(5)
        .align_y(iced::Alignment::Center);

        // Add Cancel and Stack buttons at bottom if in selection mode
        let mut pane_content = column![
            pane_header,
            container(scrollable_widget)
                .width(Length::Fill)
                .height(Length::Fill)
                .style(|_| container::Style {
                    background: Some(iced::Background::Color(iced::Color::TRANSPARENT)),
                    border: iced::Border::default(),
                    text_color: Some(iced::Color::TRANSPARENT),
                    shadow: iced::Shadow::default(),
                })
        ]
        .spacing(10);

        if self.aligned_selection_mode {
            pane_content = pane_content
                .push(
                    row![
                        button("Select All")
                            .on_press(Message::SelectAllAligned)
                            .style(button::secondary),
                        button("Deselect All")
                            .on_press(Message::DeselectAllAligned)
                            .style(button::secondary),
                    ]
                    .spacing(10)
                    .padding(5)
                    .width(Length::Fill)
                )
                .push(
                    row![
                        button("Cancel")
                            .on_press(Message::CancelAlignedSelection)
                            .style(|theme, status| button::Style {
                                background: Some(iced::Background::Color(iced::Color::from_rgb(0.8, 0.3, 0.3))),
                                text_color: iced::Color::WHITE,
                                ..button::secondary(theme, status)
                            }),
                        button(text(format!("Stack ({} selected)", self.selected_aligned.len())))
                            .on_press(Message::StackSelectedAligned)
                            .style(|theme, status| button::Style {
                                background: Some(iced::Background::Color(iced::Color::from_rgb(0.2, 0.7, 0.2))),
                                text_color: iced::Color::WHITE,
                                ..button::secondary(theme, status)
                            }),
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
            .style(|_| container::Style::default()
                .background(iced::Color::from_rgb(0.15, 0.15, 0.2))
                .border(iced::Border {
                    color: iced::Color::from_rgb(0.5, 0.5, 0.6),
                    width: 2.0,
                    radius: 4.0.into(),
                }))
            .into()
    }
}
