//! Sharpness pane rendering with selection mode support
//!
//! This module handles rendering the sharpness analysis pane showing:
//! - Thumbnails of images that have been analyzed for sharpness
//! - Sharpness scores overlaid on thumbnails
//! - Visual indicators of image quality
//! - Selection mode for choosing images to stack

use iced::widget::{
    button, column, container, image as iced_image, mouse_area, row, scrollable, text,
};
use iced::{Element, Length};

use crate::messages::Message;
use crate::gui::state::ImageStacker;
use crate::sharpness_cache::SharpnessInfo;

impl ImageStacker {
    pub(crate) fn render_sharpness_pane(&self) -> Element<'_, Message> {
        const THUMB_WIDTH: f32 = 120.0;
        const THUMB_HEIGHT: f32 = 90.0;
        const THUMB_SPACING: f32 = 8.0;
        let thumbs_per_row = 2;

        let mut rows_vec: Vec<Element<Message>> = Vec::new();
        
        // For each YAML file in sharpness_images, find the corresponding image and display it
        // with sharpness info overlaid
        let yaml_paths = &self.sharpness_images;
        
        // Group YAML files into chunks for rows
        for chunk in yaml_paths.chunks(thumbs_per_row) {
            let mut row_elements: Vec<Element<Message>> = Vec::new();
            
            for yaml_path in chunk {
                // Load sharpness info from YAML
                let sharpness_info = SharpnessInfo::load_from_file(yaml_path).ok();
                
                // Find the corresponding image path from the original images
                let image_path = if let Some(info) = &sharpness_info {
                    // Try to find the image with matching filename
                    self.images.iter().find(|p| {
                        p.file_name()
                            .and_then(|n| n.to_str())
                            .map(|n| n == info.image_filename)
                            .unwrap_or(false)
                    })
                } else {
                    None
                };
                
                // Create thumbnail element
                let thumb_element: Element<Message> = if let Some(img_path) = image_path {
                    let img_path_clone = img_path.clone();
                    let img_path_for_right_click = img_path.clone();
                    let is_selected = self.selected_sharpness.contains(img_path);
                    
                    // Try to get thumbnail from cache
                    let thumb_handle = {
                        let cache = self.thumbnail_cache.read().unwrap();
                        cache.get(img_path).cloned()
                    };
                    
                    let img_widget = if let Some(handle) = thumb_handle {
                        iced_image(handle)
                            .width(THUMB_WIDTH)
                            .height(THUMB_HEIGHT)
                            .content_fit(iced::ContentFit::Contain)
                    } else {
                        // Generate thumbnail if not in cache
                        match crate::thumbnail::generate_thumbnail(img_path) {
                            Ok(handle) => {
                                let mut cache = self.thumbnail_cache.write().unwrap();
                                cache.insert(img_path.clone(), handle.clone());
                                iced_image(handle)
                                    .width(THUMB_WIDTH)
                                    .height(THUMB_HEIGHT)
                                    .content_fit(iced::ContentFit::Contain)
                            }
                            Err(_) => {
                                // Fallback: show placeholder
                                iced_image(iced::widget::image::Handle::from_rgba(1, 1, vec![255, 0, 0, 255]))
                                    .width(THUMB_WIDTH)
                                    .height(THUMB_HEIGHT)
                                    .content_fit(iced::ContentFit::Contain)
                            }
                        }
                    };
                    
                    // Create overlay with sharpness score directly on the image
                    let sharpness_overlay = if let Some(info) = &sharpness_info {
                        container(
                            text(format!("Max: {:.1}", info.max_regional_sharpness))
                                .size(11)
                                .style(|_| text::Style {
                                    color: Some(iced::Color::WHITE)
                                })
                        )
                        .padding(3)
                        .style(|_theme: &iced::Theme| {
                            container::Style {
                                background: Some(iced::Background::Color(iced::Color::from_rgba(0.0, 0.0, 0.0, 0.7))),
                                border: iced::Border {
                                    radius: 3.0.into(),
                                    ..Default::default()
                                },
                                ..Default::default()
                            }
                        })
                    } else {
                        container(text(""))
                    };
                    
                    let filename = img_path
                        .file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or("?");
                    
                    // Stack image with overlay at top-right corner
                    let image_with_overlay = iced::widget::stack![
                        img_widget,
                        container(sharpness_overlay)
                            .width(Length::Fill)
                            .height(Length::Fill)
                            .align_x(iced::alignment::Horizontal::Right)
                            .align_y(iced::alignment::Vertical::Top)
                            .padding(4)
                    ];
                    
                    let label = text(filename).size(9);
                    
                    let thumbnail_content = column![
                        image_with_overlay,
                        container(label)
                            .width(Length::Fixed(THUMB_WIDTH))
                            .center_x(Length::Fill)
                    ]
                    .align_x(iced::Alignment::Center);
                    
                    // In selection mode or normal mode
                    let clickable_thumb = if self.sharpness_selection_mode {
                        // In selection mode: make clickable for selection
                        let btn = button(thumbnail_content)
                            .on_press(Message::ToggleSharpnessImage(img_path_clone))
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
                                Message::ShowImagePreview(img_path_clone, self.images.clone())
                            } else {
                                Message::OpenImage(img_path_clone)
                            })
                            .style(button::secondary)
                            .width(Length::Fixed(THUMB_WIDTH))
                            .into()
                    };
                    
                    // Wrap in mouse_area to detect right-clicks (only in normal mode)
                    let final_element = if !self.sharpness_selection_mode {
                        mouse_area(clickable_thumb)
                            .on_right_press(Message::OpenImageWithExternalEditor(img_path_for_right_click))
                            .into()
                    } else {
                        clickable_thumb
                    };
                    
                    final_element
                } else {
                    // No image found, just show YAML filename
                    let yaml_name = yaml_path
                        .file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or("?");
                    
                    column![
                        text(yaml_name).size(10),
                        text("Image not found").size(10),
                    ]
                    .spacing(4)
                    .align_x(iced::Alignment::Center)
                    .width(THUMB_WIDTH)
                    .into()
                };
                
                row_elements.push(thumb_element);
            }
            
            // Add row with proper spacing
            let row = row(row_elements)
                .spacing(THUMB_SPACING)
                .align_y(iced::Alignment::Start);
            rows_vec.push(row.into());
        }
        
        let content = column(rows_vec)
            .spacing(THUMB_SPACING)
            .align_x(iced::Alignment::Center)
            .height(Length::Shrink);

        let scrollable_widget = scrollable(content)
            .id(iced::widget::scrollable::Id::new("sharpness"))
            .on_scroll(|viewport| {
                Message::SharpnessScrollChanged(viewport.absolute_offset().y)
            })
            .width(Length::Fill);

        // Create pane header with title, count, and refresh button
        let count = yaml_paths.len();
        let pane_header = row![
            column![
                text("Sharpness")
                    .size(18)
                    .align_x(iced::Alignment::Center),
                text(format!("{} images", count))
                    .size(12)
                    .style(|_theme| text::Style {
                        color: Some(iced::Color::from_rgb(0.7, 0.7, 0.7))
                    })
                    .align_x(iced::Alignment::Center)
            ]
            .width(Length::Fill)
            .align_x(iced::Alignment::Center),
            button("Refresh")
                .on_press(Message::RefreshSharpnessPane)
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

        if self.sharpness_selection_mode {
            pane_content = pane_content
                .push(
                    row![
                        button("Select All")
                            .on_press(Message::SelectAllSharpness)
                            .style(button::secondary),
                        button("Deselect All")
                            .on_press(Message::DeselectAllSharpness)
                            .style(button::secondary),
                    ]
                    .spacing(10)
                    .padding(5)
                    .width(Length::Fill)
                )
                .push(
                    row![
                        button("Cancel")
                            .on_press(Message::CancelSharpnessSelection)
                            .style(|theme, status| button::Style {
                                background: Some(iced::Background::Color(iced::Color::from_rgb(0.8, 0.3, 0.3))),
                                text_color: iced::Color::WHITE,
                                ..button::secondary(theme, status)
                            }),
                        button(text(format!("Stack ({} selected)", self.selected_sharpness.len())))
                            .on_press(Message::StackSelectedSharpness)
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
