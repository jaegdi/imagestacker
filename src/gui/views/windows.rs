//! Window rendering (Help and Log windows)
//!
//! This module contains rendering for secondary windows like Help and Log.

use std::fs;
use iced::widget::{
    button, column, container, scrollable, text,
};
use iced::{Element, Length};

use crate::messages::Message;
use super::super::state::ImageStacker;
use super::super::log_capture::get_logs;

impl ImageStacker {
    pub(crate) fn render_help_window(&self) -> Element<'_, Message> {
        // Load help text from markdown file
        // Try multiple locations: current dir, installation dir, and source dir
        let help_paths = [
            "USER_MANUAL.md",                                    // Current directory (development)
            "/usr/share/doc/imagestacker/USER_MANUAL.md",       // System installation (Linux)
            "/usr/local/share/doc/imagestacker/USER_MANUAL.md", // Local installation
        ];
        
        let mut errors = Vec::new();
        let help_markdown = help_paths.iter()
            .find_map(|path| {
                log::info!("Trying to load USER_MANUAL from: {}", path);
                match fs::read_to_string(path) {
                    Ok(content) => {
                        log::info!("✓ Successfully loaded USER_MANUAL from: {} ({} bytes)", path, content.len());
                        Some(content)
                    }
                    Err(e) => {
                        let error_msg = format!("{}: {}", path, e);
                        log::warn!("✗ Failed to load USER_MANUAL from {}: {}", path, e);
                        errors.push(error_msg);
                        None
                    }
                }
            })
            .unwrap_or_else(|| {
                let error_details = errors.join("\n- ");
                let error_msg = format!(
                    "# Error Loading Manual\n\n\
                    Could not load USER_MANUAL.md file from any location.\n\n\
                    ## Tried locations:\n{}\n\n\
                    ## Errors:\n- {}\n\n\
                    ## Troubleshooting:\n\
                    - Check if the package is properly installed\n\
                    - Verify file exists: `ls -la /usr/share/doc/imagestacker/USER_MANUAL.md`\n\
                    - Check file permissions: `test -r /usr/share/doc/imagestacker/USER_MANUAL.md && echo OK`",
                    help_paths.iter().map(|p| format!("- {}", p)).collect::<Vec<_>>().join("\n"),
                    error_details
                );
                log::error!("Failed to load USER_MANUAL.md from any location! Errors: {:?}", errors);
                error_msg
            });
        
        // Parse markdown and create formatted text elements
        let mut elements: Vec<Element<Message>> = Vec::new();
        
        for line in help_markdown.lines() {
            let trimmed = line.trim();
            
            if trimmed.is_empty() {
                // Empty line - add small spacing
                elements.push(container(text("")).height(Length::Fixed(8.0)).into());
            } else if trimmed.starts_with("# ") {
                // H1 heading
                elements.push(
                    text(trimmed.trim_start_matches("# ").to_string())
                        .size(24)
                        .style(|_| text::Style { 
                            color: Some(iced::Color::from_rgb(0.3, 0.7, 1.0)) 
                        })
                        .into()
                );
            } else if trimmed.starts_with("## ") {
                // H2 heading
                elements.push(
                    text(trimmed.trim_start_matches("## ").to_string())
                        .size(20)
                        .style(|_| text::Style { 
                            color: Some(iced::Color::from_rgb(0.4, 0.8, 1.0)) 
                        })
                        .into()
                );
            } else if trimmed.starts_with("### ") {
                // H3 heading
                elements.push(
                    text(trimmed.trim_start_matches("### ").to_string())
                        .size(16)
                        .style(|_| text::Style { 
                            color: Some(iced::Color::from_rgb(0.5, 0.85, 1.0)) 
                        })
                        .into()
                );
            } else if trimmed.starts_with("---") || trimmed.starts_with("===") {
                // Horizontal rule
                elements.push(
                    container(text(""))
                        .width(Length::Fill)
                        .height(Length::Fixed(1.0))
                        .style(|_| container::Style::default()
                            .background(iced::Color::from_rgb(0.5, 0.5, 0.5)))
                        .into()
                );
            } else if trimmed.starts_with("- ") || trimmed.starts_with("* ") {
                // Bullet list
                let content = trimmed.trim_start_matches("- ").trim_start_matches("* ").to_string();
                elements.push(
                    container(
                        text(format!("  • {}", content))
                            .size(13)
                            .style(|_| text::Style { 
                                color: Some(iced::Color::from_rgb(0.9, 0.9, 0.9)) 
                            })
                    )
                    .padding(iced::Padding::new(0.0).left(10.0))
                    .into()
                );
            } else if trimmed.starts_with("✓ ") {
                // Checkmark list
                elements.push(
                    container(
                        text(trimmed.to_string())
                            .size(13)
                            .style(|_| text::Style { 
                                color: Some(iced::Color::from_rgb(0.4, 1.0, 0.4)) 
                            })
                    )
                    .padding(iced::Padding::new(0.0).left(10.0))
                    .into()
                );
            } else if trimmed.starts_with("```") {
                // Code block marker - skip
                continue;
            } else if trimmed.chars().next().map_or(false, |c| c.is_numeric()) 
                      && trimmed.chars().nth(1) == Some('.') {
                // Numbered list
                elements.push(
                    container(
                        text(format!("  {}", trimmed))
                            .size(13)
                            .style(|_| text::Style { 
                                color: Some(iced::Color::from_rgb(0.9, 0.9, 0.9)) 
                            })
                    )
                    .padding(iced::Padding::new(0.0).left(10.0))
                    .into()
                );
            } else if trimmed.starts_with("**") && trimmed.ends_with("**") {
                // Bold text as a paragraph
                let content = trimmed.trim_start_matches("**").trim_end_matches("**").to_string();
                elements.push(
                    text(content)
                        .size(14)
                        .style(|_| text::Style { 
                            color: Some(iced::Color::from_rgb(1.0, 1.0, 1.0)) 
                        })
                        .into()
                );
            } else {
                // Regular paragraph
                elements.push(
                    text(trimmed.to_string())
                        .size(13)
                        .style(|_| text::Style { 
                            color: Some(iced::Color::from_rgb(0.85, 0.85, 0.85)) 
                        })
                        .into()
                );
            }
        }
        
        let help_content = scrollable(
            column(elements)
                .spacing(4)
                .padding(20)
        )
        .height(Length::Fill);

        let close_button = container(
            button("Close")
                .on_press(Message::CloseHelp)
                .padding(10)
                .style(|_theme, _status| button::Style {
                    background: Some(iced::Background::Color(iced::Color::from_rgb(0.8, 0.3, 0.3))),
                    text_color: iced::Color::WHITE,
                    border: iced::Border {
                        radius: 4.0.into(),
                        ..Default::default()
                    },
                    ..button::Style::default()
                })
        )
        .padding(15)
        .width(Length::Fill)
        .center_x(Length::Fill);

        container(
            column![
                help_content,
                close_button,
            ]
            .spacing(0)
        )
        .padding(0)
        .width(Length::Fill)
        .height(Length::Fill)
        .style(|_| container::Style::default()
            .background(iced::Color::from_rgb(0.15, 0.15, 0.2)))
        .into()
    }

    pub(crate) fn render_log_window(&self) -> Element<'_, Message> {
        // Get captured log messages
        let logs = get_logs();
        
        let log_content = if logs.is_empty() {
            "No log messages yet.\n\nLog messages will appear here when the application performs operations.".to_string()
        } else {
            logs.join("\n")
        };
        
        let log_text = scrollable(
            container(
                text(log_content)
                    .size(12)
                    .font(iced::Font::MONOSPACE)
                    .style(|_| text::Style { 
                        color: Some(iced::Color::from_rgb(0.9, 0.9, 0.9)) 
                    })
            )
            .padding(20)
            .width(Length::Fill)
        )
        .height(Length::Fill);

        let close_button = container(
            button("Close")
                .on_press(Message::CloseLog)
                .padding(10)
                .style(|_theme, _status| button::Style {
                    background: Some(iced::Background::Color(iced::Color::from_rgb(0.8, 0.3, 0.3))),
                    text_color: iced::Color::WHITE,
                    border: iced::Border {
                        radius: 4.0.into(),
                        ..Default::default()
                    },
                    ..button::Style::default()
                })
        )
        .padding(15)
        .width(Length::Fill)
        .center_x(Length::Fill);

        container(
            column![
                log_text,
                close_button,
            ]
            .spacing(0)
        )
        .padding(0)
        .width(Length::Fill)
        .height(Length::Fill)
        .style(|_| container::Style::default()
            .background(iced::Color::from_rgb(0.1, 0.1, 0.15)))
        .into()
    }
}