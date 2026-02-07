//! Centralized theme for the ImageStacker application
//!
//! This module provides all color constants, button styles, container styles,
//! and other visual styling functions for a consistent, professional dark theme.

#![allow(dead_code)]

use iced::widget::{button, container, text, text_input, pane_grid};
use iced::{Background, Border, Color, Shadow};

// =============================================================================
// Color Palette
// =============================================================================

/// Base background colors (dark to light)
pub mod bg {
    use super::Color;
    /// Deepest background (window/app background)
    pub const BASE: Color = Color::from_rgb(0.10, 0.10, 0.13);
    /// Surface (panels, cards)
    pub const SURFACE: Color = Color::from_rgb(0.14, 0.14, 0.18);
    /// Elevated surface (settings panes, popups)
    pub const ELEVATED: Color = Color::from_rgb(0.17, 0.17, 0.22);
    /// Toolbar / header area
    pub const TOOLBAR: Color = Color::from_rgb(0.12, 0.12, 0.16);
    /// Status bar
    pub const STATUS_BAR: Color = Color::from_rgb(0.09, 0.09, 0.12);
    /// Overlay background (preview modal)
    pub const OVERLAY: Color = Color::from_rgba(0.0, 0.0, 0.0, 0.75);
    /// Hover highlight
    pub const HOVER: Color = Color::from_rgb(0.22, 0.22, 0.28);
    /// Loading placeholder
    pub const PLACEHOLDER: Color = Color::from_rgb(0.18, 0.18, 0.22);
}

/// Text colors
pub mod txt {
    use super::Color;
    /// Primary text (bright white)
    pub const PRIMARY: Color = Color::from_rgb(0.92, 0.92, 0.95);
    /// Secondary text (labels, descriptions)
    pub const SECONDARY: Color = Color::from_rgb(0.65, 0.65, 0.72);
    /// Muted text (hints, disabled)
    pub const MUTED: Color = Color::from_rgb(0.45, 0.45, 0.52);
    /// Disabled text
    pub const DISABLED: Color = Color::from_rgb(0.38, 0.38, 0.42);
    /// Heading text
    pub const HEADING: Color = Color::from_rgb(0.85, 0.87, 0.95);
}

/// Accent / action colors
pub mod accent {
    use super::Color;
    /// Primary accent (blue)
    pub const PRIMARY: Color = Color::from_rgb(0.30, 0.52, 0.80);
    /// Primary accent hover
    pub const PRIMARY_HOVER: Color = Color::from_rgb(0.35, 0.58, 0.88);
    /// Success / confirm (green)
    pub const SUCCESS: Color = Color::from_rgb(0.25, 0.62, 0.35);
    /// Success hover
    pub const SUCCESS_HOVER: Color = Color::from_rgb(0.30, 0.70, 0.40);
    /// Danger / destructive (red)
    pub const DANGER: Color = Color::from_rgb(0.75, 0.28, 0.28);
    /// Danger hover
    pub const DANGER_HOVER: Color = Color::from_rgb(0.82, 0.32, 0.32);
    /// Warning (orange/amber)
    pub const WARNING: Color = Color::from_rgb(0.85, 0.60, 0.15);
    /// Active / selected toggle
    pub const ACTIVE: Color = Color::from_rgb(0.30, 0.55, 0.35);
    /// Active border
    pub const ACTIVE_BORDER: Color = Color::from_rgb(0.40, 0.72, 0.45);
    /// Settings button active (amber)
    pub const SETTINGS_ACTIVE: Color = Color::from_rgb(0.78, 0.52, 0.12);
}

/// Pane-specific accent colors (subtle tints for each pipeline stage)
pub mod pane {
    use super::Color;

    // Imported pane (blue tint)
    pub const IMPORTED_BG: Color = Color::from_rgb(0.12, 0.15, 0.22);
    pub const IMPORTED_BORDER: Color = Color::from_rgb(0.28, 0.40, 0.60);

    // Sharpness pane (teal tint)
    pub const SHARPNESS_BG: Color = Color::from_rgb(0.10, 0.17, 0.19);
    pub const SHARPNESS_BORDER: Color = Color::from_rgb(0.22, 0.48, 0.52);

    // Aligned pane (purple tint)
    pub const ALIGNED_BG: Color = Color::from_rgb(0.15, 0.13, 0.24);
    pub const ALIGNED_BORDER: Color = Color::from_rgb(0.38, 0.34, 0.62);

    // Bunches pane (warm/brown tint)
    pub const BUNCHES_BG: Color = Color::from_rgb(0.18, 0.14, 0.10);
    pub const BUNCHES_BORDER: Color = Color::from_rgb(0.52, 0.38, 0.22);

    // Final pane (neutral)
    pub const FINAL_BG: Color = Color::from_rgb(0.14, 0.14, 0.18);
    pub const FINAL_BORDER: Color = Color::from_rgb(0.35, 0.35, 0.42);
}

/// Border colors
pub mod border {
    use super::Color;
    /// Subtle border (separators, groups)
    pub const SUBTLE: Color = Color::from_rgb(0.28, 0.28, 0.34);
    /// Normal border (cards, inputs)
    pub const NORMAL: Color = Color::from_rgb(0.35, 0.35, 0.42);
    /// Strong border (focused elements)
    pub const STRONG: Color = Color::from_rgb(0.50, 0.50, 0.58);
    /// Selection border (selected thumbnails)
    pub const SELECTION: Color = Color::from_rgb(0.35, 0.85, 0.40);
}

// =============================================================================
// Border Helpers
// =============================================================================

pub fn border_subtle() -> Border {
    Border {
        color: border::SUBTLE,
        width: 1.0,
        radius: 4.0.into(),
    }
}

pub fn border_normal() -> Border {
    Border {
        color: border::NORMAL,
        width: 1.0,
        radius: 4.0.into(),
    }
}

pub fn border_pane(color: Color) -> Border {
    Border {
        color,
        width: 1.5,
        radius: 6.0.into(),
    }
}

// =============================================================================
// Button Styles
// =============================================================================

/// Standard toolbar button (neutral dark)
pub fn toolbar_button(_theme: &iced::Theme, status: button::Status) -> button::Style {
    let (bg_color, border_color) = match status {
        button::Status::Hovered => (bg::HOVER, border::NORMAL),
        button::Status::Pressed => (accent::PRIMARY, accent::PRIMARY),
        _ => (bg::SURFACE, border::SUBTLE),
    };
    button::Style {
        background: Some(Background::Color(bg_color)),
        text_color: txt::PRIMARY,
        border: Border {
            color: border_color,
            width: 1.0,
            radius: 4.0.into(),
        },
        shadow: Shadow::default(),
    }
}

/// Disabled button style
pub fn button_disabled(_theme: &iced::Theme, _status: button::Status) -> button::Style {
    button::Style {
        background: Some(Background::Color(Color::from_rgb(0.20, 0.20, 0.24))),
        text_color: txt::DISABLED,
        border: Border {
            color: border::SUBTLE,
            width: 1.0,
            radius: 4.0.into(),
        },
        shadow: Shadow::default(),
    }
}

/// Action button with a specific accent color
pub fn action_button(
    accent_bg: Color,
    accent_border: Color,
) -> impl Fn(&iced::Theme, button::Status) -> button::Style {
    move |_theme: &iced::Theme, status: button::Status| {
        let bg_color = match status {
            button::Status::Hovered => {
                // Lighten slightly on hover
                Color::from_rgb(
                    (accent_bg.r + 0.06).min(1.0),
                    (accent_bg.g + 0.06).min(1.0),
                    (accent_bg.b + 0.06).min(1.0),
                )
            }
            button::Status::Pressed => {
                Color::from_rgb(
                    (accent_bg.r + 0.12).min(1.0),
                    (accent_bg.g + 0.12).min(1.0),
                    (accent_bg.b + 0.12).min(1.0),
                )
            }
            _ => accent_bg,
        };
        button::Style {
            background: Some(Background::Color(bg_color)),
            text_color: txt::PRIMARY,
            border: Border {
                color: accent_border,
                width: 1.0,
                radius: 4.0.into(),
            },
            shadow: Shadow::default(),
        }
    }
}

/// "Stack Imported" button style
pub fn stack_imported_button(_theme: &iced::Theme, status: button::Status) -> button::Style {
    action_button(
        pane::IMPORTED_BG,
        pane::IMPORTED_BORDER,
    )(_theme, status)
}

/// "Stack Sharpness" button style
pub fn stack_sharpness_button(_theme: &iced::Theme, status: button::Status) -> button::Style {
    action_button(
        pane::SHARPNESS_BG,
        pane::SHARPNESS_BORDER,
    )(_theme, status)
}

/// "Stack Aligned" button style
pub fn stack_aligned_button(_theme: &iced::Theme, status: button::Status) -> button::Style {
    action_button(
        pane::ALIGNED_BG,
        pane::ALIGNED_BORDER,
    )(_theme, status)
}

/// "Stack Bunches" button style
pub fn stack_bunches_button(_theme: &iced::Theme, status: button::Status) -> button::Style {
    action_button(
        pane::BUNCHES_BG,
        pane::BUNCHES_BORDER,
    )(_theme, status)
}

/// Danger button (Cancel, Delete)
pub fn danger_button(_theme: &iced::Theme, status: button::Status) -> button::Style {
    action_button(accent::DANGER, accent::DANGER)(_theme, status)
}

/// Success button (Stack, Confirm)
pub fn success_button(_theme: &iced::Theme, status: button::Status) -> button::Style {
    action_button(accent::SUCCESS, accent::SUCCESS)(_theme, status)
}

/// Active toggle button (selected feature detector, active settings)
pub fn toggle_active_button(_theme: &iced::Theme, status: button::Status) -> button::Style {
    action_button(accent::ACTIVE, accent::ACTIVE_BORDER)(_theme, status)
}

/// Settings button when active (amber glow)
pub fn settings_active_button(_theme: &iced::Theme, status: button::Status) -> button::Style {
    let bg_color = match status {
        button::Status::Hovered => Color::from_rgb(0.85, 0.58, 0.15),
        _ => accent::SETTINGS_ACTIVE,
    };
    button::Style {
        background: Some(Background::Color(bg_color)),
        text_color: txt::PRIMARY,
        border: Border {
            color: Color::from_rgb(0.92, 0.65, 0.18),
            width: 1.0,
            radius: 4.0.into(),
        },
        shadow: Shadow::default(),
    }
}

/// Refresh button (small green)
pub fn refresh_button(_theme: &iced::Theme, status: button::Status) -> button::Style {
    let bg_color = match status {
        button::Status::Hovered => accent::SUCCESS_HOVER,
        _ => accent::SUCCESS,
    };
    button::Style {
        background: Some(Background::Color(bg_color)),
        text_color: txt::PRIMARY,
        border: Border {
            color: accent::SUCCESS,
            width: 1.0,
            radius: 4.0.into(),
        },
        shadow: Shadow::default(),
    }
}

/// Thumbnail button (selected — green highlight)
pub fn thumb_selected(_theme: &iced::Theme, _status: button::Status) -> button::Style {
    button::Style {
        background: Some(Background::Color(Color::from_rgb(0.15, 0.38, 0.18))),
        text_color: txt::PRIMARY,
        border: Border {
            color: border::SELECTION,
            width: 3.0,
            radius: 4.0.into(),
        },
        shadow: Shadow::default(),
    }
}

/// Thumbnail button (unselected in selection mode)
pub fn thumb_unselected(_theme: &iced::Theme, _status: button::Status) -> button::Style {
    button::Style {
        background: Some(Background::Color(Color::from_rgb(0.14, 0.14, 0.18))),
        text_color: txt::PRIMARY,
        border: Border {
            color: border::SUBTLE,
            width: 1.5,
            radius: 4.0.into(),
        },
        shadow: Shadow::default(),
    }
}

/// ECC motion type toggle button
pub fn ecc_toggle_button(selected: bool) -> impl Fn(&iced::Theme, button::Status) -> button::Style {
    move |_theme: &iced::Theme, status: button::Status| {
        if selected {
            let bg_color = match status {
                button::Status::Hovered => Color::from_rgb(0.35, 0.55, 0.68),
                _ => Color::from_rgb(0.28, 0.45, 0.58),
            };
            button::Style {
                background: Some(Background::Color(bg_color)),
                text_color: txt::PRIMARY,
                border: Border {
                    color: Color::from_rgb(0.40, 0.65, 0.78),
                    width: 2.0,
                    radius: 4.0.into(),
                },
                shadow: Shadow::default(),
            }
        } else {
            toolbar_button(_theme, status)
        }
    }
}

/// Reset button (destructive)
pub fn reset_button(_theme: &iced::Theme, status: button::Status) -> button::Style {
    let bg_color = match status {
        button::Status::Hovered => Color::from_rgb(0.68, 0.28, 0.28),
        _ => Color::from_rgb(0.55, 0.25, 0.25),
    };
    button::Style {
        background: Some(Background::Color(bg_color)),
        text_color: txt::PRIMARY,
        border: Border {
            color: Color::from_rgb(0.72, 0.35, 0.35),
            width: 1.0,
            radius: 6.0.into(),
        },
        shadow: Shadow::default(),
    }
}

// =============================================================================
// Container Styles
// =============================================================================

/// Main pane container style (with pane-specific colors)
pub fn pane_container(bg_color: Color, border_color: Color) -> container::Style {
    container::Style {
        background: Some(Background::Color(bg_color)),
        border: border_pane(border_color),
        text_color: None,
        shadow: Shadow {
            color: Color::from_rgba(0.0, 0.0, 0.0, 0.3),
            offset: iced::Vector::new(0.0, 2.0),
            blur_radius: 6.0,
        },
    }
}

/// Toolbar container
pub fn toolbar_container(_theme: &iced::Theme) -> container::Style {
    container::Style {
        background: Some(Background::Color(bg::TOOLBAR)),
        border: Border {
            color: border::SUBTLE,
            width: 0.0,
            radius: 0.0.into(),
        },
        text_color: None,
        shadow: Shadow {
            color: Color::from_rgba(0.0, 0.0, 0.0, 0.2),
            offset: iced::Vector::new(0.0, 2.0),
            blur_radius: 4.0,
        },
    }
}

/// Button group container (grouped toolbar buttons)
pub fn button_group(_theme: &iced::Theme) -> container::Style {
    container::Style {
        background: Some(Background::Color(Color::from_rgba(0.0, 0.0, 0.0, 0.15))),
        border: Border {
            color: border::SUBTLE,
            width: 1.0,
            radius: 6.0.into(),
        },
        text_color: None,
        shadow: Shadow::default(),
    }
}

/// Settings panel outer container
pub fn settings_panel(_theme: &iced::Theme) -> container::Style {
    container::Style {
        background: Some(Background::Color(bg::SURFACE)),
        border: Border {
            color: border::SUBTLE,
            width: 0.0,
            radius: 0.0.into(),
        },
        text_color: None,
        shadow: Shadow::default(),
    }
}

/// Settings section pane (individual settings groups)
pub fn settings_section(_theme: &iced::Theme) -> container::Style {
    container::Style {
        background: Some(Background::Color(bg::ELEVATED)),
        border: Border {
            color: border::SUBTLE,
            width: 1.0,
            radius: 8.0.into(),
        },
        text_color: None,
        shadow: Shadow::default(),
    }
}

/// Progress bar container
pub fn progress_container(_theme: &iced::Theme) -> container::Style {
    container::Style {
        background: Some(Background::Color(Color::from_rgb(0.12, 0.20, 0.30))),
        border: Border {
            color: accent::PRIMARY,
            width: 1.0,
            radius: 4.0.into(),
        },
        text_color: None,
        shadow: Shadow::default(),
    }
}

/// Status bar container
pub fn status_bar(_theme: &iced::Theme) -> container::Style {
    container::Style {
        background: Some(Background::Color(bg::STATUS_BAR)),
        border: Border {
            color: border::SUBTLE,
            width: 0.0,
            radius: 0.0.into(),
        },
        text_color: None,
        shadow: Shadow::default(),
    }
}

/// Preview overlay container (dark background)
pub fn preview_overlay(_theme: &iced::Theme) -> container::Style {
    container::Style {
        background: Some(Background::Color(Color::from_rgb(0.12, 0.12, 0.15))),
        border: Border {
            color: border::NORMAL,
            width: 2.0,
            radius: 8.0.into(),
        },
        text_color: None,
        shadow: Shadow {
            color: Color::from_rgba(0.0, 0.0, 0.0, 0.5),
            offset: iced::Vector::new(0.0, 4.0),
            blur_radius: 16.0,
        },
    }
}

/// Sharpness overlay badge on thumbnails
pub fn sharpness_badge(_theme: &iced::Theme) -> container::Style {
    container::Style {
        background: Some(Background::Color(Color::from_rgba(0.0, 0.0, 0.0, 0.72))),
        border: Border {
            radius: 4.0.into(),
            ..Default::default()
        },
        text_color: None,
        shadow: Shadow::default(),
    }
}

/// Selected thumbnail container wrapper
pub fn thumb_selected_container(_theme: &iced::Theme) -> container::Style {
    container::Style {
        background: Some(Background::Color(Color::from_rgb(0.08, 0.25, 0.10))),
        border: Border::default(),
        text_color: None,
        shadow: Shadow::default(),
    }
}

/// Transparent scrollable inner container
pub fn scrollable_inner(_theme: &iced::Theme) -> container::Style {
    container::Style {
        background: None,
        border: Border::default(),
        text_color: None,
        shadow: Shadow::default(),
    }
}

// =============================================================================
// Text Styles
// =============================================================================

/// Section heading text
pub fn heading_text(_theme: &iced::Theme) -> text::Style {
    text::Style {
        color: Some(txt::HEADING),
    }
}

/// Secondary / subtitle text
pub fn secondary_text(_theme: &iced::Theme) -> text::Style {
    text::Style {
        color: Some(txt::SECONDARY),
    }
}

/// Muted / hint text
pub fn muted_text(_theme: &iced::Theme) -> text::Style {
    text::Style {
        color: Some(txt::MUTED),
    }
}

/// Warning text (amber)
pub fn warning_text(_theme: &iced::Theme) -> text::Style {
    text::Style {
        color: Some(accent::WARNING),
    }
}

/// Success text (green)
pub fn success_text(_theme: &iced::Theme) -> text::Style {
    text::Style {
        color: Some(Color::from_rgb(0.45, 0.72, 0.50)),
    }
}

/// Env var name text
pub fn env_var_text(_theme: &iced::Theme) -> text::Style {
    text::Style {
        color: Some(Color::from_rgb(0.55, 0.78, 0.55)),
    }
}

/// Section label text (blue-tinted heading)
pub fn section_label(_theme: &iced::Theme) -> text::Style {
    text::Style {
        color: Some(Color::from_rgb(0.65, 0.75, 0.95)),
    }
}

/// White text (for overlays)
pub fn white_text(_theme: &iced::Theme) -> text::Style {
    text::Style {
        color: Some(Color::WHITE),
    }
}

// =============================================================================
// Text Input Styles
// =============================================================================

/// Status bar text input style
pub fn status_bar_input(_theme: &iced::Theme, _status: text_input::Status) -> text_input::Style {
    text_input::Style {
        background: Background::Color(bg::STATUS_BAR),
        border: Border {
            color: border::SUBTLE,
            width: 0.0,
            radius: 0.0.into(),
        },
        icon: txt::SECONDARY,
        placeholder: txt::MUTED,
        value: txt::PRIMARY,
        selection: accent::PRIMARY,
    }
}

/// Settings text input style
pub fn settings_input(_theme: &iced::Theme, _status: text_input::Status) -> text_input::Style {
    text_input::Style {
        background: Background::Color(Color::from_rgb(0.12, 0.12, 0.16)),
        border: Border {
            color: border::NORMAL,
            width: 1.0,
            radius: 4.0.into(),
        },
        icon: txt::SECONDARY,
        placeholder: txt::MUTED,
        value: txt::PRIMARY,
        selection: accent::PRIMARY,
    }
}

// =============================================================================
// PaneGrid Styles
// =============================================================================

/// PaneGrid style — split handle highlights
pub fn pane_grid_style(_theme: &iced::Theme) -> pane_grid::Style {
    pane_grid::Style {
        hovered_region: pane_grid::Highlight {
            background: Background::Color(Color::from_rgba(0.30, 0.52, 0.80, 0.15)),
            border: Border {
                color: accent::PRIMARY,
                width: 2.0,
                radius: 6.0.into(),
            },
        },
        picked_split: pane_grid::Line {
            color: accent::PRIMARY,
            width: 3.0,
        },
        hovered_split: pane_grid::Line {
            color: Color::from_rgb(0.45, 0.65, 0.90),
            width: 3.0,
        },
    }
}

/// Title bar for a pane in the PaneGrid
pub fn pane_title_bar(bg_color: Color) -> impl Fn(&iced::Theme) -> container::Style {
    move |_theme: &iced::Theme| {
        container::Style {
            background: Some(Background::Color(Color::from_rgb(
                (bg_color.r + 0.04).min(1.0),
                (bg_color.g + 0.04).min(1.0),
                (bg_color.b + 0.04).min(1.0),
            ))),
            border: Border {
                color: border::SUBTLE,
                width: 0.0,
                radius: 0.0.into(),
            },
            text_color: None,
            shadow: Shadow::default(),
        }
    }
}
