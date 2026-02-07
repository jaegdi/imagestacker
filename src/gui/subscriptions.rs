//! Event subscriptions for the ImageStacker GUI
//!
//! This module handles all event subscriptions including keyboard,
//! mouse, window events, and periodic refresh timers.

use std::time::Duration;

use crate::messages::Message;
use super::state::ImageStacker;

impl ImageStacker {
    /// Create event subscriptions for the application
    pub fn subscription(&self) -> iced::Subscription<Message> {
        // Auto-refresh subscription during processing
        let refresh = if self.is_processing {
            iced::time::every(Duration::from_secs(2)).map(|_| Message::AutoRefreshTick)
        } else {
            iced::Subscription::none()
        };
        
        // Window resize and close subscription
        let window_events = iced::event::listen_with(|event, _status, _id| {
            match event {
                iced::Event::Window(iced::window::Event::Resized(size)) => {
                    Some(Message::WindowResized(size.width, size.height))
                }
                iced::Event::Window(iced::window::Event::CloseRequested) => {
                    Some(Message::CloseHelp)
                }
                _ => None,
            }
        });
        
        // Keyboard events for image preview navigation
        let keyboard_events = iced::event::listen_with(move |event, _status, _id| {
            if let iced::Event::Keyboard(keyboard_event) = event {
                match keyboard_event {
                    iced::keyboard::Event::KeyPressed { key, .. } => {
                        match key {
                            iced::keyboard::Key::Named(iced::keyboard::key::Named::ArrowRight) => {
                                Some(Message::NextImageInPreview)
                            }
                            iced::keyboard::Key::Named(iced::keyboard::key::Named::ArrowLeft) => {
                                Some(Message::PreviousImageInPreview)
                            }
                            iced::keyboard::Key::Named(iced::keyboard::key::Named::Escape) => {
                                // Check if we're in preview mode or if a process is running
                                // If process is running, send CancelProcessing
                                // Otherwise, close preview
                                // The update() function will handle the logic
                                Some(Message::CloseImagePreview)  // Will be handled by update() to check state
                            }
                            _ => None,
                        }
                    }
                    _ => None,
                }
            } else {
                None
            }
        });
        
        // Mouse wheel events for image preview navigation
        let mouse_events = iced::event::listen_with(move |event, _status, _id| {
            if let iced::Event::Mouse(mouse_event) = event {
                match mouse_event {
                    iced::mouse::Event::WheelScrolled { delta } => {
                        match delta {
                            iced::mouse::ScrollDelta::Lines { y, .. } => {
                                if y > 0.0 {
                                    Some(Message::PreviousImageInPreview)
                                } else if y < 0.0 {
                                    Some(Message::NextImageInPreview)
                                } else {
                                    None
                                }
                            }
                            iced::mouse::ScrollDelta::Pixels { y, .. } => {
                                if y > 0.0 {
                                    Some(Message::PreviousImageInPreview)
                                } else if y < 0.0 {
                                    Some(Message::NextImageInPreview)
                                } else {
                                    None
                                }
                            }
                        }
                    }
                    _ => None,
                }
            } else {
                None
            }
        });
        
        iced::Subscription::batch(vec![refresh, window_events, keyboard_events, mouse_events])
    }
}
