//! Window management handlers
//!
//! Handlers for:
//! - ToggleSettings, ToggleHelp, CloseHelp, ToggleLog, CloseLog
//! - Exit

use std::fs;

use iced::Task;
use iced::window;

use crate::messages::Message;

use crate::gui::state::ImageStacker;

impl ImageStacker {
    /// Handle ToggleSettings
    pub fn handle_toggle_settings(&mut self) -> Task<Message> {
        self.show_settings = !self.show_settings;
        Task::none()
    }

    /// Handle ToggleHelp - convert markdown to HTML and open in browser
    pub fn handle_toggle_help(&mut self) -> Task<Message> {
        Task::perform(
            async {
                // Read markdown file
                let markdown = match fs::read_to_string("USER_MANUAL.md") {
                    Ok(content) => content,
                    Err(e) => {
                        log::error!("Failed to read USER_MANUAL.md: {}", e);
                        return Err(format!("Failed to read USER_MANUAL.md: {}", e));
                    }
                };

                // Convert markdown to HTML
                use pulldown_cmark::{Parser, html};
                let parser = Parser::new(&markdown);
                let mut html_output = String::new();
                html::push_html(&mut html_output, parser);

                // Create a complete HTML document with CSS styling
                let full_html = format!(
                    r#"<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Image Stacker - User Manual</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            line-height: 1.6;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            background-color: #1e1e1e;
            color: #d4d4d4;
        }}
        h1 {{ color: #4ec9f0; border-bottom: 2px solid #4ec9f0; padding-bottom: 10px; }}
        h2 {{ color: #66ccff; margin-top: 30px; }}
        h3 {{ color: #88ddff; margin-top: 20px; }}
        code {{
            background-color: #2d2d2d;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
        }}
        pre {{
            background-color: #2d2d2d;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
        }}
        pre code {{
            background-color: transparent;
            padding: 0;
        }}
        a {{ color: #4ec9f0; text-decoration: none; }}
        a:hover {{ text-decoration: underline; }}
        ul, ol {{ padding-left: 30px; }}
        li {{ margin: 8px 0; }}
        hr {{ border: none; border-top: 1px solid #444; margin: 30px 0; }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #444;
            padding: 10px;
            text-align: left;
        }}
        th {{ background-color: #2d2d2d; }}
    </style>
</head>
<body>
{}
</body>
</html>"#,
                    html_output
                );

                // Write to temporary HTML file
                let temp_dir = std::env::temp_dir();
                let html_path = temp_dir.join("imagestacker_manual.html");
                
                if let Err(e) = fs::write(&html_path, full_html) {
                    log::error!("Failed to write HTML file: {}", e);
                    return Err(format!("Failed to write HTML file: {}", e));
                }

                // Open in default browser
                if let Err(e) = opener::open(&html_path) {
                    log::error!("Failed to open browser: {}", e);
                    return Err(format!("Failed to open browser: {}", e));
                }

                Ok(())
            },
            |result| {
                if let Err(e) = result {
                    log::error!("Help display error: {}", e);
                }
                Message::None
            }
        )
    }

    /// Handle CloseHelp
    pub fn handle_close_help(&mut self) -> Task<Message> {
        if let Some(id) = self.help_window_id.take() {
            window::close::<Message>(id)
        } else {
            Task::none()
        }
    }

    /// Handle ToggleLog
    pub fn handle_toggle_log(&mut self) -> Task<Message> {
        if self.log_window_id.is_some() {
            // Log window already open, do nothing
            Task::none()
        } else {
            // Create and open log window
            let (id, open) = window::open(window::Settings {
                size: iced::Size::new(900.0, 600.0),
                position: window::Position::Centered,
                exit_on_close_request: false,
                ..Default::default()
            });
            self.log_window_id = Some(id);
            open.map(|_| Message::None)
        }
    }

    /// Handle CloseLog
    pub fn handle_close_log(&mut self) -> Task<Message> {
        if let Some(id) = self.log_window_id.take() {
            window::close::<Message>(id)
        } else {
            Task::none()
        }
    }

    /// Handle Exit
    pub fn handle_exit(&mut self) -> Task<Message> {
        std::process::exit(0);
    }
}
