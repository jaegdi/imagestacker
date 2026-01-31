# GUI Module Structure

The `gui.rs` file contains the main GUI logic for ImageStacker. While it is currently a single file (~2800 lines), it has been logically organized into clear sections for better maintainability.

## File Structure

```
src/gui.rs (2,804 lines)
├── Imports & Dependencies
├── STATE MANAGEMENT (~100 lines)
│   ├── ImageStacker struct definition
│   └── Default implementation
├── MESSAGE HANDLING (~1,160 lines)
│   ├── create_processing_config()
│   └── update() - Main message handler
│       ├── File operations (AddImages, AddFolder, LoadFolder)
│       ├── Image processing (AlignImages, StackImages, etc.)
│       ├── Preview management (ShowImagePreview, LoadFullImage, etc.)
│       ├── Navigation (NextImage, PreviousImage)
│       ├── Settings (ToggleSettings, config changes)
│       ├── Help window (ToggleHelp, CloseHelp)
│       └── Scroll tracking
├── EVENT SUBSCRIPTIONS (~90 lines)
│   ├── Auto-refresh timer
│   ├── Window events (resize, close)
│   ├── Keyboard events (arrow keys, ESC)
│   └── Mouse events (wheel scrolling)
├── VIEW RENDERING (~1,450 lines)
│   ├── Main View
│   │   ├── Top button bar
│   │   ├── Four panes (Imported, Aligned, Bunches, Final)
│   │   ├── Progress bar
│   │   └── Status text
│   ├── Image Preview Modal
│   │   ├── Preview window with navigation
│   │   ├── Navigation buttons
│   │   └── Keyboard/mouse wheel support
│   ├── Settings Panel
│   │   ├── Responsive layout (horizontal/vertical)
│   │   ├── Three sections: Alignment, Post-Processing, Preview
│   │   └── Reset to defaults button
│   ├── Help Window
│   │   ├── Loads USER_MANUAL.md
│   │   ├── Markdown to HTML conversion
│   │   └── Scrollable content
│   ├── Aligned Pane (with selection mode)
│   ├── Bunches Pane (with selection mode)
│   └── Generic Pane Rendering (Imported, Final)
└── WINDOW CONFIGURATION (~15 lines)
    ├── theme() - Dark theme
    └── title() - Window titles

```

## Section Markers

The code uses clear section markers for navigation:

```rust
// ============================================================================
// SECTION NAME
// ============================================================================

// ------------------------------------------------------------------------
// Subsection Name
// ------------------------------------------------------------------------
```

## Key Components

### State Management
- **ImageStacker struct**: Main application state
- Holds all images, thumbnails, configuration, and UI state
- Uses Arc<RwLock<>> for thread-safe thumbnail cache

### Message Handling (Update)
- **~60 different message types** handled
- Asynchronous operations using `Task<Message>`
- File I/O, image processing, and UI updates
- Navigation throttling (50ms delay between image switches)

### View Rendering
- **Responsive layout**: Adapts to window size
- **Multiple windows**: Main window + help window
- **Image preview**: Modal with navigation support
- **Settings panel**: Three organized sections
- **Panes**: Four image display areas with thumbnails

### Event Subscriptions
- **Keyboard**: Arrow keys for navigation, ESC to close
- **Mouse**: Wheel scrolling for image navigation
- **Window**: Resize events for responsive layout
- **Timer**: Auto-refresh during processing

## Future Refactoring Considerations

If the file grows beyond ~3000 lines, consider splitting into:

1. **gui/mod.rs** - Main struct and trait implementations
2. **gui/update.rs** - Message handling logic
3. **gui/view.rs** - Main view rendering
4. **gui/panes.rs** - Pane-specific rendering (aligned, bunches, etc.)
5. **gui/settings.rs** - Settings panel rendering
6. **gui/preview.rs** - Image preview modal
7. **gui/subscription.rs** - Event subscriptions

## Navigation Tips

Use your editor's "Go to Symbol" or "Outline" feature to quickly jump to sections:
- Look for `// ===` for major sections
- Look for `// ---` for subsections
- All functions are well-indented and clearly named

## Dependencies

- **iced**: GUI framework (v0.13)
- **opencv**: Image processing
- **tokio**: Async runtime
- **pulldown-cmark**: Markdown rendering (help window)
