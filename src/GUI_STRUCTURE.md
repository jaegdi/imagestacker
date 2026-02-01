# GUI Module Structure

The GUI has been refactored into a modular structure for better maintainability and code organization. The original monolithic `gui.rs` (~3300 lines) is now split into 17 focused modules totaling ~3800 lines.

**Performance Note**: The application features **GPU acceleration via OpenCL** for all image processing operations (blur detection, alignment, stacking). This provides 2-6x speedup compared to CPU-only processing.

## Directory Structure

```text
src/gui/
├── mod.rs              (57 lines)   - Module entry point, theme & title
├── state.rs            (115 lines)  - Application state (ImageStacker struct)
├── update.rs           (108 lines)  - Message dispatcher
├── subscriptions.rs    (94 lines)   - Event subscriptions
├── log_capture.rs      (41 lines)   - Thread-safe log buffer
│
├── handlers/                        - Message handler implementations
│   ├── mod.rs               (12 lines)   - Handler module exports
│   ├── file_handlers.rs     (330 lines)  - File loading & thumbnails
│   ├── alignment_handlers.rs (184 lines) - Image alignment
│   ├── stacking_handlers.rs  (335 lines) - Stacking & selection
│   ├── preview_handlers.rs   (233 lines) - Image preview & navigation
│   ├── refresh_handlers.rs   (238 lines) - Pane refresh operations
│   ├── settings_handlers.rs  (175 lines) - Settings changes
│   └── window_handlers.rs    (169 lines) - Window management
│
└── views/                           - UI rendering functions
    ├── mod.rs           (10 lines)   - View module exports
    ├── main_view.rs     (355 lines)  - Main application view
    ├── settings.rs      (452 lines)  - Settings panel
    ├── pane_aligned.rs  (245 lines)  - Aligned images pane
    ├── pane_bunches.rs  (245 lines)  - Bunches pane
    ├── pane_generic.rs  (216 lines)  - Generic pane (Imported/Final)
    └── windows.rs       (233 lines)  - Help & Log windows
```

## Module Responsibilities

### Core Modules

#### mod.rs - Entry Point

- Re-exports ImageStacker and append_log
- Implements theme() (Dark theme) and title() for windows
- Contains module documentation

#### state.rs - Application State

- ImageStacker struct: Main application state with 30+ fields
  - Image lists: images, aligned_images, bunch_images, final_images
  - Thumbnail cache: Arc<RwLock<HashMap<PathBuf, Handle>>>
  - Configuration: ProcessingConfig
  - UI state: preview, scroll positions, selection modes
  - Processing state: is_processing, cancel_flag, progress
- Default implementation: Initializes all state with sensible defaults
- create_processing_config(): Creates config from current state

#### update.rs - Message Dispatcher

- Clean match statement routing ~65 message types to handler modules
- Comments indicate which handler module handles each message category
- Minimal logic - just delegation to handlers

#### subscriptions.rs - Event Subscriptions

- Auto-refresh timer (triggers during processing)
- Window events (resize, close request)
- Keyboard events (arrow keys, ESC, Ctrl+Q)
- Mouse events (wheel scrolling for navigation)

#### log_capture.rs - Log Buffer

- Thread-safe log storage using Lazy<Mutex<Vec<String>>>
- append_log() function for adding log entries
- get_logs() function for retrieving all logs

### Handler Modules (handlers/)

Each handler module contains methods on ImageStacker for a specific category of messages:

#### file_handlers.rs - File Operations

- handle_add_images() - Open file dialog
- handle_add_folder() - Open folder dialog
- handle_load_folder(path) - Scan folder for images
- handle_images_selected(paths) - Load selected images
- handle_thumbnail_updated(path) - Thumbnail generation callback
- handle_internal_paths_scanned(...) - Process scanned paths

#### alignment_handlers.rs - Alignment Processing

- handle_align_images() - Check for existing aligned images
- handle_align_images_confirmed(reuse) - Start alignment or reuse
- handle_alignment_done(result) - Process alignment result

#### stacking_handlers.rs - Stacking Operations

- Selection mode handlers for aligned images:
  - handle_stack_images(), handle_cancel_aligned_selection()
  - handle_toggle_aligned_image(), handle_select_all_aligned()
  - handle_stack_selected_aligned()
- Selection mode handlers for bunches (parallel structure)
- handle_stacking_done(result) - Process stacking result
- handle_open_image(), handle_open_image_with_external_editor()

#### preview_handlers.rs - Image Preview

- handle_show_image_preview(path, pane_images) - Show preview
- handle_image_preview_loaded(...) - Update preview state
- handle_load_full_image(path) - Load full resolution
- handle_close_image_preview() - Close preview or cancel processing
- Navigation: handle_next_image_in_preview(), handle_previous_image_in_preview()
- Scroll tracking for all four panes

#### refresh_handlers.rs - Pane Refresh

- handle_refresh_panes() - Refresh all subdirectory panes
- handle_auto_refresh_tick() - Timer-triggered refresh
- Individual pane refresh: handle_refresh_imported_pane(), etc.

#### settings_handlers.rs - Settings Changes

- Sharpness settings: threshold, grid size, IQR multiplier
- Processing options: adaptive batches, CLAHE, feature detector
- Advanced options: noise reduction, sharpening, color correction
- Preview settings: internal preview, max dimensions
- External application paths
- handle_reset_to_defaults()

#### window_handlers.rs - Window Management

- handle_toggle_settings() - Toggle settings panel
- handle_toggle_help() - Convert markdown to HTML, open in browser
- handle_close_help() - Close help window
- handle_toggle_log(), handle_close_log() - Log window
- handle_exit() - Exit application

### View Modules (views/)

#### main_view.rs - Main Application View

- view(window_id) - Entry point for view rendering
- render_main_view() - Main window layout
- render_image_preview() - Modal image preview with navigation

#### settings.rs - Settings Panel

- render_settings_panel() - Collapsible settings UI
- Three sections: Alignment, Post-Processing, Preview Settings
- Responsive layout (horizontal/vertical based on window width)

#### pane_aligned.rs - Aligned Images Pane

- render_aligned_pane() - Aligned images pane with selection mode
- Selection mode: Visual feedback (green borders/background)
- Selection mode buttons: Select All, Deselect All, Cancel, Stack
- Normal mode: Click for preview, right-click for external editor
- Fixed 2-column grid (120x90px thumbnails, 8px spacing)

#### pane_bunches.rs - Bunches Pane

- render_bunches_pane() - Bunches pane with selection mode
- Parallel structure to aligned pane with bunch-specific messages
- Selection mode tracking via selected_bunches HashSet
- Same visual design and button layout as aligned pane
- Fixed 2-column grid layout

#### pane_generic.rs - Generic Pane

- render_pane() - Generic pane for Imported and Final images
- render_pane_with_columns() - Multi-column thumbnail grid helper
- No selection mode (simpler than aligned/bunches panes)
- Supports scrolling with position preservation
- Click for preview, right-click for external editor
- Fixed 2-column grid layout

#### windows.rs - Help and Log Windows

- render_help_window() - Help window content
- render_log_window() - Log window with scrollable entries

## Message Flow

```text
User Action
    ↓
subscription() captures event
    ↓
Produces Message variant
    ↓
update() receives Message
    ↓
Dispatches to appropriate handler
    ↓
Handler updates state, returns Task<Message>
    ↓
view() re-renders based on new state
```

## Key Design Patterns

### Thread Safety

- Thumbnail cache uses Arc<RwLock<HashMap<...>>>
- Cancel flag uses Arc<AtomicBool> for processing cancellation
- Log buffer uses Mutex<Vec<String>>

### Async Processing

- Image loading, alignment, stacking run in background threads
- Progress updates via iced::stream::channel
- Cancellation via shared atomic flag

### Responsive Layout

- Window width tracked in state
- Settings panel adapts layout based on available space
- Thumbnail grid columns adjust to pane width

## Dependencies

- iced (v0.13): GUI framework with Task-based async
- opencv (v4.12.0): Image processing with **OpenCL GPU acceleration**
- tokio: Async runtime for file I/O
- rayon: Parallel processing for thumbnails and batch operations
- pulldown-cmark: Markdown to HTML conversion
- rfd: Native file dialogs
- opener: Open files with system default applications

## GPU Acceleration Architecture

### OpenCL Integration
- **UMat-based processing**: All GPU operations use OpenCV's UMat for GPU memory
- **Thread-safe GPU access**: Global OpenCL mutex serializes GPU operations
- **Minimal CPU↔GPU transfers**: Data uploaded once, all ops on GPU, downloaded once
- **Hybrid parallelism**: Rayon threads for I/O, OpenCL for compute

### Performance Optimizations
- **Adaptive batch sizing**: Adjusts based on image size (>30MP: 2-3 images/batch)
- **SIFT optimization**: Reduced from 3000 to 2000 features (~30% faster)
- **GPU pipeline**: color→CLAHE→resize→feature detection (CPU)→warp (GPU)
- **Parallel + GPU**: Multiple images load in parallel, GPU processes sequentially

### Memory Management
- **Typical usage**: 8-10GB RAM for 46×42MP images
- **No OOM errors**: Adaptive batching prevents memory exhaustion
- **UMat reference handling**: Automatic cloning to prevent deallocation errors

## Adding New Features

1. New Message Type: Add variant to src/messages.rs
2. Handler: Add handler method to appropriate handlers/*.rs module
3. Dispatcher: Add match arm in update.rs
4. UI: Add rendering in appropriate views/*.rs module
5. State: Add fields to ImageStacker in state.rs if needed
