//! Message handling (update function) for the ImageStacker GUI
//!
//! This module contains the main update function that dispatches messages
//! to the appropriate handler modules. The actual handler implementations
//! are in the `handlers` submodule.

use iced::Task;

use crate::messages::Message;

use super::state::ImageStacker;

impl ImageStacker {
    /// Main update function - dispatches messages to appropriate handlers
    pub fn update(&mut self, message: Message) -> Task<Message> {
        match message {
            // File loading handlers (handlers/file_handlers.rs)
            Message::AddImages => self.handle_add_images(),
            Message::AddFolder => self.handle_add_folder(),
            Message::LoadFolder(path) => self.handle_load_folder(path),
            Message::ImagesSelected(paths) => self.handle_images_selected(paths),
            Message::ThumbnailUpdated(path, _handle) => self.handle_thumbnail_updated(path),
            Message::InternalPathsScanned(aligned, bunches, final_imgs, paths) => {
                self.handle_internal_paths_scanned(aligned, bunches, final_imgs, paths)
            }

            // Alignment handlers (handlers/alignment_handlers.rs)
            Message::AlignImages => self.handle_align_images(),
            Message::AlignImagesConfirmed(reuse) => self.handle_align_images_confirmed(reuse),
            Message::AlignmentDone(result) => self.handle_alignment_done(result),

            // Stacking handlers (handlers/stacking_handlers.rs)
            Message::StackImages => self.handle_stack_images(),
            Message::CancelAlignedSelection => self.handle_cancel_aligned_selection(),
            Message::ToggleAlignedImage(path) => self.handle_toggle_aligned_image(path),
            Message::SelectAllAligned => self.handle_select_all_aligned(),
            Message::DeselectAllAligned => self.handle_deselect_all_aligned(),
            Message::StackSelectedAligned => self.handle_stack_selected_aligned(),
            Message::StackBunches => self.handle_stack_bunches(),
            Message::CancelBunchSelection => self.handle_cancel_bunch_selection(),
            Message::ToggleBunchImage(path) => self.handle_toggle_bunch_image(path),
            Message::SelectAllBunches => self.handle_select_all_bunches(),
            Message::DeselectAllBunches => self.handle_deselect_all_bunches(),
            Message::StackSelectedBunches => self.handle_stack_selected_bunches(),
            Message::StackingDone(result) => self.handle_stacking_done(result),
            Message::OpenImage(path) => self.handle_open_image(path),
            Message::OpenImageWithExternalEditor(path) => self.handle_open_image_with_external_editor(path),

            // Preview handlers (handlers/preview_handlers.rs)
            Message::ShowImagePreview(path, pane_images) => self.handle_show_image_preview(path, pane_images),
            Message::ImagePreviewLoaded(path, handle, is_thumbnail) => {
                self.handle_image_preview_loaded(path, handle, is_thumbnail)
            }
            Message::LoadFullImage(path) => self.handle_load_full_image(path),
            Message::CloseImagePreview => self.handle_close_image_preview(),
            Message::NextImageInPreview => self.handle_next_image_in_preview(),
            Message::PreviousImageInPreview => self.handle_previous_image_in_preview(),
            Message::NavigationThrottleReset => self.handle_navigation_throttle_reset(),
            Message::ImportedScrollChanged(offset) => self.handle_imported_scroll_changed(offset),
            Message::AlignedScrollChanged(offset) => self.handle_aligned_scroll_changed(offset),
            Message::BunchesScrollChanged(offset) => self.handle_bunches_scroll_changed(offset),
            Message::FinalScrollChanged(offset) => self.handle_final_scroll_changed(offset),

            // Refresh handlers (handlers/refresh_handlers.rs)
            Message::RefreshPanes => self.handle_refresh_panes(),
            Message::AutoRefreshTick => self.handle_auto_refresh_tick(),
            Message::RefreshImportedPane => self.handle_refresh_imported_pane(),
            Message::RefreshAlignedPane => self.handle_refresh_aligned_pane(),
            Message::RefreshBunchesPane => self.handle_refresh_bunches_pane(),
            Message::RefreshFinalPane => self.handle_refresh_final_pane(),

            // Window handlers (handlers/window_handlers.rs)
            Message::ToggleSettings => self.handle_toggle_settings(),
            Message::ToggleHelp => self.handle_toggle_help(),
            Message::CloseHelp => self.handle_close_help(),
            Message::ToggleLog => self.handle_toggle_log(),
            Message::CloseLog => self.handle_close_log(),
            Message::Exit => self.handle_exit(),

            // Settings handlers (handlers/settings_handlers.rs)
            Message::ResetToDefaults => self.handle_reset_to_defaults(),
            Message::SharpnessThresholdChanged(value) => self.handle_sharpness_threshold_changed(value),
            Message::SharpnessGridSizeChanged(value) => self.handle_sharpness_grid_size_changed(value),
            Message::SharpnessIqrMultiplierChanged(value) => self.handle_sharpness_iqr_multiplier_changed(value),
            Message::UseAdaptiveBatchSizes(enabled) => self.handle_use_adaptive_batch_sizes(enabled),
            Message::UseCLAHE(enabled) => self.handle_use_clahe(enabled),
            Message::FeatureDetectorChanged(detector) => self.handle_feature_detector_changed(detector),
            Message::ProgressUpdate(msg, value) => self.handle_progress_update(msg, value),
            Message::EnableNoiseReduction(enabled) => self.handle_enable_noise_reduction(enabled),
            Message::NoiseReductionStrengthChanged(value) => self.handle_noise_reduction_strength_changed(value),
            Message::EnableSharpening(enabled) => self.handle_enable_sharpening(enabled),
            Message::SharpeningStrengthChanged(value) => self.handle_sharpening_strength_changed(value),
            Message::EnableColorCorrection(enabled) => self.handle_enable_color_correction(enabled),
            Message::ContrastBoostChanged(value) => self.handle_contrast_boost_changed(value),
            Message::BrightnessBoostChanged(value) => self.handle_brightness_boost_changed(value),
            Message::SaturationBoostChanged(value) => self.handle_saturation_boost_changed(value),
            Message::UseInternalPreview(enabled) => self.handle_use_internal_preview(enabled),
            Message::PreviewMaxWidthChanged(width) => self.handle_preview_max_width_changed(width),
            Message::PreviewMaxHeightChanged(height) => self.handle_preview_max_height_changed(height),
            Message::ExternalViewerPathChanged(path) => self.handle_external_viewer_path_changed(path),
            Message::ExternalEditorPathChanged(path) => self.handle_external_editor_path_changed(path),
            Message::WindowResized(width, height) => self.handle_window_resized(width, height),

            // No-op message
            Message::None => Task::none(),
        }
    }
}
