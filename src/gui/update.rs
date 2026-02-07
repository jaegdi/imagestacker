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
            Message::InternalPathsScanned(paths, sharpness, aligned, bunches, final_imgs) => {
                self.handle_internal_paths_scanned(paths, sharpness, aligned, bunches, final_imgs)
            }
            Message::DetectSharpness => self.handle_detect_sharpness(),
            Message::SharpnessDetectionDone(result) => self.handle_sharpness_detection_done(result),

            // Alignment handlers (handlers/alignment_handlers.rs)
            Message::AlignImages => self.handle_align_images(),
            Message::AlignImagesConfirmed(reuse) => self.handle_align_images_confirmed(reuse),
            Message::AlignmentDone(result) => self.handle_alignment_done(result),

            // Stacking handlers (handlers/stacking_handlers.rs)
            Message::CancelAlignedSelection => self.handle_cancel_aligned_selection(),
            Message::CancelBunchSelection => self.handle_cancel_bunch_selection(),
            Message::CancelImportedSelection => self.handle_cancel_imported_selection(),
            Message::CancelSharpnessSelection => self.handle_cancel_sharpness_selection(),
            Message::DeselectAllAligned => self.handle_deselect_all_aligned(),
            Message::DeselectAllBunches => self.handle_deselect_all_bunches(),
            Message::DeselectAllImported => self.handle_deselect_all_imported(),
            Message::DeselectAllSharpness => self.handle_deselect_all_sharpness(),
            Message::OpenImage(path) => self.handle_open_image(path),
            Message::OpenImageWithExternalEditor(path) => self.handle_open_image_with_external_editor(path),
            Message::SelectAllAligned => self.handle_select_all_aligned(),
            Message::SelectAllBunches => self.handle_select_all_bunches(),
            Message::SelectAllImported => self.handle_select_all_imported(),
            Message::SelectAllSharpness => self.handle_select_all_sharpness(),
            Message::StackBunches => self.handle_stack_bunches(),
            Message::StackImages => self.handle_stack_images(),
            Message::StackImported => self.handle_stack_imported(),
            Message::StackingDone(result) => self.handle_stacking_done(result),
            Message::StackSelectedAligned => self.handle_stack_selected_aligned(),
            Message::StackSelectedBunches => self.handle_stack_selected_bunches(),
            Message::StackSelectedImported => self.handle_stack_selected_imported(),
            Message::StackSelectedSharpness => self.handle_stack_selected_sharpness(),
            Message::StackSharpness => self.handle_stack_sharpness(),
            Message::ToggleAlignedImage(path) => self.handle_toggle_aligned_image(path),
            Message::ToggleBunchImage(path) => self.handle_toggle_bunch_image(path),
            Message::ToggleImportedImage(path) => self.handle_toggle_imported_image(path),
            Message::ToggleSharpnessImage(path) => self.handle_toggle_sharpness_image(path),

            // Preview handlers (handlers/preview_handlers.rs)
            Message::ShowImagePreview(path, pane_images) => self.handle_show_image_preview(path, pane_images),
            Message::ImagePreviewLoaded(path, handle, is_thumbnail) => {
                self.handle_image_preview_loaded(path, handle, is_thumbnail)
            }
            Message::LoadFullImage(path) => self.handle_load_full_image(path),
            Message::CloseImagePreview => {
                // Delegate to handler which has correct priority:
                // 1. If preview is open → close preview (background process keeps running)
                // 2. If no preview but process running → cancel background process
                // 3. Otherwise → no-op
                self.handle_close_image_preview()
            }
            Message::DeletePreviewImage => self.handle_delete_preview_image(),
            Message::AlignedScrollChanged(offset) => self.handle_aligned_scroll_changed(offset),
            Message::BunchesScrollChanged(offset) => self.handle_bunches_scroll_changed(offset),
            Message::FinalScrollChanged(offset) => self.handle_final_scroll_changed(offset),
            Message::ImportedScrollChanged(offset) => self.handle_imported_scroll_changed(offset),
            Message::NavigationThrottleReset => self.handle_navigation_throttle_reset(),
            Message::NextImageInPreview => self.handle_next_image_in_preview(),
            Message::PreviousImageInPreview => self.handle_previous_image_in_preview(),
            Message::SharpnessScrollChanged(offset) => self.handle_sharpness_scroll_changed(offset),

            // Refresh handlers (handlers/refresh_handlers.rs)
            Message::AutoRefreshTick => self.handle_auto_refresh_tick(),
            Message::RefreshAlignedPane => self.handle_refresh_aligned_pane(),
            Message::RefreshBunchesPane => self.handle_refresh_bunches_pane(),
            Message::RefreshFinalPane => self.handle_refresh_final_pane(),
            Message::RefreshImportedPane => self.handle_refresh_imported_pane(),
            Message::RefreshPanes => self.handle_refresh_panes(),
            Message::RefreshSharpnessPane => self.handle_refresh_sharpness_pane(),

            // Window handlers (handlers/window_handlers.rs)
            Message::CloseHelp => self.handle_close_help(),
            Message::CloseLog => self.handle_close_log(),
            Message::Exit => self.handle_exit(),
            Message::ToggleHelp => self.handle_toggle_help(),
            Message::ToggleLog => self.handle_toggle_log(),
            Message::ToggleSettings => self.handle_toggle_settings(),

            // Settings handlers (handlers/settings_handlers.rs)
            Message::BrightnessBoostChanged(value) => self.handle_brightness_boost_changed(value),
            Message::ContrastBoostChanged(value) => self.handle_contrast_boost_changed(value),
            Message::EccBatchSizeChanged(value) => self.handle_ecc_batch_size_changed(value),
            Message::EccChunkSizeChanged(value) => self.handle_ecc_chunk_size_changed(value),
            Message::EccEpsilonChanged(value) => self.handle_ecc_epsilon_changed(value),
            Message::EccGaussFilterSizeChanged(value) => self.handle_ecc_gauss_filter_size_changed(value),
            Message::EccMaxIterationsChanged(value) => self.handle_ecc_max_iterations_changed(value),
            Message::EccMotionTypeChanged(motion_type) => self.handle_ecc_motion_type_changed(motion_type),
            Message::EccTimeoutChanged(value) => self.handle_ecc_timeout_changed(value),
            Message::EccUseHybridChanged(enabled) => self.handle_ecc_use_hybrid_changed(enabled),
            Message::EnableColorCorrection(enabled) => self.handle_enable_color_correction(enabled),
            Message::EnableNoiseReduction(enabled) => self.handle_enable_noise_reduction(enabled),
            Message::EnableSharpening(enabled) => self.handle_enable_sharpening(enabled),
            Message::ExternalEditorPathChanged(path) => self.handle_external_editor_path_changed(path),
            Message::ExternalViewerPathChanged(path) => self.handle_external_viewer_path_changed(path),
            Message::FeatureDetectorChanged(detector) => self.handle_feature_detector_changed(detector),
            Message::GpuConcurrencyChanged(value) => self.handle_gpu_concurrency_changed(value),
            Message::AutoBunchSizeChanged(enabled) => self.handle_auto_bunch_size_changed(enabled),
            Message::BunchSizeChanged(value) => self.handle_bunch_size_changed(value),
            Message::DefaultFontChanged(font) => self.handle_default_font_changed(font),
            Message::MaxTransformDeterminantChanged(value) => self.handle_max_transform_determinant_changed(value),
            Message::MaxTransformScaleChanged(value) => self.handle_max_transform_scale_changed(value),
            Message::MaxTransformTranslationChanged(value) => self.handle_max_transform_translation_changed(value),
            Message::NoiseReductionStrengthChanged(value) => self.handle_noise_reduction_strength_changed(value),
            Message::PreviewMaxHeightChanged(height) => self.handle_preview_max_height_changed(height),
            Message::PreviewMaxWidthChanged(width) => self.handle_preview_max_width_changed(width),
            Message::ProgressUpdate(msg, value) => self.handle_progress_update(msg, value),
            Message::ResetToDefaults => self.handle_reset_to_defaults(),
            Message::SaturationBoostChanged(value) => self.handle_saturation_boost_changed(value),
            Message::SharpeningStrengthChanged(value) => self.handle_sharpening_strength_changed(value),
            Message::SharpnessGridSizeChanged(value) => self.handle_sharpness_grid_size_changed(value),
            Message::SharpnessIqrMultiplierChanged(value) => self.handle_sharpness_iqr_multiplier_changed(value),
            Message::SharpnessThresholdChanged(value) => self.handle_sharpness_threshold_changed(value),
            Message::UseAdaptiveBatchSizes(enabled) => self.handle_use_adaptive_batch_sizes(enabled),
            Message::UseCLAHE(enabled) => self.handle_use_clahe(enabled),
            Message::UseInternalPreview(enabled) => self.handle_use_internal_preview(enabled),
            Message::WindowResized(width, height) => self.handle_window_resized(width, height),

            // PaneGrid events
            Message::PaneResized(event) => {
                self.pane_state.resize(event.split, event.ratio);
                Task::none()
            }
            Message::PaneDragged(event) => {
                if let iced::widget::pane_grid::DragEvent::Dropped { pane, target } = event {
                    self.pane_state.drop(pane, target);
                }
                Task::none()
            }

            // No-op message
            Message::None => Task::none(),
        }
    }
}
