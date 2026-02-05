//! Settings change handlers
//!
//! Handlers for all settings-related messages (toggles, sliders, paths)

use iced::Task;

use crate::config::{EccMotionType, FeatureDetector, ProcessingConfig};
use crate::messages::Message;
use crate::settings::save_settings;
use crate::system_info;

use crate::gui::state::ImageStacker;

impl ImageStacker {
    /// Handle ResetToDefaults
    pub fn handle_reset_to_defaults(&mut self) -> Task<Message> {
        self.config = ProcessingConfig::default();
        let _ = save_settings(&self.config);
        Task::none()
    }

    /// Handle SharpnessThresholdChanged
    pub fn handle_sharpness_threshold_changed(&mut self, value: f32) -> Task<Message> {
        self.config.sharpness_threshold = value;
        let _ = save_settings(&self.config);
        Task::none()
    }

    /// Handle SharpnessGridSizeChanged
    pub fn handle_sharpness_grid_size_changed(&mut self, value: f32) -> Task<Message> {
        self.config.sharpness_grid_size = value as i32;
        let _ = save_settings(&self.config);
        Task::none()
    }

    /// Handle SharpnessIqrMultiplierChanged
    pub fn handle_sharpness_iqr_multiplier_changed(&mut self, value: f32) -> Task<Message> {
        self.config.sharpness_iqr_multiplier = value;
        let _ = save_settings(&self.config);
        Task::none()
    }

    /// Handle UseAdaptiveBatchSizes
    pub fn handle_use_adaptive_batch_sizes(&mut self, enabled: bool) -> Task<Message> {
        self.config.use_adaptive_batches = enabled;
        if enabled {
            // Recalculate batch sizes based on system RAM
            let available_gb = system_info::get_available_memory_gb();
            // Estimate average image size (assume 24MP RGB image ~72MB)
            let avg_size_mb = 72.0;
            self.config.batch_config = 
                system_info::BatchSizeConfig::calculate_optimal(available_gb, avg_size_mb);
        }
        let _ = save_settings(&self.config);
        Task::none()
    }

    /// Handle UseCLAHE
    pub fn handle_use_clahe(&mut self, enabled: bool) -> Task<Message> {
        self.config.use_clahe = enabled;
        let _ = save_settings(&self.config);
        Task::none()
    }

    /// Handle FeatureDetectorChanged
    pub fn handle_feature_detector_changed(&mut self, detector: FeatureDetector) -> Task<Message> {
        self.config.feature_detector = detector;
        let _ = save_settings(&self.config);
        Task::none()
    }

    /// Handle EccMotionTypeChanged
    pub fn handle_ecc_motion_type_changed(&mut self, motion_type: EccMotionType) -> Task<Message> {
        self.config.ecc_motion_type = motion_type;
        let _ = save_settings(&self.config);
        Task::none()
    }

    /// Handle EccMaxIterationsChanged
    pub fn handle_ecc_max_iterations_changed(&mut self, value: f32) -> Task<Message> {
        self.config.ecc_max_iterations = value as i32;
        let _ = save_settings(&self.config);
        Task::none()
    }

    /// Handle EccEpsilonChanged (logarithmic slider: -8 to -4 maps to 1e-8 to 1e-4)
    pub fn handle_ecc_epsilon_changed(&mut self, value: f32) -> Task<Message> {
        self.config.ecc_epsilon = 10_f64.powf(value as f64);
        let _ = save_settings(&self.config);
        Task::none()
    }

    /// Handle EccGaussFilterSizeChanged
    pub fn handle_ecc_gauss_filter_size_changed(&mut self, value: f32) -> Task<Message> {
        // Ensure odd number (3, 5, 7, 9, 11, 13, 15)
        let size = (value as i32) | 1;  // Make odd by setting lowest bit
        self.config.ecc_gauss_filter_size = size;
        let _ = save_settings(&self.config);
        Task::none()
    }

    /// Handle EccChunkSizeChanged
    pub fn handle_ecc_chunk_size_changed(&mut self, value: f32) -> Task<Message> {
        self.config.ecc_chunk_size = value as usize;
        let _ = save_settings(&self.config);
        Task::none()
    }

    /// Handle ProgressUpdate
    pub fn handle_progress_update(&mut self, msg: String, value: f32) -> Task<Message> {
        self.progress_message = msg;
        self.progress_value = value;
        Task::none()
    }

    /// Handle EnableNoiseReduction
    pub fn handle_enable_noise_reduction(&mut self, enabled: bool) -> Task<Message> {
        self.config.enable_noise_reduction = enabled;
        let _ = save_settings(&self.config);
        Task::none()
    }

    /// Handle NoiseReductionStrengthChanged
    pub fn handle_noise_reduction_strength_changed(&mut self, value: f32) -> Task<Message> {
        self.config.noise_reduction_strength = value;
        let _ = save_settings(&self.config);
        Task::none()
    }

    /// Handle EnableSharpening
    pub fn handle_enable_sharpening(&mut self, enabled: bool) -> Task<Message> {
        self.config.enable_sharpening = enabled;
        let _ = save_settings(&self.config);
        Task::none()
    }

    /// Handle SharpeningStrengthChanged
    pub fn handle_sharpening_strength_changed(&mut self, value: f32) -> Task<Message> {
        self.config.sharpening_strength = value;
        let _ = save_settings(&self.config);
        Task::none()
    }

    /// Handle EnableColorCorrection
    pub fn handle_enable_color_correction(&mut self, enabled: bool) -> Task<Message> {
        self.config.enable_color_correction = enabled;
        let _ = save_settings(&self.config);
        Task::none()
    }

    /// Handle ContrastBoostChanged
    pub fn handle_contrast_boost_changed(&mut self, value: f32) -> Task<Message> {
        self.config.contrast_boost = value;
        let _ = save_settings(&self.config);
        Task::none()
    }

    /// Handle BrightnessBoostChanged
    pub fn handle_brightness_boost_changed(&mut self, value: f32) -> Task<Message> {
        self.config.brightness_boost = value;
        let _ = save_settings(&self.config);
        Task::none()
    }

    /// Handle SaturationBoostChanged
    pub fn handle_saturation_boost_changed(&mut self, value: f32) -> Task<Message> {
        self.config.saturation_boost = value;
        let _ = save_settings(&self.config);
        Task::none()
    }

    /// Handle UseInternalPreview
    pub fn handle_use_internal_preview(&mut self, enabled: bool) -> Task<Message> {
        self.config.use_internal_preview = enabled;
        let _ = save_settings(&self.config);
        Task::none()
    }

    /// Handle PreviewMaxWidthChanged
    pub fn handle_preview_max_width_changed(&mut self, width: f32) -> Task<Message> {
        self.config.preview_max_width = width;
        let _ = save_settings(&self.config);
        Task::none()
    }

    /// Handle PreviewMaxHeightChanged
    pub fn handle_preview_max_height_changed(&mut self, height: f32) -> Task<Message> {
        self.config.preview_max_height = height;
        let _ = save_settings(&self.config);
        Task::none()
    }

    /// Handle ExternalViewerPathChanged
    pub fn handle_external_viewer_path_changed(&mut self, path: String) -> Task<Message> {
        self.config.external_viewer_path = path;
        let _ = save_settings(&self.config);
        Task::none()
    }

    /// Handle ExternalEditorPathChanged
    pub fn handle_external_editor_path_changed(&mut self, path: String) -> Task<Message> {
        self.config.external_editor_path = path;
        let _ = save_settings(&self.config);
        Task::none()
    }

    /// Handle WindowResized
    pub fn handle_window_resized(&mut self, width: f32, _height: f32) -> Task<Message> {
        self.window_width = width;
        Task::none()
    }
}
