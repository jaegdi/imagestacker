use serde::{Deserialize, Serialize};
use crate::system_info;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum FeatureDetector {
    ORB,
    SIFT,
    AKAZE,
}

impl std::fmt::Display for FeatureDetector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FeatureDetector::ORB => write!(f, "ORB (Fast)"),
            FeatureDetector::SIFT => write!(f, "SIFT (Best Quality)"),
            FeatureDetector::AKAZE => write!(f, "AKAZE (Balanced)"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingConfig {
    pub sharpness_threshold: f32,
    pub use_adaptive_batches: bool,
    pub use_clahe: bool,
    pub feature_detector: FeatureDetector,
    pub batch_config: system_info::BatchSizeConfig,
    // Advanced processing options
    pub enable_noise_reduction: bool,
    pub noise_reduction_strength: f32,
    pub enable_sharpening: bool,
    pub sharpening_strength: f32,
    pub enable_color_correction: bool,
    pub contrast_boost: f32,
    pub brightness_boost: f32,
    pub saturation_boost: f32,
    // Preview settings
    pub use_internal_preview: bool,
    pub preview_max_width: f32,
    pub preview_max_height: f32,
}

impl Default for ProcessingConfig {
    fn default() -> Self {
        Self {
            sharpness_threshold: 30.0,
            use_adaptive_batches: true,
            use_clahe: true,
            feature_detector: FeatureDetector::ORB,
            batch_config: system_info::BatchSizeConfig::default_config(),
            // Advanced processing defaults
            enable_noise_reduction: false,
            noise_reduction_strength: 3.0,
            enable_sharpening: false,
            sharpening_strength: 1.0,
            enable_color_correction: false,
            contrast_boost: 1.0,
            brightness_boost: 0.0,
            saturation_boost: 1.0,
            // Preview settings
            use_internal_preview: true,
            preview_max_width: 900.0,
            preview_max_height: 700.0,
        }
    }
}