use serde::{Deserialize, Serialize};
use crate::system_info;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum FeatureDetector {
    ORB,
    SIFT,
    AKAZE,
    ECC,  // Enhanced Correlation Coefficient - best for macro/focus stacking
}

impl std::fmt::Display for FeatureDetector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FeatureDetector::ORB => write!(f, "ORB (Fast)"),
            FeatureDetector::SIFT => write!(f, "SIFT (Best Quality)"),
            FeatureDetector::AKAZE => write!(f, "AKAZE (Balanced)"),
            FeatureDetector::ECC => write!(f, "ECC (Macro/Precision)"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum EccMotionType {
    Translation,   // 2 DOF: x, y translation only
    Euclidean,     // 3 DOF: translation + rotation
    Affine,        // 6 DOF: translation, rotation, scale, shear
    Homography,    // 8 DOF: full perspective transform (best for macro rails)
}

impl std::fmt::Display for EccMotionType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EccMotionType::Translation => write!(f, "Translation"),
            EccMotionType::Euclidean => write!(f, "Euclidean (Rotation)"),
            EccMotionType::Affine => write!(f, "Affine (Scale/Shear)"),
            EccMotionType::Homography => write!(f, "Homography (Perspective)"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingConfig {
    pub sharpness_threshold: f32,
    pub sharpness_grid_size: i32,  // Grid size for regional sharpness detection (4-16)
    pub sharpness_iqr_multiplier: f32,  // IQR multiplier for outlier detection (1.5 = standard, 3.0 = very permissive)
    pub use_adaptive_batches: bool,
    pub use_clahe: bool,
    pub feature_detector: FeatureDetector,
    pub batch_config: system_info::BatchSizeConfig,
    // ECC-specific parameters (only used when feature_detector == ECC)
    pub ecc_motion_type: EccMotionType,
    pub ecc_max_iterations: i32,      // Maximum iterations (3000-30000, default 10000)
    pub ecc_epsilon: f64,              // Convergence threshold (1e-8 to 1e-4, default 1e-6)
    pub ecc_gauss_filter_size: i32,   // Gaussian blur kernel size (3-15, odd, default 7)
    pub ecc_chunk_size: usize,         // Images per parallel chunk (8-16, default 12)
    pub ecc_batch_size: usize,         // Images processed in parallel per batch (2-16, default 4) - controls memory usage
    pub ecc_use_hybrid: bool,          // Use SIFT initialization + ECC refinement for 40-50% speedup (default true)
    pub ecc_timeout_seconds: u64,      // Per-image ECC timeout in seconds (10-300, default 60) - prevents infinite hangs
    // Transform validation settings (for all alignment algorithms)
    pub max_transform_scale: f32,      // Maximum scale factor (0.5-2.0, default 1.5)
    pub max_transform_translation: f32, // Maximum translation in pixels (0-1000, default 600)
    pub max_transform_determinant: f32, // Maximum determinant deviation (0.1-5.0, default 3.0)
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
    // External applications
    pub external_viewer_path: String,  // For left-click when use_internal_preview is false
    pub external_editor_path: String,  // For right-click
    // GPU / Performance settings
    pub gpu_concurrency: usize,        // Max concurrent GPU operations (0=unlimited, 1=serialized, 2+=bounded)
    // Stacking bunch size
    pub auto_bunch_size: bool,         // true = auto-calculate bunch size based on RAM, false = use manual value
    pub stacking_bunch_size: usize,    // Manual bunch size (4-64, default 12) — how many images per bunch/batch
    // Font settings
    pub default_font: String,          // System font family name for the GUI (default: "DejaVu Sans")
}

impl Default for ProcessingConfig {
    fn default() -> Self {
        Self {
            sharpness_threshold: 30.0,
            sharpness_grid_size: 4,  // Default 4x4 grid
            sharpness_iqr_multiplier: 1.5,  // Standard outlier detection
            use_adaptive_batches: true,
            use_clahe: true,
            feature_detector: FeatureDetector::ORB,
            batch_config: system_info::BatchSizeConfig::default_config(),
            // ECC defaults (optimized for macro focus stacking)
            ecc_motion_type: EccMotionType::Homography,  // Best for focus rails with slight camera movement
            ecc_max_iterations: 10000,                   // Standard precision
            ecc_epsilon: 1e-6,                           // Sub-pixel accuracy
            ecc_gauss_filter_size: 7,                    // Smooth focus gradients
            ecc_chunk_size: 12,                          // Optimal for 4-8 core systems
            ecc_batch_size: 4,                           // Conservative memory usage (4 images in parallel)
            ecc_use_hybrid: true,                        // Hybrid mode: 60-70% faster with same quality
            ecc_timeout_seconds: 60,                     // 60 seconds per image before timeout
            // Transform validation defaults
            max_transform_scale: 1.5,                    // Allow up to 1.5x scale
            max_transform_translation: 600.0,            // Allow up to 600 pixels translation
            max_transform_determinant: 3.0,              // Allow determinant up to 3x
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
            // External applications (empty = use system default)
            external_viewer_path: String::new(),
            external_editor_path: String::new(),
            // GPU / Performance defaults
            gpu_concurrency: 2,                          // Allow 2 concurrent GPU operations
            // Stacking bunch size defaults
            auto_bunch_size: true,                       // Auto-calculate based on RAM
            stacking_bunch_size: 12,                     // Manual default: 12 images per bunch
            // Font settings
            default_font: "DejaVu Sans".to_string(),
        }
    }
}