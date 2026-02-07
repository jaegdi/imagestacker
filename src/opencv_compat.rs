//! OpenCV compatibility wrappers for cross-version support.
//!
//! Different OpenCV versions have different function signatures due to added parameters:
//! - OpenCV 4.11+ added `AlgorithmHint` to `cvt_color`, `gaussian_blur`, etc.
//! - OpenCV 4.8+ added `enable_precise_upscale` to `SIFT::create`
//! - OpenCV 4.7+ added `max_points` to `AKAZE::create`
//!
//! Older versions (e.g., Ubuntu 24.04's OpenCV 4.6) don't have these parameters.
//! The opencv Rust crate generates different function signatures depending on which
//! OpenCV headers are installed.
//!
//! These wrapper functions use the `_def` variants (e.g., `cvt_color_def`, `create_def`)
//! which use OpenCV's default parameter values and are available across all OpenCV versions.
//! This allows the codebase to compile with both old and new OpenCV installations.

use opencv::core::ToInputArray;
use opencv::core::ToOutputArray;
use opencv::{core, features2d, imgproc, Result};

/// Wrapper for `imgproc::cvt_color` that works across OpenCV versions.
/// Uses `cvt_color_def` which applies default values for `dst_cn` and `AlgorithmHint`
/// (if applicable), ensuring compatibility with OpenCV < 4.11 and >= 4.11.
pub fn cvt_color(
    src: &impl ToInputArray,
    dst: &mut impl ToOutputArray,
    code: i32,
    _dst_cn: i32,
) -> Result<()> {
    // cvt_color_def uses defaults: dst_cn=0, hint=ALGO_HINT_DEFAULT (on 4.11+)
    // All our call sites pass dst_cn=0, so this is equivalent.
    imgproc::cvt_color_def(src, dst, code)
}

/// Wrapper for `imgproc::gaussian_blur` that works across OpenCV versions.
/// Uses `gaussian_blur_def` which applies default values for `sigma_y`, `border_type`,
/// and `AlgorithmHint` (if applicable).
pub fn gaussian_blur(
    src: &impl ToInputArray,
    dst: &mut impl ToOutputArray,
    ksize: core::Size,
    sigma_x: f64,
    _sigma_y: f64,
    _border_type: i32,
) -> Result<()> {
    // gaussian_blur_def uses defaults: sigma_y=0, border_type=BORDER_DEFAULT, hint=ALGO_HINT_DEFAULT
    // All our call sites pass sigma_y=0.0 and border_type=BORDER_DEFAULT, so this is equivalent.
    imgproc::gaussian_blur_def(src, dst, ksize, sigma_x)
}

/// Wrapper for `features2d::SIFT::create` that works across OpenCV versions.
/// OpenCV 4.8+ added `enable_precise_upscale` as a 6th parameter.
/// Uses `create_def()` which works on all versions with default parameters:
/// nfeatures=0 (detect all), nOctaveLayers=3, contrastThreshold=0.04,
/// edgeThreshold=10, sigma=1.6, enable_precise_upscale=false.
///
/// Note: We lose the ability to set custom nfeatures, but this only affects
/// performance (slightly more features detected), not correctness.
/// The feature matching step limits how many features are actually used.
pub fn sift_create() -> Result<core::Ptr<features2d::SIFT>> {
    features2d::SIFT::create_def()
}

/// Wrapper for `features2d::AKAZE::create` that works across OpenCV versions.
/// OpenCV 4.7+ added `max_points` as an 8th parameter.
/// Uses `create_def()` which works on all versions with default parameters:
/// descriptor_type=DESCRIPTOR_MLDB, descriptor_size=0, descriptor_channels=3,
/// threshold=0.001, nOctaves=4, nOctaveLayers=4, diffusivity=DIFF_PM_G2,
/// max_points=-1 (unlimited).
pub fn akaze_create() -> Result<core::Ptr<features2d::AKAZE>> {
    features2d::AKAZE::create_def()
}
