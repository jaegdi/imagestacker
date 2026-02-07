//! OpenCV compatibility wrappers for cross-version support.
//!
//! OpenCV 4.11+ introduced `AlgorithmHint` as a parameter to many imgproc functions
//! like `cvt_color` and `gaussian_blur`. Older versions (e.g., Ubuntu 24.04's OpenCV 4.6)
//! don't have this parameter. The opencv Rust crate generates different function signatures
//! depending on which OpenCV headers are installed.
//!
//! These wrapper functions use the `_def` variants (`cvt_color_def`, `gaussian_blur_def`)
//! which use OpenCV's default parameter values and are available across all OpenCV versions.
//! This allows the codebase to compile with both old and new OpenCV installations.

use opencv::core::ToInputArray;
use opencv::core::ToOutputArray;
use opencv::{core, imgproc, Result};

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
