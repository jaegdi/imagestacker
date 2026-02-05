# ECC-Hybrid Alignment Implementation Plan

## Status: In Progress (Configuration Layer Complete)

This document outlines the plan to integrate ECC (Enhanced Correlation Coefficient) alignment as an alternative to feature-based methods (ORB/SIFT/AKAZE) for precision focus stacking.

## Completed Steps ✅

### 1. Configuration Layer (DONE)
- Added `ECC` variant to `FeatureDetector` enum in `src/config.rs`
- Added `EccMotionType` enum with 4 transformation types:
  - Translation (2 DOF)
  - Euclidean (3 DOF - rotation)
  - Affine (6 DOF - scale/shear)
  - Homography (8 DOF - perspective) **[Default for macro]**
- Added ECC parameters to `ProcessingConfig`:
  - `ecc_motion_type: EccMotionType` - Default: Homography
  - `ecc_max_iterations: i32` - Default: 10000
  - `ecc_epsilon: f64` - Default: 1e-6
  - `ecc_gauss_filter_size: i32` - Default: 7 (odd kernel)
  - `ecc_chunk_size: usize` - Default: 12 (for parallel processing)

### 2. Message Handling (DONE)
- Added 5 new messages to `src/messages.rs`:
  - `EccMotionTypeChanged(EccMotionType)`
  - `EccMaxIterationsChanged(f32)`
  - `EccEpsilonChanged(f32)` - logarithmic scale
  - `EccGaussFilterSizeChanged(f32)`
  - `EccChunkSizeChanged(f32)`
- Wired messages in `src/gui/update.rs`
- Implemented handlers in `src/gui/handlers/settings_handlers.rs`
  - Epsilon handler uses logarithmic mapping: slider -8 to -4 → 1e-8 to 1e-4
  - Gauss filter ensures odd kernel sizes

## Remaining Work 🔧

### 3. Core ECC Implementation (IN PROGRESS)
**File:** `src/alignment.rs`

Need to add:

#### A. ECC-specific helper functions
```rust
/// Convert EccMotionType to OpenCV MotionType constant
fn ecc_motion_to_opencv(motion: EccMotionType) -> i32 {
    use opencv::video;
    match motion {
        EccMotionType::Translation => video::MOTION_TRANSLATION,
        EccMotionType::Euclidean => video::MOTION_EUCLIDEAN,
        EccMotionType::Affine => video::MOTION_AFFINE,
        EccMotionType::Homography => video::MOTION_HOMOGRAPHY,
    }
}

/// Compute ECC-based transformation matrix between two grayscale images
fn compute_ecc_transform(
    reference: &Mat,
    target: &Mat,
    config: &ProcessingConfig,
) -> Result<Mat> {
    // 1. Apply Gaussian blur to reduce noise
    let mut ref_blurred = Mat::default();
    let mut tgt_blurred = Mat::default();
    
    let kernel_size = core::Size::new(config.ecc_gauss_filter_size, config.ecc_gauss_filter_size);
    imgproc::gaussian_blur(reference, &mut ref_blurred, kernel_size, 0.0, 0.0, core::BORDER_DEFAULT)?;
    imgproc::gaussian_blur(target, &mut tgt_blurred, kernel_size, 0.0, 0.0, core::BORDER_DEFAULT)?;
    
    // 2. Initialize warp matrix based on motion type
    let motion_type = ecc_motion_to_opencv(config.ecc_motion_type);
    let rows = if motion_type == video::MOTION_HOMOGRAPHY { 3 } else { 2 };
    let mut warp_matrix = Mat::eye(rows, 3, core::CV_32F)?;
    
    // 3. Define termination criteria
    let criteria = core::TermCriteria {
        typ: core::TermCriteria_COUNT + core::TermCriteria_EPS,
        max_count: config.ecc_max_iterations,
        epsilon: config.ecc_epsilon,
    };
    
    // 4. Compute ECC transformation
    use opencv::video;
    video::find_transform_ecc(
        &tgt_blurred,
        &ref_blurred,
        &mut warp_matrix,
        motion_type,
        criteria,
        &Mat::default(),  // inputMask
        5,                // gaussFiltSize (internal, additional to our pre-blur)
    )?;
    
    Ok(warp_matrix)
}
```

#### B. Main ECC alignment function
```rust
/// Align images using ECC method (for macro/precision focus stacking)
fn align_images_ecc(
    image_paths: &[PathBuf],
    output_dir: &Path,
    config: &ProcessingConfig,
    progress_cb: Option<ProgressCallback>,
    cancel_flag: Option<Arc<AtomicBool>>,
) -> Result<core::Rect> {
    // 1. Sort images by sharpness (reuse existing logic)
    //    - Load all images
    //    - Compute sharpness scores (regional)
    //    - Sort: mid-sharpness image becomes reference
    
    // 2. Convert reference to grayscale for ECC
    
    // 3. Process in parallel chunks with overlap
    //    - Chunk size from config.ecc_chunk_size
    //    - 50% overlap between chunks
    //    - Each chunk processed via Rayon
    
    // 4. For each image in chunk:
    //    - Convert to grayscale
    //    - Compute ECC transform vs reference
    //    - Apply warp_affine or warp_perspective
    //    - Save aligned image
    
    // 5. Return bounding box of aligned region
    
    todo!("Implement ECC alignment")
}
```

#### C. Update existing match arms
In `align_images()` function, add routing:
```rust
pub fn align_images(...) -> Result<core::Rect> {
    // After blur detection, before feature extraction:
    
    match config.feature_detector {
        FeatureDetector::ECC => {
            log::info!("🔬 Using ECC (Enhanced Correlation Coefficient) alignment");
            return align_images_ecc(image_paths, output_dir, config, progress_cb, cancel_flag);
        }
        _ => {
            // Existing feature-based flow (ORB/SIFT/AKAZE)
        }
    }
}
```

Also add ECC cases to these match statements:
- Line 32: `extract_features` → return empty features (ECC doesn't use them)
- Line 373: Print statement → "ECC (Precision)"
- Line 400: Parallel limit → 1 (ECC is already parallelized internally)
- Line 406: Feature batch size → N/A (use ecc_chunk_size instead)
- Line 415: Image size limit → Same as SIFT (30MP)

### 4. UI Integration (TODO)
**File:** `src/gui/views/settings.rs`

After the AKAZE button (around line 160), add ECC button and conditional parameter panel:

```rust
// ECC button (4th option)
let ecc_selected = self.config.feature_detector == FeatureDetector::ECC;
let ecc_button = button(
    text(if ecc_selected { "✓ ECC (Macro/Precision)" } else { "  ECC (Macro/Precision)" })
)
.style(move |theme, status| {
    if ecc_selected {
        button::Style {
            background: Some(iced::Background::Color(iced::Color::from_rgb(0.3, 0.5, 0.3))),
            text_color: iced::Color::WHITE,
            border: iced::Border {
                color: iced::Color::from_rgb(0.4, 0.7, 0.4),
                width: 2.0,
                radius: 4.0.into(),
            },
            ..Default::default()
        }
    } else {
        button::secondary(theme, status)
    }
})
.on_press(Message::FeatureDetectorChanged(FeatureDetector::ECC));

// ECC Parameters Panel (shown only when ECC is selected)
let ecc_params_panel = if ecc_selected {
    column![
        text("ECC Parameters:").size(14),
        
        // Motion Type buttons
        row![
            text("Motion Type:").width(label_width),
            button(if self.config.ecc_motion_type == EccMotionType::Translation { "✓ Translation" } else { "Translation" })
                .on_press(Message::EccMotionTypeChanged(EccMotionType::Translation)),
            button(if self.config.ecc_motion_type == EccMotionType::Euclidean { "✓ Euclidean" } else { "Euclidean" })
                .on_press(Message::EccMotionTypeChanged(EccMotionType::Euclidean)),
            button(if self.config.ecc_motion_type == EccMotionType::Affine { "✓ Affine" } else { "Affine" })
                .on_press(Message::EccMotionTypeChanged(EccMotionType::Affine)),
            button(if self.config.ecc_motion_type == EccMotionType::Homography { "✓ Homography" } else { "Homography" })
                .on_press(Message::EccMotionTypeChanged(EccMotionType::Homography)),
        ].spacing(5),
        
        // Max Iterations slider
        row![
            text("Max Iterations:").width(label_width),
            slider(3000.0..=30000.0, self.config.ecc_max_iterations as f32, Message::EccMaxIterationsChanged)
                .step(1000.0)
                .width(slider_width),
            text(format!("{}", self.config.ecc_max_iterations)).width(value_width),
        ].spacing(10),
        
        // Epsilon slider (logarithmic: 1e-8 to 1e-4)
        row![
            text("Epsilon (10^x):").width(label_width),
            slider(-8.0..=-4.0, self.config.ecc_epsilon.log10(), Message::EccEpsilonChanged)
                .step(0.1)
                .width(slider_width),
            text(format!("{:.1e}", self.config.ecc_epsilon)).width(value_width),
        ].spacing(10),
        
        // Gauss Filter Size slider
        row![
            text("Blur Kernel:").width(label_width),
            slider(3.0..=15.0, self.config.ecc_gauss_filter_size as f32, Message::EccGaussFilterSizeChanged)
                .step(2.0)  // Ensure odd
                .width(slider_width),
            text(format!("{}x{}", self.config.ecc_gauss_filter_size, self.config.ecc_gauss_filter_size)).width(value_width),
        ].spacing(10),
        
        // Chunk Size slider
        row![
            text("Parallel Chunks:").width(label_width),
            slider(4.0..=24.0, self.config.ecc_chunk_size as f32, Message::EccChunkSizeChanged)
                .step(2.0)
                .width(slider_width),
            text(format!("{} images", self.config.ecc_chunk_size)).width(value_width),
        ].spacing(10),
    ]
    .spacing(10)
    .padding(10)
} else {
    column![].into()  // Empty when not ECC
};
```

### 5. Documentation (TODO)

#### A. PROJECT_STATUS.md
Add section:
```markdown
### ECC-Hybrid Alignment (v1.1.0)

**Status:** Complete
**Purpose:** Sub-pixel precision alignment for macro focus stacking

- **Method:** Enhanced Correlation Coefficient (iterative Lucas-Kanade)
- **Use Case:** Focus rails, macro photography, high-precision requirements
- **Advantage:** ~10x more accurate than feature-based (0.01-0.001 pixel vs 0.1-0.5 pixel)
- **Trade-off:** 2-4x slower than ORB, requires similar lighting/exposure

**Parameters:**
- Motion Type: Translation, Euclidean, Affine, Homography
- Max Iterations: 3K-30K (default 10K)
- Epsilon: 1e-8 to 1e-4 (default 1e-6)
- Gauss Filter: 3-15 (default 7)
- Chunk Size: 4-24 images (default 12)

**GPU Support:** OpenCL-enabled Mat operations (blur, warp), mutex-serialized for thread safety
```

#### B. USER_MANUAL.md
Add section in "Alignment Methods":
```markdown
### ECC (Enhanced Correlation Coefficient) - Precision Mode

**Best for:**
- Macro photography with focus rails
- Microscopy focus stacking
- Situations requiring sub-pixel accuracy
- Images with similar exposure and lighting

**Advantages:**
- **Highest Precision:** 0.01-0.001 pixel accuracy (10-100x better than feature-based)
- **No feature detection needed:** Works on smooth/uniform surfaces
- **Subpixel warping:** Iterative refinement for maximum sharpness

**Limitations:**
- **Slower:** 2-4x slower than ORB (but still parallelized)
- **Requires similar lighting:** Doesn't handle exposure changes well
- **Memory intensive:** Processes full images (no keypoint reduction)

**When to use:**
- Focus stacking with macro rails (recommended!)
- Scientific/technical imaging requiring maximum precision
- When ORB/SIFT fail due to lack of features

**Parameters Explained:**

- **Motion Type:**
  - *Translation*: Only X/Y shifts (fastest, least flexible)
  - *Euclidean*: Translation + rotation
  - *Affine*: Translation + rotation + scale + shear
  - *Homography*: Full perspective transform (best for macro rails) ⭐

- **Max Iterations:** Higher = more precise but slower
  - 3000: Fast preview
  - 10000: Standard (recommended) ⭐
  - 25000: Maximum precision

- **Epsilon:** Convergence threshold (10^x format)
  - 1e-4: Fastest, ~0.1 pixel accuracy
  - 1e-6: Balanced, ~0.01 pixel accuracy ⭐
  - 1e-8: Slowest, ~0.001 pixel accuracy

- **Blur Kernel:** Gaussian smoothing before alignment
  - 3-5: Sharp images, low noise
  - 7: Balanced (recommended) ⭐
  - 9-15: Noisy images or strong focus gradients

- **Parallel Chunks:** Images per batch for multi-core processing
  - 8: Low RAM systems
  - 12: Standard (recommended) ⭐
  - 16-24: High-RAM workstations

**Recommended Presets:**

| Scenario | Motion | Iterations | Epsilon | Kernel | Speed |
|----------|--------|------------|---------|--------|-------|
| **Quick Preview** | Homography | 3000 | 1e-4 | 5 | Fast |
| **Standard Macro** | Homography | 10000 | 1e-6 | 7 | Medium ⭐ |
| **Maximum Quality** | Homography | 25000 | 1e-8 | 9 | Slow |
```

## Dependencies

ECC uses OpenCV's `video` module:
- `opencv::video::find_transform_ecc()` - Core algorithm
- Already included in opencv-rust crate
- Requires OpenCV 4.x (already a dependency)

## Testing Checklist

- [ ] Compile without errors
- [ ] ECC button appears in settings
- [ ] ECC parameters show/hide correctly
- [ ] Parameters save/load from settings.json
- [ ] Alignment produces registered images
- [ ] Progress reporting works
- [ ] Cancellation (ESC) works during ECC
- [ ] Memory usage reasonable for large images
- [ ] Performance comparable to reference implementation
- [ ] GPU/OpenCL compatibility maintained

## Next Steps for Completion

1. **Implement core ECC functions** in `src/alignment.rs` (est. 200-300 lines)
2. **Add UI controls** in `src/gui/views/settings.rs` (est. 100 lines)
3. **Update documentation** (PROJECT_STATUS.md, USER_MANUAL.md)
4. **Test with macro images** and verify sub-pixel accuracy
5. **Optimize GPU/OpenCL integration** for warp operations
6. **Add preset buttons** (Quick/Standard/Quality) for common scenarios

## References

- OpenCV ECC: https://docs.opencv.org/4.x/dc/d6b/group__video__track.html#ga1af661ab7475f4b1b2c831f66e5e1eb5
- Original paper: Evangelidis & Psarakis (2008) - "Parametric Image Alignment using Enhanced Correlation Coefficient Maximization"
- User research: Macro focus stacking forum discussions on sub-pixel accuracy needs
