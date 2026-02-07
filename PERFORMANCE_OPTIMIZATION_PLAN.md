# Performance Optimization Plan

**Date:** 2025-02-06
**Branch:** cleanup
**OpenCV:** 4.12.0 (crate 0.94)
**Current parallelism:** Rayon + OpenCL (UMat)

---

## Executive Summary

The application uses Rayon for multithreading and OpenCL (via UMat) for GPU acceleration, but a **global `OPENCL_MUTEX`** serializes most GPU work, negating the parallelism in several critical paths. Removing this mutex is the single highest-impact optimization. Additionally, the stacking pipeline processes images sequentially where parallel pyramid generation is possible, and the warping phase in ORB/SIFT/AKAZE/ECC alignment uses CPU `Mat` instead of GPU `UMat`.

---

## 1. Sharpness Detection

**Files:** `src/sharpness.rs` (464 lines), `src/gui/handlers/sharpness_handlers.rs` (178 lines)

### Current State

| Aspect | Status | Details |
|--------|--------|---------|
| GPU ops | ✅ Full | `compute_sharpness_umat` / `compute_regional_sharpness_umat` — cvtColor, GaussianBlur, Laplacian, Sobel, multiply, add all on UMat |
| Rayon parallelism | ✅ Used | `sharpness_handlers.rs:99` — `images.par_iter()` |
| OPENCL_MUTEX | ❌ **Bottleneck** | `sharpness_handlers.rs:116` — wraps entire `compute_regional_sharpness_auto` call, serializing ALL GPU sharpness work despite `par_iter` |

### Bottleneck Detail

```text
sharpness_handlers.rs:114-117:
    static OPENCL_MUTEX: Mutex<()> = Mutex::new(());
    let _lock = OPENCL_MUTEX.lock().unwrap();
    let (max_regional, global_sharpness, sharp_region_count) =
        crate::sharpness::compute_regional_sharpness_auto(&img, ...)?;
    drop(_lock);
```

This means: N Rayon threads load N images in parallel, then queue up single-file for GPU sharpness computation. The parallelism is completely wasted.

### Optimizations

| ID | Change | Impact | Effort | Risk |
|----|--------|--------|--------|------|
| **S1** | **Remove OPENCL_MUTEX from sharpness handler** — OpenCV 4.12 handles OpenCL thread safety internally. The ECC code already runs without mutex (`IMAGESTACKER_ECC_MUTEX=0`). Apply the same approach here. | 🔥 High — enables true parallel GPU sharpness across all Rayon threads | Easy | Low — same OpenCV version already works mutex-free in ECC |
| **S2** | **Parallel regional grid** — `compute_regional_sharpness_umat` processes grid cells sequentially (nested for-loop, lines 420-440). With 16×16 grid = 256 sequential GPU calls per image. Could batch ROI extraction or parallelize regions. | Medium — reduces per-image sharpness time | Medium | Low |
| **S3** | **Eliminate redundant global sharpness call** — `compute_regional_sharpness_umat` computes full-image sharpness via separate `compute_sharpness_umat(img_umat)` call (line 451) in addition to all regions. The global score could be derived from the mean of regional scores instead. | Low — saves one full-image GPU pass per image | Easy | None |

---

## 2. Thumbnail Generation

**Files:** `src/thumbnail.rs` (80 lines), `src/gui/handlers/file_handlers.rs` (417 lines)

### Current State

| Aspect | Status | Details |
|--------|--------|---------|
| GPU ops | ✅ Partial | UMat used only for `imgproc::resize` (thumbnail.rs:58-65) |
| Rayon parallelism | ✅ Used | `file_handlers.rs:200,404` — `paths.par_iter()` |
| OPENCL_MUTEX | ✅ None | Thumbnails already run freely in parallel (good!) |

### Analysis

The thumbnail pipeline is: `imread` (CPU) → `get_umat` (upload) → `resize` (GPU) → `get_mat` (download). For an 800px target, the GPU upload/download overhead likely dominates the actual resize computation. CPU `resize` with `INTER_AREA` would be simpler and possibly faster at this scale.

### Optimizations

| ID | Change | Impact | Effort | Risk |
|----|--------|--------|--------|------|
| **T1** | **CPU-only resize for thumbnails** — Target is only 800px max. Remove UMat upload/download overhead. Use CPU `Mat` `imgproc::resize` directly. | Medium — less overhead per thumbnail, simpler code | Easy | None |
| **T2** | **Lazy/progressive thumbnails** — Generate fast 200px thumbnails (INTER_NEAREST) immediately for responsive UI, then upgrade to 800px (INTER_AREA) in background. | Medium — perceived speed improvement | Medium | Low |

---

## 3. Alignment

**File:** `src/alignment.rs` (2101 lines)

### 3a. ORB / SIFT / AKAZE (Feature-Based) — `align_images()` line 1083+

#### Current State

| Phase | GPU | Parallel | Mutex | Lines |
|-------|-----|----------|-------|-------|
| Sharpness pre-filter | ✅ UMat | ✅ par_iter (batched) | ❌ OPENCL_MUTEX (line 1203) | 1140-1260 |
| GPU preprocessing (cvtColor, CLAHE, resize) | ✅ UMat | ✅ par_iter | ❌ **OPENCL_MUTEX (line 1508)** | 1494-1580 |
| Feature detection (detect_and_compute) | ❌ CPU Mat | ✅ par_iter (after GPU unlock) | ✅ None needed | 1580-1590 |
| Pairwise matching | ❌ CPU | ❌ Sequential | N/A | 1650-1780 |
| Warping | ❌ **CPU Mat** | ✅ par_iter | ❌ **OPENCL_MUTEX (line 1934)** | 1890-1960 |

#### Bottleneck Detail: GPU Preprocessing

```text
alignment.rs:1504-1510:
    let (preprocessed, scale) = {
        let _lock = opencl_mutex().lock().unwrap();
        let img_umat = img.get_umat(...)?;
        // ... cvtColor, CLAHE, resize all on GPU ...
        let small_img = small_umat.get_mat(...)?;
    };
    // Feature detection runs AFTER lock is released — truly parallel
```

The GPU preprocessing (upload → cvtColor → CLAHE → resize → download) is serialized. Each thread waits for the previous thread's GPU work to finish. Feature detection (CPU) runs in parallel after the lock is released — this part is fine.

#### Bottleneck Detail: Warping

```text
alignment.rs:1930-1945:
    let (warped, output_path) = {
        let _lock = opencl_mutex().lock().unwrap();
        // ... cvtColor, warp_affine, mask operations ...
    };
```

All warping is serialized AND uses CPU `Mat` instead of GPU `UMat`.

#### Optimizations

| ID | Change | Impact | Effort | Risk |
|----|--------|--------|--------|------|
| **A1** | **Remove OPENCL_MUTEX from GPU preprocessing** — OpenCV 4.12 handles OpenCL thread safety internally. This is the biggest single bottleneck for ORB/SIFT/AKAZE. | 🔥 High — true parallel GPU preprocessing for all Rayon threads | Easy | Low — test with 2-3 threads first |
| **A2** | **Remove OPENCL_MUTEX from warping** — `warp_affine` should be thread-safe in OpenCV 4.12. | 🔥 High — true parallel warping | Easy | Low |
| **A3** | **Use UMat for warping** — Currently warps on CPU `Mat` (lines 1938-1945). Convert input to UMat → GPU warp_affine → download for imwrite. For 42MP images, GPU warp is significantly faster. | High — GPU warp >> CPU warp at 42MP | Medium | Low |
| **A4** | **Parallel pairwise matching** — Feature matching (lines 1650-1780) runs sequentially for consecutive pairs. Pairs are independent and could be parallelized. | Medium — faster matching phase | Medium | Low |

### 3b. ECC — `align_images_ecc()` line 644+

#### Current State

| Phase | GPU | Parallel | Mutex | Lines |
|-------|-----|----------|-------|-------|
| Sharpness pre-filter | ✅ UMat (auto) | ❌ Sequential (per-image loop) | ✅ opencl_mutex (line 712) | 690-760 |
| Preprocessing (cvtColor, GaussianBlur) | ❌ CPU Mat | ✅ par_iter (per batch) | ✅ None | 890-905 |
| ECC transform (`find_transform_ecc`) | ❌ CPU (internally may use OpenCL) | ✅ par_iter | ✅ **No mutex by default** (`IMAGESTACKER_ECC_MUTEX=0`) | 908-920 |
| Warping | ❌ **CPU Mat** | ✅ par_iter | ✅ None | 1020-1043 |
| File I/O (imwrite) | ❌ CPU | ✅ par_iter | ✅ None | 1053-1055 |

ECC is the best-parallelized algorithm — `find_transform_ecc` already runs without mutex. The main gap is CPU-only warping.

#### Optimizations

| ID | Change | Impact | Effort | Risk |
|----|--------|--------|--------|------|
| **E1** | **GPU warping for ECC** — Use UMat for `warp_perspective`/`warp_affine` (lines 1020-1043). Currently CPU `Mat`. | High — GPU warp for 42MP images | Medium | Low |
| **E2** | **Parallel sharpness pre-filter** — ECC sharpness computation (lines 690-760) is sequential (for-loop), unlike ORB/SIFT/AKAZE which use par_iter. Add par_iter with batching. | Medium — faster sharpness phase for ECC | Easy | Low |
| **E3** | **GPU preprocessing** — ECC preprocessing (cvtColor, GaussianBlur) uses CPU `Mat`. Convert to UMat pipeline for GPU acceleration. | Medium — benefits large images | Medium | Low |

### 3c. ECC-Hybrid — `compute_hybrid_ecc_transform()` line 233+

#### Current State

| Phase | GPU | Parallel | Mutex | Lines |
|-------|-----|----------|-------|-------|
| Keypoint extraction (SIFT) | ❌ CPU | N/A (per-image) | ❌ **OPENCL_MUTEX (line 248)** | 246-255 |
| Feature matching | ❌ CPU | N/A (per-image) | ✅ None | 260-340 |
| ECC refinement | ❌ CPU (may use OpenCL internally) | N/A (per-image) | ✅ No mutex (follows ECC_MUTEX env) | 370-395 |

#### Optimizations

| ID | Change | Impact | Effort | Risk |
|----|--------|--------|--------|------|
| **H1** | **Remove OPENCL_MUTEX from keypoint extraction** — Line 248 locks mutex around `extract_features(ref_img, FeatureDetector::SIFT)`. SIFT `detect_and_compute` runs on CPU `Mat` — no GPU mutex needed. | Medium — faster hybrid init in parallel batches | Easy | None — pure CPU operation |

---

## 4. Stacking

**File:** `src/stacking.rs` (602 lines)

### Current State

| Phase | GPU | Parallel | Lines |
|-------|-----|----------|-------|
| Image loading | ❌ CPU | ✅ par_iter (line 119) | 115-130 |
| BGR/Alpha extraction | ✅ UMat | ❌ Sequential (per-image in loop) | 247-260 |
| Laplacian pyramid generation | ✅ UMat (pyr_down, pyr_up, subtract) | ❌ **Sequential** (per-image in loop) | 262 |
| Sharpness energy computation | ✅ UMat (Laplacian, absdiff, GaussianBlur) | ❌ Sequential | 268-280 |
| Layer fusion (winner-take-all) | ✅ UMat (compare, copy_to_masked) | ❌ Sequential (inherently — depends on previous) | 282-300 |
| Pyramid collapse | ✅ UMat (pyr_up, add, clip) | ❌ Sequential (inherently — level by level) | 565-600 |
| Alpha assembly + erosion | ✅ UMat | N/A | 460-510 |

### Analysis

The stacking pipeline in `stack_images_direct` (line 230) processes images one by one in a for-loop. For each image:
1. Upload to GPU → convert to float → split BGR/Alpha → generate Laplacian pyramid → compute energy → fuse with running result

Steps 1-4 are **independent per image** until the fusion step. This means Laplacian pyramid generation for all N images could run in parallel, then fusion runs sequentially.

### Optimizations

| ID | Change | Impact | Effort | Risk |
|----|--------|--------|--------|------|
| **K1** | **Parallel pyramid generation** — Pre-generate Laplacian pyramids for ALL images in the batch using Rayon, then fuse sequentially. Pyramid gen (pyr_down × 7 levels) is the most expensive step and is independent per image. | 🔥 High — pyramid gen dominates stacking time | Medium | Medium — GPU memory for N pyramids simultaneously |
| **K2** | **Parallel layer fusion** — After generating all pyramids, fuse each pyramid level independently in parallel (7 levels = 7 threads). Each level's fusion is independent of other levels. | Medium — 7-way parallelism for fusion | Medium | Low |
| **K3** | **Pre-split BGR/Alpha during parallel load** — `extract_bgr_and_alpha` (line 247) is called per-image inside the sequential loop. Move it to the parallel image loading phase. | Low — small savings per image | Easy | None |
| **K4** | **Reduce GPU↔CPU transfers** — Final conversion (line 515-518) does `convert_to(CV_8U)` on UMat → `get_mat` → `copy_to`. Could skip the final copy_to. | Low — one less copy | Easy | None |

---

## Priority Matrix

### 🥇 Tier 1 — High Impact, Easy Effort (Do First)

| ID | Area | Change | Expected Speedup |
|----|------|--------|-----------------|
| **S1** | Sharpness | Remove OPENCL_MUTEX from sharpness handler | 2-4x (N threads truly parallel) |
| **A1** | Alignment (ORB/SIFT/AKAZE) | Remove OPENCL_MUTEX from GPU preprocessing | 2-4x for preprocessing phase |
| **A2** | Alignment (ORB/SIFT/AKAZE) | Remove OPENCL_MUTEX from warping | 2-4x for warping phase |
| **H1** | Alignment (Hybrid) | Remove OPENCL_MUTEX from keypoint extraction | Minor — removes unnecessary serialization |

### 🥈 Tier 2 — High Impact, Medium Effort

| ID | Area | Change | Expected Speedup |
|----|------|--------|-----------------|
| **K1** | Stacking | Parallel pyramid generation | 2-3x for stacking phase |
| **A3** | Alignment (ORB/SIFT/AKAZE) | Use UMat for warping instead of CPU Mat | 2-5x for warping 42MP images |
| **E1** | Alignment (ECC) | Use UMat for warping | 2-5x for ECC warping |
| **E2** | Alignment (ECC) | Parallel sharpness pre-filter (add par_iter) | 2-4x for ECC sharpness phase |

### 🥉 Tier 3 — Lower Impact / Diminishing Returns

| ID | Area | Change | Expected Speedup |
|----|------|--------|-----------------|
| **T1** | Thumbnails | CPU-only resize (remove UMat overhead) | 10-30% per thumbnail |
| **K2** | Stacking | Parallel layer fusion (7 levels) | 1.5-2x for fusion phase |
| **A4** | Alignment | Parallel pairwise matching | Minor for matching phase |
| **S2** | Sharpness | Parallel regional grid computation | 1.5x per image |
| **S3** | Sharpness | Eliminate redundant global sharpness call | Minor |
| **T2** | Thumbnails | Progressive thumbnails (fast preview) | Perceived speed only |
| **E3** | Alignment (ECC) | GPU preprocessing for ECC | Medium for large images |
| **K3** | Stacking | Pre-split BGR/Alpha in parallel load | Minor |

---

## Implementation Notes

### Removing OPENCL_MUTEX (S1, A1, A2, H1)

The safest approach:

1. **Remove the mutex calls** — delete `let _lock = opencl_mutex().lock().unwrap();` and the corresponding `drop(_lock);`
2. **Add env var fallback** — like ECC already does: `IMAGESTACKER_OPENCL_MUTEX=1` to re-enable if crashes occur
3. **Test incrementally** — remove one mutex at a time, test with 46×42MP images
4. **Keep the `opencl_mutex()` function** — it's still used as an optional safety net

### GPU Warping (A3, E1)

Replace the CPU warp pattern:
```rust
// Current (CPU):
let mut warped = Mat::default();
imgproc::warp_affine(&img, &mut warped, &transform, size, ...)?;

// Optimized (GPU):
let img_umat = img.get_umat(AccessFlag::ACCESS_READ, UMatUsageFlags::USAGE_DEFAULT)?;
let mut warped_umat = UMat::new(UMatUsageFlags::USAGE_DEFAULT);
imgproc::warp_affine(&img_umat, &mut warped_umat, &transform, size, ...)?;
let warped = warped_umat.get_mat(AccessFlag::ACCESS_READ)?;
```

### Parallel Pyramid Generation (K1)

```rust
// Current (sequential):
for (idx, img) in images.iter().enumerate() {
    let pyramid = generate_laplacian_pyramid(&bgr, levels)?;
    // ... fuse immediately ...
}

// Optimized (parallel gen, sequential fuse):
let pyramids: Vec<_> = images.par_iter()
    .map(|img| {
        let float_img = /* upload + convert */;
        let (bgr, alpha) = extract_bgr_and_alpha(&float_img)?;
        let pyramid = generate_laplacian_pyramid(&bgr, levels)?;
        let energies: Vec<_> = pyramid.iter()
            .map(|layer| compute_sharpness_energy(layer))
            .collect::<Result<_>>()?;
        Ok((pyramid, energies, alpha))
    })
    .collect::<Result<Vec<_>>>()?;

// Sequential fusion (depends on running result)
for (pyramid, energies, alpha) in pyramids { ... }
```

**Memory concern:** Each pyramid for a 42MP image ≈ 500-800MB on GPU. For a batch of 6 images, that's 3-5GB of GPU memory. May need to limit parallel pyramid count based on available GPU memory.

---

## Measurement Plan

Before implementing, measure current performance baselines:

```bash
# Measure sharpness detection time
time RUST_LOG=info cargo run --release -- -i testimages_small/ 2>&1 | grep "Sharpness detection complete"

# Measure alignment time (ORB)
time RUST_LOG=info cargo run --release -- -i testimages/ 2>&1 | grep "Alignment completed"

# Measure stacking time
time RUST_LOG=info cargo run --release -- -i testimages/ 2>&1 | grep "Stacking.*complete"
```

After each optimization, re-measure and compare. Expected total improvement from Tier 1 changes alone: **30-60% faster** end-to-end for a typical 46-image workflow.
