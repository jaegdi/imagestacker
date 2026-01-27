use anyhow::Result;
use opencv::prelude::*;
use opencv::{calib3d, core, features2d, imgproc};
use std::path::{Path, PathBuf};

pub fn load_image(path: &PathBuf) -> Result<Mat> {
    use opencv::imgcodecs;
    let img = imgcodecs::imread(path.to_str().unwrap(), imgcodecs::IMREAD_COLOR)?;
    Ok(img)
}

pub fn align_images(images: &mut [Mat], output_dir: &std::path::Path) -> Result<()> {
    if images.len() < 2 {
        return Ok(());
    }

    let aligned_dir = output_dir.join("aligned");
    std::fs::create_dir_all(&aligned_dir)?;

    // Save reference image (Image 0)
    opencv::imgcodecs::imwrite(
        aligned_dir.join("0000.png").to_str().unwrap(),
        &images[0],
        &opencv::core::Vector::new(),
    )?;

    let mut sift = features2d::SIFT::create(0, 3, 0.04, 10.0, 1.6, false)?;

    // Total transformation from current image to Image 0
    let mut t_total = Mat::eye(3, 3, core::CV_64F)?.to_mat()?;

    // Pre-compute features for the first image
    let mut prev_keypoints = opencv::core::Vector::new();
    let mut prev_descriptors: core::UMat = core::UMat::new(core::UMatUsageFlags::USAGE_DEFAULT);

    let ref_umat = images[0].get_umat(
        core::AccessFlag::ACCESS_READ,
        core::UMatUsageFlags::USAGE_DEFAULT,
    )?;
    sift.detect_and_compute(
        &ref_umat,
        &core::UMat::new(core::UMatUsageFlags::USAGE_DEFAULT),
        &mut prev_keypoints,
        &mut prev_descriptors,
        false,
    )?;

    for i in 1..images.len() {
        let mut keypoints = opencv::core::Vector::new();
        let mut descriptors: core::UMat = core::UMat::new(core::UMatUsageFlags::USAGE_DEFAULT);
        let img_umat = images[i].get_umat(
            core::AccessFlag::ACCESS_READ,
            core::UMatUsageFlags::USAGE_DEFAULT,
        )?;

        sift.detect_and_compute(
            &img_umat,
            &core::UMat::new(core::UMatUsageFlags::USAGE_DEFAULT),
            &mut keypoints,
            &mut descriptors,
            false,
        )?;

        if !descriptors.empty() && !prev_descriptors.empty() {
            let mut matcher = features2d::BFMatcher::create(core::NORM_L2, false)?;
            let mut matches = opencv::core::Vector::<core::DMatch>::new();

            // Convert to Mat for matching to avoid Vector<UMat> / getMatVector issues
            let desc_mat = descriptors.get_mat(core::AccessFlag::ACCESS_READ)?;
            let prev_desc_mat = prev_descriptors.get_mat(core::AccessFlag::ACCESS_READ)?;

            let mut train_descriptors = opencv::core::Vector::<core::Mat>::new();
            train_descriptors.push(prev_desc_mat);
            matcher.add(&train_descriptors)?;

            matcher.match_(&desc_mat, &mut matches, &core::Mat::default())?;

            let mut matches_vec = matches.to_vec();
            matches_vec.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());

            // Keep top matches
            let count = (matches_vec.len() as f32 * 0.2) as usize;
            let count = count.max(10).min(matches_vec.len());
            let good_matches = &matches_vec[..count];

            let mut src_pts = opencv::core::Vector::<core::Point2f>::new();
            let mut dst_pts = opencv::core::Vector::<core::Point2f>::new();

            for m in good_matches {
                src_pts.push(keypoints.get(m.query_idx as usize)?.pt());
                dst_pts.push(prev_keypoints.get(m.train_idx as usize)?.pt());
            }

            if src_pts.len() >= 4 {
                let mut inliers: core::UMat = core::UMat::new(core::UMatUsageFlags::USAGE_DEFAULT);
                let t_step_2x3 = calib3d::estimate_affine_partial_2d(
                    &src_pts,
                    &dst_pts,
                    &mut inliers,
                    calib3d::RANSAC,
                    3.0,
                    2000,
                    0.99,
                    10,
                )?;

                if !t_step_2x3.empty() {
                    // Convert 2x3 to 3x3 for multiplication
                    let mut t_step_3x3 = Mat::eye(3, 3, core::CV_64F)?.to_mat()?;
                    for row in 0..2 {
                        for col in 0..3 {
                            *t_step_3x3.at_2d_mut::<f64>(row, col)? =
                                *t_step_2x3.at_2d::<f64>(row, col)?;
                        }
                    }

                    // Accumulate: T_total = T_total * T_step
                    let mut next_t_total = Mat::default();
                    core::gemm(
                        &t_total,
                        &t_step_3x3,
                        1.0,
                        &Mat::default(),
                        0.0,
                        &mut next_t_total,
                        0,
                    )?;
                    t_total = next_t_total;
                }
            }
        }

        // Warp current image to Image 0 frame
        let mut warped_umat: core::UMat = core::UMat::new(core::UMatUsageFlags::USAGE_DEFAULT);
        let mut t_warp_2x3: Mat = Mat::zeros(2, 3, core::CV_64F)?.to_mat()?;
        for row in 0..2 {
            for col in 0..3 {
                *t_warp_2x3.at_2d_mut::<f64>(row, col)? = *t_total.at_2d::<f64>(row, col)?;
            }
        }

        imgproc::warp_affine(
            &img_umat,
            &mut warped_umat,
            &t_warp_2x3,
            images[0].size()?,
            imgproc::INTER_LINEAR,
            core::BORDER_CONSTANT,
            core::Scalar::default(),
        )?;

        let mut aligned_mat = Mat::default();
        warped_umat
            .get_mat(core::AccessFlag::ACCESS_READ)?
            .copy_to(&mut aligned_mat)?;
        images[i] = aligned_mat;

        // Save aligned image
        opencv::imgcodecs::imwrite(
            aligned_dir.join(format!("{:04}.png", i)).to_str().unwrap(),
            &images[i],
            &opencv::core::Vector::new(),
        )?;

        // Update previous for next iteration
        prev_keypoints = keypoints;
        prev_descriptors = descriptors;
    }
    Ok(())
}

pub fn stack_images(images: &[Mat], output_dir: &Path) -> Result<Mat> {
    let result = stack_recursive(images, output_dir, 0)?;

    let final_dir = output_dir.join("final");
    std::fs::create_dir_all(&final_dir)?;
    opencv::imgcodecs::imwrite(
        final_dir.join("result.png").to_str().unwrap(),
        &result,
        &opencv::core::Vector::new(),
    )?;

    Ok(result)
}

fn stack_recursive(images: &[Mat], output_dir: &Path, level: usize) -> Result<Mat> {
    if images.is_empty() {
        return Err(anyhow::anyhow!("No images to stack"));
    }
    if images.len() == 1 {
        return Ok(images[0].clone());
    }

    const BATCH_SIZE: usize = 10;
    const OVERLAP: usize = 2;

    if images.len() <= BATCH_SIZE {
        return stack_images_direct(images);
    }

    let bunches_dir = output_dir.join("bunches");
    std::fs::create_dir_all(&bunches_dir)?;

    let mut intermediate_files = Vec::new();
    let step = BATCH_SIZE - OVERLAP;
    let mut i = 0;
    let mut batch_idx = 0;

    while i < images.len() {
        let end = (i + BATCH_SIZE).min(images.len());
        let batch = &images[i..end];

        let result = stack_images_direct(batch)?;

        let filename = format!("L{}_B{:04}.png", level, batch_idx);
        let path = bunches_dir.join(&filename);

        opencv::imgcodecs::imwrite(
            path.to_str().unwrap(),
            &result,
            &opencv::core::Vector::new(),
        )?;

        intermediate_files.push(path);
        batch_idx += 1;

        // Explicitly drop the Mat to free memory
        drop(result);

        if end == images.len() {
            break;
        }
        i += step;
    }

    // Load intermediate images from disk
    let mut intermediate_images = Vec::new();
    for path in intermediate_files {
        let img = load_image(&path)?;
        intermediate_images.push(img);
    }

    // Recursively stack the intermediate results
    stack_recursive(&intermediate_images, output_dir, level + 1)
}

fn stack_images_direct(images: &[Mat]) -> Result<Mat> {
    if images.is_empty() {
        return Err(anyhow::anyhow!("No images to stack"));
    }
    if images.len() == 1 {
        return Ok(images[0].clone());
    }

    let levels = 4; // Number of pyramid levels
    let mut fused_pyramid: Vec<core::UMat> = Vec::new();
    let mut max_energies: Vec<core::UMat> = Vec::new();

    for (idx, img) in images.iter().enumerate() {
        let mut float_img = core::UMat::new(core::UMatUsageFlags::USAGE_DEFAULT);
        img.get_umat(
            core::AccessFlag::ACCESS_READ,
            core::UMatUsageFlags::USAGE_DEFAULT,
        )?
        .convert_to(&mut float_img, core::CV_32F, 1.0, 0.0)?;

        let current_pyramid = generate_laplacian_pyramid(&float_img, levels)?;

        if idx == 0 {
            // Initialize fused pyramid with the first image's pyramid
            fused_pyramid = current_pyramid.clone();

            // Initialize max energies for Laplacian levels
            for l in 0..levels as usize {
                let layer = &current_pyramid[l];

                // Compute initial energy
                let mut gray = core::UMat::new(core::UMatUsageFlags::USAGE_DEFAULT);
                if layer.channels() == 3 {
                    imgproc::cvt_color(
                        layer,
                        &mut gray,
                        imgproc::COLOR_BGR2GRAY,
                        0,
                        core::AlgorithmHint::ALGO_HINT_DEFAULT,
                    )?;
                } else {
                    gray = layer.clone();
                }

                let mut square = core::UMat::new(core::UMatUsageFlags::USAGE_DEFAULT);
                core::multiply(&gray, &gray, &mut square, 1.0, -1)?;

                let mut energy = core::UMat::new(core::UMatUsageFlags::USAGE_DEFAULT);
                imgproc::gaussian_blur(
                    &square,
                    &mut energy,
                    core::Size::new(5, 5),
                    0.0,
                    0.0,
                    core::BORDER_DEFAULT,
                    core::AlgorithmHint::ALGO_HINT_DEFAULT,
                )?;

                max_energies.push(energy);
            }

            // For the base level (Gaussian), we'll use it for averaging later
            // The last element in fused_pyramid is the base level.
            // We need to ensure it's float for accumulation.
            let base_idx = levels as usize;
            let mut float_base = core::UMat::new(core::UMatUsageFlags::USAGE_DEFAULT);
            fused_pyramid[base_idx].convert_to(&mut float_base, core::CV_32F, 1.0, 0.0)?;
            fused_pyramid[base_idx] = float_base;
        } else {
            // Fuse with current image
            for l in 0..levels as usize {
                let layer = &current_pyramid[l];

                // Compute energy
                let mut gray = core::UMat::new(core::UMatUsageFlags::USAGE_DEFAULT);
                if layer.channels() == 3 {
                    imgproc::cvt_color(
                        layer,
                        &mut gray,
                        imgproc::COLOR_BGR2GRAY,
                        0,
                        core::AlgorithmHint::ALGO_HINT_DEFAULT,
                    )?;
                } else {
                    gray = layer.clone();
                }

                let mut square = core::UMat::new(core::UMatUsageFlags::USAGE_DEFAULT);
                core::multiply(&gray, &gray, &mut square, 1.0, -1)?;

                let mut energy = core::UMat::new(core::UMatUsageFlags::USAGE_DEFAULT);
                imgproc::gaussian_blur(
                    &square,
                    &mut energy,
                    core::Size::new(5, 5),
                    0.0,
                    0.0,
                    core::BORDER_DEFAULT,
                    core::AlgorithmHint::ALGO_HINT_DEFAULT,
                )?;

                // Update fused layer where energy is higher
                let mut mask = core::UMat::new(core::UMatUsageFlags::USAGE_DEFAULT);
                core::compare(&energy, &max_energies[l], &mut mask, core::CMP_GT)?;

                layer.copy_to_masked(&mut fused_pyramid[l], &mask)?;
                energy.copy_to_masked(&mut max_energies[l], &mask)?;
            }

            // Accumulate base level for averaging
            let base_idx = levels as usize;
            let mut float_base = core::UMat::new(core::UMatUsageFlags::USAGE_DEFAULT);
            current_pyramid[base_idx].convert_to(&mut float_base, core::CV_32F, 1.0, 0.0)?;

            let mut next_fused_base = core::UMat::new(core::UMatUsageFlags::USAGE_DEFAULT);
            core::add(
                &fused_pyramid[base_idx],
                &float_base,
                &mut next_fused_base,
                &core::UMat::new(core::UMatUsageFlags::USAGE_DEFAULT),
                -1,
            )?;
            fused_pyramid[base_idx] = next_fused_base;
        }
        // current_pyramid goes out of scope and its UMats are dropped
    }

    // Finalize base level averaging
    let base_idx = levels as usize;
    let mut final_base = core::UMat::new(core::UMatUsageFlags::USAGE_DEFAULT);
    fused_pyramid[base_idx].convert_to(&mut final_base, -1, 1.0 / images.len() as f64, 0.0)?;
    fused_pyramid[base_idx] = final_base;

    // 3. Collapse Pyramid
    let result_umat = collapse_pyramid(&fused_pyramid)?;

    let mut final_img_umat = core::UMat::new(core::UMatUsageFlags::USAGE_DEFAULT);
    result_umat.convert_to(&mut final_img_umat, core::CV_8U, 1.0, 0.0)?;

    let mut final_img = Mat::default();
    final_img_umat
        .get_mat(core::AccessFlag::ACCESS_READ)?
        .copy_to(&mut final_img)?;
    Ok(final_img)
}

fn generate_laplacian_pyramid(img: &core::UMat, levels: i32) -> Result<Vec<core::UMat>> {
    let mut current = img.clone();
    let mut pyramid = Vec::new();

    for _ in 0..levels {
        let mut down = core::UMat::new(core::UMatUsageFlags::USAGE_DEFAULT);
        imgproc::pyr_down(
            &current,
            &mut down,
            core::Size::default(),
            core::BORDER_DEFAULT,
        )?;

        let mut up = core::UMat::new(core::UMatUsageFlags::USAGE_DEFAULT);
        imgproc::pyr_up(&down, &mut up, current.size()?, core::BORDER_DEFAULT)?;

        // Resize up if needed (due to odd dimensions)
        if up.size()? != current.size()? {
            let mut resized = core::UMat::new(core::UMatUsageFlags::USAGE_DEFAULT);
            imgproc::resize(
                &up,
                &mut resized,
                current.size()?,
                0.0,
                0.0,
                imgproc::INTER_LINEAR,
            )?;
            up = resized;
        }

        let mut lap = core::UMat::new(core::UMatUsageFlags::USAGE_DEFAULT);
        core::subtract(
            &current,
            &up,
            &mut lap,
            &core::UMat::new(core::UMatUsageFlags::USAGE_DEFAULT),
            -1,
        )?;
        pyramid.push(lap);

        current = down;
    }
    pyramid.push(current); // Last level is Gaussian
    Ok(pyramid)
}

fn collapse_pyramid(pyramid: &[core::UMat]) -> Result<core::UMat> {
    let mut current = pyramid.last().unwrap().clone();

    for i in (0..pyramid.len() - 1).rev() {
        let mut up: core::UMat = core::UMat::new(core::UMatUsageFlags::USAGE_DEFAULT);
        imgproc::pyr_up(&current, &mut up, pyramid[i].size()?, core::BORDER_DEFAULT)?;

        if up.size()? != pyramid[i].size()? {
            let mut resized: core::UMat = core::UMat::new(core::UMatUsageFlags::USAGE_DEFAULT);
            imgproc::resize(
                &up,
                &mut resized,
                pyramid[i].size()?,
                0.0,
                0.0,
                imgproc::INTER_LINEAR,
            )?;
            up = resized;
        }

        let mut next: core::UMat = core::UMat::new(core::UMatUsageFlags::USAGE_DEFAULT);
        core::add(
            &up,
            &pyramid[i],
            &mut next,
            &core::UMat::new(core::UMatUsageFlags::USAGE_DEFAULT),
            -1,
        )?;
        current = next;
    }
    Ok(current)
}

#[cfg(test)]
mod tests {
    use super::*;
    use opencv::core::{Scalar, Size};

    #[test]
    fn test_stack_images_synthetic() -> Result<()> {
        // Create two images: one focused on left, one on right
        let mut img1 = Mat::new_rows_cols_with_default(100, 100, core::CV_8UC3, Scalar::all(0.0))?;
        let mut img2 = Mat::new_rows_cols_with_default(100, 100, core::CV_8UC3, Scalar::all(0.0))?;

        // Draw white circle on left of img1 (focused)
        imgproc::circle(
            &mut img1,
            core::Point::new(25, 50),
            20,
            Scalar::all(255.0),
            -1,
            imgproc::LINE_8,
            0,
        )?;
        // Blur right side of img1
        imgproc::gaussian_blur(
            &img1.clone(),
            &mut img1,
            Size::new(15, 15),
            0.0,
            0.0,
            core::BORDER_DEFAULT,
            core::AlgorithmHint::ALGO_HINT_DEFAULT,
        )?;
        // Redraw sharp circle on left
        imgproc::circle(
            &mut img1,
            core::Point::new(25, 50),
            20,
            Scalar::all(255.0),
            -1,
            imgproc::LINE_8,
            0,
        )?;

        // Draw white circle on right of img2 (focused)
        imgproc::circle(
            &mut img2,
            core::Point::new(75, 50),
            20,
            Scalar::all(255.0),
            -1,
            imgproc::LINE_8,
            0,
        )?;
        // Blur left side of img2
        imgproc::gaussian_blur(
            &img2.clone(),
            &mut img2,
            Size::new(15, 15),
            0.0,
            0.0,
            core::BORDER_DEFAULT,
            core::AlgorithmHint::ALGO_HINT_DEFAULT,
        )?;
        // Redraw sharp circle on right
        imgproc::circle(
            &mut img2,
            core::Point::new(75, 50),
            20,
            Scalar::all(255.0),
            -1,
            imgproc::LINE_8,
            0,
        )?;

        let images = vec![img1, img2];
        let temp_dir = std::env::temp_dir().join("imagestacker_test_synthetic");
        std::fs::create_dir_all(&temp_dir)?;
        let result = stack_images(&images, &temp_dir)?;

        assert!(!result.empty());
        assert_eq!(result.size()?, Size::new(100, 100));

        // Check that both circles are present (roughly)
        // Center left (25, 50) should be bright
        let p1 = result.at_2d::<core::Vec3b>(50, 25)?;
        assert!(p1[0] > 100);

        // Center right (75, 50) should be bright
        let p2 = result.at_2d::<core::Vec3b>(50, 75)?;
        assert!(p2[0] > 100);

        Ok(())
    }
    #[test]
    fn test_stack_images_batched() -> Result<()> {
        // Create 12 images (batch size 10, overlap 2 -> 2 batches)
        let mut images = Vec::new();
        for _ in 0..12 {
            let mut img =
                Mat::new_rows_cols_with_default(100, 100, core::CV_8UC3, Scalar::all(0.0))?;

            // Draw a circle
            imgproc::circle(
                &mut img,
                core::Point::new(50, 50),
                20,
                Scalar::all(255.0),
                -1,
                imgproc::LINE_8,
                0,
            )?;

            images.push(img);
        }

        let temp_dir = std::env::temp_dir().join("imagestacker_test_batched");
        std::fs::create_dir_all(&temp_dir)?;
        let result = stack_images(&images, &temp_dir)?;

        assert!(!result.empty());
        assert_eq!(result.size()?, Size::new(100, 100));

        // Check center brightness
        let p = result.at_2d::<core::Vec3b>(50, 50)?;
        assert!(p[0] > 200);

        Ok(())
    }
}
