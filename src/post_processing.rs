use anyhow::Result;
use opencv::prelude::*;
use opencv::{core, imgproc};

use crate::config::ProcessingConfig;

/// Apply advanced post-processing to the stacked image
pub fn apply_advanced_processing(mut image: Mat, config: &ProcessingConfig) -> Result<Mat> {
    if config.enable_noise_reduction {
        image = apply_noise_reduction(image, config.noise_reduction_strength)?;
    }

    if config.enable_sharpening {
        image = apply_sharpening(image, config.sharpening_strength)?;
    }

    if config.enable_color_correction {
        image = apply_color_correction(image, config.contrast_boost, config.brightness_boost, config.saturation_boost)?;
    }

    Ok(image)
}

/// Apply noise reduction using bilateral filter
pub fn apply_noise_reduction(image: Mat, strength: f32) -> Result<Mat> {
    log::info!("Applying noise reduction (strength: {:.2})", strength);

    // Bilateral filter works best with 3-channel images
    // If we have 4 channels (BGRA), convert to BGR, process, then convert back
    let has_alpha = image.channels() == 4;
    let working_img = if has_alpha {
        let mut bgr = Mat::default();
        imgproc::cvt_color(
            &image,
            &mut bgr,
            imgproc::COLOR_BGRA2BGR,
            0,
            core::AlgorithmHint::ALGO_HINT_DEFAULT,
        )?;
        bgr
    } else {
        image.clone()
    };

    // Convert to float for processing
    let mut float_img = Mat::default();
    working_img.convert_to(&mut float_img, core::CV_32F, 1.0, 0.0)?;

    // Bilateral filter for noise reduction while preserving edges
    let mut denoised = Mat::default();
    imgproc::bilateral_filter(
        &float_img,
        &mut denoised,
        9, // diameter
        (strength * 50.0) as f64, // sigmaColor - controls color similarity
        (strength * 10.0) as f64, // sigmaSpace - controls spatial proximity
        core::BORDER_DEFAULT,
    )?;

    // Convert back to original type
    let mut result = Mat::default();
    denoised.convert_to(&mut result, core::CV_8U, 1.0, 0.0)?;

    // Convert back to BGRA if needed
    if has_alpha {
        let mut bgra = Mat::default();
        imgproc::cvt_color(
            &result,
            &mut bgra,
            imgproc::COLOR_BGR2BGRA,
            0,
            core::AlgorithmHint::ALGO_HINT_DEFAULT,
        )?;
        Ok(bgra)
    } else {
        Ok(result)
    }
}

/// Apply sharpening using unsharp mask
pub fn apply_sharpening(mut image: Mat, strength: f32) -> Result<Mat> {
    log::info!("Applying sharpening (strength: {:.2})", strength);

    // Convert to float for processing
    let mut float_img = Mat::default();
    image.convert_to(&mut float_img, core::CV_32F, 1.0, 0.0)?;

    // Unsharp masking: blur - original + original = sharpened
    let mut blurred = Mat::default();
    imgproc::gaussian_blur(
        &float_img,
        &mut blurred,
        core::Size::new(0, 0),
        (strength * 3.0) as f64, // sigma
        0.0,
        core::BORDER_DEFAULT,
        core::AlgorithmHint::ALGO_HINT_DEFAULT,
    )?;

    // Calculate: sharpened = original + (original - blurred) * amount
    let mut diff = Mat::default();
    core::subtract(&float_img, &blurred, &mut diff, &core::Mat::default(), -1)?;

    let mut sharpened = Mat::default();
    core::add_weighted(
        &float_img,
        1.0,
        &diff,
        strength as f64,
        0.0,
        &mut sharpened,
        -1,
    )?;

    // Convert back to original type
    sharpened.convert_to(&mut image, core::CV_8U, 1.0, 0.0)?;

    Ok(image)
}

/// Apply color correction (contrast, brightness, saturation)
pub fn apply_color_correction(image: Mat, contrast: f32, brightness: f32, saturation: f32) -> Result<Mat> {
    log::info!("Applying color correction (contrast: {:.2}, brightness: {:.2}, saturation: {:.2})",
               contrast, brightness, saturation);

    // Handle alpha channel if present
    let has_alpha = image.channels() == 4;
    let working_img = if has_alpha {
        let mut bgr = Mat::default();
        imgproc::cvt_color(
            &image,
            &mut bgr,
            imgproc::COLOR_BGRA2BGR,
            0,
            core::AlgorithmHint::ALGO_HINT_DEFAULT,
        )?;
        bgr
    } else {
        image.clone()
    };

    // Convert to HSV for saturation adjustment
    let mut hsv = Mat::default();
    imgproc::cvt_color(
        &working_img,
        &mut hsv,
        imgproc::COLOR_BGR2HSV,
        0,
        core::AlgorithmHint::ALGO_HINT_DEFAULT,
    )?;

    // Split channels
    let mut channels = core::Vector::<Mat>::new();
    core::split(&hsv, &mut channels)?;

    // Adjust saturation (V channel in HSV)
    if saturation != 1.0 {
        let mut float_v = Mat::default();
        channels.get(2)?.convert_to(&mut float_v, core::CV_32F, 1.0, 0.0)?;

        let mut adjusted_v = Mat::default();
        core::multiply(&float_v, &core::Scalar::all(saturation as f64), &mut adjusted_v, 1.0, -1)?;

        let mut new_v = Mat::default();
        adjusted_v.convert_to(&mut new_v, core::CV_8U, 1.0, 0.0)?;
        channels.set(2, new_v)?;
    }

    // Merge channels back
    core::merge(&channels, &mut hsv)?;

    // Convert back to BGR
    let mut result = Mat::default();
    imgproc::cvt_color(
        &hsv,
        &mut result,
        imgproc::COLOR_HSV2BGR,
        0,
        core::AlgorithmHint::ALGO_HINT_DEFAULT,
    )?;

    // Apply contrast and brightness to the entire image
    if contrast != 1.0 || brightness != 0.0 {
        let mut float_img = Mat::default();
        result.convert_to(&mut float_img, core::CV_32F, 1.0, 0.0)?;

        let mut adjusted = Mat::default();
        core::multiply(&float_img, &core::Scalar::all(contrast as f64), &mut adjusted, 1.0, -1)?;
        
        let mut brightness_adjusted = Mat::default();
        core::add(&adjusted, &core::Scalar::all(brightness as f64), &mut brightness_adjusted, &core::Mat::default(), -1)?;

        brightness_adjusted.convert_to(&mut result, core::CV_8U, 1.0, 0.0)?;
    }

    // Convert back to BGRA if needed
    if has_alpha {
        let mut bgra = Mat::default();
        imgproc::cvt_color(
            &result,
            &mut bgra,
            imgproc::COLOR_BGR2BGRA,
            0,
            core::AlgorithmHint::ALGO_HINT_DEFAULT,
        )?;
        Ok(bgra)
    } else {
        Ok(result)
    }
}