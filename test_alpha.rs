use opencv::prelude::*;
use opencv::{imgcodecs, core};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = "/home/dirk/Bilder/FocusStacking/Meterstab-100_41/bunches/aligned/0001.png";
    let img = imgcodecs::imread(path, imgcodecs::IMREAD_UNCHANGED)?;
    
    println!("Original image:");
    println!("  Size: {}x{}", img.cols(), img.rows());
    println!("  Channels: {}", img.channels());
    println!("  Depth: {}", img.depth());
    println!("  Type: {}", img.typ());
    
    // Extract alpha channel
    if img.channels() == 4 {
        let mut channels = opencv::core::Vector::<core::Mat>::new();
        core::split(&img, &mut channels)?;
        let alpha = channels.get(3)?;
        
        let mut min_val = 0.0;
        let mut max_val = 0.0;
        core::min_max_loc(&alpha, Some(&mut min_val), Some(&mut max_val), None, None, &core::no_array())?;
        
        let mean = core::mean(&alpha, &core::no_array())?;
        
        println!("\nAlpha channel:");
        println!("  Min: {}", min_val);
        println!("  Max: {}", max_val);
        println!("  Mean: {}", mean[0]);
        
        // Count pixels >= 128
        let mut mask_128 = core::Mat::default();
        core::compare(&alpha, &core::Scalar::all(128.0), &mut mask_128, core::CMP_GE)?;
        let count_128 = core::count_non_zero(&mask_128)?;
        println!("  Pixels >= 128: {} ({:.1}%)", count_128, count_128 as f64 / (img.rows() * img.cols()) as f64 * 100.0);
        
        // Count pixels >= 250
        let mut mask_250 = core::Mat::default();
        core::compare(&alpha, &core::Scalar::all(250.0), &mut mask_250, core::CMP_GE)?;
        let count_250 = core::count_non_zero(&mask_250)?;
        println!("  Pixels >= 250: {} ({:.1}%)", count_250, count_250 as f64 / (img.rows() * img.cols()) as f64 * 100.0);
    }
    
    Ok(())
}
