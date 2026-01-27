use std::fs;

/// Get available system memory in GB
pub fn get_available_memory_gb() -> f64 {
    #[cfg(target_os = "linux")]
    {
        if let Ok(content) = fs::read_to_string("/proc/meminfo") {
            for line in content.lines() {
                if line.starts_with("MemAvailable:") {
                    if let Some(kb_str) = line.split_whitespace().nth(1) {
                        if let Ok(kb) = kb_str.parse::<f64>() {
                            return kb / 1024.0 / 1024.0; // Convert KB to GB
                        }
                    }
                }
            }
        }
    }
    
    #[cfg(target_os = "macos")]
    {
        use std::process::Command;
        if let Ok(output) = Command::new("sysctl")
            .arg("-n")
            .arg("hw.memsize")
            .output()
        {
            if let Ok(size_str) = String::from_utf8(output.stdout) {
                if let Ok(bytes) = size_str.trim().parse::<f64>() {
                    return bytes / 1024.0 / 1024.0 / 1024.0; // Convert bytes to GB
                }
            }
        }
    }
    
    #[cfg(target_os = "windows")]
    {
        // Simplified - would need winapi for accurate measurement
        8.0 // Default fallback
    }
    
    // Fallback: assume 8GB available
    8.0
}

/// Calculate optimal batch sizes based on available RAM and image size
#[derive(Debug, Clone)]
pub struct BatchSizeConfig {
    pub sharpness_batch_size: usize,
    pub feature_batch_size: usize,
    pub warp_batch_size: usize,
    pub stacking_batch_size: usize,
}

impl BatchSizeConfig {
    pub fn calculate_optimal(available_gb: f64, avg_image_size_mb: f64) -> Self {
        println!("System Info: {:.2} GB RAM available", available_gb);
        println!("Average image size: {:.2} MB", avg_image_size_mb);
        
        // Conservative allocation: use max 50% of available RAM
        let usable_gb = available_gb * 0.5;
        let usable_mb = usable_gb * 1024.0;
        
        // Calculate how many images we can safely load at once
        let max_images_in_memory = (usable_mb / avg_image_size_mb).max(4.0) as usize;
        
        let sharpness_batch_size = max_images_in_memory.min(16).max(4);
        let feature_batch_size = (max_images_in_memory * 2).min(32).max(8);
        let warp_batch_size = max_images_in_memory.min(24).max(8);
        let stacking_batch_size = (max_images_in_memory / 2).min(16).max(6);
        
        println!("Calculated batch sizes:");
        println!("  - Sharpness: {} images", sharpness_batch_size);
        println!("  - Feature detection: {} images", feature_batch_size);
        println!("  - Warping: {} images", warp_batch_size);
        println!("  - Stacking: {} images", stacking_batch_size);
        
        Self {
            sharpness_batch_size,
            feature_batch_size,
            warp_batch_size,
            stacking_batch_size,
        }
    }
    
    pub fn default_config() -> Self {
        Self {
            sharpness_batch_size: 8,
            feature_batch_size: 16,
            warp_batch_size: 16,
            stacking_batch_size: 12,
        }
    }
}

/// Estimate image size in MB based on dimensions
#[allow(dead_code)]
pub fn estimate_image_size_mb(width: i32, height: i32, channels: i32) -> f64 {
    // Raw pixel data + overhead for OpenCV Mat structures
    let raw_size = (width * height * channels) as f64;
    let mb = raw_size / (1024.0 * 1024.0);
    mb * 1.5 // Add 50% overhead for processing buffers
}
