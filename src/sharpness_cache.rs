//! Sharpness data caching
//!
//! This module handles saving and loading sharpness detection results to/from YAML files.
//! This avoids repeating expensive sharpness calculations for the same images.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Sharpness information for a single image
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SharpnessInfo {
    /// Original image filename
    pub image_filename: String,
    
    /// Maximum regional sharpness score (best region)
    pub max_regional_sharpness: f64,
    
    /// Global sharpness across entire image
    pub global_sharpness: f64,
    
    /// Number of sharp regions (or stored as f64 for flexibility)
    pub sharp_region_count: f64,
    
    /// Grid size used for regional analysis (e.g., 8 for 8x8 grid)
    pub grid_size: usize,
    
    /// Timestamp when sharpness was calculated
    pub timestamp: String,
    
    /// Image dimensions (width, height)
    pub image_size: (i32, i32),
}

impl SharpnessInfo {
    /// Create new SharpnessInfo
    pub fn new(
        image_filename: String,
        max_regional: f64,
        global_sharpness: f64,
        sharp_region_count: f64,
        grid_size: usize,
        image_size: (i32, i32),
    ) -> Self {
        Self {
            image_filename,
            max_regional_sharpness: max_regional,
            global_sharpness,
            sharp_region_count,
            grid_size,
            timestamp: chrono::Local::now().to_rfc3339(),
            image_size,
        }
    }
    
    /// Save sharpness info to YAML file
    pub fn save_to_file(&self, yaml_path: &Path) -> Result<()> {
        let yaml_content = serde_yaml::to_string(self)?;
        std::fs::write(yaml_path, yaml_content)?;
        Ok(())
    }
    
    /// Load sharpness info from YAML file
    pub fn load_from_file(yaml_path: &Path) -> Result<Self> {
        let yaml_content = std::fs::read_to_string(yaml_path)?;
        let info: SharpnessInfo = serde_yaml::from_str(&yaml_content)?;
        Ok(info)
    }
    
    /// Get YAML filename for an image file
    /// Example: "image.jpg" -> "image.yml"
    pub fn yaml_filename_for_image(image_path: &Path) -> String {
        image_path
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string()
            + ".yml"
    }
    
    /// Check if sharpness cache exists for an image
    #[allow(dead_code)]
    pub fn cache_exists(image_path: &Path, sharpness_dir: &Path) -> bool {
        let yaml_name = Self::yaml_filename_for_image(image_path);
        let yaml_path = sharpness_dir.join(yaml_name);
        yaml_path.exists()
    }
}

/// Create or clear sharpness cache directory
pub fn prepare_sharpness_cache_dir(output_dir: &Path) -> Result<PathBuf> {
    let sharpness_dir = output_dir.join("sharpness");
    
    if sharpness_dir.exists() {
        // Clear existing YAML files
        for entry in std::fs::read_dir(&sharpness_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_file() && path.extension().map_or(false, |ext| ext == "yml" || ext == "yaml") {
                std::fs::remove_file(&path)?;
            }
        }
    } else {
        // Create new directory
        std::fs::create_dir_all(&sharpness_dir)?;
    }
    
    Ok(sharpness_dir)
}

/// Load all cached sharpness info from directory
#[allow(dead_code)]
pub fn load_cached_sharpness(sharpness_dir: &Path) -> Result<Vec<(PathBuf, SharpnessInfo)>> {
    let mut results = Vec::new();
    
    if !sharpness_dir.exists() {
        return Ok(results);
    }
    
    for entry in std::fs::read_dir(sharpness_dir)? {
        let entry = entry?;
        let path = entry.path();
        
        if path.is_file() && path.extension().map_or(false, |ext| ext == "yml" || ext == "yaml") {
            if let Ok(info) = SharpnessInfo::load_from_file(&path) {
                results.push((path, info));
            }
        }
    }
    
    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    
    #[test]
    fn test_yaml_filename_conversion() {
        let image_path = PathBuf::from("test_image.jpg");
        let yaml_name = SharpnessInfo::yaml_filename_for_image(&image_path);
        assert_eq!(yaml_name, "test_image.yml");
    }
}
