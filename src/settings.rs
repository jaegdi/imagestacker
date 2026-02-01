use crate::config::ProcessingConfig;

pub fn save_settings(config: &ProcessingConfig) -> Result<(), Box<dyn std::error::Error>> {
    let settings_path = dirs::config_dir()
        .unwrap_or_else(|| std::env::temp_dir())
        .join("imagestacker")
        .join("settings.json");

    std::fs::create_dir_all(settings_path.parent().unwrap())?;

    let json = serde_json::to_string_pretty(config)?;
    std::fs::write(&settings_path, json)?;

    log::info!("Settings saved to: {}", settings_path.display());
    Ok(())
}

pub fn load_settings() -> ProcessingConfig {
    let settings_path = dirs::config_dir()
        .unwrap_or_else(|| std::env::temp_dir())
        .join("imagestacker")
        .join("settings.json");

    if settings_path.exists() {
        match std::fs::read_to_string(&settings_path) {
            Ok(json) => match serde_json::from_str(&json) {
                Ok(config) => {
                    log::info!("Settings loaded from: {}", settings_path.display());
                    config
                }
                Err(e) => {
                    log::warn!("Failed to parse settings file: {}. Using defaults.", e);
                    ProcessingConfig::default()
                }
            },
            Err(e) => {
                log::warn!("Failed to read settings file: {}. Using defaults.", e);
                ProcessingConfig::default()
            }
        }
    } else {
        log::info!("No settings file found. Using defaults.");
        ProcessingConfig::default()
    }
}