use log::{Metadata, Record};
use chrono::Local;
use std::fs::OpenOptions;
use std::io::Write;
use std::sync::Mutex;

pub struct DualLogger {
    env_logger: env_logger::Logger,
    log_file: Mutex<std::fs::File>,
}

impl DualLogger {
    pub fn new(env_logger: env_logger::Logger, log_file: std::fs::File) -> Self {
        Self { 
            env_logger,
            log_file: Mutex::new(log_file),
        }
    }

    pub fn init() {
        // Create or truncate log file in /tmp/
        let log_path = "/tmp/imagestacker.log";
        let log_file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(log_path)
            .expect("Failed to create log file");
        
        println!("Logging to: {}", log_path);
        
        let env_logger = env_logger::Builder::from_default_env()
            .format_timestamp(Some(env_logger::fmt::TimestampPrecision::Seconds))
            .build();
        
        let logger = Box::new(DualLogger::new(env_logger, log_file));
        log::set_boxed_logger(logger).unwrap();
        log::set_max_level(log::LevelFilter::Debug);
    }
}

impl log::Log for DualLogger {
    fn enabled(&self, metadata: &Metadata) -> bool {
        self.env_logger.enabled(metadata)
    }

    fn log(&self, record: &Record) {
        if self.enabled(record.metadata()) {
            // Log to stderr via env_logger
            self.env_logger.log(record);
            
            // Format message
            let timestamp = Local::now().format("%H:%M:%S");
            let level = record.level();
            let target = record.target();
            let message = format!(
                "[{}] {:5} {} - {}",
                timestamp,
                level,
                target,
                record.args()
            );
            
            // Write to file
            if let Ok(mut file) = self.log_file.lock() {
                let _ = writeln!(file, "{}", message);
                let _ = file.flush();
            }
            
            // Also capture to our GUI buffer
            crate::gui::append_log(message);
        }
    }

    fn flush(&self) {
        self.env_logger.flush();
    }
}
