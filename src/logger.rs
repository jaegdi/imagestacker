use log::{Metadata, Record};
use chrono::Local;

pub struct DualLogger {
    env_logger: env_logger::Logger,
}

impl DualLogger {
    pub fn new(env_logger: env_logger::Logger) -> Self {
        Self { env_logger }
    }

    pub fn init() {
        let env_logger = env_logger::Builder::from_default_env()
            .format_timestamp(Some(env_logger::fmt::TimestampPrecision::Seconds))
            .build();
        
        let logger = Box::new(DualLogger::new(env_logger));
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
            
            // Also capture to our buffer
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
            
            crate::gui::append_log(message);
        }
    }

    fn flush(&self) {
        self.env_logger.flush();
    }
}
