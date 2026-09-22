use thiserror::Error;

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON parse error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("Validation error: {0}")]
    Validation(String),
}

pub mod portable;

mod store;
mod structs;

pub use portable::{app_root, ensure_migrated};
pub use store::*;
pub use structs::*;
