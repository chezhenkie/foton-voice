//! Small helpers shared by the transcription backends.

use std::path::PathBuf;

/// How many threads a transcription backend should run on.
pub(crate) fn inference_threads() -> usize {
    static THREADS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *THREADS.get_or_init(|| {
        let physical = num_cpus::get_physical();
        if physical > 0 {
            physical
        } else {
            std::thread::available_parallelism()
                .map(|n| (n.get() / 2).max(1))
                .unwrap_or(2)
        }
    })
}

/// Base directory for on-device models: `<portable root>/models`.
pub(crate) fn models_base_dir() -> PathBuf {
    fotonvoice_config::portable::app_root().join("models")
}

/// Expand a leading `~` / `~/` to the user's home directory. Any other path is
pub(crate) fn expand_tilde(path: &str) -> PathBuf {
    let home = std::env::var("HOME")
        .map(PathBuf::from)
        .ok()
        .or_else(dirs::home_dir);
    if path == "~" {
        return home.unwrap_or_else(|| PathBuf::from("~"));
    }
    if let Some(rest) = path.strip_prefix("~/") {
        if let Some(h) = home {
            return h.join(rest);
        }
    }
    PathBuf::from(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expand_tilde_home() {
        let home = dirs::home_dir().expect("home dir must be available");
        assert_eq!(expand_tilde("~"), home);
    }

    #[test]
    fn test_expand_tilde_subdir() {
        let home = dirs::home_dir().expect("home dir must be available");
        assert_eq!(expand_tilde("~/.models"), home.join(".models"));
    }

    #[test]
    fn test_expand_tilde_absolute_unchanged() {
        assert_eq!(expand_tilde("/tmp/models"), PathBuf::from("/tmp/models"));
    }

    #[test]
    fn test_expand_tilde_relative_unchanged() {
        assert_eq!(expand_tilde("relative/path"), PathBuf::from("relative/path"));
    }

    #[test]
    fn test_models_base_dir_ends_with_models() {
        assert!(models_base_dir().ends_with("models"));
    }
}
