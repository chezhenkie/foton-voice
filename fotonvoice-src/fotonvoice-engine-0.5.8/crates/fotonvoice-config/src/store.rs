use std::path::{Path, PathBuf};

use crate::{AppConfig, ConfigError};

/// Lift a HuggingFace token stored per engine onto the single `tts.hf_token`,
fn migrate_hf_token(data: &mut AppConfig) -> bool {
    let legacy = data
        .tts
        .pocket_tts
        .legacy_hf_token
        .take()
        .or_else(|| data.tts.breeze_tts_2.legacy_hf_token.take());
    data.tts.breeze_tts_2.legacy_hf_token = None;

    let Some(token) = legacy else { return false };
    if data.tts.hf_token.is_none() {
        data.tts.hf_token = Some(token);
    }
    true
}

/// Rename `<base>/fotonvoice-engine/pocket-tts-voices` to `<base>/fotonvoice-engine/cloned-tts-voices`,
fn migrate_cloned_voices_dir_at(base: &std::path::Path) -> bool {
    let old_dir = base.join("fotonvoice-engine").join("pocket-tts-voices");
    let new_dir = base.join("fotonvoice-engine").join("cloned-tts-voices");
    if old_dir.exists() && !new_dir.exists() {
        match std::fs::rename(&old_dir, &new_dir) {
            Ok(()) => return true,
            Err(e) => tracing::error!(
                "Failed to migrate {} to {}: {e}",
                old_dir.display(),
                new_dir.display()
            ),
        }
    }
    false
}

fn migrate_cloned_voices_dir() {
    if let Some(base) = dirs::data_local_dir() {
        migrate_cloned_voices_dir_at(&base);
    }
}

/// Read a config file, keeping every section that parses.
fn parse_tolerant_report(text: &str) -> (AppConfig, bool) {
    let whole_file_error = match serde_json::from_str::<AppConfig>(text) {
        Ok(cfg) => return (cfg, false),
        Err(e) => e,
    };

    let Ok(serde_json::Value::Object(file)) = serde_json::from_str::<serde_json::Value>(text)
    else {
        tracing::warn!("Failed to load config, using defaults: {whole_file_error}");
        return (AppConfig::default(), true);
    };

    tracing::warn!(
        "Config did not load as a whole ({whole_file_error}); \
         recovering it section by section"
    );

    let Ok(serde_json::Value::Object(mut merged)) = serde_json::to_value(AppConfig::default())
    else {
        return (AppConfig::default(), true);
    };

    for (key, value) in file {
        if !merged.contains_key(&key) {
            continue;
        }
        let mut candidate = merged.clone();
        candidate.insert(key.clone(), value);
        match serde_json::from_value::<AppConfig>(serde_json::Value::Object(candidate.clone())) {
            Ok(_) => merged = candidate,
            Err(e) => tracing::warn!(
                "Config section '{key}' could not be read ({e}); it falls back to \
                 defaults, and the rest of the file is kept"
            ),
        }
    }

    (serde_json::from_value(serde_json::Value::Object(merged)).unwrap_or_default(), false)
}


pub struct Config {
    pub data: AppConfig,
    path: PathBuf,
}

impl Config {
    pub fn config_path() -> PathBuf {
        crate::portable::app_root().join("config.json")
    }

    pub fn load() -> Self {
        let path = Self::config_path();
        let mut data = if path.exists() {
            match std::fs::read_to_string(&path).map_err(ConfigError::Io) {
                Ok(text) => {
                    let (parsed, fatal) = parse_tolerant_report(&text);
                    if fatal {
                        tracing::warn!(
                            "Config at {} is unreadable; quarantining it and \
                             starting from defaults",
                            path.display()
                        );
                        Self::quarantine(&path);
                        parsed
                    } else {
                        parsed
                    }
                }
                Err(e) => {
                    tracing::warn!("Failed to read config, using defaults: {e}");
                    AppConfig::default()
                }
            }
        } else {
            AppConfig::default()
        };

        if let Some(legacy_notif) = data.features.show_notification {
            data.ui.show_notification = legacy_notif;
            data.features.show_notification = None;
            let clean_config = Self { data: data.clone(), path: path.clone() };
            if let Err(e) = clean_config.save() {
                tracing::error!("Failed to save clean migrated config: {e}");
            }
        }

        let needs_escape_fix = data.tts.stop_key.iter().any(|k| k == "KEY_ESCAPE");
        if needs_escape_fix {
            data.tts.stop_key = data.tts.stop_key
                .into_iter()
                .map(|k| if k == "KEY_ESCAPE" { "KEY_ESC".to_string() } else { k })
                .collect();
            let clean_config = Self { data: data.clone(), path: path.clone() };
            if let Err(e) = clean_config.save() {
                tracing::error!("Failed to save migrated stop_key: {e}");
            }
        }

        if data.openai.timeout_secs == 8 {
            data.openai.timeout_secs = 30;
            let clean_config = Self { data: data.clone(), path: path.clone() };
            if let Err(e) = clean_config.save() {
                tracing::error!("Failed to save migrated OpenAI timeout: {e}");
            }
        }

        if let Some(legacy_prompt) = data.openai.custom_prompt.take() {
            if !legacy_prompt.trim().is_empty() {
                data.openai.user_prompt = legacy_prompt;
                data.openai.system_prompt = String::new();
            }
            let clean_config = Self { data: data.clone(), path: path.clone() };
            if let Err(e) = clean_config.save() {
                tracing::error!("Failed to save migrated OpenAI custom prompt: {e}");
            }
        }

        if migrate_hf_token(&mut data) {
            let migrated = Self { data: data.clone(), path: path.clone() };
            if let Err(e) = migrated.save() {
                tracing::error!("Failed to save migrated HuggingFace token: {e}");
            }
        }

        migrate_cloned_voices_dir();

        data.tts.speed = if data.tts.speed <= 0.0 {
            1.0
        } else {
            data.tts.speed.clamp(0.90, 1.10)
        };

        Self { data, path }
    }

    /// Move an unreadable config file aside as `<name>.bad`.
    fn quarantine(path: &Path) {
        let bad = path.with_file_name(format!(
            "{}.bad",
            path.file_name().and_then(|n| n.to_str()).unwrap_or("config.json")
        ));
        if bad.exists() {
            let _ = std::fs::remove_file(&bad);
        }
        match std::fs::rename(path, &bad) {
            Ok(()) => tracing::warn!("Quarantined unreadable config as {}", bad.display()),
            Err(e) => tracing::error!("Failed to quarantine unreadable config: {e}"),
        }
    }

    pub fn save(&self) -> Result<(), ConfigError> {
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let json = serde_json::to_string_pretty(&self.data)?;
        #[cfg(unix)]
        {
            use std::io::Write;
            use std::os::unix::fs::OpenOptionsExt;
            let mut f = std::fs::OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .mode(0o600)
                .open(&self.path)?;
            f.write_all(json.as_bytes())?;
        }
        #[cfg(not(unix))]
        std::fs::write(&self.path, json)?;
        Ok(())
    }

    pub fn reload(&mut self) {
        *self = Self::load();
    }
}

impl Default for Config {
    fn default() -> Self {
        Self::load()
    }
}


/// Find the executable `name` the same way spawning it will.
pub fn find_in_path(name: &str) -> Option<PathBuf> {
    let search_name: std::borrow::Cow<str> = if cfg!(target_os = "windows") && !name.contains('.') {
        format!("{name}.exe").into()
    } else {
        name.into()
    };

    let mut dirs: Vec<PathBuf> = Vec::new();

    #[cfg(target_os = "windows")]
    {
        if let Ok(exe) = std::env::current_exe() {
            if let Some(parent) = exe.parent() {
                dirs.push(parent.to_path_buf());
            }
        }
        if let Some(root) = std::env::var_os("SystemRoot") {
            let root = PathBuf::from(root);
            dirs.push(root.join("System32"));
            dirs.push(root);
        }
    }

    if let Some(paths) = std::env::var_os("PATH") {
        dirs.extend(std::env::split_paths(&paths));
    }

    dirs.into_iter()
        .map(|dir| dir.join(search_name.as_ref()))
        .find(|p| p.is_file())
}


static VALID_MODEL_SIZES: &[&str] = &[
    "small", "small.en", "medium", "medium.en", "large-v3", "large-v3-turbo",
    "small-q8", "small.en-q8", "medium-q8", "medium.en-q8", "large-v3-turbo-q8",
];

pub fn validate(cfg: &AppConfig) -> Vec<String> {
    let mut errors = Vec::new();

    if !VALID_MODEL_SIZES.contains(&cfg.engine.whisper_cpp.model_size.as_str())
        && !cfg.engine.whisper_cpp.model_size.ends_with(".bin")
        && !std::path::Path::new(&cfg.engine.whisper_cpp.model_size).is_absolute()
    {
        errors.push(format!(
            "Unknown whisper_cpp model_size '{}'. Valid: {:?}",
            cfg.engine.whisper_cpp.model_size, VALID_MODEL_SIZES
        ));
    }

    if !["auto", "cuda", "vulkan", "cpu"]
        .contains(&cfg.engine.whisper_cpp.device.as_str())
    {
        errors.push(format!(
            "Invalid whisper_cpp device '{}'. Use: auto, cuda, vulkan, cpu",
            cfg.engine.whisper_cpp.device
        ));
    }

    if cfg.audio.vad_threshold < 0.0 || cfg.audio.vad_threshold > 1.0 {
        errors.push(format!(
            "vad_threshold {} out of range [0.0, 1.0]",
            cfg.audio.vad_threshold
        ));
    }

    errors
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::structs::*;
    #[test]
    fn a_name_on_path_is_found() {
        let dir = tempfile::tempdir().unwrap();
        let extension = if cfg!(target_os = "windows") { ".exe" } else { "" };
        let name = format!("fotonvoice_path_probe{extension}");
        std::fs::write(dir.path().join(&name), b"").unwrap();

        let _guard = PathGuard::prepending(dir.path());
        assert_eq!(
            find_in_path("fotonvoice_path_probe").as_deref(),
            Some(dir.path().join(&name).as_path())
        );
    }

    #[test]
    fn a_name_that_is_on_no_searched_directory_is_not_found() {
        let dir = tempfile::tempdir().unwrap();
        let _guard = PathGuard::prepending(dir.path());
        assert_eq!(find_in_path("fotonvoice_definitely_not_here_9f3a"), None);
    }

    /// The directories Windows searches before `PATH` are exactly the gap this
    #[cfg(target_os = "windows")]
    #[test]
    fn a_system_directory_tool_is_found_even_when_path_is_empty() {
        let _guard = PathGuard::replacing_with_nothing();
        let found = find_in_path("where").expect("where.exe lives in System32");
        assert!(found.is_file());
    }

    static PATH_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Puts a directory at the front of `PATH` and restores it on drop.
    #[allow(dead_code)]
    struct PathGuard<'a>(Option<std::ffi::OsString>, std::sync::MutexGuard<'a, ()>);

    impl<'a> PathGuard<'a> {
        #[cfg(target_os = "windows")]
        fn replacing_with_nothing() -> Self {
            let guard = PATH_MUTEX.lock().unwrap();
            let previous = std::env::var_os("PATH");
            std::env::set_var("PATH", "");
            Self(previous, guard)
        }

        fn prepending(dir: &std::path::Path) -> Self {
            let guard = PATH_MUTEX.lock().unwrap();
            let previous = std::env::var_os("PATH");
            let mut entries = vec![dir.to_path_buf()];
            if let Some(existing) = &previous {
                entries.extend(std::env::split_paths(existing));
            }
            std::env::set_var("PATH", std::env::join_paths(entries).unwrap());
            Self(previous, guard)
        }
    }

    impl Drop for PathGuard<'_> {
        fn drop(&mut self) {
            match self.0.take() {
                Some(previous) => std::env::set_var("PATH", previous),
                None => std::env::remove_var("PATH"),
            }
        }
    }



    /// A file with nothing wrong with it must take the ordinary path and come
    #[test]
    fn a_valid_config_parses_whole() {
        let (cfg, _) = parse_tolerant_report(
            r#"{"engine": {"backend": "moonshine",
                           "whisper_cpp": {"model_dir": "", "model_size": "small",
                                           "device": "auto", "threads": 0},
                           "moonshine": {"model_size": "base", "language": "en"}},
                "audio": {"vad_threshold": 0.65, "input_device_index": null,
                          "evdev_device": null, "noise_suppression": true,
                          "gain": 1.6, "dynamic_stream": true}}"#,
        );
        assert_eq!(cfg.engine.backend, BackendChoice::Moonshine);
        assert_eq!(cfg.engine.whisper_cpp.model_size, "small");
        assert_eq!(cfg.audio.gain, 1.6);
    }

    /// The failure this exists for: `whisper_cpp` where the enum spells it
    #[test]
    fn one_bad_section_does_not_take_the_rest_of_the_file_with_it() {
        let (cfg, _) = parse_tolerant_report(
            r#"{"engine": {"backend": "whisper_cpp"},
                "audio": {"vad_threshold": 0.65, "input_device_index": null,
                          "evdev_device": null, "noise_suppression": true,
                          "gain": 1.6, "dynamic_stream": true},
                "ui": {"show_overlay": false, "overlay_style": "pulse",
                       "overlay_position": "top", "overlay_monitor": "primary",
                       "auto_show_settings": false, "show_notification": false,
                       "show_command_overlay": true, "command_overlay_duration_secs": 3,
                       "setup_completed": true}}"#,
        );

        assert_eq!(cfg.audio.gain, 1.6);
        assert!(cfg.audio.noise_suppression);
        assert!(!cfg.ui.show_overlay);

        assert_eq!(cfg.engine.backend, BackendChoice::default());
    }

    /// A key the running build knows nothing about is left alone rather than
    #[test]
    fn an_unknown_top_level_key_is_ignored() {
        let (cfg, _) = parse_tolerant_report(
            r#"{"engine": {"backend": "whisper_cpp"},
                "some_future_section": {"whatever": 1},
                "features": {"remove_fillers": false, "custom_vocabulary": [],
                             "spoken_punctuation": true, "auto_format_lists": true,
                             "snippets": {}}}"#,
        );
        assert!(!cfg.features.remove_fillers);
    }

    /// Not a JSON object at all - there are no sections to recover, so this is
    #[test]
    fn a_file_that_is_not_an_object_falls_back_to_defaults() {
        let (cfg, _) = parse_tolerant_report("[1, 2, 3]");
        assert_eq!(cfg.engine.backend, BackendChoice::default());
        assert_eq!(cfg.audio.gain, AudioConfig::default().gain);
    }

    /// The wholesale fallback must be reported, so the caller can quarantine
    #[test]
    fn a_fatally_unreadable_file_is_reported_for_quarantine() {
        let (cfg, fatal) = parse_tolerant_report("{not json at all");
        assert!(fatal);
        assert_eq!(cfg.engine.backend, BackendChoice::default());
    }

    /// A file that is recovered section by section is not quarantined: it is
    #[test]
    fn a_recovered_file_is_not_reported_as_fatal() {
        let (_, fatal) = parse_tolerant_report(r#"{"engine": {"backend": "whisper_cpp"}}"#);
        assert!(!fatal);
    }

    /// Quarantine renames the unreadable file to `<name>.bad` and frees the
    #[test]
    fn quarantine_renames_the_bad_file_aside() {
        let dir = std::env::temp_dir().join(format!("fotonvoice-config-quarantine-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let src = dir.join("config.json");
        std::fs::write(&src, "\u{feff}{oops").unwrap();
        Config::quarantine(&src);
        assert!(!src.exists());
        let bad = dir.join("config.json.bad");
        assert!(bad.exists());
        std::fs::write(&src, "{oops again").unwrap();
        Config::quarantine(&src);
        assert!(std::fs::read_to_string(&bad).unwrap() == "{oops again");
        let _ = std::fs::remove_file(&bad);
        let _ = std::fs::remove_dir(&dir);
    }

    fn tts_json(body: &str) -> TtsConfig {
        serde_json::from_str(body).expect("tts config should parse")
    }

    /// A config written when each engine carried its own copy of the token
    #[test]
    fn migrates_per_engine_hf_tokens_onto_one_key() {
        let mut data = AppConfig::default();
        data.tts = tts_json(
            r#"{"enabled": true, "engine": "pocket_tts", "voice": "v",
                "stop_key": ["KEY_ESC"], "response_overlay": true,
                "pocket_tts": {"voice": "alba", "hf_token": "hf_from_pocket"},
                "breeze_tts_2": {"hf_token": "hf_from_pocket"}}"#,
        );
        assert_eq!(
            data.tts.pocket_tts.legacy_hf_token.as_deref(),
            Some("hf_from_pocket"),
            "the old location must still parse"
        );

        assert!(migrate_hf_token(&mut data));

        assert_eq!(data.tts.hf_token.as_deref(), Some("hf_from_pocket"));
        assert!(data.tts.pocket_tts.legacy_hf_token.is_none());
        assert!(data.tts.breeze_tts_2.legacy_hf_token.is_none());

        let written = serde_json::to_string(&data.tts).unwrap();
        assert_eq!(
            written.matches("hf_token").count(),
            1,
            "the token must be stored once, not per engine: {written}"
        );
    }

    /// A token set only on Breeze is lifted too - either copy will do.
    #[test]
    fn migrates_a_breeze_only_token() {
        let mut data = AppConfig::default();
        data.tts = tts_json(
            r#"{"enabled": false, "engine": "espeak", "voice": "v", "stop_key": [],
                "response_overlay": true, "breeze_tts_2": {"hf_token": "hf_from_breeze"}}"#,
        );

        assert!(migrate_hf_token(&mut data));
        assert_eq!(data.tts.hf_token.as_deref(), Some("hf_from_breeze"));
    }

    /// A config that already has the single key keeps it, and needs no rewrite.
    #[test]
    fn a_config_with_one_token_is_left_alone() {
        let mut data = AppConfig::default();
        data.tts = tts_json(
            r#"{"enabled": false, "engine": "espeak", "voice": "v", "stop_key": [],
                "response_overlay": true, "hf_token": "hf_single"}"#,
        );

        assert!(!migrate_hf_token(&mut data), "nothing to migrate");
        assert_eq!(data.tts.hf_token.as_deref(), Some("hf_single"));
        assert_eq!(
            serde_json::to_string(&data.tts).unwrap().matches("hf_token").count(),
            1
        );
    }

    /// The single key wins over a stale per-engine copy rather than being
    #[test]
    fn the_single_token_wins_over_a_legacy_copy() {
        let mut data = AppConfig::default();
        data.tts = tts_json(
            r#"{"enabled": false, "engine": "espeak", "voice": "v", "stop_key": [],
                "response_overlay": true, "hf_token": "hf_current",
                "pocket_tts": {"hf_token": "hf_stale"}}"#,
        );

        assert!(migrate_hf_token(&mut data));
        assert_eq!(data.tts.hf_token.as_deref(), Some("hf_current"));
        assert!(data.tts.pocket_tts.legacy_hf_token.is_none());
    }

    /// The shared voice-clip folder is renamed from its old Pocket-TTS-only
    #[test]
    fn migrates_the_cloned_voices_folder() {
        let base = tempfile::tempdir().unwrap();
        let old_dir = base.path().join("fotonvoice-engine").join("pocket-tts-voices");
        std::fs::create_dir_all(&old_dir).unwrap();
        std::fs::write(old_dir.join("narrator.wav"), b"fake wav data").unwrap();

        assert!(migrate_cloned_voices_dir_at(base.path()));

        let new_dir = base.path().join("fotonvoice-engine").join("cloned-tts-voices");
        assert!(!old_dir.exists());
        assert!(new_dir.join("narrator.wav").exists());
    }

    /// With no old folder there is nothing to do, and an existing new folder
    #[test]
    fn cloned_voices_migration_is_a_noop_without_the_old_folder() {
        let base = tempfile::tempdir().unwrap();
        assert!(!migrate_cloned_voices_dir_at(base.path()));

        let old_dir = base.path().join("fotonvoice-engine").join("pocket-tts-voices");
        let new_dir = base.path().join("fotonvoice-engine").join("cloned-tts-voices");
        std::fs::create_dir_all(&old_dir).unwrap();
        std::fs::write(old_dir.join("a.wav"), b"old").unwrap();
        std::fs::create_dir_all(&new_dir).unwrap();
        std::fs::write(new_dir.join("b.wav"), b"new").unwrap();

        assert!(
            !migrate_cloned_voices_dir_at(base.path()),
            "must not clobber an existing new folder"
        );
        assert!(new_dir.join("b.wav").exists());
        assert!(old_dir.join("a.wav").exists());
    }


    /// Configs written before the Backend dropdown lost its "Auto-detect"
    #[test]
    fn legacy_auto_backend_loads_as_whisper_cpp() {
        let parsed: BackendChoice = serde_json::from_str(r#""auto""#).unwrap();
        assert_eq!(parsed, BackendChoice::WhisperCpp);
        assert_eq!(BackendChoice::default(), BackendChoice::Parakeet);
    }

    #[test]
    fn backend_choice_serializes_kebab_case() {
        assert_eq!(
            serde_json::to_string(&BackendChoice::WhisperCpp).unwrap(),
            r#""whisper-cpp""#
        );
        assert_eq!(
            serde_json::to_string(&BackendChoice::Moonshine).unwrap(),
            r#""moonshine""#
        );
        assert_eq!(
            serde_json::to_string(&BackendChoice::Parakeet).unwrap(),
            r#""parakeet""#
        );
        assert_eq!(
            serde_json::to_string(&BackendChoice::RemoteOpenAi).unwrap(),
            r#""remote-openai""#
        );
        let parsed: BackendChoice = serde_json::from_str(r#""remote-openai""#).unwrap();
        assert_eq!(parsed, BackendChoice::RemoteOpenAi);
        let parsed_alias: BackendChoice = serde_json::from_str(r#""openai-compatible""#).unwrap();
        assert_eq!(parsed_alias, BackendChoice::RemoteOpenAi);
    }

    #[test]
    fn test_default_config_values() {
        let cfg = AppConfig::default();
        assert!(!cfg.ui.auto_show_settings);
        assert!(!cfg.ui.show_notification);
        assert_eq!(cfg.ui.overlay_position, "center");
        assert_eq!(cfg.ui.overlay_monitor, "primary");
        assert!(cfg.features.show_notification.is_none());
    }

    #[test]
    fn test_legacy_notification_migration() {
        let legacy_json = r#"{
            "engine": {
                "backend": "auto",
                "whisper_cpp": {
                    "model_dir": "",
                    "model_size": "large-v3",
                    "device": "auto",
                    "threads": 0
                },
                "moonshine": {
                    "model_size": "base",
                    "language": "en"
                }
            },
            "audio": {
                "vad_threshold": 0.5,
                "input_device_index": null,
                "evdev_device": null,
                "noise_suppression": false,
                "gain": 1.0,
                "dynamic_stream": true
            },
            "ui": {
                "show_overlay": true,
                "overlay_style": "voice_card"
            },
            "features": {
                "remove_fillers": true,
                "custom_vocabulary": [],
                "spoken_punctuation": true,
                "auto_format_lists": true,
                "show_notification": true,
                "snippets": {}
            },
            "openai": {
                "enabled": false,
                "model": "llama3.2:1b",
                "mode": "clean",
                "custom_prompt": null,
                "endpoint": "http://localhost:11434",
                "timeout_secs": 8
            },
            "tts": {
                "enabled": false,
                "engine": "piper",
                "voice": "en-us-lessac-medium",
                "stop_key": ["KEY_ESC"],
                "response_overlay": true
            },
            "mcp": {
                "server_enabled": false,
                "record_timeout": 15.0
            }
        }"#;

        let parsed: AppConfig = serde_json::from_str(legacy_json).unwrap();
        assert!(parsed.features.show_notification.is_some());
        assert_eq!(parsed.features.show_notification, Some(true));

        let temp_dir = tempfile::tempdir().unwrap();
        let config_file_path = temp_dir.path().join("config.json");
        std::fs::write(&config_file_path, legacy_json).unwrap();

        let config = Config {
            data: parsed,
            path: config_file_path.clone(),
        };

        let _migrated_config = Config::load();
        
        let mut custom_config = Config {
            data: config.data.clone(),
            path: config_file_path.clone(),
        };
        if let Some(legacy_notif) = custom_config.data.features.show_notification {
            custom_config.data.ui.show_notification = legacy_notif;
            custom_config.data.features.show_notification = None;
            custom_config.save().unwrap();
        }

        assert!(custom_config.data.ui.show_notification);
        assert!(custom_config.data.features.show_notification.is_none());

        let re_read_content = std::fs::read_to_string(&config_file_path).unwrap();
        assert!(re_read_content.contains(r#""show_notification": true"#));
        assert!(!re_read_content.contains(r#""features": {
    "remove_fillers": true,
    "custom_vocabulary": [],
    "spoken_punctuation": true,
    "auto_format_lists": true,
    "show_notification": true"#));
    }

    #[test]
    fn test_ui_config_position_monitor_defaults() {
        let partial_json = r#"{
            "show_overlay": true,
            "overlay_style": "waveform",
            "auto_show_settings": true,
            "show_notification": false
        }"#;

        let parsed: UiConfig = serde_json::from_str(partial_json).unwrap();
        assert_eq!(parsed.overlay_position, "center");
        assert_eq!(parsed.overlay_monitor, "primary");
    }

    #[test]
    fn test_openai_prompt_defaults_for_legacy_config() {
        let legacy_openai = r#"{
            "enabled": true,
            "model": "llama3.2:1b",
            "mode": "clean",
            "custom_prompt": null,
            "endpoint": "http://localhost:11434",
            "timeout_secs": 30
        }"#;

        let parsed: OpenAiConfig = serde_json::from_str(legacy_openai).unwrap();
        assert_eq!(parsed.user_prompt, "{text}");
        assert!(parsed.system_prompt.contains("Fix grammar"));
        assert_eq!(parsed.api_key, None);
    }

    #[test]
    fn test_openai_timeout_migration() {
        let mut default_cfg = AppConfig::default();
        default_cfg.openai.timeout_secs = 8;

        let legacy_json = serde_json::to_string(&default_cfg).unwrap();

        let parsed: AppConfig = serde_json::from_str(&legacy_json).unwrap();
        assert_eq!(parsed.openai.timeout_secs, 8);

        let temp_dir = tempfile::tempdir().unwrap();
        let config_file_path = temp_dir.path().join("config.json");
        std::fs::write(&config_file_path, &legacy_json).unwrap();

        let mut config = Config {
            data: parsed,
            path: config_file_path.clone(),
        };

        if config.data.openai.timeout_secs == 8 {
            config.data.openai.timeout_secs = 30;
            config.save().unwrap();
        }

        assert_eq!(config.data.openai.timeout_secs, 30);

        let re_read_content = std::fs::read_to_string(&config_file_path).unwrap();
        assert!(re_read_content.contains(r#""timeout_secs": 30"#));
    }

    #[test]
    fn test_breeze_tts_2_serde() {
        let engine = TtsEngine::BreezeTts2;
        let json = serde_json::to_string(&engine).unwrap();
        assert_eq!(json, r#""breeze_tts_2""#);

        let parsed1: TtsEngine = serde_json::from_str(r#""breeze_tts_2""#).unwrap();
        assert_eq!(parsed1, TtsEngine::BreezeTts2);

        let parsed2: TtsEngine = serde_json::from_str(r#""breeze_tts2""#).unwrap();
        assert_eq!(parsed2, TtsEngine::BreezeTts2);
    }

    #[test]
    fn test_vox_cpm_2_serde() {
        let engine = TtsEngine::VoxCpm2;
        let json = serde_json::to_string(&engine).unwrap();
        assert_eq!(json, r#""vox_cpm_2""#);

        let parsed1: TtsEngine = serde_json::from_str(r#""vox_cpm_2""#).unwrap();
        assert_eq!(parsed1, TtsEngine::VoxCpm2);

        let parsed2: TtsEngine = serde_json::from_str(r#""voxcpm2""#).unwrap();
        assert_eq!(parsed2, TtsEngine::VoxCpm2);

        let parsed3: TtsEngine = serde_json::from_str(r#""vox_cpm2""#).unwrap();
        assert_eq!(parsed3, TtsEngine::VoxCpm2);
    }

    #[test]
    fn test_lux_tts_serde() {
        let engine = TtsEngine::LuxTts;
        let json = serde_json::to_string(&engine).unwrap();
        assert_eq!(json, r#""lux_tts""#);

        let parsed1: TtsEngine = serde_json::from_str(r#""lux_tts""#).unwrap();
        assert_eq!(parsed1, TtsEngine::LuxTts);

        let parsed2: TtsEngine = serde_json::from_str(r#""luxtts""#).unwrap();
        assert_eq!(parsed2, TtsEngine::LuxTts);
    }
}
