use std::sync::Arc;
use tauri::{Emitter, State};
use tracing::info;
use fotonvoice_config::AppConfig;
use crate::state::AppState;
use super::*;
#[tauri::command]
pub async fn get_config(state: State<'_, Arc<AppState>>) -> Result<AppConfig, String> {
    let guard = state.config.lock().await;
    Ok(guard.data.clone())
}

#[tauri::command]
pub async fn save_config(
    state: State<'_, Arc<AppState>>,
    app: tauri::AppHandle,
    new_config: AppConfig,
) -> Result<(), String> {
    state.set_dynamic_stream(new_config.audio.dynamic_stream);
    state.set_input_device_index(new_config.audio.input_device_index);
    state.set_gain(new_config.audio.gain);
    state.set_noise_suppression(new_config.audio.noise_suppression);
    state.set_overlay_enabled(new_config.ui.show_overlay);

    {
        let mut handle = state.tts_handle.lock().await;
        let mut need_restart = true;

        if let Some(ref tts) = *handle {
            tts.update_config(new_config.tts.clone());
            need_restart = false;
        }

        if need_restart {
            if new_config.tts.enabled {
                let app_handle = app.clone();
                let app_handle_end = app.clone();
                let app_handle_err = app.clone();
                let state_clone = state.inner().clone();
                let state_clone_end = state.inner().clone();
                let new_tts = fotonvoice_tts::TtsEngineWorker::start(
                    new_config.tts.clone(),
                    new_config.features.custom_vocabulary.clone(),
                    Some(std::sync::Arc::new(move || {
                        state_clone.set_speaking(true);
                        let _ = app_handle.emit("tts-playback-start", ());
                    })),
                    Some(std::sync::Arc::new(move || {
                        state_clone_end.set_speaking(false);
                        let _ = app_handle_end.emit("tts-playback-end", ());
                    })),
                    Some(std::sync::Arc::new(move |msg: String| {
                        let _ = app_handle_err.emit("tts-error", msg);
                    })),
                );
                *handle = Some(new_tts.clone());
                state.spawn_fifo_responders(new_tts).await;
            } else {
                *handle = None;
            }
        }
    }

    let mut guard = state.config.lock().await;
    let stop_key_changed = guard.data.tts.stop_key != new_config.tts.stop_key;
    guard.data = new_config.clone();
    guard.save().map_err(|e| e.to_string())?;
    info!("Config saved");

    let _ = state.inference_config_tx.send(Arc::new(new_config.clone()));

    let (overlay_position, overlay_monitor) = (
        guard.data.ui.overlay_position.clone(),
        guard.data.ui.overlay_monitor.clone(),
    );

    drop(guard);
    if stop_key_changed {
        if let Some(bindings) = crate::stop_key::listener_bindings_from_disk(&state).await {
            let reloader_guard = state.hotkey_reloader.lock().await;
            if let Some(reloader) = &*reloader_guard {
                let _ = reloader.send(bindings);
            }
        }
    }

    crate::tray::update_tray_tts_memory(&app, new_config.tts.unloads_when_idle());

    let _ = app.emit("config-changed", new_config);

    let pos_msg = serde_json::json!({
        "type": "position",
        "position": overlay_position,
        "monitor": overlay_monitor,
    });
    if let Ok(json_str) = serde_json::to_string(&pos_msg) {
        let _ = state.overlay_tx.send(json_str);
    }

    Ok(())
}

/// The HuggingFace token exported into the environment, if any.
#[tauri::command]
pub async fn hf_token_env() -> Option<String> {
    fotonvoice_tts::hf_token_from_env()
}

#[tauri::command]
pub async fn check_directory_exists(path: String) -> Result<bool, String> {
    if path.is_empty() {
        return Ok(true);
    }
    Ok(expand_tilde(&path).is_dir())
}

fn expand_tilde(path: &str) -> std::path::PathBuf {
    if path == "~" {
        return dirs::home_dir().unwrap_or_else(|| std::path::PathBuf::from("~"));
    }
    if let Some(rest) = path.strip_prefix("~/") {
        if let Some(home) = dirs::home_dir() {
            return home.join(rest);
        }
    }
    std::path::PathBuf::from(path)
}

#[tauri::command]
pub async fn get_setup_status(
    state: tauri::State<'_, std::sync::Arc<crate::state::AppState>>,
) -> Result<SetupStatusPayload, String> {
    let hotkeys = hotkey_status(&state.hotkey_health);
    let hotkeys_active = hotkeys.is_active;

    let (model_size, model_dir, uses_whisper_model) = {
        let cfg = state.config.lock().await;
        let eng = &cfg.data.engine;
        (
            eng.whisper_cpp.model_size.clone(),
            eng.whisper_cpp.model_dir.clone(),
            eng.backend != fotonvoice_config::BackendChoice::RemoteOpenAi
                && (eng.backend != fotonvoice_config::BackendChoice::Moonshine
                    || !fotonvoice_inference::MOONSHINE_COMPILED)
                && (eng.backend != fotonvoice_config::BackendChoice::Parakeet
                    || !fotonvoice_inference::PARAKEET_COMPILED)
                && (eng.backend != fotonvoice_config::BackendChoice::NemotronStreaming
                    || !fotonvoice_inference::NEMOTRON_STREAMING_COMPILED),
        )
    };

    let model_ready = !uses_whisper_model
        || fotonvoice_inference::whisper_cpp::is_model_downloaded(&model_size, &model_dir);
    let model_auto_downloads =
        uses_whisper_model && fotonvoice_inference::whisper_cpp::is_small_auto_downloadable(&model_size);

    let missing_tool = missing_injection_tool();

    Ok(SetupStatusPayload {
        is_complete: hotkeys_active && model_ready && missing_tool.is_none(),
        hotkeys,
        hotkeys_active,
        model_ready,
        model_size,
        model_auto_downloads,
        missing_injection_tool: missing_tool.map(str::to_string),
        pkexec_available: crate::installer::command_exists("pkexec"),
        manual_package_commands: crate::installer::manual_setup_commands(
            crate::installer::detect_pkg_manager(),
        ),
    })
}

/// Download whichever speech model the config currently selects.
#[tauri::command]
pub async fn download_configured_model(
    state: tauri::State<'_, std::sync::Arc<crate::state::AppState>>,
) -> Result<(), String> {
    let (model_size, model_dir) = {
        let cfg = state.config.lock().await;
        (
            cfg.data.engine.whisper_cpp.model_size.clone(),
            cfg.data.engine.whisper_cpp.model_dir.clone(),
        )
    };
    fotonvoice_inference::whisper_cpp::download_model(&model_size, &model_dir)
        .await
        .map_err(|e| format!("{e:#}"))
}
