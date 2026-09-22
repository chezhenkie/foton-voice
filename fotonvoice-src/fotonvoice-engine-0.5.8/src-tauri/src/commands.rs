use std::sync::Arc;

use tauri::{Emitter, Manager, State};
use tracing::info;
use fotonvoice_config::AppConfig;
use fotonvoice_routing::{HotkeyBinding, OutputTarget};

use crate::state::AppState;


#[tauri::command]
pub async fn get_status(state: State<'_, Arc<AppState>>) -> Result<StatusPayload, String> {
    let active_target_id = state.active_target.lock().await.clone();
    let target_label = {
        let targets_guard = state.targets.lock().await;
        targets_guard.iter()
            .find(|t| t.id == active_target_id)
            .map(|t| t.label.clone())
            .unwrap_or_else(|| {
                if active_target_id == "default" {
                    "Focused Window".to_string()
                } else {
                    active_target_id.clone()
                }
            })
    };

    Ok(StatusPayload {
        recording: state.is_recording(),
        processing: state.is_processing(),
        speaking: state.is_speaking(),
        mcp_recording: state.is_mcp_recording(),
        audio_ready: state.is_audio_ready(),
        word_count: state.total_words(),
        active_target_id,
        active_target_label: target_label,
        hotkeys_active: state.hotkey_health.is_active(),
    })
}

#[derive(serde::Serialize)]
pub struct StatusPayload {
    pub recording: bool,
    pub processing: bool,
    pub speaking: bool,
    pub mcp_recording: bool,
    pub audio_ready: bool,
    pub word_count: u32,
    pub active_target_id: String,
    pub active_target_label: String,
    /// True when the global-shortcut listener can actually deliver a key right
    pub hotkeys_active: bool,
}


#[tauri::command]
pub async fn start_recording(state: State<'_, Arc<AppState>>) -> Result<(), String> {
    *state.active_binding_id.lock().await = String::new();
    state.begin_recording().await;
    info!("Recording started via command");
    let active_target = state.active_target.lock().await.clone();
    state.preload_tts_for_target(&active_target).await;
    Ok(())
}

#[tauri::command]
pub async fn stop_recording(state: State<'_, Arc<AppState>>) -> Result<(), String> {
    state.set_recording(false);
    info!("Recording stopped via command");
    Ok(())
}

#[tauri::command]
pub async fn toggle_recording(state: State<'_, Arc<AppState>>) -> Result<bool, String> {
    let was = state.is_recording();
    if !was {
        *state.active_binding_id.lock().await = String::new();
        let active_target = state.active_target.lock().await.clone();
        state.preload_tts_for_target(&active_target).await;
    }
    state.set_recording(!was);
    Ok(!was)
}


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

#[tauri::command]
pub async fn stop_tts(
    state: State<'_, Arc<AppState>>,
) -> Result<(), String> {
    info!("TTS stop requested via command");
    let handle = state.tts_handle.lock().await;
    if let Some(ref tts) = *handle {
        tts.stop();
    }
    Ok(())
}


/// Returns true when this binary was compiled with the `cuda` cargo feature.
#[tauri::command]
pub fn cuda_enabled() -> bool {
    cfg!(feature = "cuda")
}

/// What this build can actually offload to a GPU, per engine.
#[derive(Debug, Clone, serde::Serialize)]
pub struct AcceleratorSupport {
    /// `"cuda"`, `"vulkan"`, or `None` on a CPU-only build.
    pub whisper_gpu: Option<String>,
    /// `"cuda"`, `"coreml"`, or `None` - `None` in every build shipped today.
    pub moonshine_gpu: Option<String>,
    /// `"cuda"`, `"coreml"`, `"webgpu"`, or `None`.
    pub parakeet_gpu: Option<String>,
    /// True when the GPU backend is compiled in and a device is present at
    pub parakeet_gpu_present: bool,
    /// Same providers as `parakeet_gpu`, for the Nemotron streaming lane.
    pub nemotron_streaming_gpu: Option<String>,
    /// Same contract as `parakeet_gpu_present` for the Nemotron lane.
    pub nemotron_streaming_gpu_present: bool,
    /// `"vulkan"` or `None`.
    pub s1_mini_gpu: Option<String>,
}

#[tauri::command]
pub fn accelerator_support() -> AcceleratorSupport {
    AcceleratorSupport {
        whisper_gpu: fotonvoice_inference::whisper_gpu_backend().map(str::to_string),
        moonshine_gpu: fotonvoice_inference::moonshine_gpu_backend().map(str::to_string),
        parakeet_gpu: fotonvoice_inference::parakeet_gpu_backend().map(str::to_string),
        parakeet_gpu_present: fotonvoice_inference::parakeet_gpu_available(),
        nemotron_streaming_gpu: fotonvoice_inference::nemotron_gpu_backend().map(str::to_string),
        nemotron_streaming_gpu_present: fotonvoice_inference::nemotron_gpu_available(),
        s1_mini_gpu: fotonvoice_inference::s1_mini_gpu_backend().map(str::to_string),
    }
}


#[tauri::command]
pub async fn get_targets(
    _state: State<'_, Arc<AppState>>,
) -> Result<Vec<OutputTarget>, String> {
    let dir = fotonvoice_routing::config_dir();
    fotonvoice_routing::load_targets(&dir).map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn save_targets(
    state: State<'_, Arc<AppState>>,
    targets: Vec<OutputTarget>,
) -> Result<(), String> {
    let dir = fotonvoice_routing::config_dir();
    fotonvoice_routing::save_targets(&targets, &dir).map_err(|e| e.to_string())?;
    
    *state.targets.lock().await = targets.clone();

    state.router.reload(targets).await;
    info!("Targets saved and router reloaded");

    let tts_handle_opt = {
        let guard = state.tts_handle.lock().await;
        guard.clone()
    };
    if let Some(tts) = tts_handle_opt {
        state.spawn_fifo_responders(tts).await;
    }

    Ok(())
}

#[tauri::command]
pub async fn get_bindings(
    _state: State<'_, Arc<AppState>>,
) -> Result<Vec<HotkeyBinding>, String> {
    let dir = fotonvoice_routing::config_dir();
    fotonvoice_routing::load_bindings(&dir).map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn save_bindings(
    state: State<'_, Arc<AppState>>,
    bindings: Vec<HotkeyBinding>,
) -> Result<(), String> {
    let dir = fotonvoice_routing::config_dir();
    fotonvoice_routing::save_bindings(&bindings, &dir).map_err(|e| e.to_string())?;
    info!("Bindings saved");
    
    let all_bindings = crate::stop_key::listener_bindings(&state, bindings.clone()).await;
    let reloader_guard = state.hotkey_reloader.lock().await;
    if let Some(reloader) = &*reloader_guard {
        if let Err(e) = reloader.send(all_bindings) {
            tracing::warn!("Failed to hot-reload bindings: {e}");
        } else {
            info!("Hot-reload signal sent to listener");
        }
    }
    
    let backend = state.hotkey_health.backend();
    if backend == fotonvoice_hotkeys::Backend::MintDbus
        || (backend == fotonvoice_hotkeys::Backend::None
            && crate::mint_shortcuts::is_mint_desktop())
    {
        match crate::mint_shortcuts::sync_mint_shortcuts(&bindings) {
            Ok(registered) => {
                state
                    .hotkey_health
                    .set_backend(fotonvoice_hotkeys::Backend::MintDbus);
                info!(
                    "Mirrored {} binding(s) into Linux Mint's own shortcut settings",
                    registered.len()
                );
            }
            Err(e) => tracing::warn!("Failed to sync Linux Mint native shortcut settings: {e}"),
        }
    }

    Ok(())
}

/// Forget a Chat target's conversation so the next dictation starts fresh.
#[tauri::command]
pub async fn reset_chat_conversation(
    _state: State<'_, Arc<AppState>>,
    target_id: String,
) -> Result<usize, String> {
    let dropped = fotonvoice_routing::reset_chat_history(&target_id).await;
    info!("Chat conversation for '{target_id}' reset ({dropped} messages dropped)");
    Ok(dropped)
}

/// Probe a Chat target's endpoint and list the models it serves.
#[tauri::command]
pub async fn test_chat_target(target: OutputTarget) -> Result<OpenAiTestResult, String> {
    use fotonvoice_config::OpenAiConfig;

    let endpoint = target.chat_url.unwrap_or_default();
    if endpoint.trim().is_empty() {
        return Err("No server URL configured.".into());
    }
    let client = fotonvoice_llm::OpenAiClient::new(OpenAiConfig {
        enabled: true,
        endpoint: endpoint.clone(),
        api_key: target.chat_api_key,
        model: String::new(),
        timeout_secs: target.chat_timeout_secs.clamp(1, 30),
        ..Default::default()
    });

    match client.list_models().await {
        Ok(models) if models.is_empty() => Ok(OpenAiTestResult {
            success: true,
            message: format!("Connected to {endpoint}, but it reported no models."),
            models,
        }),
        Ok(models) => Ok(OpenAiTestResult {
            success: true,
            message: format!("Connected to {endpoint} - {} model(s) available.", models.len()),
            models,
        }),
        Err(e) => Ok(OpenAiTestResult {
            success: false,
            message: format!("Could not reach {endpoint}: {e}"),
            models: Vec::new(),
        }),
    }
}

#[tauri::command]
pub async fn speak_text(
    state: State<'_, Arc<AppState>>,
    text: String,
    voice: Option<String>,
) -> Result<(), String> {
    info!("TTS speak_text via command: {text}");
    let handle = state.tts_handle.lock().await;
    if let Some(ref tts) = *handle {
        tts.speak_utterance(fotonvoice_tts::Utterance {
            text,
            voice,
            source_label: None,
        });
    }
    Ok(())
}

#[tauri::command]
pub async fn check_voice_downloaded(voice_name: String, voice_dir: String) -> Result<bool, String> {
    Ok(fotonvoice_tts::is_voice_downloaded(&voice_name, &voice_dir))
}

/// Voices present in the Piper voices folder (`*.onnx` + `.onnx.json` pairs).
#[tauri::command]
pub async fn list_piper_voices(voice_dir: String) -> Result<Vec<String>, String> {
    Ok(fotonvoice_tts::list_local_voices(&voice_dir))
}

#[tauri::command]
pub async fn download_voice(voice_name: String, voice_dir: String) -> Result<(), String> {
    fotonvoice_tts::download_voice(&voice_name, &voice_dir)
        .await
        .map_err(|e| e.to_string())
}

/// Render a file target's timestamp format so the Settings UI can preview it
#[tauri::command]
pub async fn preview_timestamp_format(format: String) -> Result<String, String> {
    fotonvoice_routing::render_timestamp(&format, chrono::Utc::now())
}

/// The HuggingFace token exported into the environment, if any.
#[tauri::command]
pub async fn hf_token_env() -> Option<String> {
    fotonvoice_tts::hf_token_from_env()
}

#[tauri::command]
pub async fn check_breeze_tts_2_ready(model_dir: String) -> Result<bool, String> {
    Ok(fotonvoice_tts::is_breeze_tts_2_ready(&model_dir))
}

#[tauri::command]
pub async fn download_breeze_tts_2(model_dir: String, hf_token: Option<String>) -> Result<(), String> {
    fotonvoice_tts::download_breeze_tts_2_assets(&model_dir, hf_token)
        .await
        .map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn check_vox_cpm_2_ready(model_dir: String) -> Result<bool, String> {
    Ok(fotonvoice_tts::is_vox_cpm_2_ready(&model_dir))
}

#[tauri::command]
pub async fn check_vox_cpm_2_downloaded(model_dir: String) -> Result<bool, String> {
    Ok(fotonvoice_tts::is_vox_cpm_2_ready(&model_dir))
}

#[tauri::command]
pub async fn download_vox_cpm_2(model_dir: String, hf_token: Option<String>) -> Result<(), String> {
    fotonvoice_tts::download_vox_cpm_2_assets(&model_dir, hf_token)
        .await
        .map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn check_pocket_tts_ready(voice: String, voice_dir: String) -> Result<bool, String> {
    Ok(fotonvoice_tts::is_pocket_tts_ready(&voice, &voice_dir))
}

#[tauri::command]
pub async fn download_pocket_tts(voice: String, voice_dir: String, hf_token: Option<String>) -> Result<(), String> {
    fotonvoice_tts::download_pocket_tts_assets(&voice, &voice_dir, hf_token)
        .await
        .map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn list_pocket_tts_voices(voice_dir: String) -> Result<Vec<fotonvoice_tts::PocketTtsVoiceOption>, String> {
    Ok(fotonvoice_tts::pocket_tts_voice_catalogue(&voice_dir))
}

/// Whether the Inflect-Micro-v2 ONNX engine was compiled into this build. The UI
#[tauri::command]
pub fn inflect_micro_available() -> bool {
    fotonvoice_tts::INFLECT_MICRO_COMPILED
}

#[tauri::command]
pub async fn check_inflect_micro_downloaded(model_dir: String) -> Result<bool, String> {
    Ok(fotonvoice_tts::is_inflect_micro_downloaded(&model_dir))
}

#[tauri::command]
pub async fn download_inflect_micro(model_dir: String) -> Result<(), String> {
    fotonvoice_tts::download_inflect_micro_assets(&model_dir)
        .await
        .map_err(|e| e.to_string())
}

/// Report the tensor names a downloaded Inflect-Micro-v2 export actually
#[tauri::command]
pub async fn inflect_micro_inspect(model_dir: String) -> Result<serde_json::Value, String> {
    #[cfg(feature = "inflect-micro")]
    {
        let dir = fotonvoice_tts::inflect::resolve_model_dir(&model_dir);
        let signature = tokio::task::spawn_blocking(move || fotonvoice_tts::inflect::model::inspect(&dir))
            .await
            .map_err(|e| format!("inspect task join: {e}"))?
            .map_err(|e| format!("{e:#}"))?;
        serde_json::to_value(signature).map_err(|e| e.to_string())
    }
    #[cfg(not(feature = "inflect-micro"))]
    {
        let _ = model_dir;
        Err("This build was compiled without the `inflect-micro` feature.".to_string())
    }
}

#[tauri::command]
pub async fn check_model_downloaded(model_size: String, model_dir: Option<String>) -> Result<bool, String> {
    let dir = model_dir.unwrap_or_default();
    Ok(fotonvoice_inference::whisper_cpp::is_model_downloaded(&model_size, &dir))
}

/// Whether the LuxTTS ONNX engine was compiled into this build. The UI uses
#[tauri::command]
pub fn lux_tts_available() -> bool {
    fotonvoice_tts::LUX_TTS_COMPILED
}

/// Whether every LuxTTS graph plus tokens.txt is on disk in the model dir.
#[tauri::command]
pub async fn check_lux_tts_downloaded(model_dir: String) -> Result<bool, String> {
    Ok(fotonvoice_tts::is_lux_tts_downloaded(&model_dir))
}

#[tauri::command]
pub async fn download_model(model_size: String, model_dir: String) -> Result<(), String> {
    fotonvoice_inference::whisper_cpp::download_model(&model_size, &model_dir)
        .await
        .map_err(|e| e.to_string())
}

/// Delete the whisper.cpp GGUF file(s) for `model_size`.
#[tauri::command]
pub async fn delete_model(model_size: String, model_dir: String) -> Result<(), String> {
    fotonvoice_inference::whisper_cpp::delete_model(&model_size, &model_dir)
        .map_err(|e| e.to_string())
}

/// Whether the Moonshine ONNX backend was compiled into this build. The UI uses
#[tauri::command]
pub fn moonshine_available() -> bool {
    fotonvoice_inference::MOONSHINE_COMPILED
}

#[tauri::command]
pub async fn check_moonshine_downloaded(model_size: String) -> Result<bool, String> {
    #[cfg(feature = "moonshine")]
    {
        Ok(fotonvoice_inference::moonshine::is_model_downloaded(&model_size, ""))
    }
    #[cfg(not(feature = "moonshine"))]
    {
        let _ = model_size;
        Ok(false)
    }
}

#[tauri::command]
pub async fn download_moonshine_model(model_size: String) -> Result<(), String> {
    #[cfg(feature = "moonshine")]
    {
        fotonvoice_inference::moonshine::download_model(&model_size, "")
            .await
            .map_err(|e| e.to_string())
    }
    #[cfg(not(feature = "moonshine"))]
    {
        let _ = model_size;
        Err("This build was compiled without the Moonshine backend. Rebuild with `--features moonshine` to use it.".into())
    }
}

/// Delete the Moonshine ONNX folder for `model_size`.
#[tauri::command]
pub async fn delete_moonshine_model(model_size: String) -> Result<(), String> {
    #[cfg(feature = "moonshine")]
    {
        fotonvoice_inference::moonshine::delete_model(&model_size, "").map_err(|e| e.to_string())
    }
    #[cfg(not(feature = "moonshine"))]
    {
        let _ = model_size;
        Err("This build was compiled without the Moonshine backend. Rebuild with `--features moonshine` to use it.".into())
    }
}

/// Whether the Parakeet ONNX backend was compiled into this build. The UI uses
#[tauri::command]
pub fn parakeet_available() -> bool {
    fotonvoice_inference::PARAKEET_COMPILED
}

#[tauri::command]
pub async fn check_parakeet_downloaded(model_size: String) -> Result<bool, String> {
    #[cfg(feature = "parakeet")]
    {
        Ok(fotonvoice_inference::parakeet::is_model_downloaded(&model_size, ""))
    }
    #[cfg(not(feature = "parakeet"))]
    {
        let _ = model_size;
        Ok(false)
    }
}

#[tauri::command]
pub async fn download_parakeet_model(model_size: String) -> Result<(), String> {
    #[cfg(feature = "parakeet")]
    {
        fotonvoice_inference::parakeet::download_model(&model_size, "")
            .await
            .map_err(|e| e.to_string())
    }
    #[cfg(not(feature = "parakeet"))]
    {
        let _ = model_size;
        Err("This build was compiled without the Parakeet backend. Rebuild with `--features parakeet` to use it.".into())
    }
}

/// Delete the Parakeet ONNX folder for `model_size`.
#[tauri::command]
pub async fn delete_parakeet_model(model_size: String) -> Result<(), String> {
    #[cfg(feature = "parakeet")]
    {
        fotonvoice_inference::parakeet::delete_model(&model_size, "").map_err(|e| e.to_string())
    }
    #[cfg(not(feature = "parakeet"))]
    {
        let _ = model_size;
        Err("This build was compiled without the Parakeet backend. Rebuild with `--features parakeet` to use it.".into())
    }
}

/// Whether the Nemotron streaming ONNX backend was compiled into this build.
#[tauri::command]
pub fn nemotron_streaming_available() -> bool {
    fotonvoice_inference::NEMOTRON_STREAMING_COMPILED
}

#[tauri::command]
pub async fn check_nemotron_streaming_downloaded(model_size: String) -> Result<bool, String> {
    #[cfg(feature = "nemotron-streaming")]
    {
        Ok(fotonvoice_inference::nemotron_streaming::is_model_downloaded(
            &model_size,
            "",
        ))
    }
    #[cfg(not(feature = "nemotron-streaming"))]
    {
        let _ = model_size;
        Ok(false)
    }
}

#[tauri::command]
pub async fn download_nemotron_streaming_model(model_size: String) -> Result<(), String> {
    #[cfg(feature = "nemotron-streaming")]
    {
        fotonvoice_inference::nemotron_streaming::download_model(&model_size, "")
            .await
            .map_err(|e| e.to_string())
    }
    #[cfg(not(feature = "nemotron-streaming"))]
    {
        let _ = model_size;
        Err("This build was compiled without the Nemotron streaming backend. Rebuild with `--features nemotron-streaming` to use it.".into())
    }
}

/// Delete the Nemotron streaming ONNX folder for `model_size`.
#[tauri::command]
pub async fn delete_nemotron_streaming_model(model_size: String) -> Result<(), String> {
    #[cfg(feature = "nemotron-streaming")]
    {
        fotonvoice_inference::nemotron_streaming::delete_model(&model_size, "")
            .map_err(|e| e.to_string())
    }
    #[cfg(not(feature = "nemotron-streaming"))]
    {
        let _ = model_size;
        Err("This build was compiled without the Nemotron streaming backend. Rebuild with `--features nemotron-streaming` to use it.".into())
    }
}

#[tauri::command]
pub async fn check_s1_mini_downloaded(model_dir: Option<String>) -> Result<bool, String> {
    Ok(fotonvoice_inference::s1_mini::is_s1_mini_downloaded(model_dir.as_deref()))
}

#[tauri::command]
pub async fn download_s1_mini_model(model_dir: Option<String>) -> Result<(), String> {
    fotonvoice_inference::s1_mini::download_s1_mini_assets(model_dir.as_deref())
        .await
        .map_err(|e| e.to_string())
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


/// The resolved, absolute path to the shared voice-cloning reference-clip
#[tauri::command]
pub fn get_cloned_tts_voices_dir() -> String {
    fotonvoice_tts::cloned_tts_voices_dir().display().to_string()
}


#[tauri::command]
pub async fn start_monitoring_audio(state: State<'_, Arc<AppState>>) -> Result<(), String> {
    state.set_monitoring(true);
    info!("Audio monitoring started");
    Ok(())
}

#[tauri::command]
pub async fn stop_monitoring_audio(state: State<'_, Arc<AppState>>) -> Result<(), String> {
    state.set_monitoring(false);
    info!("Audio monitoring stopped");
    Ok(())
}

/// Tell the gesture handler to silently drop any incoming hotkey events.
#[tauri::command]
pub async fn set_hotkeys_inhibited(
    inhibited: bool,
    state: State<'_, Arc<AppState>>,
) -> Result<(), String> {
    state.set_hotkeys_inhibited(inhibited);
    info!("Hotkeys inhibited: {inhibited}");
    Ok(())
}

#[tauri::command]
pub async fn list_audio_devices() -> Result<Vec<AudioDeviceInfo>, String> {
    let devices = fotonvoice_audio::list_input_devices();
    Ok(devices
        .into_iter()
        .map(|d| AudioDeviceInfo { index: d.index, name: d.name })
        .collect())
}

#[derive(serde::Serialize)]
pub struct AudioDeviceInfo {
    pub index: u32,
    pub name: String,
}

#[derive(serde::Serialize)]
pub struct OpenAiTestResult {
    pub success: bool,
    pub message: String,
    pub models: Vec<String>,
}

#[tauri::command]
pub async fn test_openai(
    endpoint: String,
    api_key: Option<String>,
    timeout_secs: u64,
) -> Result<OpenAiTestResult, String> {
    use fotonvoice_config::OpenAiConfig;
    let cfg = OpenAiConfig {
        enabled: true,
        endpoint: endpoint.clone(),
        api_key,
        model: String::new(),
        timeout_secs,
        ..Default::default()
    };
    let client = fotonvoice_llm::OpenAiClient::new(cfg);
    if client.is_available().await {
        match client.list_models().await {
            Ok(models) => {
                Ok(OpenAiTestResult {
                    success: true,
                    message: "Successfully connected to the OpenAI API server!".to_string(),
                    models,
                })
            }
            Err(e) => {
                Ok(OpenAiTestResult {
                    success: true,
                    message: format!("Successfully connected, but failed to fetch model list: {}", e),
                    models: Vec::new(),
                })
            }
        }
    } else {
        Ok(OpenAiTestResult {
            success: false,
            message: format!("Failed to connect to the OpenAI API server at '{}'. Check the URL and API key.", endpoint),
            models: Vec::new(),
        })
    }
}

#[tauri::command]
pub async fn test_remote_stt(
    endpoint: String,
    api_key: Option<String>,
    model: String,
    timeout_secs: u64,
) -> Result<fotonvoice_inference::RemoteSttTestResult, String> {
    fotonvoice_inference::test_remote_speech_engine(
        &endpoint,
        api_key.as_deref(),
        &model,
        timeout_secs,
    )
    .await
    .map_err(|e| e.to_string())
}

#[derive(serde::Serialize, Clone)]
pub struct HotkeyStatusPayload {
    /// Global shortcuts can fire right now.
    pub is_active: bool,
    /// Which mechanism is delivering them: `portal`, `evdev`, `windows_hook`,
    pub backend: String,
    /// FotonVoice Engine receives only its own shortcuts and can read no keystrokes.
    pub is_private: bool,
    /// Why the desktop portal is not in use, when it is not.
    pub portal_error: Option<String>,
    /// The portal exists and refused FotonVoice Engine, rather than being absent. Needs
    pub portal_refused: bool,
    /// What the compositor actually bound, which may differ from what FotonVoice Engine
    pub shortcuts: Vec<fotonvoice_hotkeys::BoundShortcut>,
    /// Gesture styles the running backend can actually deliver, as the same
    pub supported_gestures: Vec<fotonvoice_routing::GestureType>,
    /// Why the X11 backend was not used, when it was not. Distinct from
    pub x11_error: Option<String>,
    /// `wayland`, `x11` or whatever `XDG_SESSION_TYPE` says.
    pub session_type: String,
    /// `/dev/input/event*` nodes present, and how many FotonVoice Engine could open.
    pub devices_total: u32,
    pub devices_readable: u32,
    /// The user has to do something outside FotonVoice Engine for shortcuts to work.
    pub needs_attention: bool,
    /// One-line, human-readable explanation of the state above.
    pub detail: String,
    /// KDE registers portal shortcuts into System Settings in a *disabled*
    pub needs_manual_enable: bool,
    /// What to tell the user about `needs_manual_enable`, and how to fix it.
    pub manual_enable_hint: Option<String>,
    /// Running on Linux Mint's Cinnamon or MATE desktop environment.
    pub is_mint_desktop: bool,
    /// FotonVoice Engine's native D-Bus shortcut is registered in Mint's gsettings registry.
    pub mint_shortcut_registered: bool,
    /// Windows only: the focused window belongs to a process elevated above
    pub elevated_window_focused: bool,
}

/// `XDG_CURRENT_DESKTOP` is a colon-separated list (e.g. `ubuntu:GNOME`); any
#[cfg(target_os = "linux")]
fn desktop_environment() -> Option<String> {
    let current = std::env::var("XDG_CURRENT_DESKTOP").unwrap_or_default();
    if current.split(':').any(|d| d.eq_ignore_ascii_case("KDE")) {
        return Some("KDE".to_string());
    }
    if std::env::var_os("KDE_FULL_SESSION").is_some() {
        return Some("KDE".to_string());
    }
    if current.split(':').any(|d| d.eq_ignore_ascii_case("GNOME")) {
        return Some("GNOME".to_string());
    }
    current
        .split(':')
        .find(|d| !d.is_empty())
        .map(|d| d.to_string())
}

#[cfg(not(target_os = "linux"))]
fn desktop_environment() -> Option<String> {
    None
}

#[cfg(target_os = "linux")]
fn session_type() -> String {
    if std::env::var_os("WAYLAND_DISPLAY").is_some() {
        "wayland".to_string()
    } else if std::env::var_os("DISPLAY").is_some() {
        "x11".to_string()
    } else {
        std::env::var("XDG_SESSION_TYPE").unwrap_or_else(|_| "unknown".to_string())
    }
}

#[cfg(not(target_os = "linux"))]
fn session_type() -> String {
    std::env::consts::OS.to_string()
}

/// Returns `(total, readable)` counts of `/dev/input/event*` nodes.
#[cfg(target_os = "linux")]
fn count_input_devices() -> (u32, u32) {
    let Ok(entries) = std::fs::read_dir("/dev/input") else {
        return (0, 0);
    };
    let mut total = 0;
    let mut readable = 0;
    for entry in entries.flatten() {
        let is_event = entry
            .file_name()
            .to_str()
            .map(|n| n.starts_with("event"))
            .unwrap_or(false);
        if !is_event {
            continue;
        }
        total += 1;
        if std::fs::File::open(entry.path()).is_ok() {
            readable += 1;
        }
    }
    (total, readable)
}

#[cfg(not(target_os = "linux"))]
fn count_input_devices() -> (u32, u32) {
    (0, 0)
}

/// How FotonVoice Engine is receiving global shortcuts, and what - if anything - is
pub fn hotkey_status(health: &fotonvoice_hotkeys::ListenerHealth) -> HotkeyStatusPayload {
    if let Ok(override_val) = std::env::var("FOTONVOICE_TEST_HOTKEY_STATUS") {
        if let Some(payload) = test_override(&override_val) {
            return payload;
        }
    }

    let backend = health.backend();
    let desktop = desktop_environment();
    let is_mint_desktop = crate::mint_shortcuts::is_mint_desktop();
    let mint_shortcut_registered = if is_mint_desktop {
        crate::mint_shortcuts::is_mint_shortcut_registered()
    } else {
        false
    };

    let effective_backend = if backend == fotonvoice_hotkeys::Backend::None && mint_shortcut_registered
    {
        fotonvoice_hotkeys::Backend::MintDbus
    } else {
        backend
    };

    let needs_manual_enable = backend == fotonvoice_hotkeys::Backend::Portal
        && desktop.as_deref() == Some("KDE");
    let (devices_total, devices_readable) = match effective_backend {
        fotonvoice_hotkeys::Backend::Portal
        | fotonvoice_hotkeys::Backend::WindowsHook
        | fotonvoice_hotkeys::Backend::X11
        | fotonvoice_hotkeys::Backend::MintDbus => (0, 0),
        _ => count_input_devices(),
    };
    let shortcuts = health.bound_shortcuts();

    let detail = match effective_backend {
        fotonvoice_hotkeys::Backend::Portal => {
            let unbound = shortcuts.iter().filter(|s| !s.bound).count();
            if unbound > 0 {
                format!(
                    "Your desktop registered FotonVoice Engine's shortcuts, but {unbound} of {} could \
                     not be bound. Pick different keys for those, or set them in your \
                     desktop's own shortcut settings.",
                    shortcuts.len()
                )
            } else {
                "Your desktop is handling FotonVoice Engine's global shortcuts. FotonVoice Engine cannot read \
                 your keyboard - it is only told when its own shortcut fires."
                    .to_string()
            }
        }
        fotonvoice_hotkeys::Backend::WindowsHook if health.elevated_window_focused() => (
            "FotonVoice Engine cannot see key presses right now: the focused window is running as \
             administrator (Task Manager, an elevated terminal, an installer), and Windows \
             blocks FotonVoice Engine's keyboard hook from receiving keys while such a window has \
             focus. Switch back to a window that is not running as administrator and your \
             shortcuts will work again."
        )
        .to_string(),
        fotonvoice_hotkeys::Backend::WindowsHook => (
            "FotonVoice Engine is receiving shortcuts through a Windows low-level keyboard hook.              Every gesture style works, including bare modifiers, and no permission setup              was needed - but in this mode every keystroke passes through FotonVoice Engine. Keys are              matched against your shortcuts and discarded; nothing is stored or sent              anywhere. Windows does not deliver keys to this hook while an elevated              application has focus, or on the secure desktop (the UAC prompt and the lock              screen), so shortcuts do not fire there."
        )
        .to_string(),
        fotonvoice_hotkeys::Backend::X11 => (
            "Your desktop has no global-shortcuts portal, so FotonVoice Engine is reading X11 key \
             events directly. Every gesture style works, including bare modifiers, and no \
             permission setup was needed - but in this mode every keystroke passes through \
             FotonVoice Engine."
        )
        .to_string(),
        fotonvoice_hotkeys::Backend::MintDbus => (
            "Your desktop is handling FotonVoice Engine's shortcuts through its own keyboard settings. \
             FotonVoice Engine cannot read your keyboard. This route only carries a press, never a \
             release, so it can serve tap-to-start/tap-to-stop bindings and no other gesture."
        )
        .to_string(),
        fotonvoice_hotkeys::Backend::Evdev => format!(
            "This desktop does not offer the global-shortcuts portal, so FotonVoice Engine is reading \
             input devices directly ({devices_readable} of {devices_total} readable). That \
             works, but it means every keystroke passes through FotonVoice Engine."
        ),
        fotonvoice_hotkeys::Backend::Starting => "Starting the shortcut listener...".to_string(),
        fotonvoice_hotkeys::Backend::None if health.portal_refused() => {
            "Global shortcuts require approval from your desktop. The system prompt was \
             closed or declined before shortcuts were registered - click Approve Shortcuts \
             below to display the prompt and confirm your keybinds."
                .to_string()
        }
        fotonvoice_hotkeys::Backend::None => {
            if is_mint_desktop {
                "This desktop (Linux Mint) does not provide the XDG global-shortcuts portal, but supports registering native custom shortcuts via System Settings / D-Bus."
                    .to_string()
            } else if devices_total == 0 {
                "No global-shortcuts portal and no input devices were found, so FotonVoice Engine \
                 cannot receive shortcuts on this system."
                    .to_string()
            } else {
                format!(
                    "This desktop does not provide the XDG global-shortcuts portal, so \
                     FotonVoice Engine has no way to receive its shortcuts. FotonVoice Engine will not grant \
                     itself keyboard access to work around it: doing so would let every \
                     program you run read everything you type. None of the {devices_total} \
                     input devices on this system is readable."
                )
            }
        }
    };

    let is_active = health.is_active() || (is_mint_desktop && mint_shortcut_registered);

    HotkeyStatusPayload {
        is_active,
        backend: match effective_backend {
            fotonvoice_hotkeys::Backend::Portal => "portal",
            fotonvoice_hotkeys::Backend::X11 => "x11",
            fotonvoice_hotkeys::Backend::MintDbus => "mint_dbus",
            fotonvoice_hotkeys::Backend::Evdev => "evdev",
            fotonvoice_hotkeys::Backend::WindowsHook => "windows_hook",
            fotonvoice_hotkeys::Backend::Starting => "starting",
            fotonvoice_hotkeys::Backend::None => "none",
        }
        .to_string(),
        is_private: health.is_private() || (is_mint_desktop && mint_shortcut_registered),
        portal_error: health.portal_error(),
        portal_refused: health.portal_refused(),
        shortcuts,
        supported_gestures: effective_backend.gestures().to_vec(),
        x11_error: health.x11_error(),
        session_type: session_type(),
        devices_total,
        devices_readable,
        needs_attention: !is_active,
        detail,
        needs_manual_enable,
        manual_enable_hint: needs_manual_enable.then(|| {
            "KDE registers FotonVoice Engine's shortcuts as disabled until you turn them on yourself: \
             open Shortcuts, find FotonVoice Engine, tick the box next to each shortcut, and click \
             Apply. This is a known KDE bug (xdg-desktop-portal-kde #483639), not something \
             FotonVoice Engine's setup missed - the portal gives FotonVoice Engine no way to tell whether a \
             shortcut is enabled, so this step cannot be automated or skipped."
                .to_string()
        }),
        is_mint_desktop,
        mint_shortcut_registered,
        elevated_window_focused: health.elevated_window_focused(),
    }
}

/// Developer/test override so the setup window can be exercised in every state
fn test_override(value: &str) -> Option<HotkeyStatusPayload> {
    let base = HotkeyStatusPayload {
        is_active: true,
        backend: "portal".to_string(),
        is_private: true,
        portal_error: None,
        portal_refused: false,
        shortcuts: Vec::new(),
        supported_gestures: fotonvoice_hotkeys::Backend::Portal.gestures().to_vec(),
        x11_error: None,
        session_type: "wayland".to_string(),
        devices_total: 0,
        devices_readable: 0,
        needs_attention: false,
        detail: "Your desktop is handling FotonVoice Engine's global shortcuts.".to_string(),
        needs_manual_enable: false,
        manual_enable_hint: None,
        is_mint_desktop: false,
        mint_shortcut_registered: false,
        elevated_window_focused: false,
    };
    match value {
        "portal" => Some(base),
        "mint" => Some(HotkeyStatusPayload {
            is_active: false,
            backend: "none".to_string(),
            is_private: false,
            portal_error: Some("no such interface".to_string()),
            is_mint_desktop: true,
            mint_shortcut_registered: false,
            needs_attention: true,
            detail: "This desktop (Linux Mint) does not provide the XDG global-shortcuts portal, but supports registering native custom shortcuts via System Settings / D-Bus.".to_string(),
            ..base
        }),
        "mint_registered" => Some(HotkeyStatusPayload {
            is_active: true,
            backend: "mint_dbus".to_string(),
            is_private: true,
            portal_error: Some("no such interface".to_string()),
            is_mint_desktop: true,
            mint_shortcut_registered: true,
            needs_attention: false,
            detail: "Linux Mint native desktop shortcut (Ctrl+Alt+Space) is registered in System Settings and triggers FotonVoice Engine over D-Bus.".to_string(),
            ..base
        }),
        "kde_manual_enable" => Some(HotkeyStatusPayload {
            needs_manual_enable: true,
            manual_enable_hint: Some(
                "KDE registers FotonVoice Engine's shortcuts as disabled until you turn them on \
                 yourself: open Shortcuts, find FotonVoice Engine, tick the box next to each \
                 shortcut, and click Apply."
                    .to_string(),
            ),
            ..base
        }),
        "evdev" => Some(HotkeyStatusPayload {
            backend: "evdev".to_string(),
            is_private: false,
            portal_error: Some("no such interface".to_string()),
            devices_total: 6,
            devices_readable: 6,
            detail: "FotonVoice Engine is reading input devices directly.".to_string(),
            ..base
        }),
        "none" => Some(HotkeyStatusPayload {
            is_active: false,
            backend: "none".to_string(),
            is_private: false,
            portal_error: Some("no such interface".to_string()),
            devices_total: 6,
            devices_readable: 0,
            needs_attention: true,
            detail: "This desktop does not provide the XDG global-shortcuts portal.".to_string(),
            ..base
        }),
        "refused" => Some(HotkeyStatusPayload {
            is_active: false,
            backend: "none".to_string(),
            is_private: false,
            portal_error: Some(
                "org.freedesktop.portal.Error.NotAllowed: An app id is required".to_string(),
            ),
            portal_refused: true,
            needs_attention: true,
            detail: "Your desktop has a global-shortcuts portal but refused FotonVoice Engine's \
                     request for one."
                .to_string(),
            ..base
        }),
        _ => None,
    }
}

/// Launch the desktop's own global-shortcuts settings panel, best-effort.
#[tauri::command]
pub async fn open_shortcut_settings() -> Result<(), String> {
    #[cfg(not(target_os = "linux"))]
    {
        return Err("Opening system shortcut settings is only supported on Linux.".to_string());
    }

    #[cfg(target_os = "linux")]
    {
        for (bin, args) in shortcut_settings_candidates() {
            if !crate::installer::command_exists(bin) {
                continue;
            }
            match spawn_shortcut_settings(bin, args) {
                Ok(()) => return Ok(()),
                Err(e) => {
                    tracing::warn!("Found `{bin}` but failed to launch it: {e}");
                    continue;
                }
            }
        }

        Err("Could not find a way to open your desktop's shortcut settings automatically. \
             Open System Settings yourself and look for Shortcuts -> FotonVoice Engine."
            .to_string())
    }
}

/// Request or retry global shortcut registration via the XDG desktop portal.
#[tauri::command]
pub async fn retry_portal_shortcuts(state: State<'_, Arc<AppState>>) -> Result<(), String> {
    #[cfg(target_os = "linux")]
    {
        let Some(bindings) = crate::stop_key::listener_bindings_from_disk(&state).await else {
            return Err("Could not read your saved shortcuts, so there is nothing to \
                        register. Check the log for what went wrong reading bindings.toml."
                .to_string());
        };

        let gesture_tx = {
            let gtx_guard = state.hotkey_gesture_tx.lock().await;
            gtx_guard.clone()
        };
        let Some(gesture_tx) = gesture_tx else {
            return Err("Hotkey gesture channel is not available.".to_string());
        };

        let (reloader_tx, reloader_rx) = crossbeam_channel::unbounded();
        {
            let mut reloader = state.hotkey_reloader.lock().await;
            *reloader = Some(reloader_tx);
        }

        fotonvoice_hotkeys::retry_portal(
            bindings,
            gesture_tx,
            reloader_rx,
            state.hotkey_health.clone(),
        )
        .await
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = state;
        Ok(())
    }
}

/// Tried in order: the KDE System Settings module directly (skips the home
#[cfg(target_os = "linux")]
fn shortcut_settings_candidates() -> &'static [(&'static str, &'static [&'static str])] {
    &[
        ("kcmshell6", &["kcm_keys"]),
        ("kcmshell5", &["kcm_keys"]),
        ("systemsettings6", &["kcm_keys"]),
        ("systemsettings", &["kcm_keys"]),
        ("gnome-control-center", &["keyboard"]),
    ]
}

/// Fire-and-forget: these are GUI apps meant to stay open long after this
#[cfg(target_os = "linux")]
fn spawn_shortcut_settings(bin: &str, args: &[&str]) -> Result<(), String> {
    #[cfg(test)]
    {
        if std::env::var_os("FOTONVOICE_INSTALLER_TEST_MOCK").is_some() {
            return Ok(());
        }
    }
    crate::host_env::host_command(bin)
        .args(args)
        .spawn()
        .map(|_| ())
        .map_err(|e| e.to_string())
}

/// Verdict on a key combination the user just recorded.
#[derive(serde::Serialize, Clone)]
pub struct HotkeyKeysCheck {
    /// The combination can be saved as-is.
    pub accepted: bool,
    /// True when a rejection is binding rather than advisory - i.e. shortcuts
    pub enforced: bool,
    /// The shortcut as the desktop will see it, e.g. `LOGO+space`.
    pub accelerator: Option<String>,
    /// Machine-readable problem: `modifiers_only`, `multiple_keys`,
    pub problem: Option<String>,
    /// What to tell the user, and what to press instead.
    pub message: Option<String>,
}

impl HotkeyKeysCheck {
    fn ok(accelerator: Option<String>) -> Self {
        Self {
            accepted: true,
            enforced: false,
            accelerator,
            problem: None,
            message: None,
        }
    }
}

/// A combination the desktop can bind, plus a warning when *holding* it would
fn standing_grab_check(
    keys: &[String],
    accelerator: String,
    health: &fotonvoice_hotkeys::ListenerHealth,
) -> HotkeyKeysCheck {
    let takes_the_key = fotonvoice_hotkeys::is_reserved_for_the_desktop(keys)
        && !health.backend().sees_raw_keys();
    if !takes_the_key {
        return HotkeyKeysCheck::ok(Some(accelerator));
    }
    HotkeyKeysCheck {
        accepted: true,
        enforced: false,
        accelerator: Some(accelerator),
        problem: Some("reserved_key".to_string()),
        message: Some(
            "Your desktop hands a registered shortcut to FotonVoice Engine alone, and a binding is \
             held for as long as FotonVoice Engine runs - so with Escape bound here, an open menu or \
             dialog would stop closing anywhere on your desktop. Add a modifier \
             (Ctrl+Escape) to avoid that. The TTS stop key is safe to leave on Escape: it \
             is held only while FotonVoice Engine is speaking."
                .to_string(),
        ),
    }
}

/// Can this key combination be registered as a global shortcut?
pub fn check_hotkey_keys_with(
    keys: &[String],
    health: &fotonvoice_hotkeys::ListenerHealth,
) -> HotkeyKeysCheck {
    use fotonvoice_hotkeys::TriggerProblem;

    let problem = match fotonvoice_hotkeys::accelerator(keys) {
        Ok(accelerator) => return standing_grab_check(keys, accelerator, health),
        Err(problem) => problem,
    };

    let enforced = !health.backend().sees_raw_keys();

    let hint = match problem {
        TriggerProblem::ModifiersOnly => Some(
            "Add a regular key to the combination - Super+Space and Ctrl+Alt+D both work.",
        ),
        TriggerProblem::MultipleKeys => {
            Some("Keep one regular key and use modifiers for the rest.")
        }
        TriggerProblem::UnsupportedKey(_) => Some("Try a letter, number, function or arrow key."),
        TriggerProblem::Empty => None,
    };

    let mut message = if enforced {
        format!("Your desktop cannot register this shortcut: {problem}.")
    } else {
        format!(
            "This works right now, because FotonVoice Engine is watching the keyboard itself rather \
             than using the desktop's shortcut service. It will stop working if that \
             changes: {problem}."
        )
    };
    if let Some(hint) = hint {
        message.push(' ');
        message.push_str(hint);
    }

    HotkeyKeysCheck {
        accepted: !enforced,
        enforced,
        accelerator: None,
        problem: Some(
            match problem {
                TriggerProblem::Empty => "empty",
                TriggerProblem::ModifiersOnly => "modifiers_only",
                TriggerProblem::MultipleKeys => "multiple_keys",
                TriggerProblem::UnsupportedKey(_) => "unsupported_key",
            }
            .to_string(),
        ),
        message: Some(message),
    }
}

#[tauri::command]
pub async fn check_hotkey_keys(
    keys: Vec<String>,
    state: tauri::State<'_, std::sync::Arc<crate::state::AppState>>,
) -> Result<HotkeyKeysCheck, String> {
    Ok(check_hotkey_keys_with(&keys, &state.hotkey_health))
}

#[tauri::command]
pub async fn check_hotkey_status(
    state: tauri::State<'_, std::sync::Arc<crate::state::AppState>>,
) -> Result<HotkeyStatusPayload, String> {
    Ok(hotkey_status(&state.hotkey_health))
}

#[tauri::command]
pub async fn register_mint_shortcut(
    state: tauri::State<'_, std::sync::Arc<crate::state::AppState>>,
) -> Result<String, String> {
    let result = crate::mint_shortcuts::register_mint_shortcut(None)?;
    state
        .hotkey_health
        .set_backend(fotonvoice_hotkeys::Backend::MintDbus);
    Ok(result)
}

#[tauri::command]
pub async fn approve_shortcuts(
    state: tauri::State<'_, std::sync::Arc<crate::state::AppState>>,
) -> Result<HotkeyStatusPayload, String> {
    if crate::mint_shortcuts::is_mint_desktop()
        && state.hotkey_health.backend() == fotonvoice_hotkeys::Backend::None
    {
        crate::mint_shortcuts::register_mint_shortcut(None)?;
        state
            .hotkey_health
            .set_backend(fotonvoice_hotkeys::Backend::MintDbus);
    } else if state.hotkey_health.backend() == fotonvoice_hotkeys::Backend::Portal {
        let _ = open_shortcut_settings().await;
    } else {
        let _ = retry_portal_shortcuts(state.clone()).await;
    }
    Ok(hotkey_status(&state.hotkey_health))
}

/// Install the host packages FotonVoice Engine needs to type text into other windows.
#[tauri::command]
pub async fn install_system_integration(
    state: tauri::State<'_, std::sync::Arc<crate::state::AppState>>,
) -> Result<HotkeyStatusPayload, String> {
    crate::installer::run_gui_installer().await?;
    Ok(hotkey_status(&state.hotkey_health))
}

/// The keystroke-injection helper this session needs, if it is not installed.
pub fn missing_injection_tool() -> Option<&'static str> {
    #[cfg(not(target_os = "linux"))]
    {
        None
    }

    #[cfg(target_os = "linux")]
    {
        let have = |name: &str| fotonvoice_config::find_in_path(name).is_some();
        if have("wtype") || have("xdotool") {
            return None;
        }
        if std::env::var_os("WAYLAND_DISPLAY").is_some() {
            Some("wtype")
        } else {
            Some("xdotool")
        }
    }
}

/// Everything first-run setup depends on, in one call: how global shortcuts are
#[derive(serde::Serialize)]
pub struct SetupStatusPayload {
    pub hotkeys: HotkeyStatusPayload,
    /// Shortcuts can fire right now. Mirrors `hotkeys.is_active` so the UI can
    pub hotkeys_active: bool,
    pub model_ready: bool,
    pub model_size: String,
    /// The configured model downloads itself in the background at first launch.
    pub model_auto_downloads: bool,
    /// Name of the missing keystroke-injection helper, if any.
    pub missing_injection_tool: Option<String>,
    /// Graphical privilege escalation is available for the one-click install.
    pub pkexec_available: bool,
    /// Commands that install the host packages by hand, for machines with no
    pub manual_package_commands: String,
    pub is_complete: bool,
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

/// Open the settings window on a specific tab. Used by the setup window so
#[tauri::command]
pub async fn open_settings_tab(app: tauri::AppHandle, tab: String) -> Result<(), String> {
    let existed = app.get_webview_window("settings").is_some();
    let window = crate::window::open_settings_window(&app)?;
    let _ = window.emit("focus-settings-tab", tab.clone());

    if !existed {
        tauri::async_runtime::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_millis(700)).await;
            let _ = window.emit("focus-settings-tab", tab);
        });
    }
    Ok(())
}

#[derive(serde::Serialize)]
pub struct MonitorInfo {
    pub name: Option<String>,
    pub width: u32,
    pub height: u32,
    pub is_primary: bool,
}

#[tauri::command]
pub async fn get_available_monitors(app: tauri::AppHandle) -> Result<Vec<MonitorInfo>, String> {
    let (tx, rx) = tokio::sync::oneshot::channel();
    let app_handle = app.clone();
    let res = app.run_on_main_thread(move || {
        let mut list = Vec::new();
        if let Some(w) = app_handle.webview_windows().values().next() {
            if let Ok(monitors) = w.available_monitors() {
                let primary = w.primary_monitor().ok().flatten();
                let primary_name = primary.as_ref().and_then(|m| m.name());

                for m in monitors {
                    let name = m.name().map(|s| s.to_string());
                    let is_primary = primary_name.is_some() && name.as_deref() == primary_name.map(|s| s.as_ref());
                    let size = m.size();
                    list.push(MonitorInfo {
                        name,
                        width: size.width,
                        height: size.height,
                        is_primary,
                    });
                }
            }
        }
        let _ = tx.send(list);
    });

    if let Err(e) = res {
        return Err(format!("Failed to run monitor query on main thread: {}", e));
    }

    rx.await.map_err(|e| format!("Failed to receive monitors: {}", e))
}

/// The overlay's frontend reporting that it has painted a frame.
#[tauri::command]
pub fn overlay_content_ready(app: tauri::AppHandle) {
    crate::window::reveal_overlay(&app);
}


