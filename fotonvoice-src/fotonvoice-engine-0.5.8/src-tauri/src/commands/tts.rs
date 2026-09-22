use std::sync::Arc;
use tauri::State;
use tracing::info;
use crate::state::AppState;
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

/// The resolved, absolute path to the shared voice-cloning reference-clip
#[tauri::command]
pub fn get_cloned_tts_voices_dir() -> String {
    fotonvoice_tts::cloned_tts_voices_dir().display().to_string()
}
