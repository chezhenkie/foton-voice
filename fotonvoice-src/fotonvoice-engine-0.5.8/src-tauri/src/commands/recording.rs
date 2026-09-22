use std::sync::Arc;
use tauri::State;
use tracing::info;
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
