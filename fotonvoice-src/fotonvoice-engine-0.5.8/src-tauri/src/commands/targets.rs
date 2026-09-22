use std::sync::Arc;
use tauri::State;
use tracing::info;
use fotonvoice_routing::{HotkeyBinding, OutputTarget};
use crate::state::AppState;
use super::*;
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
