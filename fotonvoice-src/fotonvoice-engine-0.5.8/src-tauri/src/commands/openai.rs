use super::*;
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
