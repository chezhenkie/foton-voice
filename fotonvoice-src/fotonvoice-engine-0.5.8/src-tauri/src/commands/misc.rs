use tauri::{Emitter, Manager};
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
