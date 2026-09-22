use std::sync::{Arc, OnceLock};
use std::time::Duration;
use tauri::Manager;
use crate::commands;
use crate::state::AppState;

/// Set once the Tauri app is built, so background tasks started before it (the
static APP_HANDLE: OnceLock<tauri::AppHandle> = OnceLock::new();

/// Label of the first-run setup window.
pub const SETUP_WINDOW: &str = "udev-warning";

/// Default and minimum geometry for the Settings window. Its sidebar plus the
pub const SETTINGS_WIDTH: f64 = 880.0;
pub const SETTINGS_HEIGHT: f64 = 1000.0;
pub const SETTINGS_MIN_WIDTH: f64 = 720.0;
pub const SETTINGS_MIN_HEIGHT: f64 = 640.0;

/// Minimum gap between "finish the setup" notifications, so holding a
pub const SETUP_NOTICE_INTERVAL: Duration = Duration::from_secs(60);

/// How often the setup watcher re-checks the listener. Short enough that a
#[cfg(target_os = "linux")]
pub const SETUP_POLL_INTERVAL: Duration = Duration::from_secs(2);

/// Re-alert cadence while nothing can deliver shortcuts at all. In that state
#[cfg(target_os = "linux")]
pub const BLIND_ALERT_INTERVAL: Duration = Duration::from_secs(300);

pub fn set_app_handle(handle: tauri::AppHandle) {
    let _ = APP_HANDLE.set(handle);
}

pub fn get_app_handle() -> Option<tauri::AppHandle> {
    APP_HANDLE.get().cloned()
}

/// Show the Settings window, building it if the user has closed it.
pub fn open_settings_window(app: &tauri::AppHandle) -> Result<tauri::WebviewWindow, String> {
    if let Some(existing) = app.get_webview_window("settings") {
        show_and_focus_window(&existing);
        return Ok(existing);
    }

    tauri::WebviewWindowBuilder::new(app, "settings", tauri::WebviewUrl::App("/settings".into()))
        .title("FotonVoice Engine Settings")
        .inner_size(SETTINGS_WIDTH, SETTINGS_HEIGHT)
        .min_inner_size(SETTINGS_MIN_WIDTH, SETTINGS_MIN_HEIGHT)
        .center()
        .resizable(true)
        .decorations(true)
        .build()
        .map_err(|e| format!("Could not open Settings: {e}"))
}

/// Label of the dictation overlay window.
pub const OVERLAY_WINDOW: &str = "overlay";
#[cfg(target_os = "linux")]
const OVERLAY_WIDTH: f64 = 1184.0;
#[cfg(target_os = "linux")]
const OVERLAY_HEIGHT: f64 = 444.0;
#[cfg(not(target_os = "linux"))]
const OVERLAY_WIDTH: f64 = 592.0;
#[cfg(not(target_os = "linux"))]
const OVERLAY_HEIGHT: f64 = 222.0;

/// Build (or fetch) the dictation overlay: a transparent, frameless,
const OVERLAY_REVEAL_TIMEOUT: Duration = Duration::from_millis(3000);

/// Make the overlay window visible, if it is not already.
pub fn reveal_overlay(app: &tauri::AppHandle) {
    let Some(window) = app.get_webview_window(OVERLAY_WINDOW) else {
        return;
    };
    #[cfg(target_os = "linux")]
    {
        use gtk::prelude::*;
        if let Ok(gtk_window) = window.gtk_window() {
            gtk_window.set_opacity(1.0);
        }
    }
    #[cfg(not(target_os = "linux"))]
    let _ = window;
}

pub fn open_overlay_window(
    app: &tauri::AppHandle,
    anchor: &str,
    monitor_pref: &str,
) -> Result<tauri::WebviewWindow, String> {
    let is_new = app.get_webview_window(OVERLAY_WINDOW).is_none();

    let window = match app.get_webview_window(OVERLAY_WINDOW) {
        Some(existing) => existing,
        None => tauri::WebviewWindowBuilder::new(
            app,
            OVERLAY_WINDOW,
            tauri::WebviewUrl::App("/overlay".into()),
        )
        .title("FotonVoice Engine Overlay")
        .inner_size(OVERLAY_WIDTH, OVERLAY_HEIGHT)
        .resizable(false)
        .decorations(false)
        .transparent(true)
        .shadow(false)
        .always_on_top(true)
        .skip_taskbar(true)
        .focused(false)
        .closable(false)
        .minimizable(false)
        .maximizable(false)
        .visible(false)
        .build()
        .map_err(|e| format!("Could not create the overlay window: {e}"))?,
    };

    #[cfg(target_os = "linux")]
    {
        use gtk::prelude::*;
        if let Ok(gtk_window) = window.gtk_window() {
            gtk_window.realize();
            gtk_window.set_accept_focus(false);
            gtk_window.set_can_focus(false);
            gtk_window.set_focus_on_map(false);
            if let Some(gdk_window) = gtk_window.window() {
                gdk_window.set_type_hint(gtk::gdk::WindowTypeHint::Utility);
            }
            if is_new {
                gtk_window.set_opacity(0.0);
            }
        }
    }

    reposition_overlay_inner(&window, anchor, monitor_pref);

    if let Err(e) = window.show() {
        tracing::error!("Failed to show overlay window: {:?}", e);
    }

    if let Err(e) = window.set_ignore_cursor_events(true) {
        tracing::warn!("Failed to make overlay window click-through: {:?}", e);
    }

    reposition_overlay_inner(&window, anchor, monitor_pref);

    if is_new {
        let handle = app.clone();
        tauri::async_runtime::spawn(async move {
            tokio::time::sleep(OVERLAY_REVEAL_TIMEOUT).await;
            reveal_overlay(&handle);
        });
    }

    Ok(window)
}

/// Re-apply the anchor/monitor position to the overlay window, if it
pub fn reposition_overlay(app: &tauri::AppHandle, anchor: &str, monitor_pref: &str) {
    if let Some(window) = app.get_webview_window(OVERLAY_WINDOW) {
        reposition_overlay_inner(&window, anchor, monitor_pref);
    }
}

fn reposition_overlay_inner(window: &tauri::WebviewWindow, anchor: &str, monitor_pref: &str) {
    if let Some((x, y)) = compute_overlay_window_position(window, anchor, monitor_pref) {
        let _ = window.set_position(tauri::PhysicalPosition::new(x, y));
    }
}

/// Re-assert the overlay window's always-on-top state, if it exists.
pub fn reassert_overlay_topmost(app: &tauri::AppHandle) {
    if let Some(window) = app.get_webview_window(OVERLAY_WINDOW) {
        let _ = window.set_always_on_top(false);
        let _ = window.set_always_on_top(true);
        #[cfg(target_os = "linux")]
        {
            use gtk::prelude::*;
            if let Ok(gtk_window) = window.gtk_window() {
                gtk_window.set_accept_focus(false);
                gtk_window.set_can_focus(false);
            }
        }
    }
}

/// Poll interval for [`suspend_overlay_without_outputs`]. Far coarser than
#[cfg(target_os = "linux")]
pub const NO_OUTPUTS_POLL_INTERVAL: Duration = Duration::from_secs(2);

/// Hide the overlay window while the compositor reports no real outputs, and
#[cfg(target_os = "linux")]
pub fn suspend_overlay_without_outputs(app: &tauri::AppHandle, suspended: &mut bool) {
    let Some(window) = app.get_webview_window(OVERLAY_WINDOW) else {
        return;
    };
    let has_outputs = window.available_monitors().map(|m| !m.is_empty()).unwrap_or(true);

    if !has_outputs && !*suspended {
        *suspended = true;
        tracing::info!(
            "No display outputs reported; hiding the overlay window until one returns \
             (works around a SIGFPE in the bundled WebKitGTK's vblank thread)"
        );
        let _ = window.hide();
    } else if has_outputs && *suspended {
        *suspended = false;
        let _ = window.show();
    }
}

/// Force the overlay window's client-side buffer to repaint by resizing it
#[cfg(target_os = "linux")]
pub async fn nudge_overlay_repaint(app: &tauri::AppHandle) {
    let Some(window) = app.get_webview_window(OVERLAY_WINDOW) else {
        return;
    };
    let _ = window.set_size(tauri::LogicalSize::new(OVERLAY_WIDTH + 2.0, OVERLAY_HEIGHT));
    tokio::time::sleep(Duration::from_millis(120)).await;
    let _ = window.set_size(tauri::LogicalSize::new(OVERLAY_WIDTH, OVERLAY_HEIGHT));
}

/// Top-left Y for the overlay given the anchor, in the same pixel space as
fn overlay_anchor_y(monitor_y: i32, monitor_height: i32, window_height: i32, margin: i32, anchor: &str) -> i32 {
    match anchor {
        "top" => monitor_y + margin,
        "bottom" => monitor_y + monitor_height - window_height - margin,
        _ => monitor_y + (monitor_height - window_height) / 2, // "center" default
    }
}

/// Compute the overlay's top-left pixel position for an anchor, using the
fn compute_overlay_window_position(
    window: &tauri::WebviewWindow,
    anchor: &str,
    monitor_pref: &str,
) -> Option<(i32, i32)> {
    let monitors = window.available_monitors().ok()?;
    let monitor = if monitor_pref.is_empty() || monitor_pref == "primary" {
        window.primary_monitor().ok().flatten().or_else(|| monitors.first().cloned())
    } else {
        monitors
            .iter()
            .find(|m| m.name().map(|n| n.as_str()) == Some(monitor_pref))
            .cloned()
            .or_else(|| window.primary_monitor().ok().flatten())
            .or_else(|| monitors.first().cloned())
    }?;

    let msize = monitor.size();
    let mpos = monitor.position();
    let scale = monitor.scale_factor();
    let margin = (60.0 * scale) as i32;

    let wwidth = (OVERLAY_WIDTH * scale) as i32;
    let wheight = (OVERLAY_HEIGHT * scale) as i32;

    let x = mpos.x + (msize.width as i32 - wwidth) / 2;
    let y = overlay_anchor_y(mpos.y, msize.height as i32, wheight, margin, anchor);
    Some((x, y))
}

/// Bring the setup window to the front, building it if it has been closed.
pub fn show_setup_window() {
    let Some(handle) = APP_HANDLE.get() else {
        return;
    };
    if let Some(w) = handle.get_webview_window(SETUP_WINDOW) {
        raise_window(&w, true);
        return;
    }

    let built = tauri::WebviewWindowBuilder::new(
        handle,
        SETUP_WINDOW,
        tauri::WebviewUrl::App("/udev-warning".into()),
    )
    .title("FotonVoice Engine Setup")
    .inner_size(580.0, 600.0)
    .min_inner_size(480.0, 420.0)
    .center()
    .always_on_top(true)
    .resizable(true)
    .build();
    if let Err(e) = built {
        tracing::error!("Could not open the setup window: {e}");
    }
}

/// What is stopping dictation from working end to end, as a message to show the
pub async fn setup_blocker(state: &Arc<AppState>) -> Option<String> {
    if let Some(tool) = commands::missing_injection_tool() {
        return Some(format!(
            "FotonVoice Engine cannot type text into other windows: '{tool}' is not installed. \
             Finish the setup in FotonVoice Engine -> Settings to install it."
        ));
    }

    let cfg = state.config.lock().await;
    let eng = &cfg.data.engine;
    let uses_whisper_model = eng.backend != fotonvoice_config::BackendChoice::RemoteOpenAi
        && (eng.backend != fotonvoice_config::BackendChoice::Moonshine
            || !fotonvoice_inference::MOONSHINE_COMPILED)
        && (eng.backend != fotonvoice_config::BackendChoice::Parakeet
            || !fotonvoice_inference::PARAKEET_COMPILED)
        && (eng.backend != fotonvoice_config::BackendChoice::NemotronStreaming
            || !fotonvoice_inference::NEMOTRON_STREAMING_COMPILED);
    if !uses_whisper_model {
        return None;
    }
    if fotonvoice_inference::whisper_cpp::is_model_downloaded(
        &eng.whisper_cpp.model_size,
        &eng.whisper_cpp.model_dir,
    ) {
        return None;
    }

    Some(format!(
        "Speech model '{}' is not downloaded - dictation cannot produce text. \
         Open Settings -> Engine and download it.",
        eng.whisper_cpp.model_size
    ))
}

/// Helper to robustly show, unminimize, and focus a window
pub fn show_and_focus_window(window: &tauri::WebviewWindow) {
    raise_window(window, false);
}

/// Bring a window to the front and give it keyboard focus.
pub fn raise_window(window: &tauri::WebviewWindow, keep_on_top: bool) {
    let _ = window.unminimize();
    let _ = window.show();
    let _ = window.set_always_on_top(true);
    let _ = window.set_focus();

    if !keep_on_top {
        let w = window.clone();
        tauri::async_runtime::spawn(async move {
            tokio::time::sleep(Duration::from_millis(250)).await;
            let _ = w.set_always_on_top(false);
        });
    }
}

