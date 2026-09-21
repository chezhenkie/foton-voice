use std::sync::{Arc, OnceLock};
use std::time::Duration;
use tauri::{
    tray::{MouseButton, TrayIconBuilder, TrayIconEvent},
    Emitter, Manager,
};
use crate::state::AppState;

/// The tray entry that doubles as the setup indicator.
static SETUP_MENU_ITEM: OnceLock<tauri::menu::MenuItem<tauri::Wry>> = OnceLock::new();

pub const TRAY_SETUP_OK: &str = "  Setup & Diagnostics";
#[cfg(target_os = "linux")]
pub const TRAY_SETUP_BROKEN: &str = "!  Global shortcuts unavailable";

/// Reflect setup state in the tray, which is the one piece of FotonVoice Engine UI that
/// is always on screen.
#[cfg(target_os = "linux")]
pub fn update_tray_for_setup(app: &tauri::AppHandle, ok: bool) {
    let app = app.clone();
    let handle = app.clone();
    let _ = app.run_on_main_thread(move || {
        if let Some(item) = SETUP_MENU_ITEM.get() {
            let _ = item.set_text(if ok { TRAY_SETUP_OK } else { TRAY_SETUP_BROKEN });
        }
        if let Some(tray) = handle.tray_by_id("main-tray") {
            let _ = tray.set_tooltip(Some(if ok {
                "FotonVoice Engine"
            } else {
                "FotonVoice Engine - global shortcuts are unavailable"
            }));
        }
    });
}

/// The tray entry that mirrors (and toggles) the TTS memory mode.
static TTS_MEMORY_MENU_ITEM: OnceLock<tauri::menu::CheckMenuItem<tauri::Wry>> = OnceLock::new();

pub const TRAY_TTS_MEMORY: &str = "  Unload TTS model when idle";

/// Keep the tray checkbox in step with the setting, whichever side changed it
/// (the tray item itself, or the TTS settings tab).
pub fn update_tray_tts_memory(app: &tauri::AppHandle, on_demand: bool) {
    let app = app.clone();
    let _ = app.run_on_main_thread(move || {
        if let Some(item) = TTS_MEMORY_MENU_ITEM.get() {
            let _ = item.set_checked(on_demand);
        }
    });
}

/// Flip the TTS memory mode from the tray: persist it, tell the running worker
/// (no restart - that would tear down the engine and its audio device), and
/// mirror the new state back into the settings window.
fn toggle_tts_memory_mode(app: &tauri::AppHandle) {
    use tauri::Manager;

    let app = app.clone();
    tauri::async_runtime::spawn(async move {
        let state = app.state::<Arc<AppState>>().inner().clone();

        let new_data = {
            let mut guard = state.config.lock().await;
            let on_demand = !guard.data.tts.unloads_when_idle();
            guard.data.tts.memory_mode = if on_demand {
                fotonvoice_config::TtsMemoryMode::OnDemand
            } else {
                fotonvoice_config::TtsMemoryMode::AlwaysLoaded
            };
            if let Err(e) = guard.save() {
                tracing::error!("Failed to save TTS memory mode from tray: {e}");
            }
            guard.data.clone()
        };

        let on_demand = new_data.tts.unloads_when_idle();
        if let Some(tts) = state.tts_handle.lock().await.as_ref() {
            tts.update_config(new_data.tts.clone());
        }

        update_tray_tts_memory(&app, on_demand);
        let _ = app.emit("config-changed", new_data.clone());

        fotonvoice_inject::show_notification(
            "FotonVoice Engine",
            &if on_demand {
                format!(
                    "TTS model will unload after {} minutes idle to save memory.",
                    new_data.tts.idle_unload_duration().as_secs() / 60
                )
            } else {
                "TTS model will stay loaded in memory for the fastest response.".to_string()
            },
        );
    });
}

pub fn create_tray(app: &tauri::App) -> Result<tauri::tray::TrayIcon, tauri::Error> {
    let record_off_icon = tauri::image::Image::from_bytes(include_bytes!("../../assets/record_off.png"))
        .expect("Failed to load record_off icon");
    let tray_icon = record_off_icon.clone();

    let settings_i = tauri::menu::MenuItem::with_id(app, "settings", "  Settings", true, None::<&str>)?;
    let setup_i = tauri::menu::MenuItem::with_id(app, "setup", TRAY_SETUP_OK, true, None::<&str>)?;
    let tts_memory_i = tauri::menu::CheckMenuItem::with_id(
        app,
        "tts_memory",
        TRAY_TTS_MEMORY,
        true,
        // The real value is applied right after the tray is built (see
        // `sync_tts_memory_item`); a `try_lock` here can miss at startup.
        false,
        None::<&str>,
    )?;
    let separator = tauri::menu::PredefinedMenuItem::separator(app)?;
    let quit_i = tauri::menu::MenuItem::with_id(app, "quit", "Quit FotonVoice Engine", true, None::<&str>)?;
    let menu = tauri::menu::Menu::with_items(
        app,
        &[&settings_i, &setup_i, &tts_memory_i, &separator, &quit_i],
    )?;
    let _ = SETUP_MENU_ITEM.set(setup_i);
    let _ = TTS_MEMORY_MENU_ITEM.set(tts_memory_i);

    TrayIconBuilder::with_id("main-tray")
        .icon(tray_icon)
        .tooltip("FotonVoice Engine")
        .menu(&menu)
        .on_menu_event(|app, event| {
            match event.id().as_ref() {
                "settings" => {
                    if let Err(e) = crate::window::open_settings_window(app) {
                        tracing::error!("Could not open Settings: {e}");
                    }
                }
                "setup" => {
                    crate::window::show_setup_window();
                }
                "tts_memory" => {
                    toggle_tts_memory_mode(app);
                }
                "quit" => {
                    // Exit immediately. Do NOT await TTS engine teardown here:
                    // a hanging audio/ORT session shutdown would block
                    // app.exit(0) and leave a zombie process in the tray.
                    // The worker processes are reaped by the OS on exit.
                    //
                    // Failsafe (2026-09-21): a kernel-stuck thread (cpal WASAPI
                    // stream release or the WebGPU/D3D12 teardown path) can keep
                    // the process alive after exit. Two mitigations, both
                    // fire-and-forget so the Tray Quit Law stays synchronous:
                    // 1. release the capture streams now (drop happens on the
                    //    audio thread, not awaited here);
                    // 2. a detached watchdog thread force-exits 2s later if the
                    //    graceful exit has not completed by then.
                    if let Some(state) = app.try_state::<AppState>() {
                        state.recording.store(false, std::sync::atomic::Ordering::SeqCst);
                        state.monitoring.store(false, std::sync::atomic::Ordering::SeqCst);
                    }
                    std::thread::Builder::new()
                        .name("quit-watchdog".into())
                        .spawn(|| {
                            std::thread::sleep(Duration::from_secs(2));
                            tracing::warn!("quit watchdog fired: graceful exit did not complete in 2s");
                            std::process::exit(0);
                        })
                        .ok();
                    app.exit(0);
                }
                _ => {}
            }
        })
        .on_tray_icon_event(|tray, event| {
            if let TrayIconEvent::Click { button: MouseButton::Left, .. } = event {
                if let Err(e) = crate::window::open_settings_window(tray.app_handle()) {
                    tracing::error!("Could not open Settings: {e}");
                }
            }
        })
        .build(app)
}

/// Show the stored TTS memory mode on the freshly built tray item.
pub fn sync_tts_memory_item(app: tauri::AppHandle, state: Arc<AppState>) {
    tauri::async_runtime::spawn(async move {
        let on_demand = state.config.lock().await.data.tts.unloads_when_idle();
        update_tray_tts_memory(&app, on_demand);
    });
}

/// Emits `status-tick`, animates the tray icon, and builds the dictation
/// overlay window if the startup pre-build failed (see `lib.rs`).
///
/// The overlay window is normally built during startup and kept for the rest
/// of the session; this loop builds it on the first activation only if that
/// failed. What is on screen is decided inside it by `Overlay.svelte`, which
/// renders nothing while idle. The old Linux destroy/recreate per dictation
/// (a WebKitGTK stale-frame workaround) is gone: its cause is fixed by the
/// host-first WebKitGTK AppImage packaging (see
/// `scripts/appimage-hooks/host-first-fallback.sh`), and rebuilding cost a
/// black box flash plus a page load on every keybind press. Linux also gets
/// the no-outputs suspend poll and the post-dictation repaint nudge, both
/// cfg-gated here.
pub fn spawn_status_ticker(
    handle: tauri::AppHandle,
    state_for_ticker: Arc<AppState>,
    record_on_icon: tauri::image::Image<'static>,
    record_off_icon: tauri::image::Image<'static>,
    processing_frames: [tauri::image::Image<'static>; 6],
) {
    tokio::spawn(async move {
        // The tray animation frame rate, and the ceiling on how quickly a
        // state change reaches the UI.
        let mut interval = tokio::time::interval(Duration::from_millis(150));
        // The status window falls back to polling when ticks stop arriving,
        // so an idle app still sends one of these - just not seven a second.
        const HEARTBEAT: Duration = Duration::from_millis(900);

        let mut last_recording = false;
        let mut was_animating = false;
        let mut frame_idx = 0;
        let mut last_pos: Option<(String, String)> = None;
        let mut startup_tick_count: u32 = 0;
        // What the last emitted payload said, so a tick that changes nothing
        // costs a few atomic loads instead of building a payload, two JSON
        // encodes, a webview event and a message to the overlay process.
        let mut last_flags: Option<(bool, bool, bool, bool, bool, u32, bool)> = None;
        let mut last_emit = tokio::time::Instant::now() - HEARTBEAT;
        // The label is derived from three rarely-changing strings; caching it
        // keeps the common tick from rebuilding and re-joining it.
        let mut label_inputs: Option<(String, String, bool)> = None;
        let mut cached_label = String::new();
        // Linux only: whether the overlay window has been built by this loop
        // yet (the startup pre-build in lib.rs is not visible here; a failed
        // pre-build is recovered on the first activation).
        #[cfg(target_os = "linux")]
        let mut overlay_built = false;
        // Guards against a SIGFPE in the bundled WebKitGTK's vblank thread
        // when the compositor briefly has zero outputs - see
        // `window::suspend_overlay_without_outputs`. (upstream 64209ba)
        #[cfg(target_os = "linux")]
        let mut overlay_suspended_no_outputs = false;
        #[cfg(target_os = "linux")]
        let mut last_output_check = tokio::time::Instant::now() - crate::window::NO_OUTPUTS_POLL_INTERVAL;
        // Whether the overlay had something to show on the previous tick -
        // drives the post-dictation repaint nudge below. (upstream 769f6b1)
        #[cfg(target_os = "linux")]
        let mut was_showing_overlay = false;

        loop {
            interval.tick().await;
            startup_tick_count = startup_tick_count.saturating_add(1);
            let is_recording = state_for_ticker.is_recording();
            let is_processing = state_for_ticker.is_processing();

            // Decide whether the tray icon needs updating this tick and,
            // if so, which frame to show. The actual `set_icon` call must
            // happen on the GTK main thread: on Linux the tray is backed by
            // ayatana-appindicator/GTK, which is not thread-safe. Calling it
            // from this Tokio worker thread makes icon updates unreliable -
            // the animated icon flickers or disappears entirely on
            // appindicator-based desktops (e.g. GNOME). `run_on_main_thread`
            // marshals the update onto the loop that owns the tray.
            let next_icon: Option<tauri::image::Image<'static>> = if is_processing {
                let icon = processing_frames[frame_idx].clone();
                frame_idx = (frame_idx + 1) % 6;
                was_animating = true;
                Some(icon)
            } else if was_animating || is_recording != last_recording {
                was_animating = false;
                Some(if is_recording { record_on_icon.clone() } else { record_off_icon.clone() })
            } else {
                None
            };

            if let Some(icon) = next_icon {
                let handle_for_icon = handle.clone();
                let _ = handle.run_on_main_thread(move || {
                    if let Some(tray) = handle_for_icon.tray_by_id("main-tray") {
                        let _ = tray.set_icon(Some(icon));
                    }
                });
            }

            last_recording = is_recording;

            // Overlay placement follows the user's config, which is compared
            // under the lock so an unchanged setting costs no allocation.
            let mut position_changed = false;
            {
                let cfg = state_for_ticker.config.lock().await;
                let ui = &cfg.data.ui;
                let unchanged = last_pos.as_ref().is_some_and(|(pos, mon)| {
                    pos == &ui.overlay_position && mon == &ui.overlay_monitor
                });
                if !unchanged {
                    last_pos = Some((
                        ui.overlay_position.clone(),
                        ui.overlay_monitor.clone(),
                    ));
                    position_changed = true;
                }
            }
            if position_changed || startup_tick_count < 40 {
                // Send the anchor + monitor; the overlay computes pixel
                // coordinates itself using its own display scale.
                let (position, monitor) = last_pos
                    .as_ref()
                    .map(|(pos, mon)| (pos.as_str(), mon.as_str()))
                    .expect("set above");
                let pos_msg = serde_json::json!({
                    "type": "position",
                    "position": position,
                    "monitor": monitor,
                });
                if let Ok(json_str) = serde_json::to_string(&pos_msg) {
                    let _ = state_for_ticker.overlay_tx.send(json_str);
                }
            }

            // -- Overlay window build fallback (Linux only) ------------------------
            // Build the overlay window the first time there is anything to
            // show, then keep it - normally lib.rs already built it at
            // startup and this is a no-op fetch. Mirrors the same condition
            // Overlay.svelte derives client-side for its own content.
            #[cfg(target_os = "linux")]
            let should_show_overlay = {
                let cfg = state_for_ticker.config.lock().await;
                (is_recording && state_for_ticker.is_overlay_enabled())
                    || (state_for_ticker.is_speaking()
                        && cfg.data.tts.enabled
                        && cfg.data.tts.response_overlay)
                    || (state_for_ticker.is_mcp_recording() && cfg.data.mcp.visual_feedback)
                    || state_for_ticker.is_command_overlay_active()
            };
            #[cfg(target_os = "linux")]
            if should_show_overlay && !overlay_built {
                let (position, monitor) = last_pos
                    .as_ref()
                    .map(|(pos, mon)| (pos.as_str(), mon.as_str()))
                    .unwrap_or(("center", "primary"));
                match crate::window::open_overlay_window(&handle, position, monitor) {
                    Ok(_) => overlay_built = true,
                    Err(e) => tracing::error!("Failed to open the dictation overlay: {e}"),
                }
            }

            // On some WebKitGTK/compositor combinations the overlay's client
            // buffer never repaints back to blank once the last dictation's
            // content is gone - see `window::nudge_overlay_repaint`. Nudging
            // twice, at 1s then 3s after the transition to idle, mirrors the
            // reporter's own timing; harmless when nothing was stuck.
            #[cfg(target_os = "linux")]
            if overlay_built && was_showing_overlay && !should_show_overlay {
                let nudge_handle = handle.clone();
                tokio::spawn(async move {
                    tokio::time::sleep(Duration::from_secs(1)).await;
                    crate::window::nudge_overlay_repaint(&nudge_handle).await;
                    tokio::time::sleep(Duration::from_secs(2)).await;
                    crate::window::nudge_overlay_repaint(&nudge_handle).await;
                });
            }
            #[cfg(target_os = "linux")]
            {
                was_showing_overlay = should_show_overlay;
            }

            #[cfg(target_os = "linux")]
            if overlay_built && last_output_check.elapsed() >= crate::window::NO_OUTPUTS_POLL_INTERVAL {
                last_output_check = tokio::time::Instant::now();
                crate::window::suspend_overlay_without_outputs(&handle, &mut overlay_suspended_no_outputs);
            }

            let active_target_id = state_for_ticker.active_target.lock().await.clone();
            let binding_label = state_for_ticker.active_binding_label.lock().await.clone();
            let use_binding_label = (is_recording || is_processing) && !binding_label.is_empty();
            let label_changed = match &label_inputs {
                Some((target, binding, from_binding)) => {
                    target != &active_target_id
                        || binding != &binding_label
                        || *from_binding != use_binding_label
                }
                None => true,
            };
            if label_changed {
                label_inputs = Some((
                    active_target_id.clone(),
                    binding_label.clone(),
                    use_binding_label,
                ));
                cached_label = if use_binding_label {
                    binding_label
                } else {
                    let targets_guard = state_for_ticker.targets.lock().await;
                    active_target_id
                        .split(',')
                        .map(|s| s.trim())
                        .filter(|s| !s.is_empty())
                        .map(|id| {
                            targets_guard
                                .iter()
                                .find(|t| t.id == id)
                                .map(|t| t.label.clone())
                                .unwrap_or_else(|| {
                                    if id == "default" {
                                        "Focused Window".to_string()
                                    } else {
                                        id.to_string()
                                    }
                                })
                        })
                        .collect::<Vec<_>>()
                        .join(" + ")
                };
            }

            // Everything the payload carries, compared before one is built:
            // the label and target id are the cached strings above, so an
            // unchanged tick allocates nothing at all.
            let flags = (
                is_recording,
                is_processing,
                state_for_ticker.is_speaking(),
                state_for_ticker.is_mcp_recording(),
                state_for_ticker.is_audio_ready(),
                state_for_ticker.total_words(),
                state_for_ticker.hotkey_health.is_active(),
            );
            let now = tokio::time::Instant::now();
            let unchanged = last_flags == Some(flags) && !label_changed;
            if unchanged && now.duration_since(last_emit) < HEARTBEAT {
                continue;
            }
            last_flags = Some(flags);
            last_emit = now;

            let payload = serde_json::json!({
                "recording": flags.0,
                "processing": flags.1,
                "speaking": flags.2,
                "mcp_recording": flags.3,
                "audio_ready": flags.4,
                "word_count": flags.5,
                "hotkeys_active": flags.6,
                "active_target_id": &active_target_id,
                "active_target_label": &cached_label,
            });

            let _ = handle.emit("status-tick", &payload);
        }
    });
}
