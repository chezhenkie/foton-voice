use std::sync::{Arc, OnceLock};
use std::time::Duration;
use tauri::Manager;
use crate::commands;
use crate::state::AppState;

/// Set once the Tauri app is built, so background tasks started before it (the
/// hotkey gesture loop) can raise windows.
static APP_HANDLE: OnceLock<tauri::AppHandle> = OnceLock::new();

/// Label of the first-run setup window.
pub const SETUP_WINDOW: &str = "udev-warning";

/// Default and minimum geometry for the Settings window. Its sidebar plus the
/// widest tab body need the width, and the longest tab needs the height before
/// it starts scrolling on first open.
pub const SETTINGS_WIDTH: f64 = 880.0;
pub const SETTINGS_HEIGHT: f64 = 1000.0;
pub const SETTINGS_MIN_WIDTH: f64 = 720.0;
pub const SETTINGS_MIN_HEIGHT: f64 = 640.0;

/// Minimum gap between "finish the setup" notifications, so holding a
/// push-to-talk key does not produce a wall of toasts.
pub const SETUP_NOTICE_INTERVAL: Duration = Duration::from_secs(60);

/// How often the setup watcher re-checks the listener. Short enough that a
/// change made elsewhere - the portal coming up, a shortcut reassigned in the
/// desktop's settings - visibly flips the app to working within a couple of
/// seconds.
#[cfg(target_os = "linux")]
pub const SETUP_POLL_INTERVAL: Duration = Duration::from_secs(2);

/// Re-alert cadence while nothing can deliver shortcuts at all. In that state
/// no shortcut can reach the app, so this is the only way the user hears about
/// it while they are pressing keys and getting nothing.
#[cfg(target_os = "linux")]
pub const BLIND_ALERT_INTERVAL: Duration = Duration::from_secs(300);

pub fn set_app_handle(handle: tauri::AppHandle) {
    let _ = APP_HANDLE.set(handle);
}

pub fn get_app_handle() -> Option<tauri::AppHandle> {
    APP_HANDLE.get().cloned()
}

/// Show the Settings window, building it if the user has closed it.
///
/// Closing a window destroys it, so every entry point into Settings - the tray,
/// a second launch - has to be able to make a new one. Geometry is kept in step with the `settings` entry in tauri.conf.json.
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
// The overlay's actual content (the Retro Terminal panel) is centered via
// flex inside this window with real margin to spare, so it stays comfortably
// clear of the window edges. On Linux this has to absorb the rendered size
// sometimes coming out a few px smaller than requested: forcing
// GDK_BACKEND=x11 (see lib.rs) makes GDK approximate the display's real
// scale factor - often fractional under Wayland - by rounding to an integer
// X11 scale, which showed up as the window edge clipping into rounded
// corners before this was widened.
//
// On Linux the window is doubled (upstream 0.5.9, kept): the larger surface
// gives the overlay room to render its load/outro animations into. Windows
// keeps the original size, since its layout is verified-good.
#[cfg(target_os = "linux")]
const OVERLAY_WIDTH: f64 = 1184.0;
#[cfg(target_os = "linux")]
const OVERLAY_HEIGHT: f64 = 444.0;
#[cfg(not(target_os = "linux"))]
const OVERLAY_WIDTH: f64 = 592.0;
#[cfg(not(target_os = "linux"))]
const OVERLAY_HEIGHT: f64 = 222.0;

/// Build (or fetch) the dictation overlay: a transparent, frameless,
/// always-on-top, click-through `WebviewWindow` rendering the `/overlay`
/// Svelte route (`src/lib/Overlay/Overlay.svelte`) - the same component tree
/// that renders the Retro Terminal panel and the speaking/command/MCP
/// pills, already wired to the app-wide `status-tick` / `audio-level` events.
///
/// `anchor` / `monitor_pref` are `config.ui.overlay_position` /
/// `overlay_monitor`.
///
/// How long the overlay may stay hidden waiting for its content to paint.
/// The frontend normally reports in well inside this (see `reveal_overlay`),
/// so this only matters when it cannot - a custom overlay whose script throws
/// before the report, say, or a first paint that is simply slow. Was 600ms
/// until a Hyprland/XWayland box with an older iGPU revealed the overlay
/// indefinitely after the timeout fired before WebKitGTK's first real paint;
/// 3s gives a slow first paint room to finish. (upstream 1444f4d)
const OVERLAY_REVEAL_TIMEOUT: Duration = Duration::from_millis(3000);

/// Make the overlay window visible, if it is not already.
///
/// The window is built fully transparent and revealed once the frontend
/// reports it has painted (`overlay_content_ready`, see commands.rs).
/// Opacity rather than unmapped or offscreen: an unmapped webview may never
/// render, an offscreen position can be clamped by the WM, a mapped fully
/// transparent window renders normally and is reliably invisible - including
/// any frame the WM draws around it. No compositor: set_opacity does nothing
/// and the overlay behaves as before. (upstream 9e33af5)
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
    // Only a window built by this call starts hidden. Re-entering with one
    // already on screen must not blank the overlay the user is looking at.
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

    // Everything the window manager reads when presenting a window must be
    // set BEFORE mapping: applied afterwards, the window is mapped as a
    // default-type window at a WM-chosen position and re-evaluated a moment
    // later, which on KWin draws a brief black frame on every activation.
    // `realize()` creates the underlying GdkWindow without mapping it, which
    // is what gives us somewhere to put these first. (upstream 2307242)
    //
    // Click-through is mouse only. An XWayland overlay that accepts keyboard
    // focus eats Space-up of CTRL+SPACE; Hyprland then never sends the portal
    // Deactivated until focus returns - recording hangs after release.
    // (upstream ae814d6)
    //
    // `Utility` rather than `Notification`: the app is forced through
    // XWayland on Linux (see `lib.rs`'s `GDK_BACKEND=x11` override - this
    // window's absolute positioning and always-on-top depend on it), and
    // KWin's X11 compositing path gives `_NET_WM_WINDOW_TYPE_NOTIFICATION`
    // windows different, short-lived-oriented repaint handling. Reported
    // symptom: every overlay style, on every keybind release, freezes solid
    // on screen - sometimes clearing when another application is launched,
    // never on its own, only fixed by quitting the app entirely - despite
    // the overlay's own content genuinely finishing its unmount.
    // `Utility` is KWin's ordinary type for a persistent always-on-top panel
    // and goes through the normal compositing/repaint path, while still
    // being excluded from alt-tab in every WM this has been checked against.
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
            // Mapped but fully transparent until its content has painted -
            // see `reveal_overlay`. Set before `show()` so there is no frame
            // in which the window is both mapped and opaque.
            if is_new {
                gtk_window.set_opacity(0.0);
            }
        }
    }

    // Position before mapping too: placed beforehand the window appears
    // where it belongs instead of the WM placing it and us moving it after
    // the fact.
    reposition_overlay_inner(&window, anchor, monitor_pref);

    // On Linux, tao's `set_ignore_cursor_events` reaches into the GTK
    // window's underlying GdkWindow and unwraps it unconditionally
    // (tao/src/platform_impl/linux/event_loop.rs, WindowRequest::CursorIgnoreEvents).
    // That GdkWindow does not exist until the widget is realized, so this
    // has to come after the block above (which realizes it explicitly) or
    // after `show()` (which realizes it as a side effect). Calling it any
    // earlier panics - and, being inside a GTK callback, aborts the whole
    // process instead of unwinding.
    if let Err(e) = window.show() {
        tracing::error!("Failed to show overlay window: {:?}", e);
    }

    // Click-through: mouse events pass to whatever is beneath the overlay.
    if let Err(e) = window.set_ignore_cursor_events(true) {
        tracing::warn!("Failed to make overlay window click-through: {:?}", e);
    }

    reposition_overlay_inner(&window, anchor, monitor_pref);

    // Safety net for the hidden-until-painted gate: if the frontend never
    // reports in, reveal anyway. An overlay that flashes black is a blemish;
    // one that never appears is a broken feature.
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
/// currently exists.
///
/// Called whenever `config.ui.overlay_position` / `overlay_monitor` change
/// (see `commands.rs`'s `save_config` and `tray.rs`'s config-change ticker,
/// both of which send a `{"type":"position","position":..,"monitor":..}`
/// message over `overlay_tx` - see its consumer in `lib.rs`), so
/// position/monitor changes take effect live instead of only at startup.
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
///
/// The window level is not "sticky": another window taking `_NET_WM_STATE_ABOVE`,
/// a fullscreen app, or the compositor re-stacking between dictations can push
/// the overlay behind other windows, and it never recovers on its own.
/// Toggling the level off then back on (rather than setting `true` when it
/// may already be `true`) forces the change to actually reach the window
/// manager rather than being suppressed as a no-op.
///
/// A no-op when the overlay window doesn't exist yet, so callers on the hot
/// audio-level path can call this unconditionally.
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
/// the ticker's 150ms cadence: this only ever reacts to a screen going
/// idle/locked, never to anything on the hot dictation path.
#[cfg(target_os = "linux")]
pub const NO_OUTPUTS_POLL_INTERVAL: Duration = Duration::from_secs(2);

/// Hide the overlay window while the compositor reports no real outputs, and
/// show it again once one comes back. (upstream 64209ba)
///
/// The bundled WebKitGTK paces its compositor off a vblank thread that
/// divides by the current output's refresh rate. When a compositor (seen on
/// Hyprland) drops to zero real outputs - idle, screen lock, DPMS off - and
/// falls back to a placeholder monitor, that monitor has been observed
/// reporting scale 0 to GTK and a refresh rate of 0 into the same divide
/// crashes the whole process with SIGFPE. Unmapping the window for as long
/// as there is nothing to display it on narrows the crash window to the poll
/// interval; it does nothing once WebKitGTK has the divide fixed upstream.
#[cfg(target_os = "linux")]
pub fn suspend_overlay_without_outputs(app: &tauri::AppHandle, suspended: &mut bool) {
    let Some(window) = app.get_webview_window(OVERLAY_WINDOW) else {
        return;
    };
    // Any error is treated as "outputs present": this must never hide the
    // overlay on a desktop that is working normally just because a query
    // failed.
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
/// by a couple of pixels and immediately back. (upstream 769f6b1)
///
/// Reported on KDE/XWayland even with the host's own current WebKitGTK: the
/// overlay stays frozen on its last frame after every dictation, and the
/// stale pixels were confirmed (via XGetImage) to live in the client's own
/// buffer. The reporter found that a real resize - grow 2px, then shrink
/// back, as two separate operations rather than one that cancels out -
/// reliably clears it. Same fix run by the app itself via tauri's
/// `set_size`; costs nothing where nothing was ever stuck.
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
/// the monitor geometry.
fn overlay_anchor_y(monitor_y: i32, monitor_height: i32, window_height: i32, margin: i32, anchor: &str) -> i32 {
    match anchor {
        "top" => monitor_y + margin,
        "bottom" => monitor_y + monitor_height - window_height - margin,
        _ => monitor_y + (monitor_height - window_height) / 2, // "center" default
    }
}

/// Compute the overlay's top-left pixel position for an anchor, using the
/// window's own monitor + size.
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

    // Physical-pixel size of the overlay, computed from the target monitor's
    // scale factor rather than queried via `window.outer_size()`: called this
    // soon after `show()`, outer_size() can still report 0x0 because the
    // resize hasn't round-tripped through the X11/GTK event loop yet. That
    // silently produced `None` here every time, leaving the window wherever
    // the window manager defaults new windows to (its own primary-monitor
    // center) - which is why position/monitor settings appeared to be
    // ignored entirely. OVERLAY_WIDTH/OVERLAY_HEIGHT are fixed (the window is
    // non-resizable), so there's no need to ask the window for its size at all.
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
/// user. `None` means the app is fully set up.
///
/// Called on every hotkey activation: pressing the shortcut and getting silence
/// is the moment the user needs to be told the install is unfinished, and it is
/// the only moment we know for certain they are trying to dictate.
pub async fn setup_blocker(state: &Arc<AppState>) -> Option<String> {
    if let Some(tool) = commands::missing_injection_tool() {
        return Some(format!(
            "FotonVoice Engine cannot type text into other windows: '{tool}' is not installed. \
             Finish the setup in FotonVoice Engine -> Settings to install it."
        ));
    }

    let cfg = state.config.lock().await;
    let eng = &cfg.data.engine;
    // Moonshine or Parakeet only bypasses the Whisper-model check when it is actually
    // compiled in; otherwise the app silently falls back to whisper-cpp and
    // still needs the model.
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
///
/// `set_focus` on its own is not enough on Linux. Every mainstream desktop
/// implements focus-stealing prevention: a process that does not already own
/// the focused window has its focus requests demoted to "urgent" - the taskbar
/// entry blinks and the window stays exactly where it was, behind whatever the
/// user is looking at. A tray click is precisely that case, since the tray is
/// not the app's own window.
///
/// Pinning the window above others is honoured where a bare focus request is
/// not, so the raise is done by pinning, focusing, then unpinning a moment
/// later - long enough for the compositor to restack, short enough that the
/// window does not linger above everything else.
///
/// `keep_on_top` leaves the pin in place, for windows that are meant to stay
/// above other applications.
pub fn raise_window(window: &tauri::WebviewWindow, keep_on_top: bool) {
    // Deliberately no `center()`: this is also the path for a window that is
    // already open, and moving a window the user has placed is not what
    // "bring it to the front" means.
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

