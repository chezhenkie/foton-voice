use std::sync::Arc;
use tauri::State;
use crate::state::AppState;
use super::*;
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
