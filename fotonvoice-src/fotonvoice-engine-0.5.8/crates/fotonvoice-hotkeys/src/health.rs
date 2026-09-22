//! Live state of the global hotkey listener.

use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    Mutex,
};

use fotonvoice_routing::GestureType;

/// Which mechanism is delivering shortcuts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Backend {
    /// Nothing is listening yet.
    Starting,
    /// `org.freedesktop.portal.GlobalShortcuts` - the compositor owns the key
    Portal,
    /// X11 raw key events (XInput2). Needs no permissions, and sees every
    X11,
    /// A native Cinnamon/MATE custom shortcut that pokes FotonVoice Engine over D-Bus.
    MintDbus,
    /// Reading `/dev/input/event*` directly. Only reachable when the user has
    Evdev,
    /// The Win32 low-level keyboard hook.
    WindowsHook,
    /// No mechanism is available; shortcuts cannot fire.
    None,
}

/// Gesture styles a backend that only learns about key *presses* can serve.
const PRESS_ONLY_GESTURES: &[GestureType] = &[GestureType::Toggle];

/// Every gesture style, for the backends that see raw presses and releases.
const ALL_GESTURES: &[GestureType] = &[
    GestureType::Hold,
    GestureType::Toggle,
    GestureType::DoubleTap,
    GestureType::DoubleTapHold,
];

impl Backend {
    /// Gesture styles this backend can actually deliver.
    pub fn gestures(self) -> &'static [GestureType] {
        match self {
            Self::MintDbus => PRESS_ONLY_GESTURES,
            Self::Portal | Self::X11 | Self::Evdev | Self::WindowsHook | Self::Starting
            | Self::None => ALL_GESTURES,
        }
    }

    /// True when FotonVoice Engine watches the key stream itself, so a trigger need not be
    pub fn sees_raw_keys(self) -> bool {
        matches!(self, Self::X11 | Self::Evdev | Self::WindowsHook)
    }
}

/// One shortcut as the compositor actually bound it.
#[derive(Debug, Clone, serde::Serialize)]
pub struct BoundShortcut {
    /// Bindings that fire from this shortcut.
    pub binding_ids: Vec<String>,
    /// The accelerator FotonVoice Engine asked for, if the combination could be expressed.
    pub requested: Option<String>,
    /// How the desktop describes the shortcut to the user, e.g. "Meta+Space".
    pub trigger_description: String,
    /// The compositor acknowledged this shortcut.
    pub bound: bool,
}

#[derive(Debug, Default)]
pub struct ListenerHealth {
    keyboards_open: AtomicUsize,
    devices_denied: AtomicUsize,
    devices_total: AtomicUsize,
    scanned: AtomicBool,
    supported: AtomicBool,
    backend: Mutex<Option<Backend>>,
    /// Why the portal could not be used, if it could not.
    portal_error: Mutex<Option<String>>,
    /// Why the X11 backend could not be used, if it could not. Separate from
    x11_error: Mutex<Option<String>>,
    /// The portal is present and answered, but refused the session. A different
    portal_refused: AtomicBool,
    bound_shortcuts: Mutex<Vec<BoundShortcut>>,
    /// Windows only: the foreground window belongs to a process elevated above
    elevated_window_focused: AtomicBool,
}

impl ListenerHealth {
    /// Number of keyboard devices the evdev backend currently has open. Always
    pub fn keyboards_open(&self) -> usize {
        self.keyboards_open.load(Ordering::Relaxed)
    }

    /// Input devices that exist but could not be opened because of file
    pub fn devices_denied(&self) -> usize {
        self.devices_denied.load(Ordering::Relaxed)
    }

    /// Input event devices present on the system, readable or not.
    pub fn devices_total(&self) -> usize {
        self.devices_total.load(Ordering::Relaxed)
    }

    /// True once the evdev backend has completed at least one device scan.
    pub fn has_scanned(&self) -> bool {
        self.scanned.load(Ordering::Relaxed)
    }

    /// True on platforms where a global listener is implemented at all.
    pub fn is_supported(&self) -> bool {
        self.supported.load(Ordering::Relaxed)
    }

    pub fn backend(&self) -> Backend {
        self.backend
            .lock()
            .ok()
            .and_then(|b| *b)
            .unwrap_or(Backend::Starting)
    }

    /// Why the portal backend was not used. `None` means it was.
    pub fn portal_error(&self) -> Option<String> {
        self.portal_error.lock().ok().and_then(|e| e.clone())
    }

    /// The desktop has a shortcuts portal and it turned FotonVoice Engine away, rather
    pub fn portal_refused(&self) -> bool {
        self.portal_refused.load(Ordering::Relaxed)
    }

    /// Why the X11 backend was not used. `None` means it was, or was never
    pub fn x11_error(&self) -> Option<String> {
        self.x11_error.lock().ok().and_then(|e| e.clone())
    }

    /// Windows only: the currently focused window belongs to a more
    pub fn elevated_window_focused(&self) -> bool {
        self.elevated_window_focused.load(Ordering::Relaxed)
    }

    /// Gesture styles the running backend can deliver.
    pub fn gestures(&self) -> &'static [GestureType] {
        self.backend().gestures()
    }

    pub fn bound_shortcuts(&self) -> Vec<BoundShortcut> {
        self.bound_shortcuts.lock().map(|s| s.clone()).unwrap_or_default()
    }

    /// Hotkeys can fire right now.
    pub fn is_active(&self) -> bool {
        if !self.is_supported() {
            return true;
        }
        match self.backend() {
            Backend::WindowsHook => !self.elevated_window_focused(),
            Backend::Portal | Backend::X11 | Backend::MintDbus => true,
            Backend::Evdev => self.keyboards_open() > 0,
            Backend::Starting => true,
            Backend::None => false,
        }
    }

    /// FotonVoice Engine has fallen back to reading input devices and cannot: the portal
    pub fn permission_blocked(&self) -> bool {
        self.backend() == Backend::None && self.devices_denied() > 0
    }

    /// Shortcuts are working without FotonVoice Engine seeing any key but its own.
    pub fn is_private(&self) -> bool {
        matches!(self.backend(), Backend::Portal | Backend::MintDbus)
    }

    pub fn set_supported(&self, supported: bool) {
        self.supported.store(supported, Ordering::Relaxed);
    }

    pub fn set_backend(&self, backend: Backend) {
        if let Ok(mut b) = self.backend.lock() {
            *b = Some(backend);
        }
    }

    pub fn set_portal_error(&self, error: String) {
        if let Ok(mut e) = self.portal_error.lock() {
            *e = Some(error);
        }
    }

    pub fn clear_portal_error(&self) {
        if let Ok(mut e) = self.portal_error.lock() {
            *e = None;
        }
    }

    pub fn set_x11_error(&self, error: String) {
        if let Ok(mut e) = self.x11_error.lock() {
            *e = Some(error);
        }
    }

    pub fn set_portal_refused(&self, refused: bool) {
        self.portal_refused.store(refused, Ordering::Relaxed);
    }

    /// Record the outcome of the Windows watchdog's periodic elevation check.
    pub fn set_elevated_window_focused(&self, focused: bool) {
        self.elevated_window_focused.store(focused, Ordering::Relaxed);
    }

    pub fn set_bound_shortcuts(&self, shortcuts: Vec<BoundShortcut>) {
        if let Ok(mut s) = self.bound_shortcuts.lock() {
            *s = shortcuts;
        }
    }

    /// A backend that was running has stopped and cannot recover on its own.
    pub fn set_backend_failed(&self, reason: String) {
        self.set_backend(Backend::None);
        self.set_portal_error(reason);
        self.set_bound_shortcuts(Vec::new());
    }

    pub fn set_keyboards_open(&self, n: usize) {
        self.keyboards_open.store(n, Ordering::Relaxed);
    }

    pub fn record_scan(&self, total: usize, denied: usize) {
        self.devices_total.store(total, Ordering::Relaxed);
        self.devices_denied.store(denied, Ordering::Relaxed);
        self.scanned.store(true, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unsupported_platform_never_reports_a_problem() {
        let h = ListenerHealth::default();
        h.record_scan(0, 0);
        assert!(h.is_active(), "platforms without a listener must not warn");
        assert!(!h.permission_blocked());
    }

    #[test]
    fn the_portal_backend_is_active_without_reading_any_device() {
        let h = ListenerHealth::default();
        h.set_supported(true);
        h.set_backend(Backend::Portal);
        h.record_scan(6, 6); // every device denied - and it does not matter

        assert!(h.is_active());
        assert!(h.is_private());
        assert!(
            !h.permission_blocked(),
            "the portal needs no device permissions, so nothing is blocked"
        );
    }

    #[test]
    fn no_portal_and_no_readable_keyboard_is_a_permission_problem() {
        let h = ListenerHealth::default();
        h.set_supported(true);
        h.set_portal_error("no such interface".to_string());
        h.set_backend(Backend::None);
        h.record_scan(6, 6);
        h.set_keyboards_open(0);

        assert!(!h.is_active());
        assert!(!h.is_private());
        assert!(h.permission_blocked());
    }

    #[test]
    fn the_evdev_fallback_is_active_but_not_private() {
        let h = ListenerHealth::default();
        h.set_supported(true);
        h.set_portal_error("no such interface".to_string());
        h.set_backend(Backend::Evdev);
        h.record_scan(6, 2);
        h.set_keyboards_open(1);

        assert!(h.is_active());
        assert!(!h.is_private(), "reading evdev means seeing every keystroke");
        assert!(!h.permission_blocked());
    }

    #[test]
    fn a_machine_with_no_input_devices_is_not_a_permission_problem() {
        let h = ListenerHealth::default();
        h.set_supported(true);
        h.set_backend(Backend::None);
        h.record_scan(0, 0);
        assert!(!h.permission_blocked());
    }

    #[test]
    fn nothing_is_reported_before_a_backend_is_chosen() {
        let h = ListenerHealth::default();
        h.set_supported(true);
        assert_eq!(h.backend(), Backend::Starting);
        assert!(h.is_active(), "startup must not flash a failure");
        assert!(!h.permission_blocked());
    }

    #[test]
    fn a_refused_portal_is_distinguished_from_a_missing_one() {
        let h = ListenerHealth::default();
        h.set_supported(true);
        h.set_portal_error("An app id is required".to_string());
        h.set_portal_refused(true);
        h.set_backend(Backend::None);

        assert!(h.portal_refused());
        assert!(!h.is_active());

        let missing = ListenerHealth::default();
        missing.set_supported(true);
        missing.set_portal_error("no such interface".to_string());
        missing.set_backend(Backend::None);
        assert!(!missing.portal_refused());
    }

    #[test]
    fn a_lost_portal_session_stops_reporting_as_active() {
        let h = ListenerHealth::default();
        h.set_supported(true);
        h.set_backend(Backend::Portal);
        h.set_bound_shortcuts(vec![BoundShortcut {
            binding_ids: vec!["a".into()],
            requested: Some("LOGO+space".into()),
            trigger_description: "Meta+Space".into(),
            bound: true,
        }]);
        assert!(h.is_active());

        h.set_backend_failed("the portal session ended".to_string());
        assert!(!h.is_active());
        assert!(h.bound_shortcuts().is_empty());
    }

    #[test]
    fn the_x11_backend_is_active_and_serves_every_gesture() {
        let h = ListenerHealth::default();
        h.set_supported(true);
        h.set_portal_error("no such interface".to_string());
        h.set_backend(Backend::X11);

        assert!(h.is_active());
        assert!(!h.is_private(), "raw X11 key events are every keystroke");
        assert!(
            !h.permission_blocked(),
            "X11 raw events need no device permission, so nothing is blocked"
        );
        assert_eq!(h.gestures().len(), 4);
        assert!(h.backend().sees_raw_keys());
    }

    #[test]
    fn a_press_only_backend_advertises_toggle_and_nothing_else() {
        let h = ListenerHealth::default();
        h.set_supported(true);
        h.set_backend(Backend::MintDbus);

        assert_eq!(h.gestures(), &[GestureType::Toggle]);
        assert!(h.is_active());
        assert!(h.is_private(), "the desktop holds the grab, FotonVoice Engine reads nothing");
        assert!(
            !h.backend().sees_raw_keys(),
            "a bare modifier cannot be registered as a desktop accelerator"
        );
    }

    #[test]
    fn a_backend_that_watches_keys_itself_never_hides_a_gesture() {
        for backend in [Backend::X11, Backend::Evdev, Backend::WindowsHook, Backend::Portal] {
            assert_eq!(backend.gestures().len(), 4, "{backend:?} lost a gesture");
        }
    }

    #[test]
    fn every_backend_either_sees_keys_or_is_private() {
        for backend in [
            Backend::Portal,
            Backend::X11,
            Backend::MintDbus,
            Backend::Evdev,
            Backend::WindowsHook,
        ] {
            let h = ListenerHealth::default();
            h.set_supported(true);
            h.set_backend(backend);
            assert_eq!(
                backend.sees_raw_keys(),
                !h.is_private(),
                "{backend:?} claims to both read every key and read none"
            );
        }
    }

    #[test]
    fn the_windows_hook_is_not_private() {
        let h = ListenerHealth::default();
        h.set_supported(true);
        h.set_backend(Backend::WindowsHook);

        assert!(h.is_active());
        assert!(
            !h.is_private(),
            "a WH_KEYBOARD_LL hook is called for every keystroke on the machine"
        );
        assert!(h.backend().sees_raw_keys());
    }

    #[test]
    fn an_elevated_foreground_window_blinds_the_windows_hook() {
        let h = ListenerHealth::default();
        h.set_supported(true);
        h.set_backend(Backend::WindowsHook);
        assert!(h.is_active(), "no elevated window yet, so shortcuts should fire");

        h.set_elevated_window_focused(true);
        assert!(!h.is_active(), "an elevated foreground window must stop reporting active");

        h.set_elevated_window_focused(false);
        assert!(h.is_active(), "clearing the flag once focus moves back must restore active");
    }

    #[test]
    fn an_undecided_or_broken_backend_still_offers_every_gesture() {
        for backend in [Backend::Starting, Backend::None] {
            assert_eq!(backend.gestures().len(), 4, "{backend:?} narrowed the choices");
        }
    }
}
