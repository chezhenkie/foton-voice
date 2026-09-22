//! Global shortcuts from X11 raw key events (XInput2).

use std::{collections::HashSet, sync::Arc};

use x11rb::{
    connection::Connection,
    protocol::{
        xinput::{self, ConnectionExt as _},
        Event,
    },
    rust_connection::RustConnection,
};

use fotonvoice_routing::HotkeyBinding;

use crate::{
    linux::{is_synthetic_device_name, run_coordinator, ReaderEvent},
    Backend, GestureSender, ListenerHealth,
};

/// `XIAllMasterDevices`. Raw events are reported against the master device the
const XI_ALL_MASTER_DEVICES: xinput::DeviceId = 1;

/// `XIAllDevices`. Used to query the device list - the slave keyboards that
const XI_ALL_DEVICES: xinput::DeviceId = 0;

/// X11 keycodes are evdev codes offset by 8, fixed by the X11 protocol.
const X11_KEYCODE_OFFSET: u8 = 8;

/// XInput 2.1 is the floor, and not for a feature - for two behaviours that
const MIN_XI_MINOR: u16 = 1;
const MIN_XI_MAJOR: u16 = 2;

/// Does a negotiated XInput version deliver raw events this backend can trust?
fn supports_reliable_raw_events(major: u16, minor: u16) -> bool {
    (major, minor) >= (MIN_XI_MAJOR, MIN_XI_MINOR)
}

/// Why the X11 backend could not be used. Never fatal: the caller falls through
#[derive(Debug, Clone)]
pub enum X11Error {
    /// No X server to talk to - a Wayland session, or no session at all.
    NoDisplay,
    /// There is a `DISPLAY`, but the connection failed.
    Connect(String),
    /// The X server is too old for XInput2, or has the extension disabled.
    NoXInput(String),
    /// The server refused the event selection.
    Select(String),
    /// Turned off by `FOTONVOICE_DISABLE_X11_HOTKEYS`.
    Disabled,
}

impl std::fmt::Display for X11Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoDisplay => write!(f, "this is not an X11 session"),
            Self::Connect(e) => write!(f, "cannot connect to the X server: {e}"),
            Self::NoXInput(e) => write!(f, "this X server has no usable XInput2: {e}"),
            Self::Select(e) => write!(f, "the X server refused raw key events: {e}"),
            Self::Disabled => write!(f, "disabled by FOTONVOICE_DISABLE_X11_HOTKEYS"),
        }
    }
}

/// Connect, claim raw key events, and start dispatching them into `tx`.
pub fn start(
    bindings: Vec<HotkeyBinding>,
    tx: GestureSender,
    rx_reload: crate::ReloaderReceiver,
    health: Arc<ListenerHealth>,
) -> Result<(), X11Error> {
    if std::env::var_os("FOTONVOICE_DISABLE_X11_HOTKEYS").is_some() {
        return Err(X11Error::Disabled);
    }
    if std::env::var_os("DISPLAY").is_none() {
        return Err(X11Error::NoDisplay);
    }

    let (conn, _screen) = RustConnection::connect(None)
        .map_err(|e| X11Error::Connect(format!("{e}")))?;

    let version = conn
        .xinput_xi_query_version(MIN_XI_MAJOR, MIN_XI_MINOR)
        .map_err(|e| X11Error::NoXInput(format!("{e}")))?
        .reply()
        .map_err(|e| X11Error::NoXInput(format!("{e}")))?;
    if !supports_reliable_raw_events(version.major_version, version.minor_version) {
        return Err(X11Error::NoXInput(format!(
            "the server offers XInput {}.{}, and reliable raw key events need \
             {MIN_XI_MAJOR}.{MIN_XI_MINOR}",
            version.major_version, version.minor_version
        )));
    }

    let raw_mask = xinput::EventMask {
        deviceid: XI_ALL_MASTER_DEVICES,
        mask: vec![xinput::XIEventMask::RAW_KEY_PRESS | xinput::XIEventMask::RAW_KEY_RELEASE],
    };
    let hierarchy_mask = xinput::EventMask {
        deviceid: XI_ALL_DEVICES,
        mask: vec![xinput::XIEventMask::HIERARCHY],
    };

    let roots: Vec<u32> = conn.setup().roots.iter().map(|s| s.root).collect();
    for root in &roots {
        conn.xinput_xi_select_events(*root, std::slice::from_ref(&raw_mask))
            .map_err(|e| X11Error::Select(format!("{e}")))?
            .check()
            .map_err(|e| X11Error::Select(format!("{e}")))?;

        let hotplug = conn
            .xinput_xi_select_events(*root, std::slice::from_ref(&hierarchy_mask))
            .map_err(|e| format!("{e}"))
            .and_then(|cookie| cookie.check().map_err(|e| format!("{e}")));
        if let Err(e) = hotplug {
            tracing::debug!("X11 device-hotplug notifications unavailable: {e}");
        }
    }
    conn.flush().map_err(|e| X11Error::Select(format!("{e}")))?;

    let synthetic = synthetic_source_ids(&conn);

    let rt_handle = tokio::runtime::Handle::try_current().ok();
    let (event_tx, event_rx) = crossbeam_channel::unbounded::<ReaderEvent>();

    let rt = rt_handle.clone();
    std::thread::Builder::new()
        .name("fotonvoice-hotkey-coord".into())
        .spawn(move || {
            let _guard = rt.as_ref().map(|h| h.enter());
            run_coordinator(bindings, tx, rx_reload, event_rx);
        })
        .map_err(|e| X11Error::Select(format!("cannot start the hotkey coordinator: {e}")))?;

    let reader_health = health.clone();
    std::thread::Builder::new()
        .name("fotonvoice-x11-keys".into())
        .spawn(move || run_reader(conn, synthetic, event_tx, reader_health))
        .map_err(|e| X11Error::Select(format!("cannot start the X11 reader: {e}")))?;

    health.set_backend(Backend::X11);
    tracing::info!(
        "Global shortcuts are coming from X11 raw key events; no input-device access was \
         needed and none was requested"
    );
    Ok(())
}

/// Source ids that FotonVoice Engine must ignore, so it never reads back the keystrokes
fn synthetic_source_ids(conn: &RustConnection) -> HashSet<xinput::DeviceId> {
    let Ok(cookie) = conn.xinput_xi_query_device(XI_ALL_DEVICES) else {
        return HashSet::new();
    };
    let Ok(reply) = cookie.reply() else {
        return HashSet::new();
    };
    reply
        .infos
        .iter()
        .filter(|info| is_synthetic_device_name(&String::from_utf8_lossy(&info.name)))
        .map(|info| info.deviceid)
        .collect()
}

fn run_reader(
    conn: RustConnection,
    mut synthetic: HashSet<xinput::DeviceId>,
    event_tx: crossbeam_channel::Sender<ReaderEvent>,
    health: Arc<ListenerHealth>,
) {
    loop {
        let event = match conn.wait_for_event() {
            Ok(e) => e,
            Err(e) => {
                tracing::warn!("X11 hotkey connection lost: {e}");
                let _ = event_tx.send(ReaderEvent::SourceLost);
                health.set_backend_failed(format!("the X11 connection ended: {e}"));
                return;
            }
        };

        let (raw, down) = match event {
            Event::XinputRawKeyPress(ev) => (ev, true),
            Event::XinputRawKeyRelease(ev) => (ev, false),
            Event::XinputHierarchy(_) => {
                synthetic = synthetic_source_ids(&conn);
                continue;
            }
            _ => continue,
        };

        if down && raw.flags.contains(xinput::KeyEventFlags::KEY_REPEAT) {
            continue;
        }
        if synthetic.contains(&raw.sourceid) {
            continue;
        }
        let Some(name) = key_name(raw.detail) else {
            continue;
        };
        if event_tx.send(ReaderEvent::Key { name, down }).is_err() {
            return;
        }
    }
}

/// X11 keycode -> the evdev key name the rest of FotonVoice Engine speaks.
fn key_name(keycode: u32) -> Option<String> {
    let code = u16::try_from(keycode).ok()?;
    let code = code.checked_sub(u16::from(X11_KEYCODE_OFFSET))?;
    let name = format!("{:?}", evdev::Key::new(code));
    if name.starts_with("KEY_") || name.starts_with("BTN_") {
        Some(name)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keycodes_translate_to_the_same_names_evdev_reports() {
        assert_eq!(key_name(38).as_deref(), Some("KEY_A"));
        assert_eq!(key_name(65).as_deref(), Some("KEY_SPACE"));
        assert_eq!(key_name(133).as_deref(), Some("KEY_LEFTMETA"));
        assert_eq!(key_name(37).as_deref(), Some("KEY_LEFTCTRL"));
    }

    #[test]
    fn keycodes_below_the_offset_are_not_keys() {
        for keycode in 0..8 {
            assert_eq!(key_name(keycode), None, "keycode {keycode} is not a key");
        }
    }

    #[test]
    fn unmapped_keycodes_are_dropped_rather_than_named() {
        assert_eq!(key_name(60000), None);
    }

    /// The selection sequence `start` performs, against whatever X server
    #[test]
    fn the_real_selection_sequence_is_accepted_by_an_x_server() {
        let Ok(display) = std::env::var("DISPLAY") else {
            eprintln!("skipped: no DISPLAY (run under xvfb-run to exercise this)");
            return;
        };

        let (conn, _screen) = match RustConnection::connect(None) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("skipped: DISPLAY={display} is not connectable: {e}");
                return;
            }
        };

        let version = conn
            .xinput_xi_query_version(MIN_XI_MAJOR, MIN_XI_MINOR)
            .expect("XIQueryVersion should be sendable")
            .reply()
            .expect("this server should serve XInput2");
        assert!(
            supports_reliable_raw_events(version.major_version, version.minor_version),
            "server offers XInput {}.{}",
            version.major_version,
            version.minor_version
        );

        let raw_mask = xinput::EventMask {
            deviceid: XI_ALL_MASTER_DEVICES,
            mask: vec![xinput::XIEventMask::RAW_KEY_PRESS | xinput::XIEventMask::RAW_KEY_RELEASE],
        };
        let hierarchy_mask = xinput::EventMask {
            deviceid: XI_ALL_DEVICES,
            mask: vec![xinput::XIEventMask::HIERARCHY],
        };
        let root = conn.setup().roots[0].root;

        conn.xinput_xi_select_events(root, std::slice::from_ref(&raw_mask))
            .expect("XISelectEvents should be sendable")
            .check()
            .expect("the server must accept raw key events on XIAllMasterDevices");

        conn.xinput_xi_select_events(root, std::slice::from_ref(&hierarchy_mask))
            .expect("XISelectEvents should be sendable")
            .check()
            .expect("the server must accept hierarchy events on XIAllDevices");

        let merged = xinput::EventMask {
            deviceid: XI_ALL_MASTER_DEVICES,
            mask: vec![
                xinput::XIEventMask::RAW_KEY_PRESS
                    | xinput::XIEventMask::RAW_KEY_RELEASE
                    | xinput::XIEventMask::HIERARCHY,
            ],
        };
        let merged_result = conn
            .xinput_xi_select_events(root, std::slice::from_ref(&merged))
            .expect("XISelectEvents should be sendable")
            .check();
        assert!(
            merged_result.is_err(),
            "selecting HIERARCHY on XIAllMasterDevices must be refused; if a \
             server ever accepts it, this backend's split is still correct but \
             this test's premise has changed"
        );

        let devices = conn
            .xinput_xi_query_device(XI_ALL_DEVICES)
            .expect("XIQueryDevice should be sendable")
            .reply()
            .expect("the server must answer a device query");
        assert!(!devices.infos.is_empty(), "a server always has core devices");
    }

    /// `start` itself, end to end, against a real server.
    #[tokio::test]
    async fn start_succeeds_and_claims_the_backend_on_a_real_server() {
        if std::env::var("DISPLAY").is_err() {
            eprintln!("skipped: no DISPLAY (run under xvfb-run to exercise this)");
            return;
        }

        let (tx, _rx) = crate::channel();
        let (_reload_tx, reload_rx) = crossbeam_channel::unbounded();
        let health = Arc::new(ListenerHealth::default());

        let result = start(Vec::new(), tx, reload_rx, health.clone());

        assert!(result.is_ok(), "start must succeed on a real X server: {result:?}");
        assert_eq!(health.backend(), Backend::X11);
        assert!(health.is_active());
    }

    #[test]
    fn hierarchy_events_are_never_selected_alongside_raw_events() {
        assert_ne!(
            XI_ALL_DEVICES, XI_ALL_MASTER_DEVICES,
            "the two selections must target different devices"
        );

        let raw = xinput::XIEventMask::RAW_KEY_PRESS | xinput::XIEventMask::RAW_KEY_RELEASE;
        assert_eq!(
            u32::from(raw) & u32::from(xinput::XIEventMask::HIERARCHY),
            0,
            "the raw-event mask must not carry HIERARCHY"
        );
    }

    #[test]
    fn xinput_2_0_is_refused_because_its_raw_events_cannot_be_trusted() {
        assert!(!supports_reliable_raw_events(2, 0));
        assert!(!supports_reliable_raw_events(1, 9));

        assert!(supports_reliable_raw_events(2, 1));
        assert!(supports_reliable_raw_events(2, 4));
        assert!(supports_reliable_raw_events(3, 0));
    }

    #[test]
    fn injected_keystrokes_come_from_devices_this_backend_ignores() {
        for name in ["Virtual core XTEST keyboard", "FotonVoice Engine Virtual Keyboard"] {
            assert!(is_synthetic_device_name(name), "{name} must be ignored");
        }
        assert!(!is_synthetic_device_name("AT Translated Set 2 keyboard"));
    }
}
