//! Linux hotkey backends.

use std::{collections::HashSet, sync::Arc, time::Duration};

use tracing::warn;
use fotonvoice_routing::HotkeyBinding;

use crate::{
    gestures::GestureEngine,
    keys::KeyMatcher,
    Backend, GestureSender, ListenerHealth,
};

/// How often the supervisor rescans `/dev/input` while it has no keyboard.
const RESCAN_BLOCKED_INTERVAL: Duration = Duration::from_secs(2);
/// Rescan interval once at least one keyboard is being read - this only picks
const RESCAN_HEALTHY_INTERVAL: Duration = Duration::from_secs(10);

/// What a key-reading backend's threads report to the coordinator.
pub(crate) enum ReaderEvent {
    Key { name: String, down: bool },
    /// A keyboard went away. Anything held on it is never coming back up.
    SourceLost,
}

pub fn start(
    bindings: Vec<HotkeyBinding>,
    tx: GestureSender,
    device_path: Option<String>,
    rx_reload: crate::ReloaderReceiver,
    health: Arc<ListenerHealth>,
) {
    health.set_supported(true);

    let rt_handle = tokio::runtime::Handle::try_current().ok();
    let Some(rt) = rt_handle else {
        health.set_backend_failed("no async runtime available for the hotkey listener".into());
        tracing::error!("Hotkey listener started outside a tokio runtime");
        return;
    };

    rt.spawn(async move {
        match portal_start(bindings.clone(), tx.clone(), rx_reload.clone(), health.clone()).await {
            Ok(()) => {}
            Err(e) => {
                tracing::info!("Desktop portal shortcuts unavailable ({e})");
                health.set_portal_error(e.to_string());
                health.set_portal_refused(matches!(e, crate::portal::PortalError::Rejected(_)));

                match crate::x11::start(
                    bindings.clone(),
                    tx.clone(),
                    rx_reload.clone(),
                    health.clone(),
                ) {
                    Ok(()) => {}
                    Err(e) => {
                        tracing::info!(
                            "X11 raw key events unavailable ({e}); falling back to reading \
                             input devices, which needs access this app will not request \
                             for you"
                        );
                        health.set_x11_error(e.to_string());
                        start_evdev(bindings, tx, device_path, rx_reload, health);
                    }
                }
            }
        }
    });
}

async fn portal_start(
    bindings: Vec<HotkeyBinding>,
    tx: GestureSender,
    rx_reload: crate::ReloaderReceiver,
    health: Arc<ListenerHealth>,
) -> Result<(), crate::portal::PortalError> {
    if std::env::var_os("FOTONVOICE_DISABLE_PORTAL_HOTKEYS").is_some() {
        return Err(crate::portal::PortalError::Unavailable(
            "disabled by FOTONVOICE_DISABLE_PORTAL_HOTKEYS".to_string(),
        ));
    }
    crate::portal::start(bindings, tx, rx_reload, health).await
}

pub async fn retry_portal(
    bindings: Vec<HotkeyBinding>,
    tx: GestureSender,
    rx_reload: crate::ReloaderReceiver,
    health: Arc<ListenerHealth>,
) -> Result<(), String> {
    health.clear_portal_error();
    health.set_portal_refused(false);
    match portal_start(bindings, tx, rx_reload, health.clone()).await {
        Ok(()) => {
            tracing::info!("Global shortcuts registered through the desktop portal on retry");
            Ok(())
        }
        Err(e) => {
            let msg = e.to_string();
            tracing::warn!("Retry portal shortcuts failed: {msg}");
            health.set_portal_error(msg.clone());
            health.set_portal_refused(matches!(e, crate::portal::PortalError::Rejected(_)));
            Err(msg)
        }
    }
}


fn start_evdev(
    bindings: Vec<HotkeyBinding>,
    tx: GestureSender,
    device_path: Option<String>,
    rx_reload: crate::ReloaderReceiver,
    health: Arc<ListenerHealth>,
) {
    let rt_handle = tokio::runtime::Handle::try_current().ok();
    let (event_tx, event_rx) = crossbeam_channel::unbounded::<ReaderEvent>();

    let rt = rt_handle.clone();
    if let Err(e) = std::thread::Builder::new()
        .name("fotonvoice-hotkey-coord".into())
        .spawn(move || {
            let _guard = rt.as_ref().map(|h| h.enter());
            run_coordinator(bindings, tx, rx_reload, event_rx);
        })
    {
        health.set_backend_failed(format!("cannot start the hotkey coordinator: {e}"));
        return;
    }

    let supervisor_health = health.clone();
    if let Err(e) = std::thread::Builder::new()
        .name("fotonvoice-evdev-supervisor".into())
        .spawn(move || run_supervisor(device_path, event_tx, supervisor_health))
    {
        health.set_backend_failed(format!("cannot start the evdev supervisor: {e}"));
    }
}

/// Owns the reader threads for the lifetime of the process.
fn run_supervisor(
    device_path: Option<String>,
    event_tx: crossbeam_channel::Sender<ReaderEvent>,
    health: Arc<ListenerHealth>,
) {
    let mut running: HashSet<String> = HashSet::new();
    let (exit_tx, exit_rx) = crossbeam_channel::unbounded::<String>();
    let mut warned_blocked = false;

    loop {
        while let Ok(path) = exit_rx.try_recv() {
            running.remove(&path);
        }

        let scan = scan_input_devices(device_path.as_deref());
        health.record_scan(scan.total, scan.denied);

        for path in scan.keyboards {
            if !running.insert(path.clone()) {
                continue;
            }
            let tx_clone = event_tx.clone();
            let exit_clone = exit_tx.clone();
            let reader_path = path.clone();
            match std::thread::Builder::new()
                .name("fotonvoice-evdev".into())
                .spawn(move || {
                    run_reader(reader_path.clone(), tx_clone);
                    let _ = exit_clone.send(reader_path);
                }) {
                Ok(_) => tracing::info!("Reading hotkeys from {path}"),
                Err(e) => {
                    warn!("Failed to spawn evdev reader for {path}: {e}");
                    running.remove(&path);
                }
            }
        }

        health.set_keyboards_open(running.len());
        health.set_backend(if running.is_empty() {
            Backend::None
        } else {
            Backend::Evdev
        });

        if running.is_empty() {
            if !warned_blocked {
                warned_blocked = true;
                if scan.denied > 0 {
                    warn!(
                        "This desktop has no global-shortcuts portal, and {} of {} input \
                         devices are permission-denied, so FotonVoice Engine has no way to receive \
                         hotkeys.",
                        scan.denied, scan.total
                    );
                } else {
                    warn!("No suitable keyboard evdev device found; hotkeys disabled");
                }
            }
        } else if warned_blocked {
            warned_blocked = false;
            tracing::info!("Keyboard access recovered; global hotkeys are active again");
        }

        std::thread::sleep(if running.is_empty() {
            RESCAN_BLOCKED_INTERVAL
        } else {
            RESCAN_HEALTHY_INTERVAL
        });
    }
}

fn run_reader(device_path: String, event_tx: crossbeam_channel::Sender<ReaderEvent>) {
    let mut device = match open_device(&Some(device_path.clone())) {
        Some(d) => d,
        None => return,
    };

    loop {
        match device.fetch_events() {
            Ok(events) => {
                for ev in events {
                    if ev.event_type() != evdev::EventType::KEY {
                        continue;
                    }
                    let mut key_name = match ev.kind() {
                        evdev::InputEventKind::Key(key) => format!("{:?}", key),
                        _ => format!("{:?}", ev.code()),
                    };
                    if key_name.starts_with("Key(") && key_name.ends_with(')') {
                        key_name.pop();
                        key_name.drain(..4);
                    }
                    let down = match ev.value() {
                        1 => true,
                        0 => false,
                        _ => continue,
                    };
                    if event_tx
                        .send(ReaderEvent::Key {
                            name: key_name,
                            down,
                        })
                        .is_err()
                    {
                        return;
                    }
                }
            }
            Err(e) => {
                if !std::path::Path::new(&device_path).exists() {
                    tracing::info!("evdev device {device_path} disappeared; releasing it");
                    let _ = event_tx.send(ReaderEvent::SourceLost);
                    return;
                }
                warn!("evdev read error on {device_path}: {e}; retrying in 1s");
                std::thread::sleep(Duration::from_secs(1));
            }
        }
    }
}

pub(crate) fn run_coordinator(
    bindings: Vec<HotkeyBinding>,
    tx: GestureSender,
    rx_reload: crate::ReloaderReceiver,
    event_rx: crossbeam_channel::Receiver<ReaderEvent>,
) {
    let mut engine = GestureEngine::new(bindings.clone());
    let mut matcher = KeyMatcher::new(bindings);

    loop {
        crossbeam_channel::select! {
            recv(rx_reload) -> new_bindings => {
                match new_bindings {
                    Ok(new_bindings) => {
                        tracing::info!("evdev hotkeys: reloading {} bindings", new_bindings.len());
                        engine.reset(&tx);
                        engine.reload(new_bindings.clone());
                        matcher.reload(new_bindings);
                    }
                    Err(_) => break,
                }
            }
            recv(event_rx) -> event => {
                match event {
                    Ok(ReaderEvent::Key { name, down }) => {
                        for (id, transition) in matcher.on_key(&name, down) {
                            engine.apply(&id, transition, &tx);
                        }
                    }
                    Ok(ReaderEvent::SourceLost) => {
                        for (id, transition) in matcher.clear() {
                            engine.apply(&id, transition, &tx);
                        }
                    }
                    Err(e) => {
                        tracing::error!(
                            "Hotkey coordinator: event channel closed unexpectedly \
                             ({e}); all hotkeys are disabled until the app restarts."
                        );
                        engine.reset(&tx);
                        break;
                    }
                }
            }
        }
    }
}


fn open_device(preferred: &Option<String>) -> Option<evdev::Device> {
    if let Some(path) = preferred {
        if let Ok(d) = evdev::Device::open(path) {
            return Some(d);
        }
        warn!("Saved evdev device {path} not accessible");
    }
    None
}

/// Result of one `/dev/input` sweep.
pub(crate) struct InputScan {
    /// Devices that are readable and look like a keyboard.
    pub keyboards: Vec<String>,
    /// Devices that exist but cannot be opened because of permissions.
    pub denied: usize,
    /// Every `event*` node found, readable or not.
    pub total: usize,
}

/// True for an evdev device FotonVoice Engine is willing to read hotkeys from.
pub(crate) fn is_eligible_keyboard(name: &str, has_key_a: bool) -> bool {
    if is_synthetic_device_name(name) {
        return false;
    }
    has_key_a
}

/// True for a device that reports keystrokes some program injected rather than
pub(crate) fn is_synthetic_device_name(name: &str) -> bool {
    let name = name.to_ascii_lowercase();
    name.contains("virtual")
        || name.contains("uinput")
        || name.contains("xtest")
        || name.contains("passthrough")
}

/// Sweep `/dev/input` for keyboards, counting the devices that exist but are
pub(crate) fn scan_input_devices(preferred: Option<&str>) -> InputScan {
    let mut scan = InputScan {
        keyboards: Vec::new(),
        denied: 0,
        total: 0,
    };

    if let Some(path) = preferred {
        scan.total = 1;
        match std::fs::File::open(path) {
            Ok(_) => match evdev::Device::open(path) {
                Ok(_) => scan.keyboards.push(path.to_string()),
                Err(e) => warn!("Configured evdev device {path} is not usable: {e}"),
            },
            Err(e) if e.kind() == std::io::ErrorKind::PermissionDenied => scan.denied = 1,
            Err(e) => warn!("Configured evdev device {path} is not usable: {e}"),
        }
        return scan;
    }

    let entries = match std::fs::read_dir("/dev/input") {
        Ok(entries) => entries,
        Err(e) => {
            warn!("Cannot list /dev/input: {e}");
            return scan;
        }
    };

    for entry in entries.flatten() {
        if !entry
            .file_name()
            .to_str()
            .map(|n| n.starts_with("event"))
            .unwrap_or(false)
        {
            continue;
        }
        scan.total += 1;
        let path = entry.path();

        if let Err(e) = std::fs::File::open(&path) {
            if e.kind() == std::io::ErrorKind::PermissionDenied {
                scan.denied += 1;
            }
            continue;
        }

        let Ok(dev) = evdev::Device::open(&path) else {
            continue;
        };
        let name = dev.name().unwrap_or("").to_string();
        let has_key_a = dev
            .supported_keys()
            .map(|keys| keys.contains(evdev::Key::KEY_A))
            .unwrap_or(false);
        if is_eligible_keyboard(&name, has_key_a) {
            tracing::debug!("Eligible keyboard: {name} at {path:?}");
            scan.keyboards.push(path.to_string_lossy().to_string());
        }
    }

    scan
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gestures::GestureKind;
    use fotonvoice_routing::GestureType;

    #[test]
    fn synthetic_devices_are_never_used_as_hotkey_sources() {
        for name in [
            "FotonVoice Engine Virtual Keyboard",
            "py-evdev-uinput",
            "XTEST pointer",
            "Some passthrough device",
        ] {
            assert!(!is_eligible_keyboard(name, true), "{name} must be skipped");
        }
    }

    #[test]
    fn real_keyboards_are_eligible_and_non_keyboards_are_not() {
        assert!(is_eligible_keyboard("AT Translated Set 2 keyboard", true));
        assert!(!is_eligible_keyboard("Logitech USB Mouse", false));
    }

    #[test]
    fn scanning_a_missing_configured_device_reports_it_as_unusable() {
        let scan = scan_input_devices(Some("/dev/input/does-not-exist-fotonvoice-test"));
        assert_eq!(scan.total, 1);
        assert_eq!(scan.denied, 0);
        assert!(scan.keyboards.is_empty());
    }

    #[test]
    fn scanning_reports_totals_consistently() {
        let scan = scan_input_devices(None);
        assert!(
            scan.denied <= scan.total,
            "denied ({}) cannot exceed total ({})",
            scan.denied,
            scan.total
        );
        assert!(scan.keyboards.len() <= scan.total);
    }

    fn binding(id: &str, gesture: GestureType, keys: &[&str]) -> HotkeyBinding {
        HotkeyBinding {
            id: id.to_string(),
            label: id.to_string(),
            keys: keys.iter().map(|k| k.to_string()).collect(),
            gesture,
            target_id: "t".to_string(),
            target_ids: vec!["t".to_string()],
            tap_ms: 300,
            hold_threshold_ms: 100,
            disabled: false,
            openai_enabled: Some(false),
            openai_model: None,
            openai_mode: None,
            openai_prompt: None,
            openai_system_prompt: None,
            s1_mini_enabled: None,
        }
    }

    /// Drive a raw key event through the same path the reader thread uses.
    fn key(
        matcher: &mut KeyMatcher,
        engine: &mut GestureEngine,
        tx: &GestureSender,
        name: &str,
        down: bool,
    ) {
        for (id, transition) in matcher.on_key(name, down) {
            engine.apply(&id, transition, tx);
        }
    }

    #[tokio::test]
    async fn a_real_double_tap_on_a_bare_modifier_starts_recording() {
        let (tx, mut rx) = crate::channel();
        let bindings = vec![binding("dt", GestureType::DoubleTap, &["KEY_LEFTMETA"])];
        let mut engine = GestureEngine::new(bindings.clone());
        let mut matcher = KeyMatcher::new(bindings);

        key(&mut matcher, &mut engine, &tx, "KEY_LEFTMETA", true);
        key(&mut matcher, &mut engine, &tx, "KEY_LEFTMETA", false);
        tokio::time::sleep(Duration::from_millis(60)).await;
        key(&mut matcher, &mut engine, &tx, "KEY_LEFTMETA", true);

        let event = rx.try_recv().expect("double-tap must start recording");
        assert_eq!(event.binding_id, "dt");
        assert_eq!(event.kind, GestureKind::Start);
    }

    #[tokio::test]
    async fn auto_repeat_between_taps_does_not_break_the_gesture() {
        let (tx, mut rx) = crate::channel();
        let bindings = vec![binding("dth", GestureType::DoubleTapHold, &["KEY_LEFTMETA"])];
        let mut engine = GestureEngine::new(bindings.clone());
        let mut matcher = KeyMatcher::new(bindings);

        key(&mut matcher, &mut engine, &tx, "KEY_LEFTMETA", true);
        key(&mut matcher, &mut engine, &tx, "KEY_LEFTMETA", false);
        tokio::time::sleep(Duration::from_millis(60)).await;
        key(&mut matcher, &mut engine, &tx, "KEY_LEFTMETA", true);
        tokio::time::sleep(Duration::from_millis(150)).await;

        assert_eq!(rx.try_recv().unwrap().kind, GestureKind::Start);
        key(&mut matcher, &mut engine, &tx, "KEY_LEFTMETA", false);
        assert_eq!(rx.try_recv().unwrap().kind, GestureKind::Stop);
    }

    #[tokio::test]
    async fn losing_the_keyboard_mid_hold_stops_the_recording() {
        let (tx, mut rx) = crate::channel();
        let bindings = vec![binding("h", GestureType::Hold, &["KEY_LEFTALT"])];
        let mut engine = GestureEngine::new(bindings.clone());
        let mut matcher = KeyMatcher::new(bindings);

        key(&mut matcher, &mut engine, &tx, "KEY_LEFTALT", true);
        tokio::time::sleep(Duration::from_millis(150)).await;
        assert_eq!(rx.try_recv().unwrap().kind, GestureKind::Start);

        for (id, transition) in matcher.clear() {
            engine.apply(&id, transition, &tx);
        }
        assert_eq!(rx.try_recv().unwrap().kind, GestureKind::Stop);
    }

    #[tokio::test]
    async fn a_hold_combo_survives_releasing_its_keys_in_either_order() {
        let (tx, mut rx) = crate::channel();
        let bindings = vec![binding("h", GestureType::Hold, &["KEY_LEFTMETA", "KEY_SPACE"])];
        let mut engine = GestureEngine::new(bindings.clone());
        let mut matcher = KeyMatcher::new(bindings);

        key(&mut matcher, &mut engine, &tx, "KEY_LEFTMETA", true);
        key(&mut matcher, &mut engine, &tx, "KEY_SPACE", true);
        tokio::time::sleep(Duration::from_millis(150)).await;
        assert_eq!(rx.try_recv().unwrap().kind, GestureKind::Start);

        key(&mut matcher, &mut engine, &tx, "KEY_SPACE", false);
        assert!(
            rx.try_recv().is_err(),
            "must not stop while Super is still down"
        );
        key(&mut matcher, &mut engine, &tx, "KEY_LEFTMETA", false);
        assert_eq!(rx.try_recv().unwrap().kind, GestureKind::Stop);
    }
}
