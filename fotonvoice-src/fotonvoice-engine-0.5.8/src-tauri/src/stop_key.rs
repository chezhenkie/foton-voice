//! When FotonVoice Engine is allowed to hold the TTS stop key.

use std::sync::{atomic::Ordering, Arc};
use std::time::Duration;

use fotonvoice_hotkeys::Backend;
use fotonvoice_routing::{GestureType, HotkeyBinding};

use crate::state::AppState;

/// The synthetic binding id the gesture handler matches on. Defined in
pub use fotonvoice_routing::TTS_STOP_BINDING_ID as STOP_BINDING_ID;

/// How long the grab is kept after playback stops.
const RELEASE_AFTER: Duration = Duration::from_millis(2000);

/// How often the arbiter re-reads the world.
const TICK: Duration = Duration::from_millis(500);

/// How long FotonVoice Engine may hold the stop key.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopKeyGrab {
    /// Nothing is registered: no stop key is configured, or one that would need
    None,
    /// Registered for the whole session. Either FotonVoice Engine is watching the keys
    Always,
    /// Registered only while FotonVoice Engine speaks, because a standing grab would take
    WhileSpeaking,
}

/// Set `FOTONVOICE_STOP_KEY_ARMING=0` to switch the arming off.
fn arming_disabled() -> bool {
    matches!(
        std::env::var("FOTONVOICE_STOP_KEY_ARMING").as_deref(),
        Ok("0") | Ok("false") | Ok("no")
    )
}

/// How long FotonVoice Engine may hold `stop_key` on the backend that is running.
pub fn stop_key_grab(backend: Backend, stop_key: &[String]) -> StopKeyGrab {
    if stop_key.is_empty() {
        return StopKeyGrab::None;
    }
    if !fotonvoice_hotkeys::is_reserved_for_the_desktop(stop_key) {
        return StopKeyGrab::Always;
    }
    if backend.sees_raw_keys() {
        return StopKeyGrab::Always;
    }
    if backend == Backend::Starting && !cfg!(target_os = "linux") {
        return StopKeyGrab::Always;
    }

    if arming_disabled() {
        tracing::debug!(
            "FOTONVOICE_STOP_KEY_ARMING is off: leaving the stop key unregistered rather \
             than taking it for playback"
        );
        return StopKeyGrab::None;
    }

    StopKeyGrab::WhileSpeaking
}

/// Should the stop binding be in the set handed to the listener right now?
pub fn stop_key_wanted(grab: StopKeyGrab, speaking: bool) -> bool {
    match grab {
        StopKeyGrab::None => false,
        StopKeyGrab::Always => true,
        StopKeyGrab::WhileSpeaking => speaking,
    }
}

/// The synthetic binding that carries the stop key into the listener.
pub fn stop_binding(stop_key: Vec<String>) -> HotkeyBinding {
    HotkeyBinding {
        id: STOP_BINDING_ID.to_string(),
        label: "TTS Stop Key".to_string(),
        keys: stop_key,
        gesture: GestureType::Hold,
        target_id: String::new(),
        target_ids: vec![],
        tap_ms: 250,
        hold_threshold_ms: 0,
        disabled: false,
        openai_enabled: Some(false),
        openai_model: None,
        openai_mode: None,
        openai_prompt: None,
        openai_system_prompt: None,
        s1_mini_enabled: None,
    }
}

/// The saved bindings plus the stop key, when the stop key is currently held.
pub async fn listener_bindings(state: &AppState, saved: Vec<HotkeyBinding>) -> Vec<HotkeyBinding> {
    let mut all = saved;
    if state.stop_key_held.load(Ordering::SeqCst) {
        let stop_key = state.config.lock().await.data.tts.stop_key.clone();
        if !stop_key.is_empty() {
            all.push(stop_binding(stop_key));
        }
    }
    all
}

/// Is this a set worth handing the listener?
pub fn is_usable(saved: &[HotkeyBinding]) -> bool {
    !saved.is_empty()
}

/// The saved bindings plus the stop key, or `None` when the saved bindings
pub async fn listener_bindings_from_disk(state: &AppState) -> Option<Vec<HotkeyBinding>> {
    let dir = fotonvoice_routing::config_dir();
    let saved = match fotonvoice_routing::load_bindings(&dir) {
        Ok(saved) => saved,
        Err(e) => {
            tracing::warn!(
                "Could not read the saved shortcuts ({e}); leaving the ones already \
                 registered alone rather than unregistering them"
            );
            return None;
        }
    };
    if !is_usable(&saved) {
        tracing::warn!(
            "The saved shortcuts read back empty, which only happens when the read \
             failed; leaving the ones already registered alone"
        );
        return None;
    }
    Some(listener_bindings(state, saved).await)
}

/// Start the task that takes the stop key and gives it back.
pub fn spawn(state: Arc<AppState>) {
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<bool>();
    if state.speaking_tx.set(tx).is_err() {
        return;
    }

    tokio::spawn(async move {
        let mut held = state.stop_key_held.load(Ordering::SeqCst);
        let mut release_at: Option<tokio::time::Instant> = None;

        loop {
            let speaking = tokio::select! {
                msg = rx.recv() => match msg {
                    Some(speaking) => speaking,
                    None => return,
                },
                _ = tokio::time::sleep(TICK) => state.is_speaking(),
            };

            release_at = if speaking || !held {
                None
            } else {
                Some(release_at.unwrap_or_else(|| tokio::time::Instant::now() + RELEASE_AFTER))
            };
            let wanted_now = speaking
                || release_at.is_some_and(|deadline| tokio::time::Instant::now() < deadline);

            let stop_key = match state.config.try_lock() {
                Ok(cfg) => cfg.data.tts.stop_key.clone(),
                Err(_) => continue,
            };
            let grab = stop_key_grab(state.hotkey_health.backend(), &stop_key);
            let wanted = stop_key_wanted(grab, wanted_now);
            if wanted == held {
                continue;
            }

            if state.is_recording() {
                continue;
            }

            state.stop_key_held.store(wanted, Ordering::SeqCst);
            let Some(bindings) = listener_bindings_from_disk(&state).await else {
                state.stop_key_held.store(held, Ordering::SeqCst);
                continue;
            };
            let reloader = state.hotkey_reloader.lock().await;
            let Some(reloader) = reloader.as_ref() else {
                state.stop_key_held.store(held, Ordering::SeqCst);
                continue;
            };
            let count = bindings.len();
            match reloader.send(bindings) {
                Ok(()) => {
                    held = wanted;
                    tracing::info!(
                        "TTS stop key {} ({grab:?}); listener now has {count} binding(s)",
                        if wanted { "registered" } else { "released" }
                    );
                }
                Err(e) => {
                    state.stop_key_held.store(held, Ordering::SeqCst);
                    tracing::warn!("Could not reload bindings for the stop key: {e}");
                }
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn keys(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| s.to_string()).collect()
    }

    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn escape_is_held_only_while_speaking_where_the_desktop_owns_the_grab() {
        let _lock = ENV_LOCK.lock().unwrap();
        let grab = stop_key_grab(Backend::Portal, &keys(&["KEY_ESC"]));
        assert_eq!(grab, StopKeyGrab::WhileSpeaking);
        assert!(stop_key_wanted(grab, true));
        assert!(!stop_key_wanted(grab, false));
    }

    #[test]
    fn escape_is_held_throughout_where_fotonvoice_watches_the_keyboard() {
        for backend in [Backend::X11, Backend::Evdev, Backend::WindowsHook] {
            let grab = stop_key_grab(backend, &keys(&["KEY_ESC"]));
            assert_eq!(grab, StopKeyGrab::Always, "{backend:?}");
            assert!(stop_key_wanted(grab, false), "{backend:?}");
        }
    }

    #[test]
    fn a_stop_key_with_a_modifier_is_an_ordinary_standing_shortcut() {
        let grab = stop_key_grab(Backend::Portal, &keys(&["KEY_LEFTCTRL", "KEY_ESC"]));
        assert_eq!(grab, StopKeyGrab::Always);
        assert!(stop_key_wanted(grab, false));
    }

    #[test]
    fn an_undecided_backend_does_not_get_the_benefit_of_the_doubt() {
        for backend in [Backend::None, Backend::MintDbus] {
            assert_eq!(
                stop_key_grab(backend, &keys(&["KEY_ESC"])),
                StopKeyGrab::WhileSpeaking,
                "{backend:?}"
            );
        }

        let starting = stop_key_grab(Backend::Starting, &keys(&["KEY_ESC"]));
        if cfg!(target_os = "linux") {
            assert_eq!(starting, StopKeyGrab::WhileSpeaking);
        } else {
            assert_eq!(starting, StopKeyGrab::Always);
        }
    }

    #[test]
    fn no_stop_key_means_nothing_is_ever_registered() {
        let grab = stop_key_grab(Backend::Portal, &[]);
        assert_eq!(grab, StopKeyGrab::None);
        assert!(!stop_key_wanted(grab, true));
    }

    #[test]
    fn the_arming_switch_stands_the_whole_mechanism_down() {
        let _lock = ENV_LOCK.lock().unwrap();
        let escape = keys(&["KEY_ESC"]);
        let ctrl_escape = keys(&["KEY_LEFTCTRL", "KEY_ESC"]);

        unsafe { std::env::set_var("FOTONVOICE_STOP_KEY_ARMING", "0") };
        assert_eq!(stop_key_grab(Backend::Portal, &escape), StopKeyGrab::None);
        assert_eq!(
            stop_key_grab(Backend::Portal, &ctrl_escape),
            StopKeyGrab::Always
        );
        assert_eq!(stop_key_grab(Backend::X11, &escape), StopKeyGrab::Always);

        unsafe { std::env::remove_var("FOTONVOICE_STOP_KEY_ARMING") };
        assert_eq!(
            stop_key_grab(Backend::Portal, &escape),
            StopKeyGrab::WhileSpeaking
        );
    }

    #[test]
    fn an_empty_binding_set_is_never_handed_to_the_listener() {
        assert!(!is_usable(&[]));
        assert!(is_usable(&[stop_binding(keys(&["KEY_ESC"]))]));
    }

    #[test]
    fn the_stop_binding_records_nothing_and_targets_nothing() {
        let b = stop_binding(keys(&["KEY_ESC"]));
        assert_eq!(b.id, STOP_BINDING_ID);
        assert!(b.target_id.is_empty());
        assert!(b.target_ids.is_empty());
        assert_eq!(b.hold_threshold_ms, 0);
        assert!(!b.disabled);
    }
}
