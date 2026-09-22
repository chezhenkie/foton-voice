//! Gesture recognition, independent of where the key events came from.

use std::{
    collections::{HashMap, HashSet},
    sync::{atomic::AtomicBool, Arc},
    time::Duration,
};
use tokio::time::Instant;

use tokio_util::sync::CancellationToken;
use fotonvoice_routing::{GestureType, HotkeyBinding};

use crate::GestureSender;

/// Event emitted when a gesture is fully recognized.
#[derive(Debug, Clone)]
pub struct GestureEvent {
    pub binding_id: String,
    pub binding_label: String,
    pub target_id: String,
    pub kind: GestureKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GestureKind {
    /// Hold started
    Start,
    /// Hold released / toggle activated
    Stop,
}

/// A `hold` or `double_tap_hold` that is never released - because the backend
const STUCK_HOLD_MAX: Duration = Duration::from_secs(120);

/// Shortest gap between releasing the first tap and pressing the second that
const MIN_TAP_GAP: Duration = Duration::from_millis(15);

/// Longest a first tap may be held down and still count as a tap.
const MAX_TAP_HOLD: Duration = Duration::from_millis(600);


/// What a backend reports about one binding's trigger.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Transition {
    /// The trigger just became fully satisfied.
    Activated,
    /// The trigger is no longer satisfied, but part of it may still be held.
    Deactivated,
    /// Every key of the trigger is up.
    Released,
}


/// Owns the per-binding state machines and turns trigger transitions into
pub struct GestureEngine {
    states: Vec<BindingState>,
    /// Binding id -> index into `states`.
    index: HashMap<String, usize>,
}

impl GestureEngine {
    pub fn new(bindings: Vec<HotkeyBinding>) -> Self {
        let mut engine = Self {
            states: Vec::new(),
            index: HashMap::new(),
        };
        engine.reload(bindings);
        engine
    }

    /// Replace the bindings, abandoning any gesture in flight.
    pub fn reload(&mut self, bindings: Vec<HotkeyBinding>) {
        let mut hold_signatures: HashSet<String> = HashSet::new();
        for b in &bindings {
            if !b.disabled && b.gesture == GestureType::DoubleTapHold {
                hold_signatures.insert(b.trigger_signature());
            }
        }

        self.states = bindings
            .into_iter()
            .map(|b| {
                let contended = hold_signatures.contains(&b.trigger_signature());
                BindingState::new(b, contended)
            })
            .collect();
        self.index = self
            .states
            .iter()
            .enumerate()
            .map(|(i, s)| (s.binding.id.clone(), i))
            .collect();
    }

    /// Bindings currently loaded, in configuration order.
    pub fn bindings(&self) -> impl Iterator<Item = &HotkeyBinding> {
        self.states.iter().map(|s| &s.binding)
    }

    pub fn apply(&mut self, binding_id: &str, transition: Transition, tx: &GestureSender) {
        self.apply_at(binding_id, transition, Instant::now(), tx);
    }

    /// `at` is the moment the transition happened, which is what the tap
    pub fn apply_at(
        &mut self,
        binding_id: &str,
        transition: Transition,
        at: Instant,
        tx: &GestureSender,
    ) {
        let Some(&i) = self.index.get(binding_id) else {
            return;
        };
        let signature = self.states[i].signature.clone();
        let sibling_hold_active = self.states.iter().enumerate().any(|(j, s)| {
            j != i
                && s.signature == signature
                && s.binding.gesture == GestureType::DoubleTapHold
                && s.double_tap_hold_active.load(std::sync::atomic::Ordering::SeqCst)
        });

        let state = &mut self.states[i];
        if state.binding.disabled {
            return;
        }
        match transition {
            Transition::Activated => state.on_activate(at, tx),
            Transition::Deactivated => state.on_deactivate(tx),
            Transition::Released => state.on_release(at, sibling_hold_active, tx),
        }
    }

    /// Abandon every gesture in flight, stopping any recording they started.
    pub fn reset(&mut self, tx: &GestureSender) {
        for state in &mut self.states {
            state.abort(tx);
        }
    }
}


pub struct BindingState {
    pub binding: HotkeyBinding,
    signature: String,
    /// A `double_tap_hold` shares this trigger, so `double_tap` cannot resolve
    contended: bool,
    pub hold_active: Arc<AtomicBool>,
    pub hold_cancel: Option<CancellationToken>,
    pub hold_release_cancel: Option<CancellationToken>,
    pub toggle_on: bool,
    pub double_tap: DoubleTapMachine,
    pub double_tap_hold_active: Arc<AtomicBool>,
    pub double_tap_hold_cancel: Option<CancellationToken>,
    pub double_tap_hold_release_cancel: Option<CancellationToken>,
}

impl BindingState {
    pub fn new(binding: HotkeyBinding, contended: bool) -> Self {
        let tap_ms = binding.tap_ms;
        let signature = binding.trigger_signature();
        Self {
            binding,
            signature,
            contended,
            hold_active: Arc::new(AtomicBool::new(false)),
            hold_cancel: None,
            hold_release_cancel: None,
            toggle_on: false,
            double_tap: DoubleTapMachine::new(Duration::from_millis(tap_ms as u64)),
            double_tap_hold_active: Arc::new(AtomicBool::new(false)),
            double_tap_hold_cancel: None,
            double_tap_hold_release_cancel: None,
        }
    }

    fn emit(&self, kind: GestureKind, tx: &GestureSender) {
        let _ = tx.send(GestureEvent {
            binding_id: self.binding.id.clone(),
            binding_label: self.binding.label.clone(),
            target_id: self.binding.target_ids_string(),
            kind,
        });
    }

    fn on_activate(&mut self, at: Instant, tx: &GestureSender) {
        match self.binding.gesture {
            GestureType::Hold => self.start_hold_timer(tx),
            GestureType::Toggle => {
                self.toggle_on = !self.toggle_on;
                self.emit(
                    if self.toggle_on {
                        GestureKind::Start
                    } else {
                        GestureKind::Stop
                    },
                    tx,
                );
            }
            GestureType::DoubleTap => {
                if self.double_tap.on_press(at) == TapOutcome::Completed && !self.contended {
                    self.toggle_on = !self.toggle_on;
                    self.emit(
                        if self.toggle_on {
                            GestureKind::Start
                        } else {
                            GestureKind::Stop
                        },
                        tx,
                    );
                }
            }
            GestureType::DoubleTapHold => {
                if self.double_tap.on_press(at) == TapOutcome::Completed {
                    self.start_double_tap_hold_timer(tx);
                }
            }
        }
    }

    /// The combo was broken (any key released) so keys may still be down.
    fn on_deactivate(&mut self, tx: &GestureSender) {
        if self.hold_active.load(std::sync::atomic::Ordering::SeqCst) {
            if self.hold_release_cancel.is_none() {
                let cancel = CancellationToken::new();
                self.hold_release_cancel = Some(cancel.clone());
                let active = self.hold_active.clone();
                let event = self.pending_event(GestureKind::Stop);
                let tx = tx.clone();
                tokio::spawn(async move {
                    tokio::select! {
                        _ = tokio::time::sleep(Duration::from_millis(50)) => {
                            if active.swap(false, std::sync::atomic::Ordering::SeqCst) {
                                let _ = tx.send(event);
                            }
                        }
                        _ = cancel.cancelled() => {}
                    }
                });
            }
        } else if let Some(cancel) = self.hold_cancel.take() {
            cancel.cancel();
        }

        if self
            .double_tap_hold_active
            .load(std::sync::atomic::Ordering::SeqCst)
        {
            if self.double_tap_hold_release_cancel.is_none() {
                let cancel = CancellationToken::new();
                self.double_tap_hold_release_cancel = Some(cancel.clone());
                let active = self.double_tap_hold_active.clone();
                let event = self.pending_event(GestureKind::Stop);
                let tx = tx.clone();
                tokio::spawn(async move {
                    tokio::select! {
                        _ = tokio::time::sleep(Duration::from_millis(50)) => {
                            if active.swap(false, std::sync::atomic::Ordering::SeqCst) {
                                let _ = tx.send(event);
                            }
                        }
                        _ = cancel.cancelled() => {}
                    }
                });
            }
        } else if let Some(cancel) = self.double_tap_hold_cancel.take() {
            cancel.cancel();
        }
    }

    /// Every key of the trigger is up.
    fn on_release(&mut self, at: Instant, sibling_hold_active: bool, tx: &GestureSender) {
        match self.binding.gesture {
            GestureType::Hold => {
                if let Some(cancel) = self.hold_release_cancel.take() {
                    cancel.cancel();
                }
                if self.hold_active.swap(false, std::sync::atomic::Ordering::SeqCst) {
                    self.hold_cancel.take();
                    self.emit(GestureKind::Stop, tx);
                } else if let Some(cancel) = self.hold_cancel.take() {
                    cancel.cancel();
                }
            }
            GestureType::Toggle => {}
            GestureType::DoubleTap => {
                let completed = self.double_tap.on_release(at) == TapOutcome::Completed;
                if completed && self.contended && !sibling_hold_active {
                    self.toggle_on = !self.toggle_on;
                    self.emit(
                        if self.toggle_on {
                            GestureKind::Start
                        } else {
                            GestureKind::Stop
                        },
                        tx,
                    );
                }
            }
            GestureType::DoubleTapHold => {
                if let Some(cancel) = self.double_tap_hold_release_cancel.take() {
                    cancel.cancel();
                }
                self.double_tap.on_release(at);
                if let Some(cancel) = self.double_tap_hold_cancel.take() {
                    cancel.cancel();
                }
                if self
                    .double_tap_hold_active
                    .swap(false, std::sync::atomic::Ordering::SeqCst)
                {
                    self.emit(GestureKind::Stop, tx);
                }
            }
        }
    }

    /// Tear down without waiting for a release that is never coming.
    fn abort(&mut self, tx: &GestureSender) {
        if let Some(cancel) = self.hold_cancel.take() {
            cancel.cancel();
        }
        if let Some(cancel) = self.double_tap_hold_cancel.take() {
            cancel.cancel();
        }
        let was_holding = self
            .hold_active
            .swap(false, std::sync::atomic::Ordering::SeqCst);
        let was_tap_holding = self
            .double_tap_hold_active
            .swap(false, std::sync::atomic::Ordering::SeqCst);
        let was_toggled = matches!(
            self.binding.gesture,
            GestureType::Toggle | GestureType::DoubleTap
        ) && self.toggle_on;
        let was_recording = was_holding || was_tap_holding || was_toggled;
        self.toggle_on = false;
        self.double_tap.reset();
        if was_recording {
            self.emit(GestureKind::Stop, tx);
        }
    }

    fn start_hold_timer(&mut self, tx: &GestureSender) {
        if self.hold_active.load(std::sync::atomic::Ordering::SeqCst)
            || self.hold_cancel.is_some()
        {
            return;
        }
        let cancel = CancellationToken::new();
        self.hold_cancel = Some(cancel.clone());
        let active = self.hold_active.clone();
        let start = self.pending_event(GestureKind::Start);
        let stop = self.pending_event(GestureKind::Stop);
        let tx = tx.clone();
        let threshold = Duration::from_millis(self.binding.hold_threshold_ms as u64);
        tokio::spawn(async move {
            tokio::select! {
                _ = tokio::time::sleep(threshold) => {}
                _ = cancel.cancelled() => return,
            }
            active.store(true, std::sync::atomic::Ordering::SeqCst);
            let _ = tx.send(start);

            tokio::select! {
                _ = tokio::time::sleep(STUCK_HOLD_MAX) => {
                    if active.swap(false, std::sync::atomic::Ordering::SeqCst) {
                        let _ = tx.send(stop);
                    }
                }
                _ = cancel.cancelled() => {}
            }
        });
    }

    fn start_double_tap_hold_timer(&mut self, tx: &GestureSender) {
        if let Some(cancel) = self.double_tap_hold_cancel.take() {
            cancel.cancel();
        }
        if self
            .double_tap_hold_active
            .swap(false, std::sync::atomic::Ordering::SeqCst)
        {
            self.emit(GestureKind::Stop, tx);
        }

        let cancel = CancellationToken::new();
        self.double_tap_hold_cancel = Some(cancel.clone());
        let active = self.double_tap_hold_active.clone();
        let start = self.pending_event(GestureKind::Start);
        let stop = self.pending_event(GestureKind::Stop);
        let tx = tx.clone();
        let threshold = Duration::from_millis(self.binding.hold_threshold_ms as u64);
        tokio::spawn(async move {
            tokio::select! {
                _ = tokio::time::sleep(threshold) => {}
                _ = cancel.cancelled() => return,
            }
            active.store(true, std::sync::atomic::Ordering::SeqCst);
            let _ = tx.send(start);

            tokio::select! {
                _ = tokio::time::sleep(STUCK_HOLD_MAX) => {
                    if active.swap(false, std::sync::atomic::Ordering::SeqCst) {
                        let _ = tx.send(stop);
                    }
                }
                _ = cancel.cancelled() => {}
            }
        });
    }

    fn pending_event(&self, kind: GestureKind) -> GestureEvent {
        GestureEvent {
            binding_id: self.binding.id.clone(),
            binding_label: self.binding.label.clone(),
            target_id: self.binding.target_ids_string(),
            kind,
        }
    }
}

impl Drop for BindingState {
    fn drop(&mut self) {
        if let Some(cancel) = self.hold_cancel.take() {
            cancel.cancel();
        }
        if let Some(cancel) = self.double_tap_hold_cancel.take() {
            cancel.cancel();
        }
    }
}


#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DtState {
    Idle,
    FirstDown,
    FirstUp,
    SecondDown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TapOutcome {
    None,
    /// The second press landed inside the window (from `on_press`), or the
    Completed,
}

/// Recognises "two taps in quick succession".
pub struct DoubleTapMachine {
    pub state: DtState,
    window: Duration,
    last_release: Option<Instant>,
    first_press: Option<Instant>,
}

impl DoubleTapMachine {
    pub fn new(window: Duration) -> Self {
        Self {
            state: DtState::Idle,
            window,
            last_release: None,
            first_press: None,
        }
    }

    pub fn on_press(&mut self, now: Instant) -> TapOutcome {
        match self.state {
            DtState::Idle => {
                self.state = DtState::FirstDown;
                self.first_press = Some(now);
                TapOutcome::None
            }
            DtState::FirstUp => {
                let gap = self
                    .last_release
                    .map(|r| now.saturating_duration_since(r))
                    .unwrap_or(Duration::MAX);
                if gap >= MIN_TAP_GAP && gap <= self.window {
                    self.state = DtState::SecondDown;
                    TapOutcome::Completed
                } else {
                    self.state = DtState::FirstDown;
                    self.first_press = Some(now);
                    self.last_release = None;
                    TapOutcome::None
                }
            }
            DtState::FirstDown | DtState::SecondDown => TapOutcome::None,
        }
    }

    pub fn on_release(&mut self, now: Instant) -> TapOutcome {
        match self.state {
            DtState::FirstDown => {
                let held = self
                    .first_press
                    .map(|p| now.saturating_duration_since(p))
                    .unwrap_or_default();
                if held > MAX_TAP_HOLD {
                    self.reset();
                } else {
                    self.state = DtState::FirstUp;
                    self.last_release = Some(now);
                }
                TapOutcome::None
            }
            DtState::SecondDown => {
                self.reset();
                TapOutcome::Completed
            }
            DtState::Idle | DtState::FirstUp => TapOutcome::None,
        }
    }

    pub fn reset(&mut self) {
        self.state = DtState::Idle;
        self.last_release = None;
        self.first_press = None;
    }
}


/// Given a set of currently-pressed keys and a list of bindings, return the ids
pub fn shadowed_by_longer(pressed: &HashSet<String>, bindings: &[HotkeyBinding]) -> HashSet<String> {
    let active: Vec<&HotkeyBinding> = bindings
        .iter()
        .filter(|b| !b.disabled)
        .filter(|b| !b.keys.is_empty())
        .filter(|b| b.keys.iter().all(|k| pressed.contains(k)))
        .collect();

    let mut shadowed = HashSet::new();
    for b in &active {
        for other in &active {
            if b.id != other.id
                && other.keys.len() > b.keys.len()
                && b.keys.iter().all(|k| other.keys.contains(k))
            {
                shadowed.insert(b.id.clone());
            }
        }
    }
    shadowed
}

#[cfg(test)]
mod tests {
    use super::*;

    fn binding(id: &str, gesture: GestureType, keys: &[&str]) -> HotkeyBinding {
        HotkeyBinding {
            id: id.to_string(),
            label: format!("{id} label"),
            keys: keys.iter().map(|k| k.to_string()).collect(),
            gesture,
            target_id: "target".to_string(),
            target_ids: vec!["target".to_string()],
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

    /// Press-and-release of an atomic trigger, the way the portal reports it.
    fn tap(engine: &mut GestureEngine, id: &str, tx: &GestureSender) {
        engine.apply(id, Transition::Activated, tx);
        engine.apply(id, Transition::Deactivated, tx);
        engine.apply(id, Transition::Released, tx);
    }

    async fn sleep(ms: u64) {
        tokio::time::sleep(Duration::from_millis(ms)).await;
    }

    #[tokio::test]
    async fn double_tap_starts_on_the_second_press() {
        let (tx, mut rx) = crate::channel();
        let mut engine = GestureEngine::new(vec![binding("dt", GestureType::DoubleTap, &["KEY_LEFTMETA"])]);

        tap(&mut engine, "dt", &tx);
        assert!(rx.try_recv().is_err(), "one tap must not start anything");
        sleep(60).await;
        engine.apply("dt", Transition::Activated, &tx);

        let event = rx.try_recv().expect("second press starts recording");
        assert_eq!(event.binding_id, "dt");
        assert_eq!(event.kind, GestureKind::Start);
    }

    #[tokio::test]
    async fn double_tap_toggles_off_on_the_next_double_tap() {
        let (tx, mut rx) = crate::channel();
        let mut engine = GestureEngine::new(vec![binding("dt", GestureType::DoubleTap, &["KEY_LEFTMETA"])]);

        tap(&mut engine, "dt", &tx);
        sleep(60).await;
        tap(&mut engine, "dt", &tx);
        assert_eq!(rx.try_recv().unwrap().kind, GestureKind::Start);

        sleep(60).await;
        tap(&mut engine, "dt", &tx);
        sleep(60).await;
        tap(&mut engine, "dt", &tx);
        assert_eq!(rx.try_recv().unwrap().kind, GestureKind::Stop);
    }

    #[tokio::test]
    async fn a_very_fast_double_tap_still_fires() {
        let (tx, mut rx) = crate::channel();
        let mut engine = GestureEngine::new(vec![binding("dt", GestureType::DoubleTap, &["KEY_LEFTMETA"])]);

        tap(&mut engine, "dt", &tx);
        sleep(25).await; // well under the old 50ms floor, well within human range
        engine.apply("dt", Transition::Activated, &tx);

        let event = rx.try_recv().expect("a 25ms gap is a double-tap, not bounce");
        assert_eq!(event.kind, GestureKind::Start);
    }

    #[tokio::test]
    async fn a_duplicated_press_does_not_desync_the_machine() {
        let (tx, mut rx) = crate::channel();
        let mut engine = GestureEngine::new(vec![binding("dt", GestureType::DoubleTap, &["KEY_LEFTMETA"])]);

        tap(&mut engine, "dt", &tx);
        engine.apply("dt", Transition::Released, &tx);
        sleep(60).await;
        engine.apply("dt", Transition::Activated, &tx);

        assert_eq!(
            rx.try_recv().expect("gesture survives a duplicated release").kind,
            GestureKind::Start
        );
    }

    #[tokio::test]
    async fn a_slow_second_tap_becomes_a_new_first_tap() {
        let (tx, mut rx) = crate::channel();
        let mut engine = GestureEngine::new(vec![binding("dt", GestureType::DoubleTap, &["KEY_LEFTMETA"])]);

        tap(&mut engine, "dt", &tx);
        sleep(400).await; // beyond tap_ms
        tap(&mut engine, "dt", &tx);
        assert!(rx.try_recv().is_err(), "two slow taps are not a double-tap");

        sleep(60).await;
        engine.apply("dt", Transition::Activated, &tx);
        assert_eq!(rx.try_recv().unwrap().kind, GestureKind::Start);
    }

    #[tokio::test]
    async fn holding_the_key_normally_does_not_prime_a_double_tap() {
        let (tx, mut rx) = crate::channel();
        let mut engine = GestureEngine::new(vec![binding("dt", GestureType::DoubleTap, &["KEY_LEFTMETA"])]);

        engine.apply("dt", Transition::Activated, &tx);
        sleep(700).await; // longer than MAX_TAP_HOLD
        engine.apply("dt", Transition::Deactivated, &tx);
        engine.apply("dt", Transition::Released, &tx);

        sleep(60).await;
        engine.apply("dt", Transition::Activated, &tx);
        assert!(
            rx.try_recv().is_err(),
            "a long hold followed by a press is not a double-tap"
        );
    }

    #[tokio::test]
    async fn double_tap_hold_records_while_held_and_stops_on_release() {
        let (tx, mut rx) = crate::channel();
        let mut engine =
            GestureEngine::new(vec![binding("dth", GestureType::DoubleTapHold, &["KEY_LEFTMETA"])]);

        tap(&mut engine, "dth", &tx);
        sleep(60).await;
        engine.apply("dth", Transition::Activated, &tx);
        assert!(rx.try_recv().is_err(), "not until the hold threshold passes");

        sleep(150).await;
        assert_eq!(rx.try_recv().unwrap().kind, GestureKind::Start);

        engine.apply("dth", Transition::Deactivated, &tx);
        engine.apply("dth", Transition::Released, &tx);
        assert_eq!(rx.try_recv().unwrap().kind, GestureKind::Stop);
    }

    #[tokio::test]
    async fn double_tap_hold_stops_even_if_the_tap_machine_lost_track() {
        let (tx, mut rx) = crate::channel();
        let mut engine =
            GestureEngine::new(vec![binding("dth", GestureType::DoubleTapHold, &["KEY_LEFTMETA"])]);

        tap(&mut engine, "dth", &tx);
        sleep(60).await;
        engine.apply("dth", Transition::Activated, &tx);
        sleep(150).await;
        assert_eq!(rx.try_recv().unwrap().kind, GestureKind::Start);

        engine.states[0].double_tap.reset();

        engine.apply("dth", Transition::Released, &tx);
        assert_eq!(
            rx.try_recv().expect("release must always stop recording").kind,
            GestureKind::Stop
        );
    }

    #[tokio::test]
    async fn double_tap_hold_does_not_leak_a_second_recording() {
        let (tx, mut rx) = crate::channel();
        let mut engine =
            GestureEngine::new(vec![binding("dth", GestureType::DoubleTapHold, &["KEY_LEFTMETA"])]);

        tap(&mut engine, "dth", &tx);
        sleep(60).await;
        engine.apply("dth", Transition::Activated, &tx);
        sleep(150).await;
        assert_eq!(rx.try_recv().unwrap().kind, GestureKind::Start);

        engine.states[0].double_tap.reset();
        tap(&mut engine, "dth", &tx);
        sleep(60).await;
        engine.apply("dth", Transition::Activated, &tx);

        assert_eq!(
            rx.try_recv().expect("stale recording is closed first").kind,
            GestureKind::Stop
        );
        sleep(150).await;
        assert_eq!(rx.try_recv().unwrap().kind, GestureKind::Start);
    }

    #[tokio::test(start_paused = true)]
    async fn double_tap_hold_stops_itself_after_the_safety_timeout() {
        let (tx, mut rx) = crate::channel();
        let mut engine =
            GestureEngine::new(vec![binding("dth", GestureType::DoubleTapHold, &["KEY_LEFTMETA"])]);

        tap(&mut engine, "dth", &tx);
        sleep(60).await;
        engine.apply("dth", Transition::Activated, &tx);
        sleep(150).await;
        assert_eq!(rx.try_recv().unwrap().kind, GestureKind::Start);

        sleep(121_000).await;
        assert_eq!(rx.try_recv().unwrap().kind, GestureKind::Stop);

        engine.apply("dth", Transition::Released, &tx);
        assert!(rx.try_recv().is_err(), "no duplicate stop on the real release");
    }

    #[tokio::test]
    async fn double_tap_and_double_tap_hold_coexist_on_one_key() {
        let (tx, mut rx) = crate::channel();
        let mut engine = GestureEngine::new(vec![
            binding("dt", GestureType::DoubleTap, &["KEY_LEFTMETA"]),
            binding("dth", GestureType::DoubleTapHold, &["KEY_LEFTMETA"]),
        ]);

        tap(&mut engine, "dt", &tx);
        tap(&mut engine, "dth", &tx);
        sleep(60).await;
        engine.apply("dt", Transition::Activated, &tx);
        engine.apply("dth", Transition::Activated, &tx);
        assert!(rx.try_recv().is_err(), "the tap must wait for the release here");
        engine.apply("dt", Transition::Released, &tx);
        engine.apply("dth", Transition::Released, &tx);

        let event = rx.try_recv().unwrap();
        assert_eq!(event.binding_id, "dt");
        assert_eq!(event.kind, GestureKind::Start);
        assert!(rx.try_recv().is_err(), "the hold must not also fire");

        engine.states[0].toggle_on = false;
        sleep(200).await;

        tap(&mut engine, "dt", &tx);
        tap(&mut engine, "dth", &tx);
        sleep(60).await;
        engine.apply("dt", Transition::Activated, &tx);
        engine.apply("dth", Transition::Activated, &tx);
        sleep(150).await;

        let event = rx.try_recv().unwrap();
        assert_eq!(event.binding_id, "dth");
        assert_eq!(event.kind, GestureKind::Start);

        engine.apply("dt", Transition::Released, &tx);
        engine.apply("dth", Transition::Released, &tx);
        let event = rx.try_recv().unwrap();
        assert_eq!(event.binding_id, "dth");
        assert_eq!(event.kind, GestureKind::Stop);
        assert!(rx.try_recv().is_err(), "the tap must stay out of the hold's way");
    }

    #[tokio::test]
    async fn hold_waits_for_the_threshold_and_stops_on_release() {
        let (tx, mut rx) = crate::channel();
        let mut engine = GestureEngine::new(vec![binding("h", GestureType::Hold, &["KEY_LEFTALT"])]);

        engine.apply("h", Transition::Activated, &tx);
        assert!(rx.try_recv().is_err());
        sleep(150).await;
        assert_eq!(rx.try_recv().unwrap().kind, GestureKind::Start);

        engine.apply("h", Transition::Released, &tx);
        assert_eq!(rx.try_recv().unwrap().kind, GestureKind::Stop);
    }

    #[tokio::test]
    async fn hold_released_inside_the_threshold_never_starts() {
        let (tx, mut rx) = crate::channel();
        let mut engine = GestureEngine::new(vec![binding("h", GestureType::Hold, &["KEY_LEFTALT"])]);

        engine.apply("h", Transition::Activated, &tx);
        engine.apply("h", Transition::Deactivated, &tx);
        engine.apply("h", Transition::Released, &tx);
        sleep(200).await;
        assert!(rx.try_recv().is_err(), "the pending start must be cancelled");
    }

    #[tokio::test(start_paused = true)]
    async fn hold_stops_itself_after_the_safety_timeout() {
        let (tx, mut rx) = crate::channel();
        let mut engine = GestureEngine::new(vec![binding("h", GestureType::Hold, &["KEY_LEFTALT"])]);

        engine.apply("h", Transition::Activated, &tx);
        sleep(150).await;
        assert_eq!(rx.try_recv().unwrap().kind, GestureKind::Start);

        sleep(121_000).await;
        assert_eq!(rx.try_recv().unwrap().kind, GestureKind::Stop);

        engine.apply("h", Transition::Deactivated, &tx);
        engine.apply("h", Transition::Released, &tx);
        assert!(rx.try_recv().is_err(), "no duplicate stop on the real release");
    }

    #[tokio::test]
    async fn hold_ignores_a_partial_release_of_a_combo() {
        let (tx, mut rx) = crate::channel();
        let mut engine = GestureEngine::new(vec![binding(
            "h",
            GestureType::Hold,
            &["KEY_LEFTMETA", "KEY_SPACE"],
        )]);

        engine.apply("h", Transition::Activated, &tx);
        sleep(150).await;
        assert_eq!(rx.try_recv().unwrap().kind, GestureKind::Start);

        engine.apply("h", Transition::Deactivated, &tx); // Space up, Super still down
        assert!(rx.try_recv().is_err(), "must not stop while a key is held");

        engine.apply("h", Transition::Released, &tx);
        assert_eq!(rx.try_recv().unwrap().kind, GestureKind::Stop);
    }

    #[tokio::test]
    async fn toggle_alternates_on_each_activation() {
        let (tx, mut rx) = crate::channel();
        let mut engine = GestureEngine::new(vec![binding("t", GestureType::Toggle, &["KEY_LEFTCTRL"])]);

        engine.apply("t", Transition::Activated, &tx);
        assert_eq!(rx.try_recv().unwrap().kind, GestureKind::Start);
        engine.apply("t", Transition::Released, &tx);
        assert!(rx.try_recv().is_err());
        engine.apply("t", Transition::Activated, &tx);
        assert_eq!(rx.try_recv().unwrap().kind, GestureKind::Stop);
    }

    #[tokio::test]
    async fn reset_stops_a_recording_whose_release_can_never_arrive() {
        let (tx, mut rx) = crate::channel();
        let mut engine = GestureEngine::new(vec![binding("h", GestureType::Hold, &["KEY_LEFTALT"])]);

        engine.apply("h", Transition::Activated, &tx);
        sleep(150).await;
        assert_eq!(rx.try_recv().unwrap().kind, GestureKind::Start);

        engine.reset(&tx); // keyboard unplugged mid-hold
        assert_eq!(rx.try_recv().unwrap().kind, GestureKind::Stop);
    }

    #[tokio::test]
    async fn disabled_bindings_never_fire() {
        let (tx, mut rx) = crate::channel();
        let mut b = binding("t", GestureType::Toggle, &["KEY_LEFTCTRL"]);
        b.disabled = true;
        let mut engine = GestureEngine::new(vec![b]);

        engine.apply("t", Transition::Activated, &tx);
        assert!(rx.try_recv().is_err());
    }

    #[test]
    fn shadowing_suppresses_the_shorter_combo() {
        let bindings = vec![
            binding("short", GestureType::Hold, &["KEY_LEFTMETA", "KEY_SPACE"]),
            binding(
                "long",
                GestureType::Hold,
                &["KEY_LEFTCTRL", "KEY_LEFTMETA", "KEY_SPACE"],
            ),
        ];
        let pressed: HashSet<String> = ["KEY_LEFTCTRL", "KEY_LEFTMETA", "KEY_SPACE"]
            .iter()
            .map(|s| s.to_string())
            .collect();

        let shadowed = shadowed_by_longer(&pressed, &bindings);
        assert!(shadowed.contains("short"));
        assert!(!shadowed.contains("long"));
    }
}
