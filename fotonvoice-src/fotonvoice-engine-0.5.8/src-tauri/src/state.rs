use std::collections::HashSet;
use std::sync::{Arc, atomic::{AtomicBool, AtomicU32, Ordering}};

use tokio::sync::Mutex;
use fotonvoice_config::Config;
use fotonvoice_routing::OutputTargetRouter;

/// All shared mutable state, behind Arc so it can be handed to Tauri commands.
pub struct AppState {
    pub config: Arc<Mutex<Config>>,
    pub router: Arc<OutputTargetRouter>,

    /// True while a hotkey hold/toggle is active (recording)
    pub recording: Arc<AtomicBool>,
    /// True while speech transcription/OpenAI post-processing is running
    pub processing: Arc<AtomicBool>,
    /// True while TTS is playing back
    pub speaking: Arc<AtomicBool>,
    /// Live mirror of `ui.show_overlay` so the hot status-forwarding loops can
    pub overlay_enabled: Arc<AtomicBool>,
    /// True while MCP server is actively recording/listening to the microphone
    pub mcp_recording: Arc<AtomicBool>,
    /// When the command-executed overlay pill should stop showing, if it's
    pub command_overlay_until: Arc<std::sync::Mutex<Option<std::time::Instant>>>,
    /// True while the user is recording a new keybind in the settings UI.
    pub hotkeys_inhibited: Arc<AtomicBool>,
    /// True when dynamic stream has successfully opened and is active (Option A)
    pub audio_ready: Arc<AtomicBool>,
    /// Live sync atomic flag for dynamic stream preference
    pub dynamic_stream: Arc<AtomicBool>,
    /// True when Svelte settings Audio tab is actively monitoring audio level
    pub monitoring: Arc<AtomicBool>,
    /// Live input device index, mapped to u32::MAX when None (default system device)
    pub input_device_index: Arc<AtomicU32>,
    /// Live gain value, stored as f32 bits
    pub gain: Arc<AtomicU32>,
    /// Live noise-suppression preference, read by the capture callback
    pub noise_suppression: Arc<AtomicBool>,

    /// Total words injected this session
    pub word_count: Arc<std::sync::atomic::AtomicU32>,

    /// Most recent transcription result (shown in the overlay)
    pub last_text: Arc<Mutex<String>>,

    /// Monotonic counter - incremented each time last_text is written.
    pub last_text_version: Arc<std::sync::atomic::AtomicU64>,

    /// Currently active dictation target ID
    pub active_target: Arc<Mutex<String>>,

    /// Currently active keybind display name/label
    pub active_binding_label: Arc<Mutex<String>>,

    /// Currently active hotkey binding ID
    pub active_binding_id: Arc<Mutex<String>>,

    /// Currently configured target definitions (in-memory cache for fast lookups).
    /// Replace it through [`AppState::set_targets`], which also bumps
    /// `targets_version`.
    pub targets: Arc<Mutex<Vec<fotonvoice_routing::OutputTarget>>>,

    /// Currently configured hotkey bindings (in-memory cache; replaced by the
    /// bindings save and canary stop-key plumbing reads it instead of disk).
    pub bindings: Arc<Mutex<Vec<fotonvoice_routing::HotkeyBinding>>>,

    /// Incremented every time `targets` is replaced, so anything caching a
    /// value derived from the targets (the tray's target label) can tell its
    /// cache is stale without re-deriving it on every tick.
    pub targets_version: Arc<std::sync::atomic::AtomicU64>,

    /// Channel sender to send empty audio chunks as sentinels to unblock the coordinator thread
    pub audio_tx: crossbeam_channel::Sender<Vec<f32>>,
    /// Nudges the audio capture supervisor when a flag it watches changes, so
    pub audio_wake: crossbeam_channel::Sender<()>,
    pub audio_handle: std::sync::Mutex<Option<fotonvoice_audio::RecorderHandle>>,

    /// Channel sender for pushing runtime snapshots (config, targets, bindings)
    pub inference_config_tx:
        crossbeam_channel::Sender<Arc<fotonvoice_inference::InferenceRuntimeInput>>,

    /// Load state of the STT worker thread.
    ///
    /// Read this instead of the request channel when deciding whether a
    /// recording can be handed over: the worker blocks inside `load()` and
    /// cannot answer anything until it finishes, which during a cold ONNX load
    /// is minutes.
    pub stt_load: fotonvoice_inference::LoadStatus,

    /// Playback engine handle
    pub tts_handle: Arc<Mutex<Option<fotonvoice_tts::TtsEngineHandle>>>,

    /// Set of active FIFO response pipes currently being listened to
    pub active_fifos: Arc<Mutex<HashSet<String>>>,

    /// True while the TTS stop key is part of the listener's binding set.
    pub stop_key_held: Arc<AtomicBool>,

    /// Tells the stop-key arbiter that playback started or stopped. Set once,
    pub speaking_tx: Arc<std::sync::OnceLock<tokio::sync::mpsc::UnboundedSender<bool>>>,

    /// Channel for sending hotkey configurations directly to background threads
    pub hotkey_reloader: Arc<Mutex<Option<crossbeam_channel::Sender<Vec<fotonvoice_routing::HotkeyBinding>>>>>,

    /// Channel for forwarding hotkey gestures from listener to app coordinator
    pub hotkey_gesture_tx: Arc<Mutex<Option<fotonvoice_hotkeys::GestureSender>>>,

    /// Live view of whether the global hotkey listener can actually see a
    pub hotkey_health: Arc<fotonvoice_hotkeys::ListenerHealth>,

    /// Channel sender for `{"type":"position",...}` messages that reposition
    pub overlay_tx: crossbeam_channel::Sender<String>,
}

impl AppState {
    pub fn is_recording(&self) -> bool {
        self.recording.load(Ordering::SeqCst)
    }

    pub fn is_speaking(&self) -> bool {
        self.speaking.load(Ordering::SeqCst)
    }

    pub fn is_hotkeys_inhibited(&self) -> bool {
        self.hotkeys_inhibited.load(Ordering::SeqCst)
    }

    pub fn set_hotkeys_inhibited(&self, v: bool) {
        self.hotkeys_inhibited.store(v, Ordering::SeqCst);
    }

    pub fn is_processing(&self) -> bool {
        self.processing.load(Ordering::SeqCst)
    }

    pub fn set_processing(&self, v: bool) {
        self.processing.store(v, Ordering::SeqCst);
    }

    pub fn is_audio_ready(&self) -> bool {
        self.audio_ready.load(Ordering::SeqCst)
    }

    pub fn stt_load_state(&self) -> fotonvoice_inference::LoadState {
        self.stt_load.state()
    }

    pub fn set_dynamic_stream(&self, v: bool) {
        self.dynamic_stream.store(v, Ordering::SeqCst);
        self.wake_audio();
    }

    /// Tell the capture supervisor to look at its flags now. Never blocks: the
    fn wake_audio(&self) {
        let _ = self.audio_wake.try_send(());
    }

    pub fn set_monitoring(&self, v: bool) {
        self.monitoring.store(v, Ordering::SeqCst);
        self.wake_audio();
        if !v {
            let _ = self.audio_tx.send(Vec::new());
        }
    }

    pub fn set_input_device_index(&self, v: Option<u32>) {
        self.input_device_index.store(v.unwrap_or(u32::MAX), Ordering::SeqCst);
        self.wake_audio();
    }

    pub fn set_noise_suppression(&self, v: bool) {
        self.noise_suppression.store(v, Ordering::SeqCst);
    }

    pub fn set_gain(&self, v: f32) {
        self.gain.store(v.to_bits(), Ordering::SeqCst);
    }

    /// Start capturing, and stop anything FotonVoice Engine is saying while it does.
    pub async fn begin_recording(&self) {
        if let Some(tts) = self.tts_handle.lock().await.as_ref() {
            tts.stop();
        }
        self.set_recording(true);
    }

    /// Prefer `begin_recording` to start dictation - it also interrupts
    pub fn set_recording(&self, v: bool) {
        self.recording.store(v, Ordering::SeqCst);
        self.wake_audio();
        if !v {
            let _ = self.audio_tx.send(Vec::new());
        }
    }

    pub fn set_speaking(&self, v: bool) {
        self.speaking.store(v, Ordering::SeqCst);
        if let Some(tx) = self.speaking_tx.get() {
            let _ = tx.send(v);
        }
    }

    pub fn is_overlay_enabled(&self) -> bool {
        self.overlay_enabled.load(Ordering::SeqCst)
    }

    pub fn set_overlay_enabled(&self, v: bool) {
        self.overlay_enabled.store(v, Ordering::SeqCst);
    }

    pub fn is_mcp_recording(&self) -> bool {
        self.mcp_recording.load(Ordering::SeqCst)
    }

    pub fn set_mcp_recording(&self, v: bool) {
        self.mcp_recording.store(v, Ordering::SeqCst);
    }

    /// Mark the command-executed overlay pill active for `duration` from now.
    pub fn activate_command_overlay(&self, duration: std::time::Duration) {
        *self.command_overlay_until.lock().unwrap() = Some(std::time::Instant::now() + duration);
    }

    /// Whether the command-executed overlay pill should still be showing.
    pub fn is_command_overlay_active(&self) -> bool {
        self.command_overlay_until
            .lock()
            .unwrap()
            .is_some_and(|until| std::time::Instant::now() < until)
    }

    /// Replace the in-memory targets cache and mark it changed.
    pub async fn set_targets(&self, targets: Vec<fotonvoice_routing::OutputTarget>) {
        *self.targets.lock().await = targets;
        self.targets_version.fetch_add(1, Ordering::SeqCst);
    }

    /// Replace the in-memory bindings cache.
    pub async fn set_bindings(&self, bindings: Vec<fotonvoice_routing::HotkeyBinding>) {
        *self.bindings.lock().await = bindings;
    }

    /// Build the runtime snapshot the inference engine transcribes with and
    /// push it. Called once after each change to the config, the targets or
    /// the bindings, replacing the per-utterance config-directory reads.
    pub async fn push_inference_runtime(&self) {
        let snapshot = Arc::new(fotonvoice_inference::InferenceRuntimeInput {
            app_config: Arc::new(self.config.lock().await.data.clone()),
            targets: Arc::new(self.targets.lock().await.clone()),
            bindings: Arc::new(self.bindings.lock().await.clone()),
        });
        let _ = self.inference_config_tx.send(snapshot);
    }

    pub fn targets_version(&self) -> u64 {
        self.targets_version.load(Ordering::SeqCst)
    }

    pub fn increment_words(&self, n: u32) {
        self.word_count.fetch_add(n, Ordering::SeqCst);
    }

    pub fn total_words(&self) -> u32 {
        self.word_count.load(Ordering::SeqCst)
    }

    /// Start loading the TTS model now, before anything asks it to speak.
    ///
    /// The load takes seconds, so kicking it off the moment we know speech is
    /// coming (the user has just started dictating, or a spoken command was
    /// recognised) hides most of that behind the time they spend talking. That
    /// holds in both memory modes: on-demand drops the model when idle, and
    /// always-loaded without pre-warming loads it lazily. When the model is
    /// already resident the worker treats this as a no-op.
    pub async fn preload_tts(&self) {
        if !self.config.lock().await.data.tts.enabled {
            return;
        }
        if let Some(tts) = self.tts_handle.lock().await.as_ref() {
            tts.preload();
        }
    }

    /// True when dictating through `target_id` is likely to end in speech -
    pub async fn target_leads_to_speech(&self, target_id: &str) -> bool {
        use fotonvoice_routing::DeliveryType;
        let targets = self.targets.lock().await;
        targets.iter().any(|t| {
            t.id == target_id
                && (t.delivery == DeliveryType::Speak
                    || t.response_pipe.as_deref().is_some_and(|p| !p.trim().is_empty()))
        })
    }

    /// Pre-load the TTS model if `target_id` is one that ends in speech.
    pub async fn preload_tts_for_target(&self, target_id: &str) {
        if target_id.is_empty() || self.target_leads_to_speech(target_id).await {
            self.preload_tts().await;
        }
    }

    /// Start a responder for every target response pipe not already watched.
    ///
    /// Responders look the TTS worker up per line (see
    /// `fotonvoice_tts::run_fifo_responder`), so one started while TTS is off
    /// begins speaking as soon as it is turned on, and none needs restarting
    /// when the worker is replaced.
    pub async fn spawn_fifo_responders(&self) {
        let targets_guard = self.targets.lock().await;
        let mut active_fifos_guard = self.active_fifos.lock().await;

        for target in targets_guard.iter() {
            if let Some(ref pipe_path) = target.response_pipe {
                if !pipe_path.trim().is_empty() && active_fifos_guard.insert(pipe_path.clone()) {
                    tokio::spawn(fotonvoice_tts::run_fifo_responder(
                        pipe_path.clone(),
                        self.tts_handle.clone(),
                    ));
                }
            }
        }
    }
}
