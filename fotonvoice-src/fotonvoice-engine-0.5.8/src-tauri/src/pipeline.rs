use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::time::Duration;
use tauri::Emitter;
use tokio::sync::Mutex;
use fotonvoice_hotkeys::GestureKind;

use crate::state::AppState;
#[cfg(target_os = "linux")]
use crate::tray::update_tray_for_setup;
use crate::window::{setup_blocker, show_setup_window, SETUP_NOTICE_INTERVAL};
#[cfg(target_os = "linux")]
use crate::window::{BLIND_ALERT_INTERVAL, SETUP_POLL_INTERVAL};
use fotonvoice_inference::InferenceOutput;

async fn process_remote_transcription(
    res: anyhow::Result<fotonvoice_inference::backend::TranscriptionResult>,
    audio: Vec<f32>,
    target_id: String,
    binding_id: String,
    state: Arc<AppState>,
    text_tx: crossbeam_channel::Sender<fotonvoice_inference::InferenceOutput>,
) {
    let result = match res {
        Ok(res) => res,
        Err(e) => {
            let _ = text_tx.send(InferenceOutput {
                text: String::new(),
                target_id,
                binding_id: Some(binding_id),
                raw_text: String::new(),
                inference_ms: 0,
                language: String::new(),
                error: Some(format!("{e:#}")),
            });
            return;
        }
    };

    let sum_sq: f32 = audio.iter().map(|&s| s * s).sum();
    let rms = if audio.is_empty() {
        0.0
    } else {
        (sum_sq / audio.len() as f32).sqrt()
    };
    let vad_threshold = {
        let guard = state.config.lock().await;
        guard.data.audio.vad_threshold
    };
    let rms_threshold = (1.0 - vad_threshold) * 0.006;

    if rms < rms_threshold {
        tracing::info!(
            "Remote audio skipped by noise gate: RMS is {:.5} (threshold is {:.5}, vad_threshold={:.2})",
            rms,
            rms_threshold,
            vad_threshold
        );
        let _ = text_tx.send(InferenceOutput {
            text: String::new(),
            target_id,
            binding_id: Some(binding_id),
            raw_text: String::new(),
            inference_ms: result.inference_ms,
            language: result.language,
            error: None,
        });
        return;
    }

    let dir = fotonvoice_routing::config_dir();
    let targets = fotonvoice_routing::load_targets_cached(&dir);
    let target_ids: Vec<&str> = target_id
        .split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .collect();
    let first_target_id = target_ids.first().copied().unwrap_or("default");
    let target = targets.iter().find(|t| t.id == first_target_id);

    let features = {
        let guard = state.config.lock().await;
        guard.data.features.clone()
    };

    let remove_fillers = target
        .and_then(|t| t.processing.remove_fillers)
        .unwrap_or(features.remove_fillers);

    let spoken_punctuation = target
        .and_then(|t| t.processing.spoken_punctuation)
        .unwrap_or(features.spoken_punctuation);

    let auto_format_lists = target
        .and_then(|t| t.processing.auto_format_lists)
        .unwrap_or(features.auto_format_lists);

    let code_mode = target
        .and_then(|t| t.processing.code_mode)
        .unwrap_or(false);

    let post_cfg = fotonvoice_inference::postprocess::PostProcessConfig {
        remove_fillers,
        spoken_punctuation,
        auto_format_lists,
        apply_snippets: !features.snippets.is_empty(),
        snippets: &features.snippets,
        code_mode,
        custom_vocabulary: &features.custom_vocabulary,
    };

    let mut processed = fotonvoice_inference::postprocess::run_pipeline(&result.text, &post_cfg);
    let raw_text = result.text.clone();

    if !processed.is_empty()
        && fotonvoice_inference::postprocess::is_silence_hallucination(&processed)
        && rms < 0.003
    {
        tracing::info!(
            "Discarded silence hallucination '{}' (audio RMS: {:.5})",
            processed,
            rms
        );
        processed = String::new();
    }

    let bindings = fotonvoice_routing::load_bindings_cached(&dir);
    let binding = bindings.iter().find(|b| b.id == binding_id);

    let binding_wants_openai = binding
        .and_then(|b| b.openai_enabled)
        .unwrap_or(false);

    if binding_wants_openai && !processed.is_empty() {
        let mut openai_cfg = fotonvoice_config::Config::load().data.openai;
        openai_cfg.enabled = true;

        if let Some(b) = binding {
            if let Some(ref model) = b.openai_model {
                if !model.is_empty() {
                    openai_cfg.model = model.clone();
                }
            }
            if let Some(ref mode_str) = b.openai_mode {
                let mode = match mode_str.as_str() {
                    "clean" => fotonvoice_config::OpenAiMode::Clean,
                    "formal" => fotonvoice_config::OpenAiMode::Formal,
                    "casual" => fotonvoice_config::OpenAiMode::Casual,
                    "bullet" => fotonvoice_config::OpenAiMode::Bullet,
                    "concise" => fotonvoice_config::OpenAiMode::Concise,
                    "custom" => fotonvoice_config::OpenAiMode::Custom,
                    _ => fotonvoice_config::OpenAiMode::Clean,
                };
                if mode != fotonvoice_config::OpenAiMode::Custom {
                    openai_cfg.mode = mode.clone();
                    openai_cfg.system_prompt =
                        fotonvoice_llm::preset_system_prompt(&mode).to_string();
                }
            }
            if let Some(ref system_prompt) = b.openai_system_prompt {
                if !system_prompt.is_empty() {
                    openai_cfg.system_prompt = system_prompt.clone();
                }
            }
            if let Some(ref prompt) = b.openai_prompt {
                if !prompt.is_empty() {
                    openai_cfg.user_prompt = prompt.clone();
                }
            }
        }

        let client = fotonvoice_llm::OpenAiClient::new(openai_cfg);
        processed = client.process(&processed).await;
    }

    let _ = text_tx.send(InferenceOutput {
        text: processed,
        target_id,
        binding_id: Some(binding_id),
        raw_text,
        inference_ms: result.inference_ms,
        language: result.language,
        error: None,
    });
}

pub fn spawn_audio_coordinator(
    state_for_audio: Arc<AppState>,
    audio_rx: crossbeam_channel::Receiver<fotonvoice_audio::AudioChunk>,
    inference_tx: crossbeam_channel::Sender<fotonvoice_inference::InferenceRequest>,
    text_tx: crossbeam_channel::Sender<fotonvoice_inference::InferenceOutput>,
    rt_handle: tokio::runtime::Handle,
) {
    std::thread::spawn(move || {
        let mut accumulated_audio = Vec::<f32>::new();
        let mut was_recording = false;
        let mut target_id = "default".to_string();
        let mut binding_id = String::new();
        let mut remote_session: Option<fotonvoice_inference::RemoteStreamingSession> = None;
        let mut is_remote_backend = false;

        while let Ok(chunk) = audio_rx.recv() {
            let is_recording = state_for_audio.is_recording();

            if is_recording {
                if !was_recording {
                    accumulated_audio.clear();
                    target_id = state_for_audio.active_target.blocking_lock().clone();
                    binding_id = state_for_audio.active_binding_id.blocking_lock().clone();
                    was_recording = true;

                    let backend = state_for_audio.config.blocking_lock().data.engine.backend.clone();
                    if backend == fotonvoice_config::BackendChoice::RemoteOpenAi {
                        is_remote_backend = true;
                        let (remote_openai, custom_vocabulary) = {
                            let cfg_lock = state_for_audio.config.blocking_lock();
                            (
                                cfg_lock.data.engine.remote_openai.clone(),
                                cfg_lock.data.features.custom_vocabulary.clone(),
                            )
                        };
                        let mut merged_prompt = String::from(
                            "FotonVoice Engine is a voice control assistant application. FotonVoice Engine commands start with FotonVoice Engine. ",
                        );
                        if !custom_vocabulary.is_empty() {
                            merged_prompt.push_str("Vocabulary: ");
                            merged_prompt.push_str(&custom_vocabulary.join(", "));
                            merged_prompt.push_str(". ");
                        }
                        let prompt = {
                            let end = merged_prompt.trim_end().len();
                            merged_prompt.truncate(end);
                            (!merged_prompt.is_empty()).then_some(merged_prompt)
                        };
                        let session = fotonvoice_inference::RemoteStreamingSession::start(
                            remote_openai,
                            prompt,
                            &rt_handle,
                        );
                        remote_session = Some(session);
                    } else {
                        is_remote_backend = false;
                        remote_session = None;
                    }
                }

                if let Some(ref session) = remote_session {
                    session.send_chunk(chunk);
                } else {
                    accumulated_audio.extend(chunk);
                }
            } else {
                if was_recording {
                    if is_remote_backend {
                        if let Some(session) = remote_session.take() {
                            let audio = session.take_buffered_samples();
                            if !audio.is_empty() {
                                state_for_audio.set_processing(true);
                                let state_clone = state_for_audio.clone();
                                let text_tx_clone = text_tx.clone();
                                let target_id_clone = target_id.clone();
                                let binding_id_clone = binding_id.clone();

                                rt_handle.spawn(async move {
                                    let res = session.finish().await;
                                    process_remote_transcription(
                                        res,
                                        audio,
                                        target_id_clone,
                                        binding_id_clone,
                                        state_clone,
                                        text_tx_clone,
                                    )
                                    .await;
                                });
                            }
                        }
                    } else {
                        if !accumulated_audio.is_empty() {
                            let req = fotonvoice_inference::InferenceRequest {
                                audio: std::mem::take(&mut accumulated_audio),
                                target_id: target_id.clone(),
                                binding_id: Some(binding_id.clone()),
                            };
                            state_for_audio.set_processing(true);
                            let _ = inference_tx.send(req);
                        }
                    }
                    was_recording = false;
                }
            }
        }
    });
}

pub fn spawn_text_delivery_worker(
    state: Arc<AppState>,
    text_rx: crossbeam_channel::Receiver<fotonvoice_inference::InferenceOutput>,
    rt_handle: tokio::runtime::Handle,
) {
    std::thread::spawn(move || {
        while let Ok(output) = text_rx.recv() {
            state.set_processing(false);
            if let Some(ref err) = output.error {
                tracing::error!("Transcription failed: {err}");
                fotonvoice_inject::show_notification("FotonVoice Engine - transcription failed", err);
                continue;
            }
            if output.text.trim().is_empty() {
                continue;
            }

            let (global_s1_mini_enabled, s1_mini_styling) = {
                let cfg_lock = state.config.blocking_lock();
                (cfg_lock.data.engine.s1_mini.enabled, cfg_lock.data.engine.s1_mini.styling.clone())
            };

            let dir = fotonvoice_routing::config_dir();
            let bindings = fotonvoice_routing::load_bindings_cached(&dir);
            let binding = output.binding_id.as_ref().and_then(|bid| bindings.iter().find(|b| &b.id == bid));
            let s1_mini_enabled = binding
                .and_then(|b| b.s1_mini_enabled)
                .unwrap_or(global_s1_mini_enabled);

            let targets = fotonvoice_routing::load_targets_cached(&dir);
            let (target_id, text) = if let Some(parsed) = fotonvoice_routing::targets::parse_voice_command(&output.text, &targets) {
                let matched_id = parsed.matched_target_id.clone();
                let payload = parsed.payload;
                let matched_label = targets
                    .iter()
                    .find(|t| t.id == matched_id)
                    .map(|t| if t.label.is_empty() { t.id.clone() } else { t.label.clone() })
                    .unwrap_or_else(|| matched_id.clone());
                fotonvoice_routing::targets::notify_command_trigger(&matched_label, &payload);

                let cleaned_payload = if s1_mini_enabled && !payload.trim().is_empty() {
                    fotonvoice_inference::s1_mini::clean_dictation(&payload, &s1_mini_styling, None)
                } else {
                    payload
                };
                (matched_id, cleaned_payload)
            } else {
                let cleaned_text = if s1_mini_enabled && !output.text.trim().is_empty() {
                    fotonvoice_inference::s1_mini::clean_dictation(&output.text, &s1_mini_styling, None)
                } else {
                    output.text.clone()
                };
                (output.target_id.clone(), cleaned_text)
            };

            if text.trim().is_empty() {
                continue;
            }

            tracing::info!(
                "Received transcription: \"{}\" for target '{}' (took {}ms)",
                text,
                target_id,
                output.inference_ms
            );
            let words = text.split_whitespace().count() as u32;
            state.increment_words(words);

            let router = state.router.clone();
            let state_lt = state.clone();
            let text_lt = text.clone();
            let text_to_deliver = text.clone();
            let target_to_deliver = target_id.clone();
            rt_handle.spawn(async move {
                {
                    let mut lt = state_lt.last_text.lock().await;
                    *lt = text_lt;
                    state_lt
                        .last_text_version
                        .fetch_add(1, Ordering::SeqCst);
                }
                let target_ids: Vec<String> = target_to_deliver
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect();
                for tid in target_ids {
                    router.deliver_direct(&tid, &text_to_deliver).await;
                }
            });

            let show_notif = {
                let cfg_lock = state.config.blocking_lock();
                cfg_lock.data.ui.show_notification
            };
            if show_notif {
                fotonvoice_inject::show_notification("FotonVoice Engine", &text);
            }
        }
    });
}

pub fn spawn_hotkey_gesture_handler(
    state_for_gesture: Arc<AppState>,
    gesture_rx: fotonvoice_hotkeys::GestureReceiver,
) {
    let last_setup_notice = Arc::new(Mutex::new(None::<std::time::Instant>));
    let mic_notice_shown = Arc::new(AtomicBool::new(false));
    let gesture_rx = Arc::new(Mutex::new(gesture_rx));

    tokio::spawn(async move {
        loop {
            let task = tokio::spawn(run_gesture_loop(
                state_for_gesture.clone(),
                gesture_rx.clone(),
                last_setup_notice.clone(),
                mic_notice_shown.clone(),
            ));
            match task.await {
                Ok(()) => {
                    tracing::warn!(
                        "Hotkey gesture loop ended: the listener's channel closed. No \
                         shortcut can fire until FotonVoice Engine is restarted."
                    );
                    return;
                }
                Err(e) if e.is_panic() => {
                    tracing::error!(
                        "Hotkey gesture loop panicked ({e}); restarting it so shortcuts \
                         keep working. Please report this with the backtrace above."
                    );
                }
                Err(e) => {
                    tracing::warn!("Hotkey gesture loop stopped: {e}");
                    return;
                }
            }
        }
    });
}

async fn run_gesture_loop(
    state_for_gesture: Arc<AppState>,
    gesture_rx: Arc<Mutex<fotonvoice_hotkeys::GestureReceiver>>,
    last_setup_notice: Arc<Mutex<Option<std::time::Instant>>>,
    mic_notice_shown: Arc<AtomicBool>,
) {
    let mut gesture_rx = gesture_rx.lock().await;
    while let Some(event) = gesture_rx.recv().await {
        if event.binding_id == fotonvoice_routing::TTS_STOP_BINDING_ID {
            if event.kind == GestureKind::Start {
                if let Some(tts) = state_for_gesture.tts_handle.lock().await.as_ref() {
                    tts.stop();
                }
            }
            continue;
        }

        match event.kind {
            GestureKind::Start => {
                if state_for_gesture.is_hotkeys_inhibited() {
                    tracing::debug!(
                        "Hotkey '{}' gesture suppressed: keybind recorder is active",
                        event.binding_id
                    );
                    continue;
                }

                *state_for_gesture.active_target.lock().await = event.target_id.clone();
                *state_for_gesture.active_binding_label.lock().await =
                    event.binding_label.clone();
                *state_for_gesture.active_binding_id.lock().await = event.binding_id.clone();
                state_for_gesture.begin_recording().await;

                let preload_state = state_for_gesture.clone();
                let preload_target = event.target_id.clone();
                tokio::spawn(async move {
                    preload_state.preload_tts_for_target(&preload_target).await;
                });

                if let Some(msg) = setup_blocker(&state_for_gesture).await {
                    let now = std::time::Instant::now();
                    let stale = {
                        let last = last_setup_notice.lock().await;
                        last.map(|t: std::time::Instant| {
                            now.duration_since(t) > SETUP_NOTICE_INTERVAL
                        })
                        .unwrap_or(true)
                    };
                    if stale {
                        *last_setup_notice.lock().await = Some(now);
                        fotonvoice_inject::show_notification("FotonVoice Engine - setup unfinished", &msg);
                        show_setup_window();
                    }
                }

                let st = state_for_gesture.clone();
                let mic_shown = mic_notice_shown.clone();
                tokio::spawn(async move {
                    tokio::time::sleep(Duration::from_secs(2)).await;
                    if st.is_recording()
                        && !st.is_audio_ready()
                        && !mic_shown.swap(true, Ordering::SeqCst)
                    {
                        fotonvoice_inject::show_notification(
                            "FotonVoice Engine",
                            "Recording is active but no microphone audio is arriving. Check the input device in Settings -> Audio.",
                        );
                    }
                });
            }
            GestureKind::Stop => {
                state_for_gesture.set_recording(false);
            }
        }
    }
}

#[cfg(target_os = "linux")]
pub fn spawn_setup_watcher(
    app_handle: tauri::AppHandle,
    health: Arc<fotonvoice_hotkeys::ListenerHealth>,
) {
    tauri::async_runtime::spawn(async move {
        tokio::time::sleep(Duration::from_millis(500)).await;

        let mut first_pass = true;
        let mut was_active: Option<bool> = None;
        let mut last_alert: Option<std::time::Instant> = None;
        loop {
            let active = health.is_active();

            if active && was_active == Some(false) {
                fotonvoice_inject::show_notification(
                    "FotonVoice Engine",
                    "Your global shortcuts are registered and working now.",
                );
            }

            if !active {
                let due = last_alert
                    .map(|t| t.elapsed() >= BLIND_ALERT_INTERVAL)
                    .unwrap_or(true);
                if due {
                    last_alert = Some(std::time::Instant::now());
                    if first_pass {
                        show_setup_window();
                    } else {
                        fotonvoice_inject::show_notification(
                            "FotonVoice Engine - global shortcuts unavailable",
                            "Nothing on this desktop can deliver FotonVoice Engine's \
                             shortcuts, so pressing them does nothing. Open \
                             FotonVoice Engine to see why.",
                        );
                    }
                }
            } else if first_pass && crate::commands::missing_injection_tool().is_some() {
                show_setup_window();
            }

            if was_active != Some(active) {
                was_active = Some(active);
                let _ = app_handle.emit("setup-status-changed", active);
                update_tray_for_setup(&app_handle, active);
            }

            first_pass = false;
            tokio::time::sleep(SETUP_POLL_INTERVAL).await;
        }
    });
}

pub fn spawn_audio_level_forwarder(
    handle: tauri::AppHandle,
    state_for_audio_level: Arc<AppState>,
    audio_level_rx: crossbeam_channel::Receiver<f32>,
) {
    const MIN_INTERVAL: Duration = Duration::from_millis(16);

    const TOPMOST_REASSERT_INTERVAL: Duration = Duration::from_secs(1);

    std::thread::spawn(move || {
        let mut last_sent = std::time::Instant::now() - MIN_INTERVAL;
        let mut last_topmost_reassert = std::time::Instant::now() - TOPMOST_REASSERT_INTERVAL;
        while let Ok(mut level) = audio_level_rx.recv() {
            while let Ok(newer) = audio_level_rx.try_recv() {
                level = newer;
            }
            let now = std::time::Instant::now();
            if now.duration_since(last_sent) < MIN_INTERVAL {
                continue;
            }
            last_sent = now;

            let _ = handle.emit("audio-level", level);

            let overlay_on = state_for_audio_level.is_overlay_enabled();
            let is_recording = overlay_on && state_for_audio_level.is_recording();
            let is_processing = overlay_on && state_for_audio_level.is_processing();
            let is_speaking = overlay_on && state_for_audio_level.is_speaking();

            if (is_recording || is_processing || is_speaking)
                && now.duration_since(last_topmost_reassert) >= TOPMOST_REASSERT_INTERVAL
            {
                last_topmost_reassert = now;
                crate::window::reassert_overlay_topmost(&handle);
            }
        }
    });
}
