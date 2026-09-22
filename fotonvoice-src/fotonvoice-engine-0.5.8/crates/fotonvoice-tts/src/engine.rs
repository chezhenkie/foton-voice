//! The utterance queue, worker thread, and Piper/eSpeak synthesis. Pocket-TTS

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use crossbeam_channel::{bounded, Receiver, RecvTimeoutError, Sender};
use tracing::{debug, info, warn};
use fotonvoice_config::{TtsConfig, TtsEngine};
use fotonvoice_text::{correct_custom_vocabulary, expand_snippets};

use crate::audiocpp::AudioCppSession;
use crate::breeze::{ensure_breeze_tts_2_loaded, speak_breeze_tts_2};
use crate::inflect::{ensure_inflect_micro_loaded, speak_inflect_micro};
    use crate::piper::{get_voice_path, piper_binary, sample_rate_for_voice_path};
use crate::pocket::{ensure_pocket_tts_loaded, speak_pocket_tts};
use crate::voxcpm::{ensure_vox_cpm_2_loaded, speak_vox_cpm_2};

/// How long the worker parks in `recv_timeout` when there is nothing to expire.
const IDLE_PARK: Duration = Duration::from_secs(3600);

/// The worker's cached Inflect-Micro-v2 sessions. Without the `inflect-micro`
#[cfg(feature = "inflect-micro")]
type InflectModelSlot = Option<crate::inflect::model::InflectModel>;
#[cfg(not(feature = "inflect-micro"))]
type InflectModelSlot = Option<()>;

#[cfg(feature = "luxtts")]
type LuxModelSlot = Option<crate::luxtts::model::LuxTTSModel>;
#[cfg(not(feature = "luxtts"))]
type LuxModelSlot = Option<()>;


#[derive(Debug, Clone)]
pub struct Utterance {
    pub text: String,
    pub voice: Option<String>,
    pub source_label: Option<String>,
}

#[derive(Debug, Clone)]
pub enum TtsCommand {
    Play {
        utterance: Utterance,
        generation: u32,
    },
    /// Live config swap - also how the memory policy reaches a running worker,
    UpdateConfig(TtsConfig),
    /// Load the model now (and restart the idle countdown) without speaking.
    Preload,
    Shutdown,
}

static ACTIVE_SINK: std::sync::Mutex<Option<std::sync::Arc<rodio::Sink>>> = std::sync::Mutex::new(None);

pub fn stop_current_playback() {
    let mut guard = ACTIVE_SINK.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(ref sink) = *guard {
        let _ = sink.stop();
    }
    *guard = None;
}

#[derive(Clone)]
pub struct TtsEngineHandle {
    tx: Sender<TtsCommand>,
    generation: Arc<AtomicU32>,
    /// Mirrors whether the worker currently holds a model in memory, so the UI
    model_loaded: Arc<AtomicBool>,
}

impl TtsEngineHandle {
    pub fn speak(&self, text: impl Into<String>) {
        let gen = self.generation.load(std::sync::atomic::Ordering::SeqCst);
        let _ = self.tx.send(TtsCommand::Play {
            utterance: Utterance {
                text: text.into(),
                voice: None,
                source_label: None,
            },
            generation: gen,
        });
    }

    pub fn speak_utterance(&self, u: Utterance) {
        let gen = self.generation.load(std::sync::atomic::Ordering::SeqCst);
        let _ = self.tx.send(TtsCommand::Play {
            utterance: u,
            generation: gen,
        });
    }

    /// Ask the worker to load the model now. Cheap and idempotent when the model
    pub fn preload(&self) {
        let _ = self.tx.try_send(TtsCommand::Preload);
    }

    /// Whether a model is currently resident in memory.
    pub fn is_model_loaded(&self) -> bool {
        self.model_loaded.load(Ordering::SeqCst)
    }

    pub fn stop(&self) {
        self.generation.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        stop_current_playback();
    }

    pub fn update_config(&self, config: TtsConfig) {
        let _ = self.tx.send(TtsCommand::UpdateConfig(config));
    }

    pub fn shutdown(&self) {
        let _ = self.tx.send(TtsCommand::Shutdown);
    }
}


pub type PlaybackCallback = Arc<dyn Fn() + Send + Sync + 'static>;
/// Called with a human-readable message whenever an utterance fails to play
pub type ErrorCallback = Arc<dyn Fn(String) + Send + Sync + 'static>;

pub struct TtsEngineWorker {
    config: TtsConfig,
    custom_vocabulary: Vec<String>,
    rx: Receiver<TtsCommand>,
    generation: Arc<AtomicU32>,
    model_loaded: Arc<AtomicBool>,
    on_playback_start: Option<PlaybackCallback>,
    on_playback_end: Option<PlaybackCallback>,
    on_error: Option<ErrorCallback>,
}

impl TtsEngineWorker {
    pub fn start(
        config: TtsConfig,
        custom_vocabulary: Vec<String>,
        on_playback_start: Option<PlaybackCallback>,
        on_playback_end: Option<PlaybackCallback>,
        on_error: Option<ErrorCallback>,
    ) -> TtsEngineHandle {
        let (tx, rx) = bounded(32);
        let generation = Arc::new(AtomicU32::new(0));
        let model_loaded = Arc::new(AtomicBool::new(false));
        let handle = TtsEngineHandle {
            tx,
            generation: generation.clone(),
            model_loaded: model_loaded.clone(),
        };

        let prewarm = match config.engine {
            TtsEngine::PocketTts => config.pocket_tts.prewarm,
            TtsEngine::InflectMicro => config.inflect_micro.prewarm,
            TtsEngine::BreezeTts2 => config.breeze_tts_2.prewarm,
            TtsEngine::VoxCpm2 => config.vox_cpm_2.prewarm,
            TtsEngine::LuxTts => config.lux_tts.prewarm,
            _ => false,
        };
        if prewarm && !config.unloads_when_idle() {
            let _ = handle.tx.send(TtsCommand::Play {
                utterance: Utterance {
                    text: " ".into(),
                    voice: None,
                    source_label: Some("prewarm".into()),
                },
                generation: 0,
            });
        }

        let worker = Self {
            config,
            custom_vocabulary,
            rx,
            generation,
            model_loaded,
            on_playback_start,
            on_playback_end,
            on_error,
        };
        std::thread::Builder::new()
            .name("fotonvoice-tts".into())
            .spawn(move || worker.run())
            .expect("spawn tts thread");

        handle
    }

    fn run(self) {
        info!(
            "TTS engine started (engine={:?}, memory_mode={:?})",
            self.config.engine, self.config.memory_mode
        );
        let mut current_config = self.config.clone();

        let mut inflect_model: InflectModelSlot = None;
        let mut luxtts_model: LuxModelSlot = None;
        let mut audiocpp_session: Option<AudioCppSession> = None;
        let mut piper_resident: Option<crate::piper::PiperResident> = None;

        let mut audio_context: Option<(rodio::OutputStream, rodio::OutputStreamHandle, Arc<rodio::Sink>)> = None;

        let mut snips_cache: Option<std::collections::HashMap<String, String>> = None;

        let init_audio = |ctx: &mut Option<(rodio::OutputStream, rodio::OutputStreamHandle, Arc<rodio::Sink>)>| -> Result<Arc<rodio::Sink>> {
            if let Some((_, _, ref sink)) = ctx {
                return Ok(sink.clone());
            }
            let (stream, handle) = rodio::OutputStream::try_default()
                .map_err(|e| anyhow::anyhow!("audio output device: {e}"))?;
            let sink = Arc::new(rodio::Sink::try_new(&handle)
                .map_err(|e| anyhow::anyhow!("audio sink: {e}"))?);
            *ctx = Some((stream, handle, sink.clone()));
            Ok(sink)
        };

        let mut last_used = Instant::now();

        loop {
            let unload_when_idle = current_config.unloads_when_idle();
            let idle = current_config.idle_unload_duration();
            let model_resident = inflect_model.is_some()
                || audiocpp_session.is_some()
                || luxtts_model.is_some()
                || piper_resident.is_some();

            let wait = if unload_when_idle && model_resident {
                match idle.checked_sub(last_used.elapsed()) {
                    Some(remaining) if !remaining.is_zero() => remaining,
                    _ => {
                        info!(
                            "Unloading TTS model after {}s idle (memory mode: on-demand)",
                            idle.as_secs()
                        );
                        inflect_model = None;
                        audiocpp_session = None;
                        luxtts_model = None;
                        piper_resident = None;
                        self.model_loaded.store(false, Ordering::SeqCst);
                        continue;
                    }
                }
            } else {
                IDLE_PARK
            };

            let cmd = match self.rx.recv_timeout(wait) {
                Ok(cmd) => cmd,
                Err(RecvTimeoutError::Timeout) => continue,
                Err(RecvTimeoutError::Disconnected) => break,
            };

            match cmd {
                TtsCommand::UpdateConfig(new_cfg) => {
                    info!(
                        "TTS worker config dynamically updated (engine={:?}, memory_mode={:?})",
                        new_cfg.engine, new_cfg.memory_mode
                    );
                    current_config = new_cfg;
                    snips_cache = None;
                    last_used = Instant::now();
                }
                TtsCommand::Play { mut utterance, generation } => {
                    last_used = Instant::now();
                    let current_gen = self.generation.load(std::sync::atomic::Ordering::SeqCst);
                    if generation < current_gen {
                        debug!("Discarding stale utterance: generation={generation} (current={current_gen})");
                        continue;
                    }

                    let is_prewarm = utterance.source_label.as_deref() == Some("prewarm");

                    if !is_prewarm {
                        if snips_cache.is_none() {
                            let mut snips = current_config.snippets.clone();
                            snips.entry("FotonVoice Engine".to_string()).or_insert_with(|| "Foton Voice".to_string());
                            snips.entry("fotonvoice-engine".to_string()).or_insert_with(|| "Foton Voice".to_string());
                            snips.entry("Vox Control".to_string()).or_insert_with(|| "Foton Voice".to_string());
                            snips.entry("vox control".to_string()).or_insert_with(|| "Foton Voice".to_string());
                            snips_cache = Some(snips);
                        }
                        // Built directly above in this branch, so always present.
                        if let Some(snips) = snips_cache.as_ref() {
                            utterance.text = expand_snippets(&utterance.text, snips);
                        }
                        if !self.custom_vocabulary.is_empty() {
                            utterance.text = correct_custom_vocabulary(&utterance.text, &self.custom_vocabulary);
                        }
                    }

                    let sink = match init_audio(&mut audio_context) {
                        Ok(s) => s,
                        Err(e) => {
                            warn!("TTS audio init error: {e}");
                            if !is_prewarm {
                                if let Some(ref cb) = self.on_error {
                                    cb(format!("Audio output unavailable: {e}"));
                                }
                            }
                            continue;
                        }
                    };
                    let sink_speed = match current_config.engine {
                        TtsEngine::Piper
                        | TtsEngine::InflectMicro
                        | TtsEngine::LuxTts
                        | TtsEngine::Espeak => 1.0,
                        TtsEngine::PocketTts
                        | TtsEngine::BreezeTts2
                        | TtsEngine::VoxCpm2 => {
                            if current_config.speed <= 0.0 { 1.0 } else { current_config.speed }
                        }
                    };
                    sink.set_speed(sink_speed.clamp(0.5, 2.5));

                    {
                        let mut guard = ACTIVE_SINK.lock().unwrap_or_else(|e| e.into_inner());
                        *guard = Some(sink.clone());
                    }

                    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        match current_config.engine {
                        TtsEngine::Piper => self.speak_piper(&utterance, &sink, &mut piper_resident),
                        TtsEngine::Espeak => self.speak_espeak(&utterance),
                        TtsEngine::PocketTts => speak_pocket_tts(
                            &current_config,
                            &utterance,
                            &mut audiocpp_session,
                            &self.on_playback_start,
                            &sink,
                            &self.generation,
                            generation,
                        ),
                        TtsEngine::InflectMicro => speak_inflect_micro(
                            &current_config,
                            &utterance,
                            &mut inflect_model,
                            &self.on_playback_start,
                            &sink,
                            &self.generation,
                            generation,
                        ),
                        TtsEngine::BreezeTts2 => speak_breeze_tts_2(
                            &current_config,
                            &utterance,
                            &mut audiocpp_session,
                            &self.on_playback_start,
                            &sink,
                            &self.generation,
                            generation,
                        ),
                        TtsEngine::VoxCpm2 => speak_vox_cpm_2(
                            &current_config,
                            &utterance,
                            &mut audiocpp_session,
                            &self.on_playback_start,
                            &sink,
                            &self.generation,
                            generation,
                        ),
                        TtsEngine::LuxTts => crate::luxtts::speak_lux_tts(
                            &current_config,
                            &utterance,
                            &mut luxtts_model,
                            &self.on_playback_start,
                            &sink,
                            &self.generation,
                            generation,
                        ),
                        }
                    }))
                    .unwrap_or_else(|payload| {
                        let detail = payload
                            .downcast_ref::<&str>()
                            .map(|s| (*s).to_string())
                            .or_else(|| payload.downcast_ref::<String>().cloned())
                            .unwrap_or_else(|| "unknown panic".into());
                        Err(anyhow::anyhow!(
                            "TTS engine panicked: {detail}. For the Inflect-Micro-v2 \
                             engine this usually means ONNX Runtime could not be \
                             loaded - this build resolves libonnxruntime at runtime."
                        ))
                    });

                    {
                        let mut guard = ACTIVE_SINK.lock().unwrap_or_else(|e| e.into_inner());
                        *guard = None;
                    }

                    if let Err(e) = result {
                        warn!("TTS speak error: {e:#}");
                        if !is_prewarm {
                            if let Some(ref cb) = self.on_error {
                                cb(format!("{e:#}"));
                            }
                        }
                    }
                    if !is_prewarm {
                        if let Some(ref cb) = self.on_playback_end {
                            cb();
                        }
                    }

                    self.model_loaded.store(
                        inflect_model.is_some()
                            || audiocpp_session.is_some()
                            || luxtts_model.is_some()
                            || piper_resident.is_some(),
                        Ordering::SeqCst,
                    );
                    last_used = Instant::now();
                }
                TtsCommand::Preload => {
                    let outcome = match current_config.engine {
                        TtsEngine::InflectMicro if inflect_model.is_none() => {
                            Some(ensure_inflect_micro_loaded(&current_config, &mut inflect_model))
                        }
                        TtsEngine::PocketTts if audiocpp_session.is_none() => {
                            Some(ensure_pocket_tts_loaded(&current_config, &mut audiocpp_session))
                        }
                        TtsEngine::BreezeTts2 if audiocpp_session.is_none() => {
                            Some(ensure_breeze_tts_2_loaded(&current_config, &mut audiocpp_session))
                        }
                        TtsEngine::VoxCpm2 if audiocpp_session.is_none() => {
                            Some(ensure_vox_cpm_2_loaded(&current_config, &mut audiocpp_session))
                        }
                        TtsEngine::LuxTts if luxtts_model.is_none() => {
                            Some(crate::luxtts::ensure_lux_tts_loaded(&current_config, &mut luxtts_model))
                        }
                        _ => None,
                    };
                    match outcome {
                        Some(Err(e)) => debug!("TTS preload skipped: {e:#}"),
                        Some(Ok(())) => debug!("TTS model pre-loaded and primed"),
                        None => {}
                    }
                    self.model_loaded.store(
                        inflect_model.is_some()
                            || audiocpp_session.is_some()
                            || luxtts_model.is_some()
                            || piper_resident.is_some(),
                        Ordering::SeqCst,
                    );
                    last_used = Instant::now();
                }
                TtsCommand::Shutdown => {
                    debug!("TTS shutdown signal received");
                    stop_current_playback();
                    break;
                }
            }
        }
    }

    fn speak_piper(
        &self,
        u: &Utterance,
        sink: &rodio::Sink,
        resident: &mut Option<crate::piper::PiperResident>,
    ) -> Result<()> {
        let binary = piper_binary().ok_or_else(|| {
            anyhow::anyhow!(
                "Piper binary not found. Download a voice from TTS settings (this \
                 also installs the standalone Piper engine), or install piper \
                 system-wide."
            )
        })?;
        let voice_name = u.voice.as_deref().unwrap_or(&self.config.voice);

        let voice_path =
            get_voice_path(voice_name, &self.config.voice_dir).ok_or_else(|| {
                anyhow::anyhow!("Piper voice files not found for: {}", voice_name)
            })?;

        let length_scale = 1.0 / self.config.speed;
        let fingerprint = crate::piper::PiperFingerprint {
            voice_path: voice_path.clone(),
            gpu: self.config.gpu,
            length_scale: length_scale.to_string(),
        };

        let stale = match resident {
            Some(r) => {
                r.fingerprint() != &fingerprint
                    || r.has_exited()
            }
            None => true,
        };
        if stale {
            *resident = None;
            *resident = Some(
                crate::piper::spawn_piper_resident(&binary, fingerprint)
                    .context("start resident piper")?,
            );
        }
        // Just spawned above, so always present.
        let Some(piper) = resident.as_mut() else {
            anyhow::bail!("resident piper slot missing");
        };
        piper.drain_pending();

        let line = u.text.replace("\r\n", " ").replace('\n', " ");
        piper.write_line(&line).context("write to piper stdin")?;

        let current_gen = self.generation.load(Ordering::SeqCst);

        let sample_rate = sample_rate_for_voice_path(voice_name, &voice_path);
        let mut played_any = false;
        let quiet_done = Duration::from_millis(300);
        let mut quiet = Duration::ZERO;
        let mut first_chunk_deadline = Duration::from_secs(30);
        loop {
            match piper.recv_chunk(Duration::from_millis(50)) {
                Some(bytes) => {
                    if !played_any {
                        if u.source_label.as_deref() != Some("prewarm") {
                            if let Some(ref cb) = self.on_playback_start {
                                cb();
                            }
                        }
                        played_any = true;
                    }
                    if self.generation.load(Ordering::SeqCst) != current_gen {
                        warn!("Piper utterance cancelled by stop(); restarting resident process");
                        *resident = None;
                        stop_current_playback();
                        return Ok(());
                    }
                    let samples: Vec<i16> = bytes
                        .chunks_exact(2)
                        .map(|b| i16::from_le_bytes([b[0], b[1]]))
                        .collect();
                    if !samples.is_empty() {
                        sink.append(rodio::buffer::SamplesBuffer::new(1, sample_rate, samples));
                    }
                    quiet = Duration::ZERO;
                }
                None => {
                    quiet += Duration::from_millis(50);
                    if !played_any {
                        if piper.has_exited() {
                            anyhow::bail!("piper exited before producing audio: {}", piper.stderr_tail());
                        }
                        first_chunk_deadline -= Duration::from_millis(50);
                        if first_chunk_deadline.is_zero() {
                            anyhow::bail!(
                                "piper produced no audio within 30s: {}",
                                piper.stderr_tail()
                            );
                        }
                    } else if quiet >= quiet_done {
                        break;
                    }
                }
            }
        }

        sink.sleep_until_end();
        Ok(())
    }

    fn speak_espeak(&self, u: &Utterance) -> Result<()> {
        if fotonvoice_config::find_in_path("espeak-ng").is_none() {
            anyhow::bail!(
                "espeak-ng is not installed on this system. Install it with your \
                 package manager (e.g. `sudo pacman -S espeak-ng` or `sudo apt \
                 install espeak-ng`) or switch to another TTS engine."
            );
        }

        if u.source_label.as_deref() != Some("prewarm") {
            if let Some(ref cb) = self.on_playback_start {
                cb();
            }
        }

        let wpm = (175.0 * self.config.speed) as i32;
        let status = std::process::Command::new("espeak-ng")
            .arg("-s")
            .arg(wpm.to_string())
            .arg(&u.text)
            .status()
            .context("spawn espeak-ng")?;
        if !status.success() {
            anyhow::bail!("espeak-ng exited with status {:?}", status.code());
        }
        Ok(())
    }
}


#[cfg(test)]
mod tests {
    use super::*;


    #[test]
    fn test_handle_stop_increments_generation_counter() {
        let (tx, _rx) = bounded(32);
        let generation = Arc::new(std::sync::atomic::AtomicU32::new(0));
        let handle = TtsEngineHandle {
            tx,
            generation: generation.clone(),
            model_loaded: Arc::new(AtomicBool::new(false)),
        };

        assert_eq!(generation.load(std::sync::atomic::Ordering::SeqCst), 0);
        handle.stop();
        assert_eq!(generation.load(std::sync::atomic::Ordering::SeqCst), 1);
    }

    #[test]
    fn test_raw_stop_current_playback_does_not_bump_generation() {
        let generation = Arc::new(std::sync::atomic::AtomicU32::new(0));
        stop_current_playback();
        assert_eq!(generation.load(std::sync::atomic::Ordering::SeqCst), 0);
    }

    #[test]
    fn test_streaming_loop_cancellation_check_breaks_on_stale_generation() {
        let generation_counter = Arc::new(std::sync::atomic::AtomicU32::new(0));
        let snapshotted_generation = 0u32;

        let mut frames_processed = 0;
        for _ in 0..5 {
            if generation_counter.load(std::sync::atomic::Ordering::SeqCst) != snapshotted_generation {
                break;
            }
            frames_processed += 1;
            if frames_processed == 2 {
                generation_counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            }
        }

        assert_eq!(frames_processed, 2, "loop must abandon remaining frames after stop()");
    }


    fn test_handle() -> (TtsEngineHandle, Receiver<TtsCommand>) {
        let (tx, rx) = bounded(32);
        let handle = TtsEngineHandle {
            tx,
            generation: Arc::new(AtomicU32::new(0)),
            model_loaded: Arc::new(AtomicBool::new(false)),
        };
        (handle, rx)
    }

    #[test]
    fn test_preload_sends_preload_command() {
        let (handle, rx) = test_handle();
        handle.preload();
        assert!(matches!(rx.try_recv(), Ok(TtsCommand::Preload)));
    }

    #[test]
    fn test_update_config_carries_memory_policy_to_worker() {
        let (handle, rx) = test_handle();
        let cfg = TtsConfig {
            memory_mode: fotonvoice_config::TtsMemoryMode::OnDemand,
            idle_unload_secs: 900,
            ..TtsConfig::default()
        };
        handle.update_config(cfg);
        match rx.try_recv() {
            Ok(TtsCommand::UpdateConfig(cfg)) => {
                assert!(cfg.unloads_when_idle());
                assert_eq!(cfg.idle_unload_duration(), Duration::from_secs(900));
            }
            other => panic!("expected UpdateConfig, got {other:?}"),
        }
    }

    #[test]
    fn test_model_loaded_flag_defaults_false() {
        let (handle, _rx) = test_handle();
        assert!(!handle.is_model_loaded());
    }

    #[test]
    fn test_idle_window_expiry_arithmetic() {
        let idle = Duration::from_secs(900);

        let fresh = Duration::from_secs(10);
        assert_eq!(idle.checked_sub(fresh), Some(Duration::from_secs(890)));

        let expired = Duration::from_secs(901);
        assert_eq!(idle.checked_sub(expired), None);

        let boundary = idle.checked_sub(idle).unwrap();
        assert!(boundary.is_zero());
    }

    #[test]
    fn test_use_resets_idle_countdown() {
        let idle = Duration::from_secs(900);
        let first_use = Instant::now();
        let elapsed_at_second_use = Duration::from_secs(600);

        assert_eq!(idle.checked_sub(elapsed_at_second_use), Some(Duration::from_secs(300)));

        let last_used = first_use + elapsed_at_second_use;
        let remaining = idle.checked_sub(last_used.duration_since(last_used)).unwrap();
        assert_eq!(remaining, idle);
    }

    #[test]
    fn test_config_idle_duration_defaults_to_fifteen_minutes() {
        let cfg = TtsConfig::default();
        assert_eq!(cfg.idle_unload_secs, 900);
        assert_eq!(cfg.idle_unload_duration(), Duration::from_secs(900));
    }

    #[test]
    fn test_config_defaults_to_always_loaded() {
        assert!(!TtsConfig::default().unloads_when_idle());
    }

    #[test]
    fn test_config_idle_duration_is_floored() {
        let cfg = TtsConfig { idle_unload_secs: 0, ..TtsConfig::default() };
        assert_eq!(cfg.idle_unload_duration(), Duration::from_secs(30));
    }

    #[test]
    fn test_on_demand_mode_reported_by_config() {
        let cfg = TtsConfig {
            memory_mode: fotonvoice_config::TtsMemoryMode::OnDemand,
            ..TtsConfig::default()
        };
        assert!(cfg.unloads_when_idle());
    }
}
