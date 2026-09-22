use std::{
    collections::VecDeque,
    sync::{
        atomic::{AtomicBool, AtomicU32, Ordering},
        Arc,
    },
    time::Duration,
};

use anyhow::{Context, Result};
use cpal::{
    traits::{DeviceTrait, HostTrait, StreamTrait},
    SampleRate, StreamConfig,
};
use crossbeam_channel::{Receiver, Sender};
use tracing::{info, warn};
use fotonvoice_config::AudioConfig;

mod denoise;

pub const TARGET_SAMPLE_RATE: u32 = 16_000;

/// Below this peak amplitude, a buffer is treated as silence when deciding
const SILENCE_PEAK_EPSILON: f32 = 1e-4;

/// Whether every sample in `data` falls within the noise floor a denied or
fn is_silent(data: &[f32]) -> bool {
    data.iter().all(|&s| s.abs() < SILENCE_PEAK_EPSILON)
}

/// How much of the audio from just *before* a recording started is kept and
pub const PREROLL_MS: u32 = 300;

/// A chunk of mono f32 audio at TARGET_SAMPLE_RATE Hz.
pub type AudioChunk = Vec<f32>;

pub struct AudioRecorder {
    config: AudioConfig,
    /// Currently recording (pushed to inference queue)
    recording: Arc<AtomicBool>,
    /// Currently monitoring (active settings tab VU meter level feed)
    monitoring: Arc<AtomicBool>,
    /// Live sync dynamic stream preference
    dynamic_stream: Arc<AtomicBool>,
    /// Live input device index, mapped to u32::MAX when None (default system device)
    input_device_index: Arc<AtomicU32>,
    /// Live gain value, stored as f32 bits
    gain: Arc<AtomicU32>,
    /// Live noise-suppression preference
    noise_suppression: Arc<AtomicBool>,
    /// Signalled when a flag this loop watches changes, so it reacts at once
    wake: Option<Receiver<()>>,
}

impl AudioRecorder {
    pub fn new(
        config: AudioConfig,
        recording: Arc<AtomicBool>,
        monitoring: Arc<AtomicBool>,
        dynamic_stream: Arc<AtomicBool>,
        input_device_index: Arc<AtomicU32>,
        gain: Arc<AtomicU32>,
        noise_suppression: Arc<AtomicBool>,
    ) -> Self {
        Self {
            config,
            recording,
            monitoring,
            dynamic_stream,
            input_device_index,
            gain,
            noise_suppression,
            wake: None,
        }
    }

    /// Wake the capture supervisor the moment a flag changes, rather than
    pub fn with_wake(mut self, wake: Receiver<()>) -> Self {
        self.wake = Some(wake);
        self
    }

    pub fn start_recording(&self) {
        self.recording.store(true, Ordering::SeqCst);
    }

    pub fn stop_recording(&self) {
        self.recording.store(false, Ordering::SeqCst);
    }

    pub fn is_recording(&self) -> bool {
        self.recording.load(Ordering::SeqCst)
    }

    /// Spawn the audio capture task. Returns a handle to stop it.
    pub fn run(
        self,
        tx: Sender<AudioChunk>,
        level_tx: Option<Sender<f32>>,
        audio_ready: Option<Arc<AtomicBool>>,
    ) -> Result<RecorderHandle> {
        let recording = self.recording.clone();
        let monitoring = self.monitoring.clone();
        let dynamic_stream = self.dynamic_stream.clone();
        let input_device_index = self.input_device_index.clone();
        let gain = self.gain.clone();
        let noise_suppression = self.noise_suppression.clone();
        let wake = self.wake.clone();
        let cfg = self.config.clone();

        let handle = std::thread::Builder::new()
            .name("fotonvoice-audio".into())
            .spawn(move || {
                if let Err(e) = capture_loop(cfg, gain, noise_suppression, recording, monitoring, dynamic_stream, input_device_index, audio_ready, tx, level_tx, wake) {
                    warn!("Audio capture error: {e}");
                }
            })
            .context("spawn audio thread")?;

        Ok(RecorderHandle { _thread: handle })
    }
}

pub struct RecorderHandle {
    _thread: std::thread::JoinHandle<()>,
}


pub fn test_and_detect_active_device(idx_opt: Option<u32>) -> Result<cpal::Device> {
    let host = cpal::default_host();
    
    if let Some(idx) = idx_opt {
        if let Ok(mut devices) = host.input_devices() {
            if let Some(device) = devices.nth(idx as usize) {
                if let Ok(config) = negotiate_config(&device) {
                    if build_and_start_test_stream(&device, &config) {
                        info!("Startup test: Configured device index {} ({}) is active and functional.", idx, device.name().unwrap_or_default());
                        return Ok(device);
                    } else {
                        warn!("Startup test: Configured device index {} failed test stream build.", idx);
                    }
                }
            }
        }
    }

    if let Some(device) = host.default_input_device() {
        if let Ok(config) = negotiate_config(&device) {
            if build_and_start_test_stream(&device, &config) {
                info!("Startup test: Default input device ({}) is active and functional.", device.name().unwrap_or_default());
                return Ok(device);
            } else {
                warn!("Startup test: Default input device failed test stream build.");
            }
        }
    }

    if let Ok(devices) = host.input_devices() {
        for (idx, device) in devices.enumerate() {
            if let Ok(config) = negotiate_config(&device) {
                if build_and_start_test_stream(&device, &config) {
                    info!("Startup test: Fallback device index {} ({}) is active and functional.", idx, device.name().unwrap_or_default());
                    return Ok(device);
                }
            }
        }
    }

    host.default_input_device().context("no active or functional input device found at startup")
}

/// Build a throwaway input stream on `device` and briefly start it, then stop
fn build_and_start_test_stream(device: &cpal::Device, config: &StreamConfig) -> bool {
    let Ok(stream) = device.build_input_stream(config, |_: &[f32], _| {}, |_| {}, None) else {
        return false;
    };
    if stream.play().is_err() {
        return false;
    }
    let _ = stream.pause();
    true
}


/// What one captured buffer produced.
#[derive(Debug)]
struct Captured {
    /// RMS for the VU meter and the overlay, when either is watching.
    level: Option<f32>,
    /// Audio for inference, when recording.
    chunk: Option<AudioChunk>,
}

/// The per-stream processing chain: gain, pre-roll, noise suppression and
struct CaptureProcessor {
    gain: Arc<AtomicU32>,
    recording: Arc<AtomicBool>,
    monitoring: Arc<AtomicBool>,
    noise_suppression: Arc<AtomicBool>,
    hw_rate: u32,
    needs_resample: bool,
    /// Built on the first buffer that actually needs it, so a stream that runs
    denoiser: Option<denoise::Denoiser>,
    denoising: bool,
    /// Reused across buffers so the gain stage never allocates on the audio
    gained: Vec<f32>,
    /// The tail of what the microphone heard before recording began, at the
    preroll: VecDeque<f32>,
    preroll_cap: usize,
    was_recording: bool,
    /// Set once this stream has delivered a buffer that is not silence.
    audio_ready: Option<Arc<AtomicBool>>,
}

impl CaptureProcessor {
    fn new(
        gain: Arc<AtomicU32>,
        recording: Arc<AtomicBool>,
        monitoring: Arc<AtomicBool>,
        noise_suppression: Arc<AtomicBool>,
        hw_rate: u32,
        needs_resample: bool,
        audio_ready: Option<Arc<AtomicBool>>,
    ) -> Self {
        let preroll_cap = (hw_rate as usize * PREROLL_MS as usize) / 1000;
        Self {
            gain,
            recording,
            monitoring,
            noise_suppression,
            hw_rate,
            needs_resample,
            denoiser: None,
            denoising: false,
            gained: Vec::new(),
            preroll: VecDeque::with_capacity(preroll_cap + 1),
            preroll_cap,
            was_recording: false,
            audio_ready,
        }
    }

    fn feed(&mut self, data: &[f32]) -> Captured {
        let is_recording = self.recording.load(Ordering::Relaxed);
        let current_gain = f32::from_bits(self.gain.load(Ordering::Relaxed));

        if let Some(ref ready) = self.audio_ready {
            if !ready.load(Ordering::Relaxed) && !is_silent(data) {
                ready.store(true, Ordering::Relaxed);
            }
        }

        let level = (is_recording || self.monitoring.load(Ordering::Relaxed))
            .then(|| rms(data) * current_gain);

        if !is_recording {
            self.was_recording = false;
            self.preroll.extend(data.iter().copied());
            let excess = self.preroll.len().saturating_sub(self.preroll_cap);
            self.preroll.drain(..excess);
            return Captured { level, chunk: None };
        }

        let wants_denoise = self.noise_suppression.load(Ordering::Relaxed);
        if wants_denoise != self.denoising {
            self.denoiser = denoise::make_denoiser(wants_denoise, self.hw_rate);
            self.denoising = wants_denoise;
        }

        let opening = !self.was_recording;
        self.was_recording = true;
        let carries_preroll = opening && !self.preroll.is_empty();

        let processed = if self.denoiser.is_some() || self.needs_resample || carries_preroll {
            self.gained.clear();
            self.gained.reserve(self.preroll.len() + data.len());
            self.gained
                .extend(self.preroll.drain(..).map(|s| s * current_gain));
            self.gained
                .extend(data.iter().map(|&s| s * current_gain));
            match self.denoiser {
                Some(ref mut d) => d.process(&self.gained),
                None if self.needs_resample => {
                    resample_chunk(&self.gained, self.hw_rate, TARGET_SAMPLE_RATE)
                }
                None => self.gained.clone(),
            }
        } else {
            data.iter().map(|&s| s * current_gain).collect()
        };

        Captured {
            level,
            chunk: (!processed.is_empty()).then_some(processed),
        }
    }
}


/// Build the data callback cpal invokes for each captured buffer.
#[allow(clippy::too_many_arguments)]
fn make_input_callback(
    gain: Arc<AtomicU32>,
    recording: Arc<AtomicBool>,
    monitoring: Arc<AtomicBool>,
    level_tx: Option<Sender<f32>>,
    tx: Sender<AudioChunk>,
    noise_suppression: Arc<AtomicBool>,
    hw_rate: u32,
    needs_resample: bool,
    audio_ready: Option<Arc<AtomicBool>>,
) -> impl FnMut(&[f32], &cpal::InputCallbackInfo) + Send + 'static {
    let mut processor = CaptureProcessor::new(
        gain,
        recording,
        monitoring,
        noise_suppression,
        hw_rate,
        needs_resample,
        audio_ready,
    );

    move |data: &[f32], _: &cpal::InputCallbackInfo| {
        let out = processor.feed(data);
        if let (Some(ltx), Some(level)) = (level_tx.as_ref(), out.level) {
            let _ = ltx.send(level);
        }
        if let Some(chunk) = out.chunk {
            let _ = tx.send(chunk);
        }
    }
}


/// Open and start an input stream on `device` with the standard processing
#[allow(clippy::too_many_arguments)]
fn open_stream(
    device: &cpal::Device,
    hw_config: &StreamConfig,
    hw_rate: u32,
    needs_resample: bool,
    gain: &Arc<AtomicU32>,
    recording: &Arc<AtomicBool>,
    monitoring: &Arc<AtomicBool>,
    noise_suppression: &Arc<AtomicBool>,
    tx: &Sender<AudioChunk>,
    level_tx: &Option<Sender<f32>>,
    audio_ready: &Option<Arc<AtomicBool>>,
) -> Option<cpal::Stream> {
    let stream = device
        .build_input_stream(
            hw_config,
            make_input_callback(
                gain.clone(),
                recording.clone(),
                monitoring.clone(),
                level_tx.clone(),
                tx.clone(),
                noise_suppression.clone(),
                hw_rate,
                needs_resample,
                audio_ready.clone(),
            ),
            |e| warn!("Audio stream error: {e}"),
            None,
        )
        .map_err(|e| warn!("Failed to build audio stream: {e}"))
        .ok()?;
    stream
        .play()
        .map_err(|e| warn!("Failed to play audio stream: {e}"))
        .ok()?;
    Some(stream)
}

#[allow(unused_assignments, unused_variables)]
fn capture_loop(
    cfg: AudioConfig,
    gain: Arc<AtomicU32>,
    noise_suppression: Arc<AtomicBool>,
    recording: Arc<AtomicBool>,
    monitoring: Arc<AtomicBool>,
    dynamic_stream: Arc<AtomicBool>,
    input_device_index: Arc<AtomicU32>,
    audio_ready: Option<Arc<AtomicBool>>,
    tx: Sender<AudioChunk>,
    level_tx: Option<Sender<f32>>,
    wake: Option<Receiver<()>>,
) -> Result<()> {
    let host = cpal::default_host();

    let mut current_idx = input_device_index.load(Ordering::SeqCst);
    let idx_opt = if current_idx == u32::MAX { None } else { Some(current_idx) };

    let mut device = match test_and_detect_active_device(idx_opt) {
        Ok(d) => d,
        Err(e) => {
            warn!("Startup audio device detection failed: {e}. Falling back to default.");
            host.default_input_device().context("no default input device")?
        }
    };

    info!("Using detected active audio device: {}", device.name().unwrap_or_default());

    let mut hw_config = negotiate_config(&device)?;
    let mut hw_rate = hw_config.sample_rate.0;
    info!("Hardware sample rate: {hw_rate} Hz");

    let mut needs_resample = hw_rate != TARGET_SAMPLE_RATE;

    let mut current_stream: Option<cpal::Stream> = None;
    let mut was_recording = false;
    let mut was_dynamic = dynamic_stream.load(Ordering::SeqCst);

    let active_init = recording.load(Ordering::SeqCst) || monitoring.load(Ordering::SeqCst);
    if was_dynamic {
        if active_init {
        } else if let Some(ref ready) = audio_ready {
            ready.store(false, Ordering::SeqCst);
        }
    } else {
        info!("Startup: Opening always-on stream (Option B)...");
        if let Some(stream) = open_stream(
            &device, &hw_config, hw_rate, needs_resample,
            &gain, &recording, &monitoring, &noise_suppression, &tx, &level_tx, &audio_ready,
        ) {
            current_stream = Some(stream);
            info!("Startup always-on stream successfully playing.");
        }
    }

    loop {
        let is_recording = recording.load(Ordering::SeqCst);
        let is_monitoring = monitoring.load(Ordering::SeqCst);
        let active = is_recording || is_monitoring;
        let is_dynamic = dynamic_stream.load(Ordering::SeqCst);
        let live_idx = input_device_index.load(Ordering::SeqCst);

        if live_idx != current_idx {
            info!("Device index changed from {current_idx} to {live_idx}, hot-reloading audio device...");
            current_idx = live_idx;
            let idx_opt = if current_idx == u32::MAX { None } else { Some(current_idx) };
            match test_and_detect_active_device(idx_opt) {
                Ok(new_device) => {
                    if let Ok(new_config) = negotiate_config(&new_device) {
                        device = new_device;
                        hw_config = new_config;
                        hw_rate = hw_config.sample_rate.0;
                        needs_resample = hw_rate != TARGET_SAMPLE_RATE;
                        info!("Hot-reload: successfully negotiated new device '{}' ({} Hz)", device.name().unwrap_or_default(), hw_rate);
                        current_stream = None; // Drop old stream
                        was_recording = false; // Force rebuild
                        if let Some(ref ready) = audio_ready {
                            ready.store(false, Ordering::SeqCst);
                        }
                    }
                }
                Err(e) => warn!("Hot-reload failed to find functional device for index {current_idx}: {e}"),
            }
        }

        if is_dynamic != was_dynamic {
            info!("Dynamic stream preference changed at runtime to: {is_dynamic}");
            if is_dynamic {
                if !active {
                    current_stream = None; // Closes device!
                    if let Some(ref ready) = audio_ready {
                        ready.store(false, Ordering::SeqCst);
                    }
                    was_recording = false;
                }
            } else {
                if current_stream.is_none() {
                    if let Some(stream) = open_stream(
                        &device, &hw_config, hw_rate, needs_resample,
                        &gain, &recording, &monitoring, &noise_suppression, &tx, &level_tx, &audio_ready,
                    ) {
                        current_stream = Some(stream);
                        info!("Switched to always-on mode: stream successfully playing.");
                    }
                }
            }
            was_dynamic = is_dynamic;
        }

        if is_dynamic {
            if active && !was_recording {
                info!("Dynamic microphone stream starting (Option A)...");
                match open_stream(
                    &device, &hw_config, hw_rate, needs_resample,
                    &gain, &recording, &monitoring, &noise_suppression, &tx, &level_tx, &audio_ready,
                ) {
                    Some(stream) => {
                        current_stream = Some(stream);
                        info!("Dynamic microphone stream successfully playing (Option A).");
                        was_recording = true;
                    }
                    None if !is_monitoring => recording.store(false, Ordering::SeqCst),
                    None => {}
                }
            } else if !active && was_recording {
                info!("Dynamic microphone stream stopping...");
                if let Some(ref ready) = audio_ready {
                    ready.store(false, Ordering::SeqCst);
                }
                current_stream = None; // Dropping closes the input device!
                was_recording = false;
                info!("Dynamic microphone stream stopped & device closed.");
            }
        }

        let backstop = if wake.is_some() || !is_dynamic {
            Duration::from_millis(200)
        } else {
            Duration::from_millis(30)
        };
        match wake {
            Some(ref rx) => match rx.recv_timeout(backstop) {
                Ok(()) => {
                    while rx.try_recv().is_ok() {}
                }
                Err(crossbeam_channel::RecvTimeoutError::Timeout) => {}
                Err(crossbeam_channel::RecvTimeoutError::Disconnected) => {
                    std::thread::sleep(backstop)
                }
            },
            None => std::thread::sleep(backstop),
        }
    }
}

fn negotiate_config(device: &cpal::Device) -> Result<StreamConfig> {
    let supported = device.default_input_config()?;
    Ok(StreamConfig {
        channels: 1,
        sample_rate: SampleRate(supported.sample_rate().0),
        buffer_size: cpal::BufferSize::Default,
    })
}

fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_sq: f32 = samples.iter().map(|&s| s * s).sum();
    (sum_sq / samples.len() as f32).sqrt()
}

/// Linear resampling between two rates, appended to `out`.
pub(crate) fn resample_into(input: &[f32], from_hz: u32, to_hz: u32, out: &mut Vec<f32>) {
    if input.is_empty() {
        return;
    }
    if from_hz == to_hz {
        out.extend_from_slice(input);
        return;
    }
    let ratio = to_hz as f64 / from_hz as f64;
    let out_len = (input.len() as f64 * ratio) as usize;
    let step = 1.0 / ratio;
    let last = input.len() - 1;

    out.reserve(out_len);
    let mut src = 0.0f64;
    for _ in 0..out_len {
        let lo = (src as usize).min(last);
        let frac = (src - lo as f64) as f32;
        let hi = (lo + 1).min(last);
        out.push(input[lo] * (1.0 - frac) + input[hi] * frac);
        src += step;
    }
}

fn resample_chunk(input: &[f32], from_hz: u32, to_hz: u32) -> Vec<f32> {
    let mut out = Vec::new();
    resample_into(input, from_hz, to_hz, &mut out);
    out
}


#[derive(Debug, Clone)]
pub struct AudioDeviceInfo {
    pub index: u32,
    pub name: String,
}

pub fn list_input_devices() -> Vec<AudioDeviceInfo> {
    let host = cpal::default_host();
    host.input_devices()
        .into_iter()
        .flatten()
        .enumerate()
        .map(|(i, d)| AudioDeviceInfo {
            index: i as u32,
            name: d.name().unwrap_or_else(|_| format!("Device {i}")),
        })
        .collect()
}

#[cfg(test)]
mod capture_tests {
    use super::*;

    /// A processor at the target rate, so buffer lengths pass through
    fn processor(recording: &Arc<AtomicBool>) -> CaptureProcessor {
        CaptureProcessor::new(
            Arc::new(AtomicU32::new(1.0f32.to_bits())),
            recording.clone(),
            Arc::new(AtomicBool::new(false)),
            Arc::new(AtomicBool::new(false)),
            TARGET_SAMPLE_RATE,
            false,
            None,
        )
    }

    fn ramp(n: usize, start: f32) -> Vec<f32> {
        (0..n).map(|i| start + i as f32).collect()
    }

    /// The bug: dictation starts when the shortcut fires, but people are
    #[test]
    fn the_opening_buffer_carries_what_came_before_it() {
        let recording = Arc::new(AtomicBool::new(false));
        let mut p = processor(&recording);

        p.feed(&ramp(100, 0.0));
        p.feed(&ramp(100, 100.0));
        assert!(p.feed(&[0.0; 10]).chunk.is_none(), "idle audio was recorded");

        recording.store(true, Ordering::SeqCst);
        let first = p.feed(&ramp(50, 1000.0)).chunk.expect("recording produced nothing");

        assert_eq!(first.len(), 260, "pre-roll was not prepended");
        assert_eq!(first[0], 0.0, "pre-roll did not start at the oldest sample");
        assert_eq!(first[199], 199.0, "pre-roll was not contiguous");
        assert_eq!(first[210], 1000.0, "live audio did not follow the pre-roll");
    }

    #[test]
    fn the_pre_roll_is_spent_once_and_not_repeated() {
        let recording = Arc::new(AtomicBool::new(false));
        let mut p = processor(&recording);
        p.feed(&ramp(100, 0.0));

        recording.store(true, Ordering::SeqCst);
        assert_eq!(p.feed(&ramp(50, 0.0)).chunk.unwrap().len(), 150);
        assert_eq!(
            p.feed(&ramp(50, 0.0)).chunk.unwrap().len(),
            50,
            "the pre-roll was replayed into a later buffer"
        );
    }

    /// The ring is the only thing here that grows, so it has to be bounded by
    #[test]
    fn the_pre_roll_never_grows_past_its_window() {
        let recording = Arc::new(AtomicBool::new(false));
        let mut p = processor(&recording);
        let cap = (TARGET_SAMPLE_RATE as usize * PREROLL_MS as usize) / 1000;

        for _ in 0..200 {
            p.feed(&[0.5; 512]);
        }
        assert_eq!(p.preroll.len(), cap, "the ring drifted from its window");

        recording.store(true, Ordering::SeqCst);
        assert_eq!(p.feed(&[0.0; 10]).chunk.unwrap().len(), cap + 10);
    }

    /// A second recording must open with the audio from just before *it*, not
    #[test]
    fn a_later_recording_gets_its_own_pre_roll() {
        let recording = Arc::new(AtomicBool::new(false));
        let mut p = processor(&recording);

        p.feed(&ramp(100, 0.0));
        recording.store(true, Ordering::SeqCst);
        p.feed(&ramp(10, 0.0));
        recording.store(false, Ordering::SeqCst);

        p.feed(&ramp(70, 500.0));
        recording.store(true, Ordering::SeqCst);
        let second = p.feed(&ramp(10, 0.0)).chunk.unwrap();
        assert_eq!(second.len(), 80, "the second recording lost its pre-roll");
        assert_eq!(second[0], 500.0, "stale audio led the second recording");
    }

    #[test]
    fn gain_applies_to_the_pre_roll_as_well_as_the_live_audio() {
        let recording = Arc::new(AtomicBool::new(false));
        let mut p = processor(&recording);
        p.gain.store(2.0f32.to_bits(), Ordering::SeqCst);

        p.feed(&[1.0; 20]);
        recording.store(true, Ordering::SeqCst);
        let chunk = p.feed(&[3.0; 5]).chunk.unwrap();

        assert!(chunk[..20].iter().all(|&s| s == 2.0), "pre-roll missed the gain");
        assert!(chunk[20..].iter().all(|&s| s == 6.0), "live audio missed the gain");
    }

    /// The level feed exists for the VU meter and the overlay; with neither
    #[test]
    fn levels_are_produced_only_for_someone_watching() {
        let recording = Arc::new(AtomicBool::new(false));
        let monitoring = Arc::new(AtomicBool::new(false));
        let mut p = CaptureProcessor::new(
            Arc::new(AtomicU32::new(1.0f32.to_bits())),
            recording.clone(),
            monitoring.clone(),
            Arc::new(AtomicBool::new(false)),
            TARGET_SAMPLE_RATE,
            false,
            None,
        );

        assert!(p.feed(&[0.5; 16]).level.is_none(), "idle stream reported a level");

        monitoring.store(true, Ordering::SeqCst);
        assert_eq!(p.feed(&[0.5; 16]).level, Some(0.5), "the VU meter got nothing");

        monitoring.store(false, Ordering::SeqCst);
        recording.store(true, Ordering::SeqCst);
        assert_eq!(p.feed(&[0.5; 16]).level, Some(0.5), "the overlay got nothing");
    }

    #[test]
    fn silence_classifies_exact_zero_and_near_zero_buffers() {
        assert!(is_silent(&[0.0; 32]), "an all-zero buffer must read as silent");
        assert!(
            is_silent(&[0.0, 1e-6, -1e-6, 0.0]),
            "dither far below the noise floor must still read as silent"
        );
        assert!(
            !is_silent(&[0.0, 0.0, 0.01, 0.0]),
            "a single real sample must be enough to call a buffer non-silent"
        );
    }

    /// The Windows bug this exists for: a stream can open and `play()`
    #[test]
    fn audio_ready_latches_on_first_real_signal_and_not_before() {
        let recording = Arc::new(AtomicBool::new(true));
        let ready = Arc::new(AtomicBool::new(false));
        let mut p = CaptureProcessor::new(
            Arc::new(AtomicU32::new(1.0f32.to_bits())),
            recording.clone(),
            Arc::new(AtomicBool::new(false)),
            Arc::new(AtomicBool::new(false)),
            TARGET_SAMPLE_RATE,
            false,
            Some(ready.clone()),
        );

        p.feed(&[0.0; 64]);
        assert!(!ready.load(Ordering::SeqCst), "silence must not mark the stream ready");

        p.feed(&[0.02; 64]);
        assert!(ready.load(Ordering::SeqCst), "real signal must mark the stream ready");

        p.feed(&[0.0; 64]);
        assert!(
            ready.load(Ordering::SeqCst),
            "a later quiet buffer must not un-latch a confirmed stream"
        );
    }
}
