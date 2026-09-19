//! Nemotron streaming speech-to-text backend (ONNX Runtime, Phase 2.3).
//!
//! nvidia/nemotron-speech-streaming-en-0.6b (Mar 2026 checkpoint): cache-aware
//! FastConformer-RNNT. Encoder carries self-attention + conv caches for every
//! layer across chunks (strictly non-overlapping frames, no left-context
//! re-encode), so streaming partials come from true incremental decoding.
//!
//! Pipeline (two graphs, danielbodart 560ms export):
//! 1. mel in Rust (this file)     - 16 kHz audio -> 128-mel log-spectrogram
//!    (preemph 0.97, STFT 512/hop 160/win 400 Hann, Slaney filterbank,
//!    ln(x + 2^-24), NO per-feature normalization - preprocessor.config
//!    says normalize=NA)
//! 2. encoder_model.onnx          - mel chunk [1, 128, 65] (9 pre-encode cache
//!    + 56 chunk frames) -> [1, 1024, 7] + cache round-trip
//! 3. decoder_model.onnx          - decoder+joint combined; greedy RNN-T loop,
//!    blank id 1024, max 10 symbols per frame, LSTM states [2, 1, 640]
//!
//! Streaming pattern ported from altunenes/parakeet-rs (MIT), src/nemotron.rs,
//! adapted to this crate's ort session style (load-dynamic, external
//! onnxruntime.dll 1.30). Cache shapes verified against the actual graphs
//! with nemotron-probe-fp16.py (plan doc section 4b).
//!
//! Chunk size note (G2b, revised 2026-09-18): the fp16/int8-static lane is
//! locked to the single 560ms-chunk export - partial cadence is a non-goal
//! for dictation; final-text accuracy is the requirement.

use std::{
    fs::File,
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
    sync::Mutex,
    time::Instant,
};

use anyhow::{anyhow, bail, Context, Result};
use ndarray::Array2;
use ort::{
    session::{
        builder::{GraphOptimizationLevel, SessionBuilder},
        Session,
    },
    value::Tensor,
};
use tracing::info;
use fotonvoice_config::NemotronStreamingConfig;

use crate::backend::{StreamingBackend, TranscribeRequest, TranscriptionBackend, TranscriptionResult};

// -- Model constants -----------------------------------------------------------

pub const SAMPLE_RATE: usize = 16_000;
pub const N_FFT: usize = 512;
pub const HOP_LENGTH: usize = 160;
pub const WIN_LENGTH: usize = 400;
pub const N_MELS: usize = 128;
pub const PREEMPH: f32 = 0.97;
/// NeMo log_zero_guard_type="add", value=2^-24.
pub const LOG_ZERO_GUARD: f32 = 5.960_464_5e-8;

/// FastConformer subsamples mel frames by 8x: one encoder output frame spans
/// 80 ms at 16 kHz.
pub const SUBSAMPLING_FACTOR: usize = 8;

/// Mel frames of main chunk per 560ms export (56 = 7 output frames x 8).
pub const CHUNK_SIZE_MEL: usize = 56;
/// Mel frames of pre-encode cache prepended to every chunk.
pub const PRE_ENCODE_CACHE: usize = 9;
/// Encoder output frames per chunk (56 / 8).
pub const CHUNK_OUTPUT_FRAMES: usize = 7;

pub const NUM_ENCODER_LAYERS: usize = 24;
pub const ENCODER_DIM: usize = 1024;
pub const CACHE_LEFT_CONTEXT: usize = 70;
pub const CACHE_CONV_CONTEXT: usize = 8;
pub const DECODER_STATE_DIM: usize = 640;
pub const DECODER_NUM_LAYERS: usize = 2;
pub const VOCAB_SIZE: usize = 1024;
pub const BLANK_TOKEN_ID: usize = VOCAB_SIZE;
pub const MAX_TOKENS_PER_STEP: usize = 10;

// Input tensor sizes per chunk: 9 cache + 56 main = 65 frames.
pub const CHUNK_INPUT_FRAMES: usize = PRE_ENCODE_CACHE + CHUNK_SIZE_MEL;

pub const TOKENS_FILE: &str = "tokens.txt";
pub const FILTERBANK_FILE: &str = "filterbank.bin";
pub const PREPROCESSOR_CONFIG_FILE: &str = "preprocessor.config";

pub const ENCODER_FILE: &str = "encoder_model.onnx";
pub const ENCODER_DATA_FILE: &str = "encoder_model.onnx.data";
pub const DECODER_FILE: &str = "decoder_model.onnx";
pub const DECODER_DATA_FILE: &str = "decoder_model.onnx.data";

pub const HF_BASE_URL: &str =
    "https://huggingface.co/danielbodart/nemotron-speech-600m-onnx/resolve/main";

// -- Filesystem layout ---------------------------------------------------------

/// Default parent directory: `<models_base_dir>/nemotron-streaming/<size>/`.
pub fn default_model_dir() -> PathBuf {
    crate::util::models_base_dir().join("nemotron-streaming")
}

fn model_size_dir(model_dir: &str, size: &str) -> PathBuf {
    let base = if model_dir.is_empty() {
        default_model_dir()
    } else {
        crate::util::expand_tilde(model_dir)
    };
    base.join(size)
}

pub fn valid_model_size(size: &str) -> bool {
    matches!(size, "fp16" | "int8-static")
}

/// Files a precision variant needs on disk. Both variants are danielbodart
/// exports sharing the same graph structure (fp16: fp16 weights + fp32 I/O;
/// int8-static: QDQ quantized MatMuls, fp32 decoder). The three shared files
/// (tokens, filterbank, preprocessor config) come from the repo's shared/.
pub fn model_files(size: &str) -> Vec<&'static str> {
    match size {
        "int8-static" => vec![
            TOKENS_FILE,
            FILTERBANK_FILE,
            PREPROCESSOR_CONFIG_FILE,
            ENCODER_FILE,
            ENCODER_DATA_FILE,
            DECODER_FILE,
            DECODER_DATA_FILE,
        ],
        _ => vec![
            TOKENS_FILE,
            FILTERBANK_FILE,
            PREPROCESSOR_CONFIG_FILE,
            ENCODER_FILE,
            ENCODER_DATA_FILE,
            DECODER_FILE,
            DECODER_DATA_FILE,
        ],
    }
}

/// True when all model files for `size` are present on disk.
pub fn is_model_downloaded(size: &str, model_dir: &str) -> bool {
    if !valid_model_size(size) {
        return false;
    }
    let dir = model_size_dir(model_dir, size);
    model_files(size).iter().all(|f| dir.join(f).exists())
}

/// Remove every file the `size` variant downloaded, i.e. the whole
/// `<model_dir>/<size>/` folder. Refuses to run for unknown sizes so a
/// mistyped size can never point the removal at an unexpected path.
pub fn delete_model(size: &str, model_dir: &str) -> Result<()> {
    if !valid_model_size(size) {
        bail!("Unknown Nemotron streaming model size '{size}' (expected 'fp16' or 'int8-static')");
    }
    let dir = model_size_dir(model_dir, size);
    if !dir.exists() {
        return Ok(());
    }
    std::fs::remove_dir_all(&dir)
        .with_context(|| format!("Failed to delete {}", dir.display()))?;
    Ok(())
}

// -- Download ------------------------------------------------------------------

static DOWNLOAD_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

fn file_url(size: &str, file: &str) -> String {
    match file {
        TOKENS_FILE | FILTERBANK_FILE | PREPROCESSOR_CONFIG_FILE => {
            format!("{HF_BASE_URL}/shared/{file}")
        }
        _ => format!("{HF_BASE_URL}/{size}/{file}"),
    }
}

/// Fetch the model files for `size` into `<model_dir>/<size>/`.
pub async fn download_model(size: &str, model_dir: &str) -> Result<()> {
    if !valid_model_size(size) {
        bail!("Unknown Nemotron streaming model size '{size}' (expected 'fp16' or 'int8-static')");
    }

    let dir = model_size_dir(model_dir, size);
    tokio::fs::create_dir_all(&dir).await?;

    let _guard = DOWNLOAD_LOCK.lock().await;

    for file in model_files(size) {
        let path = dir.join(file);
        if path.exists() {
            continue;
        }
        let url = file_url(size, file);
        info!("Downloading Nemotron streaming file: {url}");
        let response = reqwest::get(&url)
            .await
            .with_context(|| format!("request {url}"))?
            .error_for_status()
            .with_context(|| format!("fetch {url}"))?;
        let bytes = response.bytes().await.with_context(|| format!("read {url}"))?;
        let tmp = path.with_extension("part");
        tokio::fs::write(&tmp, &bytes)
            .await
            .with_context(|| format!("write {}", tmp.display()))?;
        tokio::fs::rename(&tmp, &path)
            .await
            .with_context(|| format!("finalize {}", path.display()))?;
    }

    info!("Nemotron streaming '{size}' model ready in {}", dir.display());
    Ok(())
}

// -- Vocabulary ----------------------------------------------------------------

/// Load danielbodart tokens.txt: lines of "<piece> <id>", 1024 SentencePiece
/// pieces, blank id = 1024 = vocab size.
fn load_vocab(path: &Path) -> Result<Vec<String>> {
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let reader = BufReader::new(file);
    let mut vocab: Vec<Option<String>> = Vec::with_capacity(VOCAB_SIZE);

    for line_res in reader.lines() {
        let line = line_res?;
        let trimmed = line.trim_end();
        if trimmed.is_empty() {
            continue;
        }
        let (token, id_str) = match trimmed.rfind(' ') {
            Some(pos) => (&trimmed[..pos], &trimmed[pos + 1..]),
            None => bail!("tokens.txt line without id: '{trimmed}'"),
        };
        let id: usize = id_str
            .trim()
            .parse()
            .with_context(|| format!("bad id in tokens.txt line '{trimmed}'"))?;
        if vocab.len() <= id {
            vocab.resize(id + 1, None);
        }
        vocab[id] = Some(token.to_string());
    }

    let missing: Vec<usize> = (0..vocab.len()).filter(|&i| vocab[i].is_none()).collect();
    if !missing.is_empty() {
        bail!("tokens.txt has gaps at ids {missing:?}");
    }
    Ok(vocab.into_iter().map(|s| s.unwrap()).collect())
}

/// Decode SentencePiece ids to text (U+2581 -> space). Empty pieces and the
/// blank id are skipped by the caller (only non-blank ids are accumulated).
fn detokenize(tokens: &[usize], vocab: &[String]) -> String {
    let mut raw = String::new();
    for &tok_id in tokens {
        if tok_id >= vocab.len() {
            continue;
        }
        raw.push_str(&vocab[tok_id]);
    }
    let converted = raw.replace('\u{2581}', " ");
    converted.trim_start().to_string()
}

// -- Mel features (pure Rust; port of parakeet-rs audio.rs, MIT) ----------------

/// Reusable mel filterbank + FFT plan, built once at model load. Both are
/// deterministic from the fixed NeMo feature config, so caching avoids
/// rebuilding them per request.
pub struct FeatureCache {
    mel_basis: Array2<f32>,
    fft_plan: std::sync::Arc<dyn realfft::RealToComplex<f32>>,
}

fn hann_window(window_length: usize) -> Vec<f32> {
    (0..window_length)
        .map(|i| 0.5 - 0.5 * ((2.0 * std::f32::consts::PI * i as f32) / (window_length as f32 - 1.0)).cos())
        .collect()
}

/// STFT power spectrogram [n_fft/2+1, audio.len()/hop], NeMo frame math:
/// center padding n_fft/2 on both sides, window centered in the FFT buffer,
/// valid frames = floor(audio_len / hop) (NeMo masks torch.stft's extra
/// trailing frame).
fn stft(
    audio: &[f32],
    plan: &std::sync::Arc<dyn realfft::RealToComplex<f32>>,
) -> Result<Array2<f32>> {
    let pad_amount = N_FFT / 2;
    let mut padded = vec![0.0f32; pad_amount];
    padded.extend_from_slice(audio);
    padded.resize(padded.len() + pad_amount, 0.0);

    let window = hann_window(WIN_LENGTH);
    let num_frames = audio.len() / HOP_LENGTH;
    let freq_bins = N_FFT / 2 + 1;
    let mut spectrogram = Array2::<f32>::zeros((freq_bins, num_frames));
    let window_offset = (N_FFT - WIN_LENGTH) / 2;

    let mut input = vec![0.0f32; N_FFT];
    let mut output = plan.make_output_vec();
    let mut scratch = plan.make_scratch_vec();

    for frame_idx in 0..num_frames {
        let start = frame_idx * HOP_LENGTH;
        input.fill(0.0);
        for i in 0..WIN_LENGTH {
            input[window_offset + i] = padded[start + window_offset + i] * window[i];
        }
        plan.process_with_scratch(&mut input, &mut output, &mut scratch)
            .map_err(|e| anyhow!("FFT failed: {e}"))?;
        for k in 0..freq_bins {
            spectrogram[[k, frame_idx]] = output[k].norm_sqr();
        }
    }
    Ok(spectrogram)
}

// Slaney mel scale (librosa semantics).
const F_SP: f64 = 200.0 / 3.0;
const MIN_LOG_HZ: f64 = 1000.0;
const MIN_LOG_MEL: f64 = MIN_LOG_HZ / F_SP;
const LOG_STEP: f64 = 0.06875177742094912;

fn hz_to_mel_slaney(hz: f64) -> f64 {
    if hz < MIN_LOG_HZ {
        hz / F_SP
    } else {
        MIN_LOG_MEL + (hz / MIN_LOG_HZ).ln() / LOG_STEP
    }
}

fn mel_to_hz_slaney(mel: f64) -> f64 {
    if mel < MIN_LOG_MEL {
        mel * F_SP
    } else {
        MIN_LOG_HZ * ((mel - MIN_LOG_MEL) * LOG_STEP).exp()
    }
}

/// Slaney-normalized mel filterbank [n_mels, n_fft/2+1], librosa ramps.
fn create_mel_filterbank(n_fft: usize, n_mels: usize, sample_rate: usize) -> Array2<f32> {
    let freq_bins = n_fft / 2 + 1;
    let mut filterbank = Array2::<f32>::zeros((n_mels, freq_bins));

    let fmax = sample_rate as f64 / 2.0;
    let mel_min = hz_to_mel_slaney(0.0);
    let mel_max = hz_to_mel_slaney(fmax);

    let mel_points: Vec<f64> = (0..=n_mels + 1)
        .map(|i| mel_to_hz_slaney(mel_min + (mel_max - mel_min) * i as f64 / (n_mels + 1) as f64))
        .collect();

    let fft_freqs: Vec<f64> = (0..freq_bins).map(|i| i as f64 * sample_rate as f64 / n_fft as f64).collect();

    let fdiff: Vec<f64> = mel_points.windows(2).map(|w| w[1] - w[0]).collect();

    for i in 0..n_mels {
        for (k, &freq) in fft_freqs.iter().enumerate() {
            let lower = (freq - mel_points[i]) / fdiff[i];
            let upper = (mel_points[i + 2] - freq) / fdiff[i + 1];
            filterbank[[i, k]] = 0.0f64.max(lower.min(upper)) as f32;
        }
    }

    for i in 0..n_mels {
        let enorm = 2.0 / (mel_points[i + 2] - mel_points[i]);
        for k in 0..freq_bins {
            filterbank[[i, k]] *= enorm as f32;
        }
    }

    filterbank
}

impl FeatureCache {
    pub fn new() -> Self {
        let mel_basis = create_mel_filterbank(N_FFT, N_MELS, SAMPLE_RATE);
        let mut planner = realfft::RealFftPlanner::<f32>::new();
        let fft_plan = planner.plan_fft_forward(N_FFT);
        Self { mel_basis, fft_plan }
    }

    fn apply_preemphasis(audio: &[f32]) -> Vec<f32> {
        if audio.is_empty() {
            return Vec::new();
        }
        let mut result = Vec::with_capacity(audio.len());
        result.push(audio[0]);
        for i in 1..audio.len() {
            result.push(audio[i] - PREEMPH * audio[i - 1]);
        }
        result
    }

    /// 128-band log-mel spectrogram, band-major [128, T]. NO per-feature
    /// normalization (preprocessor.config: normalize=NA; NeMo feeds raw
    /// dB log-mel to the encoder).
    fn compute_mel(&self, audio: &[f32]) -> Result<Array2<f32>> {
        if audio.is_empty() {
            return Ok(Array2::zeros((N_MELS, 0)));
        }
        let preemph = Self::apply_preemphasis(audio);
        let spec = stft(&preemph, &self.fft_plan)?;
        let mel = self.mel_basis.dot(&spec);
        Ok(mel.mapv(|x| (x + LOG_ZERO_GUARD).ln()))
    }
}

// -- Loaded State --------------------------------------------------------------

/// Output positions, resolved once from the graphs at load time (same
/// defensive pattern as parakeet.rs).
struct OutputIndex {
    enc_out: usize,
    enc_len: usize,
    cch_next: usize,
    cti_next: usize,
    cll_next: usize,
    dec_logits: usize,
    dec_s1: usize,
    dec_s2: usize,
}

impl OutputIndex {
    fn resolve(encoder: &Session, decoder: &Session) -> Result<Self> {
        let enc = |name: &str| -> Result<usize> {
            encoder
                .outputs()
                .iter()
                .position(|o| o.name() == name)
                .ok_or_else(|| anyhow!("encoder output '{name}' missing (unexpected export?)"))
        };
        let dec = |name: &str| -> Result<usize> {
            decoder
                .outputs()
                .iter()
                .position(|o| o.name() == name)
                .ok_or_else(|| anyhow!("decoder output '{name}' missing (unexpected export?)"))
        };
        Ok(Self {
            enc_out: enc("outputs")?,
            enc_len: enc("encoded_lengths")?,
            cch_next: enc("cache_last_channel_next")?,
            cti_next: enc("cache_last_time_next")?,
            cll_next: enc("cache_last_channel_next_len")?,
            dec_logits: dec("outputs")?,
            dec_s1: dec("output_states_1")?,
            dec_s2: dec("output_states_2")?,
        })
    }
}

struct Loaded {
    encoder: Session,
    decoder: Session,
    vocab: Vec<String>,
    features: FeatureCache,
    index: OutputIndex,
}

// -- Backend -------------------------------------------------------------------

pub struct NemotronStreamingBackend {
    cfg: NemotronStreamingConfig,
    state: Mutex<Option<Loaded>>,
    stream: Mutex<StreamState>,
    loaded: bool,
}

impl NemotronStreamingBackend {
    pub fn new(cfg: NemotronStreamingConfig) -> Self {
        Self {
            cfg,
            state: Mutex::new(None),
            stream: Mutex::new(StreamState::new()),
            loaded: false,
        }
    }

    fn build_session(path: &Path, use_gpu: bool) -> Result<Session> {
        let builder = Session::builder()
            .map_err(|e| anyhow!("ort session builder: {e}"))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| anyhow!("set optimization level: {e}"))?
            .with_intra_threads(crate::util::inference_threads())
            .map_err(|e| anyhow!("set intra threads: {e}"))?;

        let mut builder = if use_gpu { Self::with_gpu(builder) } else { builder };

        builder
            .commit_from_file(path)
            .with_context(|| format!("load ONNX graph {}", path.display()))
    }

    fn with_gpu(builder: SessionBuilder) -> SessionBuilder {
        #[cfg(any(
            feature = "nemotron-streaming-cuda",
            feature = "nemotron-streaming-coreml",
            feature = "nemotron-streaming-webgpu"
        ))]
        {
            // WebGPU is a plugin EP in ORT >= 1.24.4: it lives in
            // onnxruntime_providers_webgpu.dll beside the runtime and is
            // registered once per environment, then its devices attached per
            // session. Same road parakeet takes (see parakeet.rs).
            #[cfg(all(
                feature = "nemotron-streaming-webgpu",
                not(any(feature = "nemotron-streaming-cuda", feature = "nemotron-streaming-coreml"))
            ))]
            {
                let devices = crate::webgpu::webgpu_devices();
                let name = crate::parakeet_gpu_provider().unwrap_or("WebGPU");

                if devices.is_empty() {
                    tracing::warn!(
                        "Nemotron streaming: {name} execution provider unavailable (no plugin dll or no device); running on CPU"
                    );
                    return builder;
                }

                match builder.with_devices(devices, None) {
                    Ok(with_dev) => {
                        tracing::info!("Nemotron streaming: {name} execution provider attached");
                        return with_dev;
                    }
                    Err(e) => {
                        tracing::warn!(
                            "Nemotron streaming: {name} execution provider unavailable ({e}); running on CPU"
                        );
                        return e.recover();
                    }
                }
            }

            #[cfg(any(feature = "nemotron-streaming-cuda", feature = "nemotron-streaming-coreml"))]
            {
                #[cfg(feature = "nemotron-streaming-cuda")]
                let ep = ort::ep::CUDA::default().build().error_on_failure();
                #[cfg(all(feature = "nemotron-streaming-coreml", not(feature = "nemotron-streaming-cuda")))]
                let ep = ort::ep::CoreML::default().build().error_on_failure();

                let name = crate::parakeet_gpu_provider().unwrap_or("GPU");

                match builder.with_execution_providers([ep]) {
                    Ok(with_ep) => {
                        tracing::debug!("Nemotron streaming: {name} execution provider registered");
                        return with_ep;
                    }
                    Err(e) => {
                        tracing::warn!(
                            "Nemotron streaming: {name} execution provider unavailable ({e}); running on CPU"
                        );
                        return e.recover();
                    }
                }
            }
        }

        #[cfg(not(any(
            feature = "nemotron-streaming-cuda",
            feature = "nemotron-streaming-coreml",
            feature = "nemotron-streaming-webgpu"
        )))]
        builder
    }

    pub fn load(&mut self) -> Result<()> {
        if self.loaded {
            return Ok(());
        }
        let size = &self.cfg.model_size;
        if !valid_model_size(size) {
            bail!("Unknown Nemotron streaming model size '{size}'");
        }
        let dir = model_size_dir("", size);
        if !is_model_downloaded(size, "") {
            bail!(
                "Nemotron streaming '{size}' model is not downloaded (expected {} in {}). \
                 Open Settings -> Engine and download it, or place the files there manually.",
                model_files(size).join(", "),
                dir.display()
            );
        }

        info!(
            "Loading Nemotron streaming '{size}' model from {} (gpu={})",
            dir.display(),
            self.cfg.gpu
        );

        let encoder = Self::build_session(&dir.join(ENCODER_FILE), self.cfg.gpu)?;
        let decoder = Self::build_session(&dir.join(DECODER_FILE), self.cfg.gpu)?;

        // Shape sanity: fail fast on a wrong/foreign graph instead of
        // producing garbage transcripts later.
        for name in ["audio_signal", "length", "cache_last_channel", "cache_last_time", "cache_last_channel_len"] {
            if !encoder.inputs().iter().any(|i| i.name() == name) {
                bail!("encoder graph missing input '{name}' (unexpected export?)");
            }
        }
        for name in ["encoder_outputs", "targets", "target_length", "input_states_1", "input_states_2"] {
            if !decoder.inputs().iter().any(|i| i.name() == name) {
                bail!("decoder graph missing input '{name}' (unexpected export?)");
            }
        }

        let vocab = load_vocab(&dir.join(TOKENS_FILE))?;
        if vocab.len() != VOCAB_SIZE {
            bail!(
                "tokens.txt has {} pieces, expected {VOCAB_SIZE} (blank id {BLANK_TOKEN_ID})",
                vocab.len()
            );
        }

        let features = FeatureCache::new();
        let index = OutputIndex::resolve(&encoder, &decoder)?;

        *self.state.lock().unwrap() = Some(Loaded { encoder, decoder, vocab, features, index });
        self.loaded = true;
        Ok(())
    }

    pub fn unload(&mut self) {
        *self.state.lock().unwrap() = None;
        self.loaded = false;
    }
}

// -- Streaming decode ------------------------------------------------------------

/// Per-utterance streaming state (cache + decoder state + buffers).
#[derive(Clone)]
struct StreamState {
    cache_last_channel: Vec<f32>, // [NUM_ENCODER_LAYERS, 1, CACHE_LEFT_CONTEXT, ENCODER_DIM]
    cache_last_time: Vec<f32>,    // [NUM_ENCODER_LAYERS, 1, ENCODER_DIM, CACHE_CONV_CONTEXT]
    cache_len: i64,
    state_1: Vec<f32>, // [DECODER_NUM_LAYERS, 1, DECODER_STATE_DIM]
    state_2: Vec<f32>,
    last_token: usize,
    audio_buffer: Vec<f32>,
    audio_processed: usize,
    chunk_idx: usize,
    accumulated: Vec<usize>,
}

impl StreamState {
    fn new() -> Self {
        Self {
            cache_last_channel: vec![0.0; NUM_ENCODER_LAYERS * CACHE_LEFT_CONTEXT * ENCODER_DIM],
            cache_last_time: vec![0.0; NUM_ENCODER_LAYERS * ENCODER_DIM * CACHE_CONV_CONTEXT],
            cache_len: 0,
            state_1: vec![0.0; DECODER_NUM_LAYERS * 1 * DECODER_STATE_DIM],
            state_2: vec![0.0; DECODER_NUM_LAYERS * 1 * DECODER_STATE_DIM],
            last_token: BLANK_TOKEN_ID,
            audio_buffer: Vec::new(),
            audio_processed: 0,
            chunk_idx: 0,
            accumulated: Vec::new(),
        }
    }

    fn reset(&mut self) {
        self.cache_last_channel.fill(0.0);
        self.cache_last_time.fill(0.0);
        self.cache_len = 0;
        self.state_1.fill(0.0);
        self.state_2.fill(0.0);
        self.last_token = BLANK_TOKEN_ID;
        self.audio_buffer.clear();
        self.audio_processed = 0;
        self.chunk_idx = 0;
        self.accumulated.clear();
    }
}

/// Run one encoder chunk forward: mel [1, 128, input_frames] + cache in ->
/// encoded [1, 1024, T'] + cache out. Graph names probe-verified (plan 4b).
/// `input_frames` = PRE_ENCODE_CACHE + main_len (65 for a full chunk; the
/// last offline chunk can be shorter - the T axis is dynamic).
fn run_encoder_chunk(
    encoder: &mut Session,
    index: &OutputIndex,
    mel_chunk: Vec<f32>, // N_MELS * input_frames, band-major
    input_frames: usize,
    chunk_length: i64,
    cache: &mut StreamState,
) -> Result<(Vec<f32>, usize)> {
    let mel_tensor = Tensor::from_array(([1_usize, N_MELS, input_frames], mel_chunk))
        .context("build audio_signal tensor")?;
    let length_tensor =
        Tensor::from_array(([1_usize], vec![chunk_length])).context("build length tensor")?;
    let cch_tensor = Tensor::from_array(
        ([1_usize, NUM_ENCODER_LAYERS, CACHE_LEFT_CONTEXT, ENCODER_DIM], cache.cache_last_channel.clone()),
    )
    .context("build cache_last_channel tensor")?;
    let cti_tensor = Tensor::from_array(
        ([1_usize, NUM_ENCODER_LAYERS, ENCODER_DIM, CACHE_CONV_CONTEXT], cache.cache_last_time.clone()),
    )
    .context("build cache_last_time tensor")?;
    let cll_tensor = Tensor::from_array(([1_usize], vec![cache.cache_len]))
        .context("build cache_last_channel_len tensor")?;

    let feed: Vec<(&str, ort::session::SessionInputValue)> = vec![
        ("audio_signal", mel_tensor.into()),
        ("length", length_tensor.into()),
        ("cache_last_channel", cch_tensor.into()),
        ("cache_last_time", cti_tensor.into()),
        ("cache_last_channel_len", cll_tensor.into()),
    ];

    let outs = encoder.run(feed).context("encoder chunk run")?;

    let (_, enc_data) = outs[index.enc_out]
        .try_extract_tensor::<f32>()
        .context("extract encoder outputs")?;
    let enc_data_vec = enc_data.to_vec();

    let (_, enc_len) = outs[index.enc_len]
        .try_extract_tensor::<i64>()
        .context("extract encoded_lengths")?;

    let (_, cch_next) = outs[index.cch_next]
        .try_extract_tensor::<f32>()
        .context("extract cache_last_channel_next")?;
    let (_, cti_next) = outs[index.cti_next]
        .try_extract_tensor::<f32>()
        .context("extract cache_last_time_next")?;
    let (_, cll_next) = outs[index.cll_next]
        .try_extract_tensor::<i64>()
        .context("extract cache_last_channel_next_len")?;
    cache.cache_last_channel = cch_next.to_vec();
    cache.cache_last_time = cti_next.to_vec();
    cache.cache_len = cll_next[0];

    Ok((enc_data_vec, enc_len[0] as usize))
}

/// Greedy RNN-T decode over one chunk's encoder frames. Appends non-blank ids
/// to `accumulated` and returns the count emitted.
fn decode_chunk(
    decoder: &mut Session,
    index: &OutputIndex,
    enc_data: &[f32], // [1, 1024, enc_frames], channels-first
    enc_frames: usize,
    stream: &mut StreamState,
) -> Result<usize> {
    let mut emitted = 0;
    for t in 0..enc_frames {
        let frame: Vec<f32> = (0..ENCODER_DIM).map(|c| enc_data[c * enc_frames + t]).collect();

        for _ in 0..MAX_TOKENS_PER_STEP {
            let enc_in = Tensor::from_array(([1_usize, ENCODER_DIM, 1_usize], frame.clone()))
                .context("build decoder encoder_outputs tensor")?;
            let targets = Tensor::from_array(([1_usize, 1_usize], vec![stream.last_token as i32]))
                .context("build targets tensor")?;
            let target_length = Tensor::from_array(([1_usize], vec![1_i32]))
                .context("build target_length tensor")?;
            let s1 = Tensor::from_array(
                ([DECODER_NUM_LAYERS, 1_usize, DECODER_STATE_DIM], stream.state_1.clone()),
            )
            .context("build state_1 tensor")?;
            let s2 = Tensor::from_array(
                ([DECODER_NUM_LAYERS, 1_usize, DECODER_STATE_DIM], stream.state_2.clone()),
            )
            .context("build state_2 tensor")?;

            let feed: Vec<(&str, ort::session::SessionInputValue)> = vec![
                ("encoder_outputs", enc_in.into()),
                ("targets", targets.into()),
                ("target_length", target_length.into()),
                ("input_states_1", s1.into()),
                ("input_states_2", s2.into()),
            ];

            let outs = decoder.run(feed).context("decoder step run")?;

            let (_, ldata) = outs[index.dec_logits]
                .try_extract_tensor::<f32>()
                .context("extract decoder logits")?;
            let row_offset = ldata.len() - (VOCAB_SIZE + 1);
            let row = &ldata[row_offset..];
            let best = argmax(row);

            if best == BLANK_TOKEN_ID {
                break;
            }
            stream.accumulated.push(best);
            stream.last_token = best;
            emitted += 1;

            let (_, n1) = outs[index.dec_s1]
                .try_extract_tensor::<f32>()
                .context("extract output_states_1")?;
            let (_, n2) = outs[index.dec_s2]
                .try_extract_tensor::<f32>()
                .context("extract output_states_2")?;
            stream.state_1 = n1.to_vec();
            stream.state_2 = n2.to_vec();
        }
    }
    Ok(emitted)
}

fn argmax(slice: &[f32]) -> usize {
    let mut best_idx = 0;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &val) in slice.iter().enumerate() {
        if val > best_val {
            best_val = val;
            best_idx = i;
        }
    }
    best_idx
}

// -- Streaming interface --------------------------------------------------------

impl StreamingBackend for NemotronStreamingBackend {
    fn chunk_samples(&self) -> usize {
        CHUNK_SIZE_MEL * HOP_LENGTH // 8960 samples = 560ms
    }

    fn feed(&mut self, samples: &[f32]) -> Result<String> {
        if !self.loaded {
            bail!("Model not loaded");
        }
        let mut guard = self.state.lock().unwrap();
        let loaded = guard.as_mut().context("Nemotron state not initialised")?;
        let mut stream = self.stream.lock().unwrap();

        stream.audio_buffer.extend_from_slice(samples);
        if stream.audio_buffer.len() < WIN_LENGTH {
            return Ok(String::new());
        }

        // Mel over the entire buffer: avoids edge effects at chunk boundaries
        // (parakeet-rs pattern). full_mel is [128, total_frames].
        let full_mel = loaded.features.compute_mel(&stream.audio_buffer)?;
        let total_mel_frames = full_mel.shape()[1];
        let processed_mel_frames = stream.audio_processed / HOP_LENGTH;
        if total_mel_frames.saturating_sub(processed_mel_frames) < CHUNK_SIZE_MEL {
            return Ok(String::new());
        }

        let main_start = processed_mel_frames;
        let mut chunk_data = vec![0.0f32; N_MELS * CHUNK_INPUT_FRAMES];
        if stream.chunk_idx == 0 {
            // First chunk: zero pre-encode cache section, main from frame 0.
            for f in 0..CHUNK_SIZE_MEL {
                for m in 0..N_MELS {
                    chunk_data[m * CHUNK_INPUT_FRAMES + PRE_ENCODE_CACHE + f] = full_mel[[m, f]];
                }
            }
        } else {
            // Subsequent chunks: 9 pre-encode cache frames from before the
            // main window, then the main 56.
            let cache_start = main_start.saturating_sub(PRE_ENCODE_CACHE);
            let cache_frames = main_start - cache_start;
            let cache_offset = PRE_ENCODE_CACHE - cache_frames;
            for f in 0..cache_frames {
                for m in 0..N_MELS {
                    chunk_data[m * CHUNK_INPUT_FRAMES + cache_offset + f] = full_mel[[m, cache_start + f]];
                }
            }
            for f in 0..CHUNK_SIZE_MEL {
                for m in 0..N_MELS {
                    chunk_data[m * CHUNK_INPUT_FRAMES + PRE_ENCODE_CACHE + f] = full_mel[[m, main_start + f]];
                }
            }
        }

        let before = stream.accumulated.len();
        let (enc_data, enc_frames) = run_encoder_chunk(
            &mut loaded.encoder,
            &loaded.index,
            chunk_data,
            CHUNK_INPUT_FRAMES,
            CHUNK_INPUT_FRAMES as i64,
            &mut stream,
        )?;
        decode_chunk(&mut loaded.decoder, &loaded.index, &enc_data, enc_frames, &mut stream)?;

        stream.audio_processed += CHUNK_SIZE_MEL * HOP_LENGTH;
        stream.chunk_idx += 1;

        // Trim the buffer; keep enough for pre-encode cache context.
        let keep_samples = (PRE_ENCODE_CACHE + CHUNK_SIZE_MEL) * HOP_LENGTH + 2 * WIN_LENGTH;
        if stream.audio_buffer.len() > keep_samples * 2 {
            let remove = stream.audio_buffer.len() - keep_samples;
            let actual_remove = remove.min(stream.audio_processed);
            stream.audio_buffer.drain(0..actual_remove);
            stream.audio_processed -= actual_remove;
        }

        let new_tokens = &stream.accumulated[before..];
        let text = detokenize(new_tokens, &loaded.vocab);
        Ok(text)
    }

    fn flush(&mut self) -> Result<String> {
        if !self.loaded {
            bail!("Model not loaded");
        }
        let mut guard = self.state.lock().unwrap();
        let loaded = guard.as_mut().context("Nemotron state not initialised")?;
        let mut stream = self.stream.lock().unwrap();

        // Encode any remainder the last full chunk left behind (up to 559ms
        // of tail audio would otherwise be lost at stop).
        if stream.audio_buffer.len() >= WIN_LENGTH {
            let full_mel = loaded.features.compute_mel(&stream.audio_buffer)?;
            let total = full_mel.shape()[1];
            let processed = stream.audio_processed / HOP_LENGTH;
            let remaining = total.saturating_sub(processed);
            if remaining > 0 && total > 0 {
                let main_start = processed;
                let mut chunk_data = vec![0.0f32; N_MELS * CHUNK_INPUT_FRAMES];
                if stream.chunk_idx == 0 {
                    for f in 0..remaining.min(CHUNK_SIZE_MEL) {
                        for m in 0..N_MELS {
                            chunk_data[m * CHUNK_INPUT_FRAMES + PRE_ENCODE_CACHE + f] =
                                full_mel[[m, main_start + f]];
                        }
                    }
                } else {
                    let cache_start = main_start.saturating_sub(PRE_ENCODE_CACHE);
                    let cache_frames = main_start - cache_start;
                    let cache_offset = PRE_ENCODE_CACHE - cache_frames;
                    for f in 0..cache_frames {
                        for m in 0..N_MELS {
                            chunk_data[m * CHUNK_INPUT_FRAMES + cache_offset + f] =
                                full_mel[[m, cache_start + f]];
                        }
                    }
                    for f in 0..remaining.min(CHUNK_SIZE_MEL) {
                        for m in 0..N_MELS {
                            chunk_data[m * CHUNK_INPUT_FRAMES + PRE_ENCODE_CACHE + f] =
                                full_mel[[m, main_start + f]];
                        }
                    }
                }
                let input_frames = PRE_ENCODE_CACHE + remaining.min(CHUNK_SIZE_MEL);
                let (enc_data, enc_frames) = run_encoder_chunk(
                    &mut loaded.encoder,
                    &loaded.index,
                    chunk_data[..N_MELS * input_frames].to_vec(),
                    input_frames,
                    input_frames as i64,
                    &mut stream,
                )?;
                decode_chunk(&mut loaded.decoder, &loaded.index, &enc_data, enc_frames, &mut stream)?;
            }
        }

        let text = detokenize(&stream.accumulated, &loaded.vocab);
        stream.reset();
        Ok(text.trim().to_string())
    }

    fn reset(&mut self) {
        self.stream.lock().unwrap().reset();
    }
}

// -- Offline batch decode -------------------------------------------------------

/// Offline chunk loop over a complete clip (batch flow; G2a parity baseline):
/// mel over the full clip, stepping CHUNK_SIZE_MEL frames per chunk, last
/// chunk may be shorter (the encoder T axis is dynamic).
fn run_offline(loaded: &mut Loaded, audio: &[f32]) -> Result<Vec<usize>> {
    let mut stream = StreamState::new();

    let mel = loaded.features.compute_mel(audio)?;
    let total_frames = mel.shape()[1];
    if total_frames == 0 {
        return Ok(Vec::new());
    }

    let mut buffer_idx = 0usize;
    let mut chunk_idx = 0usize;
    while buffer_idx < total_frames {
        let chunk_end = (buffer_idx + CHUNK_SIZE_MEL).min(total_frames);
        let main_len = chunk_end - buffer_idx;
        let input_frames = PRE_ENCODE_CACHE + main_len;

        let mut chunk_data = vec![0.0f32; N_MELS * input_frames];
        if chunk_idx > 0 && buffer_idx >= PRE_ENCODE_CACHE {
            let cache_start = buffer_idx - PRE_ENCODE_CACHE;
            for f in 0..PRE_ENCODE_CACHE {
                for m in 0..N_MELS {
                    chunk_data[m * input_frames + f] = mel[[m, cache_start + f]];
                }
            }
        }
        for f in 0..main_len {
            for m in 0..N_MELS {
                chunk_data[m * input_frames + PRE_ENCODE_CACHE + f] = mel[[m, buffer_idx + f]];
            }
        }

        let (enc_data, enc_frames) = run_encoder_chunk(
            &mut loaded.encoder,
            &loaded.index,
            chunk_data,
            input_frames,
            input_frames as i64,
            &mut stream,
        )?;
        decode_chunk(&mut loaded.decoder, &loaded.index, &enc_data, enc_frames, &mut stream)?;

        buffer_idx += CHUNK_SIZE_MEL;
        chunk_idx += 1;
    }

    Ok(stream.accumulated)
}

impl TranscriptionBackend for NemotronStreamingBackend {
    fn name(&self) -> &str {
        "nemotron-streaming"
    }

    fn load(&mut self) -> Result<()> {
        self.load()
    }

    fn transcribe(&self, req: &TranscribeRequest) -> Result<TranscriptionResult> {
        if !self.loaded {
            bail!("Model not loaded");
        }
        if req.audio.is_empty() {
            return Ok(TranscriptionResult {
                text: String::new(),
                language: self.cfg.language.clone(),
                language_probability: 1.0,
                duration_ms: 0,
                inference_ms: 0,
                word_timestamps: None,
            });
        }

        let t0 = Instant::now();
        let mut guard = self.state.lock().unwrap();
        let loaded = guard.as_mut().context("Nemotron state not initialised")?;
        let tokens = run_offline(loaded, &req.audio)?;
        let text = detokenize(&tokens, &loaded.vocab);
        let inference_ms = t0.elapsed().as_millis() as u32;
        drop(guard);

        Ok(TranscriptionResult {
            text: text.trim().to_string(),
            language: self.cfg.language.clone(),
            language_probability: 1.0,
            duration_ms: (req.audio.len() / (SAMPLE_RATE / 1000)) as u32,
            inference_ms,
            word_timestamps: None,
        })
    }

    fn unload(&mut self) {
        self.unload()
    }

    fn is_loaded(&self) -> bool {
        self.loaded
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn probe_dir() -> Option<PathBuf> {
        std::env::var("FOTON_NEMOTRON_DIR")
            .ok()
            .filter(|p| Path::new(p).join(ENCODER_FILE).exists())
            .map(PathBuf::from)
    }

    /// Decode a wav file to mono f32 (average channels).
    fn load_wav_mono(path: &Path) -> Result<Vec<f32>> {
        let mut reader = hound::WavReader::open(path)?;
        let spec = reader.spec();
        let samples: Vec<f32> = match spec.sample_format {
            hound::SampleFormat::Float => reader
                .samples::<f32>()
                .collect::<std::result::Result<Vec<_>, _>>()?,
            hound::SampleFormat::Int => reader
                .samples::<i16>()
                .map(|s| s.map(|s| s as f32 / 32768.0))
                .collect::<std::result::Result<Vec<_>, _>>()?,
        };
        let samples = if spec.channels > 1 {
            samples
                .chunks(spec.channels as usize)
                .map(|c| c.iter().sum::<f32>() / spec.channels as f32)
                .collect()
        } else {
            samples
        };
        Ok(samples)
    }

    #[test]
    fn test_mel_shapes_and_preemph() {
        let cache = FeatureCache::new();
        let mel = cache.compute_mel(&vec![0.0f32; SAMPLE_RATE]).unwrap();
        assert_eq!(mel.shape(), &[N_MELS, SAMPLE_RATE / HOP_LENGTH]);
        let sine: Vec<f32> = (0..SAMPLE_RATE)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / SAMPLE_RATE as f32).sin())
            .collect();
        let mel = cache.compute_mel(&sine).unwrap();
        assert!(mel.iter().all(|x| x.is_finite()));
        let preemph = FeatureCache::apply_preemphasis(&[1.0, 1.0, 1.0]);
        assert_eq!(preemph, vec![1.0, 1.0 - PREEMPH, 1.0 - PREEMPH]);
    }

    #[test]
    fn test_vocab_loader() {
        let dir = match probe_dir() {
            Some(d) => d,
            None => return,
        };
        let vocab = load_vocab(&dir.join(TOKENS_FILE)).expect("vocab load");
        assert_eq!(vocab.len(), VOCAB_SIZE);
        assert!(vocab.iter().any(|p| p.contains('\u{2581}')));
        let ids = vec![4usize];
        assert_eq!(detokenize(&ids, &vocab), "in");
    }

    #[test]
    fn test_offline_decode_test_wav() {
        // G2a material: whole-file chunked decode of the sherpa test wav.
        // Env: FOTON_NEMOTRON_DIR = model dir; FOTON_NEMOTRON_WAV = 16 kHz wav.
        let dir = match probe_dir() {
            Some(d) => d,
            None => return,
        };
        let wav = match std::env::var("FOTON_NEMOTRON_WAV") {
            Ok(w) if Path::new(&w).exists() => PathBuf::from(w),
            _ => return,
        };
        let encoder = NemotronStreamingBackend::build_session(&dir.join(ENCODER_FILE), false)
            .expect("encoder session");
        let decoder = NemotronStreamingBackend::build_session(&dir.join(DECODER_FILE), false)
            .expect("decoder session");
        let vocab = load_vocab(&dir.join(TOKENS_FILE)).expect("vocab");
        let features = FeatureCache::new();
        let index = OutputIndex::resolve(&encoder, &decoder).expect("index");
        let mut loaded = Loaded { encoder, decoder, vocab, features, index };

        let audio = load_wav_mono(&wav).expect("wav decode");
        let tokens = run_offline(&mut loaded, &audio).expect("offline decode");
        let text = detokenize(&tokens, &loaded.vocab);
        println!("nemotron offline decode: {text}");
        assert!(!text.trim().is_empty());
        assert!(text.to_uppercase().contains("YELLOW LAMPS"));
    }

    #[test]
    fn test_streaming_feed_matches_offline() {
        // Streaming decode over the same wav: feed 560ms chunks, flush; final
        // text must contain the reference sentence.
        let dir = match probe_dir() {
            Some(d) => d,
            None => return,
        };
        let wav = match std::env::var("FOTON_NEMOTRON_WAV") {
            Ok(w) if Path::new(&w).exists() => PathBuf::from(w),
            _ => return,
        };
        let mut backend = NemotronStreamingBackend::new(NemotronStreamingConfig::default());

        let encoder = NemotronStreamingBackend::build_session(&dir.join(ENCODER_FILE), false)
            .expect("encoder session");
        let decoder = NemotronStreamingBackend::build_session(&dir.join(DECODER_FILE), false)
            .expect("decoder session");
        let vocab = load_vocab(&dir.join(TOKENS_FILE)).expect("vocab");
        let features = FeatureCache::new();
        let index = OutputIndex::resolve(&encoder, &decoder).expect("index");
        *backend.state.lock().unwrap() = Some(Loaded { encoder, decoder, vocab, features, index });
        backend.loaded = true;

        let audio = load_wav_mono(&wav).expect("wav decode");
        let chunk = CHUNK_SIZE_MEL * HOP_LENGTH;
        let mut streamed = String::new();
        for piece in audio.chunks(chunk) {
            streamed.push_str(&backend.feed(piece).expect("feed"));
        }
        let final_text = backend.flush().expect("flush");
        println!("nemotron streamed partials: {streamed}");
        println!("nemotron flush final: {final_text}");
        assert!(!final_text.trim().is_empty());
        assert!(final_text.to_uppercase().contains("YELLOW LAMPS"));
    }
}
