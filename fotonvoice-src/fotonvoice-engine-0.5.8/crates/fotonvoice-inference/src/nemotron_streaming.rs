//! Nemotron streaming speech-to-text backend (ONNX Runtime, Phase 2.3).

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
    value::{Tensor, TensorRef},
};
use tracing::{debug, info};
use fotonvoice_config::NemotronStreamingConfig;

use crate::backend::{StreamingBackend, TranscribeRequest, TranscriptionBackend, TranscriptionResult};


pub const SAMPLE_RATE: usize = 16_000;
pub const N_FFT: usize = 512;
pub const HOP_LENGTH: usize = 160;
pub const WIN_LENGTH: usize = 400;
pub const N_MELS: usize = 128;
pub const PREEMPH: f32 = 0.97;
/// NeMo log_zero_guard_type="add", value=2^-24.
pub const LOG_ZERO_GUARD: f32 = 5.960_464_5e-8;

/// FastConformer subsamples mel frames by 8x: one encoder output frame spans
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


/// Load danielbodart tokens.txt: lines of "<piece> <id>", 1024 SentencePiece
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


/// Reusable mel filterbank + FFT plan, built once at model load. Both are
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


/// Output positions, resolved once from the graphs at load time (same
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
fn run_encoder_chunk(
    encoder: &mut Session,
    index: &OutputIndex,
    mel_chunk: &[f32], // N_MELS * input_frames, band-major
    input_frames: usize,
    chunk_length: i64,
    cache: &mut StreamState,
    enc_out_scratch: &mut Vec<f32>,
) -> Result<usize> {
    let mel_tensor = TensorRef::from_array_view(([1_usize, N_MELS, input_frames], mel_chunk))
        .context("build audio_signal tensor")?;
    let length_tensor =
        Tensor::from_array(([1_usize], vec![chunk_length])).context("build length tensor")?;
    let cch_tensor = TensorRef::from_array_view(
        (
            [1_usize, NUM_ENCODER_LAYERS, CACHE_LEFT_CONTEXT, ENCODER_DIM],
            cache.cache_last_channel.as_slice(),
        ),
    )
    .context("build cache_last_channel tensor")?;
    let cti_tensor = TensorRef::from_array_view(
        (
            [1_usize, NUM_ENCODER_LAYERS, ENCODER_DIM, CACHE_CONV_CONTEXT],
            cache.cache_last_time.as_slice(),
        ),
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
    enc_out_scratch.clear();
    enc_out_scratch.extend_from_slice(enc_data);

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
    cache.cache_last_channel.copy_from_slice(cch_next);
    cache.cache_last_time.copy_from_slice(cti_next);
    cache.cache_len = cll_next[0];

    Ok(enc_len[0] as usize)
}

/// Greedy RNN-T decode over one chunk's encoder frames. Appends non-blank ids
fn decode_chunk(
    decoder: &mut Session,
    index: &OutputIndex,
    enc_data: &[f32], // [1, 1024, enc_frames], channels-first
    enc_frames: usize,
    stream: &mut StreamState,
) -> Result<usize> {
    let mut emitted = 0;
    let mut frame = vec![0.0f32; ENCODER_DIM];
    for t in 0..enc_frames {
        for c in 0..ENCODER_DIM {
            frame[c] = enc_data[c * enc_frames + t];
        }

        for _ in 0..MAX_TOKENS_PER_STEP {
            let targets_in = [stream.last_token as i32];
            let target_len_in = [1_i32];
            let enc_in = TensorRef::from_array_view(([1_usize, ENCODER_DIM, 1_usize], frame.as_slice()))
                .context("build decoder encoder_outputs tensor")?;
            let targets = TensorRef::from_array_view(([1_usize, 1_usize], targets_in.as_slice()))
                .context("build targets tensor")?;
            let target_length = TensorRef::from_array_view(([1_usize], target_len_in.as_slice()))
                .context("build target_length tensor")?;
            let s1 = TensorRef::from_array_view(
                (
                    [DECODER_NUM_LAYERS, 1_usize, DECODER_STATE_DIM],
                    stream.state_1.as_slice(),
                ),
            )
            .context("build state_1 tensor")?;
            let s2 = TensorRef::from_array_view(
                (
                    [DECODER_NUM_LAYERS, 1_usize, DECODER_STATE_DIM],
                    stream.state_2.as_slice(),
                ),
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
            stream.state_1.copy_from_slice(n1);
            stream.state_2.copy_from_slice(n2);
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

        let t_all = Instant::now();

        let t0 = Instant::now();
        let full_mel = loaded.features.compute_mel(&stream.audio_buffer)?;
        let t_mel = t0.elapsed();
        let total_mel_frames = full_mel.shape()[1];
        let processed_mel_frames = stream.audio_processed / HOP_LENGTH;
        if total_mel_frames.saturating_sub(processed_mel_frames) < CHUNK_SIZE_MEL {
            return Ok(String::new());
        }

        let main_start = processed_mel_frames;
        let mut chunk_data = vec![0.0f32; N_MELS * CHUNK_INPUT_FRAMES];
        if stream.chunk_idx == 0 {
            for f in 0..CHUNK_SIZE_MEL {
                for m in 0..N_MELS {
                    chunk_data[m * CHUNK_INPUT_FRAMES + PRE_ENCODE_CACHE + f] = full_mel[[m, f]];
                }
            }
        } else {
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
        let t1 = Instant::now();
        let mut enc_scratch = Vec::new();
        let enc_frames = run_encoder_chunk(
            &mut loaded.encoder,
            &loaded.index,
            &chunk_data,
            CHUNK_INPUT_FRAMES,
            CHUNK_INPUT_FRAMES as i64,
            &mut stream,
            &mut enc_scratch,
        )?;
        let t_enc = t1.elapsed();
        let t2 = Instant::now();
        decode_chunk(&mut loaded.decoder, &loaded.index, &enc_scratch, enc_frames, &mut stream)?;
        let t_dec = t2.elapsed();

        debug!(
            "nemotron chunk={} buf={} mel={:?} enc={:?} dec={:?} total={:?} emitted={}",
            stream.chunk_idx,
            stream.audio_buffer.len(),
            t_mel,
            t_enc,
            t_dec,
            t_all.elapsed(),
            stream.accumulated.len() - before
        );

        stream.audio_processed += CHUNK_SIZE_MEL * HOP_LENGTH;
        stream.chunk_idx += 1;

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
                let mut enc_scratch = Vec::new();
                let enc_frames = run_encoder_chunk(
                    &mut loaded.encoder,
                    &loaded.index,
                    &chunk_data[..N_MELS * input_frames],
                    input_frames,
                    input_frames as i64,
                    &mut stream,
                    &mut enc_scratch,
                )?;
                decode_chunk(&mut loaded.decoder, &loaded.index, &enc_scratch, enc_frames, &mut stream)?;
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


/// Offline chunk loop over a complete clip (batch flow; G2a parity baseline):
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

        let mut enc_scratch = Vec::new();
        let enc_frames = run_encoder_chunk(
            &mut loaded.encoder,
            &loaded.index,
            &chunk_data,
            input_frames,
            input_frames as i64,
            &mut stream,
            &mut enc_scratch,
        )?;
        decode_chunk(&mut loaded.decoder, &loaded.index, &enc_scratch, enc_frames, &mut stream)?;

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
