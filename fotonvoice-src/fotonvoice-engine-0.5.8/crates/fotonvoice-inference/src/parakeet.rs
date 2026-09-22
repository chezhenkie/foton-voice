//! Parakeet speech-to-text backend (ONNX Runtime).

use std::{
    fs::File,
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
    sync::Mutex,
    time::Instant,
};

use anyhow::{anyhow, bail, Context, Result};
use ort::{
    session::{
        builder::{GraphOptimizationLevel, SessionBuilder},
        Session, SessionInputValue,
    },
    value::TensorRef,
};
use tracing::info;
use fotonvoice_config::ParakeetConfig;

use crate::backend::{TranscribeRequest, TranscriptionBackend, TranscriptionResult};


pub const SAMPLE_RATE: usize = 16_000;
pub const BLANK_TOKEN_ID: usize = 8192;
pub const NUM_DURATION_CLASSES: usize = 5;
pub const MAX_TOKENS_PER_STEP: usize = 10;
pub const ENCODER_DIM: usize = 1024;
pub const DECODER_STATE_DIM: usize = 640;
pub const DECODER_NUM_LAYERS: usize = 2;

pub const CONFIG_FILE: &str = "config.json";
pub const VOCAB_FILE: &str = "vocab.txt";
pub const PREPROCESSOR_FILE: &str = "nemo128.onnx";
pub const ENCODER_INT8_FILE: &str = "encoder-model.int8.onnx";
pub const DECODER_INT8_FILE: &str = "decoder_joint-model.int8.onnx";
pub const ENCODER_FP32_FILE: &str = "encoder-model.onnx";
pub const ENCODER_FP32_DATA_FILE: &str = "encoder-model.onnx.data";
pub const DECODER_FP32_FILE: &str = "decoder_joint-model.onnx";

pub const HF_BASE_URL: &str =
    "https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx/resolve/main";


/// Default parent directory for Parakeet models: `<models_base_dir>/parakeet/`.
pub fn default_model_dir() -> PathBuf {
    crate::util::models_base_dir().join("parakeet")
}

/// Directory holding one model's files: `<model_dir>/<size>/`.
fn model_size_dir(model_dir: &str, size: &str) -> PathBuf {
    let base = if model_dir.is_empty() {
        default_model_dir()
    } else {
        crate::util::expand_tilde(model_dir)
    };
    base.join(size)
}

pub fn valid_model_size(size: &str) -> bool {
    matches!(
        size,
        "tdt-0.6b-v3" | "tdt-0.6b-v3-int8" | "tdt-0.6b-v3-fp32"
    )
}

/// Files a model size needs on disk. The default and `-int8` sizes share the
pub fn model_files(size: &str) -> Vec<&'static str> {
    match size {
        "tdt-0.6b-v3-fp32" => vec![
            CONFIG_FILE,
            VOCAB_FILE,
            PREPROCESSOR_FILE,
            ENCODER_FP32_FILE,
            ENCODER_FP32_DATA_FILE,
            DECODER_FP32_FILE,
        ],
        _ => vec![
            CONFIG_FILE,
            VOCAB_FILE,
            PREPROCESSOR_FILE,
            ENCODER_INT8_FILE,
            DECODER_INT8_FILE,
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

/// Remove the whole `<model_dir>/<size>/` folder. Refuses to run for unknown
pub fn delete_model(size: &str, model_dir: &str) -> Result<()> {
    if !valid_model_size(size) {
        bail!("Unknown Parakeet model size '{size}'");
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

/// Fetch the model files for `size` into `<model_dir>/<size>/`.
pub async fn download_model(size: &str, model_dir: &str) -> Result<()> {
    if !valid_model_size(size) {
        bail!(
            "Unknown Parakeet model size '{size}' (expected 'tdt-0.6b-v3' or 'tdt-0.6b-v3-fp32')"
        );
    }

    let dir = model_size_dir(model_dir, size);
    tokio::fs::create_dir_all(&dir).await?;

    let _guard = DOWNLOAD_LOCK.lock().await;

    for file in model_files(size) {
        let path = dir.join(file);
        if path.exists() {
            continue;
        }
        let url = format!("{HF_BASE_URL}/{file}");
        info!("Downloading Parakeet file: {url}");
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

    info!("Parakeet '{size}' model ready in {}", dir.display());
    Ok(())
}


fn load_vocab(path: &Path) -> Result<Vec<String>> {
    let file = File::open(path).with_context(|| format!("open vocab file {}", path.display()))?;
    let reader = BufReader::new(file);
    let mut entries: Vec<(usize, String)> = Vec::new();
    let mut max_id = 0;

    for (line_idx, line_res) in reader.lines().enumerate() {
        let line = line_res?;
        let trimmed = line.trim_end();
        if trimmed.is_empty() {
            continue;
        }
        if let Some(pos) = trimmed.rfind(' ') {
            let (token, id_str) = trimmed.split_at(pos);
            if let Ok(id) = id_str.trim().parse::<usize>() {
                max_id = max_id.max(id);
                entries.push((id, token.to_string()));
                continue;
            }
        }
        max_id = max_id.max(line_idx);
        entries.push((line_idx, trimmed.to_string()));
    }

    let mut vocab = vec![String::new(); max_id + 1];
    for (id, token) in entries {
        if id < vocab.len() {
            vocab[id] = token;
        }
    }
    Ok(vocab)
}

fn detokenize(tokens: &[usize], vocab: &[String]) -> String {
    let mut raw = String::new();
    for &tok_id in tokens {
        if tok_id >= vocab.len() {
            continue;
        }
        let tok = &vocab[tok_id];
        if tok == "<blk>" || tok == "<unk>" || tok == "<pad>" || tok.starts_with("<|") || tok.is_empty() {
            continue;
        }
        raw.push_str(tok);
    }
    let converted = raw.replace('\u{2581}', " ").replace('Ġ', " ");
    converted.trim().to_string()
}


struct Loaded {
    preprocessor: Session,
    encoder: Session,
    decoder: Session,
    vocab: Vec<String>,
    targets_is_i32: bool,
    decoder_enc_shape_time_first: bool,
    logits_idx: usize,
    state_1_idx: usize,
    state_2_idx: usize,
}


pub struct ParakeetBackend {
    cfg: ParakeetConfig,
    state: Mutex<Option<Loaded>>,
    loaded: bool,
}

impl ParakeetBackend {
    pub fn new(cfg: ParakeetConfig) -> Self {
        Self {
            cfg,
            state: Mutex::new(None),
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
            feature = "parakeet-cuda",
            feature = "parakeet-coreml",
            feature = "parakeet-webgpu"
        ))]
        {
            #[cfg(all(
                feature = "parakeet-webgpu",
                not(any(feature = "parakeet-cuda", feature = "parakeet-coreml"))
            ))]
            {
                let devices = crate::webgpu::webgpu_devices();
                let name = crate::parakeet_gpu_provider().unwrap_or("WebGPU");

                if devices.is_empty() {
                    tracing::warn!(
                        "Parakeet: {name} execution provider unavailable (no plugin dll or no device); running on CPU"
                    );
                    return builder;
                }

                match builder.with_devices(devices, None) {
                    Ok(with_dev) => {
                        tracing::info!("Parakeet: {name} execution provider attached");
                        return with_dev;
                    }
                    Err(e) => {
                        tracing::warn!(
                            "Parakeet: {name} execution provider unavailable ({e}); running on CPU"
                        );
                        return e.recover();
                    }
                }
            }

            #[cfg(any(feature = "parakeet-cuda", feature = "parakeet-coreml"))]
            {
                #[cfg(feature = "parakeet-cuda")]
                let ep = ort::ep::CUDA::default().build().error_on_failure();
                #[cfg(all(feature = "parakeet-coreml", not(feature = "parakeet-cuda")))]
                let ep = ort::ep::CoreML::default().build().error_on_failure();

                let name = crate::parakeet_gpu_provider().unwrap_or("GPU");

                match builder.with_execution_providers([ep]) {
                    Ok(with_ep) => {
                        tracing::debug!("Parakeet: {name} execution provider registered");
                        return with_ep;
                    }
                    Err(e) => {
                        tracing::warn!(
                            "Parakeet: {name} execution provider unavailable ({e}); running on CPU"
                        );
                        return e.recover();
                    }
                }
            }
        }

        #[cfg(not(any(
            feature = "parakeet-cuda",
            feature = "parakeet-coreml",
            feature = "parakeet-webgpu"
        )))]
        builder
    }
}

impl TranscriptionBackend for ParakeetBackend {
    fn name(&self) -> &str {
        "parakeet"
    }

    fn load(&mut self) -> Result<()> {
        let size = &self.cfg.model_size;
        let dir = model_size_dir("", size);
        let files = model_files(size);
        if !is_model_downloaded(size, "") {
            bail!(
                "Parakeet '{size}' model is not downloaded (expected {} in {}). \
                 Open Settings -> Engine and download it, or place the files there manually.",
                files.join(", "),
                dir.display()
            );
        }

        info!(
            "Loading Parakeet '{size}' model from {} (gpu={})",
            dir.display(),
            self.cfg.gpu
        );

        let (encoder_file, decoder_file) = match size.as_str() {
            "tdt-0.6b-v3-fp32" => (ENCODER_FP32_FILE, DECODER_FP32_FILE),
            _ => (ENCODER_INT8_FILE, DECODER_INT8_FILE),
        };
        let preprocessor = Self::build_session(&dir.join(PREPROCESSOR_FILE), false)?;
        let encoder = Self::build_session(&dir.join(encoder_file), self.cfg.gpu)?;
        let decoder = Self::build_session(&dir.join(decoder_file), self.cfg.gpu)?;

        let vocab = load_vocab(&dir.join(VOCAB_FILE))?;

        let targets_is_i32 = decoder
            .inputs()
            .iter()
            .find(|i| i.name() == "targets")
            .map(|i| match i.dtype() {
                ort::value::ValueType::Tensor { ty, .. } => {
                    *ty == ort::value::TensorElementType::Int32
                }
                _ => true,
            })
            .unwrap_or(true);

        let decoder_enc_shape_time_first = decoder
            .inputs()
            .iter()
            .find(|i| i.name() == "encoder_outputs")
            .map(|i| match i.dtype() {
                ort::value::ValueType::Tensor { shape, .. } => {
                    shape.get(2).copied() == Some(ENCODER_DIM as i64)
                }
                _ => false,
            })
            .unwrap_or(false);

        let logits_idx = decoder
            .outputs()
            .iter()
            .position(|o| o.name() == "outputs")
            .unwrap_or(0);

        let state_1_idx = decoder
            .outputs()
            .iter()
            .position(|o| o.name() == "output_states_1")
            .unwrap_or(2);

        let state_2_idx = decoder
            .outputs()
            .iter()
            .position(|o| o.name() == "output_states_2")
            .unwrap_or(3);

        *self.state.lock().unwrap() = Some(Loaded {
            preprocessor,
            encoder,
            decoder,
            vocab,
            targets_is_i32,
            decoder_enc_shape_time_first,
            logits_idx,
            state_1_idx,
            state_2_idx,
        });
        self.loaded = true;
        Ok(())
    }

    fn transcribe(&self, req: &TranscribeRequest) -> Result<TranscriptionResult> {
        if !self.loaded {
            bail!("Model not loaded");
        }
        let mut guard = self.state.lock().unwrap();
        let state = guard.as_mut().context("Parakeet state not initialised")?;

        let n_samples = req.audio.len();
        if n_samples == 0 {
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
        let text = run_inference(state, &req.audio)?;
        let inference_ms = t0.elapsed().as_millis() as u32;

        Ok(TranscriptionResult {
            text: text.trim().to_string(),
            language: self.cfg.language.clone(),
            language_probability: 1.0,
            duration_ms: (n_samples / (SAMPLE_RATE / 1000)) as u32,
            inference_ms,
            word_timestamps: None,
        })
    }

    fn unload(&mut self) {
        *self.state.lock().unwrap() = None;
        self.loaded = false;
    }

    fn is_loaded(&self) -> bool {
        self.loaded
    }
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

fn run_inference(state: &mut Loaded, audio: &[f32]) -> Result<String> {
    let n_samples = audio.len();

    let wave_lens = [n_samples as i64];
    let wave_tensor =
        TensorRef::from_array_view(([1_usize, n_samples], audio)).context("build audio waveform tensor")?;
    let wave_lens_tensor = TensorRef::from_array_view(([1_usize], wave_lens.as_slice()))
        .context("build audio waveform_lens tensor")?;

    let prep_feed: Vec<(&str, SessionInputValue)> = vec![
        ("waveforms", wave_tensor.into()),
        ("waveforms_lens", wave_lens_tensor.into()),
    ];

    let prep_out = state.preprocessor.run(prep_feed).context("preprocessor run")?;
    let (feat_shape, feat_data) = prep_out[0]
        .try_extract_tensor::<f32>()
        .context("extract mel features")?;
    let feat_shape_vec = feat_shape.to_vec();

    let num_mel_frames = if feat_shape_vec.len() >= 3 {
        feat_shape_vec[2] as i64
    } else {
        (n_samples / 160) as i64
    };

    let feat_lens_vec = if prep_out.len() > 1 {
        let (_, lens_data) = prep_out[1]
            .try_extract_tensor::<i64>()
            .context("extract mel features_lens")?;
        lens_data.to_vec()
    } else {
        vec![num_mel_frames]
    };

    let audio_signal_tensor =
        TensorRef::from_array_view((feat_shape_vec.as_slice(), feat_data))
            .context("build audio_signal tensor view")?;
    let length_tensor = TensorRef::from_array_view(([1_usize], feat_lens_vec.as_slice()))
        .context("build length tensor view")?;

    let enc_feed: Vec<(&str, SessionInputValue)> = vec![
        ("audio_signal", audio_signal_tensor.into()),
        ("length", length_tensor.into()),
    ];

    let enc_out = state.encoder.run(enc_feed).context("encoder run")?;
    let (enc_shape, enc_data) = enc_out[0]
        .try_extract_tensor::<f32>()
        .context("extract encoder outputs")?;

    let channels_first = enc_shape.len() >= 3 && enc_shape[1] == ENCODER_DIM as i64;
    let t_prime = if channels_first {
        enc_shape[2] as usize
    } else if enc_shape.len() >= 3 {
        enc_shape[1] as usize
    } else {
        bail!("Unexpected encoder output rank: {:?}", enc_shape);
    };

    if t_prime == 0 {
        return Ok(String::new());
    }

    let vocab_size = state.vocab.len();
    let output_dim = vocab_size + NUM_DURATION_CLASSES;
    let blank_idx = BLANK_TOKEN_ID;

    let mut state_1 = vec![0.0f32; DECODER_NUM_LAYERS * 1 * DECODER_STATE_DIM];
    let mut state_2 = vec![0.0f32; DECODER_NUM_LAYERS * 1 * DECODER_STATE_DIM];
    let mut current_token = blank_idx;

    let mut emitted_tokens: Vec<usize> = Vec::new();
    let mut t = 0;

    let enc_slice_shape: [usize; 3] = if state.decoder_enc_shape_time_first {
        [1, 1, ENCODER_DIM]
    } else {
        [1, ENCODER_DIM, 1]
    };

    let mut enc_frame = vec![0.0f32; ENCODER_DIM];

    while t < t_prime {
        if channels_first {
            for c in 0..ENCODER_DIM {
                enc_frame[c] = enc_data[c * t_prime + t];
            }
        } else {
            let offset = t * ENCODER_DIM;
            enc_frame.copy_from_slice(&enc_data[offset..offset + ENCODER_DIM]);
        }

        let mut tokens_in_frame = 0;

        loop {
            let enc_view = TensorRef::from_array_view((enc_slice_shape, enc_frame.as_slice()))
                .context("build decoder encoder_outputs tensor")?;
            let s1_view = TensorRef::from_array_view((
                [DECODER_NUM_LAYERS, 1_usize, DECODER_STATE_DIM],
                state_1.as_slice(),
            ))
            .context("build decoder state_1 tensor")?;
            let s2_view = TensorRef::from_array_view((
                [DECODER_NUM_LAYERS, 1_usize, DECODER_STATE_DIM],
                state_2.as_slice(),
            ))
            .context("build decoder state_2 tensor")?;

            let targets32 = [current_token as i32];
            let targets64 = [current_token as i64];
            let tl32 = [1_i32];
            let tl64 = [1_i64];

            let dec_feed: Vec<(&str, SessionInputValue)> = if state.targets_is_i32 {
                let targets_view = TensorRef::from_array_view(([1_usize, 1], targets32.as_slice()))
                    .context("build targets tensor")?;
                let target_len_view = TensorRef::from_array_view(([1_usize], tl32.as_slice()))
                    .context("build target_length tensor")?;
                vec![
                    ("encoder_outputs", enc_view.into()),
                    ("targets", targets_view.into()),
                    ("target_length", target_len_view.into()),
                    ("input_states_1", s1_view.into()),
                    ("input_states_2", s2_view.into()),
                ]
            } else {
                let targets_view = TensorRef::from_array_view(([1_usize, 1], targets64.as_slice()))
                    .context("build targets tensor")?;
                let target_len_view = TensorRef::from_array_view(([1_usize], tl64.as_slice()))
                    .context("build target_length tensor")?;
                vec![
                    ("encoder_outputs", enc_view.into()),
                    ("targets", targets_view.into()),
                    ("target_length", target_len_view.into()),
                    ("input_states_1", s1_view.into()),
                    ("input_states_2", s2_view.into()),
                ]
            };

            let dec_out = state.decoder.run(dec_feed).context("decoder step run")?;
            let (_, ldata) = dec_out[state.logits_idx]
                .try_extract_tensor::<f32>()
                .context("extract decoder logits")?;

            let row_offset = if ldata.len() >= output_dim {
                ldata.len() - output_dim
            } else {
                0
            };
            let row = &ldata[row_offset..];
            let token_logits = &row[..vocab_size];
            let duration_logits = &row[vocab_size..vocab_size + NUM_DURATION_CLASSES];

            let best_token = argmax(token_logits);
            let best_duration = argmax(duration_logits);

            if best_token != blank_idx {
                emitted_tokens.push(best_token);
                current_token = best_token;

                let (_, next_s1) = dec_out[state.state_1_idx]
                    .try_extract_tensor::<f32>()
                    .context("extract next state_1")?;
                let (_, next_s2) = dec_out[state.state_2_idx]
                    .try_extract_tensor::<f32>()
                    .context("extract next state_2")?;
                state_1.copy_from_slice(next_s1);
                state_2.copy_from_slice(next_s2);

                tokens_in_frame += 1;
                if best_duration > 0 || tokens_in_frame >= MAX_TOKENS_PER_STEP {
                    t += best_duration.max(1);
                    break;
                }
            } else {
                t += best_duration.max(1);
                break;
            }
        }
    }

    Ok(detokenize(&emitted_tokens, &state.vocab))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_model_size() {
        assert!(valid_model_size("tdt-0.6b-v3"));
        assert!(valid_model_size("tdt-0.6b-v3-int8"));
        assert!(valid_model_size("tdt-0.6b-v3-fp32"));
        assert!(!valid_model_size("base"));
        assert!(!valid_model_size(""));
    }

    #[test]
    fn test_model_files_per_size() {
        assert_eq!(model_files("tdt-0.6b-v3").len(), 5);
        assert_eq!(model_files("tdt-0.6b-v3-int8").len(), 5);
        let fp32 = model_files("tdt-0.6b-v3-fp32");
        assert_eq!(fp32.len(), 6);
        assert!(fp32.contains(&ENCODER_FP32_DATA_FILE));
        assert!(!model_files("tdt-0.6b-v3").contains(&ENCODER_FP32_DATA_FILE));
    }

    #[test]
    fn test_detokenize_sentencepiece() {
        let vocab = vec![
            "Hello".to_string(),
            " world".to_string(),
            "!".to_string(),
        ];
        let tokens = vec![0, 1, 2];
        let text = detokenize(&tokens, &vocab);
        assert_eq!(text, "Hello world!");
    }
    #[test]
    fn test_inspect_decoder() {
        let path = std::path::Path::new("/home/jrufer/.local/share/fotonvoice-engine/models/parakeet/tdt-0.6b-v3/decoder_joint-model.int8.onnx");
        if !path.exists() {
            return;
        }
        let session = ParakeetBackend::build_session(path, true).unwrap();
        let outputs = session.outputs();
        let s1_idx = outputs.iter().position(|o| o.name() == "output_states_1").unwrap_or(2);
        let s2_idx = outputs.iter().position(|o| o.name() == "output_states_2").unwrap_or(3);
        let logits_idx = outputs.iter().position(|o| o.name() == "outputs").unwrap_or(0);
        assert_eq!(logits_idx, 0);
        assert_eq!(s1_idx, 2);
        assert_eq!(s2_idx, 3);
    }

    #[test]
    fn test_parakeet_transcribe_silence() {
        let _dir = model_size_dir("", "tdt-0.6b-v3");
        if !is_model_downloaded("tdt-0.6b-v3", "") {
            return;
        }
        let mut backend = ParakeetBackend::new(ParakeetConfig::default());
        backend.load().expect("load parakeet");
        let req = TranscribeRequest {
            audio: vec![0.0f32; 16000],
            language: None,
            word_timestamps: false,
            initial_prompt: None,
        };
        let res = backend.transcribe(&req).expect("transcribe silence");
        println!("Parakeet transcribe silence result: {:?}", res.text);
        assert_eq!(res.text, "");
    }
}
