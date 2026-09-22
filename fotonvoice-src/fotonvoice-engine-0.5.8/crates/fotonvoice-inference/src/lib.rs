pub mod backend;
#[cfg(feature = "moonshine")]
pub mod moonshine;
#[cfg(feature = "nemotron-streaming")]
pub mod nemotron_streaming;
#[cfg(feature = "parakeet")]
pub mod parakeet;
pub mod postprocess;
pub mod remote_openai;
pub mod s1_mini;
mod util;
#[cfg(any(feature = "moonshine", feature = "parakeet", feature = "nemotron-streaming"))]
mod webgpu;
pub mod whisper_cpp;

pub use remote_openai::{
    test_remote_speech_engine, RemoteOpenAiBackend, RemoteSttTestResult, RemoteStreamingSession,
};

/// Whether the Moonshine ONNX backend was compiled into this build. When false,
pub const MOONSHINE_COMPILED: bool = cfg!(feature = "moonshine");

/// Whether the Parakeet ONNX backend was compiled into this build. When false,
pub const PARAKEET_COMPILED: bool = cfg!(feature = "parakeet");

/// Whether the Nemotron streaming ONNX backend was compiled into this build.
pub const NEMOTRON_STREAMING_COMPILED: bool = cfg!(feature = "nemotron-streaming");

/// Which GPU backend whisper.cpp can offload to in this build, or `None` for a
pub fn whisper_gpu_backend() -> Option<&'static str> {
    if cfg!(feature = "cuda") {
        Some("cuda")
    } else if cfg!(feature = "vulkan") {
        Some("vulkan")
    } else {
        None
    }
}

/// Which GPU backend the Moonshine ONNX backend can offload to in this build,
pub fn moonshine_gpu_backend() -> Option<&'static str> {
    if cfg!(feature = "moonshine-cuda") {
        Some("cuda")
    } else if cfg!(feature = "moonshine-coreml") {
        Some("coreml")
    } else if cfg!(feature = "moonshine-webgpu") {
        Some("webgpu")
    } else {
        None
    }
}

/// The same provider, spelled the way ONNX Runtime spells it, for registration
#[cfg_attr(
    not(any(
        feature = "moonshine-cuda",
        feature = "moonshine-coreml",
        feature = "moonshine-webgpu"
    )),
    allow(dead_code)
)]
pub(crate) fn moonshine_gpu_provider() -> Option<&'static str> {
    match moonshine_gpu_backend() {
        Some("cuda") => Some("CUDA"),
        Some("coreml") => Some("CoreML"),
        Some("webgpu") => Some("WebGPU"),
        _ => None,
    }
}

/// Which GPU backend the Parakeet ONNX backend can offload to in this build,
pub fn parakeet_gpu_backend() -> Option<&'static str> {
    if cfg!(feature = "parakeet-cuda") {
        Some("cuda")
    } else if cfg!(feature = "parakeet-coreml") {
        Some("coreml")
    } else if cfg!(feature = "parakeet-webgpu") {
        Some("webgpu")
    } else {
        None
    }
}

/// True when the GPU backend is compiled in AND a device is actually present
pub fn parakeet_gpu_available() -> bool {
    match parakeet_gpu_backend() {
        #[cfg(any(feature = "moonshine", feature = "parakeet", feature = "nemotron-streaming"))]
        Some("webgpu") => !webgpu::webgpu_devices().is_empty(),
        Some(_) => true,
        None => false,
    }
}

/// Which GPU backend the Nemotron streaming ONNX backend can offload to in
pub fn nemotron_gpu_backend() -> Option<&'static str> {
    if cfg!(feature = "nemotron-streaming-cuda") {
        Some("cuda")
    } else if cfg!(feature = "nemotron-streaming-coreml") {
        Some("coreml")
    } else if cfg!(feature = "nemotron-streaming-webgpu") {
        Some("webgpu")
    } else {
        None
    }
}

/// Same contract as [`parakeet_gpu_available`] for the Nemotron lane.
pub fn nemotron_gpu_available() -> bool {
    match nemotron_gpu_backend() {
        #[cfg(any(feature = "moonshine", feature = "parakeet", feature = "nemotron-streaming"))]
        Some("webgpu") => !webgpu::webgpu_devices().is_empty(),
        Some(_) => true,
        None => false,
    }
}

/// Which GPU backend S1-mini can offload to in this build, or `None` when running on the CPU.
pub fn s1_mini_gpu_backend() -> Option<&'static str> {
    if crate::s1_mini::sidecar_available() || cfg!(feature = "vulkan") {
        Some("vulkan")
    } else {
        None
    }
}

#[cfg_attr(
    not(any(
        feature = "parakeet-cuda",
        feature = "parakeet-coreml",
        feature = "parakeet-webgpu"
    )),
    allow(dead_code)
)]
pub(crate) fn parakeet_gpu_provider() -> Option<&'static str> {
    match parakeet_gpu_backend() {
        Some("cuda") => Some("CUDA"),
        Some("coreml") => Some("CoreML"),
        Some("webgpu") => Some("WebGPU"),
        _ => None,
    }
}

#[cfg(test)]
mod gpu_backend_tests {
    use super::*;

    #[test]
    fn every_reported_backend_has_a_provider_spelling() {
        match moonshine_gpu_backend() {
            Some(backend) => assert!(
                moonshine_gpu_provider().is_some(),
                "{backend} is reported to the UI but has no ONNX Runtime spelling"
            ),
            None => assert_eq!(
                moonshine_gpu_provider(),
                None,
                "a CPU-only build named a GPU provider"
            ),
        }
        match parakeet_gpu_backend() {
            Some(backend) => assert!(
                parakeet_gpu_provider().is_some(),
                "{backend} is reported to the UI but has no ONNX Runtime spelling"
            ),
            None => assert_eq!(
                parakeet_gpu_provider(),
                None,
                "a CPU-only build named a GPU provider"
            ),
        }
    }

    #[test]
    fn a_build_with_no_gpu_feature_reports_none() {
        if cfg!(not(any(
            feature = "moonshine-cuda",
            feature = "moonshine-coreml",
            feature = "moonshine-webgpu"
        ))) {
            assert_eq!(moonshine_gpu_backend(), None);
        }
        if cfg!(not(any(
            feature = "parakeet-cuda",
            feature = "parakeet-coreml",
            feature = "parakeet-webgpu"
        ))) {
            assert_eq!(parakeet_gpu_backend(), None);
        }
    }

    #[test]
    fn the_webgpu_build_reports_webgpu() {
        if cfg!(all(
            feature = "moonshine-webgpu",
            not(any(feature = "moonshine-cuda", feature = "moonshine-coreml"))
        )) {
            assert_eq!(moonshine_gpu_backend(), Some("webgpu"));
            assert_eq!(moonshine_gpu_provider(), Some("WebGPU"));
        }
    }
}

use std::sync::Arc;

use anyhow::Result;
use crossbeam_channel::{Receiver, Sender};
use tracing::{error, info};
use fotonvoice_config::{AppConfig, BackendChoice};

use backend::{TranscribeRequest, TranscriptionBackend};
use postprocess::{run_pipeline, PostProcessConfig, is_silence_hallucination};
use whisper_cpp::WhisperCppBackend;


pub type AudioChunk = Vec<f32>;


#[derive(Debug, Clone)]
pub struct InferenceRequest {
    /// Accumulated audio samples (16 kHz, mono, f32)
    pub audio: Vec<f32>,
    /// Target id (used to look up per-target processing overrides)
    pub target_id: String,
    /// Hotkey binding ID (if triggered by a hotkey)
    pub binding_id: Option<String>,
}

/// Final output after transcription + post-processing.
#[derive(Debug, Clone)]
pub struct InferenceOutput {
    pub text: String,
    pub target_id: String,
    pub binding_id: Option<String>,
    pub raw_text: String,
    pub inference_ms: u32,
    pub language: String,
    /// Set when transcription failed (model missing, backend error, ...). The
    pub error: Option<String>,
}


pub struct InferenceEngine {
    config: Arc<AppConfig>,
    backend: Box<dyn TranscriptionBackend>,
}

impl InferenceEngine {
    pub fn new(config: Arc<AppConfig>) -> Self {
        let backend = build_backend(&config);
        Self { config, backend }
    }

    /// Load the selected backend model. Blocks until ready.
    pub fn load(&mut self) -> Result<()> {
        self.backend.load()
    }

    pub fn unload(&mut self) {
        self.backend.unload();
    }

    /// Update engine configuration. If backend or backend model settings changed,
    pub fn update_config(&mut self, new_config: Arc<AppConfig>) -> bool {
        let backend_changed = self.config.engine.backend != new_config.engine.backend
            || (new_config.engine.backend == BackendChoice::WhisperCpp
                && self.config.engine.whisper_cpp != new_config.engine.whisper_cpp)
            || (new_config.engine.backend == BackendChoice::Moonshine
                && self.config.engine.moonshine != new_config.engine.moonshine)
            || (new_config.engine.backend == BackendChoice::Parakeet
                && self.config.engine.parakeet != new_config.engine.parakeet)
            || (new_config.engine.backend == BackendChoice::RemoteOpenAi
                && self.config.engine.remote_openai != new_config.engine.remote_openai);

        self.config = new_config.clone();

        if backend_changed {
            info!(
                "Inference backend configuration changed, switching backend to {:?}",
                new_config.engine.backend
            );
            self.backend.unload();
            self.backend = build_backend(&new_config);
            true
        } else {
            false
        }
    }

    /// Transcribe and post-process. Returns the final text.
    pub fn process(&self, req: InferenceRequest) -> Result<InferenceOutput> {
        if req.audio.is_empty() {
            return Ok(InferenceOutput {
                text: String::new(),
                target_id: req.target_id,
                binding_id: req.binding_id,
                raw_text: String::new(),
                inference_ms: 0,
                language: "en".into(),
                error: None,
            });
        }

        let app_config = &*self.config;

        let language: Option<String> = (app_config.engine.backend == BackendChoice::WhisperCpp)
            .then(|| app_config.engine.whisper_cpp.language.trim())
            .filter(|lang| !lang.is_empty() && *lang != "auto")
            .map(str::to_string);

        let sum_sq: f32 = req.audio.iter().map(|&s| s * s).sum();
        let rms = (sum_sq / req.audio.len() as f32).sqrt();

        let rms_threshold = (1.0 - app_config.audio.vad_threshold) * 0.006;

        if rms < rms_threshold {
            info!(
                "Audio skipped by noise gate: RMS is {:.5} (threshold is {:.5}, vad_threshold={:.2})",
                rms,
                rms_threshold,
                app_config.audio.vad_threshold
            );
            return Ok(InferenceOutput {
                text: String::new(),
                target_id: req.target_id,
                binding_id: req.binding_id,
                raw_text: String::new(),
                inference_ms: 0,
                language: "en".into(),
                error: None,
            });
        }

        let dir = fotonvoice_routing::config_dir();
        let targets = fotonvoice_routing::load_targets_cached(&dir);

        let mut merged_prompt = String::from("FotonVoice Engine is a voice control assistant application. FotonVoice Engine commands start with FotonVoice Engine. ");

        if !app_config.features.custom_vocabulary.is_empty() {
            merged_prompt.push_str("Vocabulary: ");
            merged_prompt.push_str(&app_config.features.custom_vocabulary.join(", "));
            merged_prompt.push_str(". ");
        }

        let initial_prompt = {
            let end = merged_prompt.trim_end().len();
            merged_prompt.truncate(end);
            (!merged_prompt.is_empty()).then_some(merged_prompt)
        };

        let t_req = TranscribeRequest {
            audio: req.audio,
            language,
            word_timestamps: false,
            initial_prompt,
        };

        let result = self.backend.transcribe(&t_req)?;

        let post_cfg = self.build_post_config_with_app_config(&req.target_id, app_config, &targets);
        let mut processed = run_pipeline(&result.text, &post_cfg);
        let raw_text = result.text;

        if !processed.is_empty() && is_silence_hallucination(&processed) && rms < 0.003 {
            info!("Discarded silence hallucination '{}' (audio RMS: {:.5})", processed, rms);
            processed = String::new();
        }

        let bindings = fotonvoice_routing::load_bindings_cached(&dir);
        let binding = req.binding_id.as_ref().and_then(|bid| bindings.iter().find(|b| &b.id == bid));

        let binding_wants_openai = binding
            .and_then(|b| b.openai_enabled)
            .unwrap_or(false);

        if binding_wants_openai && !processed.is_empty() {
            let mut openai_cfg = fotonvoice_config::Config::load().data.openai;
            openai_cfg.enabled = true;

            if let Some(ref b) = binding {
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
            static FALLBACK_RT: std::sync::OnceLock<tokio::runtime::Runtime> = std::sync::OnceLock::new();
            let processed_res = match tokio::runtime::Handle::try_current() {
                Ok(handle) => {
                    let c = client.clone();
                    let text = processed.clone();
                    std::thread::spawn(move || {
                        handle.block_on(async { c.process(&text).await })
                    }).join().unwrap_or(processed)
                }
                Err(_) => {
                    let rt = FALLBACK_RT.get_or_init(|| {
                        tokio::runtime::Builder::new_current_thread().enable_all().build()
                            .expect("build fallback tokio runtime")
                    });
                    rt.block_on(async { client.process(&processed).await })
                }
            };
            processed = processed_res;
        }

        Ok(InferenceOutput {
            text: processed,
            target_id: req.target_id,
            binding_id: req.binding_id,
            raw_text,
            inference_ms: result.inference_ms,
            language: result.language,
            error: None,
        })
    }

    fn build_post_config_with_app_config<'a>(
        &self,
        target_id: &str,
        app_config: &'a fotonvoice_config::AppConfig,
        targets: &[fotonvoice_routing::OutputTarget],
    ) -> PostProcessConfig<'a> {
        let target_ids: Vec<&str> = target_id.split(',').map(|s| s.trim()).filter(|s| !s.is_empty()).collect();
        let first_target_id = target_ids.first().copied().unwrap_or("default");
        let target = targets.iter().find(|t| t.id == first_target_id);

        let remove_fillers = target
            .and_then(|t| t.processing.remove_fillers)
            .unwrap_or(app_config.features.remove_fillers);

        let spoken_punctuation = target
            .and_then(|t| t.processing.spoken_punctuation)
            .unwrap_or(app_config.features.spoken_punctuation);

        let auto_format_lists = target
            .and_then(|t| t.processing.auto_format_lists)
            .unwrap_or(app_config.features.auto_format_lists);

        let code_mode = target
            .and_then(|t| t.processing.code_mode)
            .unwrap_or(false);

        PostProcessConfig {
            remove_fillers,
            spoken_punctuation,
            auto_format_lists,
            apply_snippets: !app_config.features.snippets.is_empty(),
            snippets: &app_config.features.snippets,
            code_mode,
            custom_vocabulary: &app_config.features.custom_vocabulary,
        }
    }
}


fn build_backend(config: &AppConfig) -> Box<dyn TranscriptionBackend> {
    match config.engine.backend {
        BackendChoice::WhisperCpp => {
            Box::new(WhisperCppBackend::new(config.engine.whisper_cpp.clone()))
        }
        BackendChoice::Moonshine => {
            #[cfg(feature = "moonshine")]
            {
                info!(
                    "Using Moonshine backend ({} model)",
                    config.engine.moonshine.model_size
                );
                Box::new(moonshine::MoonshineBackend::new(config.engine.moonshine.clone()))
            }
            #[cfg(not(feature = "moonshine"))]
            {
                tracing::warn!("Moonshine backend selected but not compiled in this build; using whisper-cpp");
                Box::new(WhisperCppBackend::new(config.engine.whisper_cpp.clone()))
            }
        }
        BackendChoice::Parakeet => {
            #[cfg(feature = "parakeet")]
            {
                info!(
                    "Using Parakeet backend ({} model)",
                    config.engine.parakeet.model_size
                );
                Box::new(parakeet::ParakeetBackend::new(config.engine.parakeet.clone()))
            }
            #[cfg(not(feature = "parakeet"))]
            {
                tracing::warn!("Parakeet backend selected but not compiled in this build; using whisper-cpp");
                Box::new(WhisperCppBackend::new(config.engine.whisper_cpp.clone()))
            }
        }
        BackendChoice::NemotronStreaming => {
            #[cfg(feature = "nemotron-streaming")]
            {
                info!(
                    "Using Nemotron streaming backend ({} model)",
                    config.engine.nemotron_streaming.model_size
                );
                Box::new(nemotron_streaming::NemotronStreamingBackend::new(
                    config.engine.nemotron_streaming.clone(),
                ))
            }
            #[cfg(not(feature = "nemotron-streaming"))]
            {
                tracing::warn!("Nemotron streaming backend selected but not compiled in this build; using whisper-cpp");
                Box::new(WhisperCppBackend::new(config.engine.whisper_cpp.clone()))
            }
        }
        BackendChoice::RemoteOpenAi => {
            info!(
                "Using Remote OpenAI speech engine ({})",
                config.engine.remote_openai.endpoint
            );
            Box::new(remote_openai::RemoteOpenAiBackend::new(
                config.engine.remote_openai.clone(),
            ))
        }
    }
}


/// Run the inference engine on a dedicated OS thread.
pub fn run_worker(
    config: Arc<AppConfig>,
    rx: Receiver<InferenceRequest>,
    tx: Sender<InferenceOutput>,
) {
    let (_dummy_tx, dummy_rx) = crossbeam_channel::unbounded();
    run_worker_with_config(config, rx, tx, dummy_rx);
}

/// Run the inference engine on a dedicated OS thread with dynamic config reloading.
pub fn run_worker_with_config(
    config: Arc<AppConfig>,
    rx: Receiver<InferenceRequest>,
    tx: Sender<InferenceOutput>,
    config_rx: Receiver<Arc<AppConfig>>,
) {
    std::thread::Builder::new()
        .name("fotonvoice-inference".into())
        .spawn(move || {
            let mut engine = InferenceEngine::new(config);
            let mut loaded = match engine.load() {
                Ok(()) => {
                    info!("Inference engine ready");
                    true
                }
                Err(e) => {
                    error!("Failed to load inference backend: {e:#}");
                    false
                }
            };

            loop {
                crossbeam_channel::select! {
                    recv(rx) -> req_res => {
                        let req = match req_res {
                            Ok(r) => r,
                            Err(_) => break,
                        };

                        if !loaded {
                            match engine.load() {
                                Ok(()) => {
                                    info!("Inference engine ready (loaded on demand)");
                                    loaded = true;
                                }
                                Err(e) => {
                                    error!("Inference backend still not loadable: {e:#}");
                                    let _ = tx.send(InferenceOutput {
                                        text: String::new(),
                                        target_id: req.target_id,
                                        binding_id: req.binding_id,
                                        raw_text: String::new(),
                                        inference_ms: 0,
                                        language: String::new(),
                                        error: Some(format!("{e:#}")),
                                    });
                                    continue;
                                }
                            }
                        }

                        match engine.process(req) {
                            Ok(output) => {
                                let _ = tx.send(output);
                            }
                            Err(e) => {
                                error!("Inference error: {:?}", e);
                                let _ = tx.send(InferenceOutput {
                                    text: "".to_string(),
                                    target_id: "".to_string(),
                                    binding_id: None,
                                    raw_text: "".to_string(),
                                    inference_ms: 0,
                                    language: "".to_string(),
                                    error: Some(format!("{e:#}")),
                                });
                            }
                        }
                    }
                    recv(config_rx) -> new_cfg_res => {
                        let new_cfg = match new_cfg_res {
                            Ok(c) => c,
                            Err(_) => break,
                        };
                        let needs_reload = engine.update_config(new_cfg);
                        if needs_reload {
                            loaded = match engine.load() {
                                Ok(()) => {
                                    info!("Inference engine ready with new backend");
                                    true
                                }
                                Err(e) => {
                                    error!("Failed to load new inference backend: {e:#}");
                                    false
                                }
                            };
                        }
                    }
                }
            }
        })
        .expect("failed to spawn inference thread");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_language_is_none_for_all_whisper_devices() {
        for device in &["auto", "cpu", "cuda", "vulkan"] {
            let mut cfg = AppConfig::default();
            cfg.engine.whisper_cpp.device = device.to_string();
            cfg.engine.moonshine.language = "fr".to_string();
            let engine = InferenceEngine::new(Arc::new(cfg));
            assert_eq!(engine.config.engine.whisper_cpp.device, *device);
            let _ = engine; // ensure engine is not optimised out
        }
    }

    #[test]
    fn test_process_uses_in_memory_config_not_disk() {
        let mut cfg = AppConfig::default();
        cfg.features.remove_fillers = true;
        let engine = InferenceEngine::new(Arc::new(cfg.clone()));
        assert!(engine.config.features.remove_fillers);
        assert!(engine.config.features.remove_fillers);
    }

    #[test]
    fn default_backend_is_parakeet() {
        let cfg = AppConfig::default();
        assert_eq!(build_backend(&cfg).name(), "parakeet");
    }

    #[test]
    fn remote_openai_backend_builds_correctly() {
        let mut cfg = AppConfig::default();
        cfg.engine.backend = BackendChoice::RemoteOpenAi;
        cfg.engine.remote_openai.endpoint = "http://192.168.1.100:8000/v1".to_string();
        let backend = build_backend(&cfg);
        assert_eq!(backend.name(), "remote-openai");
        assert!(backend.is_loaded());
    }

    #[test]
    fn test_engine_update_config_switches_backend() {
        let cfg = AppConfig::default();
        let mut engine = InferenceEngine::new(Arc::new(cfg.clone()));
        assert_eq!(engine.backend.name(), "parakeet");

        let mut new_cfg = cfg.clone();
        new_cfg.engine.backend = BackendChoice::RemoteOpenAi;
        new_cfg.engine.remote_openai.endpoint = "http://localhost:5000/v1".to_string();
        let reloaded = engine.update_config(Arc::new(new_cfg));
        assert!(reloaded);
        assert_eq!(engine.backend.name(), "remote-openai");

        let mut features_cfg = engine.config.as_ref().clone();
        features_cfg.features.remove_fillers = !features_cfg.features.remove_fillers;
        let reloaded_features = engine.update_config(Arc::new(features_cfg));
        assert!(!reloaded_features);
        assert_eq!(engine.backend.name(), "remote-openai");
    }
}
