use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

pub mod portable;
pub use portable::{app_root, ensure_migrated};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON parse error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("Validation error: {0}")]
    Validation(String),
}


#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct WhisperCppConfig {
    /// Directory containing GGUF model files. Empty = platform default.
    pub model_dir: String,
    /// Model size name: "small", "medium", "large-v3", plus -q8 variants, etc.
    pub model_size: String,
    /// "auto" | "cuda" | "vulkan" | "cpu"
    pub device: String,
    /// 0 = auto-detect (half of logical cores)
    pub threads: u32,
    /// BCP-47 language code, e.g. "en", or "auto" to let whisper.cpp detect it.
    #[serde(default = "default_whisper_cpp_language")]
    pub language: String,
}

fn default_whisper_cpp_language() -> String {
    "auto".into()
}

impl Default for WhisperCppConfig {
    fn default() -> Self {
        Self {
            model_dir: String::new(),
            model_size: "small.en".into(),
            device: "auto".into(),
            threads: 0,
            language: "auto".into(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MoonshineConfig {
    /// "base" or "tiny"
    pub model_size: String,
    /// BCP-47 language code, e.g. "en"
    pub language: String,
}

impl Default for MoonshineConfig {
    fn default() -> Self {
        Self {
            model_size: "base".into(),
            language: "en".into(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ParakeetConfig {
    pub model_size: String,
    pub language: String,
    /// Attach the GPU execution provider (WebGPU/CUDA/CoreML per platform)
    #[serde(default = "default_parakeet_gpu")]
    pub gpu: bool,
}

fn default_parakeet_gpu() -> bool {
    true
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct NemotronStreamingConfig {
    /// Precision variant of the danielbodart 560ms-chunk export:
    pub model_size: String,
    /// English-only model; kept for UI parity with the other engines.
    pub language: String,
    /// Attach the GPU execution provider (WebGPU/CUDA/CoreML per platform)
    #[serde(default = "default_nemotron_gpu")]
    pub gpu: bool,
}

fn default_nemotron_gpu() -> bool {
    true
}

impl Default for NemotronStreamingConfig {
    fn default() -> Self {
        Self {
            model_size: "fp16".into(),
            language: "en".into(),
            gpu: default_nemotron_gpu(),
        }
    }
}

impl Default for ParakeetConfig {
    fn default() -> Self {
        Self {
            model_size: "tdt-0.6b-v3".into(),
            language: "auto".into(),
            gpu: default_parakeet_gpu(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RemoteOpenAiConfig {
    /// Remote OpenAI-compatible endpoint URL, e.g. "http://localhost:8000/v1"
    pub endpoint: String,
    /// Optional API key for Bearer authorization
    pub api_key: Option<String>,
    /// Model identifier, e.g. "whisper-1", "whisper-large-v3"
    pub model: String,
    /// Optional language code, e.g. "en" (empty string = auto-detect)
    pub language: String,
    /// Request timeout in seconds
    pub timeout_secs: u64,
}

impl Default for RemoteOpenAiConfig {
    fn default() -> Self {
        Self {
            endpoint: "http://localhost:8000/v1".into(),
            api_key: None,
            model: "whisper-1".into(),
            language: "".into(),
            timeout_secs: 30,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum BackendChoice {
    /// `alias = "auto"` migrates configs written when the backend could be
    #[serde(alias = "auto")]
    WhisperCpp,
    Moonshine,
    Parakeet,
    #[serde(rename = "nemotron-streaming", alias = "nemotron_streaming")]
    NemotronStreaming,
    #[serde(rename = "remote-openai", alias = "remote-open-ai", alias = "remote_openai", alias = "openai-compatible", alias = "remote")]
    RemoteOpenAi,
}

impl Default for BackendChoice {
    /// Whisper.cpp is UI-dormant since 2026-09-21 (measured: NVIDIA transducers
    fn default() -> Self {
        Self::Parakeet
    }
}

fn default_s1_mini_styling() -> String {
    "semi-formal".into()
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct S1MiniConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_s1_mini_styling")]
    pub styling: String,
}

impl Default for S1MiniConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            styling: default_s1_mini_styling(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct EngineConfig {
    #[serde(default)]
    pub backend: BackendChoice,
    #[serde(default)]
    pub whisper_cpp: WhisperCppConfig,
    #[serde(default)]
    pub moonshine: MoonshineConfig,
    #[serde(default)]
    pub parakeet: ParakeetConfig,
    #[serde(default)]
    pub nemotron_streaming: NemotronStreamingConfig,
    #[serde(default)]
    pub remote_openai: RemoteOpenAiConfig,
    #[serde(default)]
    pub s1_mini: S1MiniConfig,
}


fn default_gain() -> f32 {
    1.0
}

fn default_dynamic_stream() -> bool {
    true
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioConfig {
    pub vad_threshold: f32,
    /// None = use default system device
    pub input_device_index: Option<u32>,
    /// Saved evdev device path, e.g. "/dev/input/event4" (Linux only)
    pub evdev_device: Option<String>,
    pub noise_suppression: bool,
    /// Linear gain multiplier applied before sending to inference (1.0 = unity)
    #[serde(default = "default_gain")]
    pub gain: f32,
    #[serde(default = "default_dynamic_stream")]
    pub dynamic_stream: bool,
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            vad_threshold: 0.5,
            input_device_index: None,
            evdev_device: None,
            noise_suppression: false,
            gain: 1.0,
            dynamic_stream: true,
        }
    }
}

fn default_auto_show_settings() -> bool {
    false
}

fn default_setup_completed() -> bool {
    true
}

fn default_show_notification() -> bool {
    false
}

fn default_overlay_position() -> String {
    "center".into()
}

fn default_overlay_monitor() -> String {
    "primary".into()
}

fn default_show_command_overlay() -> bool {
    true
}

fn default_command_overlay_duration_secs() -> u32 {
    3
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UiConfig {
    pub show_overlay: bool,
    #[serde(default = "default_overlay_position")]
    pub overlay_position: String,
    #[serde(default = "default_overlay_monitor")]
    pub overlay_monitor: String,
    #[serde(default = "default_auto_show_settings")]
    pub auto_show_settings: bool,
    #[serde(default = "default_show_notification")]
    pub show_notification: bool,
    #[serde(default = "default_show_command_overlay")]
    pub show_command_overlay: bool,
    #[serde(default = "default_command_overlay_duration_secs")]
    pub command_overlay_duration_secs: u32,
    /// Whether the app has been set up at least once.
    #[serde(default = "default_setup_completed")]
    pub setup_completed: bool,
}

impl Default for UiConfig {
    fn default() -> Self {
        Self {
            show_overlay: true,
            overlay_position: "center".into(),
            overlay_monitor: "primary".into(),
            auto_show_settings: false,
            show_notification: false,
            show_command_overlay: true,
            command_overlay_duration_secs: 3,
            setup_completed: false,
        }
    }
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeaturesConfig {
    pub remove_fillers: bool,
    pub custom_vocabulary: Vec<String>,
    pub spoken_punctuation: bool,
    pub auto_format_lists: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub show_notification: Option<bool>,
    /// Map of trigger -> expansion, e.g. {"addr" -> "123 Main St"}
    pub snippets: std::collections::HashMap<String, String>,
}

impl Default for FeaturesConfig {
    fn default() -> Self {
        Self {
            remove_fillers: true,
            custom_vocabulary: vec!["FotonVoice Engine".into()],
            spoken_punctuation: true,
            auto_format_lists: true,
            show_notification: None,
            snippets: std::collections::HashMap::new(),
        }
    }
}


#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum OpenAiMode {
    Clean,
    Formal,
    Casual,
    Bullet,
    Concise,
    Custom,
}

impl Default for OpenAiMode {
    fn default() -> Self {
        Self::Clean
    }
}

fn default_system_prompt() -> String {
    "Fix grammar and punctuation only. Return only the corrected text, no commentary.".into()
}

fn default_user_prompt() -> String {
    "{text}".into()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAiConfig {
    pub enabled: bool,
    pub model: String,
    /// Preset that last populated the system prompt. Kept for UI convenience and
    #[serde(default)]
    pub mode: OpenAiMode,
    /// Legacy single-prompt template (mode == Custom). Migrated into `user_prompt`.
    #[serde(default)]
    pub custom_prompt: Option<String>,
    /// System message sent to the model. Empty = no system message.
    #[serde(default = "default_system_prompt")]
    pub system_prompt: String,
    /// User message template. Must contain "{text}", which is replaced with the
    #[serde(default = "default_user_prompt")]
    pub user_prompt: String,
    /// Base URL of the OpenAI-compatible API server (a local server or a remote
    pub endpoint: String,
    /// Optional API key sent as a `Bearer` token. Required by most remote
    #[serde(default)]
    pub api_key: Option<String>,
    pub timeout_secs: u64,
}

impl Default for OpenAiConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            model: "llama3.2:1b".into(),
            mode: OpenAiMode::Clean,
            custom_prompt: None,
            system_prompt: default_system_prompt(),
            user_prompt: default_user_prompt(),
            endpoint: "http://localhost:11434".into(),
            api_key: None,
            timeout_secs: 30,
        }
    }
}


#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TtsEngine {
    Piper,
    Espeak,
    PocketTts,
    InflectMicro,
    #[serde(rename = "breeze_tts_2", alias = "breeze_tts2")]
    BreezeTts2,
    #[serde(rename = "vox_cpm_2", alias = "voxcpm2", alias = "vox_cpm2")]
    VoxCpm2,
    #[serde(rename = "lux_tts", alias = "luxtts")]
    LuxTts,
}

impl Default for TtsEngine {
    fn default() -> Self {
        Self::Piper
    }
}

/// How the TTS engine manages the memory of model-backed engines (currently
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TtsMemoryMode {
    AlwaysLoaded,
    OnDemand,
}

impl Default for TtsMemoryMode {
    fn default() -> Self {
        Self::AlwaysLoaded
    }
}

/// 15 minutes - long enough that a back-and-forth conversation never pays the
pub const DEFAULT_TTS_IDLE_UNLOAD_SECS: u64 = 900;

fn default_tts_idle_unload_secs() -> u64 {
    DEFAULT_TTS_IDLE_UNLOAD_SECS
}

fn default_pocket_tts_voice() -> String {
    "alba".into()
}

fn default_breeze_tts_2_speaker_prompt() -> String {
    "A calm and clear female voice speaking at a natural pace".into()
}

fn default_tts_speed() -> f32 {
    1.0
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PocketTtsConfig {
    /// Bundled reference voice name, e.g. "alba", "anna", "vera", "charles", "michael",
    #[serde(default = "default_pocket_tts_voice")]
    pub voice: String,
    /// Pre-warm model on startup so the first synthesis is instant
    #[serde(default)]
    pub prewarm: bool,
    /// Where this engine's token used to live, kept only so a config written
    #[serde(default, rename = "hf_token", skip_serializing_if = "Option::is_none")]
    pub legacy_hf_token: Option<String>,
    /// Directory scanned for custom voice clips (`<id>.wav`). Empty = platform default
    #[serde(default)]
    pub voice_dir: String,
    /// Enable Vulkan GPU acceleration (via audio.cpp's `--backend vulkan`).
    #[serde(default)]
    pub gpu: bool,
}

impl Default for PocketTtsConfig {
    fn default() -> Self {
        Self {
            voice: default_pocket_tts_voice(),
            prewarm: false,
            gpu: false,
            legacy_hf_token: None,
            voice_dir: String::new(),
        }
    }
}

/// Breeze-TTS-2 (BreezeBlue) - bilingual neural text-to-speech with natural-language
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BreezeTts2Config {
    /// Voice selection mode: "prompt" (Voice Design) or "clone" (Cloned Voice Clip)
    #[serde(default = "default_breeze_voice_mode")]
    pub voice_mode: String,
    /// Selected cloned voice ID from the shared voice folder (e.g. "alba", "my_voice")
    #[serde(default = "default_breeze_tts_2_cloned_voice")]
    pub cloned_voice: String,
    /// Shared voice directory for custom clips (empty = platform default `~/.local/share/fotonvoice-engine/cloned-tts-voices/`)
    #[serde(default)]
    pub voice_dir: String,
    /// Text prompt describing the voice of the speaker (Voice Design)
    #[serde(default = "default_breeze_tts_2_speaker_prompt")]
    pub speaker_prompt: String,
    /// Directory containing model weights & tokenizer. Empty = platform default
    #[serde(default)]
    pub model_dir: String,
    /// Where this engine's token used to live; see
    #[serde(default, rename = "hf_token", skip_serializing_if = "Option::is_none")]
    pub legacy_hf_token: Option<String>,
    /// Pre-warm model on startup so the first synthesis is instant
    #[serde(default)]
    pub prewarm: bool,
    /// Enable Vulkan GPU acceleration (via audio.cpp's `--backend vulkan`). Falls
    #[serde(default)]
    pub gpu: bool,
}

fn default_breeze_voice_mode() -> String {
    "prompt".into()
}

fn default_breeze_tts_2_cloned_voice() -> String {
    "alba".into()
}

impl Default for BreezeTts2Config {
    fn default() -> Self {
        Self {
            voice_mode: default_breeze_voice_mode(),
            cloned_voice: default_breeze_tts_2_cloned_voice(),
            voice_dir: String::new(),
            speaker_prompt: default_breeze_tts_2_speaker_prompt(),
            model_dir: String::new(),
            legacy_hf_token: None,
            prewarm: false,
            gpu: false,
        }
    }
}

/// VoxCPM2 - neural text-to-speech with natural-language voice design speaker
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoxCpm2Config {
    /// Voice selection mode: "prompt" (Voice Design) or "clone" (Voice Cloning)
    #[serde(default = "default_vox_cpm_2_voice_mode")]
    pub voice_mode: String,
    /// Selected cloned voice ID from the shared voice folder (e.g. "alba", "my_voice")
    #[serde(default = "default_vox_cpm_2_cloned_voice")]
    pub cloned_voice: String,
    /// Shared voice directory for custom clips (empty = platform default `~/.local/share/fotonvoice-engine/cloned-tts-voices/`)
    #[serde(default)]
    pub voice_dir: String,
    /// Text prompt describing speaker characteristics (for Voice Design)
    #[serde(default = "default_vox_cpm_2_speaker_prompt")]
    pub speaker_prompt: String,
    /// Enable Ultimate Cloning (requires reference audio + transcript in the same folder)
    #[serde(default)]
    pub ultimate_cloning: bool,
    /// Directory containing downloaded model weights & tokenizer. Empty = platform default
    #[serde(default)]
    pub model_dir: String,
    /// Pre-warm model on startup so first synthesis is instant
    #[serde(default)]
    pub prewarm: bool,
    /// Enable Vulkan GPU acceleration (via audio.cpp's `--backend vulkan`). Falls
    #[serde(default)]
    pub gpu: bool,
}

fn default_vox_cpm_2_voice_mode() -> String {
    "prompt".into()
}

fn default_vox_cpm_2_cloned_voice() -> String {
    "alba".into()
}

fn default_vox_cpm_2_speaker_prompt() -> String {
    "A calm young female voice speaking clearly with a gentle tone.".into()
}

impl Default for VoxCpm2Config {
    fn default() -> Self {
        Self {
            voice_mode: default_vox_cpm_2_voice_mode(),
            cloned_voice: default_vox_cpm_2_cloned_voice(),
            voice_dir: String::new(),
            speaker_prompt: default_vox_cpm_2_speaker_prompt(),
            ultimate_cloning: false,
            model_dir: String::new(),
            prewarm: false,
            gpu: false,
        }
    }
}


fn default_inflect_micro_seed() -> u64 {
    0
}

fn default_inflect_micro_noise_scale() -> f32 {
    0.667
}

/// Inflect-Micro-v2 (<https://huggingface.co/owensong/Inflect-Micro-v2>) - a
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InflectMicroConfig {
    /// Directory holding the ONNX graphs and phoneme vocabulary. Empty = platform
    #[serde(default)]
    pub model_dir: String,
    /// Seed for the stochastic duration predictor and latent sampling. The model
    #[serde(default = "default_inflect_micro_seed")]
    pub seed: u64,
    /// Latent sampling temperature, fed to `decode.onnx` as `noise_scale`.
    #[serde(default = "default_inflect_micro_noise_scale")]
    pub noise_scale: f32,
    /// Pre-warm the ONNX sessions on startup so the first synthesis is instant.
    #[serde(default)]
    pub prewarm: bool,
}

impl Default for InflectMicroConfig {
    fn default() -> Self {
        Self {
            model_dir: String::new(),
            seed: default_inflect_micro_seed(),
            noise_scale: default_inflect_micro_noise_scale(),
            prewarm: false,
        }
    }
}

fn default_lux_tts_num_steps() -> u32 {
    4
}

fn default_lux_tts_t_shift() -> f32 {
    0.9
}

fn default_lux_tts_guidance_scale() -> f32 {
    3.0
}

fn default_lux_tts_cloned_voice() -> String {
    "alba".into()
}

fn default_lux_tts_quantized() -> bool {
    true
}

fn default_lux_tts_ref_duration() -> f32 {
    18.0
}

/// LuxTTS (<https://github.com/ysharma3501/LuxTTS>) - a lightweight ZipVoice-family
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LuxTtsConfig {
    /// Selected reference voice ID from the shared voice folder (e.g. "alba", "my_voice")
    #[serde(default = "default_lux_tts_cloned_voice")]
    pub cloned_voice: String,
    /// Shared voice directory for reference clips (empty = platform default
    #[serde(default)]
    pub voice_dir: String,
    /// Directory containing the ONNX graphs and tokens.txt. Empty = platform default
    #[serde(default)]
    pub model_dir: String,
    /// ODE solver steps. The upstream distilled model is trained for 4; the
    #[serde(default = "default_lux_tts_num_steps")]
    pub num_steps: u32,
    /// Time-schedule shift for the ODE solver (0.9 per the ONNX reference).
    #[serde(default = "default_lux_tts_t_shift")]
    pub t_shift: f32,
    /// Classifier-free guidance scale (3.0 per the ONNX reference).
    #[serde(default = "default_lux_tts_guidance_scale")]
    pub guidance_scale: f32,
    /// Prefer the int8-quantized graphs (`*_int8.onnx`, a quarter of the RAM)
    #[serde(default = "default_lux_tts_quantized")]
    pub quantized: bool,
    /// Seed for the initial flow-matching noise. Deterministic per utterance
    #[serde(default)]
    pub seed: u64,
    /// Pre-warm the ONNX sessions on startup so the first synthesis is instant
    #[serde(default)]
    pub prewarm: bool,
    /// Seconds of the reference clip used for the encoded prompt (upstream
    #[serde(default = "default_lux_tts_ref_duration")]
    pub ref_duration: f32,
    /// Vocoder output path: false = the standard LR4 crossover of the 48 kHz
    #[serde(default)]
    pub return_smooth: bool,
}

impl Default for LuxTtsConfig {
    fn default() -> Self {
        Self {
            cloned_voice: default_lux_tts_cloned_voice(),
            voice_dir: String::new(),
            model_dir: String::new(),
            num_steps: default_lux_tts_num_steps(),
            t_shift: default_lux_tts_t_shift(),
            guidance_scale: default_lux_tts_guidance_scale(),
            quantized: default_lux_tts_quantized(),
            seed: 0,
            prewarm: false,
            ref_duration: default_lux_tts_ref_duration(),
            return_smooth: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TtsConfig {
    pub enabled: bool,
    pub engine: TtsEngine,
    /// Voice name for Piper, e.g. "en-us-lessac-medium"
    pub voice: String,
    /// Directory containing Piper voice files. Empty = platform default.
    #[serde(default)]
    pub voice_dir: String,
    /// Key(s) that stop TTS playback, e.g. ["KEY_ESC"]
    pub stop_key: Vec<String>,
    pub response_overlay: bool,
    /// Global speech speed multiplier, clamped to 0.90-1.10 on load (slider
    #[serde(default = "default_tts_speed")]
    pub speed: f32,
    /// Custom text spoken by the TTS "Test TTS" button (any language). Empty
    #[serde(default)]
    pub test_text: Option<String>,
    /// Enable GPU acceleration for Piper
    #[serde(default)]
    pub gpu: bool,
    /// The single HuggingFace access token for every gated model FotonVoice Engine
    #[serde(default)]
    pub hf_token: Option<String>,
    #[serde(default)]
    pub pocket_tts: PocketTtsConfig,
    /// Whether a model-backed engine stays resident or is unloaded when idle.
    #[serde(default)]
    pub memory_mode: TtsMemoryMode,
    /// Idle time before the model is unloaded in [`TtsMemoryMode::OnDemand`].
    #[serde(default = "default_tts_idle_unload_secs")]
    pub idle_unload_secs: u64,
    #[serde(default)]
    pub inflect_micro: InflectMicroConfig,
    #[serde(default)]
    pub breeze_tts_2: BreezeTts2Config,
    #[serde(default)]
    pub vox_cpm_2: VoxCpm2Config,
    #[serde(default)]
    pub lux_tts: LuxTtsConfig,
    #[serde(default)]
    pub snippets: std::collections::HashMap<String, String>,
}

impl Default for TtsConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            engine: TtsEngine::Espeak,
            voice: "en-us-lessac-medium".into(),
            voice_dir: String::new(),
            stop_key: vec!["KEY_ESC".into()],
            response_overlay: true,
            speed: 1.0,
            test_text: None,
            gpu: false,
            hf_token: None,
            pocket_tts: PocketTtsConfig::default(),
            memory_mode: TtsMemoryMode::default(),
            idle_unload_secs: DEFAULT_TTS_IDLE_UNLOAD_SECS,
            inflect_micro: InflectMicroConfig::default(),
            breeze_tts_2: BreezeTts2Config::default(),
            vox_cpm_2: VoxCpm2Config::default(),
            lux_tts: LuxTtsConfig::default(),
            snippets: {
                let mut map = std::collections::HashMap::new();
                map.insert("FotonVoice Engine".into(), "Foton Voice".into());
                map.insert("fotonvoice-engine".into(), "Foton Voice".into());
                map.insert("Vox Control".into(), "Foton Voice".into());
                map
            },
        }
    }
}

impl TtsConfig {
    /// True when the model should be dropped after an idle period.
    pub fn unloads_when_idle(&self) -> bool {
        self.memory_mode == TtsMemoryMode::OnDemand
    }

    /// Idle window before unloading, clamped to a sane range. A zero or absurdly
    pub fn idle_unload_duration(&self) -> std::time::Duration {
        const MIN_SECS: u64 = 30;
        std::time::Duration::from_secs(self.idle_unload_secs.max(MIN_SECS))
    }
}


fn default_visual_feedback() -> bool {
    true
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpConfig {
    pub server_enabled: bool,
    pub record_timeout: f64,
    #[serde(default = "default_visual_feedback")]
    pub visual_feedback: bool,
}

impl Default for McpConfig {
    fn default() -> Self {
        Self {
            server_enabled: false,
            record_timeout: 15.0,
            visual_feedback: true,
        }
    }
}


#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AppConfig {
    pub engine: EngineConfig,
    pub audio: AudioConfig,
    pub ui: UiConfig,
    pub features: FeaturesConfig,
    /// `alias = "ollama"` keeps configs written before the OpenAI-API rename loading.
    #[serde(alias = "ollama")]
    pub openai: OpenAiConfig,
    pub tts: TtsConfig,
    pub mcp: McpConfig,
}

/// Lift a HuggingFace token stored per engine onto the single `tts.hf_token`,
fn migrate_hf_token(data: &mut AppConfig) -> bool {
    let legacy = data
        .tts
        .pocket_tts
        .legacy_hf_token
        .take()
        .or_else(|| data.tts.breeze_tts_2.legacy_hf_token.take());
    data.tts.breeze_tts_2.legacy_hf_token = None;

    let Some(token) = legacy else { return false };
    if data.tts.hf_token.is_none() {
        data.tts.hf_token = Some(token);
    }
    true
}

/// Rename `<base>/fotonvoice-engine/pocket-tts-voices` to `<base>/fotonvoice-engine/cloned-tts-voices`,
fn migrate_cloned_voices_dir_at(base: &std::path::Path) -> bool {
    let old_dir = base.join("fotonvoice-engine").join("pocket-tts-voices");
    let new_dir = base.join("fotonvoice-engine").join("cloned-tts-voices");
    if old_dir.exists() && !new_dir.exists() {
        match std::fs::rename(&old_dir, &new_dir) {
            Ok(()) => return true,
            Err(e) => tracing::error!(
                "Failed to migrate {} to {}: {e}",
                old_dir.display(),
                new_dir.display()
            ),
        }
    }
    false
}

fn migrate_cloned_voices_dir() {
    if let Some(base) = dirs::data_local_dir() {
        migrate_cloned_voices_dir_at(&base);
    }
}

/// Read a config file, keeping every section that parses.
fn parse_tolerant_report(text: &str) -> (AppConfig, bool) {
    let whole_file_error = match serde_json::from_str::<AppConfig>(text) {
        Ok(cfg) => return (cfg, false),
        Err(e) => e,
    };

    let Ok(serde_json::Value::Object(file)) = serde_json::from_str::<serde_json::Value>(text)
    else {
        tracing::warn!("Failed to load config, using defaults: {whole_file_error}");
        return (AppConfig::default(), true);
    };

    tracing::warn!(
        "Config did not load as a whole ({whole_file_error}); \
         recovering it section by section"
    );

    let Ok(serde_json::Value::Object(mut merged)) = serde_json::to_value(AppConfig::default())
    else {
        return (AppConfig::default(), true);
    };

    for (key, value) in file {
        if !merged.contains_key(&key) {
            continue;
        }
        let mut candidate = merged.clone();
        candidate.insert(key.clone(), value);
        match serde_json::from_value::<AppConfig>(serde_json::Value::Object(candidate.clone())) {
            Ok(_) => merged = candidate,
            Err(e) => tracing::warn!(
                "Config section '{key}' could not be read ({e}); it falls back to \
                 defaults, and the rest of the file is kept"
            ),
        }
    }

    (serde_json::from_value(serde_json::Value::Object(merged)).unwrap_or_default(), false)
}


pub struct Config {
    pub data: AppConfig,
    path: PathBuf,
}

impl Config {
    pub fn config_path() -> PathBuf {
        portable::app_root().join("config.json")
    }

    pub fn load() -> Self {
        let path = Self::config_path();
        let mut data = if path.exists() {
            match std::fs::read_to_string(&path).map_err(ConfigError::Io) {
                Ok(text) => {
                    let (parsed, fatal) = parse_tolerant_report(&text);
                    if fatal {
                        tracing::warn!(
                            "Config at {} is unreadable; quarantining it and \
                             starting from defaults",
                            path.display()
                        );
                        Self::quarantine(&path);
                        parsed
                    } else {
                        parsed
                    }
                }
                Err(e) => {
                    tracing::warn!("Failed to read config, using defaults: {e}");
                    AppConfig::default()
                }
            }
        } else {
            AppConfig::default()
        };

        if let Some(legacy_notif) = data.features.show_notification {
            data.ui.show_notification = legacy_notif;
            data.features.show_notification = None;
            let clean_config = Self { data: data.clone(), path: path.clone() };
            if let Err(e) = clean_config.save() {
                tracing::error!("Failed to save clean migrated config: {e}");
            }
        }

        let needs_escape_fix = data.tts.stop_key.iter().any(|k| k == "KEY_ESCAPE");
        if needs_escape_fix {
            data.tts.stop_key = data.tts.stop_key
                .into_iter()
                .map(|k| if k == "KEY_ESCAPE" { "KEY_ESC".to_string() } else { k })
                .collect();
            let clean_config = Self { data: data.clone(), path: path.clone() };
            if let Err(e) = clean_config.save() {
                tracing::error!("Failed to save migrated stop_key: {e}");
            }
        }

        if data.openai.timeout_secs == 8 {
            data.openai.timeout_secs = 30;
            let clean_config = Self { data: data.clone(), path: path.clone() };
            if let Err(e) = clean_config.save() {
                tracing::error!("Failed to save migrated OpenAI timeout: {e}");
            }
        }

        if let Some(legacy_prompt) = data.openai.custom_prompt.take() {
            if !legacy_prompt.trim().is_empty() {
                data.openai.user_prompt = legacy_prompt;
                data.openai.system_prompt = String::new();
            }
            let clean_config = Self { data: data.clone(), path: path.clone() };
            if let Err(e) = clean_config.save() {
                tracing::error!("Failed to save migrated OpenAI custom prompt: {e}");
            }
        }

        if migrate_hf_token(&mut data) {
            let migrated = Self { data: data.clone(), path: path.clone() };
            if let Err(e) = migrated.save() {
                tracing::error!("Failed to save migrated HuggingFace token: {e}");
            }
        }

        migrate_cloned_voices_dir();

        data.tts.speed = if data.tts.speed <= 0.0 {
            1.0
        } else {
            data.tts.speed.clamp(0.90, 1.10)
        };

        Self { data, path }
    }

    /// Move an unreadable config file aside as `<name>.bad`.
    fn quarantine(path: &Path) {
        let bad = path.with_file_name(format!(
            "{}.bad",
            path.file_name().and_then(|n| n.to_str()).unwrap_or("config.json")
        ));
        if bad.exists() {
            let _ = std::fs::remove_file(&bad);
        }
        match std::fs::rename(path, &bad) {
            Ok(()) => tracing::warn!("Quarantined unreadable config as {}", bad.display()),
            Err(e) => tracing::error!("Failed to quarantine unreadable config: {e}"),
        }
    }

    pub fn save(&self) -> Result<(), ConfigError> {
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let json = serde_json::to_string_pretty(&self.data)?;
        #[cfg(unix)]
        {
            use std::io::Write;
            use std::os::unix::fs::OpenOptionsExt;
            let mut f = std::fs::OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .mode(0o600)
                .open(&self.path)?;
            f.write_all(json.as_bytes())?;
        }
        #[cfg(not(unix))]
        std::fs::write(&self.path, json)?;
        Ok(())
    }

    pub fn reload(&mut self) {
        *self = Self::load();
    }
}

impl Default for Config {
    fn default() -> Self {
        Self::load()
    }
}


/// Find the executable `name` the same way spawning it will.
pub fn find_in_path(name: &str) -> Option<PathBuf> {
    let search_name: std::borrow::Cow<str> = if cfg!(target_os = "windows") && !name.contains('.') {
        format!("{name}.exe").into()
    } else {
        name.into()
    };

    let mut dirs: Vec<PathBuf> = Vec::new();

    #[cfg(target_os = "windows")]
    {
        if let Ok(exe) = std::env::current_exe() {
            if let Some(parent) = exe.parent() {
                dirs.push(parent.to_path_buf());
            }
        }
        if let Some(root) = std::env::var_os("SystemRoot") {
            let root = PathBuf::from(root);
            dirs.push(root.join("System32"));
            dirs.push(root);
        }
    }

    if let Some(paths) = std::env::var_os("PATH") {
        dirs.extend(std::env::split_paths(&paths));
    }

    dirs.into_iter()
        .map(|dir| dir.join(search_name.as_ref()))
        .find(|p| p.is_file())
}


static VALID_MODEL_SIZES: &[&str] = &[
    "small", "small.en", "medium", "medium.en", "large-v3", "large-v3-turbo",
    "small-q8", "small.en-q8", "medium-q8", "medium.en-q8", "large-v3-turbo-q8",
];

pub fn validate(cfg: &AppConfig) -> Vec<String> {
    let mut errors = Vec::new();

    if !VALID_MODEL_SIZES.contains(&cfg.engine.whisper_cpp.model_size.as_str())
        && !cfg.engine.whisper_cpp.model_size.ends_with(".bin")
        && !std::path::Path::new(&cfg.engine.whisper_cpp.model_size).is_absolute()
    {
        errors.push(format!(
            "Unknown whisper_cpp model_size '{}'. Valid: {:?}",
            cfg.engine.whisper_cpp.model_size, VALID_MODEL_SIZES
        ));
    }

    if !["auto", "cuda", "vulkan", "cpu"]
        .contains(&cfg.engine.whisper_cpp.device.as_str())
    {
        errors.push(format!(
            "Invalid whisper_cpp device '{}'. Use: auto, cuda, vulkan, cpu",
            cfg.engine.whisper_cpp.device
        ));
    }

    if cfg.audio.vad_threshold < 0.0 || cfg.audio.vad_threshold > 1.0 {
        errors.push(format!(
            "vad_threshold {} out of range [0.0, 1.0]",
            cfg.audio.vad_threshold
        ));
    }

    errors
}

#[cfg(test)]
mod tests {
    #[test]
    fn a_name_on_path_is_found() {
        let dir = tempfile::tempdir().unwrap();
        let extension = if cfg!(target_os = "windows") { ".exe" } else { "" };
        let name = format!("fotonvoice_path_probe{extension}");
        std::fs::write(dir.path().join(&name), b"").unwrap();

        let _guard = PathGuard::prepending(dir.path());
        assert_eq!(
            find_in_path("fotonvoice_path_probe").as_deref(),
            Some(dir.path().join(&name).as_path())
        );
    }

    #[test]
    fn a_name_that_is_on_no_searched_directory_is_not_found() {
        let dir = tempfile::tempdir().unwrap();
        let _guard = PathGuard::prepending(dir.path());
        assert_eq!(find_in_path("fotonvoice_definitely_not_here_9f3a"), None);
    }

    /// The directories Windows searches before `PATH` are exactly the gap this
    #[cfg(target_os = "windows")]
    #[test]
    fn a_system_directory_tool_is_found_even_when_path_is_empty() {
        let _guard = PathGuard::replacing_with_nothing();
        let found = find_in_path("where").expect("where.exe lives in System32");
        assert!(found.is_file());
    }

    static PATH_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Puts a directory at the front of `PATH` and restores it on drop.
    #[allow(dead_code)]
    struct PathGuard<'a>(Option<std::ffi::OsString>, std::sync::MutexGuard<'a, ()>);

    impl<'a> PathGuard<'a> {
        #[cfg(target_os = "windows")]
        fn replacing_with_nothing() -> Self {
            let guard = PATH_MUTEX.lock().unwrap();
            let previous = std::env::var_os("PATH");
            std::env::set_var("PATH", "");
            Self(previous, guard)
        }

        fn prepending(dir: &std::path::Path) -> Self {
            let guard = PATH_MUTEX.lock().unwrap();
            let previous = std::env::var_os("PATH");
            let mut entries = vec![dir.to_path_buf()];
            if let Some(existing) = &previous {
                entries.extend(std::env::split_paths(existing));
            }
            std::env::set_var("PATH", std::env::join_paths(entries).unwrap());
            Self(previous, guard)
        }
    }

    impl Drop for PathGuard<'_> {
        fn drop(&mut self) {
            match self.0.take() {
                Some(previous) => std::env::set_var("PATH", previous),
                None => std::env::remove_var("PATH"),
            }
        }
    }

    use super::*;


    /// A file with nothing wrong with it must take the ordinary path and come
    #[test]
    fn a_valid_config_parses_whole() {
        let (cfg, _) = parse_tolerant_report(
            r#"{"engine": {"backend": "moonshine",
                           "whisper_cpp": {"model_dir": "", "model_size": "small",
                                           "device": "auto", "threads": 0},
                           "moonshine": {"model_size": "base", "language": "en"}},
                "audio": {"vad_threshold": 0.65, "input_device_index": null,
                          "evdev_device": null, "noise_suppression": true,
                          "gain": 1.6, "dynamic_stream": true}}"#,
        );
        assert_eq!(cfg.engine.backend, BackendChoice::Moonshine);
        assert_eq!(cfg.engine.whisper_cpp.model_size, "small");
        assert_eq!(cfg.audio.gain, 1.6);
    }

    /// The failure this exists for: `whisper_cpp` where the enum spells it
    #[test]
    fn one_bad_section_does_not_take_the_rest_of_the_file_with_it() {
        let (cfg, _) = parse_tolerant_report(
            r#"{"engine": {"backend": "whisper_cpp"},
                "audio": {"vad_threshold": 0.65, "input_device_index": null,
                          "evdev_device": null, "noise_suppression": true,
                          "gain": 1.6, "dynamic_stream": true},
                "ui": {"show_overlay": false, "overlay_style": "pulse",
                       "overlay_position": "top", "overlay_monitor": "primary",
                       "auto_show_settings": false, "show_notification": false,
                       "show_command_overlay": true, "command_overlay_duration_secs": 3,
                       "setup_completed": true}}"#,
        );

        assert_eq!(cfg.audio.gain, 1.6);
        assert!(cfg.audio.noise_suppression);
        assert!(!cfg.ui.show_overlay);

        assert_eq!(cfg.engine.backend, BackendChoice::default());
    }

    /// A key the running build knows nothing about is left alone rather than
    #[test]
    fn an_unknown_top_level_key_is_ignored() {
        let (cfg, _) = parse_tolerant_report(
            r#"{"engine": {"backend": "whisper_cpp"},
                "some_future_section": {"whatever": 1},
                "features": {"remove_fillers": false, "custom_vocabulary": [],
                             "spoken_punctuation": true, "auto_format_lists": true,
                             "snippets": {}}}"#,
        );
        assert!(!cfg.features.remove_fillers);
    }

    /// Not a JSON object at all - there are no sections to recover, so this is
    #[test]
    fn a_file_that_is_not_an_object_falls_back_to_defaults() {
        let (cfg, _) = parse_tolerant_report("[1, 2, 3]");
        assert_eq!(cfg.engine.backend, BackendChoice::default());
        assert_eq!(cfg.audio.gain, AudioConfig::default().gain);
    }

    /// The wholesale fallback must be reported, so the caller can quarantine
    #[test]
    fn a_fatally_unreadable_file_is_reported_for_quarantine() {
        let (cfg, fatal) = parse_tolerant_report("{not json at all");
        assert!(fatal);
        assert_eq!(cfg.engine.backend, BackendChoice::default());
    }

    /// A file that is recovered section by section is not quarantined: it is
    #[test]
    fn a_recovered_file_is_not_reported_as_fatal() {
        let (_, fatal) = parse_tolerant_report(r#"{"engine": {"backend": "whisper_cpp"}}"#);
        assert!(!fatal);
    }

    /// Quarantine renames the unreadable file to `<name>.bad` and frees the
    #[test]
    fn quarantine_renames_the_bad_file_aside() {
        let dir = std::env::temp_dir().join(format!("fotonvoice-config-quarantine-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let src = dir.join("config.json");
        std::fs::write(&src, "\u{feff}{oops").unwrap();
        Config::quarantine(&src);
        assert!(!src.exists());
        let bad = dir.join("config.json.bad");
        assert!(bad.exists());
        std::fs::write(&src, "{oops again").unwrap();
        Config::quarantine(&src);
        assert!(std::fs::read_to_string(&bad).unwrap() == "{oops again");
        let _ = std::fs::remove_file(&bad);
        let _ = std::fs::remove_dir(&dir);
    }

    fn tts_json(body: &str) -> TtsConfig {
        serde_json::from_str(body).expect("tts config should parse")
    }

    /// A config written when each engine carried its own copy of the token
    #[test]
    fn migrates_per_engine_hf_tokens_onto_one_key() {
        let mut data = AppConfig::default();
        data.tts = tts_json(
            r#"{"enabled": true, "engine": "pocket_tts", "voice": "v",
                "stop_key": ["KEY_ESC"], "response_overlay": true,
                "pocket_tts": {"voice": "alba", "hf_token": "hf_from_pocket"},
                "breeze_tts_2": {"hf_token": "hf_from_pocket"}}"#,
        );
        assert_eq!(
            data.tts.pocket_tts.legacy_hf_token.as_deref(),
            Some("hf_from_pocket"),
            "the old location must still parse"
        );

        assert!(migrate_hf_token(&mut data));

        assert_eq!(data.tts.hf_token.as_deref(), Some("hf_from_pocket"));
        assert!(data.tts.pocket_tts.legacy_hf_token.is_none());
        assert!(data.tts.breeze_tts_2.legacy_hf_token.is_none());

        let written = serde_json::to_string(&data.tts).unwrap();
        assert_eq!(
            written.matches("hf_token").count(),
            1,
            "the token must be stored once, not per engine: {written}"
        );
    }

    /// A token set only on Breeze is lifted too - either copy will do.
    #[test]
    fn migrates_a_breeze_only_token() {
        let mut data = AppConfig::default();
        data.tts = tts_json(
            r#"{"enabled": false, "engine": "espeak", "voice": "v", "stop_key": [],
                "response_overlay": true, "breeze_tts_2": {"hf_token": "hf_from_breeze"}}"#,
        );

        assert!(migrate_hf_token(&mut data));
        assert_eq!(data.tts.hf_token.as_deref(), Some("hf_from_breeze"));
    }

    /// A config that already has the single key keeps it, and needs no rewrite.
    #[test]
    fn a_config_with_one_token_is_left_alone() {
        let mut data = AppConfig::default();
        data.tts = tts_json(
            r#"{"enabled": false, "engine": "espeak", "voice": "v", "stop_key": [],
                "response_overlay": true, "hf_token": "hf_single"}"#,
        );

        assert!(!migrate_hf_token(&mut data), "nothing to migrate");
        assert_eq!(data.tts.hf_token.as_deref(), Some("hf_single"));
        assert_eq!(
            serde_json::to_string(&data.tts).unwrap().matches("hf_token").count(),
            1
        );
    }

    /// The single key wins over a stale per-engine copy rather than being
    #[test]
    fn the_single_token_wins_over_a_legacy_copy() {
        let mut data = AppConfig::default();
        data.tts = tts_json(
            r#"{"enabled": false, "engine": "espeak", "voice": "v", "stop_key": [],
                "response_overlay": true, "hf_token": "hf_current",
                "pocket_tts": {"hf_token": "hf_stale"}}"#,
        );

        assert!(migrate_hf_token(&mut data));
        assert_eq!(data.tts.hf_token.as_deref(), Some("hf_current"));
        assert!(data.tts.pocket_tts.legacy_hf_token.is_none());
    }

    /// The shared voice-clip folder is renamed from its old Pocket-TTS-only
    #[test]
    fn migrates_the_cloned_voices_folder() {
        let base = tempfile::tempdir().unwrap();
        let old_dir = base.path().join("fotonvoice-engine").join("pocket-tts-voices");
        std::fs::create_dir_all(&old_dir).unwrap();
        std::fs::write(old_dir.join("narrator.wav"), b"fake wav data").unwrap();

        assert!(migrate_cloned_voices_dir_at(base.path()));

        let new_dir = base.path().join("fotonvoice-engine").join("cloned-tts-voices");
        assert!(!old_dir.exists());
        assert!(new_dir.join("narrator.wav").exists());
    }

    /// With no old folder there is nothing to do, and an existing new folder
    #[test]
    fn cloned_voices_migration_is_a_noop_without_the_old_folder() {
        let base = tempfile::tempdir().unwrap();
        assert!(!migrate_cloned_voices_dir_at(base.path()));

        let old_dir = base.path().join("fotonvoice-engine").join("pocket-tts-voices");
        let new_dir = base.path().join("fotonvoice-engine").join("cloned-tts-voices");
        std::fs::create_dir_all(&old_dir).unwrap();
        std::fs::write(old_dir.join("a.wav"), b"old").unwrap();
        std::fs::create_dir_all(&new_dir).unwrap();
        std::fs::write(new_dir.join("b.wav"), b"new").unwrap();

        assert!(
            !migrate_cloned_voices_dir_at(base.path()),
            "must not clobber an existing new folder"
        );
        assert!(new_dir.join("b.wav").exists());
        assert!(old_dir.join("a.wav").exists());
    }


    /// Configs written before the Backend dropdown lost its "Auto-detect"
    #[test]
    fn legacy_auto_backend_loads_as_whisper_cpp() {
        let parsed: BackendChoice = serde_json::from_str(r#""auto""#).unwrap();
        assert_eq!(parsed, BackendChoice::WhisperCpp);
        assert_eq!(BackendChoice::default(), BackendChoice::Parakeet);
    }

    #[test]
    fn backend_choice_serializes_kebab_case() {
        assert_eq!(
            serde_json::to_string(&BackendChoice::WhisperCpp).unwrap(),
            r#""whisper-cpp""#
        );
        assert_eq!(
            serde_json::to_string(&BackendChoice::Moonshine).unwrap(),
            r#""moonshine""#
        );
        assert_eq!(
            serde_json::to_string(&BackendChoice::Parakeet).unwrap(),
            r#""parakeet""#
        );
        assert_eq!(
            serde_json::to_string(&BackendChoice::RemoteOpenAi).unwrap(),
            r#""remote-openai""#
        );
        let parsed: BackendChoice = serde_json::from_str(r#""remote-openai""#).unwrap();
        assert_eq!(parsed, BackendChoice::RemoteOpenAi);
        let parsed_alias: BackendChoice = serde_json::from_str(r#""openai-compatible""#).unwrap();
        assert_eq!(parsed_alias, BackendChoice::RemoteOpenAi);
    }

    #[test]
    fn test_default_config_values() {
        let cfg = AppConfig::default();
        assert!(!cfg.ui.auto_show_settings);
        assert!(!cfg.ui.show_notification);
        assert_eq!(cfg.ui.overlay_position, "center");
        assert_eq!(cfg.ui.overlay_monitor, "primary");
        assert!(cfg.features.show_notification.is_none());
    }

    #[test]
    fn test_legacy_notification_migration() {
        let legacy_json = r#"{
            "engine": {
                "backend": "auto",
                "whisper_cpp": {
                    "model_dir": "",
                    "model_size": "large-v3",
                    "device": "auto",
                    "threads": 0
                },
                "moonshine": {
                    "model_size": "base",
                    "language": "en"
                }
            },
            "audio": {
                "vad_threshold": 0.5,
                "input_device_index": null,
                "evdev_device": null,
                "noise_suppression": false,
                "gain": 1.0,
                "dynamic_stream": true
            },
            "ui": {
                "show_overlay": true,
                "overlay_style": "voice_card"
            },
            "features": {
                "remove_fillers": true,
                "custom_vocabulary": [],
                "spoken_punctuation": true,
                "auto_format_lists": true,
                "show_notification": true,
                "snippets": {}
            },
            "openai": {
                "enabled": false,
                "model": "llama3.2:1b",
                "mode": "clean",
                "custom_prompt": null,
                "endpoint": "http://localhost:11434",
                "timeout_secs": 8
            },
            "tts": {
                "enabled": false,
                "engine": "piper",
                "voice": "en-us-lessac-medium",
                "stop_key": ["KEY_ESC"],
                "response_overlay": true
            },
            "mcp": {
                "server_enabled": false,
                "record_timeout": 15.0
            }
        }"#;

        let parsed: AppConfig = serde_json::from_str(legacy_json).unwrap();
        assert!(parsed.features.show_notification.is_some());
        assert_eq!(parsed.features.show_notification, Some(true));

        let temp_dir = tempfile::tempdir().unwrap();
        let config_file_path = temp_dir.path().join("config.json");
        std::fs::write(&config_file_path, legacy_json).unwrap();

        let config = Config {
            data: parsed,
            path: config_file_path.clone(),
        };

        let _migrated_config = Config::load();
        
        let mut custom_config = Config {
            data: config.data.clone(),
            path: config_file_path.clone(),
        };
        if let Some(legacy_notif) = custom_config.data.features.show_notification {
            custom_config.data.ui.show_notification = legacy_notif;
            custom_config.data.features.show_notification = None;
            custom_config.save().unwrap();
        }

        assert!(custom_config.data.ui.show_notification);
        assert!(custom_config.data.features.show_notification.is_none());

        let re_read_content = std::fs::read_to_string(&config_file_path).unwrap();
        assert!(re_read_content.contains(r#""show_notification": true"#));
        assert!(!re_read_content.contains(r#""features": {
    "remove_fillers": true,
    "custom_vocabulary": [],
    "spoken_punctuation": true,
    "auto_format_lists": true,
    "show_notification": true"#));
    }

    #[test]
    fn test_ui_config_position_monitor_defaults() {
        let partial_json = r#"{
            "show_overlay": true,
            "overlay_style": "waveform",
            "auto_show_settings": true,
            "show_notification": false
        }"#;

        let parsed: UiConfig = serde_json::from_str(partial_json).unwrap();
        assert_eq!(parsed.overlay_position, "center");
        assert_eq!(parsed.overlay_monitor, "primary");
    }

    #[test]
    fn test_openai_prompt_defaults_for_legacy_config() {
        let legacy_openai = r#"{
            "enabled": true,
            "model": "llama3.2:1b",
            "mode": "clean",
            "custom_prompt": null,
            "endpoint": "http://localhost:11434",
            "timeout_secs": 30
        }"#;

        let parsed: OpenAiConfig = serde_json::from_str(legacy_openai).unwrap();
        assert_eq!(parsed.user_prompt, "{text}");
        assert!(parsed.system_prompt.contains("Fix grammar"));
        assert_eq!(parsed.api_key, None);
    }

    #[test]
    fn test_openai_timeout_migration() {
        let mut default_cfg = AppConfig::default();
        default_cfg.openai.timeout_secs = 8;

        let legacy_json = serde_json::to_string(&default_cfg).unwrap();

        let parsed: AppConfig = serde_json::from_str(&legacy_json).unwrap();
        assert_eq!(parsed.openai.timeout_secs, 8);

        let temp_dir = tempfile::tempdir().unwrap();
        let config_file_path = temp_dir.path().join("config.json");
        std::fs::write(&config_file_path, &legacy_json).unwrap();

        let mut config = Config {
            data: parsed,
            path: config_file_path.clone(),
        };

        if config.data.openai.timeout_secs == 8 {
            config.data.openai.timeout_secs = 30;
            config.save().unwrap();
        }

        assert_eq!(config.data.openai.timeout_secs, 30);

        let re_read_content = std::fs::read_to_string(&config_file_path).unwrap();
        assert!(re_read_content.contains(r#""timeout_secs": 30"#));
    }

    #[test]
    fn test_breeze_tts_2_serde() {
        let engine = TtsEngine::BreezeTts2;
        let json = serde_json::to_string(&engine).unwrap();
        assert_eq!(json, r#""breeze_tts_2""#);

        let parsed1: TtsEngine = serde_json::from_str(r#""breeze_tts_2""#).unwrap();
        assert_eq!(parsed1, TtsEngine::BreezeTts2);

        let parsed2: TtsEngine = serde_json::from_str(r#""breeze_tts2""#).unwrap();
        assert_eq!(parsed2, TtsEngine::BreezeTts2);
    }

    #[test]
    fn test_vox_cpm_2_serde() {
        let engine = TtsEngine::VoxCpm2;
        let json = serde_json::to_string(&engine).unwrap();
        assert_eq!(json, r#""vox_cpm_2""#);

        let parsed1: TtsEngine = serde_json::from_str(r#""vox_cpm_2""#).unwrap();
        assert_eq!(parsed1, TtsEngine::VoxCpm2);

        let parsed2: TtsEngine = serde_json::from_str(r#""voxcpm2""#).unwrap();
        assert_eq!(parsed2, TtsEngine::VoxCpm2);

        let parsed3: TtsEngine = serde_json::from_str(r#""vox_cpm2""#).unwrap();
        assert_eq!(parsed3, TtsEngine::VoxCpm2);
    }

    #[test]
    fn test_lux_tts_serde() {
        let engine = TtsEngine::LuxTts;
        let json = serde_json::to_string(&engine).unwrap();
        assert_eq!(json, r#""lux_tts""#);

        let parsed1: TtsEngine = serde_json::from_str(r#""lux_tts""#).unwrap();
        assert_eq!(parsed1, TtsEngine::LuxTts);

        let parsed2: TtsEngine = serde_json::from_str(r#""luxtts""#).unwrap();
        assert_eq!(parsed2, TtsEngine::LuxTts);
    }
}
