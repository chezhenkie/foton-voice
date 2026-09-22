use serde::{Deserialize, Serialize};

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
