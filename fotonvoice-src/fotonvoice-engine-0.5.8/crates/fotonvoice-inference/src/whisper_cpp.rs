use std::{
    io::Read,
    path::{Path, PathBuf},
    sync::Mutex,
    time::Instant,
};

use anyhow::{bail, Context, Result};
use tracing::info;
use fotonvoice_config::WhisperCppConfig;

use crate::backend::{TranscribeRequest, TranscriptionBackend, TranscriptionResult};

// -- GGUF model resolution -----------------------------------------------------
//
// One size, one file. Presence, download, delete, and load all use the same
// single filename; there is no fallback chain. A configured size whose file
// is missing or fails the sanity check is an error the user sees, never a
// silent switch to another quant.

static GGUF_MAP: &[(&str, &str)] = &[
    ("small",             "ggml-small-q5_1.bin"),
    ("small.en",          "ggml-small.en-q5_1.bin"),
    ("medium",            "ggml-medium-q5_0.bin"),
    ("medium.en",         "ggml-medium.en-q5_0.bin"),
    ("large-v3",          "ggml-large-v3-q5_0.bin"),
    ("large-v3-turbo",    "ggml-large-v3-turbo-q5_0.bin"),
    // Q8_0 variants: near-lossless int8, preferred on GPU builds.
    ("small-q8",          "ggml-small-q8_0.bin"),
    ("small.en-q8",       "ggml-small.en-q8_0.bin"),
    ("medium-q8",         "ggml-medium-q8_0.bin"),
    ("medium.en-q8",      "ggml-medium.en-q8_0.bin"),
    ("large-v3-q8",       "ggml-large-v3-q8_0.bin"),
    ("large-v3-turbo-q8", "ggml-large-v3-turbo-q8_0.bin"),
];

const GGUF_BASE_URL: &str =
    "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/";

/// Smallest byte size any shipped model file can plausibly have. Every model
/// in GGUF_MAP is hundreds of MB, so anything under this floor is a truncated
/// or zero-length download and must never be reported as present.
const MODEL_FILE_MIN_BYTES: u64 = 1024 * 1024;

/// True when `path` is a usable Whisper model file: it exists, is large enough
/// to be real, and begins with a magic whisper.cpp accepts at load time. Both
/// on-disk variants seen here: the ggml binary format (bytes "lmgg", i.e.
/// little-endian "ggml" - measured on this machine's q5_0/q5_1/q8_0 files) and
/// GGUF ("GGUF" ASCII, per gguf.h). Existence-only checks would report a
/// zero-byte or half-written file as "downloaded", which then fails at load
/// time - the UI shows green "ready" for a model that will not run.
pub fn is_valid_model_file(path: &Path) -> bool {
    let Ok(meta) = std::fs::metadata(path) else {
        return false;
    };
    if meta.len() < MODEL_FILE_MIN_BYTES {
        return false;
    }
    let mut magic = [0u8; 4];
    match std::fs::File::open(path).and_then(|mut f| f.read_exact(&mut magic)) {
        Ok(()) => &magic == b"lmgg" || &magic == b"GGUF",
        Err(_) => false,
    }
}

/// Retired: the tiny/base sizes this auto-downloaded silently no longer ship,
/// and every remaining model is large enough that pulling it without asking
/// would be a bad surprise. All model downloads are explicit now.
pub fn is_small_auto_downloadable(_size: &str) -> bool {
    false
}

/// What a configured `device` actually resolves to in this build.
///
/// `whisper_cpp.device` chooses *whether* to offload, never *to what*: ggml
/// links one compute backend at compile time. So `cuda` on a Vulkan build runs
/// on Vulkan, and any GPU choice on a CPU build runs on the CPU.
#[derive(Debug, PartialEq, Eq)]
pub enum DeviceOutcome<'a> {
    /// The user asked for the CPU and gets it.
    Cpu,
    /// Offloading to the build's one GPU backend.
    Gpu(&'a str),
    /// A GPU was asked for and this build has none.
    NoGpuInBuild,
    /// A GPU was asked for by name, and this build has a different one.
    DifferentGpuInBuild(&'a str),
}

/// Resolve a configured device name against what this build can actually do.
///
/// The Engine tab only offers what the build has, but a config file can be
/// copied between machines or hand-edited, so a mismatch is reported rather
/// than obeyed silently.
pub fn resolve_device<'a>(requested: &str, compiled: Option<&'a str>) -> DeviceOutcome<'a> {
    if requested == "cpu" {
        return DeviceOutcome::Cpu;
    }
    match compiled {
        None => DeviceOutcome::NoGpuInBuild,
        Some(backend) if requested != "auto" && requested != backend => {
            DeviceOutcome::DifferentGpuInBuild(backend)
        }
        Some(backend) => DeviceOutcome::Gpu(backend),
    }
}

fn log_device_choice(requested: &str, compiled: Option<&str>) {
    match resolve_device(requested, compiled) {
        DeviceOutcome::Cpu => info!("Whisper acceleration: none (device = cpu)"),
        DeviceOutcome::Gpu(backend) => info!("Whisper acceleration: {backend}"),
        DeviceOutcome::NoGpuInBuild => tracing::warn!(
            "Whisper device is '{requested}', but this build has no GPU backend \
             compiled in; running on the CPU"
        ),
        DeviceOutcome::DifferentGpuInBuild(backend) => tracing::warn!(
            "Whisper device is '{requested}', but this build compiles the {backend} \
             backend; using {backend}"
        ),
    }
}

// -- Backend -------------------------------------------------------------------

pub struct WhisperCppBackend {
    cfg: WhisperCppConfig,
    model_path: Option<PathBuf>,
    loaded: bool,

    // Model context - kept alive so the state's Arc ref remains valid.
    ctx: Option<whisper_rs::WhisperContext>,

    // Inference state - KV cache + attention buffers, reused across calls.
    // WhisperState holds Arc<WhisperInnerContext> so it is self-contained; no
    // lifetime trickery needed.  We lock during each transcribe() call (the
    // inference worker is single-threaded so there is never real contention).
    state: Mutex<Option<whisper_rs::WhisperState>>,
}

impl WhisperCppBackend {
    pub fn new(cfg: WhisperCppConfig) -> Self {
        Self {
            cfg,
            model_path: None,
            loaded: false,
            ctx: None,
            state: Mutex::new(None),
        }
    }

    pub fn default_model_dir() -> PathBuf {
        crate::util::models_base_dir()
    }

    fn resolve_model_path(&self) -> Result<PathBuf> {
        let size = &self.cfg.model_size;

        // Absolute path or ends with .bin -> use directly
        if size.ends_with(".bin") || Path::new(size).is_absolute() {
            let p = PathBuf::from(size);
            if p.exists() {
                return Ok(p);
            }
            bail!("Model file not found: {}", p.display());
        }

        let model_dir = if self.cfg.model_dir.is_empty() {
            Self::default_model_dir()
        } else {
            crate::util::expand_tilde(&self.cfg.model_dir)
        };

        let filename = GGUF_MAP
            .iter()
            .find(|(name, _)| *name == size.as_str())
            .map(|(_, file)| *file)
            .ok_or_else(|| anyhow::anyhow!("Unknown model size '{size}'"))?;

        let path = model_dir.join(filename);
        if is_valid_model_file(&path) {
            return Ok(path);
        }

        bail!(
            "Whisper model '{size}' is not usable in {} (missing or not a valid GGUF/ggml file). \
             Open Settings -> Engine and download it.",
            model_dir.display()
        )
    }

    fn threads(&self) -> u32 {
        if self.cfg.threads == 0 {
            crate::util::inference_threads() as u32
        } else {
            self.cfg.threads
        }
    }
}

impl TranscriptionBackend for WhisperCppBackend {
    fn name(&self) -> &str {
        "whisper-cpp"
    }

    fn load(&mut self) -> Result<()> {
        static LOGGING_INIT: std::sync::Once = std::sync::Once::new();
        LOGGING_INIT.call_once(|| {
            whisper_rs::install_logging_hooks();
        });

        let path = self.resolve_model_path()?;
        info!("Loading whisper.cpp model: {}", path.display());

        let mut params = whisper_rs::WhisperContextParameters::default();
        let requested = self.cfg.device.to_lowercase();
        params.use_gpu = requested != "cpu";
        log_device_choice(&requested, crate::whisper_gpu_backend());

        let ctx = whisper_rs::WhisperContext::new_with_params(path.to_str().unwrap(), params)
            .context("whisper-rs load")?;

        // Create the inference state once. Allocates KV cache + attention buffers
        // up front so transcribe() never has to reallocate them.
        let state = ctx.create_state().context("whisper state init")?;

        *self.state.lock().unwrap() = Some(state);
        self.ctx = Some(ctx);
        self.model_path = Some(path);
        self.loaded = true;
        Ok(())
    }

    fn transcribe(&self, req: &TranscribeRequest) -> Result<TranscriptionResult> {
        if !self.loaded {
            bail!("Model not loaded");
        }
        let mut guard = self.state.lock().unwrap();
        let state = guard.as_mut().context("whisper state not initialised")?;
        transcribe_with_state(state, req, self.threads())
    }

    fn unload(&mut self) {
        // Drop state first so its Arc ref to the inner context is released before ctx.
        *self.state.lock().unwrap() = None;
        self.ctx = None;
        self.loaded = false;
    }

    fn is_loaded(&self) -> bool {
        self.loaded
    }
}

// -- whisper-rs transcription (reuses pre-allocated state) --------------------

fn transcribe_with_state(
    state: &mut whisper_rs::WhisperState,
    req: &TranscribeRequest,
    threads: u32,
) -> Result<TranscriptionResult> {
    use whisper_rs::{FullParams, SamplingStrategy};

    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
    params.set_n_threads(threads as i32);
    params.set_translate(false);
    params.set_print_special(false);
    params.set_print_progress(false);
    params.set_print_timestamps(false);

    if let Some(lang) = &req.language {
        params.set_language(Some(lang));
    }
    if let Some(prompt) = &req.initial_prompt {
        params.set_initial_prompt(prompt);
    }

    let t0 = Instant::now();
    state.full(params, &req.audio).context("whisper full")?;
    let inference_ms = t0.elapsed().as_millis() as u32;

    let n = state.full_n_segments();
    let mut parts = Vec::new();
    for i in 0..n {
        if let Some(segment) = state.get_segment(i) {
            if let Ok(text) = segment.to_str() {
                parts.push(text.trim().to_string());
            }
        }
    }
    let text = parts.join(" ");
    let language = req.language.clone().unwrap_or_else(|| "en".into());

    Ok(TranscriptionResult {
        text,
        language,
        language_probability: 1.0,
        duration_ms: (req.audio.len() as u32) / 16,
        inference_ms,
        word_timestamps: None,
    })
}

/// Presence is keyed to the size's single file - the same file the download
/// fetches, the delete removes, and the loader opens.
pub fn is_model_downloaded(size: &str, model_dir: &str) -> bool {
    let filename = match GGUF_MAP.iter().find(|(name, _)| *name == size) {
        Some((_, file)) => *file,
        None => return false,
    };
    let dir = if model_dir.is_empty() {
        WhisperCppBackend::default_model_dir()
    } else {
        crate::util::expand_tilde(model_dir)
    };
    is_valid_model_file(&dir.join(filename))
}

/// Remove the GGUF file for `size` from `model_dir` - the same single file the
/// presence check and download use, so "installed" and "deleted" can never
/// disagree. Refuses to run for unknown sizes so a mistyped size can never
/// point the removal at an unexpected path.
pub fn delete_model(size: &str, model_dir: &str) -> Result<()> {
    let filename = match GGUF_MAP.iter().find(|(name, _)| *name == size) {
        Some((_, file)) => *file,
        None => bail!("Unknown whisper.cpp model size '{size}'"),
    };
    let dir = if model_dir.is_empty() {
        WhisperCppBackend::default_model_dir()
    } else {
        crate::util::expand_tilde(model_dir)
    };
    let path = dir.join(filename);
    match std::fs::remove_file(&path) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => return Err(e).with_context(|| format!("Failed to delete {}", path.display())),
    }
    Ok(())
}

/// Serializes calls to `download_model`. Guards against two independent
/// triggers racing on the same file - e.g. a download trigger in Settings and
/// a retry from the transcription worker firing at the same time.
static DOWNLOAD_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

pub async fn download_model(size: &str, model_dir: &str) -> Result<()> {
    let filename = GGUF_MAP
        .iter()
        .find(|(name, _)| *name == size)
        .map(|(_, file)| *file)
        .ok_or_else(|| anyhow::anyhow!("Unknown model size '{size}'"))?;

    let model_dir = if model_dir.is_empty() {
        WhisperCppBackend::default_model_dir()
    } else {
        crate::util::expand_tilde(model_dir)
    };
    tokio::fs::create_dir_all(&model_dir).await?;

    let path = model_dir.join(filename);

    // An existing file that fails the sanity check is a broken download, not a
    // model - drop it and fetch a fresh copy rather than "succeeding".
    match is_valid_model_file(&path) {
        true => return Ok(()),
        false => {
            if path.exists() {
                tracing::warn!("Discarding invalid model file before download: {}", path.display());
                let _ = std::fs::remove_file(&path);
            }
        }
    }

    let _guard = DOWNLOAD_LOCK.lock().await;
    // Re-check after acquiring the lock: another caller may have just
    // finished downloading this exact file while we were waiting.
    match is_valid_model_file(&path) {
        true => return Ok(()),
        false => {
            if path.exists() {
                let _ = std::fs::remove_file(&path);
            }
        }
    }

    let url = format!("{}{}", GGUF_BASE_URL, filename);
    info!("Downloading Whisper model: {}", url);

    let response = reqwest::get(&url).await?.error_for_status()?;
    let bytes = response.bytes().await?;

    tokio::fs::write(&path, bytes)
        .await
        .context("save model file")?;

    // Never report a download as done unless the bytes on disk pass the same
    // check `is_model_downloaded` uses. A truncated or bad write is removed so
    // the next attempt starts clean.
    if !is_valid_model_file(&path) {
        let _ = std::fs::remove_file(&path);
        bail!(
            "Downloaded model failed verification (missing, too small, or not a GGUF file): {}",
            path.display()
        );
    }

    info!("Whisper model downloaded successfully to: {}", path.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use fotonvoice_config::WhisperCppConfig;

    // -- resolve_device --------------------------------------------------------

    /// "auto" is the default and must never produce a warning about a mismatch:
    /// it is a request to use whatever the build has.
    #[test]
    fn auto_takes_whatever_the_build_compiled() {
        assert_eq!(resolve_device("auto", Some("vulkan")), DeviceOutcome::Gpu("vulkan"));
        assert_eq!(resolve_device("auto", Some("cuda")), DeviceOutcome::Gpu("cuda"));
    }

    /// The case that made the Device dropdown misleading: a config naming CUDA
    /// on the Vulkan AppImage. It runs - on Vulkan - and says so.
    #[test]
    fn naming_the_other_gpu_backend_reports_the_one_in_the_build() {
        assert_eq!(
            resolve_device("cuda", Some("vulkan")),
            DeviceOutcome::DifferentGpuInBuild("vulkan")
        );
    }

    /// A GPU asked for in a CPU-only build silently did nothing at all before.
    #[test]
    fn asking_for_a_gpu_a_cpu_build_lacks_is_reported() {
        for requested in ["auto", "cuda", "vulkan"] {
            assert_eq!(resolve_device(requested, None), DeviceOutcome::NoGpuInBuild);
        }
    }

    #[test]
    fn cpu_is_honoured_whatever_the_build_has() {
        assert_eq!(resolve_device("cpu", Some("cuda")), DeviceOutcome::Cpu);
        assert_eq!(resolve_device("cpu", None), DeviceOutcome::Cpu);
    }

    // -- is_small_auto_downloadable --------------------------------------------

    #[test]
    fn test_small_auto_downloadable_retired() {
        assert!(!is_small_auto_downloadable("small.en"));
        assert!(!is_small_auto_downloadable("small.en-q8"));
        assert!(!is_small_auto_downloadable("large-v3"));
        assert!(!is_small_auto_downloadable("anything"));
    }

    #[test]
    fn test_is_small_auto_downloadable_rejects_larger_models() {
        for size in ["small", "medium", "large-v3", "large-v3-turbo", "small-q8", "small.en-q8"] {
            assert!(!is_small_auto_downloadable(size), "{size} must not auto-download silently");
        }
    }

    #[test]
    fn test_is_small_auto_downloadable_rejects_unknown_and_paths() {
        assert!(!is_small_auto_downloadable("unknown-size"));
        assert!(!is_small_auto_downloadable("/tmp/custom.bin"));
    }

    #[test]
    fn test_new_backend() {
        let cfg = WhisperCppConfig {
            model_dir: "/tmp".to_string(),
            model_size: "small.en".to_string(),
            device: "cpu".to_string(),
            threads: 4,
            language: "auto".to_string(),
        };
        let backend = WhisperCppBackend::new(cfg);
        assert_eq!(backend.name(), "whisper-cpp");
        assert!(!backend.is_loaded());
    }

    #[test]
    fn test_threads_calculation() {
        // Explicit threads count
        let cfg = WhisperCppConfig {
            model_dir: "".to_string(),
            model_size: "small.en".to_string(),
            device: "cpu".to_string(),
            threads: 5,
            language: "auto".to_string(),
        };
        let backend = WhisperCppBackend::new(cfg);
        assert_eq!(backend.threads(), 5);

        // Auto threads count (0)
        let cfg_auto = WhisperCppConfig {
            model_dir: "".to_string(),
            model_size: "small.en".to_string(),
            device: "cpu".to_string(),
            threads: 0,
            language: "auto".to_string(),
        };
        let backend_auto = WhisperCppBackend::new(cfg_auto);
        assert!(backend_auto.threads() >= 1);
    }

    #[test]
    fn test_resolve_model_path_absolute() {
        let cfg = WhisperCppConfig {
            model_dir: "".to_string(),
            model_size: "/tmp/nonexistent.bin".to_string(),
            device: "cpu".to_string(),
            threads: 0,
            language: "auto".to_string(),
        };
        let backend = WhisperCppBackend::new(cfg);
        // Should bail because path does not exist
        assert!(backend.resolve_model_path().is_err());
    }

    #[test]
    fn test_resolve_model_path_unknown_size() {
        let cfg = WhisperCppConfig {
            model_dir: "".to_string(),
            model_size: "invalid_size".to_string(),
            device: "cpu".to_string(),
            threads: 0,
            language: "auto".to_string(),
        };
        let backend = WhisperCppBackend::new(cfg);
        assert!(backend.resolve_model_path().is_err());
    }

    #[test]
    fn test_transcribe_unloaded_error() {
        let cfg = WhisperCppConfig {
            model_dir: "".to_string(),
            model_size: "small.en".to_string(),
            device: "cpu".to_string(),
            threads: 0,
            language: "auto".to_string(),
        };
        let backend = WhisperCppBackend::new(cfg);
        let req = crate::backend::TranscribeRequest {
            audio: vec![0.0; 16000],
            language: None,
            word_timestamps: false,
            initial_prompt: None,
        };
        assert!(backend.transcribe(&req).is_err());
    }

    // -- model_dir tests -------------------------------------------------------

    /// Write a file that passes `is_valid_model_file`: the ggml format magic
/// (bytes "lmgg") plus enough zero padding to clear the size floor.
    fn write_valid_model(dir: &Path, name: &str) {
        use std::io::Write;
        let path = dir.join(name);
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(b"lmgg").unwrap();
        let padding = vec![0u8; MODEL_FILE_MIN_BYTES as usize];
        f.write_all(&padding).unwrap();
    }

    #[test]
    fn test_is_model_downloaded_default_dir_not_present() {
        // With empty model_dir (default) and no models on disk, returns false.
        // We isolate HOME to ensure default directory is empty.
        let old_home = std::env::var_os("HOME");
        let temp_home = tempfile::tempdir().expect("create temp home");
        let home = temp_home.path().to_path_buf();
        std::env::set_var("HOME", &home);

        let result = is_model_downloaded("small.en", "");

        if let Some(old) = old_home {
            std::env::set_var("HOME", old);
        } else {
            std::env::remove_var("HOME");
        }

        assert!(!result);
    }

    #[test]
    fn test_is_model_downloaded_custom_dir_with_file() {
        let dir = tempfile::tempdir().expect("tempdir");
        write_valid_model(dir.path(), "ggml-small.en-q5_1.bin");

        assert!(is_model_downloaded(
            "small.en",
            dir.path().to_str().unwrap()
        ));
    }

    #[test]
    fn test_is_model_downloaded_rejects_empty_file() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::File::create(dir.path().join("ggml-small.en-q5_1.bin")).unwrap();

        assert!(!is_model_downloaded(
            "small.en",
            dir.path().to_str().unwrap()
        ));
    }

    #[test]
    fn test_is_model_downloaded_rejects_wrong_magic() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("ggml-small.en-q5_1.bin");
        std::fs::File::create(&path)
            .unwrap()
            .set_len(MODEL_FILE_MIN_BYTES)
            .unwrap();

        assert!(!is_model_downloaded(
            "small.en",
            dir.path().to_str().unwrap()
        ));
    }

    #[test]
    fn test_is_valid_model_file_accepts_only_real_gguf() {
        use std::io::Write;
        let dir = tempfile::tempdir().expect("tempdir");

        let missing = dir.path().join("missing.bin");
        assert!(!is_valid_model_file(&missing));

        let empty = dir.path().join("empty.bin");
        std::fs::File::create(&empty).unwrap();
        assert!(!is_valid_model_file(&empty));

        let junk_no_magic = dir.path().join("junk.bin");
        std::fs::File::create(&junk_no_magic)
            .unwrap()
            .set_len(MODEL_FILE_MIN_BYTES)
            .unwrap();
        assert!(!is_valid_model_file(&junk_no_magic));

        let too_small_with_magic = dir.path().join("small-magic.bin");
        let mut f = std::fs::File::create(&too_small_with_magic).unwrap();
        f.write_all(b"lmgg").unwrap();
        assert!(!is_valid_model_file(&too_small_with_magic));

        let good = dir.path().join("good.bin");
        write_valid_model(dir.path(), "good.bin");
        assert!(is_valid_model_file(&good));

        // GGUF-format files (ASCII "GGUF" header) must also pass.
        let gguf = dir.path().join("gguf.bin");
        let mut f = std::fs::File::create(&gguf).unwrap();
        f.write_all(b"GGUF").unwrap();
        let padding = vec![0u8; MODEL_FILE_MIN_BYTES as usize];
        f.write_all(&padding).unwrap();
        assert!(is_valid_model_file(&gguf));

        // The real files on this machine's disk use the ggml magic; reject a
        // wrong-magic file of full size to prove the magic is actually read.
        let wrong_magic = dir.path().join("wrong.bin");
        std::fs::File::create(&wrong_magic)
            .unwrap()
            .set_len(MODEL_FILE_MIN_BYTES)
            .unwrap();
        assert!(!is_valid_model_file(&wrong_magic));
    }

    #[test]
    fn test_is_model_downloaded_nonexistent_path() {
        // A path that does not exist on disk: no models found there.
        assert!(!is_model_downloaded("small.en", "/nonexistent/path/that/does/not/exist"));
    }

    #[test]
    fn test_is_model_downloaded_unknown_size() {
        // Unknown model size always returns false regardless of dir.
        assert!(!is_model_downloaded("unknown-size", ""));
        assert!(!is_model_downloaded("unknown-size", "/tmp"));
    }

    // -- Q8_0 variants --------------------------------------------------------

    #[test]
    fn test_q8_entries_point_at_their_own_q8_0_file() {
        for (name, file) in [
            ("small-q8", "ggml-small-q8_0.bin"),
            ("small.en-q8", "ggml-small.en-q8_0.bin"),
            ("medium-q8", "ggml-medium-q8_0.bin"),
            ("medium.en-q8", "ggml-medium.en-q8_0.bin"),
            ("large-v3-q8", "ggml-large-v3-q8_0.bin"),
            ("large-v3-turbo-q8", "ggml-large-v3-turbo-q8_0.bin"),
        ] {
            let (_, mapped) = GGUF_MAP
                .iter()
                .find(|(n, _)| *n == name)
                .expect("q8 entry registered");
            assert_eq!(*mapped, file, "{name} single target");
        }
    }

    /// No fallbacks: a q8 size resolves to - and only to - its own q8_0 file.
    /// A q5 file on disk must not satisfy presence or loading for a q8 size.
    #[test]
    fn test_q8_size_ignores_a_q5_file_on_disk() {
        let dir = tempfile::tempdir().expect("tempdir");
        write_valid_model(dir.path(), "ggml-small-q5_1.bin");

        assert!(!is_model_downloaded("small-q8", dir.path().to_str().unwrap()));

        let cfg = WhisperCppConfig {
            model_dir: dir.path().to_str().unwrap().to_string(),
            model_size: "small-q8".to_string(),
            device: "cpu".to_string(),
            threads: 0,
            language: "auto".to_string(),
        };
        let backend = WhisperCppBackend::new(cfg);
        assert!(backend.resolve_model_path().is_err());
    }

    #[test]
    fn test_q8_resolves_its_own_file() {
        let dir = tempfile::tempdir().expect("tempdir");
        write_valid_model(dir.path(), "ggml-small-q8_0.bin");
        let cfg = WhisperCppConfig {
            model_dir: dir.path().to_str().unwrap().to_string(),
            model_size: "small-q8".to_string(),
            device: "cpu".to_string(),
            threads: 0,
            language: "auto".to_string(),
        };
        let backend = WhisperCppBackend::new(cfg);
        let resolved = backend.resolve_model_path().expect("should resolve");
        assert_eq!(resolved, dir.path().join("ggml-small-q8_0.bin"));
    }

    #[test]
    fn test_q8_not_downloaded_in_empty_dir() {
        assert!(!is_model_downloaded("small-q8", "/nonexistent/path/for/q8"));
    }

    #[test]
    fn test_delete_model_removes_unknown_size_refused() {
        assert!(delete_model("unknown-size", "").is_err());
        assert!(delete_model("small-q8", "/nonexistent/path/that/does/not/exist").is_ok());
    }

    #[test]
    fn test_resolve_model_path_uses_custom_dir() {
        let dir = tempfile::tempdir().expect("tempdir");
        write_valid_model(dir.path(), "ggml-small.en-q5_1.bin");
        let model_path = dir.path().join("ggml-small.en-q5_1.bin");

        let cfg = WhisperCppConfig {
            model_dir: dir.path().to_str().unwrap().to_string(),
            model_size: "small.en".to_string(),
            device: "cpu".to_string(),
            threads: 0,
            language: "auto".to_string(),
        };
        let backend = WhisperCppBackend::new(cfg);
        let resolved = backend.resolve_model_path().expect("should resolve");
        assert_eq!(resolved, model_path);
    }

    #[test]
    fn test_resolve_model_path_falls_back_to_default_when_dir_empty() {
        // When model_dir is empty, resolve_model_path uses the default dir.
        // The default dir almost certainly does not contain models in CI, so we
        // just verify the error message mentions the default path rather than a
        // custom one.
        let cfg = WhisperCppConfig {
            model_dir: "".to_string(),
            model_size: "small.en".to_string(),
            device: "cpu".to_string(),
            threads: 0,
            language: "auto".to_string(),
        };
        let backend = WhisperCppBackend::new(cfg);
        let default_dir = WhisperCppBackend::default_model_dir();
        match backend.resolve_model_path() {
            Err(e) => assert!(
                e.to_string().contains(default_dir.to_str().unwrap()),
                "error should mention default dir: {e}"
            ),
            Ok(p) => {
                // If there happens to be a model on this machine, just check it's under the default dir.
                assert!(p.starts_with(&default_dir));
            }
        }
    }

    // Tilde expansion itself is covered by `crate::util` tests; this exercises
    // it through the whisper model-dir resolution path.
    #[test]
    fn test_is_model_downloaded_tilde_path() {
        let old_home = std::env::var_os("HOME");
        let temp_home = tempfile::tempdir().expect("create temp home");
        let home = temp_home.path().to_path_buf();
        
        std::env::set_var("HOME", &home);

        let dir = tempfile::tempdir_in(&home).expect("tempdir in home");
        write_valid_model(dir.path(), "ggml-small.en-q5_1.bin");

        // Construct a ~/... path pointing at the temp dir
        let rel = dir.path().strip_prefix(&home).unwrap();
        let tilde_path = format!("~/{}", rel.display());

        let result = is_model_downloaded("small.en", &tilde_path);

        if let Some(old) = old_home {
            std::env::set_var("HOME", old);
        } else {
            std::env::remove_var("HOME");
        }

        assert!(result);
    }
}

