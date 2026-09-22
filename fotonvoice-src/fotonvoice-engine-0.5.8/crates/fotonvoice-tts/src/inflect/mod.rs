//! Inflect-Micro-v2 (<https://huggingface.co/owensong/Inflect-Micro-v2>) - a

pub mod phonemes;

#[cfg(feature = "inflect-micro")]
pub mod model;

use std::path::PathBuf;

use anyhow::{Context, Result};
use tracing::{info, warn};

use crate::piper::expand_tilde;


/// Inflect-Micro-v2 emits 24 kHz mono audio.
pub const SAMPLE_RATE: u32 = 24_000;

/// Stage 1: phoneme ids -> aligned latent sequence.
pub const DURATION_FILE: &str = "duration.onnx";
/// Stage 2: latent sequence -> waveform.
pub const DECODE_FILE: &str = "decode.onnx";

/// The graphs that must be on disk before the engine can load.
pub const MODEL_FILES: [&str; 2] = [DURATION_FILE, DECODE_FILE];

/// Whether this build includes the ONNX Runtime half of the engine.
pub const INFLECT_MICRO_COMPILED: bool = cfg!(feature = "inflect-micro");

/// A repository plus the subdirectory the export lives in.
#[derive(Debug, Clone, Copy)]
struct Layout {
    repo: &'static str,
    /// Subdirectory within the repo; empty for the repo root.
    subdir: &'static str,
}

/// Candidate upstream locations, tried in order. The publisher ships the
const CANDIDATE_LAYOUTS: [Layout; 4] = [
    Layout { repo: "owensong/Inflect-Micro-v2-ONNX", subdir: "onnx" },
    Layout { repo: "owensong/Inflect-Micro-v2-ONNX", subdir: "" },
    Layout { repo: "owensong/Inflect-Micro-v2", subdir: "onnx" },
    Layout { repo: "owensong/Inflect-Micro-v2", subdir: "" },
];

impl Layout {
    /// The hub API endpoint listing this layout's files.
    fn tree_url(&self) -> String {
        if self.subdir.is_empty() {
            format!("https://huggingface.co/api/models/{}/tree/main", self.repo)
        } else {
            format!("https://huggingface.co/api/models/{}/tree/main/{}", self.repo, self.subdir)
        }
    }

    /// The download URL for one file in this layout.
    fn file_url(&self, file: &str) -> String {
        if self.subdir.is_empty() {
            format!("https://huggingface.co/{}/resolve/main/{file}", self.repo)
        } else {
            format!("https://huggingface.co/{}/resolve/main/{}/{file}", self.repo, self.subdir)
        }
    }
}

/// Repositories searched for the ordered symbol list, most likely first.
const SYMBOL_LIST_REPOS: [&str; 2] =
    ["owensong/Inflect-Micro-v2", "owensong/Inflect-Micro-v2-ONNX"];

/// Upper bound on an auxiliary file fetched alongside the graphs. The phoneme
const MAX_AUX_FILE_BYTES: u64 = 4 * 1024 * 1024;


/// Default model directory: `<portable root>/models/inflect-micro/`, keeping
pub fn inflect_micro_model_dir() -> PathBuf {
    fotonvoice_config::portable::app_root()
        .join("models")
        .join("inflect-micro")
}

/// Resolve the configured directory, falling back to the platform default.
pub fn resolve_model_dir(model_dir: &str) -> PathBuf {
    if model_dir.is_empty() {
        inflect_micro_model_dir()
    } else {
        expand_tilde(model_dir)
    }
}

/// True when both ONNX graphs and a phoneme vocabulary are present.
pub fn is_inflect_micro_downloaded(model_dir: &str) -> bool {
    let dir = resolve_model_dir(model_dir);
    if !MODEL_FILES.iter().all(|f| dir.join(f).exists()) {
        return false;
    }
    matches!(phonemes::PhonemeVocab::load(&dir), Ok(Some(_)))
}


/// Serializes downloads so a Settings click and an on-demand load can't fight
static DOWNLOAD_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

/// Fetch the model into `model_dir` (or the platform default). Files already
pub async fn download_inflect_micro_assets(model_dir: &str) -> Result<()> {
    let dir = resolve_model_dir(model_dir);
    tokio::fs::create_dir_all(&dir)
        .await
        .with_context(|| format!("create model dir {}", dir.display()))?;

    let _guard = DOWNLOAD_LOCK.lock().await;

    let (layout, listing) = resolve_layout().await?;

    for file in MODEL_FILES {
        let path = dir.join(file);
        if path.exists() {
            continue;
        }
        let url = layout.file_url(file);
        info!("Downloading Inflect-Micro-v2 graph: {url}");
        download_to(&url, &path)
            .await
            .with_context(|| format!("download {file}"))?;
    }

    for entry in listing.iter().filter(|e| e.is_auxiliary()) {
        let path = dir.join(&entry.name);
        if path.exists() {
            continue;
        }
        let url = layout.file_url(&entry.name);
        match download_to(&url, &path).await {
            Ok(()) => info!("Downloaded Inflect-Micro-v2 auxiliary file: {}", entry.name),
            Err(e) => warn!("Could not fetch {}: {e:#}", entry.name),
        }
    }

    let mut symbol_search = Vec::new();
    if phonemes::PhonemeVocab::load(&dir)?.is_none() {
        if let Err(e) = fetch_symbol_list(&dir, &mut symbol_search).await {
            warn!("Could not fetch the Inflect-Micro-v2 symbol list: {e:#}");
        }
    }

    if phonemes::PhonemeVocab::load(&dir)?.is_none() {
        let names: Vec<&str> = listing.iter().map(|e| e.name.as_str()).collect();
        anyhow::bail!(
            "Downloaded the ONNX graphs from {}, but none of the accompanying files \
             parse as a phoneme table. The table maps eSpeak IPA to the model's \
             phoneme ids and synthesis cannot be correct without it.\n\
             Files published there: {}\n\
             Phoneme ids are positions in the ordered `symbols` list from the \
             model's text frontend, published apart from the graphs. Searching \
             for it also failed:\n{}\n\
             Download the symbol list by hand and drop it in {}.",
            layout.file_url("").trim_end_matches('/'),
            if names.is_empty() { "(none listed)".to_string() } else { names.join(", ") },
            if symbol_search.is_empty() {
                "  (no repositories reachable)".to_string()
            } else {
                symbol_search.join("\n")
            },
            dir.display()
        );
    }

    info!("Inflect-Micro-v2 ready in {}", dir.display());
    Ok(())
}

/// One file entry from the hub's tree listing.
#[derive(Debug, Clone)]
struct RepoFile {
    /// Repo-relative path, e.g. `text/symbols.py`.
    path: String,
    /// Basename, e.g. `symbols.py`.
    name: String,
    size: u64,
}

impl RepoFile {
    /// Whether this is a small non-graph file worth fetching alongside the
    fn is_auxiliary(&self) -> bool {
        if self.size > MAX_AUX_FILE_BYTES {
            return false;
        }
        let lower = self.name.to_ascii_lowercase();
        if lower.ends_with(".onnx") || lower.ends_with(".onnx_data") {
            return false;
        }
        [".json", ".txt", ".csv", ".tsv", ".py"]
            .iter()
            .any(|e| lower.ends_with(e))
    }
}

/// Find which candidate layout actually hosts the export by listing each through
async fn resolve_layout() -> Result<(Layout, Vec<RepoFile>)> {
    let client = reqwest::Client::new();
    let mut attempts = Vec::with_capacity(CANDIDATE_LAYOUTS.len());

    for layout in CANDIDATE_LAYOUTS {
        let url = layout.tree_url();
        match fetch_listing(&client, &url).await {
            Ok(files) => {
                if files.iter().any(|f| f.name == DURATION_FILE) {
                    info!("Inflect-Micro-v2 export found in {} ({} files)", url, files.len());
                    return Ok((layout, files));
                }
                attempts.push(format!("  {url} -> listed, but no {DURATION_FILE}"));
            }
            Err(e) => attempts.push(format!("  {url} -> {e}")),
        }
    }

    anyhow::bail!(
        "Could not find the Inflect-Micro-v2 ONNX export at any known location.\n\
         Tried:\n{}\n\
         If the model has moved, download {} plus its phoneme table by hand and \
         point the model directory at them in TTS settings.",
        attempts.join("\n"),
        MODEL_FILES.join(" + ")
    )
}

/// Fetch and parse one hub tree listing into its file entries (directories and
async fn fetch_listing(client: &reqwest::Client, url: &str) -> Result<Vec<RepoFile>> {
    let response = client
        .get(url)
        .send()
        .await
        .with_context(|| format!("request {url}"))?
        .error_for_status()
        .with_context(|| format!("fetch {url}"))?;
    let body = response.text().await.with_context(|| format!("read {url}"))?;
    parse_listing(&body).with_context(|| format!("parse listing from {url}"))
}

/// Locate and download the ordered symbol list.
async fn fetch_symbol_list(dir: &std::path::Path, report: &mut Vec<String>) -> Result<()> {
    let client = reqwest::Client::new();

    for repo in SYMBOL_LIST_REPOS {
        let url = format!("https://huggingface.co/api/models/{repo}/tree/main?recursive=true");
        let files = match fetch_listing(&client, &url).await {
            Ok(f) => f,
            Err(e) => {
                report.push(format!("  {repo} -> {e}"));
                continue;
            }
        };

        let candidate = files
            .iter()
            .find(|f| f.name == "symbols.py")
            .or_else(|| {
                files.iter().find(|f| {
                    f.name.ends_with(".py") && f.name.to_ascii_lowercase().contains("symbol")
                })
            });

        let Some(file) = candidate else {
            report.push(format!("  {repo} -> listed {} files, no symbols.py", files.len()));
            continue;
        };

        let file_url = format!("https://huggingface.co/{repo}/resolve/main/{}", file.path);
        let target = dir.join("symbols.py");
        match download_to(&file_url, &target).await {
            Ok(()) => {
                info!("Downloaded Inflect-Micro-v2 symbol list from {file_url}");
                return Ok(());
            }
            Err(e) => {
                report.push(format!("  {file_url} -> {e}"));
                let _ = tokio::fs::remove_file(&target).await;
            }
        }
    }

    anyhow::bail!("no symbol list found in {}", SYMBOL_LIST_REPOS.join(" or "))
}

/// Parse the hub's tree JSON: an array of `{type, path, size}` objects, where
fn parse_listing(body: &str) -> Result<Vec<RepoFile>> {
    let value: serde_json::Value = serde_json::from_str(body).context("invalid JSON")?;
    let array = value.as_array().context("expected a JSON array")?;

    let mut files = Vec::with_capacity(array.len());
    for entry in array {
        if entry.get("type").and_then(|t| t.as_str()) != Some("file") {
            continue;
        }
        let Some(path) = entry.get("path").and_then(|p| p.as_str()) else { continue };
        let name = path.rsplit('/').next().unwrap_or(path).to_string();
        if name.is_empty() {
            continue;
        }
        files.push(RepoFile {
            path: path.to_string(),
            name,
            size: entry.get("size").and_then(|s| s.as_u64()).unwrap_or(0),
        });
    }
    Ok(files)
}

/// Download one URL to `path`, writing via a `.part` temp file so an interrupted
async fn download_to(url: &str, path: &std::path::Path) -> Result<()> {
    let response = reqwest::get(url)
        .await
        .with_context(|| format!("request {url}"))?
        .error_for_status()
        .with_context(|| format!("fetch {url}"))?;
    let bytes = response
        .bytes()
        .await
        .with_context(|| format!("read {url}"))?;

    let tmp = path.with_extension("part");
    tokio::fs::write(&tmp, &bytes)
        .await
        .with_context(|| format!("write {}", tmp.display()))?;
    tokio::fs::rename(&tmp, path)
        .await
        .with_context(|| format!("finalize {}", path.display()))?;
    Ok(())
}


/// Maximum characters per synthesis call, matching `split_text`'s default in the
pub const CHUNK_LIMIT_CHARS: usize = 280;

/// Length of the taper applied to each end of a chunk, matching `edge_fade`.
pub const EDGE_FADE_MS: f32 = 5.0;

/// Silence inserted after a chunk, keyed by the punctuation that ended it.
pub fn boundary_pause_seconds(chunk: &str) -> f32 {
    match chunk.trim_end().chars().last() {
        Some('?') => 0.28,
        Some('!') => 0.24,
        Some('.') => 0.22,
        Some(';') => 0.16,
        Some(':') => 0.13,
        Some(',') => 0.09,
        _ => 0.08,
    }
}

/// Taper the first and last 5 ms of a chunk so concatenation doesn't click.
pub fn edge_fade(waveform: &mut [f32], sample_rate: u32, milliseconds: f32) {
    let frames = ((sample_rate as f32 * milliseconds / 1000.0).round() as usize)
        .min(waveform.len() / 2);
    if frames == 0 {
        return;
    }
    let last = frames.saturating_sub(1).max(1) as f32;
    for i in 0..frames {
        let ramp = i as f32 / last;
        waveform[i] *= ramp;
        let end = waveform.len() - 1 - i;
        waveform[end] *= ramp;
    }
}

/// Split `text` into synthesis chunks, following the reference `split_text`.
pub fn chunk_text(text: &str) -> Vec<String> {
    let normalized = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if normalized.is_empty() {
        return Vec::new();
    }

    let mut chunks = Vec::new();
    for sentence in split_sentences(&normalized) {
        let mut sentence = sentence;
        while sentence.chars().count() > CHUNK_LIMIT_CHARS {
            let split_at = pick_split_point(&sentence, CHUNK_LIMIT_CHARS);
            let head: String = sentence.chars().take(split_at).collect();
            let tail: String = sentence.chars().skip(split_at).collect();
            let head = head.trim().to_string();
            if !head.is_empty() {
                chunks.push(head);
            }
            sentence = tail.trim().to_string();
            if sentence.is_empty() {
                break;
            }
        }
        if !sentence.is_empty() {
            chunks.push(sentence);
        }
    }
    chunks
}

/// Choose where to cut an over-long sentence: the last `,`/`;`/`:` within the
fn pick_split_point(sentence: &str, limit: usize) -> usize {
    let chars: Vec<char> = sentence.chars().collect();
    let search_end = (limit + 1).min(chars.len());
    let window = &chars[..search_end];

    let punctuation = window
        .iter()
        .rposition(|c| matches!(c, ',' | ';' | ':'))
        .map(|i| i + 1);

    match punctuation {
        Some(p) if p >= limit / 2 => p,
        _ => match window.iter().rposition(|c| *c == ' ') {
            Some(space) if space >= limit / 2 => space,
            _ => limit,
        },
    }
}

/// Break after `.`/`!`/`?`/`;`/`:` followed by whitespace. The `regex` crate has
fn split_sentences(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let mut out = Vec::new();
    let mut current = String::new();
    let mut i = 0;

    while i < chars.len() {
        current.push(chars[i]);
        let is_terminator = matches!(chars[i], '.' | '!' | '?' | ';' | ':');
        let next_is_space = chars.get(i + 1).is_some_and(|c| c.is_whitespace());
        if is_terminator && next_is_space {
            let trimmed = current.trim();
            if !trimmed.is_empty() {
                out.push(trimmed.to_string());
            }
            current.clear();
            i += 1; // consume the separating space
        }
        i += 1;
    }

    let trimmed = current.trim();
    if !trimmed.is_empty() {
        out.push(trimmed.to_string());
    }
    out
}


/// Loads the Inflect-Micro-v2 ONNX sessions into the worker's cache if they are
#[cfg(feature = "inflect-micro")]
pub(crate) fn ensure_inflect_micro_loaded(
    config: &fotonvoice_config::TtsConfig,
    model: &mut Option<model::InflectModel>,
) -> Result<()> {
    let cfg = &config.inflect_micro;

    if !is_inflect_micro_downloaded(&cfg.model_dir) {
        anyhow::bail!(
            "Inflect-Micro-v2 model files not found in {}. Download them from TTS settings.",
            resolve_model_dir(&cfg.model_dir).display()
        );
    }

    if model.is_none() {
        let started = std::time::Instant::now();
        let dir = resolve_model_dir(&cfg.model_dir);
        *model = Some(model::InflectModel::load(&dir)?);
        tracing::info!("Inflect-Micro-v2 sessions loaded in {:?}", started.elapsed());
    }
    Ok(())
}

/// Called from `TtsEngineWorker::run` when `config.engine == TtsEngine::InflectMicro`.
#[cfg(feature = "inflect-micro")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn speak_inflect_micro(
    config: &fotonvoice_config::TtsConfig,
    u: &crate::engine::Utterance,
    model: &mut Option<model::InflectModel>,
    on_playback_start: &Option<crate::engine::PlaybackCallback>,
    sink: &rodio::Sink,
    generation_counter: &std::sync::Arc<std::sync::atomic::AtomicU32>,
    generation: u32,
) -> Result<()> {
    use std::sync::atomic::Ordering;

    let cfg = &config.inflect_micro;
    let is_prewarm = u.source_label.as_deref() == Some("prewarm");

    ensure_inflect_micro_loaded(config, model)?;
    let model = model.as_mut().unwrap();

    if is_prewarm {
        let _ = model.synthesize("warm up", cfg, config.speed, cfg.seed)?;
        return Ok(());
    }

    let chunks = chunk_text(&u.text);
    let mut callback_fired = false;

    for (index, chunk) in chunks.iter().enumerate() {
        if generation_counter.load(Ordering::SeqCst) != generation {
            break; // stop() was called - abandon the rest of the utterance
        }

        let seed = cfg.seed.wrapping_add(index as u64);
        let mut audio = model.synthesize(chunk, cfg, config.speed, seed)?;
        if audio.is_empty() {
            continue;
        }
        edge_fade(&mut audio, SAMPLE_RATE, EDGE_FADE_MS);

        if generation_counter.load(Ordering::SeqCst) != generation {
            break;
        }

        if !callback_fired {
            callback_fired = true;
            if let Some(ref cb) = on_playback_start {
                cb();
            }
        }

        if index > 0 {
            let pause = boundary_pause_seconds(&chunks[index - 1]);
            let frames = (SAMPLE_RATE as f32 * pause).round() as usize;
            if frames > 0 {
                sink.append(rodio::buffer::SamplesBuffer::new(
                    1,
                    SAMPLE_RATE,
                    vec![0.0f32; frames],
                ));
            }
        }

        tracing::debug!(
            "Inflect-Micro-v2 chunk {}/{}: {} samples",
            index + 1,
            chunks.len(),
            audio.len()
        );
        sink.append(rodio::buffer::SamplesBuffer::new(1, SAMPLE_RATE, audio));
    }

    if !callback_fired && generation_counter.load(Ordering::SeqCst) == generation {
        anyhow::bail!(
            "Inflect-Micro-v2 produced no audio for {} chunk(s) of text. The \
             phoneme frontend or the symbol table yielded an empty sequence - \
             check the log (RUST_LOG=fotonvoice_tts=info) for skipped symbols.",
            chunks.len()
        );
    }

    sink.sleep_until_end();
    Ok(())
}

/// Stand-in for builds without the `inflect-micro` feature: there is no model to
#[cfg(not(feature = "inflect-micro"))]
pub(crate) fn ensure_inflect_micro_loaded(
    _config: &fotonvoice_config::TtsConfig,
    _model: &mut Option<()>,
) -> Result<()> {
    Ok(())
}

/// Stand-in used when the crate is built without the `inflect-micro` feature, so
#[cfg(not(feature = "inflect-micro"))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn speak_inflect_micro(
    _config: &fotonvoice_config::TtsConfig,
    _u: &crate::engine::Utterance,
    _model: &mut Option<()>,
    _on_playback_start: &Option<crate::engine::PlaybackCallback>,
    _sink: &rodio::Sink,
    _generation_counter: &std::sync::Arc<std::sync::atomic::AtomicU32>,
    _generation: u32,
) -> Result<()> {
    anyhow::bail!(
        "This build of FotonVoice Engine was compiled without the `inflect-micro` feature, so \
         the Inflect-Micro-v2 engine is unavailable. Rebuild with \
         `--features inflect-micro`, or pick another TTS engine in settings."
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;


    #[test]
    fn test_model_dir_has_inflect_micro_segment() {
        assert!(inflect_micro_model_dir().ends_with("inflect-micro"));
    }

    #[test]
    fn test_resolve_model_dir_empty_uses_default() {
        assert_eq!(resolve_model_dir(""), inflect_micro_model_dir());
    }

    #[test]
    fn test_resolve_model_dir_honours_explicit_path() {
        assert_eq!(resolve_model_dir("/opt/models"), PathBuf::from("/opt/models"));
    }


    #[test]
    fn test_not_downloaded_when_dir_empty() {
        let dir = tempdir().unwrap();
        assert!(!is_inflect_micro_downloaded(dir.path().to_str().unwrap()));
    }

    #[test]
    fn test_not_downloaded_when_vocab_missing() {
        let dir = tempdir().unwrap();
        for f in MODEL_FILES {
            std::fs::write(dir.path().join(f), b"not a real graph").unwrap();
        }
        assert!(
            !is_inflect_micro_downloaded(dir.path().to_str().unwrap()),
            "graphs alone are not enough - the phoneme table is required"
        );
    }

    #[test]
    fn test_not_downloaded_when_one_graph_missing() {
        let dir = tempdir().unwrap();
        std::fs::write(dir.path().join(DURATION_FILE), b"x").unwrap();
        write_vocab(dir.path(), "phonemes.json");
        assert!(!is_inflect_micro_downloaded(dir.path().to_str().unwrap()));
    }

    #[test]
    fn test_downloaded_when_graphs_and_vocab_present() {
        let dir = tempdir().unwrap();
        for f in MODEL_FILES {
            std::fs::write(dir.path().join(f), b"x").unwrap();
        }
        write_vocab(dir.path(), "tokens.txt");
        assert!(is_inflect_micro_downloaded(dir.path().to_str().unwrap()));
    }

    #[test]
    fn test_downloaded_recognises_vocab_under_an_unexpected_name() {
        let dir = tempdir().unwrap();
        for f in MODEL_FILES {
            std::fs::write(dir.path().join(f), b"x").unwrap();
        }
        write_vocab(dir.path(), "inflect_symbols.txt");
        assert!(is_inflect_micro_downloaded(dir.path().to_str().unwrap()));
    }

    #[test]
    fn test_not_downloaded_when_only_a_hyperparameter_config_is_present() {
        let dir = tempdir().unwrap();
        for f in MODEL_FILES {
            std::fs::write(dir.path().join(f), b"x").unwrap();
        }
        std::fs::write(
            dir.path().join("config.json"),
            br#"{"sample_rate": 24000, "hidden_channels": 192, "n_layers": 6}"#,
        )
        .unwrap();
        assert!(!is_inflect_micro_downloaded(dir.path().to_str().unwrap()));
    }

    /// Write a table with enough short symbols to pass the plausibility check.
    fn write_vocab(dir: &std::path::Path, name: &str) {
        let mut body = String::new();
        for (i, c) in "_^$abdefhijklmnopqrstuvwxyzəɪˈː".chars().enumerate() {
            body.push_str(&format!("{c} {i}\n"));
        }
        if name.ends_with(".json") {
            let entries: Vec<String> = body
                .lines()
                .map(|l| {
                    let (s, i) = l.rsplit_once(' ').unwrap();
                    format!("{}: {i}", serde_json::to_string(s).unwrap())
                })
                .collect();
            std::fs::write(dir.join(name), format!("{{{}}}", entries.join(","))).unwrap();
        } else {
            std::fs::write(dir.join(name), body).unwrap();
        }
    }


    #[test]
    fn test_parse_listing_extracts_basenames_and_sizes() {
        let body = r#"[
            {"type":"file","path":"onnx/duration.onnx","size":1234},
            {"type":"file","path":"onnx/tokens.txt","size":56},
            {"type":"directory","path":"onnx/nested"}
        ]"#;
        let files = parse_listing(body).unwrap();
        assert_eq!(files.len(), 2, "directories are skipped");
        assert_eq!(files[0].name, DURATION_FILE, "subdir prefix is stripped");
        assert_eq!(files[0].size, 1234);
    }

    #[test]
    fn test_parse_listing_keeps_full_path_for_nested_files() {
        let body = r#"[{"type":"file","path":"text/symbols.py","size":900}]"#;
        let files = parse_listing(body).unwrap();
        assert_eq!(files[0].path, "text/symbols.py");
        assert_eq!(files[0].name, "symbols.py");
    }

    #[test]
    fn test_parse_listing_rejects_non_array() {
        assert!(parse_listing(r#"{"error":"not found"}"#).is_err());
    }

    #[test]
    fn test_parse_listing_tolerates_missing_size() {
        let files = parse_listing(r#"[{"type":"file","path":"tokens.txt"}]"#).unwrap();
        assert_eq!(files[0].size, 0);
    }


    #[test]
    fn test_auxiliary_selects_small_text_files() {
        for name in ["tokens.txt", "phonemes.json", "symbols.csv"] {
            let f = RepoFile { path: name.into(), name: name.into(), size: 4096 };
            assert!(f.is_auxiliary(), "{name} should be fetched");
        }
    }

    #[test]
    fn test_auxiliary_includes_reference_scripts() {
        for name in ["inference_onnx.py", "export_onnx.py"] {
            let f = RepoFile { path: name.into(), name: name.into(), size: 8192 };
            assert!(f.is_auxiliary(), "{name} should be fetched");
        }
    }

    #[test]
    fn test_auxiliary_skips_graphs_and_large_files() {
        assert!(!RepoFile { path: "decode.onnx".into(), name: "decode.onnx".into(), size: 100 }.is_auxiliary());
        assert!(!RepoFile { path: "model.onnx_data".into(), name: "model.onnx_data".into(), size: 100 }.is_auxiliary());
        assert!(
            !RepoFile { path: "huge.json".into(), name: "huge.json".into(), size: MAX_AUX_FILE_BYTES + 1 }.is_auxiliary(),
            "oversized files are not auxiliary"
        );
        assert!(!RepoFile { path: "README.md".into(), name: "README.md".into(), size: 100 }.is_auxiliary());
    }


    #[test]
    fn test_layout_urls_with_subdir() {
        let l = Layout { repo: "owner/repo", subdir: "onnx" };
        assert_eq!(l.tree_url(), "https://huggingface.co/api/models/owner/repo/tree/main/onnx");
        assert_eq!(l.file_url("a.onnx"), "https://huggingface.co/owner/repo/resolve/main/onnx/a.onnx");
    }

    #[test]
    fn test_layout_urls_at_repo_root() {
        let l = Layout { repo: "owner/repo", subdir: "" };
        assert_eq!(l.tree_url(), "https://huggingface.co/api/models/owner/repo/tree/main");
        assert_eq!(l.file_url("a.onnx"), "https://huggingface.co/owner/repo/resolve/main/a.onnx");
    }

    #[test]
    fn test_known_good_layout_is_tried_first() {
        assert_eq!(CANDIDATE_LAYOUTS[0].repo, "owensong/Inflect-Micro-v2-ONNX");
        assert_eq!(CANDIDATE_LAYOUTS[0].subdir, "onnx");
    }


    #[test]
    fn test_split_sentences_keeps_terminal_punctuation() {
        let s = split_sentences("One. Two! Three?");
        assert_eq!(s, vec!["One.", "Two!", "Three?"]);
    }

    #[test]
    fn test_split_sentences_handles_trailing_fragment() {
        let s = split_sentences("Complete. Incomplete");
        assert_eq!(s, vec!["Complete.", "Incomplete"]);
    }

    #[test]
    fn test_split_sentences_requires_whitespace_after_terminator() {
        assert_eq!(split_sentences("pi is 3.14 exactly"), vec!["pi is 3.14 exactly"]);
    }


    #[test]
    fn test_boundary_pause_varies_by_terminator() {
        assert!(boundary_pause_seconds("what?") > boundary_pause_seconds("stop."));
        assert!(boundary_pause_seconds("stop.") > boundary_pause_seconds("and,"));
    }

    #[test]
    fn test_boundary_pause_defaults_for_unpunctuated_text() {
        assert_eq!(boundary_pause_seconds("no terminator"), 0.08);
        assert_eq!(boundary_pause_seconds(""), 0.08);
    }

    #[test]
    fn test_edge_fade_tapers_both_ends() {
        let mut wave = vec![1.0f32; 4800];
        edge_fade(&mut wave, SAMPLE_RATE, EDGE_FADE_MS);
        assert_eq!(wave[0], 0.0, "starts silent");
        assert_eq!(*wave.last().unwrap(), 0.0, "ends silent");
        assert_eq!(wave[2400], 1.0, "middle is untouched");
    }

    #[test]
    fn test_edge_fade_handles_short_buffers() {
        let mut wave = vec![1.0f32; 3];
        edge_fade(&mut wave, SAMPLE_RATE, EDGE_FADE_MS);
        assert!(wave.iter().all(|s| s.is_finite()));
        let mut empty: Vec<f32> = Vec::new();
        edge_fade(&mut empty, SAMPLE_RATE, EDGE_FADE_MS);
    }

    #[test]
    fn test_split_sentences_empty_input() {
        assert!(split_sentences("").is_empty());
        assert!(split_sentences("   \n  ").is_empty());
    }


    #[test]
    fn test_chunk_text_gives_each_sentence_its_own_chunk() {
        let chunks = chunk_text("One. Two. Three.");
        assert_eq!(chunks, vec!["One.", "Two.", "Three."]);
    }

    #[test]
    fn test_chunk_text_normalises_whitespace() {
        let chunks = chunk_text("  One.\n\n  Two.  ");
        assert_eq!(chunks, vec!["One.", "Two."]);
    }

    #[test]
    fn test_chunk_text_splits_sentence_over_the_limit() {
        let long = "word ".repeat(100);
        let chunks = chunk_text(&format!("{long}end."));
        assert!(chunks.len() > 1, "a 500-char sentence must be split");
        for c in &chunks {
            assert!(
                c.chars().count() <= CHUNK_LIMIT_CHARS,
                "chunk of {} chars exceeds the limit",
                c.chars().count()
            );
        }
    }

    #[test]
    fn test_chunk_text_splits_on_inner_punctuation_when_available() {
        let head = "a".repeat(200);
        let tail = "b".repeat(200);
        let chunks = chunk_text(&format!("{head}, {tail}"));
        assert!(chunks[0].ends_with(','), "should break at the comma: {:?}", chunks[0]);
    }

    #[test]
    fn test_chunk_text_breaks_on_semicolon_and_colon() {
        assert_eq!(chunk_text("one; two: three"), vec!["one;", "two:", "three"]);
    }

    #[test]
    fn test_chunk_text_empty_input_yields_no_chunks() {
        assert!(chunk_text("").is_empty());
        assert!(chunk_text("   ").is_empty());
    }

    #[test]
    fn test_chunk_text_preserves_all_content() {
        let text = "First sentence. Second sentence! Third one? And a fragment";
        let rejoined = chunk_text(text).join(" ");
        for word in ["First", "Second", "Third", "fragment"] {
            assert!(rejoined.contains(word), "lost {word:?}");
        }
    }


    #[test]
    fn test_compiled_flag_tracks_feature() {
        assert_eq!(INFLECT_MICRO_COMPILED, cfg!(feature = "inflect-micro"));
    }
}
