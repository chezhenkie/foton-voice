//! LuxTTS (<https://github.com/ysharma3501/LuxTTS>) - lightweight ZipVoice-family

#[cfg(feature = "luxtts")]
pub mod g2p;
#[cfg(feature = "luxtts")]
pub mod mel;
#[cfg(feature = "luxtts")]
pub mod model;
#[cfg(feature = "luxtts")]
pub mod prompt;
#[cfg(feature = "luxtts")]
pub mod sampler;
#[cfg(feature = "luxtts")]
pub mod vocoder;
#[cfg(all(feature = "luxtts", test))]
mod probe;

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
#[cfg(feature = "luxtts")]
use tracing::warn;

use crate::piper::expand_tilde;


/// Default model directory: `<portable root>/models/lux-tts/`, keeping the
pub fn lux_tts_model_dir() -> PathBuf {
    fotonvoice_config::portable::app_root()
        .join("models")
        .join("lux-tts")
}

/// Resolve a configured model dir (`~` expansion; empty = platform default).
pub fn resolve_model_dir(configured: &str) -> PathBuf {
    if configured.trim().is_empty() {
        lux_tts_model_dir()
    } else {
        expand_tilde(configured)
    }
}

/// The files a working LuxTTS model folder must contain.
pub const LUX_TTS_REQUIRED_FILES: [&str; 4] =
    ["text_encoder.onnx", "fm_decoder.onnx", "vocos.onnx", "tokens.txt"];

/// Whether all required graphs + tokens.txt are on disk in `dir`.
pub fn is_lux_tts_downloaded(dir: &str) -> bool {
    LUX_TTS_REQUIRED_FILES
        .iter()
        .all(|f| resolve_model_dir(dir).join(f).exists())
}

#[cfg(feature = "luxtts")]
pub const LUX_TTS_COMPILED: bool = true;
#[cfg(not(feature = "luxtts"))]
pub const LUX_TTS_COMPILED: bool = false;


/// Resolve a reference voice id to a local `.wav` path in the voice dir.
#[cfg(feature = "luxtts")]
fn resolve_reference_clip(voice_id: &str, voice_dir: &str) -> Result<PathBuf> {
    let dir = resolve_voice_dir(voice_dir);
    let clip = Path::new(&dir).join(format!("{voice_id}.wav"));
    if clip.exists() {
        Ok(clip)
    } else {
        anyhow::bail!(
            "LuxTTS reference voice '{voice_id}' not found. Place {voice_id}.wav \
             (plus {voice_id}.txt with its transcript) in {}.",
            dir.display()
        )
    }
}

#[cfg(feature = "luxtts")]
fn resolve_voice_dir(voice_dir: &str) -> PathBuf {
    if voice_dir.trim().is_empty() {
        default_voice_dir()
    } else {
        expand_tilde(voice_dir)
    }
}

/// Shared voice directory default, matching the other cloning engines.
#[cfg(feature = "luxtts")]
fn default_voice_dir() -> PathBuf {
    fotonvoice_config::portable::app_root().join("cloned-tts-voices")
}


#[cfg(feature = "luxtts")]
pub(crate) fn ensure_lux_tts_loaded(
    config: &fotonvoice_config::TtsConfig,
    model: &mut Option<model::LuxTTSModel>,
) -> Result<()> {
    let cfg = &config.lux_tts;
    if !is_lux_tts_downloaded(&cfg.model_dir) {
        anyhow::bail!(
            "LuxTTS model files not found in {}. Place text_encoder.onnx, \
             fm_decoder.onnx, vocos.onnx and tokens.txt there.",
            resolve_model_dir(&cfg.model_dir).display()
        );
    }
    if model.is_none() {
        let started = std::time::Instant::now();
        let dir = resolve_model_dir(&cfg.model_dir);
        *model = Some(model::LuxTTSModel::load(&dir, cfg.quantized)?);
        tracing::info!("LuxTTS sessions loaded in {:?}", started.elapsed());
    }
    Ok(())
}

#[cfg(not(feature = "luxtts"))]
pub(crate) fn ensure_lux_tts_loaded(
    _config: &fotonvoice_config::TtsConfig,
    _model: &mut Option<()>,
) -> Result<()> {
    Ok(())
}


/// Maximum characters per synthesis chunk. LuxTTS conditions on a prompt, so
#[cfg(feature = "luxtts")]
const EDGE_FADE_MS: f32 = 12.0;
/// Prompt + generation budget per chunk, in seconds - the reference chunks
#[cfg(feature = "luxtts")]
const MAX_CHUNK_SECONDS: f32 = 25.0;

/// Called from `TtsEngineWorker::run` when `config.engine == TtsEngine::LuxTts`.
#[cfg(feature = "luxtts")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn speak_lux_tts(
    config: &fotonvoice_config::TtsConfig,
    u: &crate::engine::Utterance,
    model: &mut Option<model::LuxTTSModel>,
    on_playback_start: &Option<crate::engine::PlaybackCallback>,
    sink: &rodio::Sink,
    generation_counter: &std::sync::Arc<std::sync::atomic::AtomicU32>,
    generation: u32,
) -> Result<()> {
    let cfg = &config.lux_tts;
    let is_prewarm = u.source_label.as_deref() == Some("prewarm");

    ensure_lux_tts_loaded(config, model)?;
    let model = model
        .as_mut()
        .ok_or_else(|| anyhow::anyhow!("luxtts model slot missing after ensure"))?;

    let clip = resolve_reference_clip(&cfg.cloned_voice, &cfg.voice_dir)?;
    let transcript = std::fs::read_to_string(clip.with_extension("txt"))
        .with_context(|| format!("read transcript {}", clip.with_extension("txt").display()))?;
    let ref_duration = cfg.ref_duration.clamp(12.0, 25.0);
    let prompt = prompt::load_or_encode(&clip, transcript.trim(), &model.vocab, ref_duration)?;

    if is_prewarm {
        let _ = model.synthesize(
            &[model.vocab.reserved.start],
            &prompt,
            cfg.num_steps,
            cfg.t_shift,
            cfg.guidance_scale,
            1.0,
            cfg.seed,
            cfg.return_smooth,
        )?;
        return Ok(());
    }

    let text = u.text.trim();
    if text.is_empty() {
        return Ok(());
    }
    if !g2p::espeak_available() {
        anyhow::bail!(
            "espeak-ng is not installed on this system, and LuxTTS needs it for \
             grapheme-to-phoneme conversion."
        );
    }

    let mut symbols = g2p::text_to_symbols(text)?;
    let (tokens, skipped) = model.vocab.encode(&symbols);
    let _ = &mut symbols;
    if !skipped.is_empty() {
        tracing::warn!(
            "LuxTTS: {} symbol(s) absent from the vocabulary and skipped: {}",
            skipped.len(),
            skipped.join(" ")
        );
    }
    if tokens.is_empty() {
        anyhow::bail!(
            "LuxTTS: no phonemes produced for {:?} - check that espeak-ng works \
             and the text is not empty.",
            text
        );
    }

    let frames_per_token = (prompt.features_len as f32 / prompt.tokens.len().max(1) as f32).max(1.0);
    let max_chunk_tokens = {
        let window_frames =
            (MAX_CHUNK_SECONDS * mel::SAMPLE_RATE as f32 / mel::HOP_LENGTH as f32) as usize;
        let budget = window_frames.saturating_sub(prompt.features_len);
        (budget as f32 / frames_per_token).floor() as usize
    };

    let mut chunks: Vec<Vec<i64>> = Vec::new();
    if max_chunk_tokens == 0 {
        chunks.push(tokens);
    } else {
        let mut current: Vec<i64> = Vec::new();
        for clause in g2p::split_clauses(&g2p::map_punctuations(text)) {
            let clause_symbols = g2p::clause_to_symbols(&clause);
            let (clause_tokens, clause_skipped) = model.vocab.encode(&clause_symbols);
            if !clause_skipped.is_empty() {
                tracing::warn!(
                    "LuxTTS: {} clause symbol(s) absent from the vocabulary and skipped: {}",
                    clause_skipped.len(),
                    clause_skipped.join(" ")
                );
            }
            if clause_tokens.is_empty() {
                warn!("LuxTTS: clause phonemized to nothing, skipping: {:?}", clause.text);
                continue;
            }
            if current.len() + clause_tokens.len() > max_chunk_tokens && !current.is_empty() {
                chunks.push(std::mem::take(&mut current));
            }
            current.extend(clause_tokens);
        }
        if !current.is_empty() {
            chunks.push(current);
        }
    }
    if chunks.is_empty() {
        anyhow::bail!(
            "LuxTTS: no phonemes produced for {:?} - check that espeak-ng works \
             and the text is not empty.",
            text
        );
    }

    let num_steps = cfg.num_steps.max(1);
    let speed = if config.speed <= 0.0 { 1.0 } else { config.speed };

    let mut audio: Vec<f32> = Vec::new();
    for chunk in &chunks {
        let part = model.synthesize(
            chunk,
            &prompt,
            num_steps,
            cfg.t_shift,
            cfg.guidance_scale,
            speed,
            cfg.seed,
            cfg.return_smooth,
        )?;
        audio.extend(part);
    }
    if audio.is_empty() {
        anyhow::bail!("LuxTTS produced no audio");
    }

    if generation_counter.load(std::sync::atomic::Ordering::SeqCst) != generation {
        return Ok(());
    }

    crate::inflect::edge_fade(&mut audio, model::OUTPUT_SAMPLE_RATE, EDGE_FADE_MS);
    if let Some(ref cb) = on_playback_start {
        cb();
    }
    sink.append(rodio::buffer::SamplesBuffer::new(
        1,
        model::OUTPUT_SAMPLE_RATE,
        audio,
    ));
    sink.sleep_until_end();
    Ok(())
}

/// Stand-in for builds without the `luxtts` feature: a pre-load is a no-op
#[cfg(not(feature = "luxtts"))]
pub(crate) fn speak_lux_tts(
    _config: &fotonvoice_config::TtsConfig,
    _u: &crate::engine::Utterance,
    _model: &mut Option<()>,
    _on_playback_start: &Option<crate::engine::PlaybackCallback>,
    _sink: &rodio::Sink,
    _generation_counter: &std::sync::Arc<std::sync::atomic::AtomicU32>,
    _generation: u32,
) -> Result<()> {
    anyhow::bail!(
        "This build of FotonVoice Engine was compiled without the `luxtts` \
         feature, so the LuxTTS engine cannot synthesize."
    )
}
