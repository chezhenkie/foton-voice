//! The encoded reference voice: mel features + its transcript's token ids.

use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use tracing::{info, warn};

use super::g2p;
use super::mel;
use super::model::ENCODE_TARGET_RMS;

/// Seconds of the reference clip fed to the prompt encoder. Configurable
pub fn max_prompt_duration(configured: f32) -> f32 {
    if configured > 0.0 {
        configured
    } else {
        15.0
    }
}

#[derive(Debug, Clone)]
pub struct Prompt {
    pub tokens: Vec<i64>,
    /// `[T][N_MELS]` log-mel rows, already scaled by FEAT_SCALE.
    pub features: Vec<Vec<f32>>,
    pub features_len: usize,
    pub rms: f32,
    /// The `ref_duration` this prompt was encoded with (cache v4+).
    pub source_ref_duration: Option<f32>,
}

/// Decode a clip to mono f32 at its native rate via rodio's decoder.
fn decode_clip(path: &Path) -> Result<(Vec<f32>, u32)> {
    use rodio::Source;

    let bytes = std::fs::read(path)
        .with_context(|| format!("read reference clip {}", path.display()))?;
    let cursor = std::io::Cursor::new(bytes);
    let mut decoder = rodio::Decoder::new(cursor)
        .map_err(|e| anyhow::anyhow!("decode {}: {e}", path.display()))?;
    let sample_rate = decoder.sample_rate();
    let channels = rodio::Source::channels(&mut decoder).max(1) as usize;
    let raw: Vec<i16> = decoder.collect();
    let samples: Vec<f32> = raw
        .chunks(channels)
        .map(|c| c.iter().map(|&s| s as f32 / 32768.0).sum::<f32>() / c.len() as f32)
        .collect();
    Ok((samples, sample_rate))
}

/// RMS normalize, only ever scaling UP (the reference `rms_norm`).
fn rms_norm(audio: &mut [f32], target_rms: f32) -> f32 {
    let rms = (audio.iter().map(|s| s * s).sum::<f32>() / audio.len().max(1) as f32).sqrt();
    if rms < target_rms && rms > 0.0 {
        let gain = target_rms / rms;
        for s in audio.iter_mut() {
            *s *= gain;
        }
    }
    rms
}


const SILENCE_THRESH_DB: f32 = -50.0;
const SEEK_SECS: f32 = 0.010;
const MIN_SILENCE_SECS: f32 = 1.0;
const SPLIT_KEEP_SECS: f32 = 1.0;
const EDGE_KEEP_SECS: f32 = 0.1;
const TRAIL_SECS: f32 = 0.2;

fn silence_amp() -> f32 {
    10f32.powf(SILENCE_THRESH_DB / 20.0)
}

fn rms_of(samples: &[f32]) -> f32 {
    (samples.iter().map(|s| s * s).sum::<f32>() / samples.len().max(1) as f32).sqrt()
}

/// Split the clip on silences longer than 1 s, keeping 1 s of silence around
fn split_long_silences(samples: &[f32], rate: u32) -> Vec<f32> {
    let step = (rate as f32 * SEEK_SECS).max(1.0) as usize;
    let window = (rate as f32 * MIN_SILENCE_SECS) as usize;
    let keep = (rate as f32 * SPLIT_KEEP_SECS) as usize;
    let n = samples.len();
    if n < window {
        return samples.to_vec();
    }
    let amp = silence_amp();
    let mut silences: Vec<(usize, usize)> = Vec::new();
    for k in 0..=(n - window) / step {
        let s = k * step;
        let e = (s + window).min(n);
        if rms_of(&samples[s..e]) <= amp {
            match silences.last_mut() {
                Some(last) if s <= last.1 => last.1 = e,
                _ => silences.push((s, e)),
            }
        }
    }
    let mut nonsilent: Vec<(usize, usize)> = Vec::new();
    let mut pos = 0;
    for (s, e) in &silences {
        if *s > pos {
            nonsilent.push((pos, *s));
        }
        pos = pos.max(*e);
    }
    if pos < n {
        nonsilent.push((pos, n));
    }
    if nonsilent.is_empty() {
        return Vec::new();
    }
    let mut ranges: Vec<(usize, usize)> = nonsilent
        .iter()
        .map(|&(s, e)| (s.saturating_sub(keep), (e + keep).min(n)))
        .collect();
    for i in 1..ranges.len() {
        let prev_end = ranges[i - 1].1;
        let start = ranges[i].0;
        if prev_end >= start {
            let mid = (start + prev_end) / 2;
            ranges[i - 1].1 = mid;
            ranges[i].0 = mid;
        }
    }
    let mut out = Vec::with_capacity(n);
    for (s, e) in ranges {
        if e > s {
            out.extend_from_slice(&samples[s..e]);
        }
    }
    out
}

/// Trim edge silences to 100 ms (upstream `remove_silence_edges`): scan 10 ms
fn trim_edges(samples: &mut Vec<f32>, rate: u32) {
    let step = (rate as f32 * SEEK_SECS).max(1.0) as usize;
    let keep = (rate as f32 * EDGE_KEEP_SECS) as usize;
    let amp = silence_amp();
    let n = samples.len();
    let mut lead = 0;
    while lead + step <= n && rms_of(&samples[lead..lead + step]) < amp {
        lead += step;
    }
    let mut tail = 0;
    while tail + step <= n && rms_of(&samples[n - tail - step..n - tail]) < amp {
        tail += step;
    }
    let start = lead.saturating_sub(keep);
    let end = n - tail.saturating_sub(keep);
    if start < end {
        *samples = samples[start..end].to_vec();
    } else {
        samples.clear();
    }
}

/// Port of upstream `remove_silence` (zipvoice utils/infer.py): split
fn remove_silence(samples: &mut Vec<f32>, rate: u32) {
    let mut split = split_long_silences(samples, rate);
    trim_edges(&mut split, rate);
    let trail = (rate as f32 * TRAIL_SECS) as usize;
    split.extend(std::iter::repeat(0.0).take(trail));
    *samples = split;
}

/// Encode a reference clip + transcript into a [`Prompt`].
pub fn encode(
    clip: &Path,
    transcript: &str,
    vocab: &g2p::TokenVocab,
    ref_duration: f32,
) -> Result<Prompt> {
    let (samples, native_rate) = decode_clip(clip)?;
    if samples.is_empty() {
        bail!("reference clip {} decoded to no samples", clip.display());
    }
    let max_samples = (max_prompt_duration(ref_duration) * native_rate as f32) as usize;
    let trimmed = &samples[..max_samples.min(samples.len())];
    let mut audio24 = super::vocoder::resample(trimmed, native_rate, mel::SAMPLE_RATE);

    remove_silence(&mut audio24, mel::SAMPLE_RATE);
    if audio24.is_empty() {
        bail!(
            "reference clip {} has no speech above the silence threshold (-50 dBFS)",
            clip.display()
        );
    }
    let prompt_rms = rms_norm(&mut audio24, ENCODE_TARGET_RMS);

    let features = mel::extract_mel_features(&audio24);
    let features_len = features.len();

    let symbols = g2p::text_to_symbols(transcript)
        .with_context(|| format!("phonemize the transcript of {}", clip.display()))?;
    let (tokens, skipped) = vocab.encode(&symbols);
    if !skipped.is_empty() {
        info!(
            "LuxTTS: {} transcript symbol(s) skipped as OOV: {}",
            skipped.len(),
            skipped.join(" ")
        );
    }
    let speech_seconds = features_len as f32 * mel::HOP_LENGTH as f32 / mel::SAMPLE_RATE as f32;
    let symbols_per_second = symbols.len() as f32 / speech_seconds.max(0.1);
    if symbols_per_second > 25.0 {
        warn!(
            "LuxTTS: transcript of {} implies {:.1} phonemes/s over {:.1}s of prompt audio \
             - the clip is probably trimmed shorter than its transcript (ref_duration). \
             Trim the transcript to match, or raise tts.lux_tts.ref_duration.",
            clip.display(),
            symbols_per_second,
            speech_seconds
        );
    }

    Ok(Prompt {
        tokens,
        features,
        features_len,
        rms: prompt_rms,
        source_ref_duration: Some(ref_duration),
    })
}

/// Prompt cache path: `<clip-stem>.luxtprompt` next to the reference wav.
pub fn cache_path_for(clip: &Path) -> PathBuf {
    let stem = clip
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "voice".into());
    clip.with_file_name(format!("{stem}.luxtprompt"))
}

/// The cache is stale when the clip or its transcript file is newer than it.
fn cache_is_stale(cache: &Path, clip: &Path) -> bool {
    let cached_modified = std::fs::metadata(cache)
        .ok()
        .and_then(|m| m.modified().ok());
    let Some(cached_modified) = cached_modified else {
        return true;
    };
    [clip.to_path_buf(), clip.with_extension("txt")]
        .into_iter()
        .any(|src| {
            std::fs::metadata(&src)
                .ok()
                .and_then(|m| m.modified().ok())
                .map(|t| t > cached_modified)
                .unwrap_or(false)
        })
}

/// Load a cached prompt, or encode and save it. The cache is keyed by the
pub fn load_or_encode(
    clip: &Path,
    transcript: &str,
    vocab: &g2p::TokenVocab,
    ref_duration: f32,
) -> Result<Prompt> {
    let cache = cache_path_for(clip);
    if !cache_is_stale(&cache, clip) {
        if let Ok(p) = load(&cache) {
            if p.source_ref_duration.map(|d| (d - ref_duration).abs() < 0.01).unwrap_or(false) {
                return Ok(p);
            }
        }
    }
    let prompt = encode(clip, transcript, vocab, ref_duration)?;
    if let Err(e) = save(&prompt, &cache) {
        info!("LuxTTS: prompt cache not written: {e}");
    } else {
        info!("LuxTTS: prompt cached at {}", cache.display());
    }
    Ok(prompt)
}

/// Prompt cache format version. v1 caches were written by a decoder that
pub const CACHE_VERSION: u32 = 5;

/// Serialize to the compact binary cache format.
pub fn save(prompt: &Prompt, path: &Path) -> Result<()> {
    let mut out: Vec<u8> = Vec::with_capacity(prompt.features.len() * FEAT_SIZE_BYTES + 64);
    out.extend_from_slice(b"LUXP");
    out.extend_from_slice(&CACHE_VERSION.to_le_bytes());
    out.extend_from_slice(&(prompt.tokens.len() as u32).to_le_bytes());
    for t in &prompt.tokens {
        out.extend_from_slice(&t.to_le_bytes());
    }
    out.extend_from_slice(&prompt.rms.to_le_bytes());
    out.extend_from_slice(&prompt.source_ref_duration.unwrap_or(0.0).to_le_bytes());
    out.extend_from_slice(&(prompt.features_len as u32).to_le_bytes());
    for row in &prompt.features {
        for v in row {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }
    std::fs::write(path, out).with_context(|| format!("write {}", path.display()))
}

const FEAT_SIZE_BYTES: usize = 4;

/// Deserialize the binary cache format.
pub fn load(path: &Path) -> Result<Prompt> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let mut pos = 0usize;

    fn take<'a>(bytes: &'a [u8], pos: &mut usize, n: usize) -> Result<&'a [u8]> {
        if *pos + n > bytes.len() {
            bail!("truncated prompt cache");
        }
        let s = &bytes[*pos..*pos + n];
        *pos += n;
        Ok(s)
    }

    if take(&bytes, &mut pos, 4)? != b"LUXP" {
        bail!("not a LuxTTS prompt cache");
    }
    let version = u32::from_le_bytes(take(&bytes, &mut pos, 4)?.try_into().unwrap());
    if version != CACHE_VERSION {
        bail!("prompt cache version {version} is stale (expected {CACHE_VERSION})");
    }
    let n_tokens = u32::from_le_bytes(take(&bytes, &mut pos, 4)?.try_into().unwrap()) as usize;
    let mut tokens = Vec::with_capacity(n_tokens);
    for _ in 0..n_tokens {
        tokens.push(i64::from_le_bytes(take(&bytes, &mut pos, 8)?.try_into().unwrap()));
    }
    let rms = f32::from_le_bytes(take(&bytes, &mut pos, 4)?.try_into().unwrap());
    let ref_duration = f32::from_le_bytes(take(&bytes, &mut pos, 4)?.try_into().unwrap());
    let t = u32::from_le_bytes(take(&bytes, &mut pos, 4)?.try_into().unwrap()) as usize;

    let mut features = Vec::with_capacity(t);
    for _ in 0..t {
        let mut row = Vec::with_capacity(mel::N_MELS);
        for _ in 0..mel::N_MELS {
            row.push(f32::from_le_bytes(take(&bytes, &mut pos, 4)?.try_into().unwrap()));
        }
        features.push(row);
    }
    Ok(Prompt {
        tokens,
        features_len: t,
        features,
        rms,
        source_ref_duration: if ref_duration > 0.0 { Some(ref_duration) } else { None },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_prompt() -> Prompt {
        Prompt {
            tokens: vec![1, 2, 3],
            features: vec![vec![0.1; mel::N_MELS]; 7],
            features_len: 7,
            rms: 0.02,
            source_ref_duration: Some(5.0),
        }
    }

    #[test]
    fn test_save_load_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("v.luxtprompt");
        let prompt = test_prompt();
        save(&prompt, &path).unwrap();
        let loaded = load(&path).unwrap();
        assert_eq!(loaded.tokens, prompt.tokens);
        assert_eq!(loaded.features_len, 7);
        assert_eq!(loaded.features.len(), 7);
        assert_eq!(loaded.features[0].len(), mel::N_MELS);
        assert!((loaded.rms - 0.02).abs() < 1e-6);
        assert_eq!(loaded.source_ref_duration, Some(5.0));
    }

    #[test]
    fn test_load_rejects_bad_magic() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("v.luxtprompt");
        std::fs::write(&path, b"NOPE").unwrap();
        assert!(load(&path).is_err());
    }


    const TEST_RATE: u32 = 1000; // 10 ms step = 10 samples, keeps sizes tiny

    fn silence(secs: f32) -> Vec<f32> {
        vec![0.0; (TEST_RATE as f32 * secs) as usize]
    }

    fn speech(secs: f32) -> Vec<f32> {
        vec![0.5; (TEST_RATE as f32 * secs) as usize]
    }

    #[test]
    fn test_split_long_silences_keeps_1s_around_speech() {
        let mut input = silence(2.5);
        input.extend(speech(1.0));
        input.extend(silence(3.0));
        let out = split_long_silences(&input, TEST_RATE);
        assert_eq!(out.len(), (TEST_RATE * 3) as usize, "1 s silence + speech + 1 s silence");
        assert!(out[..1000].iter().all(|s| *s == 0.0));
        assert!(out[1000..2000].iter().all(|s| *s == 0.5));
        assert!(out[2000..].iter().all(|s| *s == 0.0));
    }

    #[test]
    fn test_split_long_silences_keeps_short_silences_intact() {
        let mut input = silence(0.5);
        input.extend(speech(1.0));
        input.extend(silence(0.5));
        let out = split_long_silences(&input, TEST_RATE);
        assert_eq!(out.len(), input.len());
    }

    #[test]
    fn test_split_all_silence_yields_empty() {
        let out = split_long_silences(&silence(3.0), TEST_RATE);
        assert!(out.is_empty());
    }

    #[test]
    fn test_trim_edges_keeps_100ms() {
        let mut input = silence(0.5);
        input.extend(speech(1.0));
        let mut out = input.clone();
        trim_edges(&mut out, TEST_RATE);
        assert_eq!(out.len(), (TEST_RATE as f32 * 1.1) as usize);
        assert!(out.iter().take(100).all(|s| *s == 0.0));
        assert_eq!(out[100], 0.5);
    }

    #[test]
    fn test_remove_silence_appends_trail() {
        let mut input = speech(1.0);
        input.extend(silence(0.5));
        let mut out = input.clone();
        remove_silence(&mut out, TEST_RATE);
        assert_eq!(out.len(), (TEST_RATE as f32 * 1.3) as usize);
        let trail = &out[out.len() - 200..];
        assert!(trail.iter().all(|s| *s == 0.0), "trail must be silence");
    }
}
