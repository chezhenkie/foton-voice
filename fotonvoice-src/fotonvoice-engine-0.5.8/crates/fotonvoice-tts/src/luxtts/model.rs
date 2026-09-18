//! The three ONNX graphs behind LuxTTS, and the flow-matching synthesis path.
//!
//! The export splits the model in three:
//!
//! 1. `text_encoder.onnx` - text ids + prompt ids + prompt length + speed
//!    -> aligned text conditioning `[1, T, D]` (the predicted total frame count
//!    is that output's time axis).
//! 2. `fm_decoder.onnx` - `(t, x, text_condition, speech_condition, guidance)`
//!    -> velocity field; the sampler integrates it over a shifted schedule.
//! 3. `vocos.onnx` - mel features `[1, D, T]` -> two waveforms (48 kHz and
//!    24 kHz legs), merged downstream by the Linkwitz-Riley crossover.
//!
//! Tensor bindings are positional, mirroring `infer_zipvoice_onnx.py`, which
//! itself addresses the graphs through `get_inputs()[i]`. The sampler is the
//! reference `sample()` loop verbatim: anchor-based ODE updates on a
//! `t_shift`-transformed schedule with classifier-free guidance.

use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};
use ort::session::{builder::GraphOptimizationLevel, Session, SessionInputValue};
use ort::value::Tensor;
use tracing::{debug, info};

use crate::luxtts::g2p::TokenVocab;
use crate::luxtts::mel::FEAT_SCALE;
use crate::luxtts::prompt::Prompt;
use crate::luxtts::sampler::{self, StandardNormal};

pub const TE_FILE: &str = "text_encoder.onnx";
pub const TE_FILE_INT8: &str = "text_encoder_int8.onnx";
pub const FM_FILE: &str = "fm_decoder.onnx";
pub const FM_FILE_INT8: &str = "fm_decoder_int8.onnx";
pub const VOCOS_FILE: &str = "vocos.onnx";
pub const TOKENS_FILE: &str = "tokens.txt";

/// Everything that must be on disk before the engine can load.
pub const MODEL_FILES: [&str; 4] = [TE_FILE, FM_FILE, VOCOS_FILE, TOKENS_FILE];

pub const OUTPUT_SAMPLE_RATE: u32 = 48_000;

/// Defaults from the reference `generate()` signature.
pub const DEFAULT_TARGET_RMS: f32 = 0.1;
/// The prompt conditioning normalization. The exported graphs were trained on
/// prompts at this RMS (the official ONNX inference normalizes the reference
/// clip to 0.1 before mel extraction, `infer_zipvoice_onnx.py`); feeding a
/// quieter prompt pushes the model off-distribution and the output comes out
/// proportionally quiet. (The luxtts-onnx README's `encode_prompt` 0.01 default
/// does not match its own exporter's inference path.)
pub const ENCODE_TARGET_RMS: f32 = 0.1;
/// ~170 ms at hop=256, sr=24000 - the shortest generation the vocoder accepts.
pub const MIN_GEN_FRAMES: usize = 16;

fn threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get().min(4))
        .unwrap_or(1)
}

fn build_session(path: &Path) -> Result<Session> {
    if !path.exists() {
        bail!(
            "LuxTTS graph {} is missing. Place the model files there manually or \
             download them from Settings > Text-to-Speech.",
            path.display()
        );
    }
    Session::builder()
        .map_err(|e| anyhow!("ort session builder: {e}"))?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| anyhow!("set optimization level: {e}"))?
        .with_intra_threads(threads())
        .map_err(|e| anyhow!("set intra threads: {e}"))?
        .commit_from_file(path)
        .with_context(|| format!("load ONNX graph {}", path.display()))
}

/// Pick the requested graph variant, falling back to fp32 when the int8 export
/// is absent.
fn graph_file(dir: &Path, base: &str, int8_name: &str, quantized: bool) -> (PathBuf, bool) {
    let fp32 = dir.join(base);
    if !quantized {
        return (fp32, false);
    }
    let int8 = dir.join(int8_name);
    if int8.exists() {
        (int8, true)
    } else {
        (fp32, false)
    }
}

pub struct LuxTTSModel {
    text_encoder: Session,
    fm_decoder: Session,
    vocos: Session,
    pub vocab: TokenVocab,
    pub feat_dim: usize,
    dir: PathBuf,
}

impl LuxTTSModel {
    /// Load all three graphs plus the token vocabulary.
    pub fn load(dir: &Path, quantized: bool) -> Result<Self> {
        info!(
            "Loading LuxTTS from {} (quantized={quantized})",
            dir.display()
        );
        let (te_path, te_q) = graph_file(dir, TE_FILE, TE_FILE_INT8, quantized);
        let (fm_path, fm_q) = graph_file(dir, FM_FILE, FM_FILE_INT8, quantized);
        let vocos_path = dir.join(VOCOS_FILE);
        let tokens_path = dir.join(TOKENS_FILE);

        let text_encoder = build_session(&te_path)?;
        let fm_decoder = build_session(&fm_path)?;
        let vocos = build_session(&vocos_path)?;
        let vocab = TokenVocab::load(&tokens_path)?;

        // feat_dim rides on the fm_decoder's custom metadata (the export writes
        // it there); fall back to the 100-mel layout when absent.
        let feat_dim = fm_decoder
            .metadata()
            .ok()
            .and_then(|m| m.custom("feat_dim"))
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(crate::luxtts::mel::N_MELS);
        info!(
            "LuxTTS graphs loaded (text_encoder={}{}, fm_decoder={}{}, feat_dim={feat_dim})",
            te_path.file_name().map(|f| f.to_string_lossy()).unwrap_or_default(),
            if te_q { " [int8]" } else { "" },
            fm_path.file_name().map(|f| f.to_string_lossy()).unwrap_or_default(),
            if fm_q { " [int8]" } else { "" },
        );

        Ok(Self {
            text_encoder,
            fm_decoder,
            vocos,
            vocab,
            feat_dim,
            dir: dir.to_path_buf(),
        })
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// text ids + prompt ids -> aligned text conditioning `[1, T, D]`.
    fn run_text_encoder(
        &mut self,
        tokens: &[i64],
        prompt_tokens: &[i64],
        prompt_features_len: i64,
        speed: f32,
    ) -> Result<(Vec<usize>, Vec<f32>)> {
        let names: Vec<String> = self
            .text_encoder
            .inputs()
            .iter()
            .map(|i| i.name().to_string())
            .collect();
        let tokens_t = Tensor::from_array(([1_usize, tokens.len()], tokens.to_vec()))
            .context("build tokens tensor")?;
        let prompt_t = Tensor::from_array(([1_usize, prompt_tokens.len()], prompt_tokens.to_vec()))
            .context("build prompt tokens tensor")?;
        let len_t = Tensor::from_array(([1_usize], vec![prompt_features_len]))
            .context("build prompt length tensor")?;
        let speed_t = Tensor::from_array(([0_usize; 0], vec![speed]))
            .context("build speed tensor")?;

        let outputs = self
            .text_encoder
            .run(vec![
                (names[0].as_str(), SessionInputValue::from(tokens_t)),
                (names[1].as_str(), SessionInputValue::from(prompt_t)),
                (names[2].as_str(), SessionInputValue::from(len_t)),
                (names[3].as_str(), SessionInputValue::from(speed_t)),
            ])
            .map_err(|e| anyhow!("text_encoder run failed: {e}"))?;

        let (shape, data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow!("text_encoder output is not float32: {e}"))?;
        Ok((
            shape.iter().map(|d| *d as usize).collect(),
            data.to_vec(),
        ))
    }

    /// One velocity evaluation of the flow-matching decoder.
    fn run_fm_decoder(
        &mut self,
        t: f32,
        x: &[f32],
        x_shape: &[usize],
        text_condition: &[f32],
        speech_condition: &[f32],
        guidance: f32,
    ) -> Result<Vec<f32>> {
        let names: Vec<String> = self
            .fm_decoder
            .inputs()
            .iter()
            .map(|i| i.name().to_string())
            .collect();
        let t_t = Tensor::from_array(([0_usize; 0], vec![t])).context("build t tensor")?;
        let x_t = Tensor::from_array((x_shape.to_vec(), x.to_vec())).context("build x tensor")?;
        let cond_shape = vec![1usize, x_shape[1], self.feat_dim];
        let text_t = Tensor::from_array((cond_shape.clone(), text_condition.to_vec()))
            .context("build text condition tensor")?;
        let speech_t = Tensor::from_array(
            (cond_shape.clone(), speech_condition.to_vec()),
        )
        .context("build speech condition tensor")?;
        let g_t = Tensor::from_array(([0_usize; 0], vec![guidance]))
            .context("build guidance tensor")?;

        let outputs = self
            .fm_decoder
            .run(vec![
                (names[0].as_str(), SessionInputValue::from(t_t)),
                (names[1].as_str(), SessionInputValue::from(x_t)),
                (names[2].as_str(), SessionInputValue::from(text_t)),
                (names[3].as_str(), SessionInputValue::from(speech_t)),
                (names[4].as_str(), SessionInputValue::from(g_t)),
            ])
            .map_err(|e| anyhow!("fm_decoder run failed: {e}"))?;
        let (_, data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow!("fm_decoder output is not float32: {e}"))?;
        Ok(data.to_vec())
    }

    /// Full synthesis: tokens + prompt -> 48 kHz mono waveform in `[-1, 1]`.
    ///
    /// Mirrors the reference `generate()`: text conditioning, frame-count
    /// guard, ODE loop, prompt slice-off, vocos, crossover merge, RMS match.
    pub fn synthesize(
        &mut self,
        tokens: &[i64],
        prompt: &Prompt,
        num_steps: u32,
        t_shift: f32,
        guidance: f32,
        speed: f32,
        seed: u64,
        smooth: bool,
    ) -> Result<Vec<f32>> {
        // Graph-level speed feed. Upstream (k2-fsa ZipVoice
        // `infer_zipvoice_onnx.py`) feeds the speed value RAW to the exported
        // text encoder; at speed 1.0 the graph then predicts the paper's
        // ratio-based duration exactly (`T_syn = T_prompt * |y_syn|/|y_prompt|`).
        // LuxTTS's torch wrapper multiplies by 1.3 ("default is too slow") -
        // that hack does not belong to the exported-graph path: with 1.3 the
        // encoder under-predicts the total whenever the generation text is
        // shorter than ~1.3x the prompt, collapsing generation to a few frames
        // (audio that "cuts off after the first syllable" for medium text on
        // long reference prompts). Probe-verified 2026-09-17: speed_in 1.3 with
        // a 65-token text on a 1100-frame prompt yielded 116 gen frames (1.2s);
        // speed 1.0 is the upstream-correct feed.
        let speed = if speed <= 0.0 { 1.0 } else { speed };
        let (cond_shape, mut text_condition) = self.run_text_encoder(
            tokens,
            &prompt.tokens,
            prompt.features_len as i64,
            speed,
        )?;
        let mut num_frames = *cond_shape.get(1).unwrap_or(&0);

        // Guard from the reference: the encoder's predicted duration can be
        // shorter than the prompt itself for very short text; edge-pad so the
        // vocoder never sees an empty generation window.
        let prompt_t = prompt.features_len;
        let required = prompt_t + MIN_GEN_FRAMES;
        if num_frames < required {
            debug!(
                "LuxTTS: text_encoder predicted {num_frames} frames, prompt needs \
                 {required}; padding by {} frames",
                required - num_frames
            );
            let dim = self.feat_dim;
            let mut padded = text_condition.clone();
            padded.resize(num_frames * dim, 0.0);
            let last = if num_frames > 0 {
                text_condition[(num_frames - 1) * dim..num_frames * dim].to_vec()
            } else {
                vec![0.0f32; dim]
            };
            for _ in 0..(required - num_frames) {
                padded.extend_from_slice(&last);
            }
            text_condition = padded;
            num_frames = required;
        }

        let timesteps = sampler::time_steps(num_steps as usize, t_shift);
        let noise = StandardNormal::new(seed).fill(num_frames * self.feat_dim);
        let mut x = noise;

        // Prompt features padded (or truncated) to the full frame count.
        let dim = self.feat_dim;
        let mut speech_condition = vec![0.0f32; num_frames * dim];
        let copy_frames = prompt_t.min(num_frames);
        for (f, row) in prompt.features.iter().take(copy_frames).enumerate() {
            speech_condition[f * dim..(f + 1) * dim].copy_from_slice(row);
        }

        let x_shape = vec![1usize, num_frames, dim];
        for step in 0..num_steps as usize {
            let t_cur = timesteps[step];
            let v = self.run_fm_decoder(
                t_cur,
                &x,
                &x_shape,
                &text_condition,
                &speech_condition,
                guidance,
            )?;
            // Anchor-based ODE update, exactly the reference arithmetic.
            let t = t_cur as f32;
            if step < num_steps as usize - 1 {
                let t_next = timesteps[step + 1] as f32;
                for i in 0..x.len() {
                    let x1 = x[i] + (1.0 - t) * v[i];
                    let x0 = x[i] - t * v[i];
                    x[i] = (1.0 - t_next) * x0 + t_next * x1;
                }
            } else {
                for i in 0..x.len() {
                    x[i] = x[i] + (1.0 - t) * v[i];
                }
            }
        }

        // Remove the prompt portion.
        let gen_len = num_frames.saturating_sub(prompt_t.min(num_frames - 1)) * dim;
        let generated: Vec<f32> = x[x.len() - gen_len..].to_vec();

        // Vocos: [B, T, D] -> [B, D, T] / scale -> two waveforms.
        let frames = gen_len / dim;
        let mut vocos_input = vec![0.0f32; gen_len];
        for t in 0..frames {
            for d in 0..dim {
                vocos_input[d * frames + t] = generated[t * dim + d] / FEAT_SCALE;
            }
        }
        let (audio_48k, audio_24k) = self.run_vocos(&vocos_input, dim, frames)?;

        let merged: Vec<f32> = if smooth {
            // Upstream "return_smooth" (`vocos.return_48k = False`): the 24 kHz
            // leg only, upsampled to 48 kHz - smoother, less metallic, slightly
            // less crisp than the LR4 crossover of both legs.
            crate::luxtts::vocoder::resample(&audio_24k, 24_000, OUTPUT_SAMPLE_RATE)
        } else {
            crate::luxtts::vocoder::crossover_merge(&audio_48k, &audio_24k)
        };
        let mut merged: Vec<f32> = merged.iter().map(|s| s.clamp(-1.0, 1.0)).collect();

        // Volume matching: never louder than the reference voice.
        if prompt.rms < DEFAULT_TARGET_RMS {
            let gain = prompt.rms / DEFAULT_TARGET_RMS;
            for s in merged.iter_mut() {
                *s *= gain;
            }
        }
        Ok(merged)
    }

    /// vocos: mel features `[1, D, T]` -> (48 kHz waveform, 24 kHz waveform).
    fn run_vocos(
        &mut self,
        features: &[f32],
        dim: usize,
        frames: usize,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let input = Tensor::from_array((vec![1usize, dim, frames], features.to_vec()))
            .map_err(|e| anyhow!("build vocos input tensor: {e}"))?;
        let input_name = self.vocos.inputs()[0].name().to_string();
        let outputs = self
            .vocos
            .run(vec![(input_name.as_str(), SessionInputValue::from(input))])
            .map_err(|e| anyhow!("vocos run failed: {e}"))?;
        let (_, a48) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow!("vocos 48k output is not float32: {e}"))?;
        let (_, a24) = outputs[1]
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow!("vocos 24k output is not float32: {e}"))?;
        Ok((a48.to_vec(), a24.to_vec()))
    }
}
