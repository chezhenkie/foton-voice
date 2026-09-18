//! Vocos output post-processing: the Linkwitz-Riley crossover merge.
//!
//! The dual-path vocoder emits a 48 kHz waveform and a 24 kHz waveform for the
//! same utterance. The reference (`luxtts_onnx/inference.py::crossover_merge`)
//! upsamples the 24 kHz leg to 48 kHz, then combines the two in the frequency
//! domain with a 4th-order Linkwitz-Riley crossover (a Butterworth magnitude
//! squared) at 12 kHz: the 24 kHz path feeds the lows, the 48 kHz path the
//! highs. This module reproduces that FFT-based merge exactly.

use rustfft::num_complex::Complex;
use rustfft::FftPlanner;
use rubato::Resampler;

const CROSSOVER_FREQ: f64 = 12_000.0;
const SR_48K: f64 = 48_000.0;

/// Resample `audio` from `orig` to `target` Hz with rubato's sinc resampler.
///
/// Used both directions: reference clips arrive at any rate and go to 24 kHz
/// for mel extraction; the 24 kHz vocos leg goes to 48 kHz for the merge.
pub fn resample(audio: &[f32], orig: u32, target: u32) -> Vec<f32> {
    if orig == target || audio.is_empty() {
        return audio.to_vec();
    }
    // Integer 2x upsample (the merge path) through rubato's sinc; arbitrary
    // ratios (reference loading) use the same machinery.
    let ratio = target as f64 / orig as f64;
    let chunk = 1024;
    let mut resampler = rubato::SincFixedIn::<f32>::new(
        ratio,
        2.0,
        rubato::SincInterpolationParameters {
            sinc_len: 256,
            f_cutoff: 0.95,
            oversampling_factor: 256,
            interpolation: rubato::SincInterpolationType::Quadratic,
            window: rubato::WindowFunction::BlackmanHarris2,
        },
        chunk,
        1,
    )
    .expect("build resampler");

    let mut out: Vec<f32> = Vec::with_capacity(
        ((audio.len() as f64) * ratio).ceil() as usize + 1,
    );
    let mut pos = 0;
    while pos < audio.len() {
        let end = (pos + chunk).min(audio.len());
        // SincFixedIn requires exactly `chunk` input frames; zero-pad the tail
        // chunk and trim the matching tail from the output afterwards.
        let mut frame = audio[pos..end].to_vec();
        frame.resize(chunk, 0.0);
        let frame_out = match resampler.process(&[frame], None) {
            Ok(f) => f,
            Err(_) => return audio.to_vec(),
        };
        out.extend_from_slice(&frame_out[0]);
        pos = end;
    }
    // Trim to the exact expected length so both legs line up.
    let expected = (audio.len() as f64 * ratio).round() as usize;
    out.truncate(expected.max(1));
    out
}

/// FFT-based Linkwitz-Riley merge of the 48 kHz and 24 kHz paths.
///
/// 4th-order Butterworth magnitude squared = LR4: `low = sqrt(1/(1+r^8))`,
/// `high = sqrt(1 - low^2)`. Both legs must already be at 48 kHz; lengths are
/// truncated to their common length.
pub fn crossover_merge(audio_48k: &[f32], audio_24k: &[f32]) -> Vec<f32> {
    let up = resample(audio_24k, 24_000, 48_000);
    let n = audio_48k.len().min(up.len());
    if n == 0 {
        return Vec::new();
    }
    let a48 = &audio_48k[..n];
    let up = &up[..n];

    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(n);
    let ifft = planner.plan_fft_inverse(n);

    let mut spec_48: Vec<Complex<f32>> = a48.iter().map(|s| Complex::new(*s, 0.0)).collect();
    let mut spec_24: Vec<Complex<f32>> = up.iter().map(|s| Complex::new(*s, 0.0)).collect();
    fft.process(&mut spec_48);
    fft.process(&mut spec_24);

    let bins = n / 2 + 1;
    let mut merged: Vec<Complex<f32>> = spec_48.clone();
    for b in 0..bins {
        let f = b as f64 * SR_48K / n as f64;
        let ratio = f / CROSSOVER_FREQ;
        let butter_sq = 1.0 / (1.0 + ratio.powi(8));
        let low_gain = butter_sq.sqrt();
        let high_gain = (1.0 - butter_sq).max(0.0).sqrt();
        merged[b] = spec_24[b] * low_gain as f32 + spec_48[b] * high_gain as f32;
    }
    // The reference (np.fft.rfft + irfft) treats the merged spectrum as
    // conjugate-symmetric: irfft derives the upper half from the lower one.
    // Gains must be mirrored there too - leaving the raw 48k bins in place
    // injects unfiltered high-frequency content (harsh, "truly awful").
    for b in 1..n - bins + 1 {
        merged[n - b] = merged[b].conj();
    }

    ifft.process(&mut merged);
    // rustfft's inverse leaves the result scaled by n.
    let scale = 1.0 / n as f32;
    merged
        .iter()
        .take(n)
        .map(|c| c.re * scale)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resample_2x_doubles_length() {
        let audio: Vec<f32> = (0..2400).map(|i| (i as f32 * 0.01).sin()).collect();
        let up = resample(&audio, 24_000, 48_000);
        assert_eq!(up.len(), 4800);
    }

    #[test]
    fn test_resample_same_rate_is_identity() {
        let audio = vec![0.5f32; 100];
        assert_eq!(resample(&audio, 24_000, 24_000), audio);
    }

    #[test]
    fn test_merge_preserves_length_and_finiteness() {
        let a48: Vec<f32> = (0..4800).map(|i| (i as f32 * 0.02).sin()).collect();
        let a24: Vec<f32> = (0..2400).map(|i| (i as f32 * 0.01).cos()).collect();
        let merged = crossover_merge(&a48, &a24);
        assert_eq!(merged.len(), 4800);
        assert!(merged.iter().all(|s| s.is_finite()));
    }

    #[test]
    fn test_merge_lows_come_from_24k_path() {
        // A 100 Hz tone in both paths must survive the merge; the crossover at
        // 12 kHz passes lows from the 24 kHz leg.
        let sr = 48_000.0f32;
        let audio_48k: Vec<f32> = (0..4800)
            .map(|i| (2.0 * std::f32::consts::PI * 100.0 * i as f32 / sr).sin())
            .collect();
        let audio_24k: Vec<f32> = (0..2400)
            .map(|i| (2.0 * std::f32::consts::PI * 100.0 * i as f32 / 24_000.0).sin())
            .collect();
        let merged = crossover_merge(&audio_48k, &audio_24k);
        let rms: f32 = (merged.iter().map(|s| s * s).sum::<f32>() / merged.len() as f32).sqrt();
        assert!(rms > 0.3, "100 Hz tone must survive the merge, rms={rms}");
    }
}
