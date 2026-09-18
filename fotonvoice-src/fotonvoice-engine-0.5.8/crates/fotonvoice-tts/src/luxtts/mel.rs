//! Mel-spectrogram extraction matching the reference pipeline choice for choice.
//!
//! `luxtts_onnx/inference.py::extract_mel_features` calls librosa with
//! `power=1` (magnitude), `center=True`, `norm=None`, `htk=True`, 100 mels,
//! then `log(clamp(1e-7))` and a 0.1 feature scale. Every one of those choices
//! matters - `norm='slaney'` alone shifts magnitudes by 33x and produces
//! silence - so this module reproduces each one:
//!
//! * periodic Hann window (librosa's `fftbins=True` default),
//! * reflect padding on both sides (`center=True`),
//! * magnitude spectrum (`|X|`, not power),
//! * HTK mel filterbank, no slaney normalization, triangles over
//!   `fmin=0 .. fmax=sr/2`.

use rustfft::num_complex::Complex;
use rustfft::FftPlanner;

/// Everything downstream is fixed to this rate: prompts, the flow model, and
/// the 24 kHz leg of the vocoder.
pub const SAMPLE_RATE: u32 = 24_000;
pub const N_MELS: usize = 100;
pub const N_FFT: usize = 1024;
pub const HOP_LENGTH: usize = 256;
pub const FEAT_SCALE: f32 = 0.1;

/// Periodic Hann window (librosa `fftbins=True`).
fn hann_periodic(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / n as f32).cos()))
        .collect()
}

/// HTK mel scale: mel = 2595 log10(1 + f/700).
fn hz_to_mel_htk(hz: f64) -> f64 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

fn mel_to_hz_htk(mel: f64) -> f64 {
    700.0 * (10f64.powf(mel / 2595.0) - 1.0)
}

/// Triangular HTK filterbank, `norm=None` (no slaney area normalization).
///
/// N_MELS+2 edge frequencies are linearly spaced on the mel scale between
/// fmin and fmax; each row is a triangle between three consecutive edges.
fn mel_filterbank_htk() -> Vec<Vec<f64>> {
    let fmax = SAMPLE_RATE as f64 / 2.0;
    let mel_min = hz_to_mel_htk(0.0);
    let mel_max = hz_to_mel_htk(fmax);
    let edges: Vec<f64> = (0..N_MELS + 2)
        .map(|i| {
            mel_to_hz_htk(mel_min + (mel_max - mel_min) * i as f64 / (N_MELS + 1) as f64)
        })
        .collect();

    let n_bins = N_FFT / 2 + 1;
    let mut bank = vec![vec![0.0f64; n_bins]; N_MELS];
    for (m, row) in bank.iter_mut().enumerate() {
        let (left, center, right) = (edges[m], edges[m + 1], edges[m + 2]);
        for (b, slot) in row.iter_mut().enumerate() {
            let f = b as f64 * SAMPLE_RATE as f64 / N_FFT as f64;
            if f > left && f < right {
                *slot = if f <= center {
                    (f - left) / (center - left)
                } else {
                    (right - f) / (right - center)
                };
            }
        }
    }
    bank
}

/// Reflect-pad `data` by `pad` samples on both ends, like `np.pad(mode="reflect")`.
fn reflect_pad(data: &[f32], pad: usize) -> Vec<f32> {
    let n = data.len();
    if n == 0 {
        return Vec::new();
    }
    let idx = |i: usize| i.min(n - 1);
    let mut out = Vec::with_capacity(n + 2 * pad);
    // Mirror without repeating the edge sample: [1,2,3] pad 2 -> [3,2,1,2,3,2,1].
    for i in 0..pad {
        out.push(data[idx(pad - i)]);
    }
    out.extend_from_slice(data);
    for i in 0..pad {
        out.push(data[idx(n - 2 - (i % n))]);
    }
    out
}

/// Magnitude STFT of `audio` (already feature-rate compatible): returns
/// `[n_frames][n_bins]` with `n_bins = N_FFT/2 + 1`.
fn magnitude_stft(audio: &[f32]) -> Vec<Vec<f32>> {
    let window = hann_periodic(N_FFT);
    let n_bins = N_FFT / 2 + 1;
    let n_frames = if audio.len() >= N_FFT {
        (audio.len() - N_FFT) / HOP_LENGTH + 1
    } else {
        return Vec::new();
    };
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(N_FFT);
    let mut frames = Vec::with_capacity(n_frames);
    for t in 0..n_frames {
        let start = t * HOP_LENGTH;
        let mut frame: Vec<Complex<f32>> = window
            .iter()
            .enumerate()
            .map(|(i, w)| Complex::new(audio[start + i] * *w, 0.0))
            .collect();
        fft.process(&mut frame);
        frames.push(frame.iter().take(n_bins).map(|c| c.norm()).collect());
    }
    frames
}

/// Extract `[T][N_MELS]` log-mel features scaled by `FEAT_SCALE`, the exact
/// tensor layout the reference feeds its graphs (`[1, T, 100]` once batched).
pub fn extract_mel_features(audio: &[f32]) -> Vec<Vec<f32>> {
    let padded = reflect_pad(audio, N_FFT / 2);
    let mags = magnitude_stft(&padded);
    let bank = mel_filterbank_htk();
    let mut out = Vec::with_capacity(mags.len());
    for frame in mags {
        let mut row = Vec::with_capacity(N_MELS);
        for filter in &bank {
            let energy: f64 = filter
                .iter()
                .zip(frame.iter())
                .map(|(w, m)| w * *m as f64)
                .sum();
            let clamped = energy.max(1e-7);
            row.push((clamped.ln() as f32) * FEAT_SCALE);
        }
        out.push(row);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mel_frame_rate_matches_hop() {
        // One second of audio: T frames ~= sr / hop + 1 (center padding).
        let audio = vec![0.0f32; SAMPLE_RATE as usize];
        let feats = extract_mel_features(&audio);
        let expected = SAMPLE_RATE as usize / HOP_LENGTH + 1;
        assert!(
            (feats.len() as i64 - expected as i64).abs() <= 2,
            "T={} expected ~{}",
            feats.len(),
            expected
        );
        assert_eq!(feats[0].len(), N_MELS);
    }

    #[test]
    fn test_reflect_pad_matches_numpy_for_known_case() {
        assert_eq!(reflect_pad(&[1.0, 2.0, 3.0], 2), vec![3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0]);
    }

    #[test]
    fn test_filterbank_rows_are_triangles_peaking_once() {
        let bank = mel_filterbank_htk();
        for row in &bank {
            let peak_count = row.iter().filter(|w| **w > 0.99).count();
            assert!(peak_count <= 1, "each triangle peaks at most once");
            assert!(row.iter().any(|w| *w > 0.0), "each filter overlaps some bin");
        }
    }

    #[test]
    fn test_sine_gives_peak_at_its_bin() {
        // 1 kHz tone at 24 kHz sr: the mel row nearest 1 kHz must dominate.
        let sr = SAMPLE_RATE as f32;
        let audio: Vec<f32> = (0..SAMPLE_RATE as usize)
            .map(|i| (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / sr).sin())
            .collect();
        let feats = extract_mel_features(&audio);
        let frame = &feats[feats.len() / 2];
        let peak = frame
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        // 1 kHz sits in the low third of the 0..12 kHz mel range.
        assert!(peak < N_MELS / 3, "peak mel row {peak} too high for a 1 kHz tone");
    }
}
