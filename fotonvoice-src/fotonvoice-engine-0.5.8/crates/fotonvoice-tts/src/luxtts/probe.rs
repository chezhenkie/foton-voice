//! End-to-end LuxTTS probe: real ONNX graphs, real reference clip, real espeak.

use std::path::PathBuf;

use crate::luxtts::model::LuxTTSModel;
use crate::luxtts::prompt;
use crate::luxtts::g2p;

fn probe_dir() -> PathBuf {
    PathBuf::from(std::env::var("LUX_PROBE_DIR").expect("set LUX_PROBE_DIR"))
}

#[test]
#[ignore]
fn g2p_symbol_stream_prints() {
    let text = "Take this kiss upon the brow! And, in parting from you now, thus much let me avow.";
    let symbols = g2p::text_to_symbols(text).expect("espeak-ng");
    let joined = symbols.join("");
    println!("{} symbols: {:?}", symbols.len(), joined);
    assert!(symbols.len() > 40, "symbol stream suspiciously short");
}

#[test]
#[ignore]
fn synthesize_short_sentence_dumps_wav() {
    let dir = probe_dir();
    let quantized = std::env::var("LUX_PROBE_QUANTIZED").map(|v| v != "0").unwrap_or(true);
    let mut model = LuxTTSModel::load(&dir, quantized).expect("load model");

    let clip = PathBuf::from(std::env::var("LUX_PROBE_CLIP").expect("set LUX_PROBE_CLIP"));
    let transcript =
        std::fs::read_to_string(clip.with_extension("txt")).expect("read transcript");
    let prompt = prompt::load_or_encode(&clip, transcript.trim(), &model.vocab, std::env::var("LUX_PROBE_REF_DURATION").ok().and_then(|v| v.parse().ok()).unwrap_or(15.0)).expect("prompt");
    println!("prompt features_len = {} rms = {:.5}", prompt.features_len, prompt.rms);

    let text = std::env::var("LUX_PROBE_TEXT")
        .unwrap_or_else(|_| "Hello, this is a voice cloning test.".to_string());
    let symbols = g2p::text_to_symbols(&text).expect("espeak-ng");
    let (tokens, skipped) = model.vocab.encode(&symbols);
    println!("tokens: {} skipped: {:?}", tokens.len(), skipped);
    assert!(!tokens.is_empty());

    let steps = std::env::var("LUX_PROBE_STEPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8);
    let smooth = std::env::var("LUX_PROBE_SMOOTH").map(|v| v != "0").unwrap_or(false);
    let audio = model
        .synthesize(&tokens, &prompt, steps, 0.9, 3.0, 1.0, 0, smooth)
        .expect("synthesize");
    println!("audio samples: {} ({}s @ 48k)", audio.len(), audio.len() / 48000);

    let rms = (audio.iter().map(|s| s * s).sum::<f32>() / audio.len().max(1) as f32).sqrt();
    let peak = audio.iter().fold(0.0f32, |a, s| a.max(s.abs()));
    println!("rms = {:.5} peak = {:.5}", rms, peak);
    assert!(audio.len() > 24000, "suspiciously short audio");
    assert!(peak > 0.01, "audio is near-silence");

    let mut out = dir.join("lux-probe-out.wav");
    if let Some(p) = std::env::var_os("LUX_PROBE_OUT") {
        out = PathBuf::from(p);
    }
    write_wav(&out, &audio, crate::luxtts::model::OUTPUT_SAMPLE_RATE);
    println!("wrote {}", out.display());
}

fn write_wav(path: &std::path::Path, samples: &[f32], rate: u32) {
    use std::io::Write;
    let mut f = std::fs::File::create(path).expect("create wav");
    let data_len = samples.len() * 2;
    let mut header = Vec::with_capacity(44);
    header.extend_from_slice(b"RIFF");
    header.extend_from_slice(&(36 + data_len as u32).to_le_bytes());
    header.extend_from_slice(b"WAVE");
    header.extend_from_slice(b"fmt ");
    header.extend_from_slice(&16u32.to_le_bytes());
    header.extend_from_slice(&1u16.to_le_bytes());
    header.extend_from_slice(&1u16.to_le_bytes());
    header.extend_from_slice(&rate.to_le_bytes());
    header.extend_from_slice(&(rate * 2).to_le_bytes());
    header.extend_from_slice(&2u16.to_le_bytes());
    header.extend_from_slice(&16u16.to_le_bytes());
    header.extend_from_slice(b"data");
    header.extend_from_slice(&(data_len as u32).to_le_bytes());
    f.write_all(&header).unwrap();
    for s in samples {
        let v = (s.clamp(-1.0, 1.0) * 32767.0) as i16;
        f.write_all(&v.to_le_bytes()).unwrap();
    }
}
