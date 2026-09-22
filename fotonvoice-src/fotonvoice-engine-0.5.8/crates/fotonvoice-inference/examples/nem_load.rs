//! Minimal repro 2: load encoder + decoder, run one zero-mel chunk forward
use ort::session::builder::GraphOptimizationLevel;
use ort::session::{Session, SessionInputValue};
use ort::value::Tensor;

fn main() {
    let dir = std::env::var("FOTON_NEMOTRON_DIR").expect("FOTON_NEMOTRON_DIR");
    println!("loading sessions...");
    let mut encoder = Session::builder()
        .expect("builder")
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .expect("opt")
        .with_intra_threads(8)
        .expect("threads")
        .commit_from_file(format!("{dir}/encoder_model.onnx"))
        .expect("load encoder");
    println!("encoder loaded");
    let mut decoder = Session::builder()
        .expect("builder")
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .expect("opt")
        .with_intra_threads(8)
        .expect("threads")
        .commit_from_file(format!("{dir}/decoder_model.onnx"))
        .expect("load decoder");
    println!("decoder loaded");

    let mel = vec![0.0f32; 128 * 65];
    let mel_tensor = Tensor::from_array(([1_usize, 128_usize, 65_usize], mel)).expect("mel");
    let length = Tensor::from_array(([1_usize], vec![65i64])).expect("len");
    let cch = vec![0.0f32; 24 * 70 * 1024];
    let cch_tensor = Tensor::from_array(([1_usize, 24_usize, 70_usize, 1024_usize], cch)).expect("cch");
    let cti = vec![0.0f32; 24 * 1024 * 8];
    let cti_tensor = Tensor::from_array(([1_usize, 24_usize, 1024_usize, 8_usize], cti)).expect("cti");
    let cll = Tensor::from_array(([1_usize], vec![0i64])).expect("cll");

    let feed: Vec<(&str, SessionInputValue)> = vec![
        ("audio_signal", mel_tensor.into()),
        ("length", length.into()),
        ("cache_last_channel", cch_tensor.into()),
        ("cache_last_time", cti_tensor.into()),
        ("cache_last_channel_len", cll.into()),
    ];
    println!("encoder run...");
    let outs = encoder.run(feed).expect("encoder run");
    println!("encoder done; outputs: {}", outs.len());
    let (_, enc_data) = outs[0].try_extract_tensor::<f32>().expect("enc out");
    println!("enc shape len {}", enc_data.len());

    let frame: Vec<f32> = (0..1024).map(|c| enc_data[c * 7]).collect();
    let frame_tensor = Tensor::from_array(([1_usize, 1024_usize, 1_usize], frame)).expect("frame");
    let targets = Tensor::from_array(([1_usize, 1_usize], vec![1024i32])).expect("targets");
    let target_length = Tensor::from_array(([1_usize], vec![1i32])).expect("tl");
    let s1 = vec![0.0f32; 2 * 640];
    let s1_tensor = Tensor::from_array(([2_usize, 1_usize, 640_usize], s1)).expect("s1");
    let s2_tensor = Tensor::from_array((
        [2_usize, 1_usize, 640_usize],
        vec![0.0f32; 2 * 640],
    ))
    .expect("s2");

    let dfeed: Vec<(&str, SessionInputValue)> = vec![
        ("encoder_outputs", frame_tensor.into()),
        ("targets", targets.into()),
        ("target_length", target_length.into()),
        ("input_states_1", s1_tensor.into()),
        ("input_states_2", s2_tensor.into()),
    ];
    println!("decoder run...");
    let douts = decoder.run(dfeed).expect("decoder run");
    println!("decoder done; outputs: {}", douts.len());
    println!("OK");
}
