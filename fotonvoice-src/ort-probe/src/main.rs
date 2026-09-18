use std::time::Instant;

use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;

fn main() {
    let mut args = std::env::args().skip(1);
    let mode = args.next().unwrap_or_else(|| "cpu".into());
    let dir = args
        .next()
        .unwrap_or_else(|| r"C:\apps\fotonvoice-engine\models\parakeet\tdt-0.6b-v3".into());
    let threads = std::thread::available_parallelism()
        .map(|n| (n.get() / 2).max(1))
        .unwrap_or(2);
    println!("probe start, mode = {mode}, intra threads = {threads}");

    let mut lib_handle: Option<ort::ep::ExecutionProviderLibrary> = None;
    let env = if mode == "webgpu" {
        let lib = std::env::var("ORT_DYLIB_PATH")
            .ok()
            .map(|p| std::path::PathBuf::from(p).parent().unwrap().join("onnxruntime_providers_webgpu.dll"))
            .unwrap();
        let env = ort::environment::Environment::current().unwrap();
        lib_handle = Some(env.register_ep_library("webgpu", lib).unwrap());
        for d in env.devices() {
            println!("DEVICE: ep = {:?}, id = {:?}", d.ep(), d.id());
        }
        Some(env)
    } else {
        None
    };

    for f in [
        "nemo128.onnx",
        "encoder-model.int8.onnx",
        "decoder_joint-model.int8.onnx",
    ] {
        let t0 = Instant::now();
        println!("LOAD {f}");
        let r = (|| -> Result<Session, Box<dyn std::error::Error>> {
            let b = Session::builder()?;
            let b = b.with_optimization_level(GraphOptimizationLevel::Level1)?;
            let mut b = b.with_intra_threads(threads)?;
            if mode == "cuda" {
                b = b.with_execution_providers([ort::ep::CUDA::default().build().error_on_failure()])?;
            } else if mode == "dml" {
                b = b.with_execution_providers([
                    ort::ep::DirectML::default().with_device_id(0).build().error_on_failure()
                ])?;
            } else if mode == "webgpu" {
                let wg: Vec<_> = env
                    .as_ref()
                    .unwrap()
                    .devices()
                    .filter(|d| d.ep().unwrap().to_ascii_lowercase().contains("webgpu"))
                    .collect();
                b = b.with_devices(wg, None)?;
            }
            Ok(b.commit_from_file(format!("{dir}\\{f}"))?)
        })();
        match r {
            Ok(_s) => println!("DONE {f} in {:.1?}", t0.elapsed()),
            Err(e) => {
                println!("ERROR {f}: {e}");
                return;
            }
        }
    }
    println!("ALL-DONE");
    drop(lib_handle);
    std::process::exit(0);
}
