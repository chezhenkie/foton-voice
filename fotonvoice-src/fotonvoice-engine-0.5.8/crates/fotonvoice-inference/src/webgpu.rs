use std::path::PathBuf;
use std::sync::{Arc, OnceLock};

use ort::device::Device;
use ort::environment::Environment;
use ort::ep::ExecutionProviderLibrary;

static WEBGPU: OnceLock<Option<Registration>> = OnceLock::new();

struct Registration {
    env: Arc<Environment>,
    _lib: ExecutionProviderLibrary,
}

fn provider_dll() -> Option<PathBuf> {
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Ok(p) = std::env::var("ORT_DYLIB_PATH") {
        if let Some(parent) = PathBuf::from(&p).parent() {
            candidates.push(parent.join("onnxruntime_providers_webgpu.dll"));
        }
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            candidates.push(dir.join("onnxruntime_providers_webgpu.dll"));
        }
    }
    candidates.into_iter().find(|c| c.is_file())
}

pub(crate) fn webgpu_devices() -> Vec<Device<'static>> {
    let reg = WEBGPU.get_or_init(|| match provider_dll() {
        None => {
            tracing::warn!(
                "WebGPU: onnxruntime_providers_webgpu.dll not found beside the runtime; running on the CPU"
            );
            None
        }
        Some(dll) => match Environment::current().and_then(|env| {
            let lib = env.register_ep_library("webgpu", &dll)?;
            Ok::<_, ort::Error>(Registration { env, _lib: lib })
        }) {
            Ok(reg) => {
                tracing::info!("WebGPU: plugin EP registered from {}", dll.display());
                Some(reg)
            }
            Err(e) => {
                tracing::warn!("WebGPU: plugin EP registration failed ({e}); running on the CPU");
                None
            }
        },
    });

    let Some(reg) = reg.as_ref() else {
        return Vec::new();
    };
    reg.env
        .devices()
        .filter(|d| {
            d.ep()
                .map(|ep| ep.to_ascii_lowercase().contains("webgpu"))
                .unwrap_or(false)
        })
        .collect()
}
