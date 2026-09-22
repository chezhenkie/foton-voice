#[cfg(feature = "moonshine")]
use fotonvoice_inference::moonshine;
#[cfg(feature = "nemotron-streaming")]
use fotonvoice_inference::nemotron_streaming;
#[cfg(feature = "parakeet")]
use fotonvoice_inference::parakeet;

#[tauri::command]
pub async fn check_model_downloaded(model_size: String, model_dir: Option<String>) -> Result<bool, String> {
    let dir = model_dir.unwrap_or_default();
    Ok(fotonvoice_inference::whisper_cpp::is_model_downloaded(&model_size, &dir))
}

#[tauri::command]
pub async fn download_model(model_size: String, model_dir: String) -> Result<(), String> {
    fotonvoice_inference::whisper_cpp::download_model(&model_size, &model_dir)
        .await
        .map_err(|e| e.to_string())
}

/// Delete the whisper.cpp GGUF file(s) for `model_size`.
#[tauri::command]
pub async fn delete_model(model_size: String, model_dir: String) -> Result<(), String> {
    fotonvoice_inference::whisper_cpp::delete_model(&model_size, &model_dir)
        .map_err(|e| e.to_string())
}

/// One set of check/download/delete commands per in-process ONNX STT backend.
/// The three commands differ only in the feature gate, module path and the
/// "not compiled" error text, so a single macro keeps them in lockstep.
macro_rules! onnx_engine_commands {
    ($module:ident, $feature:literal, $compiled:path, $avail:ident, $check:ident, $download:ident, $delete:ident, $label:literal) => {
        #[tauri::command]
        pub fn $avail() -> bool {
            $compiled
        }

        #[tauri::command]
        pub async fn $check(model_size: String) -> Result<bool, String> {
            #[cfg(feature = $feature)]
            {
                Ok($module::is_model_downloaded(&model_size, ""))
            }
            #[cfg(not(feature = $feature))]
            {
                let _ = model_size;
                Ok(false)
            }
        }

        #[tauri::command]
        pub async fn $download(model_size: String) -> Result<(), String> {
            #[cfg(feature = $feature)]
            {
                $module::download_model(&model_size, "")
                    .await
                    .map_err(|e| e.to_string())
            }
            #[cfg(not(feature = $feature))]
            {
                let _ = model_size;
                Err(concat!("This build was compiled without the ", $label, " backend. Rebuild with `--features ", $feature, "` to use it.").into())
            }
        }

        #[tauri::command]
        pub async fn $delete(model_size: String) -> Result<(), String> {
            #[cfg(feature = $feature)]
            {
                $module::delete_model(&model_size, "").map_err(|e| e.to_string())
            }
            #[cfg(not(feature = $feature))]
            {
                let _ = model_size;
                Err(concat!("This build was compiled without the ", $label, " backend. Rebuild with `--features ", $feature, "` to use it.").into())
            }
        }
    };
}

onnx_engine_commands!(
    moonshine,
    "moonshine",
    fotonvoice_inference::MOONSHINE_COMPILED,
    moonshine_available,
    check_moonshine_downloaded,
    download_moonshine_model,
    delete_moonshine_model,
    "Moonshine"
);
onnx_engine_commands!(
    parakeet,
    "parakeet",
    fotonvoice_inference::PARAKEET_COMPILED,
    parakeet_available,
    check_parakeet_downloaded,
    download_parakeet_model,
    delete_parakeet_model,
    "Parakeet"
);
onnx_engine_commands!(
    nemotron_streaming,
    "nemotron-streaming",
    fotonvoice_inference::NEMOTRON_STREAMING_COMPILED,
    nemotron_streaming_available,
    check_nemotron_streaming_downloaded,
    download_nemotron_streaming_model,
    delete_nemotron_streaming_model,
    "Nemotron streaming"
);

#[tauri::command]
pub async fn check_s1_mini_downloaded(model_dir: Option<String>) -> Result<bool, String> {
    Ok(fotonvoice_inference::s1_mini::is_s1_mini_downloaded(model_dir.as_deref()))
}

#[tauri::command]
pub async fn download_s1_mini_model(model_dir: Option<String>) -> Result<(), String> {
    fotonvoice_inference::s1_mini::download_s1_mini_assets(model_dir.as_deref())
        .await
        .map_err(|e| e.to_string())
}
