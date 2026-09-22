use std::path::PathBuf;

fn main() {
    ensure_sidecar_placeholder("fotonvoice-llm-sidecar");

    println!("cargo:rerun-if-env-changed=FOTONVOICE_BUILD_SHA");

    tauri_build::build()
}

fn ensure_sidecar_placeholder(name: &str) {
    let manifest_dir = match std::env::var("CARGO_MANIFEST_DIR") {
        Ok(dir) => PathBuf::from(dir),
        Err(_) => return,
    };
    let target = std::env::var("TARGET").unwrap_or_default();
    if target.is_empty() {
        return;
    }
    let exe_suffix = if target.contains("windows") { ".exe" } else { "" };

    let dir = manifest_dir.join("binaries");
    let sidecar = dir.join(format!("{name}-{target}{exe_suffix}"));
    if sidecar.exists() {
        return;
    }
    if let Err(e) = std::fs::create_dir_all(&dir) {
        println!("cargo:warning=could not create {}: {e}", dir.display());
        return;
    }
    if let Err(e) = std::fs::write(&sidecar, []) {
        println!("cargo:warning=could not write placeholder sidecar {}: {e}", sidecar.display());
    }
}
