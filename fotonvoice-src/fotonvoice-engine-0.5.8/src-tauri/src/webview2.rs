//! Fixed-version WebView2 runtime bootstrap.

use std::fs;
use std::path::Path;

/// Extract `WebView2Runtime.zip` from the portable root (if present) and set
pub fn ensure_fixed_runtime(app_root: &Path) {
    let zip_path = app_root.join("WebView2Runtime.zip");
    let metadata = match fs::metadata(&zip_path) {
        Ok(m) if m.is_file() => m,
        _ => return,
    };
    let expected = format!(
        "zip-{}-{}",
        metadata.len(),
        stamp(&zip_path)
    );

    let out_dir = app_root.join("WebView2Runtime");
    let engine_exe = out_dir.join("msedgewebview2.exe");
    let marker = out_dir.join(".fotonvoice-runtime-marker");

    let up_to_date = engine_exe.is_file()
        && fs::read_to_string(&marker)
            .map(|s| s.trim() == expected)
            .unwrap_or(false);
    if !up_to_date {
        let _ = fs::remove_dir_all(&out_dir);
        if let Err(e) = extract(&zip_path, &out_dir) {
            eprintln!("Could not extract WebView2Runtime.zip: {e}");
            return;
        }
        let _ = fs::write(&marker, &expected);
    }

    if engine_exe.is_file() {
        std::env::set_var("WEBVIEW2_BROWSER_EXECUTABLE_FOLDER", &out_dir);
    }
}

/// Cheap zip identity: length + mtime seconds. Changes whenever the runtime
fn stamp(p: &Path) -> String {
    p.metadata()
        .and_then(|m| m.modified())
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs())
        .unwrap_or(0)
        .to_string()
}

fn extract(zip_path: &Path, out_dir: &Path) -> Result<(), String> {
    let file = fs::File::open(zip_path).map_err(|e| e.to_string())?;
    let mut archive = zip::ZipArchive::new(file).map_err(|e| e.to_string())?;
    archive.extract(out_dir).map_err(|e| e.to_string())
}
