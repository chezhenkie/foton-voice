//! Piper voice catalogue, path resolution, and standalone binary/voice

#[cfg(unix)]
use std::path::Path;
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::Arc;

use anyhow::{Context, Result};
use tracing::{info, warn};


#[derive(Debug, Clone)]
pub struct VoiceInfo {
    pub name: &'static str,
    pub quality: &'static str,
    pub sample_rate: u32,
    pub filename: &'static str,
}

pub static PIPER_VOICES: &[VoiceInfo] = &[
    VoiceInfo { name: "announcer",                       quality: "medium", sample_rate: 22050, filename: "announcer.onnx" },
    VoiceInfo { name: "BT7274",                          quality: "medium", sample_rate: 22050, filename: "BT7274.onnx" },
    VoiceInfo { name: "de_DE-mls-medium",                quality: "medium", sample_rate: 22050, filename: "de_DE-mls-medium.onnx" },
    VoiceInfo { name: "de_DE-thorsten_emotional-medium", quality: "medium", sample_rate: 22050, filename: "de_DE-thorsten_emotional-medium.onnx" },
    VoiceInfo { name: "de_DE-thorsten-high",             quality: "high",   sample_rate: 22050, filename: "de_DE-thorsten-high.onnx" },
    VoiceInfo { name: "de_DE-thorsten-medium",           quality: "medium", sample_rate: 22050, filename: "de_DE-thorsten-medium.onnx" },
    VoiceInfo { name: "en-gb-southern_english_female-low", quality: "low", sample_rate: 16000, filename: "en-gb-southern_english_female-low.onnx" },
    VoiceInfo { name: "en-us-lessac-medium",             quality: "medium", sample_rate: 16000, filename: "en-us-lessac-medium.onnx" },
    VoiceInfo { name: "en_GB-alba-medium",               quality: "medium", sample_rate: 22050, filename: "en_GB-alba-medium.onnx" },
    VoiceInfo { name: "en_GB-cori-high",                 quality: "high",   sample_rate: 22050, filename: "en_GB-cori-high.onnx" },
    VoiceInfo { name: "en_GB-cori-medium",               quality: "medium", sample_rate: 22050, filename: "en_GB-cori-medium.onnx" },
    VoiceInfo { name: "en_GB-jenny_dioco-medium",        quality: "medium", sample_rate: 22050, filename: "en_GB-jenny_dioco-medium.onnx" },
    VoiceInfo { name: "en_GB-northern_english_male-medium", quality: "medium", sample_rate: 22050, filename: "en_GB-northern_english_male-medium.onnx" },
    VoiceInfo { name: "en_GB-semaine-medium",            quality: "medium", sample_rate: 22050, filename: "en_GB-semaine-medium.onnx" },
    VoiceInfo { name: "en_GB-vctk-medium",               quality: "medium", sample_rate: 22050, filename: "en_GB-vctk-medium.onnx" },
    VoiceInfo { name: "en_US-carlin-high",               quality: "high",   sample_rate: 22050, filename: "en_US-carlin-high.onnx" },
    VoiceInfo { name: "en_US-data_7024-medium",          quality: "medium", sample_rate: 22050, filename: "en_US-data_7024-medium.onnx" },
    VoiceInfo { name: "en_US-eminem-medium",             quality: "medium", sample_rate: 22050, filename: "en_US-eminem-medium.onnx" },
    VoiceInfo { name: "en_US-hal_6409-medium",           quality: "medium", sample_rate: 22050, filename: "en_US-hal_6409-medium.onnx" },
    VoiceInfo { name: "es_ES-davefx-medium",             quality: "medium", sample_rate: 22050, filename: "es_ES-davefx-medium.onnx" },
    VoiceInfo { name: "es_ES-sharvard-medium",           quality: "medium", sample_rate: 22050, filename: "es_ES-sharvard-medium.onnx" },
    VoiceInfo { name: "es_MX-claude-high",               quality: "high",   sample_rate: 22050, filename: "es_MX-claude-high.onnx" },
    VoiceInfo { name: "fr_FR-mls-medium",                quality: "medium", sample_rate: 22050, filename: "fr_FR-mls-medium.onnx" },
    VoiceInfo { name: "fr_FR-siwis-medium",              quality: "medium", sample_rate: 22050, filename: "fr_FR-siwis-medium.onnx" },
    VoiceInfo { name: "fr_FR-tom-medium",                quality: "medium", sample_rate: 44100, filename: "fr_FR-tom-medium.onnx" },
    VoiceInfo { name: "fr_FR-upmc-medium",               quality: "medium", sample_rate: 22050, filename: "fr_FR-upmc-medium.onnx" },
    VoiceInfo { name: "glados",                          quality: "medium", sample_rate: 22050, filename: "glados.onnx" },
    VoiceInfo { name: "nl_BE-nathalie-medium",           quality: "medium", sample_rate: 22050, filename: "nl_BE-nathalie-medium.onnx" },
    VoiceInfo { name: "nl_BE-rdh-medium",                quality: "medium", sample_rate: 22050, filename: "nl_BE-rdh-medium.onnx" },
    VoiceInfo { name: "nl_NL-alex-medium",               quality: "medium", sample_rate: 22050, filename: "nl_NL-alex-medium.onnx" },
    VoiceInfo { name: "nl_NL-pim-medium",                quality: "medium", sample_rate: 22050, filename: "nl_NL-pim-medium.onnx" },
    VoiceInfo { name: "nl_NL-ronnie-medium",             quality: "medium", sample_rate: 22050, filename: "nl_NL-ronnie-medium.onnx" },
    VoiceInfo { name: "PDA",                             quality: "medium", sample_rate: 22050, filename: "PDA.onnx" },
    VoiceInfo { name: "ScorchAI",                        quality: "medium", sample_rate: 22050, filename: "ScorchAI.onnx" },
];


pub fn piper_voices_dir() -> PathBuf {
    fotonvoice_config::portable::app_root().join("piper-voices")
}

/// Expands a leading `~` to the user's home directory. Shared with `pocket.rs`,
pub(crate) fn expand_tilde(path: &str) -> PathBuf {
    if path == "~" {
        return dirs::home_dir().unwrap_or_else(|| PathBuf::from("~"));
    }
    if let Some(rest) = path.strip_prefix("~/") {
        if let Some(home) = dirs::home_dir() {
            return home.join(rest);
        }
    }
    PathBuf::from(path)
}

fn resolve_voices_dir(voice_dir: &str) -> PathBuf {
    if voice_dir.is_empty() {
        piper_voices_dir()
    } else {
        expand_tilde(voice_dir)
    }
}

pub fn piper_binary() -> Option<PathBuf> {
    let exe = if cfg!(target_os = "windows") { "piper.exe" } else { "piper" };
    let local_dir = fotonvoice_config::portable::app_root().join("piper");
    let local = local_dir.join(exe);
    let local_healthy = if cfg!(unix) {
        local.exists() && local_dir.join("espeak-ng-data").is_dir()
    } else {
        local.exists()
    };
    if local_healthy {
        return Some(local);
    }
    fotonvoice_config::find_in_path("piper")
}


fn voice_name_to_filename(name: &str) -> Option<String> {
    PIPER_VOICES
        .iter()
        .find(|v| v.name == name)
        .map(|v| v.filename.to_string())
}

/// Used by `TtsEngineWorker::speak_piper` to pick the correct playback sample rate.
pub(crate) fn sample_rate_for_voice_path(name: &str, voice_path: &std::path::Path) -> u32 {
    if let Some(v) = PIPER_VOICES.iter().find(|v| v.name == name) {
        return v.sample_rate;
    }
    let json_path = voice_path.with_file_name(format!(
        "{}.json",
        voice_path.file_name().and_then(|f| f.to_str()).unwrap_or_default()
    ));
    if let Ok(raw) = std::fs::read_to_string(&json_path) {
        if let Ok(json) = serde_json::from_str::<serde_json::Value>(&raw) {
            if let Some(rate) = json.pointer("/audio/sample_rate").and_then(|r| r.as_u64()) {
                let rate = u32::try_from(rate).unwrap_or(22050);
                if rate > 0 {
                    return rate;
                }
            }
        }
    }
    22050
}


pub(crate) struct PiperResident {
    child: std::process::Child,
    stdin: std::process::ChildStdin,
    chunks: std::sync::mpsc::Receiver<Vec<u8>>,
    stderr_tail: Arc<std::sync::Mutex<Vec<u8>>>,
    fingerprint: PiperFingerprint,
}

#[derive(PartialEq, Clone, Debug)]
pub(crate) struct PiperFingerprint {
    pub voice_path: PathBuf,
    pub gpu: bool,
    pub length_scale: String,
}

impl PiperResident {
    pub fn fingerprint(&self) -> &PiperFingerprint {
        &self.fingerprint
    }

    /// Whether the piper child has exited on its own (crash / killed).
    pub fn has_exited(&mut self) -> bool {
        matches!(
            self.child.try_wait(),
            Ok(Some(_))
        )
    }    /// Feed one utterance (a single line; newlines must already be normalized).
    pub fn write_line(&mut self, line: &str) -> std::io::Result<()> {
        use std::io::Write;
        self.stdin.write_all(line.as_bytes())?;
        self.stdin.write_all(b"\n")?;
        self.stdin.flush()
    }

    /// Next PCM chunk from piper (up to 8 KiB), or `None` when the process
    pub fn recv_chunk(&self, timeout: std::time::Duration) -> Option<Vec<u8>> {
        match self.chunks.recv_timeout(timeout) {
            Ok(bytes) => Some(bytes),
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => None,
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                None
            }
        }
    }

    /// Discard any PCM still queued from an abandoned utterance so it cannot
    pub fn drain_pending(&self) {
        while let Ok(_stale) = self.chunks.try_recv() {}
    }

    /// Last stderr output (bounded), for error reports.
    pub fn stderr_tail(&self) -> String {
        let tail = self.stderr_tail.lock().unwrap_or_else(|e| e.into_inner());
        String::from_utf8_lossy(&tail).trim().to_string()
    }

    /// Kill the child (stdin closes when this struct is dropped).
    #[allow(dead_code)]
    pub fn kill(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl Drop for PiperResident {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

pub(crate) fn spawn_piper_resident(
    binary: &std::path::Path,
    fingerprint: PiperFingerprint,
) -> Result<PiperResident> {
    let mut cmd = std::process::Command::new(binary);
    #[cfg(target_os = "windows")]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }
    cmd.arg("--model")
        .arg(&fingerprint.voice_path)
        .arg("--length-scale")
        .arg(&fingerprint.length_scale)
        .arg("--output-raw");
    if fingerprint.gpu {
        cmd.arg("--cuda");
    }
    cmd.env("PYTHONUNBUFFERED", "1");
    cmd.stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let mut child = cmd.spawn().context("spawn piper (resident)")?;
    let stdout = child.stdout.take().context("piper stdout")?;
    let stderr = child.stderr.take().context("piper stderr")?;
    let stdin = child.stdin.take().context("piper stdin")?;

    let (chunk_tx, chunk_rx) = std::sync::mpsc::channel::<Vec<u8>>();
    std::thread::Builder::new()
        .name("piper-stdout".into())
        .spawn(move || {
            let mut out = stdout;
            loop {
                let mut block = vec![0u8; 8192];
                match std::io::Read::read(&mut out, &mut block) {
                    Ok(0) => break,
                    Ok(n) => {
                        block.truncate(n);
                        if chunk_tx.send(block).is_err() {
                            break;
                        }
                    }
                    Err(_) => break,
                }
            }
        })?;

    let stderr_tail: Arc<std::sync::Mutex<Vec<u8>>> = Arc::new(std::sync::Mutex::new(Vec::new()));
    let stderr_tail_reader = stderr_tail.clone();
    std::thread::Builder::new()
        .name("piper-stderr".into())
        .spawn(move || {
            use std::io::Read;
            let mut err = stderr;
            let mut buf = [0u8; 4096];
            loop {
                match err.read(&mut buf) {
                    Ok(0) | Err(_) => break,
                    Ok(n) => {
                        let mut tail = stderr_tail_reader.lock().unwrap_or_else(|e| e.into_inner());
                        tail.extend_from_slice(&buf[..n]);
                        if tail.len() > 64 * 1024 {
                            let excess = tail.len() - 64 * 1024;
                            tail.drain(..excess);
                        }
                    }
                }
            }
        })?;

    Ok(PiperResident {
        child,
        stdin,
        chunks: chunk_rx,
        stderr_tail,
        fingerprint,
    })
}

/// Voices present in the voices folder: every `*.onnx` with a paired
pub fn list_local_voices(voice_dir: &str) -> Vec<String> {
    let dir = resolve_voices_dir(voice_dir);
    let mut out = Vec::new();
    if let Ok(entries) = std::fs::read_dir(&dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("onnx") {
                continue;
            }
            let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
                continue;
            };
            if path.with_file_name(format!("{stem}.onnx.json")).exists() {
                out.push(stem.to_string());
            }
        }
    }
    out.sort();
    out
}

pub fn is_voice_downloaded(voice_name: &str, voice_dir: &str) -> bool {
    get_voice_path(voice_name, voice_dir).is_some()
}


const PIPER_RELEASE_BASE: &str =
    "https://github.com/rhasspy/piper/releases/download/v0.0.2/";

pub fn get_voice_path(voice_name: &str, voice_dir: &str) -> Option<PathBuf> {
    let filename = voice_name_to_filename(voice_name)
        .unwrap_or_else(|| format!("{voice_name}.onnx"));

    let voices_dir = resolve_voices_dir(voice_dir);

    let path_onnx = voices_dir.join(&filename);
    let path_json = voices_dir.join(format!("{filename}.json"));
    if path_onnx.exists() && path_json.exists() {
        return Some(path_onnx);
    }

    let filename_lower = filename.to_lowercase();
    let path_onnx_lower = voices_dir.join(&filename_lower);
    let path_json_lower = voices_dir.join(format!("{filename_lower}.json"));
    if path_onnx_lower.exists() && path_json_lower.exists() {
        return Some(path_onnx_lower);
    }

    let path_raw_lower = voices_dir.join(format!("{}.onnx", voice_name.to_lowercase()));
    let path_raw_json_lower =
        voices_dir.join(format!("{}.onnx.json", voice_name.to_lowercase()));
    if path_raw_lower.exists() && path_raw_json_lower.exists() {
        return Some(path_raw_lower);
    }

    None
}

/// Extracts the piper release tarball into `dest_dir`, preserving the archive's
#[cfg(unix)]
fn extract_piper_archive(bytes: &[u8], dest_dir: &Path) -> Result<()> {
    let cursor = std::io::Cursor::new(bytes);
    let tar = flate2::read::GzDecoder::new(cursor);
    let mut archive = tar::Archive::new(tar);

    for entry in archive.entries()? {
        let mut entry = entry?;
        let path = entry.path()?.into_owned();

        let rel: PathBuf = path
            .components()
            .skip(1)
            .filter(|c| matches!(c, std::path::Component::Normal(_)))
            .collect();
        if rel.as_os_str().is_empty() {
            continue;
        }

        let dest = dest_dir.join(&rel);
        if let Some(parent) = dest.parent() {
            std::fs::create_dir_all(parent)?;
        }
        entry.unpack(&dest)?;
    }

    use std::os::unix::fs::PermissionsExt;
    let exe = dest_dir.join("piper");
    if let Ok(metadata) = std::fs::metadata(&exe) {
        let mut perms = metadata.permissions();
        perms.set_mode(0o755);
        let _ = std::fs::set_permissions(&exe, perms);
    }

    Ok(())
}

/// Fetch the standalone Piper binary FotonVoice Engine manages for itself.
#[cfg(not(unix))]
pub async fn download_piper_binary() -> Result<()> {
    {
        let dir = fotonvoice_config::portable::app_root().join("piper");
        anyhow::bail!(
            "FotonVoice Engine cannot install Piper automatically on this platform yet. \
             Download the Piper release for Windows from \
             https://github.com/rhasspy/piper/releases and put piper.exe in \
             {}, or choose a different engine in Settings -> Text-to-Speech \
             (Pocket-TTS needs no external binary).",
            dir.display()
        );
    }
}

#[cfg(unix)]
pub async fn download_piper_binary() -> Result<()> {
    {
        let local_dir = fotonvoice_config::portable::app_root().join("piper");

        let dest_exe = local_dir.join("piper");
        if dest_exe.exists() && local_dir.join("espeak-ng-data").is_dir() {
            return Ok(());
        }
        if local_dir.exists() {
            info!("Repairing broken standalone Piper install at {}", local_dir.display());
            tokio::fs::remove_dir_all(&local_dir).await?;
        }
        tokio::fs::create_dir_all(&local_dir).await?;

        info!("Downloading standalone Piper binary...");
        let url =
            "https://github.com/rhasspy/piper/releases/download/v1.2.0/piper_amd64.tar.gz";

        let response = reqwest::get(url).await?.error_for_status()?;
        let bytes = response.bytes().await?;

        info!("Extracting Piper binary...");
        extract_piper_archive(&bytes, &local_dir)?;
        info!("Standalone Piper binary installed to {}", dest_exe.display());
    }
    Ok(())
}

pub async fn download_voice(voice_name: &str, voice_dir: &str) -> Result<()> {
    if piper_binary().is_none() {
        if let Err(e) = download_piper_binary().await {
            warn!("Failed to download standalone piper binary: {e}");
        }
    }

    let voices_dir = resolve_voices_dir(voice_dir);
    tokio::fs::create_dir_all(&voices_dir).await?;

    if get_voice_path(voice_name, voice_dir).is_some() {
        info!("Voice {} is already downloaded.", voice_name);
        return Ok(());
    }

    let tarball_url = format!("{PIPER_RELEASE_BASE}voice-{voice_name}.tar.gz");
    info!("Downloading voice tarball: {tarball_url}");

    let response = reqwest::get(&tarball_url).await?.error_for_status()?;
    let bytes = response.bytes().await?;

    info!("Extracting voice files...");
    let cursor = std::io::Cursor::new(bytes);
    let tar = flate2::read::GzDecoder::new(cursor);
    let mut archive = tar::Archive::new(tar);

    for entry in archive.entries()? {
        let mut entry = entry?;
        let path = entry.path()?.into_owned();
        let file_name = match path.file_name() {
            Some(name) => name.to_string_lossy().to_string(),
            None => continue,
        };

        if file_name.ends_with(".onnx") || file_name.ends_with(".onnx.json") {
            let dest_path = voices_dir.join(&file_name);
            let mut temp_file = tempfile::NamedTempFile::new_in(&voices_dir)?;
            std::io::copy(&mut entry, &mut temp_file)?;
            temp_file.persist(&dest_path)?;
            info!("Extracted: {}", dest_path.display());
        }
    }

    info!("Voice files successfully downloaded and extracted.");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    fn create_fake_voice(dir: &std::path::Path, filename: &str) {
        fs::write(dir.join(filename), b"fake onnx model").unwrap();
        fs::write(dir.join(format!("{filename}.json")), b"{}").unwrap();
    }


    #[test]
    fn resident_piper_streams_two_lines() {
        let Some(binary) = piper_binary() else {
            println!("piper not installed - skipping");
            return;
        };
        let voice_name = match list_local_voices("").first() {
            Some(name) => name.clone(),
            None => {
                println!("no local piper voice - skipping");
                return;
            }
        };
        let Some(voice_path) = get_voice_path(&voice_name, "") else {
            println!("voice files missing - skipping");
            return;
        };

        let fingerprint = PiperFingerprint {
            voice_path,
            gpu: false,
            length_scale: "1".into(),
        };
        let mut resident =
            spawn_piper_resident(&binary, fingerprint).expect("spawn resident piper");

        resident
            .write_line("First resident line.")
            .expect("write line 1");
        let mut total1 = 0usize;
        let mut quiet = 0;
        while quiet < 10 {
            match resident.recv_chunk(std::time::Duration::from_millis(100)) {
                Some(bytes) => {
                    total1 += bytes.len();
                    quiet = 0;
                }
                None => quiet += 1,
            }
        }
        assert!(total1 > 1000, "line 1 produced only {total1} bytes");

        resident
            .write_line("Second line of the resident test.")
            .expect("write line 2");
        let mut total2 = 0usize;
        quiet = 0;
        while quiet < 10 {
            match resident.recv_chunk(std::time::Duration::from_millis(100)) {
                Some(bytes) => {
                    total2 += bytes.len();
                    quiet = 0;
                }
                None => quiet += 1,
            }
        }
        assert!(total2 > 100, "line 2 produced only {total2} bytes");
        assert!(!resident.has_exited(), "resident piper exited during test");
    }


    #[test]
    fn test_resolve_voices_dir_empty_uses_default() {
        let result = resolve_voices_dir("");
        assert_eq!(result, piper_voices_dir());
    }

    #[test]
    fn test_resolve_voices_dir_absolute_path() {
        let dir = tempdir().unwrap();
        let path = dir.path().to_str().unwrap();
        let result = resolve_voices_dir(path);
        assert_eq!(result, dir.path());
    }

    #[test]
    fn test_resolve_voices_dir_tilde_expands() {
        let result = resolve_voices_dir("~/my-voices");
        let home = dirs::home_dir().unwrap();
        assert_eq!(result, home.join("my-voices"));
    }

    #[test]
    fn test_resolve_voices_dir_tilde_alone_expands() {
        let result = resolve_voices_dir("~");
        let home = dirs::home_dir().unwrap();
        assert_eq!(result, home);
    }


    #[test]
    fn test_expand_tilde_home() {
        let home = dirs::home_dir().unwrap();
        assert_eq!(expand_tilde("~"), home);
    }

    #[test]
    fn test_expand_tilde_subdir() {
        let home = dirs::home_dir().unwrap();
        assert_eq!(expand_tilde("~/.piper-voices"), home.join(".piper-voices"));
    }

    #[test]
    fn test_expand_tilde_absolute_unchanged() {
        assert_eq!(expand_tilde("/usr/share/voices"), PathBuf::from("/usr/share/voices"));
    }

    #[test]
    fn test_expand_tilde_relative_unchanged() {
        assert_eq!(expand_tilde("relative/path"), PathBuf::from("relative/path"));
    }


    #[test]
    fn test_is_voice_downloaded_default_dir_not_present() {
        let _ = is_voice_downloaded("en-us-lessac-medium", "");
    }

    #[test]
    fn test_is_voice_downloaded_returns_true_when_files_exist() {
        let dir = tempdir().unwrap();
        let path = dir.path().to_str().unwrap();
        create_fake_voice(dir.path(), "en-us-amy-low.onnx");
        assert!(is_voice_downloaded("en-us-amy-low", path));
    }

    #[test]
    fn test_is_voice_downloaded_returns_false_when_files_missing() {
        let dir = tempdir().unwrap();
        let path = dir.path().to_str().unwrap();
        assert!(!is_voice_downloaded("en-us-amy-low", path));
    }

    #[test]
    fn test_is_voice_downloaded_returns_false_for_nonexistent_dir() {
        assert!(!is_voice_downloaded("en-us-amy-low", "/nonexistent/path/xyz"));
    }

    #[test]
    fn test_is_voice_downloaded_tilde_path() {
        let _ = is_voice_downloaded("en-us-lessac-medium", "~/.local/share/fotonvoice-engine/piper-voices");
    }

    #[test]
    fn test_is_voice_downloaded_only_onnx_not_sufficient() {
        let dir = tempdir().unwrap();
        let path = dir.path().to_str().unwrap();
        fs::write(dir.path().join("en_US-amy-low.onnx"), b"fake").unwrap();
        assert!(!is_voice_downloaded("en-us-amy-low", path));
    }


    #[test]
    fn test_list_local_voices_finds_onnx_json_pairs() {
        let dir = tempdir().unwrap();
        let path = dir.path().to_str().unwrap();
        create_fake_voice(dir.path(), "zz-custom-high.onnx");
        create_fake_voice(dir.path(), "aa-custom-low.onnx");
        fs::write(dir.path().join("broken.onnx"), b"fake").unwrap();
        fs::write(dir.path().join("notes.txt"), b"x").unwrap();
        assert_eq!(list_local_voices(path), vec!["aa-custom-low", "zz-custom-high"]);
    }

    #[test]
    fn test_list_local_voices_empty_dir() {
        let dir = tempdir().unwrap();
        let path = dir.path().to_str().unwrap();
        assert!(list_local_voices(path).is_empty());
    }

    #[test]
    fn test_sample_rate_falls_back_to_voice_json() {
        let dir = tempdir().unwrap();
        create_fake_voice(dir.path(), "zz-custom-high.onnx");
        fs::write(
            dir.path().join("zz-custom-high.onnx.json"),
            r#"{"audio":{"sample_rate":48000}}"#,
        )
        .unwrap();
        let voice_path = dir.path().join("zz-custom-high.onnx");
        assert_eq!(sample_rate_for_voice_path("zz-custom-high", &voice_path), 48000);
        assert_eq!(
            sample_rate_for_voice_path("en_US-carlin-high", &dir.path().join("en_US-carlin-high.onnx")),
            22050
        );
    }


    #[test]
    fn test_get_voice_path_returns_none_when_missing() {
        let dir = tempdir().unwrap();
        let path = dir.path().to_str().unwrap();
        assert!(get_voice_path("en-us-ryan-high", path).is_none());
    }

    #[test]
    fn test_get_voice_path_returns_some_when_present() {
        let dir = tempdir().unwrap();
        let path = dir.path().to_str().unwrap();
        create_fake_voice(dir.path(), "en_US-carlin-high.onnx");
        let result = get_voice_path("en_US-carlin-high", path);
        assert!(result.is_some());
        assert!(result.unwrap().exists());
    }

    #[test]
    fn test_get_voice_path_accepts_custom_dir() {
        let dir = tempdir().unwrap();
        let other_dir = tempdir().unwrap();
        let path = dir.path().to_str().unwrap();
        let other_path = other_dir.path().to_str().unwrap();
        create_fake_voice(other_dir.path(), "en_US-carlin-high.onnx");
        assert!(get_voice_path("en_US-carlin-high", path).is_none());
        assert!(get_voice_path("en_US-carlin-high", other_path).is_some());
    }

    #[test]
    fn test_get_voice_path_lowercase_fallback() {
        let dir = tempdir().unwrap();
        let path = dir.path().to_str().unwrap();
        let lc_name = "de_de-thorsten-medium.onnx";
        fs::write(dir.path().join(lc_name), b"fake").unwrap();
        fs::write(dir.path().join(format!("{lc_name}.json")), b"{}").unwrap();
        assert!(get_voice_path("de_DE-thorsten-medium", path).is_some());
    }


    #[test]
    fn test_piper_voices_not_empty() {
        assert!(!PIPER_VOICES.is_empty());
    }

    #[test]
    fn test_piper_voices_have_required_fields() {
        for v in PIPER_VOICES {
            assert!(!v.name.is_empty());
            assert!(!v.quality.is_empty());
            assert!(!v.filename.is_empty());
            assert!(v.sample_rate > 0);
        }
    }

    #[test]
    fn test_piper_voices_names_unique() {
        let mut seen = std::collections::HashSet::new();
        for v in PIPER_VOICES {
            assert!(seen.insert(v.name), "duplicate piper voice name: {}", v.name);
        }
    }

    #[test]
    fn test_piper_voices_filenames_unique() {
        let mut seen = std::collections::HashSet::new();
        for v in PIPER_VOICES {
            assert!(seen.insert(v.filename), "duplicate piper filename: {}", v.filename);
        }
    }

    #[test]
    fn test_piper_voices_quality_values_are_valid() {
        let valid = ["high", "medium", "low"];
        for v in PIPER_VOICES {
            assert!(valid.contains(&v.quality), "unexpected quality '{}' for {}", v.quality, v.name);
        }
    }

    #[test]
    fn test_piper_voices_sample_rates_are_valid() {
        let valid_rates = [16000u32, 22050u32, 44100u32];
        for v in PIPER_VOICES {
            assert!(valid_rates.contains(&v.sample_rate), "unexpected sample_rate {} for {}", v.sample_rate, v.name);
        }
    }

    #[test]
    fn test_piper_voices_filenames_end_with_onnx() {
        for v in PIPER_VOICES {
            assert!(v.filename.ends_with(".onnx"), "filename should end with .onnx: {}", v.filename);
        }
    }


    #[test]
    fn test_sample_rate_for_known_high_quality_voice() {
        let p = PathBuf::from("en-us-ryan-high.onnx");
        assert_eq!(sample_rate_for_voice_path("en-us-ryan-high", &p), 22050);
    }

    #[test]
    fn test_sample_rate_for_known_low_quality_voice() {
        let p = PathBuf::from("en-us-lessac-medium.onnx");
        assert_eq!(sample_rate_for_voice_path("en-us-lessac-medium", &p), 16000);
    }

    #[test]
    fn test_sample_rate_for_unknown_voice_defaults_to_22050() {
        let p = PathBuf::from("xx-unknown-voice.onnx");
        assert_eq!(sample_rate_for_voice_path("xx-unknown-voice", &p), 22050);
    }


    #[test]
    fn test_piper_binary_returns_option_without_panicking() {
        let _ = piper_binary();
    }


    #[cfg(unix)]
    fn build_fake_piper_tarball() -> Vec<u8> {
        use flate2::write::GzEncoder;
        use flate2::Compression;

        let gz = GzEncoder::new(Vec::new(), Compression::fast());
        let mut builder = tar::Builder::new(gz);

        let mut add_file = |path: &str, content: &[u8], mode: u32| {
            let mut header = tar::Header::new_gnu();
            header.set_size(content.len() as u64);
            header.set_mode(mode);
            header.set_cksum();
            builder.append_data(&mut header, path, content).unwrap();
        };

        add_file("piper/piper", b"#!/bin/sh\necho fake piper\n", 0o755);
        add_file("piper/libespeak-ng.so.1", b"fake lib", 0o644);
        add_file("piper/espeak-ng-data/phondata", b"fake phondata", 0o644);
        add_file("piper/espeak-ng-data/lang/gmw/en-US", b"fake lang", 0o644);

        builder.into_inner().unwrap().finish().unwrap()
    }

    #[cfg(unix)]
    #[test]
    fn test_extract_piper_archive_preserves_directory_structure() {
        let dir = tempdir().unwrap();
        let bytes = build_fake_piper_tarball();

        extract_piper_archive(&bytes, dir.path()).unwrap();

        assert!(dir.path().join("piper").is_file());
        assert!(dir.path().join("libespeak-ng.so.1").is_file());
        assert!(dir.path().join("espeak-ng-data").is_dir());
        assert!(dir.path().join("espeak-ng-data/phondata").is_file());
        assert!(dir.path().join("espeak-ng-data/lang/gmw/en-US").is_file());

        use std::os::unix::fs::PermissionsExt;
        let mode = fs::metadata(dir.path().join("piper")).unwrap().permissions().mode();
        assert_eq!(mode & 0o111, 0o111, "piper binary must be executable");
    }

    #[cfg(unix)]
    #[test]
    fn test_extract_piper_archive_skips_unsafe_paths() {
        use flate2::write::GzEncoder;
        use flate2::Compression;

        let gz = GzEncoder::new(Vec::new(), Compression::fast());
        let mut builder = tar::Builder::new(gz);
        let content: &[u8] = b"evil";
        let mut header = tar::Header::new_gnu();
        {
            let name = b"piper/../../escape.txt";
            let gnu = header.as_gnu_mut().unwrap();
            gnu.name[..name.len()].copy_from_slice(name);
        }
        header.set_size(content.len() as u64);
        header.set_mode(0o644);
        header.set_cksum();
        builder.append(&header, content).unwrap();
        let bytes = builder.into_inner().unwrap().finish().unwrap();

        let dir = tempdir().unwrap();
        extract_piper_archive(&bytes, dir.path()).unwrap();

        assert!(!dir.path().parent().unwrap().join("escape.txt").exists());
        assert!(!dir.path().parent().unwrap().parent().unwrap().join("escape.txt").exists());
        assert!(dir.path().join("escape.txt").exists());
    }


    #[test]
    fn test_piper_voices_dir_not_empty() {
        let d = piper_voices_dir();
        assert!(d.components().count() > 0);
    }

    #[test]
    fn test_piper_voices_dir_ends_with_piper_voices() {
        let d = piper_voices_dir();
        assert!(d.ends_with("piper-voices"));
    }


    #[test]
    fn test_voice_name_to_filename_known() {
        assert_eq!(
            voice_name_to_filename("en-us-lessac-medium"),
            Some("en-us-lessac-medium.onnx".to_string())
        );
    }

    #[test]
    fn test_voice_name_to_filename_unknown_returns_none() {
        assert_eq!(voice_name_to_filename("xx-unknown-voice"), None);
    }

    #[test]
    fn test_voice_name_to_filename_all_piper_voices_resolve() {
        for v in PIPER_VOICES {
            let result = voice_name_to_filename(v.name);
            assert!(result.is_some(), "voice_name_to_filename should resolve {}", v.name);
            assert_eq!(result.unwrap(), v.filename);
        }
    }
}
