//! Shared runtime for every audio.cpp-backed engine (Pocket-TTS, Breeze-TTS-2,

use std::io::Read as _;
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Child, Stdio};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use serde::Serialize;
use tracing::info;

use crate::piper::expand_tilde;

/// The audio.cpp release FotonVoice Engine downloads. Bump alongside a verified test of
pub const AUDIOCPP_RELEASE_VERSION: &str = "v0.8.0";

/// GGUF model family names audio.cpp registers for the engines FotonVoice Engine offers.
pub const FAMILY_POCKET_TTS: &str = "pocket_tts";
pub const FAMILY_BREEZE_TTS: &str = "breeze_tts";
pub const FAMILY_VOXCPM2: &str = "voxcpm2";

pub fn audiocpp_dir() -> PathBuf {
    fotonvoice_config::portable::app_root()
        .join("audiocpp")
}

/// Where FotonVoice Engine caches downloaded reference-voice clips (the `hf://` built-in
fn voice_clip_cache_dir() -> PathBuf {
    audiocpp_dir().join("voice-clips")
}

pub(crate) fn resolve_model_dir(model_dir: &str, default: impl FnOnce() -> PathBuf) -> PathBuf {
    if model_dir.trim().is_empty() {
        default()
    } else {
        expand_tilde(model_dir.trim())
    }
}

/// Resolves the `audiocpp_cli` binary: FotonVoice Engine's own managed install first,
pub fn audiocpp_binary() -> Option<PathBuf> {
    let exe = if cfg!(target_os = "windows") { "audiocpp_cli.exe" } else { "audiocpp_cli" };
    let local = audiocpp_dir().join(exe);
    if local.exists() {
        return Some(local);
    }
    fotonvoice_config::find_in_path("audiocpp_cli")
}

/// Resolves the `audiocpp_server` binary the same way as [`audiocpp_binary`].
fn audiocpp_server_binary() -> Option<PathBuf> {
    let exe = if cfg!(target_os = "windows") { "audiocpp_server.exe" } else { "audiocpp_server" };
    let local = audiocpp_dir().join(exe);
    if local.exists() {
        return Some(local);
    }
    fotonvoice_config::find_in_path("audiocpp_server")
}


/// Fetches and unpacks the prebuilt audio.cpp release archive for the current
#[cfg(target_os = "linux")]
pub async fn download_audiocpp_binary() -> Result<()> {
    let dest_dir = audiocpp_dir();
    let dest_exe = dest_dir.join("audiocpp_cli");
    if dest_exe.exists() {
        return Ok(());
    }
    tokio::fs::create_dir_all(&dest_dir).await?;

    let asset = format!("audio-{AUDIOCPP_RELEASE_VERSION}-bin-ubuntu-x64-vulkan.tar.gz");
    let url = format!(
        "https://github.com/0xShug0/audio.cpp/releases/download/{AUDIOCPP_RELEASE_VERSION}/{asset}"
    );
    info!("Downloading audio.cpp runtime: {url}");

    let response = reqwest::get(&url).await?.error_for_status()?;
    let bytes = response.bytes().await?;

    tokio::task::spawn_blocking(move || extract_audiocpp_archive(&bytes, &dest_dir))
        .await
        .context("extract_audiocpp_archive task join")??;

    info!("audio.cpp runtime installed to {}", dest_exe.display());
    Ok(())
}

/// FotonVoice Engine has no verified Windows/macOS release asset name or install path
#[cfg(not(target_os = "linux"))]
pub async fn download_audiocpp_binary() -> Result<()> {
    anyhow::bail!(
        "FotonVoice Engine cannot install audio.cpp automatically on this platform yet. Download a \
         release from https://github.com/0xShug0/audio.cpp/releases and place \
         audiocpp_cli{} in {}, or put it on PATH.",
        if cfg!(target_os = "windows") { ".exe" } else { "" },
        audiocpp_dir().display()
    );
}

#[cfg(target_os = "linux")]
fn extract_audiocpp_archive(bytes: &[u8], dest_dir: &Path) -> Result<()> {
    let cursor = std::io::Cursor::new(bytes);
    let tar = flate2::read::GzDecoder::new(cursor);
    let mut archive = tar::Archive::new(tar);

    for entry in archive.entries()? {
        let mut entry = entry?;
        let path = entry.path()?.into_owned();

        if path.components().next().map(|c| c.as_os_str() == "tools").unwrap_or(false) {
            continue;
        }
        if path.extension().and_then(|e| e.to_str()) == Some("py") {
            continue;
        }

        let rel: PathBuf = path
            .components()
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
    for exe in ["audiocpp_cli", "audiocpp_server"] {
        let path = dest_dir.join(exe);
        if let Ok(metadata) = std::fs::metadata(&path) {
            let mut perms = metadata.permissions();
            perms.set_mode(0o755);
            let _ = std::fs::set_permissions(&path, perms);
        }
    }

    Ok(())
}


/// Splits an `hf://owner/repo/path[@revision]` reference into its parts and
fn hf_reference_parts(reference: &str) -> Result<(String, String, String, PathBuf)> {
    let rest = reference
        .strip_prefix("hf://")
        .ok_or_else(|| anyhow::anyhow!("not an hf:// reference: {reference}"))?;
    let mut parts = rest.splitn(3, '/');
    let (owner, repo, filename_rev) = match (parts.next(), parts.next(), parts.next()) {
        (Some(o), Some(r), Some(f)) => (o, r, f),
        _ => anyhow::bail!("malformed hf:// reference: {reference}"),
    };
    let (filename, revision) = match filename_rev.rfind('@') {
        Some(at) => (&filename_rev[..at], &filename_rev[at + 1..]),
        None => (filename_rev, "main"),
    };
    let dest = voice_clip_cache_dir().join(owner).join(repo).join(filename);
    let url = format!("https://huggingface.co/{owner}/{repo}/resolve/{revision}/{filename}");
    Ok((format!("{owner}/{repo}"), url, filename.to_string(), dest))
}

/// Downloads an `hf://owner/repo/path[@revision]` reference into FotonVoice Engine's own
pub async fn resolve_hf_reference(reference: &str, hf_token: Option<&str>) -> Result<PathBuf> {
    let Ok((repo, url, _filename, dest)) = hf_reference_parts(reference) else {
        return Ok(PathBuf::from(reference));
    };
    if dest.exists() {
        return Ok(dest);
    }
    if let Some(parent) = dest.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(300))
        .build()
        .context("build reqwest client")?;
    let mut request = client.get(&url);
    if let Some(token) = crate::hf::effective_hf_token(hf_token) {
        request = request.bearer_auth(token);
    }

    let resp = request.send().await.with_context(|| format!("fetch {url}"))?;
    let status = resp.status();
    if status == reqwest::StatusCode::UNAUTHORIZED || status == reqwest::StatusCode::FORBIDDEN {
        return Err(crate::hf::token_rejected(&repo));
    }
    if !status.is_success() {
        anyhow::bail!("fetch {url}: HTTP {status}");
    }
    let bytes = resp.bytes().await.with_context(|| format!("read {url}"))?;
    tokio::fs::write(&dest, &bytes).await.with_context(|| format!("write {}", dest.display()))?;
    Ok(dest)
}

/// Blocking counterpart of [`resolve_hf_reference`], for the synchronous TTS
pub(crate) fn resolve_hf_reference_blocking(reference: &str, hf_token: Option<&str>) -> Result<PathBuf> {
    let Ok((repo, url, _filename, dest)) = hf_reference_parts(reference) else {
        return Ok(PathBuf::from(reference));
    };
    if dest.exists() {
        return Ok(dest);
    }
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(300))
        .build()
        .context("build reqwest client")?;
    let mut request = client.get(&url);
    if let Some(token) = crate::hf::effective_hf_token(hf_token) {
        request = request.bearer_auth(token);
    }

    let resp = request.send().with_context(|| format!("fetch {url}"))?;
    let status = resp.status();
    if status == reqwest::StatusCode::UNAUTHORIZED || status == reqwest::StatusCode::FORBIDDEN {
        return Err(crate::hf::token_rejected(&repo));
    }
    if !status.is_success() {
        anyhow::bail!("fetch {url}: HTTP {status}");
    }
    let bytes = resp.bytes().with_context(|| format!("read {url}"))?;
    std::fs::write(&dest, &bytes).with_context(|| format!("write {}", dest.display()))?;
    Ok(dest)
}


/// Either a reference clip to clone (`voice_ref`), a built-in voice id
pub enum SpeakerRef<'a> {
    Clone(&'a Path),
    VoiceId(&'a str),
    Design(&'a str),
}

pub struct SpeakRequest<'a> {
    pub text: &'a str,
    pub speaker: Option<SpeakerRef<'a>>,
    /// Reference transcript for a `voice_ref` clip. `breeze_tts` requires
    pub reference_text: Option<&'a str>,
}

/// A running `audiocpp_server` process bound to one model (family + model
pub(crate) struct AudioCppServer {
    child: Child,
    base_url: String,
    http: reqwest::blocking::Client,
    family: &'static str,
    model_dir: PathBuf,
    gpu: bool,
    _config_file: tempfile::NamedTempFile,
}

impl Drop for AudioCppServer {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn free_local_port() -> Result<u16> {
    let listener = TcpListener::bind(("127.0.0.1", 0)).context("bind ephemeral port")?;
    Ok(listener.local_addr()?.port())
}

/// Makes the spawned child ask the kernel to SIGTERM it the instant *this*
#[cfg(target_os = "linux")]
fn die_with_this_process(cmd: &mut std::process::Command) {
    use std::os::unix::process::CommandExt;
    unsafe {
        cmd.pre_exec(|| {
            if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGTERM) != 0 {
                return Err(std::io::Error::last_os_error());
            }
            Ok(())
        });
    }
}

impl AudioCppServer {
    /// Spawns `audiocpp_server` configured with exactly one model - the one
    fn spawn(family: &'static str, model_dir: &Path, gpu: bool) -> Result<Self> {
        let binary = audiocpp_server_binary().ok_or_else(|| {
            anyhow::anyhow!(
                "audio.cpp server binary not found. Download it from TTS settings, or install \
                 audiocpp_server system-wide."
            )
        })?;

        let port = free_local_port().context("find a free port for audiocpp_server")?;
        let config = serde_json::json!({
            "host": "127.0.0.1",
            "port": port,
            "backend": if gpu { "vulkan" } else { "cpu" },
            "lazy_load": true,
            "models": [{
                "id": family,
                "family": family,
                "path": model_dir,
                "task": "tts",
                "mode": "offline",
            }],
        });
        let mut config_file = tempfile::Builder::new()
            .prefix("fotonvoice-audiocpp-server-")
            .suffix(".json")
            .tempfile()
            .context("create audiocpp_server config file")?;
        {
            use std::io::Write;
            config_file
                .write_all(serde_json::to_string_pretty(&config)?.as_bytes())
                .context("write audiocpp_server config file")?;
            config_file.flush()?;
        }

        let mut cmd = std::process::Command::new(&binary);
        cmd.arg("--config")
            .arg(config_file.path())
            .arg("--no-ui")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::piped());
        #[cfg(target_os = "linux")]
        die_with_this_process(&mut cmd);

        let mut child = cmd.spawn().with_context(|| format!("spawn {}", binary.display()))?;

        let base_url = format!("http://127.0.0.1:{port}");
        let http = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(120))
            .build()
            .context("build reqwest client")?;

        let deadline = Instant::now() + Duration::from_secs(15);
        loop {
            if let Ok(resp) = http.get(format!("{base_url}/health")).send() {
                if resp.status().is_success() {
                    break;
                }
            }
            if let Ok(Some(status)) = child.try_wait() {
                anyhow::bail!(
                    "audiocpp_server exited during startup ({status:?}): {}",
                    read_stderr(&mut child)
                );
            }
            if Instant::now() > deadline {
                let _ = child.kill();
                anyhow::bail!("audiocpp_server did not become ready in time: {}", read_stderr(&mut child));
            }
            std::thread::sleep(Duration::from_millis(150));
        }

        Ok(Self { child, base_url, http, family, model_dir: model_dir.to_path_buf(), gpu, _config_file: config_file })
    }

    fn speak(&self, req: &SpeakRequest) -> Result<Vec<u8>> {
        #[derive(Serialize)]
        struct RequestOptions<'a> {
            instruction: &'a str,
        }
        #[derive(Serialize)]
        struct Body<'a> {
            model: &'static str,
            input: &'a str,
            #[serde(skip_serializing_if = "Option::is_none")]
            voice: Option<&'a str>,
            #[serde(skip_serializing_if = "Option::is_none")]
            voice_ref: Option<&'a Path>,
            #[serde(skip_serializing_if = "Option::is_none")]
            instruct: Option<&'a str>,
            #[serde(skip_serializing_if = "Option::is_none")]
            request_options: Option<RequestOptions<'a>>,
            #[serde(skip_serializing_if = "Option::is_none")]
            reference_text: Option<&'a str>,
        }

        let mut body = Body {
            model: self.family,
            input: req.text,
            voice: None,
            voice_ref: None,
            instruct: None,
            request_options: None,
            reference_text: req.reference_text,
        };
        match req.speaker {
            Some(SpeakerRef::Clone(clip)) => body.voice_ref = Some(clip),
            Some(SpeakerRef::VoiceId(id)) => body.voice = Some(id),
            Some(SpeakerRef::Design(prompt)) => {
                if self.family == FAMILY_BREEZE_TTS {
                    body.request_options = Some(RequestOptions { instruction: prompt });
                } else {
                    body.instruct = Some(prompt);
                }
            }
            None => {}
        }

        let resp = self
            .http
            .post(format!("{}/v1/audio/speech", self.base_url))
            .json(&body)
            .send()
            .context("POST /v1/audio/speech")?;
        let status = resp.status();
        let bytes = resp.bytes().context("read /v1/audio/speech response")?;
        if !status.is_success() {
            let message = serde_json::from_slice::<serde_json::Value>(&bytes)
                .ok()
                .and_then(|v| v.get("error")?.get("message")?.as_str().map(str::to_string))
                .unwrap_or_else(|| String::from_utf8_lossy(&bytes).trim().to_string());
            anyhow::bail!("audiocpp_server (family={}) request failed: {message}", self.family);
        }
        Ok(bytes.to_vec())
    }
}

fn read_stderr(child: &mut Child) -> String {
    let mut stderr = String::new();
    if let Some(mut s) = child.stderr.take() {
        let _ = s.read_to_string(&mut stderr);
    }
    stderr.trim().to_string()
}

/// A resident audio.cpp model session, reused across utterances for as long
pub(crate) struct AudioCppSession {
    server: AudioCppServer,
}

impl AudioCppSession {
    /// Ensures `slot` holds a session for `family`/`model_dir`/`gpu`,
    pub(crate) fn ensure(
        slot: &mut Option<AudioCppSession>,
        family: &'static str,
        model_dir: &Path,
        gpu: bool,
    ) -> Result<()> {
        let matches = slot.as_ref().is_some_and(|s| {
            s.server.family == family && s.server.model_dir == model_dir && s.server.gpu == gpu
        });
        if matches {
            return Ok(());
        }
        *slot = None; // drop first: frees the old port before the new server binds one
        info!("Starting audio.cpp server session (family={family}, gpu={gpu})");
        *slot = Some(AudioCppSession { server: AudioCppServer::spawn(family, model_dir, gpu)? });
        Ok(())
    }

    pub(crate) fn speak(&self, req: &SpeakRequest) -> Result<Vec<u8>> {
        self.server.speak(req)
    }
}

/// Decodes WAV bytes audio.cpp's server returned and queues them onto
pub fn play_wav_bytes(sink: &rodio::Sink, bytes: Vec<u8>) -> Result<()> {
    let source = rodio::Decoder::new(std::io::Cursor::new(bytes)).context("decode audio.cpp response")?;
    sink.append(source);
    sink.sleep_until_end();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_hf_reference_parts_parses_owner_repo_file() {
        let (repo, url, filename, _dest) =
            hf_reference_parts("hf://kyutai/tts-voices/alba-mackenna/casual.wav").unwrap();
        assert_eq!(repo, "kyutai/tts-voices");
        assert_eq!(filename, "alba-mackenna/casual.wav");
        assert!(url.starts_with("https://huggingface.co/kyutai/tts-voices/resolve/main/"));
    }

    #[test]
    fn test_hf_reference_parts_rejects_non_hf_uri() {
        assert!(hf_reference_parts("/plain/local/path.wav").is_err());
    }

    #[test]
    fn test_resolve_model_dir_empty_uses_default() {
        let dir = tempdir().unwrap();
        let default_dir = dir.path().to_path_buf();
        let result = resolve_model_dir("", || default_dir.clone());
        assert_eq!(result, default_dir);
    }

    #[test]
    fn test_resolve_model_dir_explicit_expands_tilde() {
        let home = dirs::home_dir().unwrap();
        let result = resolve_model_dir("~/my-model", || PathBuf::from("unused"));
        assert_eq!(result, home.join("my-model"));
    }

    #[test]
    fn test_audiocpp_binary_returns_option_without_panicking() {
        let _ = audiocpp_binary();
    }
}
