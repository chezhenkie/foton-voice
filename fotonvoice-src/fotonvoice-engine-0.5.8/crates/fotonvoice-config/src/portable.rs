//! Portable root: the whole app file tree lives beside the exe.
//!
//! Every path the app reads or writes resolves under `app_root()`, the folder
//! containing the running executable. Earlier builds scattered files across
//! the platform data directories; `ensure_migrated()` copies whatever it finds
//! there into the portable root exactly once, then never looks back.

use std::path::{Path, PathBuf};

/// The folder holding the running exe. Every app file lives under it.
///
/// Windows only, and deliberately so: the portable install (a folder copied
/// between machines) is a Windows concept, and there the exe folder is
/// writable. On Linux the exe lives somewhere read-only or root-owned - an
/// AppImage's squashfs mount, or /usr/bin for a .deb - so the portable root
/// would be unwritable and every config/model/log write would fail (observed:
/// startup error log, overlays, and model lookup all failing against
/// `/tmp/.mount_*/usr/bin`). There the tree lives in the per-user XDG data
/// directory instead; `ensure_migrated` still runs once and copies any legacy
/// per-user content into it, and its marker file is writable here.
#[cfg(target_os = "windows")]
pub fn app_root() -> PathBuf {
    std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(Path::to_path_buf))
        .unwrap_or_else(|| PathBuf::from("."))
}

/// Non-Windows variant - see the Windows doc comment for why these differ.
#[cfg(not(target_os = "windows"))]
pub fn app_root() -> PathBuf {
    dirs::data_local_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("fotonvoice-engine")
}

/// Legacy per-user roaming config directory, read only during migration.
fn legacy_roaming() -> PathBuf {
    dirs::config_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("fotonvoice-engine")
}

/// Legacy per-user local data directory, read only during migration.
fn legacy_local() -> PathBuf {
    dirs::data_local_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("fotonvoice-engine")
}

/// Copy `src` into `dst` when `dst` is missing and `src` exists.
///
/// `true` means nothing more to do here: copied, already present, or nothing
/// to copy. `false` means the copy failed and a later run should retry.
fn migrate_file(dst: &Path, src: &Path) -> bool {
    if dst.exists() || !src.is_file() {
        return true;
    }
    match std::fs::copy(src, dst) {
        Ok(_) => {
            tracing::info!("Migrated {} into portable root", dst.display());
            true
        }
        Err(e) => {
            tracing::error!("Failed to migrate {} to {}: {e}", src.display(), dst.display());
            false
        }
    }
}

/// Recursively copy a directory tree.
fn copy_dir(src: &Path, dst: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(dst)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let to = dst.join(entry.file_name());
        if entry.file_type()?.is_dir() {
            copy_dir(&entry.path(), &to)?;
        } else {
            std::fs::copy(entry.path(), &to)?;
        }
    }
    Ok(())
}

/// Copy `src` into `dst` when `dst` is missing and `src` exists.
///
/// The copy lands at `<dst>.migrating` first and is renamed into place only
/// once complete, so a failed copy leaves no half-populated `dst` that would
/// make later runs skip the retry.
fn migrate_dir(dst: &Path, src: &Path) -> bool {
    if dst.exists() || !src.is_dir() {
        return true;
    }
    let staging = dst.with_extension("migrating");
    let _ = std::fs::remove_dir_all(&staging);
    match copy_dir(src, &staging).and_then(|()| {
        if dst.exists() {
            std::fs::remove_dir_all(&staging)
        } else {
            std::fs::rename(&staging, dst)
        }
    }) {
        Ok(()) => {
            tracing::info!("Migrated {} into portable root", dst.display());
            true
        }
        Err(e) => {
            let _ = std::fs::remove_dir_all(&staging);
            tracing::error!("Failed to migrate {} to {}: {e}", src.display(), dst.display());
            false
        }
    }
}

/// One-time migration of legacy AppData content into the portable root.
///
/// Guarded by a `.portable-migrated` marker in the root: once it is written
/// the app never reads the legacy directories again, so deleting a file from
/// the root stays deleted. Migration is copy, not move - the legacy copy is
/// left untouched for the user to remove once everything checks out.
pub fn ensure_migrated() {
    let root = app_root();
    let marker = root.join(".portable-migrated");
    if marker.exists() {
        return;
    }

    let roaming = legacy_roaming();
    let local = legacy_local();
    let ok = migrate_file(&root.join("config.json"), &roaming.join("config.json"))
        & migrate_file(&root.join("bindings.toml"), &roaming.join("bindings.toml"))
        & migrate_file(&root.join("targets.toml"), &roaming.join("targets.toml"))
        & migrate_dir(&root.join("backups"), &roaming.join("backups"))
        & migrate_dir(&root.join("models"), &local.join("models"))
        & migrate_dir(&root.join("piper-voices"), &local.join("piper-voices"))
        & migrate_dir(&root.join("piper"), &local.join("piper"))
        & migrate_dir(&root.join("overlays"), &local.join("overlays"))
        & migrate_dir(&root.join("cloned-tts-voices"), &local.join("cloned-tts-voices"))
        & migrate_dir(&root.join("config"), &local.join("config"))
        & migrate_file(&root.join("startup_errors.log"), &local.join("startup_errors.log"))
        & migrate_file(&root.join("bug-reports.json"), &local.join("bug-reports.json"));

    if ok {
        if let Err(e) = std::fs::write(&marker, "FotonVoice Engine portable root: AppData migration done\n") {
            tracing::warn!("Could not write portable-root marker: {e}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn migrate_file_copies_only_when_dst_missing() {
        let dir = std::env::temp_dir().join(format!("vc-portable-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let src = dir.join("src.json");
        std::fs::write(&src, "{}").unwrap();
        let dst = dir.join("dst.json");
        assert!(!dst.exists());
        assert!(migrate_file(&dst, &src));
        assert!(dst.exists());
        assert_eq!(std::fs::read_to_string(&dst).unwrap(), "{}");
        // Present dst is never overwritten, missing src is a no-op.
        std::fs::write(&dst, "kept").unwrap();
        assert!(migrate_file(&dst, &src));
        assert_eq!(std::fs::read_to_string(&dst).unwrap(), "kept");
        assert!(migrate_file(&dir.join("other.json"), &dir.join("no-such-file")));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn migrate_dir_copies_trees_and_leaves_no_staging() {
        let dir = std::env::temp_dir().join(format!("vc-portable-d{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let src = dir.join("models");
        std::fs::create_dir_all(&src.join("sub")).unwrap();
        std::fs::write(src.join("sub").join("a.bin"), "x").unwrap();
        let dst = dir.join("out").join("models");
        assert!(migrate_dir(&dst, &src));
        assert!(dst.join("sub").join("a.bin").exists());
        // Staging copy is renamed away once complete.
        assert!(!dst.with_extension("migrating").exists());
        // Existing dst is never touched.
        std::fs::write(dst.join("sub").join("a.bin"), "kept").unwrap();
        assert!(migrate_dir(&dst, &src));
        assert_eq!(std::fs::read_to_string(dst.join("sub").join("a.bin")).unwrap(), "kept");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
