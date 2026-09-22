//! LuxTTS text frontend: eSpeak-NG phonemization and the token vocabulary.

use std::collections::HashMap;
use std::path::Path;
use std::process::Command;

use anyhow::{Context, Result};
use tracing::warn;

/// eSpeak-NG voice the reference phonemizes with.
pub const ESPEAK_VOICE: &str = "en-us";

/// The reserved token ids the text encoder expects around a sequence.
#[derive(Debug, Clone)]
pub struct Reserved {
    pub pad: i64,
    pub start: i64,
    pub end: i64,
}

/// Symbol -> id table parsed from `tokens.txt` (`<symbol>\t<id>` per line).
pub struct TokenVocab {
    token2id: HashMap<String, i64>,
    pub reserved: Reserved,
}

impl TokenVocab {
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("read token vocabulary {}", path.display()))?;
        let mut token2id = HashMap::new();
        for line in text.lines() {
            let Some((symbol, id)) = line.rsplit_once('\t') else {
                continue;
            };
            if let Ok(id) = id.trim().parse::<i64>() {
                token2id.insert(symbol.to_string(), id);
            }
        }
        let reserved = Reserved {
            pad: *token2id.get("_").context("tokens.txt lacks the `_` pad token")?,
            start: *token2id.get("^").context("tokens.txt lacks the `^` start token")?,
            end: *token2id.get("$").context("tokens.txt lacks the `$` end token")?,
        };
        Ok(Self { token2id, reserved })
    }

    pub fn len(&self) -> usize {
        self.token2id.len()
    }

    pub fn is_empty(&self) -> bool {
        self.token2id.is_empty()
    }

    /// Map phoneme symbols to ids, skipping unknown ones. Returns the ids and
    pub fn encode(&self, symbols: &[String]) -> (Vec<i64>, Vec<String>) {
        let mut ids = Vec::with_capacity(symbols.len());
        let mut skipped = Vec::new();
        for s in symbols {
            match self.token2id.get(s) {
                Some(id) => ids.push(*id),
                None => skipped.push(s.clone()),
            }
        }
        (ids, skipped)
    }
}

/// Replaces fullwidth/CJK punctuation and dot-leaders with their ASCII forms,
pub fn map_punctuations(text: &str) -> String {
    text.replace('，', ",")
        .replace('。', ".")
        .replace('！', "!")
        .replace('？', "?")
        .replace('；', ";")
        .replace('：', ":")
        .replace('、', ",")
        .replace('‘', "'")
        .replace('“', "\"")
        .replace('”', "\"")
        .replace('’', "'")
        .replace('⋯', "…")
        .replace("···", "…")
        .replace("・・・", "…")
        .replace("...", "…")
}

/// Split `text` into clauses on terminator punctuation, keeping the terminator.
pub struct Clause {
    pub text: String,
    pub terminator: String,
}

pub fn split_clauses(text: &str) -> Vec<Clause> {
    let mut clauses = Vec::new();
    let mut current = String::new();
    for ch in text.chars() {
        current.push(ch);
        if ch == '.' || ch == '!' || ch == '?' || ch == '…' {
            clauses.push(Clause {
                text: current.trim().to_string(),
                terminator: ch.to_string(),
            });
            current = String::new();
        }
    }
    if !current.trim().is_empty() {
        clauses.push(Clause {
            text: current.trim().to_string(),
            terminator: String::new(),
        });
    }
    clauses
}

/// True when the `espeak-ng` binary is callable.
pub fn espeak_available() -> bool {
    fotonvoice_config::find_in_path("espeak-ng").is_some()
}

/// The espeak-ng command with a portable data-dir override.
fn espeak_command() -> Command {
    let mut command = Command::new("espeak-ng");
    #[cfg(target_os = "windows")]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        command.creation_flags(CREATE_NO_WINDOW);
    }
    if let Some(exe) = fotonvoice_config::find_in_path("espeak-ng") {
        if let Some(dir) = exe.parent() {
            let data = dir.join("espeak-ng-data");
            if data.is_dir() {
                if let Some(parent) = data.parent() {
                    command.env("ESPEAK_DATA_PATH", parent);
                }
            }
        }
    }
    command
}

/// Phonemize one clause with eSpeak-NG (`en-us`, `--ipa`), returning bare IPA.
fn phonemize_clause(text: &str) -> Result<String> {
    let output = espeak_command()
        .arg("-v")
        .arg(ESPEAK_VOICE)
        .arg("--ipa")
        .arg("-q")
        .arg("--")
        .arg(text)
        .output()
        .context("spawn espeak-ng for phonemization")?;
    if !output.status.success() {
        anyhow::bail!(
            "espeak-ng phonemization failed with status {:?}: {}",
            output.status.code(),
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    let ipa = String::from_utf8_lossy(&output.stdout);
    Ok(ipa
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect::<Vec<_>>()
        .join(" "))
}

/// Convert one clause to the phoneme symbol list the vocabulary expects.
pub fn clause_to_symbols(clause: &Clause) -> Vec<String> {
    let Ok(ipa) = phonemize_clause(&clause.text) else {
        return Vec::new();
    };
    let mut symbols: Vec<String> = ipa
        .chars()
        .filter(|c| !c.is_whitespace() || *c == ' ')
        .map(|c| c.to_string())
        .collect();
    if !clause.terminator.is_empty() {
        symbols.push(clause.terminator.clone());
    }
    symbols
}

/// Full frontend: normalized text -> vocabulary symbol stream.
pub fn text_to_symbols(text: &str) -> Result<Vec<String>> {
    if !espeak_available() {
        anyhow::bail!(
            "espeak-ng is not installed on this system, and LuxTTS needs it for \
             grapheme-to-phoneme conversion. Install it with your package manager \
             (e.g. `sudo apt install espeak-ng` or `sudo pacman -S espeak-ng`)."
        );
    }
    let mut out = Vec::new();
    for clause in split_clauses(&map_punctuations(text)) {
        let mut symbols = clause_to_symbols(&clause);
        if symbols.is_empty() {
            warn!("LuxTTS: clause phonemized to nothing, skipping: {:?}", clause.text);
            continue;
        }
        out.append(&mut symbols);
    }
    Ok(out)
}
