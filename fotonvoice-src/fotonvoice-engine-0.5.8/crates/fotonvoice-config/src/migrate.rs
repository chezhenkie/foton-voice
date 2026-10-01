//! Rewriting legacy key names saved by older recorders so bindings match the
//! names the backends report.

/// The canonical evdev name a legacy name corresponds to, if any.
///
/// A name that no longer matches what a backend reports, that is:
///
/// - Punctuation keys saved by the typed character (`KEY_.`, `KEY_LEFTBRACE`
///   could never fire. The unshifted and shifted characters of a US layout are
///   both mapped, since Shift changes the character typed on the same key.
///
/// - Keys whose DOM name differs from evdev's were saved under the DOM name
///   (`KEY_PRINTSCREEN`, `KEY_AUDIOVOLUMEMUTE`, `KEY_NUMPADENTER`).
/// - The TTS stop key recorder saved letters without the underscore (`KEYV`).
///
/// Numpad digits were saved as the top-row digit (`KEY_1`), which is a real
/// key name, so those cannot be told apart and are left alone. So is `KEY_*`,
/// which could be the keypad `*` or Shift+8.
pub static CANONICAL_SPELLINGS: &[&str] = &[
    "KEY_APOSTROPHE", "KEY_BACKSLASH", "KEY_COMMA", "KEY_COMPOSE", "KEY_DOT",
    "KEY_EQUAL", "KEY_ENTER", "KEY_ESC", "KEY_GRAVE", "KEY_KPENTER",
    "KEY_LEFTBRACE", "KEY_MINUS", "KEY_MUTE", "KEY_NEXTSONG", "KEY_PLAYPAUSE",
    "KEY_PREVIOUSSONG", "KEY_RIGHTBRACE", "KEY_SEMICOLON", "KEY_SLASH",
    "KEY_SPACE", "KEY_STOPCD", "KEY_SYSRQ", "KEY_TAB", "KEY_VOLUMEDOWN",
    "KEY_VOLUMEUP",
];

pub fn canonical_key_name(name: &str) -> Option<&'static str> {
    // A name that is already canonical maps to nothing, so a second
    // canonicalization pass is a no-op. The canonical spellings must stay out
    // of the legacy arms below.
    if CANONICAL_SPELLINGS.contains(&name) {
        return None;
    }
    Some(match name {
        "KEY_ESCAPE" => "KEY_ESC",
        "KEY_SEMICOLON" | "KEY_COLON" | "KEY_;" => "KEY_SEMICOLON",
        "KEY_COMMA" | "KEY_LESS" | "KEY_," | "KEY_<" => "KEY_COMMA",
        "KEY_DOT" | "KEY_GREATER" | "KEY_." | "KEY_>" => "KEY_DOT",
        "KEY_SLASH" | "KEY_QUESTION" | "KEY_/" | "KEY_?" => "KEY_SLASH",
        "KEY_MINUS" | "KEY_UNDERSCORE" | "KEY_-" => "KEY_MINUS",
        "KEY_EQUAL" | "KEY_PLUS" | "KEY_=" | "KEY_+" => "KEY_EQUAL",
        "KEY_LEFTBRACE" | "KEY_LEFTCURLY" | "KEY_[" | "KEY_{" => "KEY_LEFTBRACE",
        "KEY_RIGHTBRACE" | "KEY_RIGHTCURLY" | "KEY_]" | "KEY_}" => "KEY_RIGHTBRACE",
        "KEY_BACKSLASH" | "KEY_PIPE" | "KEY_\\" | "KEY_|" => "KEY_BACKSLASH",
        "KEY_APOSTROPHE" | "KEY_QUOTE" | "KEY_QUOTEDBL" | "KEY_'" | "KEY_\"" => "KEY_APOSTROPHE",
        "KEY_GRAVE" | "KEY_TILDE" | "KEY_`" | "KEY_~" => "KEY_GRAVE",
        "KEY_TAB" => "KEY_TAB",
        "KEY_SPACE" | "KEY_SPACEBAR" => "KEY_SPACE",
        "KEY_ENTER" | "KEY_RETURN" => "KEY_ENTER",
        "KEY_PRINTSCREEN" => "KEY_SYSRQ",
        "KEY_CONTEXTMENU" => "KEY_COMPOSE",
        "KEY_NUMPADENTER" => "KEY_KPENTER",
        "KEY_AUDIOVOLUMEMUTE" => "KEY_MUTE",
        "KEY_AUDIOVOLUMEDOWN" => "KEY_VOLUMEDOWN",
        "KEY_AUDIOVOLUMEUP" => "KEY_VOLUMEUP",
        "KEY_MEDIATRACKNEXT" => "KEY_NEXTSONG",
        "KEY_MEDIATRACKPREVIOUS" => "KEY_PREVIOUSSONG",
        "KEY_MEDIASTOP" => "KEY_STOPCD",
        "KEY_MEDIAPLAYPAUSE" => "KEY_PLAYPAUSE",
        _ => return legacy_letter(name),
    })
}

/// `KEYA` -> `KEY_Z`, as the old TTS stop key recorder wrote letters.
fn legacy_letter(name: &str) -> Option<&'static str> {
    const LETTERS: [&str; 26] = [
        "KEY_A", "KEY_B", "KEY_C", "KEY_D", "KEY_E", "KEY_F", "KEY_G", "KEY_H", "KEY_I", "KEY_J",
        "KEY_K", "KEY_L", "KEY_M", "KEY_N", "KEY_O", "KEY_P", "KEY_Q", "KEY_R", "KEY_S", "KEY_T",
        "KEY_U", "KEY_V", "KEY_W", "KEY_X", "KEY_Y", "KEY_Z",
    ];
    match name.strip_prefix("KEY")?.as_bytes() {
        [c @ b'A'..=b'Z'] => Some(LETTERS[usize::from(c - b'A')]),
        _ => None,
    }
}

/// Rewrite every legacy key name in `keys` to its canonical spelling,
/// returning whether anything changed.
pub fn canonicalize_key_names(keys: &mut [String]) -> bool {
    let mut changed = false;
    for key in keys.iter_mut() {
        if let Some(canonical) = canonical_key_name(key) {
            *key = canonical.to_string();
            changed = true;
        }
    }
    changed
}

#[cfg(test)]
mod tests {
    use super::{canonical_key_name, canonicalize_key_names};

    /// Every name the match arms can emit is already canonical, so feeding one
    /// back in is a no-op. Without this, a second pass would keep rewriting.
    #[test]
    fn no_canonical_name_is_also_a_legacy_input() {
        for name in super::CANONICAL_SPELLINGS {
            assert_eq!(canonical_key_name(name), None, "{name} is already canonical");
        }
    }

    /// The unshifted and shifted characters of every US punctuation key both
    /// reach the same evdev name, because Shift changes the character typed on
    /// the same physical key.
    #[test]
    fn both_characters_of_each_punctuation_key_resolve() {
        for (unshifted, shifted, canonical) in [
            ("KEY_;", "KEY_COLON", "KEY_SEMICOLON"),
            ("KEY_,", "KEY_<", "KEY_COMMA"),
            ("KEY_.", "KEY_>", "KEY_DOT"),
            ("KEY_/", "KEY_?", "KEY_SLASH"),
            ("KEY_-", "KEY_UNDERSCORE", "KEY_MINUS"),
            ("KEY_=", "KEY_+", "KEY_EQUAL"),
            ("KEY_[", "KEY_{", "KEY_LEFTBRACE"),
            ("KEY_]", "KEY_}", "KEY_RIGHTBRACE"),
            ("KEY_\\", "KEY_|", "KEY_BACKSLASH"),
            ("KEY_'", "KEY_\"", "KEY_APOSTROPHE"),
            ("KEY_`", "KEY_~", "KEY_GRAVE"),
        ] {
            assert_eq!(canonical_key_name(unshifted), Some(canonical), "{unshifted}");
            assert_eq!(canonical_key_name(shifted), Some(canonical), "{shifted}");
        }
    }

    #[test]
    fn legacy_key_names_become_the_names_the_backends_report() {
        assert_eq!(canonical_key_name("KEY_ESCAPE"), Some("KEY_ESC"));
        assert_eq!(canonical_key_name("KEY_."), Some("KEY_DOT"));
        assert_eq!(canonical_key_name("KEY_>"), Some("KEY_DOT"), "shifted period");
        assert_eq!(canonical_key_name("KEY_\\"), Some("KEY_BACKSLASH"));
        assert_eq!(canonical_key_name("KEY_DOT"), None, "already canonical");
        assert_eq!(canonical_key_name("KEY_1"), None, "a real key is left alone");

        let mut keys = vec!["KEY_LEFTCTRL".to_string(), "KEY_/".to_string()];
        assert!(canonicalize_key_names(&mut keys));
        assert_eq!(keys, vec!["KEY_LEFTCTRL", "KEY_SLASH"]);
        assert!(!canonicalize_key_names(&mut keys), "a second pass changes nothing");
    }
}
