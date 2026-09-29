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
pub fn canonical_key_name(name: &str) -> Option<&'static str> {
    Some(match name {
        "KEY_ESCAPE" => "KEY_ESC",
        "KEY_SEMICOLON" | "KEY_COLON" => "KEY_SEMICOLON",
        "KEY_COMMA" | "KEY_LESS" => "KEY_COMMA",
        "KEY_DOT" | "KEY_GREATER" => "KEY_DOT",
        "KEY_SLASH" | "KEY_QUESTION" => "KEY_SLASH",
        "KEY_MINUS" | "KEY_UNDERSCORE" => "KEY_MINUS",
        "KEY_EQUAL" | "KEY_PLUS" => "KEY_EQUAL",
        "KEY_LEFTBRACE" | "KEY_LEFTCURLY" => "KEY_LEFTBRACE",
        "KEY_RIGHTBRACE" | "KEY_RIGHTCURLY" => "KEY_RIGHTBRACE",
        "KEY_BACKSLASH" | "KEY_PIPE" => "KEY_BACKSLASH",
        "KEY_APOSTROPHE" | "KEY_QUOTE" | "KEY_QUOTEDBL" => "KEY_APOSTROPHE",
        "KEY_GRAVE" | "KEY_TILDE" => "KEY_GRAVE",
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
