//! Translating FotonVoice Engine key combinations into desktop shortcut accelerators.

/// Why a key combination cannot be a portal shortcut.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TriggerProblem {
    /// Nothing was captured.
    Empty,
    /// Modifiers only. A lone Super, or Ctrl+Shift with no other key, is not a
    ModifiersOnly,
    /// More than one non-modifier key. An accelerator is modifiers plus exactly
    MultipleKeys,
    /// A key with no keysym equivalent in the shortcuts specification.
    UnsupportedKey(String),
}

impl std::fmt::Display for TriggerProblem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Empty => write!(f, "no keys were captured"),
            Self::ModifiersOnly => write!(
                f,
                "a shortcut needs at least one regular key - modifiers on their own \
                 (Super, Ctrl, Alt, Shift) cannot be registered with your desktop"
            ),
            Self::MultipleKeys => write!(
                f,
                "a shortcut is any number of modifiers plus exactly one regular key"
            ),
            Self::UnsupportedKey(k) => write!(
                f,
                "{} has no equivalent in the desktop shortcut specification",
                k.trim_start_matches("KEY_")
            ),
        }
    }
}

/// Translate evdev key names into the accelerator syntax of the XDG shortcuts
pub fn accelerator(keys: &[String]) -> Result<String, TriggerProblem> {
    if keys.is_empty() {
        return Err(TriggerProblem::Empty);
    }

    let mut modifiers: Vec<&str> = Vec::new();
    let mut key: Option<String> = None;

    for k in keys {
        match modifier_name(k) {
            Some(m) => {
                if !modifiers.contains(&m) {
                    modifiers.push(m);
                }
            }
            None => {
                let named =
                    keysym_name(k).ok_or_else(|| TriggerProblem::UnsupportedKey(k.clone()))?;
                if key.is_some() {
                    return Err(TriggerProblem::MultipleKeys);
                }
                key = Some(named);
            }
        }
    }

    let key = key.ok_or(TriggerProblem::ModifiersOnly)?;
    let mut parts: Vec<&str> = Vec::new();
    for m in ["CTRL", "ALT", "SHIFT", "LOGO"] {
        if modifiers.contains(&m) {
            parts.push(m);
        }
    }
    let mut out = parts.join("+");
    if !out.is_empty() {
        out.push('+');
    }
    out.push_str(&key);
    Ok(out)
}

/// The preferred trigger to hand the portal, or `None` when the combination
pub fn portal_trigger(keys: &[String]) -> Option<String> {
    accelerator(keys).ok()
}

fn modifier_name(key: &str) -> Option<&'static str> {
    match key {
        "KEY_LEFTCTRL" | "KEY_RIGHTCTRL" => Some("CTRL"),
        "KEY_LEFTALT" | "KEY_RIGHTALT" => Some("ALT"),
        "KEY_LEFTSHIFT" | "KEY_RIGHTSHIFT" => Some("SHIFT"),
        "KEY_LEFTMETA" | "KEY_RIGHTMETA" => Some("LOGO"),
        _ => None,
    }
}

/// evdev name -> XKB keysym name, as the shortcuts specification expects.
fn keysym_name(key: &str) -> Option<String> {
    let name = key.strip_prefix("KEY_")?;
    let sym = match name {
        "SPACE" => "space".to_string(),
        "ENTER" | "KPENTER" => "Return".to_string(),
        "TAB" => "Tab".to_string(),
        "ESC" | "ESCAPE" => "Escape".to_string(),
        "BACKSPACE" => "BackSpace".to_string(),
        "DELETE" => "Delete".to_string(),
        "INSERT" => "Insert".to_string(),
        "HOME" => "Home".to_string(),
        "END" => "End".to_string(),
        "PAGEUP" => "Prior".to_string(),
        "PAGEDOWN" => "Next".to_string(),
        "UP" => "Up".to_string(),
        "DOWN" => "Down".to_string(),
        "LEFT" => "Left".to_string(),
        "RIGHT" => "Right".to_string(),
        "MINUS" => "minus".to_string(),
        "EQUAL" => "equal".to_string(),
        "COMMA" => "comma".to_string(),
        "DOT" => "period".to_string(),
        "SLASH" => "slash".to_string(),
        "SEMICOLON" => "semicolon".to_string(),
        "APOSTROPHE" => "apostrophe".to_string(),
        "GRAVE" => "grave".to_string(),
        "BACKSLASH" => "backslash".to_string(),
        "LEFTBRACE" => "bracketleft".to_string(),
        "RIGHTBRACE" => "bracketright".to_string(),
        "CAPSLOCK" => "Caps_Lock".to_string(),
        _ => {
            if let Some(n) = name.strip_prefix('F') {
                if !n.is_empty() && n.chars().all(|c| c.is_ascii_digit()) {
                    return Some(format!("F{n}"));
                }
            }
            if name.len() == 1 {
                let c = name.chars().next()?;
                if c.is_ascii_alphabetic() {
                    return Some(c.to_ascii_lowercase().to_string());
                }
                if c.is_ascii_digit() {
                    return Some(c.to_string());
                }
            }
            return None;
        }
    };
    Some(sym)
}

/// Keys FotonVoice Engine will not hold a *standing* grab on, in every spelling a saved
pub const RESERVED_KEYS: &[&str] = &["KEY_ESC", "KEY_ESCAPE"];

/// Would a standing grab on these keys take something the rest of the desktop
pub fn is_reserved_for_the_desktop(keys: &[String]) -> bool {
    !keys.is_empty()
        && keys
            .iter()
            .all(|k| RESERVED_KEYS.contains(&k.to_ascii_uppercase().as_str()))
}

/// True for a key that only ever acts as a modifier, so it cannot be the one
pub fn is_modifier(key: &str) -> bool {
    modifier_name(key).is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn keys(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn combos_translate_to_accelerator_syntax() {
        assert_eq!(
            accelerator(&keys(&["KEY_LEFTMETA", "KEY_SPACE"])).unwrap(),
            "LOGO+space"
        );
        assert_eq!(
            accelerator(&keys(&["KEY_LEFTCTRL", "KEY_LEFTALT", "KEY_D"])).unwrap(),
            "CTRL+ALT+d"
        );
        assert_eq!(accelerator(&keys(&["KEY_F5"])).unwrap(), "F5");
    }

    #[test]
    fn modifier_order_does_not_change_the_trigger() {
        assert_eq!(
            accelerator(&keys(&["KEY_SPACE", "KEY_LEFTMETA"])),
            accelerator(&keys(&["KEY_LEFTMETA", "KEY_SPACE"])),
        );
    }

    #[test]
    fn left_and_right_modifiers_are_the_same_accelerator() {
        assert_eq!(
            accelerator(&keys(&["KEY_RIGHTCTRL", "KEY_A"])).unwrap(),
            "CTRL+a"
        );
    }

    #[test]
    fn a_bare_modifier_is_rejected_with_a_reason() {
        assert_eq!(
            accelerator(&keys(&["KEY_LEFTMETA"])),
            Err(TriggerProblem::ModifiersOnly)
        );
        assert_eq!(
            accelerator(&keys(&["KEY_LEFTCTRL", "KEY_LEFTSHIFT"])),
            Err(TriggerProblem::ModifiersOnly)
        );
        assert_eq!(
            accelerator(&keys(&["KEY_RIGHTMETA"])),
            Err(TriggerProblem::ModifiersOnly)
        );
    }

    #[test]
    fn bare_escape_is_a_valid_accelerator_that_may_only_be_held_transiently() {
        assert_eq!(accelerator(&keys(&["KEY_ESC"])).unwrap(), "Escape");
        assert!(is_reserved_for_the_desktop(&keys(&["KEY_ESC"])));
        assert!(is_reserved_for_the_desktop(&keys(&["KEY_ESCAPE"])));
    }

    #[test]
    fn escape_with_a_modifier_is_an_ordinary_shortcut() {
        assert_eq!(
            accelerator(&keys(&["KEY_LEFTCTRL", "KEY_ESC"])).unwrap(),
            "CTRL+Escape"
        );
        let ctrl_escape = keys(&["KEY_LEFTCTRL", "KEY_ESC"]);
        assert!(!is_reserved_for_the_desktop(&ctrl_escape));
        assert_eq!(
            accelerator(&keys(&["KEY_LEFTMETA", "KEY_ESCAPE"])).unwrap(),
            "LOGO+Escape"
        );
    }

    #[test]
    fn only_escape_is_reserved() {
        for k in ["KEY_F5", "KEY_SPACE", "KEY_A", "KEY_TAB", "KEY_ENTER"] {
            assert!(
                !is_reserved_for_the_desktop(&keys(&[k])),
                "{k} must stay an ordinary standing shortcut"
            );
        }
        assert!(!is_reserved_for_the_desktop(&[]));
    }

    #[test]
    fn two_regular_keys_are_rejected() {
        assert_eq!(
            accelerator(&keys(&["KEY_A", "KEY_B"])),
            Err(TriggerProblem::MultipleKeys)
        );
        assert_eq!(
            accelerator(&keys(&["KEY_LEFTCTRL", "KEY_A", "KEY_B"])),
            Err(TriggerProblem::MultipleKeys)
        );
    }

    #[test]
    fn an_empty_capture_is_rejected() {
        assert_eq!(accelerator(&[]), Err(TriggerProblem::Empty));
    }

    #[test]
    fn an_unmappable_key_names_itself() {
        let err = accelerator(&keys(&["KEY_FN_F1"])).unwrap_err();
        assert_eq!(err, TriggerProblem::UnsupportedKey("KEY_FN_F1".into()));
        assert!(err.to_string().contains("FN_F1"));
        assert!(!err.to_string().contains("KEY_FN_F1"));
    }

    #[test]
    fn every_problem_explains_itself_without_jargon() {
        for problem in [
            TriggerProblem::Empty,
            TriggerProblem::ModifiersOnly,
            TriggerProblem::MultipleKeys,
            TriggerProblem::UnsupportedKey("KEY_ZZZ".into()),
        ] {
            let msg = problem.to_string();
            assert!(!msg.is_empty());
            assert!(
                !msg.contains("accelerator") || matches!(problem, TriggerProblem::MultipleKeys),
                "user-facing text should avoid spec jargon: {msg}"
            );
        }
    }

    #[test]
    fn modifiers_are_recognised_on_both_sides_of_the_keyboard() {
        for k in [
            "KEY_LEFTCTRL", "KEY_RIGHTCTRL", "KEY_LEFTALT", "KEY_RIGHTALT",
            "KEY_LEFTSHIFT", "KEY_RIGHTSHIFT", "KEY_LEFTMETA", "KEY_RIGHTMETA",
        ] {
            assert!(is_modifier(k), "{k} must count as a modifier");
        }
        assert!(!is_modifier("KEY_SPACE"));
        assert!(!is_modifier("KEY_CAPSLOCK"), "Caps Lock is bindable as a key");
    }

    #[test]
    fn the_bindings_fotonvoice_ships_with_are_all_bindable() {
        for binding in fotonvoice_routing::loader::default_bindings() {
            assert!(
                accelerator(&binding.keys).is_ok(),
                "default binding `{}` is not a valid shortcut: {:?}",
                binding.id,
                accelerator(&binding.keys)
            );
        }
    }

    #[test]
    fn portal_trigger_collapses_problems_to_none() {
        assert_eq!(portal_trigger(&keys(&["KEY_LEFTMETA"])), None);
        assert_eq!(
            portal_trigger(&keys(&["KEY_LEFTMETA", "KEY_SPACE"])).as_deref(),
            Some("LOGO+space")
        );
    }
}
