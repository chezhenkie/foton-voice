//! The Windows keyboard vocabulary, kept off the platform gate on purpose.

use std::sync::atomic::{AtomicBool, Ordering};

use fotonvoice_routing::HotkeyBinding;

use crate::trigger::is_modifier;

pub mod keymap {
    //! Win32 keyboard identity -> the evdev key name the rest of FotonVoice Engine speaks.

    /// Every key FotonVoice Engine can name, in a fixed order. The index into this table is
    pub const NAMES: &[&str] = &[
        "KEY_ESC", "KEY_1", "KEY_2", "KEY_3", "KEY_4", "KEY_5", "KEY_6", "KEY_7",
        "KEY_8", "KEY_9", "KEY_0", "KEY_MINUS", "KEY_EQUAL", "KEY_BACKSPACE",
        "KEY_TAB", "KEY_Q", "KEY_W", "KEY_E", "KEY_R", "KEY_T", "KEY_Y", "KEY_U",
        "KEY_I", "KEY_O", "KEY_P", "KEY_LEFTBRACE", "KEY_RIGHTBRACE", "KEY_ENTER",
        "KEY_LEFTCTRL", "KEY_A", "KEY_S", "KEY_D", "KEY_F", "KEY_G", "KEY_H",
        "KEY_J", "KEY_K", "KEY_L", "KEY_SEMICOLON", "KEY_APOSTROPHE", "KEY_GRAVE",
        "KEY_LEFTSHIFT", "KEY_BACKSLASH", "KEY_Z", "KEY_X", "KEY_C", "KEY_V",
        "KEY_B", "KEY_N", "KEY_M", "KEY_COMMA", "KEY_DOT", "KEY_SLASH",
        "KEY_RIGHTSHIFT",
        "KEY_KPASTERISK", "KEY_LEFTALT", "KEY_SPACE", "KEY_CAPSLOCK",
        "KEY_F1", "KEY_F2", "KEY_F3", "KEY_F4", "KEY_F5", "KEY_F6", "KEY_F7",
        "KEY_F8", "KEY_F9", "KEY_F10", "KEY_F11", "KEY_F12",
        "KEY_NUMLOCK", "KEY_SCROLLLOCK", "KEY_KP7", "KEY_KP8", "KEY_KP9",
        "KEY_KPMINUS", "KEY_KP4", "KEY_KP5", "KEY_KP6", "KEY_KPPLUS", "KEY_KP1",
        "KEY_KP2", "KEY_KP3", "KEY_KP0", "KEY_KPDOT", "KEY_102ND",
        "KEY_KPENTER", "KEY_RIGHTCTRL", "KEY_KPSLASH", "KEY_SYSRQ", "KEY_RIGHTALT",
        "KEY_HOME", "KEY_UP", "KEY_PAGEUP", "KEY_LEFT", "KEY_RIGHT", "KEY_END",
        "KEY_DOWN", "KEY_PAGEDOWN", "KEY_INSERT", "KEY_DELETE", "KEY_LEFTMETA",
        "KEY_RIGHTMETA", "KEY_COMPOSE", "KEY_PAUSE",
        "KEY_MUTE", "KEY_VOLUMEDOWN", "KEY_VOLUMEUP", "KEY_NEXTSONG",
        "KEY_PREVIOUSSONG", "KEY_STOPCD", "KEY_PLAYPAUSE",
    ];

    /// `(scan_code, name)` for keys reported without the extended flag.
    const BASE: &[(u32, &str)] = &[
        (0x01, "KEY_ESC"), (0x02, "KEY_1"), (0x03, "KEY_2"), (0x04, "KEY_3"),
        (0x05, "KEY_4"), (0x06, "KEY_5"), (0x07, "KEY_6"), (0x08, "KEY_7"),
        (0x09, "KEY_8"), (0x0A, "KEY_9"), (0x0B, "KEY_0"), (0x0C, "KEY_MINUS"),
        (0x0D, "KEY_EQUAL"), (0x0E, "KEY_BACKSPACE"), (0x0F, "KEY_TAB"),
        (0x10, "KEY_Q"), (0x11, "KEY_W"), (0x12, "KEY_E"), (0x13, "KEY_R"),
        (0x14, "KEY_T"), (0x15, "KEY_Y"), (0x16, "KEY_U"), (0x17, "KEY_I"),
        (0x18, "KEY_O"), (0x19, "KEY_P"), (0x1A, "KEY_LEFTBRACE"),
        (0x1B, "KEY_RIGHTBRACE"), (0x1C, "KEY_ENTER"), (0x1D, "KEY_LEFTCTRL"),
        (0x1E, "KEY_A"), (0x1F, "KEY_S"), (0x20, "KEY_D"), (0x21, "KEY_F"),
        (0x22, "KEY_G"), (0x23, "KEY_H"), (0x24, "KEY_J"), (0x25, "KEY_K"),
        (0x26, "KEY_L"), (0x27, "KEY_SEMICOLON"), (0x28, "KEY_APOSTROPHE"),
        (0x29, "KEY_GRAVE"), (0x2A, "KEY_LEFTSHIFT"), (0x2B, "KEY_BACKSLASH"),
        (0x2C, "KEY_Z"), (0x2D, "KEY_X"), (0x2E, "KEY_C"), (0x2F, "KEY_V"),
        (0x30, "KEY_B"), (0x31, "KEY_N"), (0x32, "KEY_M"), (0x33, "KEY_COMMA"),
        (0x34, "KEY_DOT"), (0x35, "KEY_SLASH"), (0x36, "KEY_RIGHTSHIFT"),
        (0x37, "KEY_KPASTERISK"), (0x38, "KEY_LEFTALT"), (0x39, "KEY_SPACE"),
        (0x3A, "KEY_CAPSLOCK"), (0x3B, "KEY_F1"), (0x3C, "KEY_F2"),
        (0x3D, "KEY_F3"), (0x3E, "KEY_F4"), (0x3F, "KEY_F5"), (0x40, "KEY_F6"),
        (0x41, "KEY_F7"), (0x42, "KEY_F8"), (0x43, "KEY_F9"), (0x44, "KEY_F10"),
        (0x45, "KEY_NUMLOCK"), (0x46, "KEY_SCROLLLOCK"), (0x47, "KEY_KP7"),
        (0x48, "KEY_KP8"), (0x49, "KEY_KP9"), (0x4A, "KEY_KPMINUS"),
        (0x4B, "KEY_KP4"), (0x4C, "KEY_KP5"), (0x4D, "KEY_KP6"),
        (0x4E, "KEY_KPPLUS"), (0x4F, "KEY_KP1"), (0x50, "KEY_KP2"),
        (0x51, "KEY_KP3"), (0x52, "KEY_KP0"), (0x53, "KEY_KPDOT"),
        (0x56, "KEY_102ND"), (0x57, "KEY_F11"), (0x58, "KEY_F12"),
    ];

    /// `(scan_code, name)` for keys the hook reports with `LLKHF_EXTENDED`. These
    const EXTENDED: &[(u32, &str)] = &[
        (0x1C, "KEY_KPENTER"), (0x1D, "KEY_RIGHTCTRL"), (0x35, "KEY_KPSLASH"),
        (0x37, "KEY_SYSRQ"), (0x38, "KEY_RIGHTALT"), (0x47, "KEY_HOME"),
        (0x48, "KEY_UP"), (0x49, "KEY_PAGEUP"), (0x4B, "KEY_LEFT"),
        (0x4D, "KEY_RIGHT"), (0x4F, "KEY_END"), (0x50, "KEY_DOWN"),
        (0x51, "KEY_PAGEDOWN"), (0x52, "KEY_INSERT"), (0x53, "KEY_DELETE"),
        (0x5B, "KEY_LEFTMETA"), (0x5C, "KEY_RIGHTMETA"), (0x5D, "KEY_COMPOSE"),
    ];

    /// `(virtual_key, name)` consulted before the scan-code tables for keys whose
    const BY_VIRTUAL_KEY: &[(u32, &str)] = &[
        (0x13, "KEY_PAUSE"),          // VK_PAUSE - shares scan code 0x45 with NumLock
        (0x90, "KEY_NUMLOCK"),        // VK_NUMLOCK
        (0x2C, "KEY_SYSRQ"),          // VK_SNAPSHOT (Print Screen)
        (0xAD, "KEY_MUTE"),           // VK_VOLUME_MUTE
        (0xAE, "KEY_VOLUMEDOWN"),     // VK_VOLUME_DOWN
        (0xAF, "KEY_VOLUMEUP"),       // VK_VOLUME_UP
        (0xB0, "KEY_NEXTSONG"),       // VK_MEDIA_NEXT_TRACK
        (0xB1, "KEY_PREVIOUSSONG"),   // VK_MEDIA_PREV_TRACK
        (0xB2, "KEY_STOPCD"),         // VK_MEDIA_STOP
        (0xB3, "KEY_PLAYPAUSE"),      // VK_MEDIA_PLAY_PAUSE
    ];

    /// Virtual keys resolved before the scan-code tables, because their scan code
    const VK_WINS_OVER_SCANCODE: &[u32] = &[0x13, 0x90, 0x2C];

    /// `VK_PACKET` - the virtual key Windows reports for a `KEYEVENTF_UNICODE`
    pub const VK_PACKET: u32 = 0xE7;

    /// Resolve one hook event to a compact key id, or `None` for a key FotonVoice Engine has
    pub fn lookup(scan_code: u32, extended: bool, virtual_key: u32) -> Option<usize> {
        if virtual_key == VK_PACKET {
            return None;
        }
        if VK_WINS_OVER_SCANCODE.contains(&virtual_key) {
            return by_virtual_key(virtual_key);
        }
        let table = if extended { EXTENDED } else { BASE };
        if let Some(name) = table.iter().find(|(c, _)| *c == scan_code).map(|(_, n)| *n) {
            return index_of(name);
        }
        by_virtual_key(virtual_key)
    }

    fn by_virtual_key(virtual_key: u32) -> Option<usize> {
        BY_VIRTUAL_KEY
            .iter()
            .find(|(vk, _)| *vk == virtual_key)
            .and_then(|(_, name)| index_of(name))
    }

    /// The evdev name for a key id.
    pub fn name(id: usize) -> &'static str {
        NAMES[id]
    }

    /// The key id for an evdev name, for turning saved bindings into ids the hook
    pub fn index_of(name: &str) -> Option<usize> {
        NAMES.iter().position(|n| *n == name)
    }
    #[cfg(test)]
    mod tests {
        use super::*;

        fn resolve(scan: u32, extended: bool, vk: u32) -> Option<&'static str> {
            lookup(scan, extended, vk).map(name)
        }

        #[test]
        fn the_default_binding_resolves() {
            assert_eq!(resolve(0x5B, true, 0x5B), Some("KEY_LEFTMETA"));
            assert_eq!(resolve(0x39, false, 0x20), Some("KEY_SPACE"));
        }

        #[test]
        fn modifiers_keep_their_side_like_evdev() {
            assert_eq!(resolve(0x1D, false, 0xA2), Some("KEY_LEFTCTRL"));
            assert_eq!(resolve(0x1D, true, 0xA3), Some("KEY_RIGHTCTRL"));
            assert_eq!(resolve(0x2A, false, 0xA0), Some("KEY_LEFTSHIFT"));
            assert_eq!(resolve(0x36, false, 0xA1), Some("KEY_RIGHTSHIFT"));
            assert_eq!(resolve(0x38, false, 0xA4), Some("KEY_LEFTALT"));
            assert_eq!(resolve(0x38, true, 0xA5), Some("KEY_RIGHTALT"));
        }

        #[test]
        fn letters_are_positional_not_layout_dependent() {
            assert_eq!(resolve(0x1E, false, 0x41), Some("KEY_A"));
            assert_eq!(resolve(0x1E, false, 0x51), Some("KEY_A"));
        }

        #[test]
        fn escape_is_evdevs_shorter_spelling() {
            assert_eq!(resolve(0x01, false, 0x1B), Some("KEY_ESC"));
        }

        #[test]
        fn pause_does_not_masquerade_as_numlock() {
            assert_eq!(resolve(0x45, false, 0x13), Some("KEY_PAUSE"));
            assert_eq!(resolve(0x45, false, 0x90), Some("KEY_NUMLOCK"));
        }

        #[test]
        fn the_extended_flag_separates_the_keypad_from_the_edit_block() {
            assert_eq!(resolve(0x1C, false, 0x0D), Some("KEY_ENTER"));
            assert_eq!(resolve(0x1C, true, 0x0D), Some("KEY_KPENTER"));
            assert_eq!(resolve(0x52, false, 0x60), Some("KEY_KP0"));
            assert_eq!(resolve(0x52, true, 0x2D), Some("KEY_INSERT"));
            assert_eq!(resolve(0x53, true, 0x2E), Some("KEY_DELETE"));
        }

        #[test]
        fn injected_unicode_is_not_a_key() {
            assert_eq!(lookup(u32::from('a'), false, VK_PACKET), None);
            assert_eq!(lookup(u32::from('%'), false, VK_PACKET), None);
        }

        #[test]
        fn an_unknown_key_is_dropped_rather_than_named() {
            assert_eq!(lookup(0x00, false, 0x00), None);
            assert_eq!(lookup(0xFE, false, 0xFE), None);
        }

        #[test]
        fn every_table_entry_has_a_slot_in_names() {
            for (code, n) in BASE.iter().chain(EXTENDED) {
                assert!(index_of(n).is_some(), "{n} (scan {code:#04x}) is missing from NAMES");
            }
            for (vk, n) in BY_VIRTUAL_KEY {
                assert!(index_of(n).is_some(), "{n} (vk {vk:#04x}) is missing from NAMES");
            }
        }

        #[test]
        fn names_are_unique_so_ids_are_stable() {
            let mut seen = std::collections::HashSet::new();
            for n in NAMES {
                assert!(seen.insert(*n), "{n} appears twice in NAMES");
            }
        }

        #[test]
        fn every_key_the_settings_ui_can_record_is_reachable() {
            let recordable = [
                "KEY_LEFTCTRL", "KEY_LEFTALT", "KEY_LEFTSHIFT", "KEY_LEFTMETA",
                "KEY_SPACE", "KEY_ENTER", "KEY_ESC", "KEY_TAB", "KEY_BACKSPACE",
                "KEY_DELETE", "KEY_A", "KEY_Z", "KEY_0", "KEY_9", "KEY_UP",
                "KEY_DOWN", "KEY_LEFT", "KEY_RIGHT", "KEY_F1", "KEY_F12",
            ];
            for name in recordable {
                assert!(
                    index_of(name).is_some(),
                    "the settings UI can record {name}, but no key maps to it"
                );
            }
        }
    }

}

/// What the hook needs to know to decide suppression, rebuilt on every reload.
#[derive(Default)]
pub struct SuppressPlan {
    /// `(key ids of the binding, whether each id is a modifier)`.
    combos: Vec<Vec<(usize, bool)>>,
}

impl SuppressPlan {
    pub fn build(bindings: &[HotkeyBinding]) -> Self {
        let mut combos = Vec::new();
        for b in bindings {
            if b.disabled || b.keys.is_empty() {
                continue;
            }
            let mut ids = Vec::with_capacity(b.keys.len());
            for k in &b.keys {
                match keymap::index_of(k) {
                    Some(id) => ids.push((id, is_modifier(k))),
                    None => {
                        ids.clear();
                        break;
                    }
                }
            }
            if !ids.is_empty() {
                combos.push(ids);
            }
        }
        Self { combos }
    }

    /// Whether pressing `id` - with `pressed` describing what is already held -
    pub fn should_swallow(&self, id: usize, pressed: &[AtomicBool]) -> bool {
        self.combos.iter().any(|combo| {
            let Some(&(_, this_is_modifier)) = combo.iter().find(|(k, _)| *k == id) else {
                return false;
            };
            if this_is_modifier {
                return false;
            }
            combo
                .iter()
                .all(|(k, _)| *k == id || pressed[*k].load(Ordering::Relaxed))
        })
    }
}

#[cfg(test)]
mod suppress_tests {
    use super::*;
    use fotonvoice_routing::GestureType;

    fn binding(id: &str, keys: &[&str]) -> HotkeyBinding {
        HotkeyBinding {
            id: id.to_string(),
            label: id.to_string(),
            keys: keys.iter().map(|k| k.to_string()).collect(),
            gesture: GestureType::Hold,
            target_id: "t".to_string(),
            target_ids: vec!["t".to_string()],
            tap_ms: 300,
            hold_threshold_ms: 100,
            disabled: false,
            openai_enabled: Some(false),
            openai_model: None,
            openai_mode: None,
            openai_prompt: None,
            openai_system_prompt: None,
            s1_mini_enabled: None,
        }
    }

    fn held(down: &[&str]) -> Vec<AtomicBool> {
        let state: Vec<AtomicBool> = (0..keymap::NAMES.len())
            .map(|_| AtomicBool::new(false))
            .collect();
        for name in down {
            state[keymap::index_of(name).unwrap()].store(true, Ordering::Relaxed);
        }
        state
    }

    fn id(name: &str) -> usize {
        keymap::index_of(name).unwrap()
    }

    #[test]
    fn the_finishing_key_of_a_combo_is_swallowed() {
        let plan = SuppressPlan::build(&[binding("b", &["KEY_LEFTMETA", "KEY_SPACE"])]);
        assert!(plan.should_swallow(id("KEY_SPACE"), &held(&["KEY_LEFTMETA"])));
    }

    #[test]
    fn a_modifier_is_never_swallowed() {
        let combo = SuppressPlan::build(&[binding("b", &["KEY_LEFTMETA", "KEY_SPACE"])]);
        assert!(!combo.should_swallow(id("KEY_LEFTMETA"), &held(&["KEY_SPACE"])));

        let modifier_only = SuppressPlan::build(&[binding("m", &["KEY_RIGHTALT"])]);
        assert!(!modifier_only.should_swallow(id("KEY_RIGHTALT"), &held(&[])));
    }

    #[test]
    fn an_incomplete_combo_passes_through() {
        let plan = SuppressPlan::build(&[binding("b", &["KEY_LEFTMETA", "KEY_SPACE"])]);
        assert!(!plan.should_swallow(id("KEY_SPACE"), &held(&[])));
    }

    #[test]
    fn an_unrelated_key_passes_through() {
        let plan = SuppressPlan::build(&[binding("b", &["KEY_LEFTMETA", "KEY_SPACE"])]);
        assert!(!plan.should_swallow(id("KEY_T"), &held(&["KEY_LEFTMETA"])));
    }

    #[test]
    fn a_disabled_binding_takes_no_key_from_the_desktop() {
        let mut b = binding("b", &["KEY_LEFTMETA", "KEY_SPACE"]);
        b.disabled = true;
        let plan = SuppressPlan::build(&[b]);
        assert!(!plan.should_swallow(id("KEY_SPACE"), &held(&["KEY_LEFTMETA"])));
    }

    #[test]
    fn a_binding_naming_a_key_this_backend_cannot_see_takes_nothing() {
        let plan = SuppressPlan::build(&[binding("b", &["KEY_NOT_A_REAL_KEY", "KEY_SPACE"])]);
        assert!(!plan.should_swallow(id("KEY_SPACE"), &held(&[])));
    }

    #[test]
    fn a_lone_non_modifier_binding_is_swallowed() {
        let plan = SuppressPlan::build(&[binding("b", &["KEY_PAUSE"])]);
        assert!(plan.should_swallow(id("KEY_PAUSE"), &held(&[])));
    }
}
