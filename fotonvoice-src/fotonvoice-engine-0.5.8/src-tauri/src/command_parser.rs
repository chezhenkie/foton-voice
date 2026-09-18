use regex::Regex;

/// Attempts to intercept a command from the transcription.
///
/// Patterns:
/// 1. "FotonVoice Engine, [command], [text]"
/// 2. "[text]. FotonVoice Engine, [command]"
///
/// Returns `Some((target_id, payload))` if a command is successfully parsed,
/// otherwise `None`.
pub fn parse_command_routing(text: &str) -> Option<(String, String)> {
    // Every transcription reaches this, so the cheap rejection comes first and
    // without allocating a lowercased copy of the text to do it.
    if !contains_ignore_ascii_case(text, "fotonvoice-engine") {
        return None;
    }

    // Pattern 1: "FotonVoice Engine, [command], [text]" (or "FotonVoice Engine [command] [text]")
    // Regex: (?i)fotonvoice-engine[,\s]+([a-z0-9_-]+)[,\s]+(.*)
    static RE_PREFIX: std::sync::OnceLock<Regex> = std::sync::OnceLock::new();
    let re_prefix = RE_PREFIX
        .get_or_init(|| Regex::new(r"(?i)fotonvoice-engine[,\s]+([a-z0-9_-]+)[,\s]+(.*)").unwrap());
    if let Some(caps) = re_prefix.captures(text) {
        let cmd = caps.get(1)?.as_str().to_string();
        let payload = caps.get(2)?.as_str().trim().to_string();
        return Some((cmd, payload));
    }

    // Pattern 2: "[text]. FotonVoice Engine, [command]"
    // Regex: (.*)[.,\s]+(?i)fotonvoice-engine[,\s]+([a-z0-9_-]+)
    static RE_SUFFIX: std::sync::OnceLock<Regex> = std::sync::OnceLock::new();
    let re_suffix = RE_SUFFIX
        .get_or_init(|| Regex::new(r"(.*)[.,\s]+(?i)fotonvoice-engine[,\s]+([a-z0-9_-]+)").unwrap());
    if let Some(caps) = re_suffix.captures(text) {
        let payload = caps.get(1)?.as_str().trim().to_string();
        let cmd = caps.get(2)?.as_str().to_string();
        return Some((cmd, payload));
    }

    None
}

/// ASCII-case-insensitive substring search that does not allocate.
fn contains_ignore_ascii_case(haystack: &str, needle_lower: &str) -> bool {
    let (h, n) = (haystack.as_bytes(), needle_lower.as_bytes());
    if n.is_empty() || h.len() < n.len() {
        return n.is_empty();
    }
    h.windows(n.len())
        .any(|w| w.eq_ignore_ascii_case(n))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prefix_pattern() {
        let input = "FotonVoice Engine, notes, Hello there!";
        let result = parse_command_routing(input);
        assert!(result.is_some());
        let (cmd, payload) = result.unwrap();
        assert_eq!(cmd, "notes");
        assert_eq!(payload, "Hello there!");
    }

    #[test]
    fn test_prefix_no_comma() {
        let input = "fotonvoice-engine notes hello there";
        let result = parse_command_routing(input);
        assert!(result.is_some());
        let (cmd, payload) = result.unwrap();
        assert_eq!(cmd, "notes");
        assert_eq!(payload, "hello there");
    }

    #[test]
    fn test_suffix_pattern() {
        let input = "Hello there. FotonVoice Engine, notes";
        let result = parse_command_routing(input);
        assert!(result.is_some());
        let (cmd, payload) = result.unwrap();
        assert_eq!(cmd, "notes");
        assert_eq!(payload, "Hello there");
    }

    #[test]
    fn test_no_trigger() {
        let input = "Just a normal sentence.";
        let result = parse_command_routing(input);
        assert!(result.is_none());
    }

    #[test]
    fn test_invalid_command_format() {
        let input = "FotonVoice Engine , , text";
        let result = parse_command_routing(input);
        // The regex might match but cmd would be empty if we aren't careful.
        // [a-z0-9_-]+ requires at least one character.
        assert!(result.is_none());
    }
}
