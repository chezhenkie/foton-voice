// Browser KeyboardEvent key/code -> evdev key name. Shared by the hotkey
// capture in HotkeysTab and the stop-key capture in TtsTab.
export function mapBrowserKeyToEvdev(key: string, code: string): string {
  const codeUpper = code.toUpperCase();
  if (key === "Control") return "KEY_LEFTCTRL";
  if (key === "Alt") return "KEY_LEFTALT";
  if (key === "Shift") return "KEY_LEFTSHIFT";
  if (key === "Meta" || key === "OS" || key === "Super") return "KEY_LEFTMETA";

  if (codeUpper === "SPACE") return "KEY_SPACE";
  if (codeUpper === "ENTER") return "KEY_ENTER";
  if (codeUpper === "ESCAPE" || codeUpper === "ESC") return "KEY_ESC";
  if (codeUpper === "TAB") return "KEY_TAB";
  if (codeUpper === "BACKSPACE") return "KEY_BACKSPACE";
  if (codeUpper === "DELETE") return "KEY_DELETE";

  if (/^KEY[A-Z]$/.test(codeUpper)) {
    return `KEY_${codeUpper.slice(3)}`;
  }
  if (codeUpper.startsWith("KEY")) return codeUpper;
  if (codeUpper.startsWith("DIGIT")) return `KEY_${codeUpper.replace("DIGIT", "")}`;
  if (codeUpper.startsWith("ARROW")) return `KEY_${codeUpper.replace("ARROW", "")}`;
  if (codeUpper.startsWith("F") && codeUpper.length > 1) return `KEY_${codeUpper}`;

  if (key.length === 1) return `KEY_${key.toUpperCase()}`;
  return `KEY_${codeUpper}`;
}
