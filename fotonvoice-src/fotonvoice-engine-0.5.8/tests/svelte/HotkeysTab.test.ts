import { describe, test, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent } from "@testing-library/svelte";
import HotkeysTab from "../../src/lib/Settings/HotkeysTab.svelte";
import type { HotkeyBinding, OutputTarget } from "../../src/lib/Settings/routing-types";

let mockBindings: HotkeyBinding[] = [];
let mockTargets: OutputTarget[] = [];
let mockHotkeyStatus: Record<string, unknown> = {};
let mockKeysChecks: Array<Record<string, unknown>> = [];
let keysCheckCalls: string[][] = [];
let openShortcutSettingsCalls = 0;
let openShortcutSettingsResult: "ok" | string = "ok";

vi.mock("@tauri-apps/api/core", () => ({
  invoke: vi.fn(async (cmd, args) => {
    if (cmd === "get_targets") {
      return mockTargets;
    }
    if (cmd === "get_bindings") {
      return mockBindings;
    }
    if (cmd === "check_hotkey_status") {
      return mockHotkeyStatus;
    }
    if (cmd === "check_hotkey_keys") {
      keysCheckCalls.push([...((args as { keys: string[] })?.keys ?? [])]);
      return mockKeysChecks.shift() ?? { accepted: true, enforced: false, accelerator: null, problem: null, message: null };
    }
    if (cmd === "open_shortcut_settings") {
      openShortcutSettingsCalls += 1;
      if (openShortcutSettingsResult !== "ok") {
        throw new Error(openShortcutSettingsResult);
      }
      return null;
    }
    return {};
  }),
}));

function rejected(message: string) {
  return {
    accepted: false,
    enforced: true,
    accelerator: null,
    problem: "modifiers_only",
    message,
  };
}

function hotkeyStatus(overrides: Record<string, unknown> = {}) {
  return {
    is_active: true,
    backend: "portal",
    is_private: true,
    portal_error: null,
    portal_refused: false,
    shortcuts: [],
    supported_gestures: ["hold", "toggle", "double_tap", "double_tap_hold"],
    x11_error: null,
    session_type: "wayland",
    devices_total: 0,
    devices_readable: 0,
    needs_attention: false,
    needs_manual_enable: false,
    manual_enable_hint: null,
    detail: "Your desktop is handling FotonVoice Engine's global shortcuts.",
    ...overrides,
  };
}

describe("HotkeysTab.svelte Conflict Detection and Nested Modal", () => {
  beforeEach(() => {
    mockTargets = [
      {
        id: "default",
        label: "Focused Window",
        delivery: "inject",
        file_prefix: "",
        file_timestamp: true,
        strip_newlines: false,
        tts_engine: "None",
      },
    ];
    mockBindings = [];
    mockHotkeyStatus = hotkeyStatus();
    mockKeysChecks = [];
    keysCheckCalls = [];
    openShortcutSettingsCalls = 0;
    openShortcutSettingsResult = "ok";
  });

  test("refuses a bare-modifier capture and says why", async () => {
    mockBindings = [
      {
        id: "bind1",
        keys: ["KEY_LEFTCTRL", "KEY_LEFTALT", "KEY_D"],
        gesture: "double_tap",
        target_id: "default",
        tap_ms: 300,
        hold_threshold_ms: 200,
        label: "Dictate",
        disabled: false,
      },
    ];
    mockKeysChecks = [
      rejected(
        "Your desktop cannot register this shortcut: a shortcut needs at least one regular key. Add a regular key to the combination - Super+Space and Ctrl+Alt+D both work.",
      ),
    ];

    const { container } = render(HotkeysTab);
    const editBtn = await screen.findByRole("button", { name: /Edit/i });
    await fireEvent.click(editBtn);

    const recorder = container.querySelector('[aria-label="Base Hotkey recorder input"]')!;
    await fireEvent.focus(recorder);
    await fireEvent.keyDown(recorder, { key: "Meta", code: "MetaLeft" });
    await fireEvent.keyUp(recorder, { key: "Meta", code: "MetaLeft" });

    expect(await screen.findByText(/That combination was not accepted/i)).toBeTruthy();
    expect(await screen.findByText(/needs at least one regular key/i)).toBeTruthy();
    expect(keysCheckCalls).toEqual([["KEY_LEFTMETA"]]);
  });

  test("keeps the previous combination when a capture is refused", async () => {
    mockBindings = [
      {
        id: "bind1",
        keys: ["KEY_LEFTCTRL", "KEY_LEFTALT", "KEY_D"],
        gesture: "hold",
        target_id: "default",
        tap_ms: 300,
        hold_threshold_ms: 200,
        label: "Dictate",
        disabled: false,
      },
    ];
    mockKeysChecks = [rejected("Your desktop cannot register this shortcut.")];

    const { container } = render(HotkeysTab);
    await fireEvent.click(await screen.findByRole("button", { name: /Edit/i }));

    const recorder = container.querySelector('[aria-label="Base Hotkey recorder input"]')!;
    await fireEvent.focus(recorder);
    await fireEvent.keyDown(recorder, { key: "Shift", code: "ShiftLeft" });
    await fireEvent.keyUp(recorder, { key: "Shift", code: "ShiftLeft" });

    await screen.findByText(/not accepted/i);
    for (const key of ["LEFTCTRL", "LEFTALT", "D"]) {
      expect(screen.getAllByText(key).length).toBeGreaterThan(0);
    }
    expect(screen.queryByText("LEFTSHIFT")).toBeNull();
  });

  test("accepts a valid combination and shows what the desktop will bind", async () => {
    mockBindings = [
      {
        id: "bind1",
        keys: ["KEY_LEFTCTRL", "KEY_LEFTALT", "KEY_D"],
        gesture: "hold",
        target_id: "default",
        tap_ms: 300,
        hold_threshold_ms: 200,
        label: "Dictate",
        disabled: false,
      },
    ];
    mockKeysChecks = [
      { accepted: true, enforced: false, accelerator: "LOGO+space", problem: null, message: null },
    ];

    const { container } = render(HotkeysTab);
    await fireEvent.click(await screen.findByRole("button", { name: /Edit/i }));

    const recorder = container.querySelector('[aria-label="Base Hotkey recorder input"]')!;
    await fireEvent.focus(recorder);
    await fireEvent.keyDown(recorder, { key: "Meta", code: "MetaLeft" });
    await fireEvent.keyDown(recorder, { key: " ", code: "Space" });
    await fireEvent.keyUp(recorder, { key: " ", code: "Space" });

    expect(await screen.findByText("LOGO+space")).toBeTruthy();
    expect(screen.queryByText(/not accepted/i)).toBeNull();
    expect(keysCheckCalls).toEqual([["KEY_LEFTMETA", "KEY_SPACE"]]);
  });

  test("nudges the user while only modifiers are held", async () => {
    mockBindings = [
      {
        id: "bind1",
        keys: ["KEY_LEFTCTRL", "KEY_LEFTALT", "KEY_D"],
        gesture: "hold",
        target_id: "default",
        tap_ms: 300,
        hold_threshold_ms: 200,
        label: "Dictate",
        disabled: false,
      },
    ];

    const { container } = render(HotkeysTab);
    await fireEvent.click(await screen.findByRole("button", { name: /Edit/i }));

    const recorder = container.querySelector('[aria-label="Base Hotkey recorder input"]')!;
    await fireEvent.focus(recorder);
    await fireEvent.keyDown(recorder, { key: "Meta", code: "MetaLeft" });

    expect(await screen.findByText(/add a regular key/i)).toBeTruthy();
  });

  test("flags a saved binding the desktop cannot register", async () => {
    mockBindings = [
      {
        id: "legacy",
        keys: ["KEY_LEFTMETA"],
        gesture: "double_tap",
        target_id: "default",
        tap_ms: 300,
        hold_threshold_ms: 200,
        label: "Legacy Super Tap",
        disabled: false,
      },
    ];

    render(HotkeysTab);

    expect(await screen.findByText(/needs a regular key/i)).toBeTruthy();
  });

  test("does not block bare modifiers when FotonVoice Engine watches the keyboard itself", async () => {
    mockHotkeyStatus = hotkeyStatus({
      backend: "evdev",
      is_private: false,
      detail: "FotonVoice Engine is reading input devices directly.",
    });
    mockBindings = [
      {
        id: "bind1",
        keys: ["KEY_LEFTCTRL", "KEY_LEFTALT", "KEY_D"],
        gesture: "hold",
        target_id: "default",
        tap_ms: 300,
        hold_threshold_ms: 200,
        label: "Dictate",
        disabled: false,
      },
    ];
    mockKeysChecks = [
      {
        accepted: true,
        enforced: false,
        accelerator: null,
        problem: "modifiers_only",
        message: "This works right now, because FotonVoice Engine is watching the keyboard itself.",
      },
    ];

    const { container } = render(HotkeysTab);
    await fireEvent.click(await screen.findByRole("button", { name: /Edit/i }));

    const recorder = container.querySelector('[aria-label="Base Hotkey recorder input"]')!;
    await fireEvent.focus(recorder);
    await fireEvent.keyDown(recorder, { key: "Meta", code: "MetaLeft" });
    await fireEvent.keyUp(recorder, { key: "Meta", code: "MetaLeft" });

    expect(await screen.findByText(/watching the keyboard itself/i)).toBeTruthy();
    expect(screen.queryByText(/not accepted/i)).toBeNull();
    expect(screen.getAllByText("LEFTMETA").length).toBeGreaterThan(0);
  });

  test("offers only the four supported gestures", async () => {
    mockBindings = [
      {
        id: "bind1",
        keys: ["KEY_LEFTMETA"],
        gesture: "double_tap",
        target_id: "default",
        tap_ms: 300,
        hold_threshold_ms: 200,
        label: "Tap",
        disabled: false,
      },
    ];

    render(HotkeysTab);
    const editBtn = await screen.findByRole("button", { name: /Edit/i });
    await fireEvent.click(editBtn);
    expect(screen.getByText("Edit Hotkey Binding")).not.toBeNull();

    expect(screen.queryByText(/chord/i)).toBeNull();
    expect(screen.queryByText(/sub ?key/i)).toBeNull();
    expect(screen.queryByText(/Base Combo/i)).toBeNull();
    expect(
      await screen.findByText(/Double-tap hotkey to trigger recording/i),
    ).toBeTruthy();
  });

  test("tells the user their desktop owns the shortcut keys when expanded", async () => {
    render(HotkeysTab);

    expect(
      await screen.findByText("Your desktop is handling these shortcuts"),
    ).toBeTruthy();
    expect(screen.queryByText(/Your desktop decides which keys/i)).toBeNull();

    const toggle = await screen.findByRole("button", {
      name: /Toggle shortcut backend details/i,
    });
    await fireEvent.click(toggle);

    expect(await screen.findByText(/Your desktop decides which keys/i)).toBeTruthy();

    await fireEvent.click(toggle);
    expect(screen.queryByText(/Your desktop decides which keys/i)).toBeNull();
  });

  test("shows the keys the compositor actually bound, not the ones requested", async () => {
    mockBindings = [
      {
        id: "bind1",
        keys: ["KEY_LEFTMETA", "KEY_SPACE"],
        gesture: "hold",
        target_id: "default",
        tap_ms: 300,
        hold_threshold_ms: 200,
        label: "Dictate",
        disabled: false,
      },
    ];
    mockHotkeyStatus = hotkeyStatus({
      shortcuts: [
        {
          binding_ids: ["bind1"],
          requested: "LOGO+space",
          trigger_description: "Ctrl+Alt+D",
          bound: true,
        },
      ],
    });

    render(HotkeysTab);

    expect(await screen.findByText(/desktop: Ctrl\+Alt\+D/)).toBeTruthy();
  });

  test("flags a shortcut the desktop refused to bind", async () => {
    mockBindings = [
      {
        id: "bind1",
        keys: ["KEY_LEFTMETA"],
        gesture: "double_tap",
        target_id: "default",
        tap_ms: 300,
        hold_threshold_ms: 200,
        label: "Dictate",
        disabled: false,
      },
    ];
    mockHotkeyStatus = hotkeyStatus({
      shortcuts: [
        {
          binding_ids: ["bind1"],
          requested: null,
          trigger_description: "",
          bound: false,
        },
      ],
    });

    render(HotkeysTab);

    expect(await screen.findByText(/not bound by your desktop/i)).toBeTruthy();
  });

  test("warns when shortcuts come from reading input devices", async () => {
    mockHotkeyStatus = hotkeyStatus({
      backend: "evdev",
      is_private: false,
      portal_error: "no such interface",
      detail: "FotonVoice Engine is reading input devices directly.",
    });

    render(HotkeysTab);

    expect(await screen.findByText("Reading input devices directly")).toBeTruthy();
  });

  test("flags shortcuts KDE registered but left disabled, with a button to fix it", async () => {
    mockHotkeyStatus = hotkeyStatus({
      needs_manual_enable: true,
      manual_enable_hint:
        "KDE registers FotonVoice Engine's shortcuts disabled by default (KDE bug 483639). Open Shortcut Settings, tick the box next to each FotonVoice Engine shortcut, and press Apply.",
    });

    render(HotkeysTab);

    expect(await screen.findByText(/One more step on KDE/i)).toBeTruthy();
    expect(screen.queryByText(/KDE bug 483639/i)).toBeNull();

    const toggle = await screen.findByRole("button", {
      name: /Toggle manual shortcut enable details/i,
    });
    await fireEvent.click(toggle);

    expect(await screen.findByText(/KDE bug 483639/i)).toBeTruthy();
    const btn = await screen.findByRole("button", { name: /Open Shortcut Settings/i });
    await fireEvent.click(btn);

    expect(openShortcutSettingsCalls).toBe(1);

    await fireEvent.click(toggle);
    expect(screen.queryByText(/KDE bug 483639/i)).toBeNull();
  });

  test("shows an error if opening shortcut settings fails", async () => {
    mockHotkeyStatus = hotkeyStatus({
      needs_manual_enable: true,
      manual_enable_hint: "Open Shortcut Settings and enable FotonVoice Engine's shortcuts.",
    });
    openShortcutSettingsResult = "Could not find a way to open your desktop's shortcut settings automatically.";

    render(HotkeysTab);

    const toggle = await screen.findByRole("button", {
      name: /Toggle manual shortcut enable details/i,
    });
    await fireEvent.click(toggle);

    const btn = await screen.findByRole("button", { name: /Open Shortcut Settings/i });
    await fireEvent.click(btn);

    expect(await screen.findByText(/Could not find a way to open/i)).toBeTruthy();
  });

  test("does not show the manual-enable notice when the desktop needs no extra step", async () => {
    render(HotkeysTab);

    expect(screen.queryByText(/One more step on KDE/i)).toBeNull();
  });

  test("does not show conflict warnings when there are no conflicts", async () => {
    mockBindings = [
      {
        id: "bind1",
        keys: ["KEY_LEFTMETA", "KEY_SPACE"],
        gesture: "hold",
        target_id: "default",
        target_ids: ["default"],
        tap_ms: 300,
        hold_threshold_ms: 1000,
        label: "Binding 1",
        disabled: false,
      },
      {
        id: "bind2",
        keys: ["KEY_LEFTMETA", "KEY_ENTER"],
        gesture: "hold",
        target_id: "default",
        target_ids: ["default"],
        tap_ms: 300,
        hold_threshold_ms: 1000,
        label: "Binding 2",
        disabled: false,
      },
    ];

    render(HotkeysTab);

    const banner = screen.queryByText(/Conflict detected/i);
    expect(banner).toBeNull();

    const marker = screen.queryByText("CONFLICT");
    expect(marker).toBeNull();
  });

  test("shows active conflicts with yellow background and CONFLICT marker when both are enabled", async () => {
    mockBindings = [
      {
        id: "bind1",
        keys: ["KEY_LEFTMETA", "KEY_SPACE"],
        gesture: "hold",
        target_id: "default",
        target_ids: ["default"],
        tap_ms: 300,
        hold_threshold_ms: 1000,
        label: "Binding 1",
        disabled: false,
      },
      {
        id: "bind2",
        keys: ["KEY_SPACE", "KEY_LEFTMETA"], // Same keys, different order
        gesture: "hold",
        target_id: "default",
        target_ids: ["default"],
        tap_ms: 300,
        hold_threshold_ms: 1000,
        label: "Binding 2",
        disabled: false,
      },
    ];

    const { container } = render(HotkeysTab);

    const banner = await screen.findByText(/Conflict detected/i);
    expect(banner).not.toBeNull();

    const markers = await screen.findAllByText("CONFLICT");
    expect(markers.length).toBe(2);

    const conflictItems = container.querySelectorAll(".active-conflict");
    expect(conflictItems.length).toBe(2);
  });

  test("shows conflict borders but no CONFLICT markers or active-conflict background when one is disabled", async () => {
    mockBindings = [
      {
        id: "bind1",
        keys: ["KEY_LEFTMETA", "KEY_SPACE"],
        gesture: "hold",
        target_id: "default",
        target_ids: ["default"],
        tap_ms: 300,
        hold_threshold_ms: 1000,
        label: "Binding 1",
        disabled: false,
      },
      {
        id: "bind2",
        keys: ["KEY_SPACE", "KEY_LEFTMETA"],
        gesture: "hold",
        target_id: "default",
        target_ids: ["default"],
        tap_ms: 300,
        hold_threshold_ms: 1000,
        label: "Binding 2",
        disabled: true, // One is disabled
      },
    ];

    const { container } = render(HotkeysTab);

    const banner = await screen.findByText(/Conflict detected/i);
    expect(banner).not.toBeNull();

    const marker = screen.queryByText("CONFLICT");
    expect(marker).toBeNull();

    const hasConflictItems = container.querySelectorAll(".has-conflict");
    expect(hasConflictItems.length).toBe(2);

    const activeConflictItems = container.querySelectorAll(".active-conflict");
    expect(activeConflictItems.length).toBe(0);
  });

  test("shows LLM badge when openai_enabled is true", async () => {
    mockBindings = [
      {
        id: "bind1",
        keys: ["KEY_LEFTMETA", "KEY_SPACE"],
        gesture: "hold",
        target_id: "default",
        target_ids: ["default"],
        tap_ms: 300,
        hold_threshold_ms: 1000,
        label: "Binding 1",
        disabled: false,
        openai_enabled: true,
      },
    ];

    render(HotkeysTab);

    const badge = await screen.findByText(/^LLM$/);
    expect(badge).not.toBeNull();
  });

  test("does not show LLM badge when openai_enabled is false", async () => {
    mockBindings = [
      {
        id: "bind1",
        keys: ["KEY_LEFTMETA", "KEY_SPACE"],
        gesture: "hold",
        target_id: "default",
        target_ids: ["default"],
        tap_ms: 300,
        hold_threshold_ms: 1000,
        label: "Binding 1",
        disabled: false,
        openai_enabled: false,
      },
    ];

    render(HotkeysTab);

    const badge = screen.queryByText(/^LLM$/);
    expect(badge).toBeNull();
  });

  test("opens nested Target modal and cancels to revert select value", async () => {
    mockBindings = [
      {
        id: "bind1",
        keys: ["KEY_LEFTMETA", "KEY_SPACE"],
        gesture: "hold",
        target_id: "default",
        target_ids: ["default"],
        tap_ms: 300,
        hold_threshold_ms: 1000,
        label: "Binding 1",
        disabled: false,
      },
    ];

    const { container } = render(HotkeysTab);

    const editBtn = await screen.findByRole("button", { name: /Edit/i });
    await fireEvent.click(editBtn);

    expect(screen.getByText("Edit Hotkey Binding")).not.toBeNull();

    const trigger = container.querySelector(".custom-select-trigger") as HTMLButtonElement;
    expect(trigger).not.toBeNull();
    expect(trigger.textContent).toContain("Focused Window");

    await fireEvent.click(trigger);

    const createBtn = screen.getByText("Create New Target");
    expect(createBtn).not.toBeNull();
    await fireEvent.click(createBtn);

    expect(await screen.findByText("Create Target")).not.toBeNull();

    const targetIdInput = screen.queryByPlaceholderText("e.g. obsidian_vault");
    expect(targetIdInput).toBeNull();

    const cancelButtons = screen.getAllByRole("button", { name: /Cancel/i });
    await fireEvent.click(cancelButtons[1]);

    expect(screen.queryByText("Create Target")).toBeNull();

    expect(trigger.textContent).toContain("Focused Window");
  });

  test("opens nested Target modal, creates a new target, and auto-selects it", async () => {
    mockBindings = [
      {
        id: "bind1",
        keys: ["KEY_LEFTMETA", "KEY_SPACE"],
        gesture: "hold",
        target_id: "default",
        target_ids: ["default"],
        tap_ms: 300,
        hold_threshold_ms: 1000,
        label: "Binding 1",
        disabled: false,
      },
    ];

    const { container } = render(HotkeysTab);

    const editBtn = await screen.findByRole("button", { name: /Edit/i });
    await fireEvent.click(editBtn);

    const trigger = container.querySelector(".custom-select-trigger") as HTMLButtonElement;
    expect(trigger).not.toBeNull();

    await fireEvent.click(trigger);
    
    const createBtn = screen.getByText("Create New Target");
    await fireEvent.click(createBtn);

    expect(await screen.findByText("Create Target")).not.toBeNull();

    const labelInput = screen.getByPlaceholderText("e.g. Obsidian Notes");
    await fireEvent.input(labelInput, { target: { value: "My Nested Target" } });

    const doneButtons = screen.getAllByRole("button", { name: /Done/i });
    await fireEvent.click(doneButtons[1]);

    expect(screen.queryByText("Create Target")).toBeNull();

    expect(trigger.textContent).toContain("My Nested Target (inject)");
  });

  test("saving an existing binding with unchanged keys does not trigger a false key-validation error", async () => {
    mockBindings = [
      {
        id: "bind1",
        keys: ["KEY_LEFTMETA", "KEY_SPACE"],
        gesture: "hold",
        target_id: "default",
        tap_ms: 300,
        hold_threshold_ms: 200,
        label: "Old Name",
        disabled: false,
      },
    ];
    mockKeysChecks = [rejected("This should not appear")];

    render(HotkeysTab);
    await fireEvent.click(await screen.findByRole("button", { name: /Edit/i }));

    const labelInput = screen.getByDisplayValue("Old Name");
    await fireEvent.input(labelInput, { target: { value: "New Name" } });

    await fireEvent.click(await screen.findByRole("button", { name: /^Done$/i }));

    expect(screen.queryByText(/not accepted/i)).toBeNull();
    expect(keysCheckCalls).toHaveLength(0);
  });

  test("re-capturing the exact same keys clears any stale error from a previous capture", async () => {
    mockBindings = [
      {
        id: "bind1",
        keys: ["KEY_LEFTMETA", "KEY_SPACE"],
        gesture: "hold",
        target_id: "default",
        tap_ms: 300,
        hold_threshold_ms: 200,
        label: "Dictate",
        disabled: false,
      },
    ];
    mockKeysChecks = [rejected("Your desktop cannot register this shortcut.")];

    const { container } = render(HotkeysTab);
    await fireEvent.click(await screen.findByRole("button", { name: /Edit/i }));

    const recorder = container.querySelector('[aria-label="Base Hotkey recorder input"]')!;

    await fireEvent.focus(recorder);
    await fireEvent.keyDown(recorder, { key: "Meta", code: "MetaLeft" });
    await fireEvent.keyUp(recorder, { key: "Meta", code: "MetaLeft" });
    expect(await screen.findByText(/not accepted/i)).toBeTruthy();

    await fireEvent.focus(recorder);
    await fireEvent.keyDown(recorder, { key: "Meta", code: "MetaLeft" });
    await fireEvent.keyDown(recorder, { key: " ", code: "Space" });
    await fireEvent.keyUp(recorder, { key: " ", code: "Space" });

    expect(screen.queryByText(/not accepted/i)).toBeNull();
    expect(keysCheckCalls).toHaveLength(1);
  });

  describe("gesture styles the running backend cannot deliver", () => {
    const ALL_GESTURE_LABELS = [
      /Hold keys to dictate/i,
      /Tap once to start recording/i,
      /Double-tap hotkey to trigger/i,
      /Double-tap & hold keys/i,
    ];

    function toggleBinding() {
      return [
        {
          id: "bind1",
          keys: ["KEY_LEFTCTRL", "KEY_LEFTALT", "KEY_D"],
          gesture: "toggle",
          target_id: "default",
          tap_ms: 300,
          hold_threshold_ms: 200,
          label: "Dictate",
          disabled: false,
        },
      ] as HotkeyBinding[];
    }

    async function openGestureOptions(container: HTMLElement): Promise<string[]> {
      const field = Array.from(container.querySelectorAll("label")).find(l =>
        l.textContent?.includes("Input Gesture Style"),
      );
      if (!field) throw new Error("gesture field not rendered");
      const trigger = field.querySelector("button.custom-select-trigger");
      if (!trigger) throw new Error("gesture dropdown has no trigger");
      await fireEvent.click(trigger);
      return Array.from(field.querySelectorAll("button.custom-dropdown-item")).map(
        b => b.textContent?.trim() ?? "",
      );
    }

    test("offers every style when the backend can deliver every style", async () => {
      mockBindings = toggleBinding();
      mockHotkeyStatus = hotkeyStatus({ backend: "x11", is_private: false });

      const { container } = render(HotkeysTab);
      await fireEvent.click(await screen.findByRole("button", { name: /Edit/i }));
      const options = await openGestureOptions(container);

      expect(options).toHaveLength(4);
      for (const label of ALL_GESTURE_LABELS) {
        expect(options.some(o => label.test(o))).toBe(true);
      }
    });

    test("hides hold and double-tap when the desktop only reports key presses", async () => {
      mockBindings = toggleBinding();
      mockHotkeyStatus = hotkeyStatus({
        backend: "mint_dbus",
        supported_gestures: ["toggle"],
      });

      const { container } = render(HotkeysTab);
      await fireEvent.click(await screen.findByRole("button", { name: /Edit/i }));
      const options = await openGestureOptions(container);

      expect(options).toHaveLength(1);
      expect(options[0]).toMatch(/Tap once to start recording/i);
    });

    test("says why the missing styles are missing", async () => {
      mockBindings = toggleBinding();
      mockHotkeyStatus = hotkeyStatus({
        backend: "mint_dbus",
        supported_gestures: ["toggle"],
      });

      render(HotkeysTab);
      await fireEvent.click(await screen.findByRole("button", { name: /Edit/i }));

      expect(await screen.findByText(/never coming back up/i)).toBeTruthy();
    });

    test("keeps a saved binding's unsupported gesture visible instead of rewriting it", async () => {
      mockBindings = [{ ...toggleBinding()[0], gesture: "hold" }];
      mockHotkeyStatus = hotkeyStatus({
        backend: "mint_dbus",
        supported_gestures: ["toggle"],
      });

      const { container } = render(HotkeysTab);
      await fireEvent.click(await screen.findByRole("button", { name: /Edit/i }));
      const options = await openGestureOptions(container);

      expect(options).toHaveLength(2);
      expect(options.some(o => /not supported on this system/i.test(o))).toBe(true);
      expect(screen.queryByText(/will not fire/i)).toBeTruthy();
    });

    test("falls back to offering everything when the backend reports nothing", async () => {
      mockBindings = toggleBinding();
      mockHotkeyStatus = hotkeyStatus({ supported_gestures: [] });

      const { container } = render(HotkeysTab);
      await fireEvent.click(await screen.findByRole("button", { name: /Edit/i }));
      const options = await openGestureOptions(container);

      expect(options).toHaveLength(4);
    });
  });
});
