import { describe, test, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/svelte";
import CommandsTab from "../../src/lib/Settings/CommandsTab.svelte";

const invoke = vi.hoisted(() => vi.fn());

vi.mock("@tauri-apps/api/core", () => ({
  invoke: (cmd: string, args?: unknown) => invoke(cmd, args),
}));

function target(overrides: Record<string, unknown> = {}) {
  return {
    id: "notes",
    label: "Notes",
    delivery: "file",
    file_path: "~/notes.md",
    processing: {},
    ...overrides,
  };
}

describe("Output Commands tab", () => {
  beforeEach(() => {
    invoke.mockReset();
    invoke.mockImplementation(async (cmd: string) => {
      if (cmd === "get_targets") return [target()];
      if (cmd === "get_bindings") return [];
      return null;
    });
  });

  test("is named Output Commands, in the heading and on the add button", async () => {
    render(CommandsTab);

    expect(await screen.findByText("Output Commands")).toBeTruthy();
    expect(screen.getByText(/Add New Output Command/)).toBeTruthy();
  });

  test("explains how to trigger a command by voice", async () => {
    const { container } = render(CommandsTab);
    await screen.findByText("Output Commands");

    const note = container.querySelector(".usage-note") as HTMLElement;
    expect(note).toBeTruthy();

    const text = note.textContent ?? "";
    expect(text).toContain("FotonVoice Engine");
    expect(text.indexOf("FotonVoice Engine notes")).toBeGreaterThan(-1);
    expect(text).toMatch(/command's name/);
  });

  test("still lists the commands that are configured", async () => {
    render(CommandsTab);
    expect(await screen.findByText("Notes")).toBeTruthy();
  });
});
