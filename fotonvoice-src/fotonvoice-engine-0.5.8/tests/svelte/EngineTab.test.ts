import { describe, test, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, within, waitFor } from "@testing-library/svelte";
import { invoke } from "@tauri-apps/api/core";
import EngineTab from "../../src/lib/Settings/EngineTab.svelte";

vi.mock("@tauri-apps/api/core", () => ({
  invoke: vi.fn(async (cmd, args) => {
    if (cmd === "check_model_downloaded") {
      return args.modelSize === "base"; // mock base downloaded, others missing
    }
    return true;
  }),
}));

vi.mock("@tauri-apps/api/event", () => ({
  listen: vi.fn(async () => {
    return () => {};
  }),
}));

const mockConfig = {
  engine: {
    backend: "whisper-cpp",
    whisper_cpp: {
      model_dir: "",
      model_size: "large-v3", // missing
      device: "auto",
      threads: 0,
    },
    : {
      model_size: "base",
      language: "en",
    },
    nemotron_streaming: {
      model_size: "fp16",
      language: "en",
    },
  },
} as any;

describe("EngineTab.svelte model status", () => {
  test("shows the  model as installed when the backend is ", async () => {
    const Config = {
      ...mockConfig,
      engine: {
        ...mockConfig.engine,
        backend: "",
      },
    };
    render(EngineTab, { cfg: Config });

    expect(await screen.findByText("model installed")).not.toBeNull();
  });

  test("shows the Nemotron streaming model as installed when the backend is Nemotron streaming", async () => {
    const nemotronConfig = {
      ...mockConfig,
      engine: {
        ...mockConfig.engine,
        backend: "nemotron-streaming",
      },
    };
    render(EngineTab, { cfg: nemotronConfig });

    expect(await screen.findByText("model installed")).not.toBeNull();
  });
});

describe("EngineTab.svelte Backend selector", () => {
  /** Open the first CustomSelect (Backend) and read its option labels. */
  async function backendOptionLabels(cfg: any) {
    const { container } = render(EngineTab, { cfg });
    const trigger = container.querySelector(".custom-select-trigger") as HTMLElement;
    await fireEvent.click(trigger);
    const menu = trigger.parentElement as HTMLElement;
    return within(menu)
      .getAllByRole("button")
      .map(b => b.textContent?.trim())
      .filter(Boolean);
  }

  test("offers only the selectable backends: whisper.cpp is dormant, no auto-detect entry", async () => {
    const labels = await backendOptionLabels({ ...mockConfig });

    expect(labels).not.toContain("Whisper.cpp");
    expect(labels.some(l => /auto/i.test(l!))).toBe(false);
  });

  test("shows the selected backend in the trigger", async () => {
    const nemotronConfig = {
      ...mockConfig,
      engine: { ...mockConfig.engine, backend: "nemotron-streaming" },
    };
    const { container } = render(EngineTab, { cfg: nemotronConfig });
    const trigger = container.querySelector(".custom-select-trigger") as HTMLElement;
    expect(trigger.textContent).toContain("Nemotron");
  });
});
