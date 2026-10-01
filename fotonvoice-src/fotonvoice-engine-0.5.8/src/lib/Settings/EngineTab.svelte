<script lang="ts">
  import type { AppConfig } from "../../stores/config";
  import { config, configDirty } from "../../stores/config";
  import { invoke } from "@tauri-apps/api/core";
  import { onMount } from "svelte";

  import CustomSelect from "./CustomSelect.svelte";
  import ModelStatusRow from "./ModelStatusRow.svelte";
  import { createModelManager } from "./models.svelte";

  let { cfg = $bindable() } = $props<{ cfg: AppConfig }>();
  function markDirty() {
    config.set(cfg);
    configDirty.set(true);
  }

  let moonshineGpu = $state<string | null>(null);
  let nemotronStreamingGpu = $state<string | null>(null);

  const GPU_LABELS: Record<string, string> = {
    cuda: "CUDA (NVIDIA)",
    vulkan: "Vulkan (AMD/Intel/NVIDIA)",
    coreml: "CoreML (Apple)",
    webgpu: "WebGPU (AMD/Intel/NVIDIA)",
  };
  const gpuLabel = (id: string) => GPU_LABELS[id] ?? id;

  let backendOptions = $derived([
    {
      value: "moonshine",
      label: moonshineGpu
        ? `Moonshine (${gpuLabel(moonshineGpu)})`
        : "Moonshine (CPU only)",
    },
    {
      value: "nemotron-streaming",
      label: nemotronStreamingGpu
        ? `Nemotron Streaming (${gpuLabel(nemotronStreamingGpu)})`
        : "Nemotron Streaming (CPU only)",
    },
    { value: "remote-openai", label: "Remote Speech Engine (OpenAI API)" },
  ]);

  const moonshineModelSizeOptions = [
    { value: "base", label: "Base" },
    { value: "tiny", label: "Tiny" }
  ];

  const nemotronModelSizeOptions = [
    { value: "fp16", label: "FP16 (~1.2 GB, GPU-accel)" },
    { value: "int8-static", label: "INT8 static (~876 MB, CUDA / CPU)" }
  ];

  let moonshineAvailable = $state(true);
  const moonshineModels = createModelManager({
    sizes: moonshineModelSizeOptions.map((o) => o.value),
    check: (size) => invoke<boolean>("check_moonshine_downloaded", { modelSize: size }),
    download: (size) => invoke("download_moonshine_model", { modelSize: size }),
    remove: (size) => invoke("delete_moonshine_model", { modelSize: size }),
  });

  let nemotronAvailable = $state(true);
  let nemotronGpuPresent = $state(true);
  const nemotronModels = createModelManager({
    sizes: nemotronModelSizeOptions.map((o) => o.value),
    check: (size) => invoke<boolean>("check_nemotron_streaming_downloaded", { modelSize: size }),
    download: (size) => invoke("download_nemotron_streaming_model", { modelSize: size }),
    remove: (size) => invoke("delete_nemotron_streaming_model", { modelSize: size }),
  });

  interface RemoteSttTestResult {
    success: boolean;
    message: string;
    models: string[];
  }

  let remoteTesting = $state(false);
  let remoteTestStatus = $state<{ success: boolean; message: string } | null>(null);
  let remoteDiscoveredModels = $state<string[]>([]);

  function ensureRemoteConfig() {
    if (!cfg.engine.remote_openai) {
      cfg.engine.remote_openai = {
        endpoint: "http://localhost:8000/v1",
        api_key: null,
        model: "whisper-1",
        language: "",
        timeout_secs: 30,
      };
    }
  }

  async function testRemoteConnection() {
    ensureRemoteConfig();
    if (remoteTesting) return;
    remoteTesting = true;
    remoteTestStatus = null;
    try {
      const res = await invoke<RemoteSttTestResult>("test_remote_stt", {
        endpoint: cfg.engine.remote_openai.endpoint || "http://localhost:8000/v1",
        apiKey: cfg.engine.remote_openai.api_key || null,
        model: cfg.engine.remote_openai.model || "whisper-1",
        timeoutSecs: cfg.engine.remote_openai.timeout_secs || 30,
      });
      remoteTestStatus = { success: res.success, message: res.message };
      if (res.models && res.models.length > 0) {
        remoteDiscoveredModels = res.models;
        if (!cfg.engine.remote_openai.model) {
          cfg.engine.remote_openai.model = res.models[0];
          markDirty();
        }
      }
    } catch (e: any) {
      remoteTestStatus = { success: false, message: e.toString() };
    } finally {
      remoteTesting = false;
    }
  }

  async function onMoonshineModelChanged() {
    markDirty();
    await moonshineModels.verify(cfg.engine.moonshine.model_size);
  }

  async function onNemotronModelChanged() {
    markDirty();
    await nemotronModels.verify(cfg.engine.nemotron_streaming.model_size);
  }

  onMount(async () => {
    try {
      moonshineAvailable = await invoke<boolean>("moonshine_available");
    } catch (e) {
      console.error("Failed to query Moonshine availability", e);
    }
    moonshineModels.refreshAll();
    try {
      nemotronAvailable = await invoke<boolean>("nemotron_streaming_available");
    } catch (e) {
      console.error("Failed to query Nemotron streaming availability", e);
    }
    nemotronModels.refreshAll();
    try {
      const support = await invoke<{
        moonshine_gpu: string | null;
        nemotron_streaming_gpu: string | null;
        nemotron_streaming_gpu_present: boolean;
        s1_mini_gpu: string | null;
      }>("accelerator_support");
      moonshineGpu = support.moonshine_gpu ?? null;
      nemotronStreamingGpu = support.nemotron_streaming_gpu ?? null;
      nemotronGpuPresent = support.nemotron_streaming_gpu_present;
      if (!nemotronGpuPresent && cfg.engine.nemotron_streaming?.gpu) {
        cfg.engine.nemotron_streaming.gpu = false;
        markDirty();
      }
    } catch (e) {
      console.error("Failed to query GPU support", e);
    }
  });
</script>

<section>
  <h2>Inference Engine</h2>

  <div class="field-group">
    <h3>Backend</h3>
    <label class="field">
      <span>Backend</span>
      <CustomSelect bind:value={cfg.engine.backend} options={backendOptions} onchange={markDirty} />
    </label>
  </div>

  {#if cfg.engine.backend === "whisper-cpp"}
    <div class="field-group">
      <h3>Whisper.cpp</h3>
      <p class="hint">
        The whisper.cpp backend is deactivated in this version - it cannot run
        or download models. Pick another backend above.
      </p>
    </div>
  {:else if cfg.engine.backend === "moonshine"}
    <div class="field-group">
      <h3>Moonshine Settings</h3>

      {#if !moonshineAvailable}
        <div
          class="flex items-center gap-4 bg-yellow-500/10 border border-yellow-500/30 rounded-xl p-4 mb-4"
        >
          <span class="text-2xl leading-none text-yellow-500">!</span>
          <div class="flex-1">
            <strong class="block text-yellow-200 font-semibold text-sm mb-1"
              >Moonshine backend not included in this build</strong
            >
            <p class="m-0 text-slate-200 text-xs leading-relaxed">
              This build was compiled without Moonshine, so selecting it will
              report the backend as unavailable. Rebuild with
              <code>--features moonshine</code> to enable it.
            </p>
          </div>
        </div>
      {/if}

      {#if moonshineAvailable && !moonshineGpu}
        <div
          class="flex items-center gap-4 bg-slate-500/10 border border-slate-500/30 rounded-xl p-4 mb-4"
        >
          <span class="text-2xl leading-none text-slate-300"></span>
          <div class="flex-1">
            <strong class="block text-slate-100 font-semibold text-sm mb-1"
              >Moonshine runs on the CPU in this build</strong
            >
            <p class="m-0 text-slate-200 text-xs leading-relaxed">
              ONNX Runtime, which Moonshine uses, has no Vulkan backend, so
              Moonshine runs on the CPU in Vulkan builds. It holds its weights
              in RAM as fp32 - roughly <code>530&nbsp;MB</code> for
              <code>base</code>, <code>240&nbsp;MB</code> for <code>tiny</code>.
            </p>
          </div>
        </div>
      {/if}

      <label class="field">
        <span>Model size</span>
        <CustomSelect bind:value={cfg.engine.moonshine.model_size} options={moonshineModelSizeOptions} onchange={onMoonshineModelChanged} />
      </label>

      {#if moonshineAvailable}
        <ModelStatusRow
          mgr={moonshineModels}
          size={cfg.engine.moonshine.model_size}
        />
      {/if}

      <label class="field">
        <span>Language</span>
        <input
          type="text"
          bind:value={cfg.engine.moonshine.language}
          onchange={markDirty}
        />
      </label>
    </div>
  {:else if cfg.engine.backend === "nemotron-streaming"}
    <div class="field-group">
      <h3>Nemotron Streaming Settings</h3>

      {#if !nemotronAvailable}
        <div
          class="flex items-center gap-4 bg-yellow-500/10 border border-yellow-500/30 rounded-xl p-4 mb-4"
        >
          <span class="text-2xl leading-none text-yellow-500">!</span>
          <div class="flex-1">
            <strong class="block text-yellow-200 font-semibold text-sm mb-1"
              >Nemotron streaming backend not included in this build</strong
            >
            <p class="m-0 text-slate-200 text-xs leading-relaxed">
              This build was compiled without the Nemotron streaming backend, so
              selecting it will report the backend as unavailable. Rebuild with
              <code>--features nemotron-streaming</code> to enable it.
            </p>
          </div>
        </div>
      {/if}

      <label class="field">
        <span>Precision variant</span>
        <CustomSelect bind:value={cfg.engine.nemotron_streaming.model_size} options={nemotronModelSizeOptions} onchange={onNemotronModelChanged} />
      </label>

      {#if nemotronAvailable}
        <ModelStatusRow
          mgr={nemotronModels}
          size={cfg.engine.nemotron_streaming.model_size}
        />
        <p class="hint" style="margin-top: 6px;">
          Model source: <a class="credit-name-link" href="https://huggingface.co/danielbodart/nemotron-speech-600m-onnx" target="_blank" rel="noreferrer">danielbodart/nemotron-speech-600m-onnx</a>
        </p>
      {/if}

      <label class="field">
        <span>GPU acceleration</span>
        <input
          type="checkbox"
          bind:checked={cfg.engine.nemotron_streaming.gpu}
          onchange={markDirty}
          disabled={!nemotronGpuPresent}
        />
      </label>
      <p class="hint">
        {#if nemotronGpuPresent}
          Attach the GPU execution provider (WebGPU, CUDA or CoreML per platform) when
          one is available. Off keeps the GPU free for other processes. FP16 is the
          GPU-offload preference; INT8 static (QDQ) also runs on CUDA and Intel VNNI
          CPUs. Changing either reloads the model.
        {:else}
          No GPU acceleration is available in this build or on this machine, so the
          toggle is disabled and the engine runs on the CPU.
        {/if}
      </p>

      <p class="hint">
        NVIDIA Nemotron Speech Streaming 0.6B (Mar 2026 checkpoint): cache-aware
        FastConformer-RNNT, English only. Streams with true incremental decoding -
        partial text appears while you speak, final text pastes on stop. 560ms chunks.
      </p>
    </div>
  {:else if cfg.engine.backend === "remote-openai"}
    {@const _ = ensureRemoteConfig()}
    <div class="field-group">
      <h3>Remote Speech Engine Settings</h3>
      <p class="hint">
        Connect to a network speech-to-text service supporting the OpenAI
        <code>/v1/audio/transcriptions</code> API (e.g., Faster-Whisper-Server, vLLM, LocalAI, Whisper-standalone, or OpenAI Whisper).
        Audio will begin streaming to the endpoint as soon as your keybind is triggered.
      </p>

      <label class="field">
        <span>Endpoint URL</span>
        <input
          type="text"
          bind:value={cfg.engine.remote_openai.endpoint}
          placeholder="http://192.168.1.50:8000/v1"
          onchange={markDirty}
        />
      </label>
      <p class="hint">
        Base URL or full transcriptions path (e.g. <code>http://192.168.1.50:8000/v1</code> or <code>https://api.openai.com/v1</code>).
      </p>

      <label class="field">
        <span>API Key (optional)</span>
        <input
          type="password"
          bind:value={cfg.engine.remote_openai.api_key}
          placeholder="Bearer token or leave blank"
          onchange={markDirty}
        />
      </label>
      <p class="hint">
        Leave blank if your local network server does not require authentication.
      </p>

      <label class="field">
        <span>Model</span>
        <input
          type="text"
          bind:value={cfg.engine.remote_openai.model}
          placeholder="whisper-1"
          onchange={markDirty}
        />
      </label>
      {#if remoteDiscoveredModels.length > 0}
        <div class="discovered-models">
          <span class="hint">Discovered models from server:</span>
          <div class="model-tags">
            {#each remoteDiscoveredModels as m}
              <button
                type="button"
                class="tag-btn"
                class:active={cfg.engine.remote_openai.model === m}
                onclick={() => {
                  cfg.engine.remote_openai.model = m;
                  markDirty();
                }}
              >
                {m}
              </button>
            {/each}
          </div>
        </div>
      {/if}

      <label class="field">
        <span>Language (optional)</span>
        <input
          type="text"
          bind:value={cfg.engine.remote_openai.language}
          placeholder="auto"
          onchange={markDirty}
        />
      </label>
      <p class="hint">
        Language code (e.g. <code>en</code>, <code>es</code>, <code>fr</code>) or leave blank/auto for automatic detection.
      </p>

      <label class="field">
        <span>Timeout (seconds)</span>
        <input
          type="number"
          min="5"
          max="300"
          bind:value={cfg.engine.remote_openai.timeout_secs}
          onchange={markDirty}
        />
      </label>

      <div class="test-row">
        <button
          type="button"
          class="btn-test"
          onclick={testRemoteConnection}
          disabled={remoteTesting}
        >
          {#if remoteTesting}
            ... Testing Connection...
          {:else}
             Test Connection
          {/if}
        </button>

        {#if remoteTestStatus}
          <div
            class="status-pill"
            class:success={remoteTestStatus.success}
            class:error={!remoteTestStatus.success}
          >
            {#if remoteTestStatus.success}
              + {remoteTestStatus.message}
            {:else}
              ! {remoteTestStatus.message}
            {/if}
          </div>
        {/if}
      </div>
    </div>
  {/if}
</section>

<style>
  @reference "../../app.css";

  .test-row {
    @apply flex items-center gap-3 mt-4 flex-wrap;
  }
  .btn-test {
    @apply bg-[var(--accent)] hover:bg-[var(--accent2)] text-white text-xs font-semibold py-2 px-4 rounded-[var(--radius)] border-none cursor-pointer transition-colors duration-200 disabled:opacity-50 disabled:cursor-not-allowed;
  }
  .status-pill {
    @apply text-xs px-3 py-1.5 rounded-[var(--radius)] font-medium leading-normal;
  }
  .status-pill.success {
    @apply bg-emerald-500/15 text-emerald-300 border border-emerald-500/30;
  }
  .status-pill.error {
    @apply bg-red-500/15 text-red-300 border border-red-500/30;
  }
  .discovered-models {
    @apply flex items-center gap-2 flex-wrap mb-3;
  }
  .model-tags {
    @apply flex items-center gap-1.5 flex-wrap;
  }
  .tag-btn {
    @apply text-xs px-2 py-0.5 rounded bg-[var(--bg)] hover:bg-[var(--border)] border border-[var(--border)] text-[var(--text-muted)] hover:text-white cursor-pointer transition-colors;
  }
  .tag-btn.active {
    @apply border-[var(--accent)] text-[var(--accent)] bg-[var(--accent)]/10;
  }
  .credit-name-link {
    @apply text-[13px] font-normal text-white no-underline transition-colors duration-150 ease-out;
  }
  .credit-name-link:hover {
    @apply text-[var(--color-accent-blue)] underline;
  }

</style>
