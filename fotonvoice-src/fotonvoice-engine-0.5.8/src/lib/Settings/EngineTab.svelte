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

  let whisperGpu = $state<string | null>(null);
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
    { value: "whisper-cpp", label: "Whisper.cpp" },
    { value: "moonshine", label: "Moonshine" },
    { value: "nemotron-streaming", label: "Nemotron Streaming" },
    { value: "remote-openai", label: "Remote Speech Engine (OpenAI API)" },
  ]);

// Sizes are decimal MB/GB, matching what the downloads report. Largest first
  // within each engine so the size difference is obvious at a glance.
  // Whisper sizes: ggerganov/whisper.cpp ggml file sizes.
  const whisperModelSizeOptions = [
    { value: "large-v3-turbo-q8", label: "Large-v3-turbo - Q8_0 - 874 MB (best quality)" },
    { value: "large-v3-turbo", label: "Large-v3-turbo - Q5_0 - 574 MB (fastest)" },
    { value: "large-v3", label: "Large-v3 - Q5_0 - 1.08 GB (most accurate)" }
  ];

  let whisperDeviceOptions = $derived.by(() => [
    { value: "auto", label: "Auto" },
    ...(whisperGpu === "cuda" ? [{ value: "cuda", label: "CUDA" }] : []),
    ...(whisperGpu === "vulkan" ? [{ value: "vulkan", label: "Vulkan" }] : []),
    { value: "cpu", label: "CPU only" },
  ]);

  let whisperModels = createModelManager({
    sizes: whisperModelSizeOptions.map((o) => o.value),
    check: (size) =>
      invoke<boolean>("check_whisper_model_downloaded", {
        modelSize: size,
        modelDir: cfg.engine.whisper_cpp.model_dir || null,
      }),
    download: (size) =>
      invoke("download_whisper_model", {
        modelSize: size,
        modelDir: cfg.engine.whisper_cpp.model_dir || null,
      }),
    remove: (size) =>
      invoke("delete_whisper_model", {
        modelSize: size,
        modelDir: cfg.engine.whisper_cpp.model_dir || null,
      }),
  });

  // Precision names mirror the upstream folder names (float / quantized) so the
  // label cannot drift from what is actually fetched. Sizes are encoder +
  // decoder, decimal MB, from the upstream file sizes. The upstream 4-bit tier
  // is deliberately not offered: its encoder is larger than the quantized one
  // (31.0 MB vs 20.5 MB) and its decoder is byte-identical, so it is bigger for
  // no benefit.
  const moonshineModelSizeOptions = [
    { value: "base", label: "Base - float - 247 MB" },
    { value: "base-quantized", label: "Base - quantized - 63 MB" },
    { value: "tiny", label: "Tiny - float - 109 MB" },
    { value: "tiny-quantized", label: "Tiny - quantized - 28 MB" }
  ];

  const nemotronModelSizeOptions = [
    { value: "fp16", label: "FP16 - 1.26 GB" },
    { value: "int8-static", label: "INT8 static - 918 MB" }
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

  async function onWhisperModelChanged() {
    markDirty();
    await whisperModels.verify(cfg.engine.whisper_cpp.model_size);
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
    whisperModels.refreshAll();
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
        whisper_gpu: string | null;
        moonshine_gpu: string | null;
        nemotron_streaming_gpu: string | null;
        nemotron_streaming_gpu_present: boolean;
        s1_mini_gpu: string | null;
      }>("accelerator_support");
      whisperGpu = support.whisper_gpu ?? null;
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
      <h3>Whisper.cpp Settings</h3>

      <label class="field">
        <span>Model</span>
        <CustomSelect
          bind:value={cfg.engine.whisper_cpp.model_size}
          options={whisperModelSizeOptions}
          defaultToFirst={true}
          onchange={onWhisperModelChanged}
        />
      </label>

      <ModelStatusRow mgr={whisperModels} size={cfg.engine.whisper_cpp.model_size} />
      <p class="hint" style="margin-top: 6px;">
        Model source: <a class="credit-name-link" href="https://huggingface.co/ggerganov/whisper.cpp" target="_blank" rel="noreferrer">ggerganov/whisper.cpp</a>
      </p>

      <label class="field">
        <span>Language</span>
        <input
          type="text"
          bind:value={cfg.engine.whisper_cpp.language}
          onchange={markDirty}
          placeholder="auto"
        />
      </label>

      <label class="field">
        <span>cpu/gpu/igpu</span>
        <CustomSelect
          bind:value={cfg.engine.whisper_cpp.device}
          options={whisperDeviceOptions}
          onchange={markDirty}
        />
      </label>
      <p class="hint">
        {#if whisperGpu}
          Auto offloads to {gpuLabel(whisperGpu)}. CPU only leaves the GPU free
          for other processes.
        {:else}
          This build has no GPU backend for whisper.cpp, so it runs on the CPU.
        {/if}
      </p>

      <label class="field">
        <span>Threads</span>
        <input
          type="number"
          min="1"
          bind:value={cfg.engine.whisper_cpp.threads}
          onchange={markDirty}
        />
      </label>
      <p class="hint">
        Large v3 Turbo Q8 is the best quality/speed point for dictation. Turbo Q5
        loads faster and uses less RAM. Large v3 Q5 is the most accurate but needs
        about 1 GB of weights.
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
              Moonshine uses ONNX Runtime, which has no Vulkan backend in this
              build. Weights stay in RAM, so the quantized variants below are much
              lighter than the float ones.
            </p>
          </div>
        </div>
      {/if}

      <label class="field">
        <span>GPU acceleration</span>
        <input
          type="checkbox"
          bind:checked={cfg.engine.moonshine.gpu}
          onchange={markDirty}
          disabled={!moonshineGpu}
        />
      </label>
      <p class="hint">
        {#if moonshineGpu}
          Attach the GPU execution provider ({gpuLabel(moonshineGpu)}) when one is
          available. Off keeps the GPU free for other processes and skips the
          provider entirely, so no WebGPU device is created. Note that the encoder
          and decoder graphs can end up on different providers: the Moonshine
          encoder uses an op the WebGPU backend rejects, so it falls back to the
          CPU while the decoder runs on the GPU. Changing this reloads the model.
        {:else}
          No GPU acceleration is available in this build or on this machine, so the
          toggle is disabled and the engine runs on the CPU.
        {/if}
      </p>

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
