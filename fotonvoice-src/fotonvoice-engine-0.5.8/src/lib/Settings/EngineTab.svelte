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

  const MODEL_SIZES = [
    "small",
    "small.en",
    "medium",
    "medium.en",
    "large-v3",
    "large-v3-turbo",
    "small-q8",
    "small.en-q8",
    "medium-q8",
    "medium.en-q8",
    "large-v3-turbo-q8",
  ];

  // GPU support is decided when the binary is compiled, per engine, and the two
  // answers differ in the build most people run: the Vulkan build accelerates
  // whisper.cpp and leaves Moonshine on the CPU, because ONNX Runtime has no
  // Vulkan execution provider. Defaults assume no GPU, so an older backend that
  // does not answer this call is described as CPU-only rather than as more than
  // it is.
  let whisperGpu = $state<string | null>(null);
  let moonshineGpu = $state<string | null>(null);
  let parakeetGpu = $state<string | null>(null);
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
    {
      value: "moonshine",
      label: moonshineGpu
        ? `Moonshine (${gpuLabel(moonshineGpu)})`
        : "Moonshine (CPU only)",
    },
    {
      value: "parakeet",
      label: parakeetGpu
        ? `Parakeet TDT (${gpuLabel(parakeetGpu)})`
        : "Parakeet TDT (CPU only)",
    },
    {
      value: "nemotron-streaming",
      label: nemotronStreamingGpu
        ? `Nemotron Streaming (${gpuLabel(nemotronStreamingGpu)})`
        : "Nemotron Streaming (CPU only)",
    },
    { value: "remote-openai", label: "Remote Speech Engine (OpenAI API)" },
  ]);

  let whisperModelSizeOptions = $derived(
    MODEL_SIZES.map(s => ({
      value: s,
      label: `${s}${s.endsWith("-q8") ? "" : " q5"}${whisperModels.downloaded[s] ? " +" : ""}`
    }))
  );

  // Only what this build can actually do. Offering "Vulkan" unconditionally -
  // as this list used to - meant a CUDA build and a CPU-only build both showed
  // a Vulkan option that selecting changed nothing about: the device setting
  // says *whether* to offload, and ggml links exactly one backend to offload
  // to. So there is at most one GPU entry, named after the one in the build.
  let deviceOptions = $derived([
    { value: "auto", label: whisperGpu ? `Auto (${gpuLabel(whisperGpu)})` : "Auto" },
    ...(whisperGpu ? [{ value: whisperGpu, label: gpuLabel(whisperGpu) }] : []),
    { value: "cpu", label: "CPU" },
  ]);

  const moonshineModelSizeOptions = [
    { value: "base", label: "Base" },
    { value: "tiny", label: "Tiny" }
  ];

  const parakeetModelSizeOptions = [
    { value: "tdt-0.6b-v3", label: "TDT 0.6B v3 (INT8, ~665 MB, CPU-lean)" },
    { value: "tdt-0.6b-v3-fp32", label: "TDT 0.6B v3 (FP32, ~2.4 GB, GPU-accel)" }
  ];

  // -- Nemotron streaming ----------------------------------------------------
  // English-only cache-aware streaming engine; fp16 is the GPU-offload
  // preference, int8-static (QDQ) runs on CUDA and Intel VNNI CPUs.
  const nemotronModelSizeOptions = [
    { value: "fp16", label: "FP16 (~1.2 GB, GPU-accel)" },
    { value: "int8-static", label: "INT8 static (~876 MB, CUDA / CPU)" }
  ];

  // -- Model managers -----------------------------------------------------------
  // One state machine per backend, defined in models.svelte.ts. Each one owns
  // its presence map, its single download/delete lock, and its error string;
  // nothing below re-implements any of that.
  const whisperModels = createModelManager({
    sizes: MODEL_SIZES,
    check: (size) =>
      invoke<boolean>("check_model_downloaded", {
        modelSize: size,
        modelDir: cfg.engine.whisper_cpp.model_dir,
      }),
    download: (size) =>
      invoke("download_model", { modelSize: size, modelDir: cfg.engine.whisper_cpp.model_dir }),
    remove: (size) =>
      invoke("delete_model", { modelSize: size, modelDir: cfg.engine.whisper_cpp.model_dir }),
  });

  let modelDirError = $state<string | null>(null);

  // -- Moonshine ------------------------------------------------------------
  // Whether the app was built with the Moonshine backend. When false, choosing
  // Moonshine silently runs whisper-cpp, so we surface that to the user.
  let moonshineAvailable = $state(true);
  const moonshineModels = createModelManager({
    sizes: moonshineModelSizeOptions.map((o) => o.value),
    check: (size) => invoke<boolean>("check_moonshine_downloaded", { modelSize: size }),
    download: (size) => invoke("download_moonshine_model", { modelSize: size }),
    remove: (size) => invoke("delete_moonshine_model", { modelSize: size }),
  });

  // -- Parakeet -------------------------------------------------------------
  let parakeetAvailable = $state(true);
  let parakeetGpuPresent = $state(true);
  const parakeetModels = createModelManager({
    sizes: parakeetModelSizeOptions.map((o) => o.value),
    check: (size) => invoke<boolean>("check_parakeet_downloaded", { modelSize: size }),
    download: (size) => invoke("download_parakeet_model", { modelSize: size }),
    remove: (size) => invoke("delete_parakeet_model", { modelSize: size }),
  });

  // -- Nemotron streaming ----------------------------------------------------
  let nemotronAvailable = $state(true);
  let nemotronGpuPresent = $state(true);
  const nemotronModels = createModelManager({
    sizes: nemotronModelSizeOptions.map((o) => o.value),
    check: (size) => invoke<boolean>("check_nemotron_streaming_downloaded", { modelSize: size }),
    download: (size) => invoke("download_nemotron_streaming_model", { modelSize: size }),
    remove: (size) => invoke("delete_nemotron_streaming_model", { modelSize: size }),
  });

  async function onParakeetModelChanged() {
    markDirty();
    await parakeetModels.verify(cfg.engine.parakeet.model_size);
  }

  // -- Remote Speech Engine (OpenAI API) ------------------------------------
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

  async function onModelChanged() {
    markDirty();
    await whisperModels.verify(cfg.engine.whisper_cpp.model_size);
  }

  async function validateModelDir() {
    const path = cfg.engine.whisper_cpp.model_dir;
    if (!path) {
      modelDirError = null;
      return;
    }
    const exists = await invoke<boolean>("check_directory_exists", { path });
    modelDirError = exists
      ? null
      : "This folder does not exist. Please create it first or leave blank for the default location.";
    if (!modelDirError) {
      await whisperModels.refreshAll();
    }
  }

  function onModelDirChange() {
    markDirty();
  }

  function onModelDirKeydown(e: KeyboardEvent) {
    if (e.key === "Enter") {
      (e.currentTarget as HTMLInputElement).blur();
    }
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
      parakeetAvailable = await invoke<boolean>("parakeet_available");
    } catch (e) {
      console.error("Failed to query Parakeet availability", e);
    }
    parakeetModels.refreshAll();
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
        parakeet_gpu: string | null;
        parakeet_gpu_present: boolean;
        nemotron_streaming_gpu: string | null;
        nemotron_streaming_gpu_present: boolean;
        s1_mini_gpu: string | null;
      }>("accelerator_support");
      whisperGpu = support.whisper_gpu ?? null;
      moonshineGpu = support.moonshine_gpu ?? null;
      parakeetGpu = support.parakeet_gpu ?? null;
      parakeetGpuPresent = support.parakeet_gpu_present;
      nemotronStreamingGpu = support.nemotron_streaming_gpu ?? null;
      nemotronGpuPresent = support.nemotron_streaming_gpu_present;
      if (!parakeetGpuPresent && cfg.engine.parakeet?.gpu) {
        cfg.engine.parakeet.gpu = false;
        markDirty();
      }
      if (!nemotronGpuPresent && cfg.engine.nemotron_streaming?.gpu) {
        cfg.engine.nemotron_streaming.gpu = false;
        markDirty();
      }
    } catch (e) {
      console.error("Failed to query GPU support", e);
    }
    // A config naming a GPU backend this build does not have - copied from
    // another machine, or left behind by a switch between the CPU and Vulkan
    // downloads - would otherwise sit in the dropdown as a value with no entry,
    // reading as a working GPU setting while the backend quietly ran on the CPU.
    const device = cfg.engine.whisper_cpp.device;
    if (device !== "auto" && device !== "cpu" && device !== whisperGpu) {
      cfg.engine.whisper_cpp.device = "auto";
      markDirty();
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
        <span>Model size</span>
        <CustomSelect bind:value={cfg.engine.whisper_cpp.model_size} options={whisperModelSizeOptions} onchange={onModelChanged} />
      </label>

      <ModelStatusRow
        mgr={whisperModels}
        size={cfg.engine.whisper_cpp.model_size}
      />

      <label class="field">
        <span>Device</span>
        <CustomSelect bind:value={cfg.engine.whisper_cpp.device} options={deviceOptions} onchange={markDirty} />
      </label>
      <div class="field">
        <span>Model directory (leave blank for default)</span>
        <input
          type="text"
          bind:value={cfg.engine.whisper_cpp.model_dir}
          onchange={onModelDirChange}
          onblur={validateModelDir}
          onkeydown={onModelDirKeydown}
          class:field-input-error={!!modelDirError}
        />
        {#if modelDirError}
          <p class="field-error-msg">{modelDirError}</p>
        {/if}
      </div>
      <label class="field">
        <span>Threads (0 = auto)</span>
        <input
          type="number"
          min="0"
          max="64"
          bind:value={cfg.engine.whisper_cpp.threads}
          onchange={markDirty}
        />
      </label>
      <label class="field">
        <span>Language</span>
        <input
          type="text"
          bind:value={cfg.engine.whisper_cpp.language}
          placeholder="auto"
          onchange={markDirty}
        />
      </label>
      <p class="hint">
        Language code (e.g. <code>en</code>, <code>da</code>, <code>fr</code>) to force transcription in that
        language, or <code>auto</code> to let whisper.cpp detect it. Forcing a language helps short phrases
        that auto-detect sometimes misidentifies.
      </p>
      <p class="hint">
        Default model directory: <code>~/.local/share/fotonvoice-engine/models/</code>
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
              This build was compiled without Moonshine, so selecting it will fall
              back to Whisper.cpp (using the Whisper model configured above).
              Rebuild with <code>--features moonshine</code> to enable it.
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
              ONNX Runtime, which Moonshine uses, has no Vulkan backend, so the
              Device setting above applies to Whisper.cpp only. Moonshine holds
              its weights in RAM as fp32 - roughly <code>530&nbsp;MB</code> for
              <code>base</code>, <code>240&nbsp;MB</code> for <code>tiny</code> -
              where Whisper.cpp{whisperGpu
                ? ` puts a quantized model in VRAM via ${gpuLabel(whisperGpu)}`
                : " uses a smaller quantized model"}. Pick Whisper.cpp if memory
              matters more than Moonshine's latency.
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
  {:else if cfg.engine.backend === "parakeet"}
    <div class="field-group">
      <h3>Parakeet Settings</h3>

      {#if !parakeetAvailable}
        <div
          class="flex items-center gap-4 bg-yellow-500/10 border border-yellow-500/30 rounded-xl p-4 mb-4"
        >
          <span class="text-2xl leading-none text-yellow-500">!</span>
          <div class="flex-1">
            <strong class="block text-yellow-200 font-semibold text-sm mb-1"
              >Parakeet backend not included in this build</strong
            >
            <p class="m-0 text-slate-200 text-xs leading-relaxed">
              This build was compiled without Parakeet, so selecting it will fall
              back to Whisper.cpp. Rebuild with <code>--features parakeet</code> to enable it.
            </p>
          </div>
        </div>
      {/if}

      <label class="field">
        <span>Model size</span>
        <CustomSelect bind:value={cfg.engine.parakeet.model_size} options={parakeetModelSizeOptions} onchange={onParakeetModelChanged} />
      </label>

      {#if parakeetAvailable}
        <ModelStatusRow
          mgr={parakeetModels}
          size={cfg.engine.parakeet.model_size}
        />
        <p class="hint" style="margin-top: 6px;">
          Model source: <a class="credit-name-link" href="https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx" target="_blank" rel="noreferrer">istupakov/parakeet-tdt-0.6b-v3-onnx</a>
        </p>
      {/if}

      <label class="field">
        <span>GPU acceleration</span>
        <input
          type="checkbox"
          bind:checked={cfg.engine.parakeet.gpu}
          onchange={markDirty}
          disabled={!parakeetGpuPresent}
        />
      </label>
      <p class="hint">
        {#if parakeetGpuPresent}
          Attach the GPU execution provider (WebGPU, CUDA or CoreML per platform) when
          one is available. Off keeps the GPU free for other processes. INT8 graphs are
          CPU-lean and dequantize on the GPU; FP32 runs natively on the GPU but needs
          the larger download. Changing either reloads the model.
        {:else}
          No GPU acceleration is available in this build or on this machine, so the
          toggle is disabled and Parakeet runs on the CPU.
        {/if}
      </p>

      <label class="field">
        <span>Language</span>
        <input
          type="text"
          bind:value={cfg.engine.parakeet.language}
          placeholder="auto"
          onchange={markDirty}
        />
      </label>
      <p class="hint">
        NVIDIA FastConformer TDT 0.6B with 128-mel ONNX preprocessor. INT8 (~665 MB)
        or FP32 (~2.4 GB) graph export, 25 European languages.
      </p>
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
              This build was compiled without the Nemotron streaming backend, so selecting it
              will fall back to Whisper.cpp. Rebuild with
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

  .field-input-error {
    @apply border-red-500!;
  }
  .field-input-error:focus {
    @apply border-red-500 shadow-[0_0_0_2px_rgba(239,68,68,0.15),_inset_0_2px_4px_rgba(0,0,0,0.2)];
  }
  .field-error-msg {
    @apply mt-1 text-sm leading-5 text-red-400;
  }

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

</style>
