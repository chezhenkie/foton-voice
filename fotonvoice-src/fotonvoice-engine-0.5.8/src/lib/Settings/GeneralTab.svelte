<script lang="ts">
  import type { AppConfig } from "../../stores/config";
  import { config, configDirty } from "../../stores/config";
  import { invoke } from "@tauri-apps/api/core";
  import { onMount } from "svelte";

  let { cfg = $bindable() } = $props<{ cfg: AppConfig }>();

  function markDirty() {
    config.set(cfg);
    configDirty.set(true);
  }

  /**
   * The single HuggingFace access token used by every gated model FotonVoice Engine can
   * download (Pocket-TTS, Breeze-TTS-2, VoxCPM2). Entering it here - or during
   * the setup wizard - writes the same `tts.hf_token` field, so the two stay
   * in sync automatically. A token exported as `HF_TOKEN` wins at download
   * time, so when the session has one it is shown here read-only and never
   * written to the config.
   */
  let envHfToken = $state<string | null>(null);
  const hfFromEnv = $derived(!!envHfToken);
  const hfTokenShown = $derived(envHfToken ?? cfg.tts.hf_token ?? "");

  function onHfTokenChanged(e: Event) {
    if (hfFromEnv) return;
    const val = (e.target as HTMLInputElement).value;
    cfg.tts.hf_token = val.trim() ? val.trim() : null;
    markDirty();
  }

  onMount(() => {
    invoke<string | null>("hf_token_env")
      .then((t) => (envHfToken = t && t.trim() ? t.trim() : null))
      .catch(() => (envHfToken = null));
  });
</script>

<section>
  <h2>General</h2>

  <div class="field-group">
    <h3>Hugging Face Access Token</h3>
    <div class="field">
      <span>Hugging Face access token</span>
      <input
        type="password"
        value={hfTokenShown}
        readonly={hfFromEnv}
        title={hfFromEnv ? "Set by the HF_TOKEN environment variable" : undefined}
        oninput={onHfTokenChanged}
      />
    </div>
    {#if hfFromEnv}
      <p class="hint">
        Using the <code>HF_TOKEN</code> environment variable. It takes precedence over a saved
        token and is not written to your config.
      </p>
    {/if}
    <p class="hint">
      Used by every gated model FotonVoice Engine can download - Pocket-TTS, Breeze-TTS-2, and VoxCPM2 in
      the TTS tab. Create a token at <code>huggingface.co/settings/tokens</code> and accept each
      model's license on its Hugging Face page before downloading. Enter it once here.
    </p>
  </div>


  <div class="field-group">
    <h3>MCP Server</h3>
    <label class="field">
      <span>Enable MCP JSON-RPC server</span>
      <input type="checkbox" bind:checked={cfg.mcp.server_enabled} onchange={markDirty} />
    </label>
    <label class="field">
      <span>Visual Feedback</span>
      <input type="checkbox" bind:checked={cfg.mcp.visual_feedback} onchange={markDirty} />
    </label>
    <label class="field">
      <span>Record timeout (seconds)</span>
      <input
        type="number"
        min="1"
        max="120"
        bind:value={cfg.mcp.record_timeout}
        onchange={markDirty}
      />
    </label>
    <p class="hint">
      How long <code>transcribe_voice</code> listens when the calling agent does not ask for a
      specific timeout. An explicit <code>timeout_seconds</code> in the tool call still wins.
    </p>
    <p class="hint">Socket: <code>/tmp/fotonvoice-mcp.sock</code> (Linux) / <code>\\.\pipe\fotonvoice-mcp</code> (Windows)</p>
  </div>
</section>

<style>
  @reference "../../app.css";

  .btn-action {
    @apply bg-[var(--surface2)] text-[var(--text)] border border-[var(--border)] rounded-[var(--radius)] p-1.5 px-3.5 text-xs font-semibold cursor-pointer transition-all duration-150 ease-out;
  }
  .btn-action:hover {
    @apply bg-[var(--border)] border-[var(--text-muted)];
  }

  .hint.error {
    @apply text-red-400;
  }

  .btn-action:disabled {
    @apply opacity-60 cursor-default;
  }

  .link {
    @apply text-[var(--color-accent-blue)] underline underline-offset-2 bg-transparent border-0 p-0 cursor-pointer text-inherit;
  }
</style>
