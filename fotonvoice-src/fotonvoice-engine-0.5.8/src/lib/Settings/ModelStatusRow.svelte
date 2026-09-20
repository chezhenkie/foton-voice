<script lang="ts">
  let {
    state = "missing",
    busy = false,
    error = null,
    onDownload = () => {},
    onDelete = () => {},
  }: {
    state: "checking" | "downloading" | "present" | "missing";
    busy?: boolean;
    error?: string | null;
    onDownload?: () => void;
    onDelete?: () => void;
  } = $props();

  const present = state === "present";
  const downloading = state === "downloading";
  const checking = state === "checking";
  const missing = state === "missing";
  const disabled = busy || checking;
</script>

<div class="model-status-container">
  <span
    class="status"
    class:present
    class:downloading
    class:missing
    class:checking
  >
    {#if checking}
      checking...
    {:else if downloading}
      model downloading
    {:else if present}
      model present
    {:else}
      model not installed
    {/if}
  </span>
  <div class="model-actions">
    <button
      class="btn btn-download"
      type="button"
      onclick={onDownload}
      disabled={disabled || present}
    >
      Download
    </button>
    <button
      class="btn btn-delete"
      type="button"
      onclick={onDelete}
      disabled={disabled || !present}
    >
      Delete
    </button>
  </div>
</div>
{#if error}
  <span class="status-error">{error}</span>
{/if}

<style>
  @reference "../../app.css";

  .model-status-container {
    @apply flex items-center justify-between gap-3 bg-[var(--bg)] border border-[var(--border)] rounded-[var(--radius)] p-2.5 px-3.5 text-[13px] min-h-[42px] mb-3;
  }
  .status {
    @apply font-semibold lowercase;
  }
  .status.present {
    @apply text-emerald-400;
  }
  .status.downloading,
  .status.missing {
    @apply text-red-400;
  }
  .status.checking {
    @apply text-[var(--text-muted)];
  }
  .model-actions {
    @apply flex items-center gap-2 flex-shrink-0;
  }
  .btn {
    @apply rounded-[var(--radius)] p-1.5 px-3 text-xs font-semibold transition-colors duration-200 border;
  }
  .btn-download {
    @apply bg-[var(--accent)] border-transparent text-white;
  }
  .btn-download:hover:not(:disabled) {
    @apply bg-[var(--accent2)];
  }
  .btn-delete {
    @apply bg-red-500/10 border-red-500/30 text-red-300;
  }
  .btn-delete:hover:not(:disabled) {
    @apply bg-red-500/25 text-red-200;
  }
  .btn:disabled {
    @apply opacity-40 cursor-not-allowed;
  }
  .status-error {
    @apply mt-1 text-xs leading-5 text-red-400 w-full;
  }
</style>