
export type ModelRowState = "checking" | "downloading" | "present" | "missing";

export interface ModelManagerOpts {
  sizes: readonly string[];
  check: (size: string) => Promise<boolean>;
  download: (size: string) => Promise<unknown>;
  remove: (size: string) => Promise<unknown>;
}

const VERIFY_FAIL =
  "Download finished but the model file failed its on-disk check. Please try again.";

export function createModelManager(opts: ModelManagerOpts) {
  let downloaded = $state<Record<string, boolean>>({});
  let checking = $state(false);
  let op = $state<{ kind: "download" | "delete"; size: string } | null>(null);
  let error = $state<string | null>(null);

  async function verify(size: string): Promise<boolean> {
    try {
      return await opts.check(size);
    } catch {
      return false;
    }
  }

  async function refreshAll(): Promise<void> {
    checking = true;
    error = null;
    try {
      const results = await Promise.all(
        opts.sizes.map(async (s) => [s, await verify(s)] as const),
      );
      downloaded = Object.fromEntries(results);
    } finally {
      checking = false;
    }
  }

  async function download(size: string): Promise<void> {
    if (op !== null || checking) return;
    op = { kind: "download", size };
    error = null;
    try {
      await opts.download(size);
      downloaded[size] = await verify(size);
      if (!downloaded[size]) error = VERIFY_FAIL;
    } catch (e) {
      error = `${e}`;
      downloaded[size] = await verify(size);
    } finally {
      op = null;
    }
  }

  async function remove(size: string): Promise<void> {
    if (op !== null || checking) return;
    op = { kind: "delete", size };
    error = null;
    try {
      await opts.remove(size);
    } catch (e) {
      error = `${e}`;
    } finally {
      op = null;
    }
    downloaded[size] = await verify(size);
  }

  function state(size: string): ModelRowState {
    if (checking) return "checking";
    if (op !== null && op.kind === "download" && op.size === size) {
      return "downloading";
    }
    return downloaded[size] ? "present" : "missing";
  }

  return {
    get downloaded() {
      return downloaded;
    },
    get checking() {
      return checking;
    },
    get busy() {
      return checking || op !== null;
    },
    get error() {
      return error;
    },
    verify,
    refreshAll,
    download,
    remove,
    state,
  };
}

export type ModelManager = ReturnType<typeof createModelManager>;