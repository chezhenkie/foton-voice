# Architecture

## High-Level Design

FotonVoice Engine is a **Tauri 2** application: a compiled Rust backend that spawns a WebView window running a Svelte SPA. The two halves communicate via Tauri's IPC bridge (invoke commands + event emitters).

```
+---------------------------------------------------------+
|                    Tauri Desktop App                     |
|                                                          |
|  +-------------------+      +--------------------------+ |
|  |   Svelte Frontend |<---->|    Rust Backend (lib.rs) | |
|  |   (WebView)       | IPC  |    + crates workspace     | |
|  +-------------------+      +--------------------------+ |
+---------------------------------------------------------+
         |                              |
   Settings                      Audio devices,
   windows                       Filesystem, DBus,
                                 Network (OpenAI API/HTTP)
```

In addition, the backend opens a second **`WebviewWindow`** (`window::open_overlay_window` in `src-tauri/src/window.rs`) that renders the always-on-top, click-through recording HUD by loading the `/overlay` Svelte route - the same `Overlay.svelte` component tree used for the style preview in Settings. It gets its state through the same app-wide `status-tick` / `audio-level` Tauri events every other window uses, rather than a separate protocol; see `docs/overlays.md` for the window-management details (click-through, always-on-top reassertion, Wayland/XWayland).

---

## Crate Workspace

The backend is organized as a Cargo workspace of focused, single-responsibility crates:

```
fotonvoice-engine/
+-- src-tauri/         # Tauri app entry + IPC command handlers
|   +-- src/
|       +-- main.rs    # App bootstrap
|       +-- lib.rs     # Pipeline coordinator
|       +-- commands.rs# Tauri #[command] handlers
|       +-- state.rs   # Shared AppState
|
+-- crates/
    +-- fotonvoice-config/     # AppConfig struct, TOML/JSON persistence
    +-- fotonvoice-audio/      # Microphone capture, resampling, VU meter
    +-- fotonvoice-hotkeys/    # Global shortcuts (XDG portal / evdev / Windows hook / D-Bus)
    +-- fotonvoice-inference/  # whisper.cpp/Moonshine/Parakeet/Remote STT + post-processing
    +-- fotonvoice-routing/    # OutputTarget + HotkeyBinding data models, 11-target router
    +-- fotonvoice-inject/     # Text injection orchestrator (Wayland/X11/Windows)
    +-- fotonvoice-winput/     # Windows native synthesised Unicode keyboard input
    +-- fotonvoice-tts/        # Neural & local TTS (Breeze/Piper/Pocket/Inflect/VoxCPM2/eSpeak)
    +-- fotonvoice-mcp/        # MCP JSON-RPC server (Unix socket / named pipe)
    +-- fotonvoice-dbus/       # DBus service (Linux session bus)
    +-- fotonvoice-llm/        # OpenAI-compatible LLM HTTP client
    +-- fotonvoice-text/       # Shared snippet expansion + fuzzy vocab correction
    +-- fotonvoice-update/     # GitHub release check, download, verify, self-replace
    +-- fotonvoice-llm-sidecar/# Vulkan-accelerated llama.cpp sidecar for S1-mini dictation cleanup
```

---

## Fundamental Rule (binding, above every other rule)

We aim for minimum latency, maximum quality and robustness, and maximum speed - with the resources a given machine provides. No rule shall ever hinder that.

Standalone and modularity are the aim, but they are means, not ends: they must never become dogma when they hinder these basic design goals.

Worked example of what this rule prevents: the Parakeet backend shipped for a while with the int8 graphs hardwired. int8 is "portable" - it runs on any CPU - but WebGPU and CUDA cannot execute int8 ops, so the encoder silently fell back to CPU nodes or dequantized to fp32 and the GPU bought nothing.
A portability convenience had quietly overridden the actual design goal: latency, speed and quality on the hardware present.
The fix, not an exception: every engine must expose the precision variants its model format offers (int8 / fp32 / fp16), a per-engine GPU toggle, and the freedom to use or refuse each device - because deployment machines range from Intel iGPU laptops to 16 GB RTX rigs, and the GPU must be usable when wanted and freeable when other processes need it.

---

## Why Several Inference Runners (not everything through audio.cpp)

This is a locked design decision: we stick with the current engine split. Each
model runs through the runner its format and interaction pattern demands.

- **Model format rules the runner.** audio.cpp is a GGUF runner. Whisper is
  natively GGUF, so it keeps its own whisper.cpp stack. Parakeet and Moonshine
  are ONNX graphs (conformer / RNN-T / hybrid encoder architectures that
  audio.cpp does not implement); converting them to GGUF would quantize
  conformer encoders lossily. ONNX Runtime executes them directly, int8 or
  fp16, on the same runtime DLL the app already ships for WebGPU - no extra
  dependency, no per-engine native builds.
- **Latency profile.** Dictation wants first-token speed. In-process ONNX
  sessions give zero process-spawn overhead and let the pipeline feed mic
  frames directly into a session - the control level the Phase 2 streaming
  loop needs. audio.cpp is built around blocking generation calls, which fits
  TTS batches but not a live partials stream.
- **audio.cpp is not excluded - it owns the big GGUF lane.** VoxCPM2, Breeze
  and Pocket-TTS already run through audio.cpp as child processes with Vulkan
  acceleration, and the same route is the planned path for larger STT models
  that outgrow the fp16 parakeet family on 16 GB RTX cards (near-real-time
  batches rather than streaming).
- **GPU peculiarities are per-runtime, not per-model.** Vulkan pairs with the
  audio.cpp GGUF path; WebGPU pairs with onnxruntime.dll and dequantizes int8
  to fp32 (so prefer fp16 for GPU offload); CUDA only exists in the whisper.cpp
  CUDA artifact, and whisper.cpp stays CPU by decision. Each runner carries
  exactly the device flags its backend understands, which is why GPU offload is
  a per-engine config knob rather than one global switch.

Small and fast engines run in-process; heavy GGUF engines run through
audio.cpp. This split stays.

---

## Data Flow

```
Hotkey press
     |
     v
fotonvoice-hotkeys --gesture_tx--> lib.rs coordinator
                                      |
                         +------------+
                         |            |
                  Start AudioRecorder  Determine target from binding
                  (fotonvoice-audio)
                         |
                    audio_tx chunks
                         |
                         v
                  Audio Accumulator
                  (lib.rs buffer)
                         |
                  Hotkey release / VAD stop
                         |
                    inference_tx
                         |
                         v
                  InferenceEngine.process()
                  (fotonvoice-inference)
                    |  Noise gate (VAD)
                    |  Speech transcription (whisper.cpp / Moonshine / Parakeet / Remote)
                    |  Filler removal
                    |  Spoken punctuation
                    |  Auto-format lists
                    |  Snippet expansion
                    |  Custom vocab fuzzy correction
                    |  Code mode
                    |  Silence hallucination filter
                    |  LLM rewrite via OpenAI API (optional, per-target)
                    |  S1-mini text normalization via fotonvoice-llm-sidecar
                         |
                    text_tx (InferenceOutput)
                         |
                         v
                  OutputTargetRouter.route()
                  (fotonvoice-routing)
                    +-- inject -> fotonvoice-inject / fotonvoice-winput
                    +-- clipboard -> arboard
                    +-- file -> tokio::fs
                    +-- http/webhook -> reqwest
                    +-- exec -> std::process
                    +-- socket -> UnixStream / TcpStream
                    +-- dbus -> fotonvoice-dbus
                    +-- mcp -> fotonvoice-mcp response queue
                    +-- pipe -> named FIFO
                    +-- speak -> fotonvoice-tts
                    +-- chat -> fotonvoice-llm conversational loop
                         |
                    Tauri event -> frontend
                    (status-tick)
```

---

## Concurrency Model

FotonVoice Engine uses Tokio for async I/O plus dedicated OS threads for latency-sensitive work:

| Thread / Task | Type | Purpose |
|---|---|---|
| Main Tauri thread | OS thread | Window management, IPC dispatch |
| Audio capture | OS thread (cpal) | Microphone streaming at hardware rate |
| Audio level emitter | OS thread | Forwards RMS levels to the UI and the overlay, coalesced to one per frame (16 ms); silent unless recording or monitoring |
| Hotkey listener | async task + OS threads | XDG GlobalShortcuts portal, with an evdev/Win32 fallback |
| Inference worker | OS thread | Blocking Whisper computation; `WhisperState` (KV cache + attention buffers) is allocated once at load and reused across all calls |
| Status ticker | Tokio task | Ticks at 150 ms to animate the tray icon; emits `status-tick` only when the state changed, plus a 900 ms heartbeat |
| Config watcher | Tokio task | `inotify`/`kqueue` on config files |
| MCP server | Tokio task | Unix socket accept loop |
| DBus service | Tokio task | Session bus method handler |
| TTS FIFO watcher | Tokio task | Named pipe reader for TTS input |

**Shared state** is an `Arc<AppState>` with `AtomicBool`/`AtomicU32` for hot-path flags and `Mutex` for heavier data (targets, TTS handle).

**Channels** (crossbeam/tokio):
- `audio_tx` / `audio_rx` - `Vec<f32>` chunks
- `inference_tx` / `inference_rx` - `InferenceRequest`
- `text_tx` / `text_rx` - `InferenceOutput`
- `audio_level_tx` / `level_rx` - `f32` RMS
- `gesture_tx` / `gesture_rx` - `GestureEvent`
- `hotkey_reloader` - updated bindings list sent to listener thread (hot-reload)

---

## Frontend Architecture

The Svelte frontend is a single-page app with three logical "pages" rendered as separate Tauri windows:

```
App.svelte  (route switcher)
  +-- /settings  -> Settings component (sidebar with 9 tabs)
  |     +-- GeneralTab
  |     +-- EngineTab
  |     +-- RoutingTab     (targets + bindings editor)
  |     +-- VisualTab
  |     +-- AudioTab
  |     +-- TtsTab
  |     +-- FeaturesTab
  |     +-- OpenAiTab  (labeled "OpenAI API")
  |     +-- AboutTab
  |
  +-- /wizard    -> SetupWizard component (first-run setup, 7 steps)
  |     +-- WelcomeStep
  |     +-- EngineStep     (engine + model, downloads before continuing)
  |     +-- HotkeyStep     (gesture + key capture, desktop registration)
  |     +-- OverlayStep    (style + position, bundled webm previews)
  |     +-- TestStep       (live end-to-end dictation)
  |     +-- VoiceStep      (optional TTS engine, per-card download)
  |     +-- DoneStep       (summary + anything that failed)
  |
  +-- /overlay   -> Overlay component (the on-screen recording HUD, loaded into
  |     |          its own WebviewWindow - see "High-Level Design" above)
  |     +-- BlueWave       (default - "Ocean Wave" tide pool)
  |     +-- VoiceCard      ("Voice Card" VU LED matrix card)
  |     +-- Waveform       (green-phosphor oscilloscope)
  |     +-- Pulse          ("Pulse Ring" sonar dial)
  |
```

**State management:**
- `src/stores/config.ts` - reactive `AppConfig` with 400ms debounced auto-save via `save_config` IPC; also listens for `config-changed` events
- `src/stores/status.ts` - live state from `status-tick` events + derived stores (`recording`, `speaking`, `wordCount`, `activeTargetLabel`). Falls back to polling `get_status` once a second when ticks have been absent for two, so a window the events do not reach still shows live state; while ticks are arriving it does no IPC at all

---

## File Locations

| Path | Contents |
|---|---|
| `~/.config/fotonvoice-engine/config.json` | Main application config |
| `~/.config/fotonvoice-engine/targets.toml` | Output target definitions |
| `~/.config/fotonvoice-engine/bindings.toml` | Hotkey binding definitions |
| `~/.local/share/fotonvoice-engine/models/` | Downloaded Whisper GGUF models |
| `~/.local/share/fotonvoice-engine/piper-voices/` | Downloaded Piper voice packs |
| `~/.local/share/fotonvoice-engine/models/inflect-micro/` | Inflect-Micro-v2 ONNX graphs and symbol list |
| `~/.local/share/fotonvoice-engine/models/lux-tts/` | LuxTTS ONNX graphs (text encoder, flow decoder, vocos vocoder) and token vocabulary |
| `~/.local/share/fotonvoice-engine/cloned-tts-voices/` | Reference voice clips (.wav) with paired transcripts (.txt) and encoded `.luxtprompt` caches |
| `/tmp/fotonvoice-mcp.sock` | MCP Unix domain socket (Linux) |
| `\\.\pipe\fotonvoice-mcp` | MCP named pipe (Windows) |
