# Third-Party Notices

FotonVoice Engine is MIT-licensed (see LICENSE). This file covers the
third-party components it builds on or vendors, as of 2026-09-22.

The exact crate versions in any build are recorded in `Cargo.lock`, npm
versions in `package-lock.json` (where present). Licenses below were read
from the crates.io and npm registries at the versions listed. All code
dependencies below are permissive licenses; none are copyleft.

## Rust crates (direct dependencies, locked versions)

| Crate | Version | License |
|---|---|---|
| anyhow | 1.0.102 | MIT OR Apache-2.0 |
| arboard | 3.6.1 | MIT OR Apache-2.0 |
| ashpd | 0.13.13 | MIT |
| base64 | 0.22.1 | MIT OR Apache-2.0 |
| bytes | 1.11.1 | MIT |
| candle-core | 0.9.2 | MIT OR Apache-2.0 |
| candle-transformers | 0.9.2 | MIT OR Apache-2.0 |
| chrono | 0.4.44 | MIT OR Apache-2.0 |
| cpal | 0.15.3 | Apache-2.0 |
| crossbeam-channel | 0.5.15 | MIT OR Apache-2.0 |
| dirs | 5.0.1 | MIT OR Apache-2.0 |
| flate2 | 1.1.9 | MIT OR Apache-2.0 |
| futures-util | 0.3.32 | MIT OR Apache-2.0 |
| hex | 0.4.3 | MIT OR Apache-2.0 |
| hmac | 0.12.1 | MIT OR Apache-2.0 |
| libc | 0.2.186 | MIT OR Apache-2.0 |
| ndarray | 0.16.1 | MIT OR Apache-2.0 |
| notify-rust | 4.17.0 | MIT OR Apache-2.0 |
| num_cpus | 1.17.0 | MIT OR Apache-2.0 |
| ort | 2.0.0-rc.12 | MIT OR Apache-2.0 |
| realfft | 3.5.0 | MIT |
| regex | 1.12.3 | MIT OR Apache-2.0 |
| reqwest | 0.12.28 | MIT OR Apache-2.0 |
| rodio | 0.17.3 | MIT OR Apache-2.0 |
| rubato | 0.15.0 | MIT |
| rustfft | 6.4.1 | MIT OR Apache-2.0 |
| serde | 1.0.228 | MIT OR Apache-2.0 |
| serde_json | 1.0.151 | MIT OR Apache-2.0 |
| sha2 | 0.10.9 | MIT OR Apache-2.0 |
| tar | 0.4.46 | MIT OR Apache-2.0 |
| tauri | 2.11.2 | Apache-2.0 OR MIT |
| tauri-plugin-dialog | 2.7.1 | Apache-2.0 OR MIT |
| tauri-plugin-fs | 2.5.1 | Apache-2.0 OR MIT |
| tauri-plugin-notification | 2.4.0 | Apache-2.0 OR MIT |
| tauri-plugin-shell | 2.3.5 | Apache-2.0 OR MIT |
| tauri-plugin-single-instance | 2.4.2 | Apache-2.0 OR MIT |
| tempfile | 3.27.0 | MIT OR Apache-2.0 |
| thiserror | 2.0.18 | MIT OR Apache-2.0 |
| tokenizers | 0.21.4 | Apache-2.0 |
| tokio | 1.52.3 | MIT |
| tokio-util | 0.7.18 | MIT |
| toml | 0.8.23 | MIT OR Apache-2.0 |
| tracing | 0.1.44 | MIT |
| tracing-subscriber | 0.3.23 | MIT |
| whisper-rs | 0.16.0 | Unlicense |
| zbus | 4.4.0 | MIT |
| zip | 2.4.2 | MIT |

Build-time-only: `cc` 1.2.62 (MIT OR Apache-2.0), `tauri-build` 2.6.2
(Apache-2.0 OR MIT). Test/dev only: `hound` 3.5.1 (Apache-2.0).

Linux builds additionally link `gtk` 0.18.2 (Rust bindings, MIT); the GTK
toolkit underneath is LGPL-2.1 and provided by the host system at runtime,
not bundled.

A complete dump of every transitive dependency with full license text can be
generated on demand with `cargo about` against `Cargo.lock`.

## Native runtime libraries

- **ONNX Runtime** (Microsoft) - MIT. Loaded at runtime; builds using the
  `ort` crate's prebuilt binaries fetch it at build time, GPU builds ship
  `onnxruntime.dll`/`libonnxruntime.so` beside the executable.
- **whisper.cpp** (ggml-org) - MIT, wrapped by whisper-rs (Unlicense).
- **eSpeak NG** - required at runtime by the Inflect-Micro and LuxTTS
  phonemizers; used as an external command-line binary on the host PATH, not
  bundled or modified. eSpeak NG is GPL-3.0; consult its source for the exact
  terms that apply to the binary you install.

## npm packages

| Package | License |
|---|---|
| @tauri-apps/api | Apache-2.0 OR MIT |
| @tauri-apps/plugin-dialog | MIT OR Apache-2.0 |
| @tauri-apps/plugin-fs | MIT OR Apache-2.0 |
| @tauri-apps/plugin-notification | MIT OR Apache-2.0 |
| @tauri-apps/plugin-shell | MIT OR Apache-2.0 |
| @tauri-apps/cli | Apache-2.0 OR MIT |
| @sveltejs/vite-plugin-svelte | MIT |
| @tailwindcss/vite | MIT |
| @testing-library/svelte | MIT |
| jsdom | MIT |
| svelte | MIT |
| svelte-check | MIT |
| tailwindcss | MIT |
| typescript | Apache-2.0 |
| vite | MIT |
| vitest | MIT |

## Vendored data assets

- `crates/fotonvoice-inference/assets/moonshine_tokenizer.json` - the
  Moonshine ASR tokenizer, vendored from the Moonshine project
  (<https://github.com/moonshine-ai/moonshine>), which ships it with the
  model packages on Hugging Face. Moonshine is Apache-2.0; the tokenizer
  data is redistributed here under those terms. The file is identical
  across all Moonshine model sizes and is embedded in the binary (see
  `crates/fotonvoice-inference/src/moonshine.rs`).

Speech models and TTS voices are NOT bundled: they are downloaded on
explicit user command from their upstream hosts, and their licensing is
governed by those hosts (see `docs/privacy.md` for what is downloaded when).
