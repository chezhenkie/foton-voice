use std::fs::File;
use std::io::Write;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use tracing::{Event, Subscriber};
use tracing_subscriber::layer::Context;
use tracing_subscriber::Layer;

pub static STARTUP_COMPLETE: AtomicBool = AtomicBool::new(false);

/// Record panics to `<app_root>/crash.log`.
///
/// The release binary is a Windows GUI subsystem app, so there is no stderr to
/// read and `tracing` never sees a panic. Without this a crash leaves no trace
/// at all beyond whatever happened to be logged before it.
pub fn install_panic_hook(crash_path: std::path::PathBuf) {
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        use std::io::Write;
        let thread = std::thread::current();
        let name = thread.name().unwrap_or("<unnamed>").to_string();
        let location = info
            .location()
            .map(|l| format!("{}:{}:{}", l.file(), l.line(), l.column()))
            .unwrap_or_else(|| "<unknown location>".to_string());
        let payload = info
            .payload()
            .downcast_ref::<&str>()
            .map(|s| s.to_string())
            .or_else(|| info.payload().downcast_ref::<String>().cloned())
            .unwrap_or_else(|| "<non-string panic payload>".to_string());

        let report = format!(
            "{}\n--- PANIC ---\nthread: {}\nlocation: {}\npayload: {}\n",
            chrono::Utc::now().to_rfc3339(),
            name,
            location,
            payload
        );

        if let Ok(mut file) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&crash_path)
        {
            let _ = file.write_all(report.as_bytes());
            let _ = file.flush();
        }

        previous(info);
    }));
}

struct MessageVisitor {
    message: String,
}

impl tracing::field::Visit for MessageVisitor {
    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        if field.name() == "message" {
            self.message = format!("{:?}", value);
        }
    }

    fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
        if field.name() == "message" {
            self.message = value.to_string();
        }
    }
}

pub struct StartupErrorLayer {
    file: Mutex<File>,
}

impl StartupErrorLayer {
    pub fn new(path: std::path::PathBuf) -> std::io::Result<Self> {
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;
        Ok(Self {
            file: Mutex::new(file),
        })
    }
}

impl<S: Subscriber> Layer<S> for StartupErrorLayer {
    fn on_event(&self, event: &Event<'_>, _ctx: Context<'_, S>) {
        let metadata = event.metadata();
        let level = metadata.level();

        let mut visitor = MessageVisitor { message: String::new() };
        event.record(&mut visitor);

        let msg = visitor.message;

        // whisper.cpp/ggml log about the compute backend, not about user speech,
        // and its wording ("transcribe", "payload") collides with the privacy
        // keywords below. Filtering it would discard GPU failure diagnostics,
        // so it is exempt.
        let is_native_backend_log = metadata.target().starts_with("whisper_rs");
        // The capture thread's own diagnostics. Without this the audio INFO lines
        // are dropped once startup finishes, so "Using detected active audio
        // device" never appeared and a capture thread stuck inside device init
        // looked identical to one that had not started.
        let is_audio_log = metadata.target().starts_with("fotonvoice_audio");

        let lower_msg = msg.to_lowercase();
        if !is_native_backend_log
            && (lower_msg.contains("received transcription")
                || lower_msg.contains("delivered target_id")
                || lower_msg.contains("transcribe")
                || lower_msg.contains("transcription")
                || lower_msg.contains("speaking")
                || lower_msg.contains("speak")
                || lower_msg.contains("openai")
                || lower_msg.contains("payload")
                || lower_msg.contains("status-tick"))
        {
            return;
        }

        let is_error_or_warn = *level == tracing::Level::ERROR || *level == tracing::Level::WARN;
        let is_startup = !STARTUP_COMPLETE.load(Ordering::SeqCst);
        let is_whisper_diagnostic =
            msg.starts_with("Whisper acceleration:") || msg.starts_with("Whisper run:");
        // Same gate for the ONNX engines. Without this their INFO lines are
        // dropped once startup finishes, so "Moonshine acceleration:", the
        // per-graph load timings and the run line never reached the log at all -
        // which is why a stalled load and a completed one looked identical.
        let is_engine_diagnostic = is_whisper_diagnostic
            || msg.starts_with("Inference engine ready")
            || msg.starts_with("Inference backend changed")
            || msg.starts_with("STT audio:")
            || msg.starts_with("STT request:")
            || msg.starts_with("Moonshine acceleration:")
            || msg.starts_with("Moonshine load:")
            || msg.starts_with("Moonshine run:")
            || msg.starts_with("Nemotron streaming acceleration:")
            || msg.starts_with("Nemotron streaming load:")
            || msg.starts_with("Nemotron streaming run:");
        // whisper.cpp initialises the GPU backend during the lazy model load,
        // which happens after STARTUP_COMPLETE, so its INFO lines must not be
        // dropped by the startup gate.
        if is_error_or_warn
            || ((is_startup || is_engine_diagnostic || is_native_backend_log || is_audio_log)
                && *level == tracing::Level::INFO)
        {
            let timestamp = chrono::Utc::now().to_rfc3339();
            let log_line = format!(
                "{} [{}] {}: {}\n",
                timestamp,
                level,
                metadata.target(),
                msg
            );
            if let Ok(mut file) = self.file.lock() {
                let _ = file.write_all(log_line.as_bytes());
                let _ = file.flush();
            }
        }
    }
}
