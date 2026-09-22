//! Windows hotkey backend: a Win32 low-level keyboard hook.

use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

use tracing::{info, warn};
use fotonvoice_routing::HotkeyBinding;

use crate::win_keys::{keymap, SuppressPlan};
use crate::{gestures::GestureEngine, keys::KeyMatcher, Backend, GestureSender, ListenerHealth};

use windows_sys::Win32::Foundation::{CloseHandle, HANDLE, LPARAM, LRESULT, WPARAM};
use windows_sys::Win32::Security::{GetTokenInformation, TokenElevation, TOKEN_ELEVATION, TOKEN_QUERY};
use windows_sys::Win32::System::Threading::{
    GetCurrentProcess, OpenProcess, OpenProcessToken, PROCESS_QUERY_LIMITED_INFORMATION,
};
use windows_sys::Win32::UI::WindowsAndMessaging::{
    CallNextHookEx, DispatchMessageW, GetForegroundWindow, GetMessageW, GetWindowThreadProcessId,
    SetWindowsHookExW, TranslateMessage, UnhookWindowsHookEx, HC_ACTION, HHOOK, KBDLLHOOKSTRUCT,
    LLKHF_EXTENDED, MSG, WH_KEYBOARD_LL, WM_KEYDOWN, WM_KEYUP, WM_SYSKEYDOWN, WM_SYSKEYUP,
};

/// Marker on every keystroke FotonVoice Engine synthesises. Defined by `fotonvoice-winput`,
use fotonvoice_winput::INJECTED_TAG;

/// How long the hook may go unheard before it is presumed dead and reinstalled.
const WATCHDOG_SILENCE: Duration = Duration::from_secs(180);

/// How often the watchdog wakes. Besides checking hook silence, this tick
const WATCHDOG_TICK: Duration = Duration::from_secs(5);

/// One key transition as the hook saw it.
struct RawKey {
    id: usize,
    down: bool,
}


static EVENTS: Mutex<Option<crossbeam_channel::Sender<RawKey>>> = Mutex::new(None);
static PLAN: RwLock<Option<SuppressPlan>> = RwLock::new(None);
static PRESSED: [AtomicBool; keymap::NAMES.len()] =
    [const { AtomicBool::new(false) }; keymap::NAMES.len()];
static SWALLOWED: [AtomicBool; keymap::NAMES.len()] =
    [const { AtomicBool::new(false) }; keymap::NAMES.len()];
/// Monotonic count of events the hook has handled, read by the watchdog.
static SEEN: AtomicUsize = AtomicUsize::new(0);
/// Thread id of the pump, so the watchdog can ask it to exit.
static PUMP_THREAD: AtomicU32 = AtomicU32::new(0);

/// Sender cached outside the mutex for the hook's fast path.
static SENDER: RwLock<Option<crossbeam_channel::Sender<RawKey>>> = RwLock::new(None);

unsafe extern "system" fn hook_proc(code: i32, wparam: WPARAM, lparam: LPARAM) -> LRESULT {
    if code != HC_ACTION as i32 {
        return unsafe { CallNextHookEx(std::ptr::null_mut(), code, wparam, lparam) };
    }

    let info = unsafe { &*(lparam as *const KBDLLHOOKSTRUCT) };

    if info.dwExtraInfo == INJECTED_TAG {
        return unsafe { CallNextHookEx(std::ptr::null_mut(), code, wparam, lparam) };
    }

    let msg = wparam as u32;
    let down = matches!(msg, WM_KEYDOWN | WM_SYSKEYDOWN);
    let up = matches!(msg, WM_KEYUP | WM_SYSKEYUP);
    if !down && !up {
        return unsafe { CallNextHookEx(std::ptr::null_mut(), code, wparam, lparam) };
    }

    let extended = info.flags & LLKHF_EXTENDED != 0;
    let Some(id) = keymap::lookup(info.scanCode, extended, info.vkCode) else {
        return unsafe { CallNextHookEx(std::ptr::null_mut(), code, wparam, lparam) };
    };

    SEEN.fetch_add(1, Ordering::Relaxed);

    let repeat = down && PRESSED[id].load(Ordering::Relaxed);

    let swallow = if down {
        if repeat {
            SWALLOWED[id].load(Ordering::Relaxed)
        } else {
            let decided = match PLAN.try_read() {
                Ok(plan) => plan
                    .as_ref()
                    .map(|plan| plan.should_swallow(id, &PRESSED))
                    .unwrap_or(false),
                Err(_) => false,
            };
            SWALLOWED[id].store(decided, Ordering::Relaxed);
            decided
        }
    } else {
        SWALLOWED[id].swap(false, Ordering::Relaxed)
    };

    if down {
        PRESSED[id].store(true, Ordering::Relaxed);
    } else {
        PRESSED[id].store(false, Ordering::Relaxed);
    }

    if !repeat {
        if let Ok(guard) = SENDER.try_read() {
            if let Some(tx) = guard.as_ref() {
                let _ = tx.send(RawKey { id, down });
            }
        }
    }

    if swallow {
        return 1;
    }
    unsafe { CallNextHookEx(std::ptr::null_mut(), code, wparam, lparam) }
}


pub fn start(
    bindings: Vec<HotkeyBinding>,
    tx: GestureSender,
    rx_reload: crate::ReloaderReceiver,
    health: Arc<ListenerHealth>,
) {
    health.set_supported(true);
    health.record_scan(1, 0);
    health.set_keyboards_open(1);
    health.set_backend(Backend::WindowsHook);

    let failed = health.clone();
    if let Err(e) = std::thread::Builder::new()
        .name("fotonvoice-winhook".into())
        .spawn(move || run(bindings, tx, rx_reload, health))
    {
        failed.set_backend_failed(format!("cannot start the keyboard hook: {e}"));
    }
}

fn run(
    bindings: Vec<HotkeyBinding>,
    tx: GestureSender,
    rx_reload: crate::ReloaderReceiver,
    health: Arc<ListenerHealth>,
) {
    let (raw_tx, raw_rx) = crossbeam_channel::unbounded::<RawKey>();
    *PLAN.write().unwrap() = Some(SuppressPlan::build(&bindings));
    *SENDER.write().unwrap() = Some(raw_tx.clone());
    *EVENTS.lock().unwrap() = Some(raw_tx);

    let worker_health = health.clone();
    let worker = std::thread::Builder::new()
        .name("fotonvoice-winhook-match".into())
        .spawn(move || match_loop(bindings, tx, rx_reload, raw_rx, worker_health));
    if let Err(e) = worker {
        health.set_backend_failed(format!("cannot start the hotkey matcher: {e}"));
        return;
    }

    spawn_watchdog(health.clone());

    loop {
        match pump_once() {
            PumpOutcome::Reinstall => {
                warn!("keyboard hook stopped; reinstalling");
                release_all(&raw_tx_from_static());
            }
            PumpOutcome::CannotHook(e) => {
                health.set_backend_failed(format!("cannot install the keyboard hook: {e}"));
                return;
            }
        }
        std::thread::sleep(Duration::from_millis(250));
    }
}

fn raw_tx_from_static() -> Option<crossbeam_channel::Sender<RawKey>> {
    EVENTS.lock().ok().and_then(|g| g.clone())
}

/// Tell the matcher that everything held is now up.
fn release_all(tx: &Option<crossbeam_channel::Sender<RawKey>>) {
    let Some(tx) = tx else { return };
    for (id, held) in PRESSED.iter().enumerate() {
        if held.swap(false, Ordering::Relaxed) {
            let _ = tx.send(RawKey { id, down: false });
        }
        SWALLOWED[id].store(false, Ordering::Relaxed);
    }
}

enum PumpOutcome {
    /// The hook is gone and should be installed again.
    Reinstall,
    /// The hook could not be installed at all.
    CannotHook(std::io::Error),
}

fn pump_once() -> PumpOutcome {
    let hook: HHOOK = unsafe {
        SetWindowsHookExW(
            WH_KEYBOARD_LL,
            Some(hook_proc),
            std::ptr::null_mut(),
            0,
        )
    };
    if hook.is_null() {
        return PumpOutcome::CannotHook(std::io::Error::last_os_error());
    }
    info!("Win32 low-level keyboard hook installed");

    PUMP_THREAD.store(
        unsafe { windows_sys::Win32::System::Threading::GetCurrentThreadId() },
        Ordering::SeqCst,
    );

    let mut msg: MSG = unsafe { std::mem::zeroed() };
    loop {
        let got = unsafe { GetMessageW(&mut msg, std::ptr::null_mut(), 0, 0) };
        if got <= 0 {
            break;
        }
        unsafe {
            let _ = TranslateMessage(&msg);
            DispatchMessageW(&msg);
        }
    }

    unsafe {
        let _ = UnhookWindowsHookEx(hook);
    }
    PumpOutcome::Reinstall
}

/// Whether `process` is running at a higher integrity level than this one -
fn process_is_elevated(process: HANDLE) -> Option<bool> {
    unsafe {
        let mut token: HANDLE = std::ptr::null_mut();
        if OpenProcessToken(process, TOKEN_QUERY, &mut token) == 0 {
            return None;
        }
        let mut elevation = TOKEN_ELEVATION { TokenIsElevated: 0 };
        let mut returned = 0u32;
        let ok = GetTokenInformation(
            token,
            TokenElevation,
            &mut elevation as *mut TOKEN_ELEVATION as *mut core::ffi::c_void,
            std::mem::size_of::<TOKEN_ELEVATION>() as u32,
            &mut returned,
        );
        CloseHandle(token);
        (ok != 0).then_some(elevation.TokenIsElevated != 0)
    }
}

/// True when the focused window belongs to a process elevated above our own -
fn foreground_window_is_elevated() -> bool {
    unsafe {
        let hwnd = GetForegroundWindow();
        if hwnd.is_null() {
            return false;
        }
        let mut pid = 0u32;
        GetWindowThreadProcessId(hwnd, &mut pid);
        if pid == 0 {
            return false;
        }
        let process = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid);
        if process.is_null() {
            return false;
        }
        let foreground_elevated = process_is_elevated(process);
        CloseHandle(process);

        let self_elevated = process_is_elevated(GetCurrentProcess()).unwrap_or(false);
        foreground_elevated == Some(true) && !self_elevated
    }
}

/// Watch for the silent unhook, and for the foreground window outranking
fn spawn_watchdog(health: Arc<ListenerHealth>) {
    let _ = std::thread::Builder::new()
        .name("fotonvoice-winhook-watchdog".into())
        .spawn(move || {
            let mut last_seen = SEEN.load(Ordering::Relaxed);
            let mut quiet_since = Instant::now();
            loop {
                std::thread::sleep(WATCHDOG_TICK);

                health.set_elevated_window_focused(foreground_window_is_elevated());

                let seen = SEEN.load(Ordering::Relaxed);
                if seen != last_seen {
                    last_seen = seen;
                    quiet_since = Instant::now();
                    continue;
                }
                if quiet_since.elapsed() < WATCHDOG_SILENCE {
                    continue;
                }
                let thread = PUMP_THREAD.load(Ordering::SeqCst);
                if thread == 0 {
                    continue;
                }
                warn!(
                    "no keyboard events for {}s; reinstalling the hook in case Windows \
                     removed it for overrunning LowLevelHooksTimeout",
                    WATCHDOG_SILENCE.as_secs()
                );
                health.set_backend(Backend::WindowsHook);
                unsafe {
                    let _ = windows_sys::Win32::UI::WindowsAndMessaging::PostThreadMessageW(
                        thread,
                        windows_sys::Win32::UI::WindowsAndMessaging::WM_QUIT,
                        0,
                        0,
                    );
                }
                quiet_since = Instant::now();
            }
        });
}

/// Match raw keys against bindings and drive the gesture engine.
fn match_loop(
    bindings: Vec<HotkeyBinding>,
    tx: GestureSender,
    rx_reload: crate::ReloaderReceiver,
    raw_rx: crossbeam_channel::Receiver<RawKey>,
    _health: Arc<ListenerHealth>,
) {
    let mut engine = GestureEngine::new(bindings.clone());
    let mut matcher = KeyMatcher::new(bindings);

    loop {
        crossbeam_channel::select! {
            recv(rx_reload) -> msg => {
                let Ok(new_bindings) = msg else { break };
                info!("windows hotkey loop: reloading {} bindings", new_bindings.len());
                engine.reset(&tx);
                for (id, transition) in matcher.clear() {
                    engine.apply(&id, transition, &tx);
                }
                engine.reload(new_bindings.clone());
                matcher.reload(new_bindings.clone());
                *PLAN.write().unwrap() = Some(SuppressPlan::build(&new_bindings));
            }
            recv(raw_rx) -> msg => {
                let Ok(raw) = msg else { break };
                let name = keymap::name(raw.id);
                for (id, transition) in matcher.on_key(name, raw.down) {
                    engine.apply(&id, transition, &tx);
                }
            }
        }
    }

    engine.reset(&tx);
}
