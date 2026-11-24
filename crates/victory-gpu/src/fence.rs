//! Fences: a 64-bit value and the events waiting for it. Signals (the
//! title's and our queues') move one, and so does any write to its GPU
//! address, such as a list's bottom-of-pipe write: [`poll`] then sets the
//! events of the waiters it reached.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard};

use crate::object::{self, E_NOTIMPL, FENCE, HRESULT, Iid, Object, S_OK, Table, This, guid};
use crate::resource::PAGEABLE_IIDS;

pub(crate) const FENCE_IIDS: [Iid; 4] = [
    guid(
        0x0a753dcf,
        0xc4d8,
        0x4b91,
        [0xad, 0xf6, 0xbe, 0x5a, 0x60, 0xd9, 0x5a, 0x76],
    ),
    PAGEABLE_IIDS[0],
    PAGEABLE_IIDS[1],
    PAGEABLE_IIDS[2],
];

#[repr(C)]
struct Fence {
    object: Object,
    /// Titles also read it straight from memory, at the fence's GPU
    /// address.
    value: AtomicU64,
    flags: u32,
    /// Events to set once the value reaches theirs (handles as integers).
    waiters: Mutex<Vec<(u64, usize)>>,
}

impl Drop for Fence {
    fn drop(&mut self) {
        let me = std::ptr::from_ref(self) as usize;
        live().retain(|&f| f != me);
    }
}

/// Every fence alive, by address, for [`poll`]. A fence leaves it when
/// dropped, under this lock, so one listed here is alive while it is held.
static LIVE: Mutex<Vec<usize>> = Mutex::new(Vec::new());

fn live() -> MutexGuard<'static, Vec<usize>> {
    LIVE.lock().unwrap_or_else(|e| e.into_inner())
}

/// Sets the events of waiters whose fence value was reached without a
/// signal: written at the fence's GPU address, which Grand Theft Auto VI
/// does with lists' bottom-of-pipe writes. Run once such writes are done.
pub(crate) fn poll() {
    static SEEN: AtomicBool = AtomicBool::new(false);
    let live = live();
    for &f in live.iter() {
        // SAFETY: listed, so alive (see `LIVE`).
        let fence = unsafe { &*(f as *const Fence) };
        let mut waiters = fence.waiters.lock().unwrap_or_else(|e| e.into_inner());
        if waiters.is_empty() {
            continue;
        }
        let value = fence.value.load(Ordering::Acquire);
        waiters.retain(|&(wanted, event)| {
            let reached = wanted <= value;
            if reached {
                if !SEEN.swap(true, Ordering::Relaxed) {
                    tracing::info!(
                        value,
                        "a fence written at its GPU address, not signaled, set its waiting events"
                    );
                }
                set_event(event);
            }
            !reached
        });
    }
}

fn fence<'a>(this: This) -> &'a Fence {
    // SAFETY: called from fence methods only.
    unsafe { &*this.cast::<Fence>() }
}

pub(crate) extern "system" fn create(
    _: This,
    initial: u64,
    flags: u32,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    tracing::debug!(initial, flags, "fence");
    let fence = Box::new(Fence {
        object: Object::new(&FUNCTIONS),
        value: AtomicU64::new(initial),
        flags,
        waiters: Mutex::new(Vec::new()),
    });
    let at = std::ptr::from_ref::<Fence>(&fence) as usize;
    // Not under `LIVE`'s lock: a fence `hand_out` drops takes it.
    let result = object::hand_out(FENCE, fence, iid, &FENCE_IIDS, out);
    // Handed out (and not dropped): alive until its last release, which
    // only the title can make, and not before this returns.
    if result == S_OK && !out.is_null() {
        live().push(at);
    }
    result
}

fn set_event(event: usize) {
    #[cfg(windows)]
    // SAFETY: the title passed an event handle.
    unsafe {
        windows_sys::Win32::System::Threading::SetEvent(event as *mut c_void)
    };
    #[cfg(not(windows))]
    let _ = event;
}

extern "system" fn query(this: This, iid: *const Iid, out: *mut This) -> HRESULT {
    object::answer(
        FENCE,
        iid,
        object::is(iid, &FENCE_IIDS).then_some(this),
        out,
    )
}
extern "system" fn release(this: This) -> u32 {
    object::release_boxed::<Fence>(this)
}
extern "system" fn completed(this: This) -> u64 {
    fence(this).value.load(Ordering::Acquire)
}
extern "system" fn set_event_on_completion(this: This, value: u64, event: *mut c_void) -> HRESULT {
    let fence = fence(this);
    let mut waiters = fence.waiters.lock().unwrap_or_else(|e| e.into_inner());
    if fence.value.load(Ordering::Acquire) >= value {
        if !event.is_null() {
            set_event(event as usize);
        }
        return S_OK;
    }
    if event.is_null() {
        // shortcut: a null event means "block until reached"; nothing can
        // reach it while no GPU work runs, so fail instead of hanging.
        tracing::warn!(value, "blocking fence wait not implemented");
        return E_NOTIMPL;
    }
    waiters.push((value, event as usize));
    S_OK
}
pub(crate) extern "system" fn signal(this: This, value: u64) -> HRESULT {
    let fence = fence(this);
    let mut waiters = fence.waiters.lock().unwrap_or_else(|e| e.into_inner());
    fence.value.store(value, Ordering::Release);
    waiters.retain(|&(wanted, event)| {
        let reached = wanted <= value;
        if reached {
            set_event(event);
        }
        !reached
    });
    S_OK
}

extern "system" fn creation_flags(this: This) -> u32 {
    fence(this).flags
}
/// Where the value lives (GPU addresses are CPU addresses).
extern "system" fn gpu_address(this: This) -> u64 {
    (&raw const fence(this).value) as u64
}

/// IUnknown 3, object 4, device child 1, fence 3, fence 1 (2, a console
/// one among them).
static FUNCTIONS: Table<24> = {
    let mut f = object::d3d12_object(object::missing_table::<FENCE, 24>());
    f[0] = query as *const ();
    f[2] = release as *const ();
    f[7] = object::parent_device as *const (); // 0x38
    f[8] = completed as *const (); // 0x40
    f[9] = set_event_on_completion as *const (); // 0x48
    f[10] = signal as *const (); // 0x50
    f[11] = creation_flags as *const (); // 0x58
    f[12] = gpu_address as *const (); // 0x60
    Table(f)
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object::slot;

    #[cfg(windows)]
    #[test]
    fn signals_set_waiting_events() {
        use windows_sys::Win32::Foundation::{CloseHandle, WAIT_OBJECT_0};
        use windows_sys::Win32::System::Threading::{CreateEventW, WaitForSingleObject};

        let mut f = std::ptr::null();
        assert_eq!(create(std::ptr::null(), 1, 0, &FENCE_IIDS[0], &mut f), S_OK);
        assert_eq!(slot::<extern "system" fn(This) -> u64>(f, 0x40)(f), 1);
        // SAFETY: plain auto-reset event.
        let event = unsafe { CreateEventW(std::ptr::null(), 0, 0, std::ptr::null()) };
        assert!(!event.is_null());
        let wait = slot::<extern "system" fn(This, u64, *mut c_void) -> HRESULT>(f, 0x48);
        let signal = slot::<extern "system" fn(This, u64) -> HRESULT>(f, 0x50);
        assert_eq!(wait(f, 5, event), S_OK);
        // SAFETY: a valid event.
        assert_ne!(unsafe { WaitForSingleObject(event, 0) }, WAIT_OBJECT_0);
        assert_eq!(signal(f, 5), S_OK);
        // SAFETY: as above.
        assert_eq!(unsafe { WaitForSingleObject(event, 0) }, WAIT_OBJECT_0);
        assert_eq!(wait(f, 5, event), S_OK, "already reached");
        // SAFETY: as above.
        assert_eq!(unsafe { WaitForSingleObject(event, 0) }, WAIT_OBJECT_0);
        assert_eq!(wait(f, 9, std::ptr::null_mut()), E_NOTIMPL);
        // Grand Theft Auto VI reads the value at the fence's address.
        let at = slot::<extern "system" fn(This) -> u64>(f, 0x60)(f);
        // SAFETY: the fence's value.
        assert_eq!(unsafe { *(at as *const u64) }, 5);
        // SAFETY: ours.
        unsafe { CloseHandle(event) };
        assert_eq!(slot::<extern "system" fn(This) -> u32>(f, 0x10)(f), 0);
    }

    /// A list's bottom-of-pipe write to the fence's GPU address moves its
    /// value without a signal; the waiters it reached are set once polled.
    #[cfg(windows)]
    #[test]
    fn writes_at_the_gpu_address_set_waiting_events_once_polled() {
        use windows_sys::Win32::Foundation::{CloseHandle, WAIT_OBJECT_0};
        use windows_sys::Win32::System::Threading::{CreateEventW, WaitForSingleObject};

        let mut f = std::ptr::null();
        assert_eq!(create(std::ptr::null(), 0, 0, &FENCE_IIDS[0], &mut f), S_OK);
        assert!(live().contains(&(f as usize)));
        // SAFETY: plain auto-reset events.
        let (near, far) = unsafe {
            (
                CreateEventW(std::ptr::null(), 0, 0, std::ptr::null()),
                CreateEventW(std::ptr::null(), 0, 0, std::ptr::null()),
            )
        };
        let wait = slot::<extern "system" fn(This, u64, *mut c_void) -> HRESULT>(f, 0x48);
        assert_eq!(wait(f, 3, near), S_OK);
        assert_eq!(wait(f, 9, far), S_OK);
        let at = slot::<extern "system" fn(This) -> u64>(f, 0x60)(f);
        crate::queue::Written::Value(3).write(at as *mut u8, 8);
        poll();
        // SAFETY: valid events.
        unsafe {
            assert_eq!(
                WaitForSingleObject(near, 0),
                WAIT_OBJECT_0,
                "reached by the write"
            );
            assert_ne!(
                WaitForSingleObject(far, 0),
                WAIT_OBJECT_0,
                "not reached yet"
            );
        }
        assert_eq!(slot::<extern "system" fn(This) -> u32>(f, 0x10)(f), 0);
        assert!(
            !live().contains(&(f as usize)),
            "a released fence leaves the list"
        );
        poll();
        // SAFETY: ours.
        unsafe {
            CloseHandle(near);
            CloseHandle(far);
        }
    }
}
