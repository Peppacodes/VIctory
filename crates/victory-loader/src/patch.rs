//! Debugging aid: bytes written over the title's image once it is loaded,
//! `VICTORY_PATCH=<rva>:<hex bytes>,...` (both hex), for experiments such
//! as keeping a title's own hang watchdog from ending a run that is only
//! slow. Diagnostics builds only.

use windows_sys::Win32::System::Diagnostics::Debug::FlushInstructionCache;
use windows_sys::Win32::System::Memory::{PAGE_EXECUTE_READWRITE, VirtualProtect};
use windows_sys::Win32::System::Threading::GetCurrentProcess;

/// The patches in `list` (`<rva>:<hex bytes>`, comma-separated); malformed
/// entries are left out.
pub fn parse_patches(list: &str) -> Vec<(usize, Vec<u8>)> {
    list.split(',')
        .filter_map(|entry| {
            let (rva, hex) = entry.trim().split_once(':')?;
            let rva = usize::from_str_radix(rva.trim().trim_start_matches("0x"), 16).ok()?;
            let hex = hex.trim();
            if hex.is_empty() || hex.len() % 2 != 0 {
                return None;
            }
            let bytes = (0..hex.len())
                .step_by(2)
                .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).ok())
                .collect::<Option<Vec<u8>>>()?;
            Some((rva, bytes))
        })
        .collect()
}

/// Writes each patch over the image at `base` (`size` bytes long), whatever
/// its pages' protection, and returns how many it wrote; one reaching past
/// the image is skipped. Run before the title starts.
pub fn patch_image(base: usize, size: usize, patches: &[(usize, Vec<u8>)]) -> usize {
    let mut written = 0;
    for (rva, bytes) in patches {
        if bytes.is_empty() || rva.checked_add(bytes.len()).is_none_or(|end| end > size) {
            tracing::warn!(rva, len = bytes.len(), "patch outside the image: skipped");
            continue;
        }
        let at = (base + rva) as *mut u8;
        let mut old = 0;
        // SAFETY: inside the mapped image, checked above; its pages are made
        // writable for the copy and put back as they were.
        unsafe {
            if VirtualProtect(at.cast(), bytes.len(), PAGE_EXECUTE_READWRITE, &mut old) == 0 {
                tracing::warn!(rva, "patch: pages not made writable; skipped");
                continue;
            }
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), at, bytes.len());
            VirtualProtect(at.cast(), bytes.len(), old, &mut old);
            FlushInstructionCache(GetCurrentProcess(), at.cast(), bytes.len());
        }
        tracing::info!(
            rva = format_args!("{rva:#x}"),
            len = bytes.len(),
            "title patched"
        );
        written += 1;
    }
    written
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_EXECUTE_READ, PAGE_READWRITE, VirtualAlloc,
        VirtualFree,
    };

    #[test]
    fn patches_are_parsed_and_malformed_ones_left_out() {
        assert_eq!(
            parse_patches("2215c70:9090, 0x10:ff ,zz:00,20:abc,30:,40:0g"),
            [(0x2215c70, vec![0x90, 0x90]), (0x10, vec![0xff])]
        );
        assert!(parse_patches("").is_empty());
    }

    #[test]
    fn read_only_code_is_patched_and_keeps_its_protection() {
        // SAFETY: a fresh page of our own, freed below.
        unsafe {
            let page = VirtualAlloc(
                std::ptr::null(),
                0x1000,
                MEM_COMMIT | MEM_RESERVE,
                PAGE_READWRITE,
            )
            .cast::<u8>();
            assert!(!page.is_null());
            page.write_bytes(0xcc, 0x1000);
            let mut old = 0;
            assert_ne!(
                VirtualProtect(page.cast(), 0x1000, PAGE_EXECUTE_READ, &mut old),
                0
            );
            let base = page as usize;
            let patches = [(0x10, vec![0x90, 0x90]), (0xfff, vec![1, 2])];
            assert_eq!(
                patch_image(base, 0x1000, &patches),
                1,
                "the second reaches past the end"
            );
            assert_eq!(*page.add(0x10), 0x90);
            assert_eq!(*page.add(0x11), 0x90);
            assert_eq!(*page.add(0x12), 0xcc);
            assert_ne!(
                VirtualProtect(page.cast(), 0x1000, PAGE_READWRITE, &mut old),
                0
            );
            assert_eq!(old, PAGE_EXECUTE_READ, "protection put back");
            assert_ne!(VirtualFree(page.cast(), 0, MEM_RELEASE), 0);
        }
    }
}
