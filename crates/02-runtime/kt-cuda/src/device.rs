//! The CUDA executor (the `cuda` feature). Packed weight tiles live in device memory; the kernel in `digit_gemm.cu`
//! runs the product on the Tensor Cores. Every CUDA call goes through the small C interface at the end of that file.

use crate::backend::Exec;
use crate::digits::TILE;
use std::ffi::{c_char, c_int, c_void, CStr};
use std::sync::Mutex;

extern "C" {
    fn kt_cuda_device_count(n: *mut c_int) -> c_int;
    fn kt_cuda_device_info(
        dev: c_int,
        name: *mut c_char,
        len: c_int,
        major: *mut c_int,
        minor: *mut c_int,
        free: *mut usize,
    ) -> c_int;
    fn kt_cuda_alloc(dev: c_int, bytes: usize, p: *mut *mut c_void) -> c_int;
    fn kt_cuda_free(dev: c_int, p: *mut c_void) -> c_int;
    fn kt_cuda_upload(dev: c_int, dst: *mut c_void, src: *const c_void, bytes: usize) -> c_int;
    #[allow(clippy::too_many_arguments)]
    fn kt_cuda_digit_gemm(
        dev: c_int,
        xa_host: *const i8,
        xa_bytes: usize,
        xa_dev: *mut c_void,
        wb_dev: *const c_void,
        y_dev: *mut c_void,
        y_host: *mut i64,
        rows: c_int,
        out: c_int,
        kts: c_int,
        kt_flush: c_int,
        drop22: c_int,
    ) -> c_int;
    fn kt_cuda_error_string(e: c_int) -> *const c_char;
}

/// Panic with CUDA's message: the backend contract has no error path, and a wrong answer is worse than a stop.
fn check(e: c_int, what: &str) {
    if e != 0 {
        // SAFETY: cudaGetErrorString returns a static, NUL-terminated string for any code.
        let msg = unsafe { CStr::from_ptr(kt_cuda_error_string(e)) }.to_string_lossy();
        panic!("kt-cuda: {what}: {msg} (cudaError {e})");
    }
}

struct Info {
    name: String,
    capability: (i32, i32),
    free: usize,
}

fn info(dev: c_int) -> Option<Info> {
    let (mut name, mut major, mut minor, mut free) = ([0 as c_char; 256], 0, 0, 0usize);
    // SAFETY: every pointer is to a local of the declared type; the name buffer's length is passed.
    let e = unsafe { kt_cuda_device_info(dev, name.as_mut_ptr(), 256, &mut major, &mut minor, &mut free) };
    // SAFETY: the C side NUL-terminates within the buffer.
    let name = unsafe { CStr::from_ptr(name.as_ptr()) }.to_string_lossy().into_owned();
    (e == 0).then_some(Info { name, capability: (major, minor), free })
}

/// The CUDA devices, by name, in index order. A host without a driver has none.
pub fn devices() -> Vec<String> {
    let mut n: c_int = 0;
    // SAFETY: writes one int.
    if unsafe { kt_cuda_device_count(&mut n) } != 0 {
        return Vec::new();
    }
    (0..n).map(|d| info(d).map_or_else(|| format!("device {d} (unreadable)"), |i| i.name)).collect()
}

/// Device memory, freed on drop.
pub struct DevBuf {
    dev: c_int,
    ptr: *mut c_void,
    bytes: usize,
}

// SAFETY: a device pointer is an address in the device's memory, valid from any host thread.
unsafe impl Send for DevBuf {}
// SAFETY: as above; the buffer is only read after its upload.
unsafe impl Sync for DevBuf {}

impl DevBuf {
    fn empty(dev: c_int) -> DevBuf {
        DevBuf { dev, ptr: std::ptr::null_mut(), bytes: 0 }
    }

    fn alloc(dev: c_int, bytes: usize) -> DevBuf {
        let mut b = DevBuf::empty(dev);
        if bytes > 0 {
            // SAFETY: cudaMalloc writes one device pointer.
            check(unsafe { kt_cuda_alloc(dev, bytes, &mut b.ptr) }, "allocate");
            b.bytes = bytes;
        }
        b
    }

    /// At least `bytes`, reallocating only to grow.
    fn grow(&mut self, bytes: usize) {
        if bytes > self.bytes {
            *self = DevBuf::alloc(self.dev, bytes);
        }
    }
}

impl Drop for DevBuf {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            // SAFETY: the pointer came from cudaMalloc on this device and is freed once. An error at process exit
            // (the runtime already unloading) is ignored.
            unsafe { kt_cuda_free(self.dev, self.ptr) };
        }
    }
}

/// One CUDA device with Tensor-Core `int8` (compute capability 7.2 or later).
pub struct DeviceExec {
    dev: c_int,
    name: String,
    free: usize,
    /// The activation and result buffers, reused across calls; the lock also orders the calls.
    scratch: Mutex<(DevBuf, DevBuf)>,
}

impl DeviceExec {
    /// Device `index`, or `None` if it does not exist or lacks `int8` Tensor Cores.
    pub fn open(index: usize) -> Option<DeviceExec> {
        let dev = c_int::try_from(index).ok()?;
        let i = info(dev).filter(|i| i.capability >= (7, 2))?;
        let scratch = Mutex::new((DevBuf::empty(dev), DevBuf::empty(dev)));
        Some(DeviceExec { dev, name: i.name, free: i.free, scratch })
    }
}

impl Exec for DeviceExec {
    type Buf = DevBuf;

    fn kind(&self) -> &'static str {
        "cuda"
    }
    fn name(&self) -> String {
        format!("cuda:{}", self.name)
    }
    fn default_budget(&self) -> u64 {
        self.free as u64 / 5 * 4
    }
    fn upload(&self, packed: Vec<i8>) -> DevBuf {
        let b = DevBuf::alloc(self.dev, packed.len());
        if !packed.is_empty() {
            // SAFETY: the device buffer holds packed.len() bytes, the host slice is that long.
            check(unsafe { kt_cuda_upload(self.dev, b.ptr, packed.as_ptr().cast(), packed.len()) }, "upload weights");
        }
        b
    }
    fn gemm(
        &self,
        xa: &[i8],
        wb: &DevBuf,
        (rows, out, k): (usize, usize, usize),
        flush: usize,
        skip22: bool,
    ) -> Vec<i64> {
        let mut y = vec![0i64; rows * out];
        if y.is_empty() {
            return y;
        }
        let int =
            |v: usize, what: &str| c_int::try_from(v).unwrap_or_else(|_| panic!("kt-cuda: {what} {v} exceeds int"));
        let (r, o, kts) = (int(rows, "rows"), int(out, "outputs"), int(k.div_ceil(TILE), "k tiles"));
        let flush = flush.min(c_int::MAX as usize) as c_int;
        let mut s = self.scratch.lock().unwrap_or_else(|e| e.into_inner());
        s.0.grow(xa.len());
        s.1.grow(y.len() * std::mem::size_of::<i64>());
        // SAFETY: xa is xa.len() bytes; the scratch buffers hold at least xa.len() and rows·out·8 bytes; wb holds the
        // packed weights for (out, k); y is rows·out long. The call returns after the result is copied back.
        let e = unsafe {
            kt_cuda_digit_gemm(
                self.dev,
                xa.as_ptr(),
                xa.len(),
                s.0.ptr,
                wb.ptr,
                s.1.ptr,
                y.as_mut_ptr(),
                r,
                o,
                kts,
                flush,
                skip22 as c_int,
            )
        };
        check(e, "digit product");
        y
    }
}
