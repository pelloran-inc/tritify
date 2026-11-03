//! kt-runtime — the automatic compute backend. At startup it detects what the host offers (CPU instruction sets,
//! cores, GPU adapters), picks the fastest backend that yields the same bits, and installs it into the engine
//! through `kt_core::backend`. It runs in the layer above the engine and is injected from the application, so the
//! engine never depends on it. Every choice can be overridden ([`RuntimeConfig`]); forcing something the host
//! lacks is refused, never silently replaced.
//!
//! The bits do not depend on the choice: integer accumulation is exact and order-independent, and the
//! [`equality_gate`] checks that every available backend returns the same bytes.
//!
//! # Example
//!
//! ```
//! use kt_runtime::{install, BackendChoice, CpuVariant, RuntimeConfig};
//!
//! // Auto: the best CPU variant this host has.
//! let sel = install(&RuntimeConfig::default()).unwrap();
//! assert!(CpuVariant::from_name(&sel.backend).is_some());
//!
//! // An override the host cannot honour is refused.
//! let wrong = if cfg!(target_arch = "aarch64") { CpuVariant::Avx2 } else { CpuVariant::Neon };
//! let cfg = RuntimeConfig { backend: BackendChoice::Cpu(Some(wrong)), ..Default::default() };
//! assert!(install(&cfg).is_err());
//! kt_core::backend::reset_backend();
//! ```

mod cpu;
mod gate;
#[cfg(any(test, feature = "plants"))]
pub mod planted;

pub use cpu::{CpuBackend, CpuVariant};
pub use gate::{corpus, equality_gate, flush_case, residency_gate, Case, GateReport};
pub use kt_cuda;
#[cfg(feature = "gpu")]
pub use kt_gpu;

use kt_core::backend::{install_backend, LinearBackend};
use std::sync::Arc;

/// What this host offers.
#[derive(Clone, Debug)]
pub struct HostInfo {
    /// `std::env::consts::ARCH` (`aarch64`, `x86_64`, …).
    pub arch: &'static str,
    /// `std::env::consts::OS`.
    pub os: &'static str,
    /// Logical cores reported by the OS.
    pub cores: usize,
    /// The CPU variants this host can run.
    pub cpu_variants: Vec<CpuVariant>,
    /// The GPU adapters, in the order [`BackendChoice::Gpu`] indexes them (empty without the `gpu` feature).
    pub gpus: Vec<GpuAdapter>,
    /// The CUDA devices, in the order [`BackendChoice::Cuda`] indexes them (empty without the `cuda` feature).
    pub cudas: Vec<String>,
}

/// One GPU adapter.
#[derive(Clone, Debug)]
pub struct GpuAdapter {
    pub name: String,
    /// The graphics API (`Metal`, `Vulkan`, `Dx12`, `Gl`).
    pub api: String,
    /// `DiscreteGpu`, `IntegratedGpu`, `VirtualGpu`, `Cpu` (a software rasterizer), `Other`.
    pub kind: String,
}

#[cfg(feature = "gpu")]
fn gpu_adapters() -> Vec<GpuAdapter> {
    kt_gpu::adapters().into_iter().map(|a| GpuAdapter { name: a.name, api: a.backend, kind: a.device_type }).collect()
}

#[cfg(not(feature = "gpu"))]
fn gpu_adapters() -> Vec<GpuAdapter> {
    Vec::new()
}

/// Detect the host.
pub fn detect() -> HostInfo {
    HostInfo {
        arch: std::env::consts::ARCH,
        os: std::env::consts::OS,
        cores: std::thread::available_parallelism().map_or(1, |n| n.get()),
        cpu_variants: CpuVariant::ALL.into_iter().filter(|v| v.available()).collect(),
        gpus: gpu_adapters(),
        cudas: kt_cuda::devices(),
    }
}

/// The auto rule: the best-ranked CPU variant the host has. A GPU is chosen only by override: the GPU backend uploads
/// the weights on every call, and on the measured host the CPU is faster. On x86-64: `avx512` > `avx2` > `portable`. On aarch64:
/// `portable` > `neon`, because the compiler's vectorised loop measured faster than the explicit intrinsics.
/// `neon` stays available as an override and as an independent implementation for the equality gate.
pub fn auto_variant(host: &HostInfo) -> CpuVariant {
    const RANK: [CpuVariant; 4] = [CpuVariant::Avx512, CpuVariant::Avx2, CpuVariant::Portable, CpuVariant::Neon];
    RANK.into_iter().find(|v| host.cpu_variants.contains(v)).unwrap_or(CpuVariant::Portable)
}

/// Which backend to run.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum BackendChoice {
    /// Pick by the auto rule ([`auto_variant`]).
    #[default]
    Auto,
    /// The CPU, with a forced variant or (`None`) the auto variant.
    Cpu(Option<CpuVariant>),
    /// A GPU: adapter `i` of [`HostInfo::gpus`], or (`None`) the high-performance default.
    Gpu(Option<usize>),
    /// The exact INT8 Tensor-Core backend on CUDA device `i` of [`HostInfo::cudas`], or (`None`) device 0.
    Cuda(Option<usize>),
}

/// The runtime configuration: everything the console can override.
#[derive(Clone, Debug, Default)]
pub struct RuntimeConfig {
    pub backend: BackendChoice,
    /// Cap on worker threads (`0` = all cores, `1` = serial). `None` leaves the engine's setting alone.
    pub max_threads: Option<usize>,
    /// Cap on the bytes of weights a GPU or CUDA backend keeps resident (`None` = its default: 8 GiB on a GPU, 4/5 of
    /// the free device memory under CUDA).
    pub gpu_budget_bytes: Option<u64>,
}

/// What was installed, and why.
#[derive(Clone, Debug)]
pub struct Selection {
    /// The backend's name, as `LinearBackend::name` reports it.
    pub backend: String,
    pub reason: String,
}

/// Build the backend a configuration asks for on this host, without installing it.
pub fn build(cfg: &RuntimeConfig, host: &HostInfo) -> Result<(Arc<dyn LinearBackend>, Selection), String> {
    let (variant, reason) = match &cfg.backend {
        BackendChoice::Auto | BackendChoice::Cpu(None) => {
            let v = auto_variant(host);
            (v, format!("auto: best CPU variant on {} of {:?}", host.arch, names(&host.cpu_variants)))
        }
        BackendChoice::Cpu(Some(v)) => (*v, "forced by configuration".to_string()),
        BackendChoice::Gpu(index) => return gpu_backend(*index, cfg.gpu_budget_bytes),
        BackendChoice::Cuda(index) => return cuda_backend(*index, cfg.gpu_budget_bytes),
    };
    let b = CpuBackend::new(variant)?;
    Ok((Arc::new(b), Selection { backend: variant.name().into(), reason }))
}

#[cfg(feature = "gpu")]
fn gpu_backend(index: Option<usize>, budget: Option<u64>) -> Result<(Arc<dyn LinearBackend>, Selection), String> {
    let gpu = kt_gpu::Gpu::open(index).ok_or_else(|| match index {
        Some(i) => format!("GPU adapter {i} is not available on this host"),
        None => "no GPU adapter on this host".to_string(),
    })?;
    let b = kt_gpu::GpuLinear::new(gpu).with_budget(budget.unwrap_or(kt_gpu::GpuLinear::DEFAULT_BUDGET));
    let backend = b.name();
    Ok((Arc::new(b), Selection { backend, reason: "forced by configuration".into() }))
}

#[cfg(not(feature = "gpu"))]
fn gpu_backend(_index: Option<usize>, _budget: Option<u64>) -> Result<(Arc<dyn LinearBackend>, Selection), String> {
    Err("this runtime was built without the gpu feature".into())
}

fn cuda_backend(index: Option<usize>, budget: Option<u64>) -> Result<(Arc<dyn LinearBackend>, Selection), String> {
    if !kt_cuda::CUDA {
        return Err("this runtime was built without the cuda feature".into());
    }
    let b = kt_cuda::open(index, budget).ok_or_else(|| match index {
        Some(i) => format!("CUDA device {i} is not available on this host"),
        None => "no CUDA device on this host".to_string(),
    })?;
    let backend = b.name();
    Ok((b, Selection { backend, reason: "forced by configuration".into() }))
}

/// Detect, build and install: the one call an application makes at startup.
pub fn install(cfg: &RuntimeConfig) -> Result<Selection, String> {
    let host = detect();
    let (b, sel) = build(cfg, &host)?;
    if let Some(n) = cfg.max_threads {
        kt_core::set_max_threads(n);
    }
    install_backend(b);
    Ok(sel)
}

/// Every backend this host can run (each CPU variant, then each GPU adapter, then each CUDA device), for the equality
/// gate and the per-backend anchor runs.
pub fn available_backends(host: &HostInfo) -> Vec<Arc<dyn LinearBackend>> {
    let mut v: Vec<Arc<dyn LinearBackend>> = host
        .cpu_variants
        .iter()
        .filter_map(|&v| CpuBackend::new(v).ok())
        .map(|b| Arc::new(b) as Arc<dyn LinearBackend>)
        .collect();
    v.extend((0..host.gpus.len()).filter_map(|i| gpu_backend(Some(i), None).ok()).map(|(b, _)| b));
    v.extend((0..host.cudas.len()).filter_map(|i| cuda_backend(Some(i), None).ok()).map(|(b, _)| b));
    v
}

fn names(vs: &[CpuVariant]) -> Vec<&'static str> {
    vs.iter().map(|v| v.name()).collect()
}

#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_cuda;
