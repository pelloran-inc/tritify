//! Compiles the CUDA kernel (`src/digit_gemm.cu`) when the `cuda` feature is on. Without the feature it does
//! nothing. With the feature but without `nvcc` it warns and links nothing: `cargo check` and `clippy` still work on a
//! host without CUDA, and a link fails, as it should.
//!
//! Environment: `NVCC` (the compiler), `CUDA_HOME` or `CUDA_PATH` (default `/usr/local/cuda`), `KT_CUDA_ARCHS`
//! (compute capabilities, default `75,80,86,89,90`; Tensor-Core `int8` needs 7.2 or later).
//!
//! The CUDA runtime is linked dynamically (`libcudart.so`). It is searched for under the toolkit (`$CUDA_HOME/lib64`,
//! the directory beside the resolved `nvcc`) and in the distribution's multiarch directory (`/usr/lib/<arch>-linux-gnu`,
//! where an apt install puts it). At run time the loader finds it there; a toolkit outside the loader's path needs
//! `LD_LIBRARY_PATH`.

use std::env;
use std::path::PathBuf;
use std::process::Command;

/// The first `nvcc` on `PATH`, with its symlinks resolved.
fn nvcc_on_path() -> Option<PathBuf> {
    let path = env::var_os("PATH")?;
    env::split_paths(&path).map(|d| d.join("nvcc")).find(|p| p.is_file()).map(|p| p.canonicalize().unwrap_or(p))
}

/// Directories that may hold `libcudart.so`, in search order, the toolkit's `stubs` excluded.
fn cudart_dirs(home: &str, nvcc: &std::path::Path) -> Vec<PathBuf> {
    let arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_else(|_| "x86_64".into());
    let mut dirs: Vec<PathBuf> = ["lib64", "lib", "targets/x86_64-linux/lib", "targets/sbsa-linux/lib"]
        .iter()
        .map(|d| PathBuf::from(home).join(d))
        .collect();
    if let Some(bin) = nvcc.canonicalize().ok().as_deref().and_then(|p| p.parent()) {
        dirs.extend(["../lib64", "../lib"].iter().map(|d| bin.join(d)));
    }
    dirs.push(PathBuf::from(format!("/usr/lib/{arch}-linux-gnu")));
    dirs.push(PathBuf::from("/usr/lib64"));
    dirs.into_iter()
        .filter(|d| {
            std::fs::read_dir(d).is_ok_and(|mut r| {
                r.any(|e| e.is_ok_and(|e| e.file_name().to_string_lossy().starts_with("libcudart.so")))
            })
        })
        .collect()
}

fn main() {
    for v in ["NVCC", "CUDA_HOME", "CUDA_PATH", "KT_CUDA_ARCHS"] {
        println!("cargo:rerun-if-env-changed={v}");
    }
    println!("cargo:rerun-if-changed=src/digit_gemm.cu");
    if env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }
    let home = env::var("CUDA_HOME").or_else(|_| env::var("CUDA_PATH")).unwrap_or_else(|_| "/usr/local/cuda".into());
    let nvcc = env::var("NVCC").map(PathBuf::from).unwrap_or_else(|_| {
        let p = PathBuf::from(&home).join("bin/nvcc");
        if p.exists() {
            p
        } else {
            nvcc_on_path().unwrap_or_else(|| PathBuf::from("nvcc"))
        }
    });
    let out = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    let archs = env::var("KT_CUDA_ARCHS").unwrap_or_else(|_| "75,80,86,89,90".into());
    let archs: Vec<&str> = archs.split(',').map(str::trim).filter(|a| !a.is_empty()).collect();
    let mut cmd = Command::new(&nvcc);
    cmd.args(["-O3", "-std=c++17", "--lib", "-Xcompiler", "-fPIC", "-o"]).arg(out.join("libktcuda.a"));
    for a in &archs {
        cmd.arg(format!("-gencode=arch=compute_{a},code=sm_{a}"));
    }
    if let Some(last) = archs.last() {
        cmd.arg(format!("-gencode=arch=compute_{last},code=compute_{last}")); // PTX for newer devices
    }
    cmd.arg("src/digit_gemm.cu");
    match cmd.status() {
        Ok(s) if s.success() => {}
        Ok(s) => panic!("kt-cuda: nvcc failed ({s}) on src/digit_gemm.cu"),
        Err(e) => {
            println!("cargo:warning=kt-cuda: the cuda feature is on but {} did not run ({e}); the kernel is not compiled and a link will fail", nvcc.display());
            return;
        }
    }
    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=ktcuda");
    let dirs = cudart_dirs(&home, &nvcc);
    if dirs.is_empty() {
        println!("cargo:warning=kt-cuda: libcudart.so was not found under {home} or /usr/lib; set CUDA_HOME");
    }
    for d in &dirs {
        println!("cargo:rustc-link-search=native={}", d.display());
    }
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");
}
