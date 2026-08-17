use std::env;
use std::path::PathBuf;

// Build the candle kernels.
//
// - Default / CUDA mode: compile the .cu sources to PTX with cudaforge (nvcc) and
//   embed the PTX as strings, exactly as before.
// - ROCm mode (`rocm` feature): compile the same .cu sources with hipcc to AMD
//   code objects and embed them as byte slices, loaded at runtime via
//   rocm_rs::hip::Module::load_data.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    if env::var("CARGO_FEATURE_ROCM").is_ok() {
        build_rocm()?;
    } else {
        build_cuda()?;
    }
    Ok(())
}

fn build_cuda() -> Result<(), Box<dyn std::error::Error>> {
    use cudaforge::KernelBuilder;

    use cudaforge::Result;

    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-changed=src/compatibility.cuh");
    println!("cargo::rerun-if-changed=src/cuda_utils.cuh");
    println!("cargo::rerun-if-changed=src/binary_op_macros.cuh");

    // Build for PTX
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let ptx_path = out_dir.join("ptx.rs");
    let bindings = KernelBuilder::new()
        .source_dir("src") // Scan src/ for .cu files
        .exclude(&[
            "moe_*.cu",
            "mmvq_gguf.cu",
            "mmq_*.cu",
            "hip_gemm.cu",
            "hip_conv.cu",
        ]) // Exclude statically compiled kernels from ptx build
        .arg("--expt-relaxed-constexpr")
        .arg("-std=c++17")
        .arg("-O3")
        .build_ptx()?;

    bindings.write(&ptx_path)?;

    let mut moe_builder = KernelBuilder::default()
        .source_files(vec![
            "src/moe/moe_gguf.cu",
            "src/moe/moe_wmma.cu",
            "src/moe/moe_wmma_gguf.cu",
            "src/mmvq_gguf.cu",
            "src/mmq_gguf/mmq_quantize.cu",
            "src/mmq_gguf/mmq_instance_q4_0.cu",
            "src/mmq_gguf/mmq_instance_q4_1.cu",
            "src/mmq_gguf/mmq_instance_q5_0.cu",
            "src/mmq_gguf/mmq_instance_q5_1.cu",
            "src/mmq_gguf/mmq_instance_q8_0.cu",
            "src/mmq_gguf/mmq_instance_q2_k.cu",
            "src/mmq_gguf/mmq_instance_q3_k.cu",
            "src/mmq_gguf/mmq_instance_q4_k.cu",
            "src/mmq_gguf/mmq_instance_q5_k.cu",
            "src/mmq_gguf/mmq_instance_q6_k.cu",
        ])
        .arg("--expt-relaxed-constexpr")
        .arg("-std=c++17")
        .arg("-O3");

    // Disable bf16 WMMA kernels on GPUs older than sm_80 (Ampere).
    // bf16 WMMA fragments require compute capability >= 8.0.
    let compute_cap = cudaforge::detect_compute_cap()
        .map(|arch| arch.base())
        .unwrap_or(80);
    if compute_cap < 80 {
        moe_builder = moe_builder.arg("-DNO_BF16_KERNEL");
    }

    let mut is_target_msvc = false;
    if let Ok(target) = env::var("TARGET") {
        if target.contains("msvc") {
            is_target_msvc = true;
            moe_builder = moe_builder.arg("-D_USE_MATH_DEFINES");
        }
    }

    if !is_target_msvc {
        moe_builder = moe_builder.arg("-Xcompiler").arg("-fPIC");
    }

    moe_builder.build_lib(out_dir.join("libmoe.a"))?;
    println!("cargo:rustc-link-search={}", out_dir.display());
    println!("cargo:rustc-link-lib=moe");
    println!("cargo:rustc-link-lib=dylib=cudart");
    if !is_target_msvc {
        println!("cargo:rustc-link-lib=stdc++");
    }
    Ok(())
}

const SIMPLE_KERNELS: &[(&str, &str)] = &[
    ("affine", "affine.cu"),
    ("binary", "binary.cu"),
    ("cast", "cast.cu"),
    ("fill", "fill.cu"),
    ("indexing", "indexing.cu"),
    ("reduce", "reduce.cu"),
    ("sort", "sort.cu"),
    ("ternary", "ternary.cu"),
    ("unary", "unary.cu"),
];

// Custom kernels implemented directly for the HIP backend.
const CUSTOM_KERNELS: &[(&str, &str)] = &[
    ("gemm", "hip_kernels/hip_gemm.cu"),
    ("conv", "hip_kernels/hip_conv.cu"),
];

fn find_hipcc() -> Option<PathBuf> {
    for var in ["HIP_PATH", "ROCM_PATH", "CUDA_HOME"] {
        if let Ok(base) = env::var(var) {
            let cand = PathBuf::from(&base).join("bin").join("hipcc");
            if cand.exists() {
                return Some(cand);
            }
        }
    }
    for dir in ["/opt/rocm/bin", "/opt/therock-tarball/install/bin"] {
        let cand = PathBuf::from(dir).join("hipcc");
        if cand.exists() {
            return Some(cand);
        }
    }
    // Fall back to PATH.
    if let Ok(path) = env::var("PATH") {
        for p in path.split(':') {
            let cand = PathBuf::from(p).join("hipcc");
            if cand.exists() {
                return Some(cand);
            }
        }
    }
    None
}

fn rocm_include_dir() -> PathBuf {
    for var in ["ROCM_PATH", "HIP_PATH"] {
        if let Ok(base) = env::var(var) {
            return PathBuf::from(base).join("include");
        }
    }
    PathBuf::from("/opt/rocm/include")
}

fn detect_gfx_arch() -> String {
    if let Ok(arch) = env::var("CANDLE_ROCM_ARCH") {
        return arch;
    }
    if let Some(hipcc) = find_hipcc() {
        if let Some(parent) = hipcc.parent() {
            let agent = parent.join("rocm_agent_enumerator");
            if let Ok(out) = std::process::Command::new(&agent).output() {
                let stdout = String::from_utf8_lossy(&out.stdout);
                for line in stdout.lines() {
                    let line = line.trim();
                    if line.starts_with("gfx") && line != "gfx000" {
                        return line.to_string();
                    }
                }
            }
        }
    }
    "gfx1201".to_string()
}

fn build_rocm() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-changed=src/hip_compat/compatibility.cuh");
    println!("cargo::rerun-if-changed=src/cuda_utils.cuh");
    println!("cargo::rerun-if-changed=src/binary_op_macros.cuh");

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let src_dir = out_dir.join("src");
    std::fs::create_dir_all(&src_dir)?;

    // Stage the kernel sources and the HIP compatibility headers into OUT_DIR so
    // that quote-includes resolve to the HIP versions of the CUDA headers.
    let repo_src = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap()).join("src");
    let stage = |rel: &str| -> Result<(), Box<dyn std::error::Error>> {
        let dst = src_dir.join(PathBuf::from(rel).file_name().unwrap());
        std::fs::copy(repo_src.join(rel), dst)?;
        Ok(())
    };
    for (_, file) in SIMPLE_KERNELS.iter().chain(CUSTOM_KERNELS.iter()) {
        stage(file)?;
    }
    for f in ["cuda_utils.cuh", "binary_op_macros.cuh"] {
        stage(f)?;
    }
    for f in ["compatibility.cuh", "cuda_fp16.h", "cuda_bf16.h", "cuda_fp8.h", "cuda.h"] {
        stage(&format!("hip_compat/{f}"))?;
    }
    // Stage the minimal `cuda::std::numeric_limits` shim, preserving its path so
    // that `#include <cuda/std/limits>` resolves to it (the full libhipcxx header
    // crashes when embedded in a hipModuleLoadData code object).
    let limits_src = repo_src.join("hip_compat/cuda/std/limits");
    let limits_dst = src_dir.join("cuda/std/limits");
    if let Some(parent) = limits_dst.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::copy(limits_src, limits_dst)?;

    let hipcc = find_hipcc().ok_or("hipcc not found: set HIP_PATH or ROCM_PATH, or install ROCm")?;
    let rocm_include = rocm_include_dir();
    if !rocm_include.exists() {
        return Err(format!("ROCm include dir not found at {}", rocm_include.display()).into());
    }
    let arch = detect_gfx_arch();
    println!("cargo::warning=Building candle kernels for ROCm arch {arch}");

    let mut modules = String::new();
    let mut emit = |name: &str, file: &str, out: &str| -> Result<(), Box<dyn std::error::Error>> {
        let staged = src_dir.join(PathBuf::from(file).file_name().unwrap());
        let status = std::process::Command::new(&hipcc)
            .arg(format!("--offload-arch={arch}"))
            .arg("-O3")
            .arg("-std=c++17")
            .arg(format!("-I{}", src_dir.display()))
            .arg(format!("-I{}", rocm_include.display()))
            .arg("-x")
            .arg("hip")
            .arg("--genco")
            .arg("-o")
            .arg(&out)
            .arg(&staged)
            .status()?;
        if !status.success() {
            return Err(format!("hipcc failed for {file}").into());
        }
        let up = name.to_uppercase();
        modules.push_str(&format!(
            "pub const {up}: &[u8] = include_bytes!({});\n",
            format!(r#""{}""#, out.replace('\\', "/"))
        ));
        println!("cargo::rerun-if-changed=src/{file}");
        Ok(())
    };

    for (name, file) in SIMPLE_KERNELS.iter().chain(CUSTOM_KERNELS.iter()) {
        let out = out_dir.join(format!("{name}.co"));
        emit(name, file, out.to_str().unwrap())?;
    }

    std::fs::write(out_dir.join("rocm.rs"), modules)?;
    Ok(())
}
