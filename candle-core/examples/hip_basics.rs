//! Basic sanity and micro-benchmark for the HIP/ROCm backend.
use anyhow::Result;
use candle_core::{Device, Tensor, DType};

fn main() -> Result<()> {
    let device = Device::new_hip(0)?;
    println!("device: {device:?}");

    // Basic unary / binary / reduce ops on the GPU.
    let a = Tensor::randn(0f32, 1.0, (128, 256), &device)?;
    let b = Tensor::randn(0f32, 1.0, (128, 256), &device)?;
    let c = (&a + &b)?;
    let r = c.relu()?;
    let m = r.mean(1)?;
    let s = r.sum_all()?;
    device.synchronize()?;

    // Cross-check against the CPU backend using the same data.
    let cpu = Device::Cpu;
    let a_cpu = a.to_device(&cpu)?;
    let b_cpu = b.to_device(&cpu)?;
    let c_cpu = (&a_cpu + &b_cpu)?;
    let r_cpu = c_cpu.relu()?;
    let m_cpu = r_cpu.mean(1)?;
    let s_cpu = r_cpu.sum_all()?;

    let m_gpu = m.to_vec1::<f32>()?;
    let m_cpu = m_cpu.to_vec1::<f32>()?;
    let s_gpu = s.to_scalar::<f32>()?;
    let s_cpu = s_cpu.to_scalar::<f32>()?;
    let max_mean_diff = m_gpu
        .iter()
        .zip(m_cpu.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(max_mean_diff < 1e-4, "mean mismatch {max_mean_diff}");
    assert!((s_gpu - s_cpu).abs() < 1e-2, "sum mismatch {s_gpu} vs {s_cpu}");
    println!("add/relu/mean/sum OK");

    // conv2d + conv_transpose2d (the ops used by CUGAN super-resolution models).
    let x = Tensor::randn(0f32, 1.0, (1, 12, 64, 64), &device)?;
    let w = Tensor::randn(0f32, 1.0, (32, 12, 3, 3), &device)?;
    let y = x.conv2d(&w, 1, 1, 1, 1)?;
    assert_eq!(y.shape().dims(), &[1, 32, 64, 64]);

    let x2 = Tensor::randn(0f32, 1.0, (1, 64, 64, 64), &device)?;
    let w2 = Tensor::randn(0f32, 1.0, (64, 32, 2, 2), &device)?;
    let y2 = x2.conv_transpose2d(&w2, 0, 0, 2, 1)?;
    assert_eq!(y2.shape().dims(), &[1, 32, 128, 128]);
    let w3 = Tensor::randn(0f32, 1.0, (64, 64, 4, 4), &device)?;
    let y3 = x2.conv_transpose2d(&w3, 3, 2, 1, 1)?;
    assert_eq!(y3.shape().dims(), &[1, 64, 63, 63]);
    device.synchronize()?;

    // Cross-check conv results against CPU (max abs diff; float accumulation noise).
    let x_cpu = x.to_device(&cpu)?;
    let w_cpu = w.to_device(&cpu)?;
    let y_cpu = x_cpu.conv2d(&w_cpu, 1, 1, 1, 1)?;
    let diff = (y.to_device(&cpu)? - &y_cpu)?.abs()?.max_all()?.to_scalar::<f32>()?;
    assert!(diff < 1e-3, "conv2d mismatch {diff}");

    let x2_cpu = x2.to_device(&cpu)?;
    let w2_cpu = w2.to_device(&cpu)?;
    let y2_cpu = x2_cpu.conv_transpose2d(&w2_cpu, 0, 0, 2, 1)?;
    let diff2 = (y2.to_device(&cpu)? - &y2_cpu)?.abs()?.max_all()?.to_scalar::<f32>()?;
    assert!(diff2 < 1e-3, "conv_transpose2d (stride 2) mismatch {diff2}");
    let w3_cpu = w3.to_device(&cpu)?;
    let y3_cpu = x2_cpu.conv_transpose2d(&w3_cpu, 3, 2, 1, 1)?;
    let diff3 = (y3.to_device(&cpu)? - &y3_cpu)?.abs()?.max_all()?.to_scalar::<f32>()?;
    assert!(diff3 < 1e-3, "conv_transpose2d (pad 3) mismatch {diff3}");
    println!("conv2d / conv_transpose2d OK");

    // Micro-benchmark a few iterations.
    let xb = Tensor::randn(0f32, 1.0, (1, 12, 256, 256), &device)?;
    for _ in 0..10 {
        let start_time = std::time::Instant::now();
        let _y = xb.conv2d(&w, 1, 1, 1, 1)?;
        device.synchronize()?;
        println!("conv2d(12->32, 256x256): {:?}", start_time.elapsed());
    }

    println!("dtype f32 only; bf16/f16 ops supported via cast kernels");
    let _ = DType::F32;
    Ok(())
}
