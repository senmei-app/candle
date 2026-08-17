//! Implementation of `BackendStorage` for the HIP/ROCm device.
use crate::backend::{BackendDevice, BackendStorage};
use crate::op::{BinaryOpT, CmpOp, ReduceOp, UnaryOpT};
use crate::{CpuStorage, DType, Layout, Result, Shape, WithDType};
use candle_kernels as kernels;
use half::{bf16, f16};
use rocm_rs::hip::memory::KernelArg;
use rocm_rs::hip::{DeviceMemory, Dim3, Stream};

mod device;
mod error;
mod utils;
pub use device::{DeviceId, HipDevice, HipFunc};
pub use error::{HipError, WrapErr};
pub use utils::{Map1, Map1Any, Map2, NullPtr, PtrArg, BLOCK_DIM};

pub type S = HipStorageSlice;

pub enum HipStorageSlice {
    U8(DeviceMemory<u8>),
    U32(DeviceMemory<u32>),
    I16(DeviceMemory<i16>),
    I32(DeviceMemory<i32>),
    I64(DeviceMemory<i64>),
    BF16(DeviceMemory<bf16>),
    F16(DeviceMemory<f16>),
    F32(DeviceMemory<f32>),
    F64(DeviceMemory<f64>),
    F8E4M3(DeviceMemory<u8>),
}

pub struct HipStorage {
    pub slice: HipStorageSlice,
    pub device: HipDevice,
}

impl std::fmt::Debug for HipStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "HipStorage")
    }
}

pub fn kernel_name<T: WithDType>(root: &str) -> String {
    format!("{root}_{}", T::DTYPE.as_str())
}

/// Build a kernel launch argument array.
///
/// `hipModuleLaunchKernel` expects `kernelParams[i]` to be a *pointer to the
/// argument value* (not the value itself), so all values are copied into a fixed
/// buffer whose addresses are stable for the duration of the launch.
pub struct Args {
    values: Box<[u64; 64]>,
    n: usize,
    args: Vec<KernelArg>,
}

impl Args {
    pub fn new() -> Self {
        Self {
            values: Box::new([0u64; 64]),
            n: 0,
            args: Vec::with_capacity(32),
        }
    }

    /// Store the raw bytes of `v` (padded to 8) and append a pointer to them.
    pub fn push<T: Copy>(&mut self, v: &T) {
        let size = std::mem::size_of::<T>();
        assert!(size <= 8 && self.n < 64, "kernel arg too large or too many args");
        let mut buf = [0u8; 8];
        unsafe {
            std::ptr::copy_nonoverlapping((v as *const T).cast::<u8>(), buf.as_mut_ptr(), size);
        }
        self.values[self.n] = u64::from_ne_bytes(buf);
        self.args.push(&self.values[self.n] as *const u64 as KernelArg);
        self.n += 1;
    }

    pub fn push_ptr(&mut self, v: *mut std::ffi::c_void) {
        self.push(&v);
    }

    pub fn as_mut_slice(&mut self) -> &mut [KernelArg] {
        &mut self.args
    }
}

pub fn launch(
    func: &rocm_rs::hip::Function,
    stream: &Stream,
    grid: u32,
    block: u32,
    args: &mut [KernelArg],
) -> Result<()> {
    func.launch(
        Dim3 {
            x: grid.max(1),
            y: 1,
            z: 1,
        },
        Dim3 {
            x: block.max(1),
            y: 1,
            z: 1,
        },
        0,
        Some(stream),
        args,
    )
    .w()
}

pub fn launch1d(
    func: &rocm_rs::hip::Function,
    stream: &Stream,
    num_elems: usize,
    args: &mut [KernelArg],
) -> Result<()> {
    let grid = ((num_elems as u64 + BLOCK_DIM as u64 - 1) / BLOCK_DIM as u64) as u32;
    launch(func, stream, grid, BLOCK_DIM, args)
}

/// The `info` kernel argument: a device pointer to `[dims, strides]` or null.
pub enum InfoArg {
    Null,
    Ptr(DeviceMemory<usize>),
}

impl InfoArg {
    fn kernel_arg(&self) -> KernelArg {
        match self {
            InfoArg::Null => std::ptr::null_mut(),
            InfoArg::Ptr(m) => m.as_ptr(),
        }
    }
}

fn params_from_layout(dev: &HipDevice, l: &Layout) -> Result<InfoArg> {
    if l.is_contiguous() {
        Ok(InfoArg::Null)
    } else {
        let data = [l.dims(), l.stride()].concat();
        Ok(InfoArg::Ptr(dev.copy_htod(&data)?))
    }
}

fn offset_ptr<T>(m: &DeviceMemory<T>, offset: usize) -> KernelArg {
    unsafe { m.as_ptr().byte_add(offset * std::mem::size_of::<T>()) }
}

trait CloneD2H<T: Copy + Default> {
    fn clone_dtoh(&self) -> Result<Vec<T>>;
}
impl<T: Copy + Default> CloneD2H<T> for DeviceMemory<T> {
    fn clone_dtoh(&self) -> Result<Vec<T>> {
        let mut out = vec![T::default(); self.count()];
        self.copy_to_host(&mut out).w()?;
        Ok(out)
    }
}

trait TryClone<T> {
    fn try_clone(&self) -> Result<DeviceMemory<T>>;
}
impl<T: Copy> TryClone<T> for DeviceMemory<T> {
    fn try_clone(&self) -> Result<DeviceMemory<T>> {
        let mut out = DeviceMemory::new(self.count()).w()?;
        out.copy_from_device(self).w()?;
        Ok(out)
    }
}

fn storage_to_cpu(s: &S) -> Result<CpuStorage> {
    match s {
        S::U8(m) => Ok(CpuStorage::U8(m.clone_dtoh()?)),
        S::U32(m) => Ok(CpuStorage::U32(m.clone_dtoh()?)),
        S::I16(m) => Ok(CpuStorage::I16(m.clone_dtoh()?)),
        S::I32(m) => Ok(CpuStorage::I32(m.clone_dtoh()?)),
        S::I64(m) => Ok(CpuStorage::I64(m.clone_dtoh()?)),
        S::BF16(m) => Ok(CpuStorage::BF16(m.clone_dtoh()?)),
        S::F16(m) => Ok(CpuStorage::F16(m.clone_dtoh()?)),
        S::F32(m) => Ok(CpuStorage::F32(m.clone_dtoh()?)),
        S::F64(m) => Ok(CpuStorage::F64(m.clone_dtoh()?)),
        S::F8E4M3(m) => {
            let bytes: Vec<u8> = m.clone_dtoh()?;
            Ok(CpuStorage::F8E4M3(
                bytes
                    .into_iter()
                    .map(float8::F8E4M3::from_bits)
                    .collect(),
            ))
        }
    }
}

pub struct Affine(f64, f64);
impl Map1 for Affine {
    fn f<T: WithDType + 'static>(
        &self,
        src: &DeviceMemory<T>,
        dev: &HipDevice,
        layout: &Layout,
    ) -> Result<DeviceMemory<T>> {
        let shape = layout.shape();
        let dims = shape.dims();
        let el = shape.elem_count();
        let ndims = dims.len();
        let info = params_from_layout(dev, layout)?;
        let src_ptr = offset_ptr(src, layout.start_offset());
        let func = dev.get_or_load_func(&kernel_name::<T>("affine"), &kernels::AFFINE)?;
        let mut out = dev.alloc::<T>(el)?;
        let mul = T::from_f64(self.0);
        let add = T::from_f64(self.1);
        let mut args = Args::new();
        args.push(&el);
        args.push(&ndims);
        args.push_ptr(info.kernel_arg());
        args.push_ptr(src_ptr);
        args.push_ptr(out.as_ptr());
        args.push(&mul);
        args.push(&add);
        launch1d(&func, dev.stream(), el, args.as_mut_slice())?;
        Ok(out)
    }
}

pub struct Powf(f64);
impl Map1 for Powf {
    fn f<T: WithDType + 'static>(
        &self,
        src: &DeviceMemory<T>,
        dev: &HipDevice,
        layout: &Layout,
    ) -> Result<DeviceMemory<T>> {
        let shape = layout.shape();
        let dims = shape.dims();
        let el = shape.elem_count();
        let ndims = dims.len();
        let info = params_from_layout(dev, layout)?;
        let src_ptr = offset_ptr(src, layout.start_offset());
        let func = dev.get_or_load_func(&kernel_name::<T>("upowf"), &kernels::UNARY)?;
        let mut out = dev.alloc::<T>(el)?;
        let pow = T::from_f64(self.0);
        let mut args = Args::new();
        args.push(&el);
        args.push(&ndims);
        args.push_ptr(info.kernel_arg());
        args.push(&pow);
        args.push_ptr(src_ptr);
        args.push_ptr(out.as_ptr());
        launch1d(&func, dev.stream(), el, args.as_mut_slice())?;
        Ok(out)
    }
}

pub struct Elu(f64);
impl Map1 for Elu {
    fn f<T: WithDType + 'static>(
        &self,
        src: &DeviceMemory<T>,
        dev: &HipDevice,
        layout: &Layout,
    ) -> Result<DeviceMemory<T>> {
        let shape = layout.shape();
        let dims = shape.dims();
        let el = shape.elem_count();
        let ndims = dims.len();
        let info = params_from_layout(dev, layout)?;
        let src_ptr = offset_ptr(src, layout.start_offset());
        let func = dev.get_or_load_func(&kernel_name::<T>("uelu"), &kernels::UNARY)?;
        let mut out = dev.alloc::<T>(el)?;
        let alpha = T::from_f64(self.0);
        let mut args = Args::new();
        args.push(&el);
        args.push(&ndims);
        args.push_ptr(info.kernel_arg());
        args.push(&alpha);
        args.push_ptr(src_ptr);
        args.push_ptr(out.as_ptr());
        launch1d(&func, dev.stream(), el, args.as_mut_slice())?;
        Ok(out)
    }
}

pub struct FastReduce<'a>(pub &'a [usize], pub ReduceOp);
impl Map1Any for FastReduce<'_> {
    fn f<T: WithDType + 'static, W: Fn(DeviceMemory<T>) -> S>(
        &self,
        src: &DeviceMemory<T>,
        dev: &HipDevice,
        layout: &Layout,
        wrap: W,
    ) -> Result<S> {
        let src_stride = layout.stride();
        let src_dims = layout.shape().dims();
        let src_el: usize = src_dims.iter().product();
        let mut dims = vec![];
        let mut stride = vec![];
        let mut dst_el: usize = 1;
        for (dim_idx, &d) in src_dims.iter().enumerate() {
            if !self.0.contains(&dim_idx) {
                dst_el *= d;
                dims.push(d);
                stride.push(src_stride[dim_idx]);
            }
        }
        for &dim_idx in self.0.iter() {
            dims.push(src_dims[dim_idx]);
            stride.push(src_stride[dim_idx]);
        }
        let el_to_sum_per_block = src_el / dst_el;
        let block_dim = usize::min(1024, el_to_sum_per_block)
            .next_power_of_two()
            .max(1);
        let info = dev.copy_htod(&[dims.as_slice(), stride.as_slice()].concat())?;
        let src_ptr = offset_ptr(src, layout.start_offset());
        let (name, check_empty, return_index) = match self.1 {
            ReduceOp::Sum => ("fast_sum", false, false),
            ReduceOp::Min => ("fast_min", true, false),
            ReduceOp::Max => ("fast_max", true, false),
            ReduceOp::ArgMin => ("fast_argmin", true, true),
            ReduceOp::ArgMax => ("fast_argmax", true, true),
        };
        if check_empty && layout.shape().elem_count() == 0 {
            Err(crate::Error::EmptyTensor { op: "reduce" }.bt())?
        }
        let func = dev.get_or_load_func(&kernel_name::<T>(name), &kernels::REDUCE)?;
        let ndims = src_dims.len();
        let dst_el_u32 = dst_el as u32;
        let block_u32 = block_dim as u32;
        let mut args = Args::new();
        args.push(&src_el);
        args.push(&el_to_sum_per_block);
        args.push(&ndims);
        args.push_ptr(info.as_ptr());
        args.push_ptr(src_ptr);
        if return_index {
            let mut out = dev.alloc::<u32>(dst_el)?;
            args.push_ptr(out.as_ptr());
            launch(&func, dev.stream(), dst_el_u32, block_u32, args.as_mut_slice())?;
            Ok(S::U32(out))
        } else {
            let mut out = dev.alloc::<T>(dst_el)?;
            args.push_ptr(out.as_ptr());
            launch(&func, dev.stream(), dst_el_u32, block_u32, args.as_mut_slice())?;
            Ok(wrap(out))
        }
    }
}

impl<U: UnaryOpT> Map1 for U {
    fn f<T: WithDType + 'static>(
        &self,
        src: &DeviceMemory<T>,
        dev: &HipDevice,
        layout: &Layout,
    ) -> Result<DeviceMemory<T>> {
        let shape = layout.shape();
        let dims = shape.dims();
        let el = shape.elem_count();
        let ndims = dims.len();
        let info = params_from_layout(dev, layout)?;
        let src_ptr = offset_ptr(src, layout.start_offset());
        let func = dev.get_or_load_func(&kernel_name::<T>(U::KERNEL), &kernels::UNARY)?;
        let mut out = dev.alloc::<T>(el)?;
        let mut args = Args::new();
        args.push(&el);
        args.push(&ndims);
        args.push_ptr(info.kernel_arg());
        args.push_ptr(src_ptr);
        args.push_ptr(out.as_ptr());
        launch1d(&func, dev.stream(), el, args.as_mut_slice())?;
        Ok(out)
    }
}

impl<B: BinaryOpT> Map2 for B {
    fn f<T: WithDType + 'static>(
        &self,
        lhs: &DeviceMemory<T>,
        lhs_l: &Layout,
        rhs: &DeviceMemory<T>,
        rhs_l: &Layout,
        dev: &HipDevice,
    ) -> Result<DeviceMemory<T>> {
        let dims = lhs_l.shape().dims();
        let el = lhs_l.shape().elem_count();
        let ndims = dims.len();
        let info = dev.copy_htod(&[dims, lhs_l.stride(), rhs_l.stride()].concat())?;
        let lhs_ptr = offset_ptr(lhs, lhs_l.start_offset());
        let rhs_ptr = offset_ptr(rhs, rhs_l.start_offset());
        let func = dev.get_or_load_func(&kernel_name::<T>(B::KERNEL), &kernels::BINARY)?;
        let mut out = dev.alloc::<T>(el)?;
        let mut args = Args::new();
        args.push(&el);
        args.push(&ndims);
        args.push_ptr(info.as_ptr());
        args.push_ptr(lhs_ptr);
        args.push_ptr(rhs_ptr);
        args.push_ptr(out.as_ptr());
        launch1d(&func, dev.stream(), el, args.as_mut_slice())?;
        Ok(out)
    }
}

struct ConstSet(crate::scalar::Scalar);

impl ConstSet {
    fn to_f64(&self) -> f64 {
        use crate::scalar::Scalar;
        match self.0 {
            Scalar::U8(v) => v as f64,
            Scalar::U32(v) => v as f64,
            Scalar::I16(v) => v as f64,
            Scalar::I32(v) => v as f64,
            Scalar::I64(v) => v as f64,
            Scalar::F32(v) => v as f64,
            Scalar::F64(v) => v,
            Scalar::F16(v) => v.to_f32() as f64,
            Scalar::BF16(v) => v.to_f32() as f64,
            Scalar::F8E4M3(v) => v.to_f32() as f64,
        }
    }

    fn apply<T: WithDType + 'static>(
        &self,
        dev: &HipDevice,
        layout: &Layout,
        out: &mut DeviceMemory<T>,
    ) -> Result<()> {
        let shape = layout.shape();
        let dims = shape.dims();
        let el = shape.elem_count();
        let ndims = dims.len();
        let info = params_from_layout(dev, layout)?;
        let value = T::from_f64(self.to_f64());
        let mut args = Args::new();
        args.push(&el);
        args.push(&ndims);
        args.push_ptr(info.kernel_arg());
        args.push(&value);
        args.push_ptr(out.as_ptr());
        let func = dev.get_or_load_func(&kernel_name::<T>("const_set"), &kernels::FILL)?;
        launch1d(&func, dev.stream(), el, args.as_mut_slice())
    }
}

fn cast_run<S: Copy + WithDType + 'static, D: Copy + WithDType + 'static>(
    dev: &HipDevice,
    src: &DeviceMemory<S>,
    layout: &Layout,
    kernel: &str,
) -> Result<DeviceMemory<D>> {
    let shape = layout.shape();
    let dims = shape.dims();
    let el = shape.elem_count();
    let ndims = dims.len();
    let info = params_from_layout(dev, layout)?;
    let src_ptr = offset_ptr(src, layout.start_offset());
    let func = dev.get_or_load_func(kernel, &kernels::CAST)?;
    let mut out = dev.alloc::<D>(el)?;
    let mut args = Args::new();
    args.push(&el);
    args.push(&ndims);
    args.push_ptr(info.kernel_arg());
    args.push_ptr(src_ptr);
    args.push_ptr(out.as_ptr());
    launch1d(&func, dev.stream(), el, args.as_mut_slice())?;
    Ok(out)
}

impl BackendStorage for HipStorage {
    type Device = HipDevice;

    fn try_clone(&self, _: &Layout) -> Result<Self> {
        let slice = match &self.slice {
            S::U8(s) => S::U8(s.try_clone()?),
            S::U32(s) => S::U32(s.try_clone()?),
            S::I16(s) => S::I16(s.try_clone()?),
            S::I32(s) => S::I32(s.try_clone()?),
            S::I64(s) => S::I64(s.try_clone()?),
            S::BF16(s) => S::BF16(s.try_clone()?),
            S::F16(s) => S::F16(s.try_clone()?),
            S::F32(s) => S::F32(s.try_clone()?),
            S::F64(s) => S::F64(s.try_clone()?),
            S::F8E4M3(s) => S::F8E4M3(s.try_clone()?),
        };
        Ok(Self {
            slice,
            device: self.device.clone(),
        })
    }

    fn dtype(&self) -> DType {
        match &self.slice {
            S::U8(_) => DType::U8,
            S::U32(_) => DType::U32,
            S::I16(_) => DType::I16,
            S::I32(_) => DType::I32,
            S::I64(_) => DType::I64,
            S::BF16(_) => DType::BF16,
            S::F16(_) => DType::F16,
            S::F32(_) => DType::F32,
            S::F64(_) => DType::F64,
            S::F8E4M3(_) => DType::F8E4M3,
        }
    }

    fn device(&self) -> &Self::Device {
        &self.device
    }

    fn to_cpu_storage(&self) -> Result<CpuStorage> {
        storage_to_cpu(&self.slice)
    }

    fn affine(&self, l: &Layout, mul: f64, add: f64) -> Result<Self> {
        let slice = Affine(mul, add).map(&self.slice, &self.device, l)?;
        Ok(Self {
            slice,
            device: self.device.clone(),
        })
    }

    fn powf(&self, l: &Layout, pow: f64) -> Result<Self> {
        let slice = Powf(pow).map(&self.slice, &self.device, l)?;
        Ok(Self {
            slice,
            device: self.device.clone(),
        })
    }

    fn elu(&self, l: &Layout, alpha: f64) -> Result<Self> {
        let slice = Elu(alpha).map(&self.slice, &self.device, l)?;
        Ok(Self {
            slice,
            device: self.device.clone(),
        })
    }

    fn reduce_op(&self, r: ReduceOp, l: &Layout, dims: &[usize]) -> Result<Self> {
        let slice = FastReduce(dims, r).map(&self.slice, &self.device, l)?;
        Ok(Self {
            slice,
            device: self.device.clone(),
        })
    }

    fn cmp(&self, op: CmpOp, rhs: &Self, l: &Layout, rl: &Layout) -> Result<Self> {
        // Comparisons return a u8 mask; run them on the CPU backend for now.
        let lhs = storage_to_cpu(&self.slice)?;
        let rhs = storage_to_cpu(&rhs.slice)?;
        let out = lhs.cmp(op, &rhs, l, rl)?;
        self.device.storage_from_cpu_storage(&out)
    }

    fn to_dtype(&self, l: &Layout, dtype: DType) -> Result<Self> {
        let src_dtype = self.dtype();
        let kernel = format!("cast_{}_{}", src_dtype.as_str(), dtype.as_str());
        let slice = match (&self.slice, dtype) {
            (S::U8(s), DType::F32) => S::F32(cast_run::<u8, f32>(&self.device, s, l, &kernel)?),
            (S::U8(s), DType::U32) => S::U32(cast_run::<u8, u32>(&self.device, s, l, &kernel)?),
            (S::U32(s), DType::F32) => S::F32(cast_run::<u32, f32>(&self.device, s, l, &kernel)?),
            (S::U32(s), DType::U8) => S::U8(cast_run::<u32, u8>(&self.device, s, l, &kernel)?),
            (S::I64(s), DType::F32) => S::F32(cast_run::<i64, f32>(&self.device, s, l, &kernel)?),
            (S::F32(s), DType::F64) => S::F64(cast_run::<f32, f64>(&self.device, s, l, &kernel)?),
            (S::F32(s), DType::U8) => S::U8(cast_run::<f32, u8>(&self.device, s, l, &kernel)?),
            (S::F32(s), DType::U32) => S::U32(cast_run::<f32, u32>(&self.device, s, l, &kernel)?),
            (S::F32(s), DType::I64) => S::I64(cast_run::<f32, i64>(&self.device, s, l, &kernel)?),
            (S::F32(s), DType::BF16) => {
                S::BF16(cast_run::<f32, bf16>(&self.device, s, l, &kernel)?)
            }
            (S::F32(s), DType::F16) => S::F16(cast_run::<f32, f16>(&self.device, s, l, &kernel)?),
            (S::F64(s), DType::F32) => S::F32(cast_run::<f64, f32>(&self.device, s, l, &kernel)?),
            (S::F64(s), DType::BF16) => {
                S::BF16(cast_run::<f64, bf16>(&self.device, s, l, &kernel)?)
            }
            (S::F64(s), DType::F16) => S::F16(cast_run::<f64, f16>(&self.device, s, l, &kernel)?),
            (S::F16(s), DType::F32) => S::F32(cast_run::<f16, f32>(&self.device, s, l, &kernel)?),
            (S::F16(s), DType::F64) => S::F64(cast_run::<f16, f64>(&self.device, s, l, &kernel)?),
            (S::F16(s), DType::BF16) => {
                S::BF16(cast_run::<f16, bf16>(&self.device, s, l, &kernel)?)
            }
            (S::BF16(s), DType::F32) => {
                S::F32(cast_run::<bf16, f32>(&self.device, s, l, &kernel)?)
            }
            (S::BF16(s), DType::F64) => {
                S::F64(cast_run::<bf16, f64>(&self.device, s, l, &kernel)?)
            }
            (S::BF16(s), DType::F16) => {
                S::F16(cast_run::<bf16, f16>(&self.device, s, l, &kernel)?)
            }
            _ => {
                let cpu = storage_to_cpu(&self.slice)?;
                let out = cpu.to_dtype(l, dtype)?;
                self.device
                    .storage_from_cpu_storage(&out)
                    .map(|st| st.slice)?
            }
        };
        Ok(Self {
            slice,
            device: self.device.clone(),
        })
    }

    fn unary_impl<U: UnaryOpT>(&self, l: &Layout) -> Result<Self> {
        let slice = U::V.map(&self.slice, &self.device, l)?;
        Ok(Self {
            slice,
            device: self.device.clone(),
        })
    }

    fn binary_impl<B: BinaryOpT>(&self, rhs: &Self, l: &Layout, rl: &Layout) -> Result<Self> {
        let slice = B::V.map(&self.slice, l, &rhs.slice, rl, &self.device)?;
        Ok(Self {
            slice,
            device: self.device.clone(),
        })
    }

    fn where_cond(
        &self,
        l: &Layout,
        t: &Self,
        tl: &Layout,
        f: &Self,
        fl: &Layout,
    ) -> Result<Self> {
        let cond = storage_to_cpu(&self.slice)?;
        let t_cpu = t.to_cpu_storage()?;
        let f_cpu = f.to_cpu_storage()?;
        let out = cond.where_cond(l, &t_cpu, tl, &f_cpu, fl)?;
        self.device.storage_from_cpu_storage(&out)
    }

    fn conv1d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConv1D,
    ) -> Result<Self> {
        let inp = self.to_cpu_storage()?;
        let ker = kernel.to_cpu_storage()?;
        let out = inp.conv1d(l, &ker, kernel_l, params)?;
        self.device.storage_from_cpu_storage(&out)
    }

    fn conv_transpose1d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConvTranspose1D,
    ) -> Result<Self> {
        let inp = self.to_cpu_storage()?;
        let ker = kernel.to_cpu_storage()?;
        let out = inp.conv_transpose1d(l, &ker, kernel_l, params)?;
        self.device.storage_from_cpu_storage(&out)
    }

    fn conv2d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConv2D,
    ) -> Result<Self> {
        conv2d_hip(self, l, kernel, kernel_l, params)
    }

    fn conv_transpose2d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConvTranspose2D,
    ) -> Result<Self> {
        conv_transpose2d_hip(self, l, kernel, kernel_l, params)
    }

    fn avg_pool2d(&self, l: &Layout, kh: (usize, usize), stride: (usize, usize)) -> Result<Self> {
        let inp = self.to_cpu_storage()?;
        let out = inp.avg_pool2d(l, kh, stride)?;
        self.device.storage_from_cpu_storage(&out)
    }

    fn max_pool2d(&self, l: &Layout, kh: (usize, usize), stride: (usize, usize)) -> Result<Self> {
        let inp = self.to_cpu_storage()?;
        let out = inp.max_pool2d(l, kh, stride)?;
        self.device.storage_from_cpu_storage(&out)
    }

    fn upsample_nearest1d(&self, l: &Layout, f: usize) -> Result<Self> {
        let inp = self.to_cpu_storage()?;
        let out = inp.upsample_nearest1d(l, f)?;
        self.device.storage_from_cpu_storage(&out)
    }

    fn upsample_nearest2d(&self, l: &Layout, fh: usize, fw: usize) -> Result<Self> {
        let inp = self.to_cpu_storage()?;
        let out = inp.upsample_nearest2d(l, fh, fw)?;
        self.device.storage_from_cpu_storage(&out)
    }

    fn upsample_bilinear2d(
        &self,
        l: &Layout,
        fh: usize,
        fw: usize,
        align_corners: bool,
        scales_h: Option<f64>,
        scales_w: Option<f64>,
    ) -> Result<Self> {
        let inp = self.to_cpu_storage()?;
        let out = inp.upsample_bilinear2d(l, fh, fw, align_corners, scales_h, scales_w)?;
        self.device.storage_from_cpu_storage(&out)
    }

    fn gather(&self, l: &Layout, idx: &Self, idx_l: &Layout, dim: usize) -> Result<Self> {
        let inp = self.to_cpu_storage()?;
        let idx_cpu = idx.to_cpu_storage()?;
        let out = inp.gather(l, &idx_cpu, idx_l, dim)?;
        self.device.storage_from_cpu_storage(&out)
    }

    fn scatter_set(
        &mut self,
        l: &Layout,
        idx: &Self,
        idx_l: &Layout,
        src: &Self,
        src_l: &Layout,
        dim: usize,
    ) -> Result<()> {
        let mut inp = self.to_cpu_storage()?;
        let idx_cpu = idx.to_cpu_storage()?;
        let src_cpu = src.to_cpu_storage()?;
        inp.scatter_set(l, &idx_cpu, idx_l, &src_cpu, src_l, dim)?;
        *self = self.device.storage_from_cpu_storage(&inp)?;
        Ok(())
    }

    fn scatter_add_set(
        &mut self,
        l: &Layout,
        idx: &Self,
        idx_l: &Layout,
        src: &Self,
        src_l: &Layout,
        dim: usize,
    ) -> Result<()> {
        let mut inp = self.to_cpu_storage()?;
        let idx_cpu = idx.to_cpu_storage()?;
        let src_cpu = src.to_cpu_storage()?;
        inp.scatter_add_set(l, &idx_cpu, idx_l, &src_cpu, src_l, dim)?;
        *self = self.device.storage_from_cpu_storage(&inp)?;
        Ok(())
    }

    fn index_select(&self, idx: &Self, idx_l: &Layout, l: &Layout, dim: usize) -> Result<Self> {
        let inp = self.to_cpu_storage()?;
        let idx_cpu = idx.to_cpu_storage()?;
        let out = inp.index_select(&idx_cpu, idx_l, l, dim)?;
        self.device.storage_from_cpu_storage(&out)
    }

    fn index_add(
        &self,
        l: &Layout,
        idx: &Self,
        idx_l: &Layout,
        src: &Self,
        src_l: &Layout,
        dim: usize,
    ) -> Result<Self> {
        let inp = self.to_cpu_storage()?;
        let idx_cpu = idx.to_cpu_storage()?;
        let src_cpu = src.to_cpu_storage()?;
        let out = inp.index_add(l, &idx_cpu, idx_l, &src_cpu, src_l, dim)?;
        self.device.storage_from_cpu_storage(&out)
    }

    fn matmul(
        &self,
        rhs: &Self,
        mnk_b: (usize, usize, usize, usize),
        l: &Layout,
        rl: &Layout,
    ) -> Result<Self> {
        let inp = self.to_cpu_storage()?;
        let rhs_cpu = rhs.to_cpu_storage()?;
        let out = inp.matmul(&rhs_cpu, mnk_b, l, rl)?;
        self.device.storage_from_cpu_storage(&out)
    }

    fn copy_strided_src(&self, dst: &mut Self, dst_offset: usize, src_l: &Layout) -> Result<()> {
        let mut dst_cpu = dst.to_cpu_storage()?;
        let src_cpu = self.to_cpu_storage()?;
        src_cpu.copy_strided_src(&mut dst_cpu, dst_offset, src_l)?;
        *dst = self.device.storage_from_cpu_storage(&dst_cpu)?;
        Ok(())
    }

    fn copy2d(
        &self,
        dst: &mut Self,
        d1: usize,
        d2: usize,
        src_stride1: usize,
        dst_stride1: usize,
        src_offset: usize,
        dst_offset: usize,
    ) -> Result<()> {
        let mut dst_cpu = dst.to_cpu_storage()?;
        let src_cpu = self.to_cpu_storage()?;
        src_cpu.copy2d(
            &mut dst_cpu,
            d1,
            d2,
            src_stride1,
            dst_stride1,
            src_offset,
            dst_offset,
        )?;
        *dst = self.device.storage_from_cpu_storage(&dst_cpu)?;
        Ok(())
    }

    fn const_set(&mut self, s: crate::scalar::Scalar, layout: &Layout) -> Result<()> {
        let dev = &self.device;
        let cs = ConstSet(s);
        match &mut self.slice {
            S::U8(out) => cs.apply::<u8>(dev, layout, out),
            S::U32(out) => cs.apply::<u32>(dev, layout, out),
            S::I16(out) => cs.apply::<i16>(dev, layout, out),
            S::I32(out) => cs.apply::<i32>(dev, layout, out),
            S::I64(out) => cs.apply::<i64>(dev, layout, out),
            S::BF16(out) => cs.apply::<bf16>(dev, layout, out),
            S::F16(out) => cs.apply::<f16>(dev, layout, out),
            S::F32(out) => cs.apply::<f32>(dev, layout, out),
            S::F64(out) => cs.apply::<f64>(dev, layout, out),
            S::F8E4M3(out) => cs.apply::<u8>(dev, layout, out),
        }
    }
}

fn conv2d_hip(
    inp: &HipStorage,
    l: &Layout,
    kernel: &HipStorage,
    kernel_l: &Layout,
    params: &crate::conv::ParamsConv2D,
) -> Result<HipStorage> {
    let dev = &inp.device;
    let (n, c, h, w) = l.shape().dims4()?;
    let (oc, c_per_group, kh, kw) = kernel_l.shape().dims4()?;
    let stride = params.stride;
    let padding = params.padding;
    let dilation = params.dilation;
    let groups = c / c_per_group.max(1);
    let h_out = (h + 2 * padding - dilation * (kh - 1) - 1) / stride + 1;
    let w_out = (w + 2 * padding - dilation * (kw - 1) - 1) / stride + 1;
    let out_shape = Shape::from((n, oc, h_out, w_out));
    let out_el = out_shape.elem_count();

    if !l.is_contiguous() || !kernel_l.is_contiguous() {
        let inp_cpu = inp.to_cpu_storage()?;
        let ker_cpu = kernel.to_cpu_storage()?;
        let out = inp_cpu.conv2d(l, &ker_cpu, kernel_l, params)?;
        return dev.storage_from_cpu_storage(&out);
    }

    let slice = match (&inp.slice, &kernel.slice) {
        (S::F32(input), S::F32(weight)) => S::F32(conv2d_run::<f32>(
            dev, input, weight, l, n, c, h, w, oc, kh, kw, h_out, w_out, stride, padding,
            dilation, groups, out_el, "hip_conv_f32_conv2d",
        )?),
        (S::F64(input), S::F64(weight)) => S::F64(conv2d_run::<f64>(
            dev, input, weight, l, n, c, h, w, oc, kh, kw, h_out, w_out, stride, padding,
            dilation, groups, out_el, "hip_conv_f64_conv2d",
        )?),
        _ => {
            let inp_cpu = inp.to_cpu_storage()?;
            let ker_cpu = kernel.to_cpu_storage()?;
            let out = inp_cpu.conv2d(l, &ker_cpu, kernel_l, params)?;
            return dev.storage_from_cpu_storage(&out);
        }
    };
    Ok(HipStorage {
        slice,
        device: dev.clone(),
    })
}

#[allow(clippy::too_many_arguments)]
fn conv2d_run<T: Copy + WithDType + 'static>(
    dev: &HipDevice,
    input: &DeviceMemory<T>,
    weight: &DeviceMemory<T>,
    l: &Layout,
    n: usize,
    c: usize,
    h: usize,
    w: usize,
    oc: usize,
    kh: usize,
    kw: usize,
    h_out: usize,
    w_out: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
    out_el: usize,
    kernel: &str,
) -> Result<DeviceMemory<T>> {
    let mut out = dev.alloc::<T>(out_el)?;
    let input_ptr = offset_ptr(input, l.start_offset());
    let func = dev.get_or_load_func(kernel, &kernels::CONV)?;
    let mut args = Args::new();
    args.push_ptr(input_ptr);
    args.push_ptr(weight.as_ptr());
    args.push_ptr(std::ptr::null_mut()); // no bias
    args.push_ptr(out.as_ptr());
    let ni = n as i32;
    let ci = c as i32;
    let hi = h as i32;
    let wi = w as i32;
    let oci = oc as i32;
    let khi = kh as i32;
    let kwi = kw as i32;
    let ohi = h_out as i32;
    let owi = w_out as i32;
    let ph = padding as i32;
    let pw = padding as i32;
    let sh = stride as i32;
    let sw = stride as i32;
    let dh = dilation as i32;
    let dw = dilation as i32;
    let gi = groups as i32;
    args.push(&ni);
    args.push(&ci);
    args.push(&hi);
    args.push(&wi);
    args.push(&oci);
    args.push(&khi);
    args.push(&kwi);
    args.push(&ohi);
    args.push(&owi);
    args.push(&ph);
    args.push(&pw);
    args.push(&sh);
    args.push(&sw);
    args.push(&dh);
    args.push(&dw);
    args.push(&gi);
    launch1d(&func, dev.stream(), out_el, args.as_mut_slice())?;
    Ok(out)
}

fn conv_transpose2d_hip(
    inp: &HipStorage,
    l: &Layout,
    kernel: &HipStorage,
    kernel_l: &Layout,
    params: &crate::conv::ParamsConvTranspose2D,
) -> Result<HipStorage> {
    let dev = &inp.device;
    let (n, c, h, w) = l.shape().dims4()?;
    let (_, oc_per_group, kh, kw) = kernel_l.shape().dims4()?;
    let oc = params.c_out;
    let stride = params.stride;
    let padding = params.padding;
    let dilation = params.dilation;
    let output_padding = params.output_padding;
    let groups = oc / oc_per_group.max(1);
    let h_out = (h - 1) * stride - 2 * padding + output_padding + dilation * (kh - 1) + 1;
    let w_out = (w - 1) * stride - 2 * padding + output_padding + dilation * (kw - 1) + 1;
    let out_shape = Shape::from((n, oc, h_out, w_out));
    let out_el = out_shape.elem_count();

    if !l.is_contiguous() || !kernel_l.is_contiguous() {
        let inp_cpu = inp.to_cpu_storage()?;
        let ker_cpu = kernel.to_cpu_storage()?;
        let out = inp_cpu.conv_transpose2d(l, &ker_cpu, kernel_l, params)?;
        return dev.storage_from_cpu_storage(&out);
    }

    let slice = match (&inp.slice, &kernel.slice) {
        (S::F32(input), S::F32(weight)) => S::F32(conv_transpose2d_run::<f32>(
            dev, input, weight, l, n, c, h, w, oc, kh, kw, h_out, w_out, stride, padding,
            dilation, groups, out_el, "hip_conv_f32_conv_transpose2d",
        )?),
        (S::F64(input), S::F64(weight)) => S::F64(conv_transpose2d_run::<f64>(
            dev, input, weight, l, n, c, h, w, oc, kh, kw, h_out, w_out, stride, padding,
            dilation, groups, out_el, "hip_conv_f64_conv_transpose2d",
        )?),
        _ => {
            let inp_cpu = inp.to_cpu_storage()?;
            let ker_cpu = kernel.to_cpu_storage()?;
            let out = inp_cpu.conv_transpose2d(l, &ker_cpu, kernel_l, params)?;
            return dev.storage_from_cpu_storage(&out);
        }
    };
    Ok(HipStorage {
        slice,
        device: dev.clone(),
    })
}

#[allow(clippy::too_many_arguments)]
fn conv_transpose2d_run<T: Copy + WithDType + 'static>(
    dev: &HipDevice,
    input: &DeviceMemory<T>,
    weight: &DeviceMemory<T>,
    l: &Layout,
    n: usize,
    c: usize,
    h: usize,
    w: usize,
    oc: usize,
    kh: usize,
    kw: usize,
    h_out: usize,
    w_out: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
    out_el: usize,
    kernel: &str,
) -> Result<DeviceMemory<T>> {
    let mut out = dev.alloc::<T>(out_el)?;
    let input_ptr = offset_ptr(input, l.start_offset());
    let func = dev.get_or_load_func(kernel, &kernels::CONV)?;
    let mut args = Args::new();
    args.push_ptr(input_ptr);
    args.push_ptr(weight.as_ptr());
    args.push_ptr(std::ptr::null_mut()); // no bias
    args.push_ptr(out.as_ptr());
    let ni = n as i32;
    let ci = c as i32;
    let hi = h as i32;
    let wi = w as i32;
    let oci = oc as i32;
    let khi = kh as i32;
    let kwi = kw as i32;
    let ohi = h_out as i32;
    let owi = w_out as i32;
    let ph = padding as i32;
    let pw = padding as i32;
    let sh = stride as i32;
    let sw = stride as i32;
    let dh = dilation as i32;
    let dw = dilation as i32;
    let gi = groups as i32;
    args.push(&ni);
    args.push(&ci);
    args.push(&hi);
    args.push(&wi);
    args.push(&oci);
    args.push(&khi);
    args.push(&kwi);
    args.push(&ohi);
    args.push(&owi);
    args.push(&ph);
    args.push(&pw);
    args.push(&sh);
    args.push(&sw);
    args.push(&dh);
    args.push(&dw);
    args.push(&gi);
    launch1d(&func, dev.stream(), out_el, args.as_mut_slice())?;
    Ok(out)
}
