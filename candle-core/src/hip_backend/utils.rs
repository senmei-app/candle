//! Helper functions to plug HIP kernels into candle.
use crate::{Layout, Result, WithDType};
use rocm_rs::hip::kernel::AsKernelArg;
use rocm_rs::hip::{DeviceMemory, Dim3, Stream, memory::KernelArg};

use super::device::HipDevice;
use super::{HipError, HipStorageSlice as S, WrapErr};

/// Raw device-pointer kernel argument (used for offset views into a `DeviceMemory`).
#[derive(Clone, Copy)]
pub struct PtrArg(pub *mut std::ffi::c_void);

impl AsKernelArg for PtrArg {
    fn as_kernel_arg(&self) -> KernelArg {
        self.0
    }
}

impl<T> From<&DeviceMemory<T>> for PtrArg {
    fn from(m: &DeviceMemory<T>) -> Self {
        PtrArg(m.as_ptr())
    }
}

impl<T> From<&mut DeviceMemory<T>> for PtrArg {
    fn from(m: &mut DeviceMemory<T>) -> Self {
        PtrArg(m.as_ptr())
    }
}

/// A null device pointer (contiguous kernel fast-path).
pub struct NullPtr;
impl AsKernelArg for NullPtr {
    fn as_kernel_arg(&self) -> KernelArg {
        std::ptr::null_mut()
    }
}

pub const BLOCK_DIM: u32 = 256;

pub fn launch1d(
    func: &rocm_rs::hip::Function,
    stream: &Stream,
    num_elems: usize,
    args: &mut [KernelArg],
) -> Result<()> {
    let grid = ((num_elems as u64 + BLOCK_DIM as u64 - 1) / BLOCK_DIM as u64) as u32;
    let block = Dim3 {
        x: BLOCK_DIM,
        y: 1,
        z: 1,
    };
    func.launch(
        Dim3 {
            x: grid.max(1),
            y: 1,
            z: 1,
        },
        block,
        0,
        Some(stream),
        args,
    )
    .w()
}

pub trait Map1 {
    fn f<T: WithDType + 'static>(
        &self,
        src: &DeviceMemory<T>,
        dev: &HipDevice,
        layout: &Layout,
    ) -> Result<DeviceMemory<T>>;

    fn map(&self, s: &S, d: &HipDevice, l: &Layout) -> Result<S> {
        let out = match s {
            S::U8(s) => S::U8(self.f(s, d, l)?),
            S::U32(s) => S::U32(self.f(s, d, l)?),
            S::I16(s) => S::I16(self.f(s, d, l)?),
            S::I32(s) => S::I32(self.f(s, d, l)?),
            S::I64(s) => S::I64(self.f(s, d, l)?),
            S::BF16(s) => S::BF16(self.f(s, d, l)?),
            S::F16(s) => S::F16(self.f(s, d, l)?),
            S::F32(s) => S::F32(self.f(s, d, l)?),
            S::F64(s) => S::F64(self.f(s, d, l)?),
            S::F8E4M3(_) => {
                return Err(HipError::UnsupportedDtype {
                    dtype: crate::DType::F8E4M3,
                    op: "hip Map1",
                }
                .into())
            }
        };
        Ok(out)
    }
}

pub trait Map2 {
    fn f<T: WithDType + 'static>(
        &self,
        src1: &DeviceMemory<T>,
        layout1: &Layout,
        src2: &DeviceMemory<T>,
        layout2: &Layout,
        dev: &HipDevice,
    ) -> Result<DeviceMemory<T>>;

    fn map(&self, s1: &S, l1: &Layout, s2: &S, l2: &Layout, d: &HipDevice) -> Result<S> {
        let out = match (s1, s2) {
            (S::U8(s1), S::U8(s2)) => S::U8(self.f(s1, l1, s2, l2, d)?),
            (S::U32(s1), S::U32(s2)) => S::U32(self.f(s1, l1, s2, l2, d)?),
            (S::I16(s1), S::I16(s2)) => S::I16(self.f(s1, l1, s2, l2, d)?),
            (S::I32(s1), S::I32(s2)) => S::I32(self.f(s1, l1, s2, l2, d)?),
            (S::I64(s1), S::I64(s2)) => S::I64(self.f(s1, l1, s2, l2, d)?),
            (S::BF16(s1), S::BF16(s2)) => S::BF16(self.f(s1, l1, s2, l2, d)?),
            (S::F16(s1), S::F16(s2)) => S::F16(self.f(s1, l1, s2, l2, d)?),
            (S::F32(s1), S::F32(s2)) => S::F32(self.f(s1, l1, s2, l2, d)?),
            (S::F64(s1), S::F64(s2)) => S::F64(self.f(s1, l1, s2, l2, d)?),
            _ => {
                return Err(HipError::InternalError("dtype mismatch in binary op".to_string())
                    .into())
            }
        };
        Ok(out)
    }
}

pub trait Map1Any {
    fn f<T: WithDType + 'static, W: Fn(DeviceMemory<T>) -> S>(
        &self,
        src: &DeviceMemory<T>,
        dev: &HipDevice,
        layout: &Layout,
        wrap: W,
    ) -> Result<S>;

    fn map(&self, s: &S, d: &HipDevice, l: &Layout) -> Result<S> {
        let out = match s {
            S::U8(s) => self.f(s, d, l, S::U8)?,
            S::U32(s) => self.f(s, d, l, S::U32)?,
            S::I16(s) => self.f(s, d, l, S::I16)?,
            S::I32(s) => self.f(s, d, l, S::I32)?,
            S::I64(s) => self.f(s, d, l, S::I64)?,
            S::BF16(s) => self.f(s, d, l, S::BF16)?,
            S::F16(s) => self.f(s, d, l, S::F16)?,
            S::F32(s) => self.f(s, d, l, S::F32)?,
            S::F64(s) => self.f(s, d, l, S::F64)?,
            S::F8E4M3(_) => {
                return Err(HipError::UnsupportedDtype {
                    dtype: crate::DType::F8E4M3,
                    op: "hip Map1Any",
                }
                .into())
            }
        };
        Ok(out)
    }
}
