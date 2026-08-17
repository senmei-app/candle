//! Implementation of `BackendDevice` for the HIP/ROCm device.
use crate::backend::{BackendDevice, BackendStorage};
use crate::hip_backend::{HipError, HipStorage, HipStorageSlice, WrapErr};
use crate::{CpuStorage, CpuStorageRef, DType, Result, Shape};
use candle_kernels as kernels;
use half::{bf16, f16};
use rocm_rs::hip::{Device, DeviceMemory, Function, Module, Stream};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};

/// Unique identifier for HIP devices.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DeviceId(usize);

impl DeviceId {
    fn new() -> Self {
        static COUNTER: AtomicUsize = AtomicUsize::new(1);
        Self(COUNTER.fetch_add(1, Ordering::Relaxed))
    }
}

pub struct ModuleStore {
    mdls: [Option<Arc<Module>>; kernels::ROCM_ALL_IDS.len()],
}

pub type HipFunc = Function;

/// A handle to an AMD GPU exposed through the HIP runtime.
#[derive(Clone)]
pub struct HipDevice {
    id: DeviceId,
    device: Device,
    // Shared so clones do not destroy the underlying hipStream_t on drop.
    stream: Arc<Stream>,
    modules: Arc<RwLock<ModuleStore>>,
    seed_value: Arc<RwLock<u64>>,
}

// rocm-rs handle types wrap raw pointers and don't implement Send/Sync; the
// handles are thread-safe here because candle synchronizes before handing
// buffers back. Mirrors the CUDA backend.
unsafe impl Send for HipDevice {}
unsafe impl Sync for HipDevice {}

impl std::fmt::Debug for HipDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "HipDevice({:?})", self.id)
    }
}

impl HipDevice {
    pub fn new_with_stream(ordinal: usize) -> Result<Self> {
        let device = Device::new(ordinal as i32).w()?;
        device.set_current().w()?;
        let stream = Arc::new(device.get_stream().w()?);
        let module_store = ModuleStore {
            mdls: [const { None }; kernels::ROCM_ALL_IDS.len()],
        };
        Ok(Self {
            id: DeviceId::new(),
            device,
            stream,
            modules: Arc::new(RwLock::new(module_store)),
            seed_value: Arc::new(RwLock::new(0)),
        })
    }

    pub fn id(&self) -> DeviceId {
        self.id
    }

    pub fn stream(&self) -> &Stream {
        &self.stream
    }

    pub fn alloc<T: 'static>(&self, len: usize) -> Result<DeviceMemory<T>> {
        DeviceMemory::new(len).w()
    }

    pub fn alloc_zeros<T: 'static>(&self, len: usize) -> Result<DeviceMemory<T>> {
        let mut data = self.alloc::<T>(len)?;
        data.memset(0).w()?;
        Ok(data)
    }

    pub fn copy_htod<T: Copy + 'static>(&self, data: &[T]) -> Result<DeviceMemory<T>> {
        let mut dev_data = self.alloc::<T>(data.len())?;
        dev_data.copy_from_host(data).w()?;
        Ok(dev_data)
    }

    pub fn clone_dtoh<T: Copy + Default + 'static>(
        &self,
        data: &DeviceMemory<T>,
    ) -> Result<Vec<T>> {
        let mut out = vec![T::default(); data.count()];
        data.copy_to_host(&mut out).w()?;
        Ok(out)
    }

    pub fn get_or_load_func(&self, name: &str, mdl: &kernels::RocmModule) -> Result<HipFunc> {
        {
            let ms = self.modules.read().unwrap();
            if let Some(mdl) = ms.mdls[mdl.index()].as_ref() {
                return mdl.get_function(name).w();
            }
        }
        let mut ms = self.modules.write().unwrap();
        let module = Module::load_data(mdl.co()).w()?;
        let func = module.get_function(name).w()?;
        ms.mdls[mdl.index()] = Some(Arc::new(module));
        Ok(func)
    }
}

impl BackendDevice for HipDevice {
    type Storage = HipStorage;

    fn new(ordinal: usize) -> Result<Self> {
        Self::new_with_stream(ordinal)
    }

    fn location(&self) -> crate::DeviceLocation {
        crate::DeviceLocation::Hip {
            gpu_id: self.id.0,
        }
    }

    fn same_device(&self, rhs: &Self) -> bool {
        self.id == rhs.id
    }

    fn zeros_impl(&self, shape: &Shape, dtype: DType) -> Result<HipStorage> {
        let elem_count = shape.elem_count();
        let slice = match dtype {
            DType::U8 => HipStorageSlice::U8(self.alloc_zeros::<u8>(elem_count)?),
            DType::U32 => HipStorageSlice::U32(self.alloc_zeros::<u32>(elem_count)?),
            DType::I16 => HipStorageSlice::I16(self.alloc_zeros::<i16>(elem_count)?),
            DType::I32 => HipStorageSlice::I32(self.alloc_zeros::<i32>(elem_count)?),
            DType::I64 => HipStorageSlice::I64(self.alloc_zeros::<i64>(elem_count)?),
            DType::BF16 => HipStorageSlice::BF16(self.alloc_zeros::<bf16>(elem_count)?),
            DType::F16 => HipStorageSlice::F16(self.alloc_zeros::<f16>(elem_count)?),
            DType::F32 => HipStorageSlice::F32(self.alloc_zeros::<f32>(elem_count)?),
            DType::F64 => HipStorageSlice::F64(self.alloc_zeros::<f64>(elem_count)?),
            DType::F8E4M3 => HipStorageSlice::F8E4M3(self.alloc_zeros::<u8>(elem_count)?),
            DType::F6E2M3 | DType::F6E3M2 | DType::F4 | DType::F8E8M0 => {
                return Err(
                    HipError::InternalError("Dummy types not supported in HIP backend".to_string())
                        .into(),
                )
            }
        };
        Ok(HipStorage {
            slice,
            device: self.clone(),
        })
    }

    unsafe fn alloc_uninit(&self, shape: &Shape, dtype: DType) -> Result<HipStorage> {
        let elem_count = shape.elem_count();
        let slice = match dtype {
            DType::U8 => HipStorageSlice::U8(self.alloc::<u8>(elem_count)?),
            DType::U32 => HipStorageSlice::U32(self.alloc::<u32>(elem_count)?),
            DType::I16 => HipStorageSlice::I16(self.alloc::<i16>(elem_count)?),
            DType::I32 => HipStorageSlice::I32(self.alloc::<i32>(elem_count)?),
            DType::I64 => HipStorageSlice::I64(self.alloc::<i64>(elem_count)?),
            DType::BF16 => HipStorageSlice::BF16(self.alloc::<bf16>(elem_count)?),
            DType::F16 => HipStorageSlice::F16(self.alloc::<f16>(elem_count)?),
            DType::F32 => HipStorageSlice::F32(self.alloc::<f32>(elem_count)?),
            DType::F64 => HipStorageSlice::F64(self.alloc::<f64>(elem_count)?),
            DType::F8E4M3 => HipStorageSlice::F8E4M3(self.alloc::<u8>(elem_count)?),
            DType::F6E2M3 | DType::F6E3M2 | DType::F4 | DType::F8E8M0 => {
                return Err(
                    HipError::InternalError("Dummy types not supported in HIP backend".to_string())
                        .into(),
                )
            }
        };
        Ok(HipStorage {
            slice,
            device: self.clone(),
        })
    }

    fn storage_from_slice<T: crate::WithDType>(&self, s: &[T]) -> Result<HipStorage> {
        let slice = match T::cpu_storage_ref(s) {
            CpuStorageRef::U8(storage) => {
                HipStorageSlice::U8(self.copy_htod(storage)?)
            }
            CpuStorageRef::U32(storage) => {
                HipStorageSlice::U32(self.copy_htod(storage)?)
            }
            CpuStorageRef::I16(storage) => {
                HipStorageSlice::I16(self.copy_htod(storage)?)
            }
            CpuStorageRef::I32(storage) => {
                HipStorageSlice::I32(self.copy_htod(storage)?)
            }
            CpuStorageRef::I64(storage) => {
                HipStorageSlice::I64(self.copy_htod(storage)?)
            }
            CpuStorageRef::BF16(storage) => {
                HipStorageSlice::BF16(self.copy_htod(storage)?)
            }
            CpuStorageRef::F16(storage) => {
                HipStorageSlice::F16(self.copy_htod(storage)?)
            }
            CpuStorageRef::F32(storage) => {
                HipStorageSlice::F32(self.copy_htod(storage)?)
            }
            CpuStorageRef::F64(storage) => {
                HipStorageSlice::F64(self.copy_htod(storage)?)
            }
            CpuStorageRef::F8E4M3(_) => {
                return Err(HipError::UnsupportedDtype {
                    dtype: DType::F8E4M3,
                    op: "storage_from_slice",
                }
                .into())
            }
            CpuStorageRef::F4(_)
            | CpuStorageRef::F6E2M3(_)
            | CpuStorageRef::F6E3M2(_)
            | CpuStorageRef::F8E8M0(_) => {
                return Err(HipError::UnsupportedDtype {
                    dtype: T::DTYPE,
                    op: "storage_from_slice",
                }
                .into())
            }
        };
        Ok(HipStorage {
            slice,
            device: self.clone(),
        })
    }

    fn storage_from_cpu_storage(&self, storage: &CpuStorage) -> Result<HipStorage> {
        let slice = match storage {
            CpuStorage::U8(storage) => HipStorageSlice::U8(self.copy_htod(storage)?),
            CpuStorage::U32(storage) => HipStorageSlice::U32(self.copy_htod(storage)?),
            CpuStorage::I16(storage) => HipStorageSlice::I16(self.copy_htod(storage)?),
            CpuStorage::I32(storage) => HipStorageSlice::I32(self.copy_htod(storage)?),
            CpuStorage::I64(storage) => HipStorageSlice::I64(self.copy_htod(storage)?),
            CpuStorage::BF16(storage) => HipStorageSlice::BF16(self.copy_htod(storage)?),
            CpuStorage::F16(storage) => HipStorageSlice::F16(self.copy_htod(storage)?),
            CpuStorage::F32(storage) => HipStorageSlice::F32(self.copy_htod(storage)?),
            CpuStorage::F64(storage) => HipStorageSlice::F64(self.copy_htod(storage)?),
            CpuStorage::F8E4M3(_) => {
                return Err(HipError::UnsupportedDtype {
                    dtype: DType::F8E4M3,
                    op: "storage_from_cpu_storage",
                }
                .into())
            }
            CpuStorage::F4(_)
            | CpuStorage::F6E2M3(_)
            | CpuStorage::F6E3M2(_)
            | CpuStorage::F8E8M0(_) => {
                return Err(HipError::UnsupportedDtype {
                    dtype: storage.dtype(),
                    op: "storage_from_cpu_storage",
                }
                .into())
            }
        };
        Ok(HipStorage {
            slice,
            device: self.clone(),
        })
    }

    fn storage_from_cpu_storage_owned(&self, storage: CpuStorage) -> Result<HipStorage> {
        self.storage_from_cpu_storage(&storage)
    }

    fn rand_uniform(&self, shape: &Shape, dtype: DType, lo: f64, up: f64) -> Result<HipStorage> {
        // RNG is performed on the host for now, then uploaded.
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let elem_count = shape.elem_count();
        let slice = match dtype {
            DType::F32 => {
                let data: Vec<f32> = (0..elem_count)
                    .map(|_| rng.gen_range(lo as f32..up as f32))
                    .collect();
                HipStorageSlice::F32(self.copy_htod(&data)?)
            }
            DType::F64 => {
                let data: Vec<f64> = (0..elem_count)
                    .map(|_| rng.gen_range(lo..up))
                    .collect();
                HipStorageSlice::F64(self.copy_htod(&data)?)
            }
            _ => {
                return Err(HipError::UnsupportedDtype {
                    dtype,
                    op: "rand_uniform",
                }
                .into())
            }
        };
        Ok(HipStorage {
            slice,
            device: self.clone(),
        })
    }

    fn rand_normal(&self, shape: &Shape, dtype: DType, mean: f64, std: f64) -> Result<HipStorage> {
        use rand_distr::Distribution;
        let mut rng = rand::thread_rng();
        let elem_count = shape.elem_count();
        let slice = match dtype {
            DType::F32 => {
                let normal = rand_distr::Normal::new(mean as f32, std as f32)
                    .map_err(crate::Error::wrap)?;
                let data: Vec<f32> = (0..elem_count).map(|_| normal.sample(&mut rng)).collect();
                HipStorageSlice::F32(self.copy_htod(&data)?)
            }
            DType::F64 => {
                let normal = rand_distr::Normal::new(mean, std).map_err(crate::Error::wrap)?;
                let data: Vec<f64> = (0..elem_count).map(|_| normal.sample(&mut rng)).collect();
                HipStorageSlice::F64(self.copy_htod(&data)?)
            }
            _ => {
                return Err(HipError::UnsupportedDtype {
                    dtype,
                    op: "rand_normal",
                }
                .into())
            }
        };
        Ok(HipStorage {
            slice,
            device: self.clone(),
        })
    }

    fn set_seed(&self, seed: u64) -> Result<()> {
        *self.seed_value.write().unwrap() = seed;
        Ok(())
    }

    fn get_current_seed(&self) -> Result<u64> {
        Ok(*self.seed_value.read().unwrap())
    }

    fn synchronize(&self) -> Result<()> {
        self.stream.synchronize().w()
    }
}
