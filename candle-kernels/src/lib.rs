#[cfg(not(feature = "rocm"))]
mod ptx {
    include!(concat!(env!("OUT_DIR"), "/ptx.rs"));
}

#[cfg(not(feature = "rocm"))]
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Id {
    Affine,
    Binary,
    Cast,
    Conv,
    Fill,
    Indexing,
    Quantized,
    Reduce,
    Sort,
    Ternary,
    Unary,
}

#[cfg(not(feature = "rocm"))]
pub const ALL_IDS: [Id; 11] = [
    Id::Affine,
    Id::Binary,
    Id::Cast,
    Id::Conv,
    Id::Fill,
    Id::Indexing,
    Id::Quantized,
    Id::Reduce,
    Id::Sort,
    Id::Ternary,
    Id::Unary,
];

#[cfg(not(feature = "rocm"))]
pub struct Module {
    index: usize,
    ptx: &'static str,
}

#[cfg(not(feature = "rocm"))]
impl Module {
    pub fn index(&self) -> usize {
        self.index
    }

    pub fn ptx(&self) -> &'static str {
        self.ptx
    }
}

#[cfg(not(feature = "rocm"))]
const fn module_index(id: Id) -> usize {
    let mut i = 0;
    while i < ALL_IDS.len() {
        if ALL_IDS[i] as u32 == id as u32 {
            return i;
        }
        i += 1;
    }
    panic!("id not found")
}

#[cfg(not(feature = "rocm"))]
macro_rules! mdl {
    ($cst:ident, $id:ident) => {
        pub const $cst: Module = Module {
            index: module_index(Id::$id),
            ptx: ptx::$cst,
        };
    };
}

#[cfg(not(feature = "rocm"))]
mdl!(AFFINE, Affine);
#[cfg(not(feature = "rocm"))]
mdl!(BINARY, Binary);
#[cfg(not(feature = "rocm"))]
mdl!(CAST, Cast);
#[cfg(not(feature = "rocm"))]
mdl!(CONV, Conv);
#[cfg(not(feature = "rocm"))]
mdl!(FILL, Fill);
#[cfg(not(feature = "rocm"))]
mdl!(INDEXING, Indexing);
#[cfg(not(feature = "rocm"))]
mdl!(QUANTIZED, Quantized);
#[cfg(not(feature = "rocm"))]
mdl!(REDUCE, Reduce);
#[cfg(not(feature = "rocm"))]
mdl!(SORT, Sort);
#[cfg(not(feature = "rocm"))]
mdl!(TERNARY, Ternary);
#[cfg(not(feature = "rocm"))]
mdl!(UNARY, Unary);

pub mod ffi;

// HIP/ROCm kernel surface. The code objects are produced by the build script
// (build.rs, `rocm` feature) from the same .cu sources, compiled with hipcc.
#[cfg(feature = "rocm")]
mod rocm {
    include!(concat!(env!("OUT_DIR"), "/rocm.rs"));
}

#[cfg(feature = "rocm")]
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RocmId {
    Affine,
    Binary,
    Cast,
    Conv,
    Fill,
    Indexing,
    Reduce,
    Sort,
    Ternary,
    Unary,
    Gemm,
}

#[cfg(feature = "rocm")]
pub const ROCM_ALL_IDS: [RocmId; 11] = [
    RocmId::Affine,
    RocmId::Binary,
    RocmId::Cast,
    RocmId::Conv,
    RocmId::Fill,
    RocmId::Indexing,
    RocmId::Reduce,
    RocmId::Sort,
    RocmId::Ternary,
    RocmId::Unary,
    RocmId::Gemm,
];

/// A HIP code object (amdgcn binary) that can be loaded with `hipModuleLoadData`.
#[cfg(feature = "rocm")]
pub struct RocmModule {
    index: usize,
    co: &'static [u8],
}

#[cfg(feature = "rocm")]
impl RocmModule {
    pub fn index(&self) -> usize {
        self.index
    }

    pub fn co(&self) -> &'static [u8] {
        self.co
    }
}

#[cfg(feature = "rocm")]
const fn rocm_module_index(id: RocmId) -> usize {
    let mut i = 0;
    while i < ROCM_ALL_IDS.len() {
        if ROCM_ALL_IDS[i] as u32 == id as u32 {
            return i;
        }
        i += 1;
    }
    panic!("id not found")
}

#[cfg(feature = "rocm")]
macro_rules! rmdl {
    ($cst:ident, $id:ident) => {
        pub const $cst: RocmModule = RocmModule {
            index: rocm_module_index(RocmId::$id),
            co: rocm::$cst,
        };
    };
}

#[cfg(feature = "rocm")]
rmdl!(AFFINE, Affine);
#[cfg(feature = "rocm")]
rmdl!(BINARY, Binary);
#[cfg(feature = "rocm")]
rmdl!(CAST, Cast);
#[cfg(feature = "rocm")]
rmdl!(CONV, Conv);
#[cfg(feature = "rocm")]
rmdl!(FILL, Fill);
#[cfg(feature = "rocm")]
rmdl!(INDEXING, Indexing);
#[cfg(feature = "rocm")]
rmdl!(REDUCE, Reduce);
#[cfg(feature = "rocm")]
rmdl!(SORT, Sort);
#[cfg(feature = "rocm")]
rmdl!(TERNARY, Ternary);
#[cfg(feature = "rocm")]
rmdl!(UNARY, Unary);
#[cfg(feature = "rocm")]
rmdl!(GEMM, Gemm);
