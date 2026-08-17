//! Errors for the HIP/ROCm backend.
use crate::{DType, Error};

#[derive(Debug)]
pub enum HipError {
    InternalError(String),
    UnexpectedDType {
        msg: String,
        expected: DType,
        got: DType,
    },
    UnsupportedDtype {
        dtype: DType,
        op: &'static str,
    },
    MatMulNonContiguous {
        lhs_stride: crate::Layout,
        rhs_stride: crate::Layout,
        mnk: (usize, usize, usize),
    },
    Driver(String),
}

impl std::fmt::Display for HipError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InternalError(s) => write!(f, "{s}"),
            Self::UnexpectedDType { msg, expected, got } => {
                write!(f, "{msg}, expected {expected:?}, got {got:?}")
            }
            Self::UnsupportedDtype { dtype, op } => {
                write!(f, "{op} not supported for dtype {dtype:?}")
            }
            Self::MatMulNonContiguous {
                lhs_stride,
                rhs_stride,
                mnk,
            } => write!(
                f,
                "lhs and rhs matrices must be contiguous for matmul, lhs stride {lhs_stride:?}, \
                 rhs stride {rhs_stride:?}, mnk {mnk:?}"
            ),
            Self::Driver(s) => write!(f, "HIP error: {s}"),
        }
    }
}

impl std::error::Error for HipError {}

impl From<HipError> for Error {
    fn from(e: HipError) -> Self {
        Error::wrap(Box::new(e))
    }
}

pub trait WrapErr<T> {
    fn w(self) -> crate::Result<T>;
}

impl<T> WrapErr<T> for std::result::Result<T, rocm_rs::hip::error::Error> {
    fn w(self) -> crate::Result<T> {
        self.map_err(|e| Error::wrap(Box::new(HipError::Driver(e.to_string()))))
    }
}
