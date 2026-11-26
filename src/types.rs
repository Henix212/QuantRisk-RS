//! Shared numeric types and domain errors for the crate.
//!
//! Keep linear-algebra work on `nalgebra` dense `f64` types; convert
//! to/from NumPy only at the PyO3 boundary.

use nalgebra::{DMatrix, DVector};
use std::fmt;

/// Column weight / return / PnL vector (`f64`).
pub type VecF = DVector<f64>;

/// Dense matrix (`f64`) — returns (T×N), covariance (N×N), etc.
pub type MatF = DMatrix<f64>;

/// Domain errors returned by public Rust APIs.
#[derive(Debug, Clone, PartialEq)]
pub enum QuantRiskError {
    /// Incompatible shapes (e.g. weights len ≠ cov side).
    DimensionMismatch { expected: String, got: String },
    /// Covariance (or factor) is not usable (e.g. Cholesky failed).
    NotSpd,
    /// Invalid user argument (probability, empty input, …).
    InvalidArgument(String),
}

impl fmt::Display for QuantRiskError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DimensionMismatch { expected, got } => {
                write!(f, "dimension mismatch: expected {expected}, got {got}")
            }
            Self::NotSpd => write!(f, "matrix is not symmetric positive definite"),
            Self::InvalidArgument(msg) => write!(f, "invalid argument: {msg}"),
        }
    }
}

impl std::error::Error for QuantRiskError {}

impl From<QuantRiskError> for pyo3::PyErr {
    fn from(err: QuantRiskError) -> pyo3::PyErr {
        pyo3::exceptions::PyValueError::new_err(err.to_string())
    }
}

/// Convenience alias for fallible crate operations.
pub type Result<T> = std::result::Result<T, QuantRiskError>;
