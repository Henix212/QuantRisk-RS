//! PyO3 / NumPy boundary: conversions and Python-facing functions.
//!
//! Keep pure numerics in other modules; this file only bridges to Python.

use crate::risk;
use crate::types::{MatF, QuantRiskError, VecF};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
use pyo3::prelude::*;

/// Copy a 1-D NumPy array into an owned [`VecF`].
pub fn to_vecf(arr: &PyReadonlyArray1<'_, f64>) -> VecF {
    let view = arr.as_array();
    VecF::from_iterator(view.len(), view.iter().copied())
}

/// Copy a 2-D NumPy array into an owned [`MatF`] (row `i`, col `j`).
pub fn to_matf(arr: &PyReadonlyArray2<'_, f64>) -> MatF {
    let view = arr.as_array();
    let (nrows, ncols) = (view.nrows(), view.ncols());
    MatF::from_fn(nrows, ncols, |i, j| view[(i, j)])
}

/// Smoke-test function exposed to Python.
#[pyfunction]
pub fn ping() -> &'static str {
    "ok"
}

/// Demo that domain errors surface as Python `ValueError`.
#[pyfunction]
pub fn fail_demo() -> PyResult<()> {
    Err(QuantRiskError::InvalidArgument("demo".into()).into())
}

/// Portfolio volatility \(\sigma_p = \sqrt{w^\top \Sigma w}\).
#[pyfunction]
pub fn portfolio_vol(
    weights: PyReadonlyArray1<'_, f64>,
    cov: PyReadonlyArray2<'_, f64>,
) -> PyResult<f64> {
    let w = to_vecf(&weights);
    let sigma = to_matf(&cov);
    risk::portfolio_vol(&w, &sigma).map_err(Into::into)
}

/// Echo helpers used to verify NumPy ↔ nalgebra round-trips in smoke tests.
#[pyfunction]
pub fn echo_vector<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<'_, f64>,
) -> Bound<'py, PyArray1<f64>> {
    to_vecf(&values).as_slice().to_pyarray(py)
}

#[pyfunction]
pub fn echo_matrix<'py>(
    py: Python<'py>,
    values: PyReadonlyArray2<'_, f64>,
) -> Bound<'py, PyArray2<f64>> {
    to_matf(&values).to_pyarray(py)
}
