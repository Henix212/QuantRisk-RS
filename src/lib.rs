pub mod backtest;
pub mod constant;
pub mod execution;
pub mod monte_carlo;
pub mod portfolio;
pub mod python;
pub mod risk;
pub mod stats;
pub mod types;
pub mod utils;

use pyo3::prelude::*;
pub use types::{MatF, QuantRiskError, Result, VecF};

/// Python extension entrypoint (`import quantrisk_rs`).
#[pymodule]
fn quantrisk_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(python::ping, m)?)?;
    m.add_function(wrap_pyfunction!(python::fail_demo, m)?)?;
    m.add_function(wrap_pyfunction!(python::portfolio_vol, m)?)?;
    m.add_function(wrap_pyfunction!(python::echo_vector, m)?)?;
    m.add_function(wrap_pyfunction!(python::echo_matrix, m)?)?;
    Ok(())
}
