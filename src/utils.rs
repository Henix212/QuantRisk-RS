//! Domain validation helpers and simple scaling utilities.
//!
//! Keep these free of PyO3 / NumPy concerns: callers in the Rust core use
//! `VecF` / `MatF`, and the Python boundary converts arrays separately.

use crate::constant::{DEFAULT_TOLERANCE, TRADING_DAYS};
use crate::types::*;

/// Ensure portfolio weights and the covariance matrix share the same dimension `n`
/// (`w` length `n`, `cov` shape `n×n`).
pub fn check_weights_cov(w: &VecF, cov: &MatF) -> Result<()> {
    let n: usize = w.len();

    if cov.ncols() != n || cov.nrows() != n {
        return Err(QuantRiskError::DimensionMismatch {
            expected: format!("{n}x{n}"),
            got: format!("{}x{}", cov.nrows(), cov.ncols()),
        });
    }
    Ok(())
}

/// Ensure `mat` is square (`nrows == ncols`).
pub fn check_square(mat: &MatF) -> Result<()> {
    if mat.ncols() != mat.nrows() {
        return Err(QuantRiskError::DimensionMismatch {
            expected: format!("We expected a n x n format"),
            got: format!(" {}x{}", mat.ncols(), mat.nrows()),
        });
    }
    Ok(())
}

/// Ensure vector `v` (labeled `name` in error messages) has length `n`.
pub fn check_len(v: &VecF, n: usize, name: &str) -> Result<()> {
    if v.len() != n {
        return Err(QuantRiskError::DimensionMismatch {
            expected: format!("{name} len {n}"),
            got: format!("{name}.len() = {}", v.len()),
        });
    }
    Ok(())
}

/// Ensure `cov` is symmetric (within [`DEFAULT_TOLERANCE`]) and positive definite
/// via a successful Cholesky factorization.
pub fn ensure_spd(cov: &MatF) -> Result<()> {
    let cov_t = cov.transpose();
    if !cov.relative_eq(&cov_t, DEFAULT_TOLERANCE, DEFAULT_TOLERANCE) {
        return Err(QuantRiskError::NotSpd);
    }
    cov.clone().cholesky().ok_or(QuantRiskError::NotSpd)?;
    Ok(())
}

/// Ensure a confidence level lies in the open unit interval `(0, 1)`.
pub fn check_confidence_level(confidence_level: f64) -> Result<()> {
    if !(0.0 < confidence_level && confidence_level < 1.0) {
        return Err(QuantRiskError::InvalidArgument(format!(
            "Confidence level must be in (0,1), got {confidence_level}"
        )));
    }
    Ok(())
}

/// Reject an empty weight / return vector.
pub fn check_non_empty_vec(w: &VecF) -> Result<()> {
    if w.len() == 0 {
        return Err(QuantRiskError::InvalidArgument(format!("Empty Vector")));
    }
    Ok(())
}

/// Reject a matrix with zero rows or zero columns (`name` appears in the error).
pub fn check_non_empty_mat(m: &MatF, name: &str) -> Result<()> {
    if m.nrows() == 0 || m.ncols() == 0 {
        return Err(QuantRiskError::InvalidArgument(format!(
            "{name} matrix must be non-empty"
        )));
    }
    Ok(())
}

/// Ensure a count-like argument (e.g. simulation paths) is strictly positive.
pub fn check_positive_usize(n: u64) -> Result<()> {
    if n == 0 {
        return Err(QuantRiskError::InvalidArgument(format!(
            "Invalid usize argument"
        )));
    }
    Ok(())
}

/// Compound a daily mean return to an annual effective rate using [`TRADING_DAYS`].
pub fn annualize_mean(daily_mean_return: f64) -> Result<f64> {
    Ok((1.0 + daily_mean_return).powi(TRADING_DAYS as i32) - 1.0)
}

/// Scale a daily volatility to an annual figure using [`TRADING_DAYS`].
pub fn annualize_vol(vol: f64) -> Result<f64> {
    Ok(vol * (TRADING_DAYS as f64).sqrt())
}

/// Scale a daily covariance matrix to an annual figure: `Σ_ann = Σ_daily × trading_days`.
pub fn annualize_cov(cov_daily: &MatF) -> Result<MatF> {
    check_square(cov_daily)?;
    Ok(cov_daily * (TRADING_DAYS as f64))
}