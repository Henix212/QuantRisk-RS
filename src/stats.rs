//! Sample statistics on return panels (`T` rows × `N` assets).
//!
//! Provides column means, the unbiased sample covariance, the correlation
//! matrix, and a Ledoit–Wolf shrinkage estimator toward a scaled identity.

use crate::types::{MatF, QuantRiskError, Result, VecF};
use crate::utils::check_non_empty_mat;

/// Per-asset mean of a returns matrix (`T` rows × `N` assets).
///
/// Uses nalgebra's `row_mean_tr`: average **down each column** (over time),
/// returned as a length-`N` column vector. Note that `column_mean` averages
/// across columns per row — the opposite of what we need here.
pub fn mean_returns(returns: &MatF) -> Result<VecF> {
    check_non_empty_mat(returns, "returns")?;
    Ok(returns.row_mean_tr())
}

/// Unbiased sample covariance (`N×N`) of a returns matrix (`T×N`).
///
/// Centers each column by its mean, then forms
/// `S = X_cᵀ X_c / (T - 1)` (Bessel's correction). Requires `T >= 2`.
pub fn sample_covariance(returns: &MatF) -> Result<MatF> {
    check_non_empty_mat(returns, "returns")?;
    let t = returns.nrows();
    if t < 2 {
        return Err(QuantRiskError::InvalidArgument(
            "sample covariance requires at least 2 observations".into(),
        ));
    }

    let mu = mean_returns(returns)?;
    let n = returns.ncols();
    let mut centered = MatF::zeros(t, n);
    for j in 0..n {
        for i in 0..t {
            centered[(i, j)] = returns[(i, j)] - mu[j];
        }
    }

    Ok((centered.transpose() * centered) / (t - 1) as f64)
}

/// Sample correlation (`N×N`) derived from [`sample_covariance`].
///
/// Entry `(i, j)` is `S_ij / √(S_ii S_jj)`; the diagonal is forced to `1`.
pub fn sample_correlation(returns: &MatF) -> Result<MatF> {
    let cov = sample_covariance(returns)?;
    let n = cov.nrows();
    let mut corr = MatF::zeros(n, n);

    for i in 0..n {
        for j in 0..n {
            let denom = (cov[(i, i)] * cov[(j, j)]).sqrt();
            corr[(i, j)] = if denom == 0.0 {
                0.0
            } else {
                cov[(i, j)] / denom
            };
        }
        corr[(i, i)] = 1.0;
    }
    Ok(corr)
}

/// Ledoit–Wolf shrinkage covariance toward a scaled identity target:
/// `Σ_LW = δ F + (1 - δ) S`, with `F = m I` and `m = tr(S) / N`.
///
/// Follows the LW (2004) convention of estimating `S` with divisor `T`
/// (not `T - 1`). The intensity `δ` is clipped to `[0, 1]`.
pub fn ledoit_wolf_covariance(returns: &MatF) -> Result<MatF> {
    check_non_empty_mat(returns, "returns")?;
    let t = returns.nrows();
    let n = returns.ncols();
    if t < 2 {
        return Err(QuantRiskError::InvalidArgument(
            "Ledoit-Wolf covariance requires at least 2 observations".into(),
        ));
    }

    let mu = mean_returns(returns)?;
    let mut x = MatF::zeros(t, n);
    for j in 0..n {
        for i in 0..t {
            x[(i, j)] = returns[(i, j)] - mu[j];
        }
    }

    // LW (2004) uses the 1/T estimator for S.
    let s = (x.transpose() * &x) / t as f64;
    let m = s.trace() / n as f64;
    let f = MatF::identity(n, n) * m;

    // ||S - F||_F² — distance from sample cov to the shrinkage target.
    let mut d2 = 0.0;
    for i in 0..n {
        for j in 0..n {
            let d = s[(i, j)] - f[(i, j)];
            d2 += d * d;
        }
    }

    // π̂: average Frobenius variance of the sample covariance entries.
    let mut pi_hat = 0.0;
    for k in 0..t {
        let mut outer = MatF::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                outer[(i, j)] = x[(k, i)] * x[(k, j)];
            }
        }
        for i in 0..n {
            for j in 0..n {
                let e = outer[(i, j)] - s[(i, j)];
                pi_hat += e * e;
            }
        }
    }
    pi_hat /= t as f64;

    // Shrinkage intensity δ = (π̂ / d²) / T, clipped to a valid weight.
    let mut shrinkage = if d2 > 0.0 {
        (pi_hat / d2) / t as f64
    } else {
        1.0
    };
    if !shrinkage.is_finite() {
        shrinkage = 1.0;
    }
    shrinkage = shrinkage.clamp(0.0, 1.0);

    Ok(shrinkage * f + (1.0 - shrinkage) * s)
}
