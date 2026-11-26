use crate::utils::{check_weights_cov};
use crate::types::*;

/// Portfolio volatility \(\sigma_p = \sqrt{w^\top \Sigma w}\).
pub fn portfolio_vol(w: &VecF, cov: &MatF) -> Result<f64> {
    check_weights_cov(w, cov)?;
    let variance = w.dot(&(cov * w)); // wᵀ (Σ w)
    Ok(variance.sqrt())
}
