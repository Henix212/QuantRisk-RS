//! Tests for `quantrisk_rs::stats` (means, sample cov/corr, Ledoit–Wolf).

use quantrisk_rs::stats::{
    ledoit_wolf_covariance, mean_returns, sample_correlation, sample_covariance,
};
use quantrisk_rs::types::{MatF, QuantRiskError};

/// Simple 3×2 panel used across several tests:
/// columns means are (3, 4); sample cov is [[4, 4], [4, 4]].
fn demo_returns() -> MatF {
    MatF::from_row_slice(3, 2, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
}

#[test]
fn mean_returns_ok() {
    let mat = demo_returns();
    let got = mean_returns(&mat).unwrap();
    // Column 0: (1+3+5)/3 = 3 ; column 1: (2+4+6)/3 = 4
    assert!((got[0] - 3.0).abs() < 1e-12);
    assert!((got[1] - 4.0).abs() < 1e-12);
}

#[test]
fn mean_returns_empty_matrix_err() {
    let mat = MatF::zeros(0, 3);
    assert!(matches!(
        mean_returns(&mat),
        Err(QuantRiskError::InvalidArgument(_))
    ));
}

#[test]
fn sample_covariance_ok() {
    let mat = demo_returns();
    let cov = sample_covariance(&mat).unwrap();

    // Centered rows: (-2,-2), (0,0), (2,2) → XcᵀXc = [[8,8],[8,8]] ; /(T-1=2) → 4
    assert_eq!(cov.nrows(), 2);
    assert_eq!(cov.ncols(), 2);
    assert!((cov[(0, 0)] - 4.0).abs() < 1e-12);
    assert!((cov[(0, 1)] - 4.0).abs() < 1e-12);
    assert!((cov[(1, 0)] - 4.0).abs() < 1e-12);
    assert!((cov[(1, 1)] - 4.0).abs() < 1e-12);
}

#[test]
fn sample_covariance_too_few_rows_err() {
    let mat = MatF::from_row_slice(1, 2, &[1.0, 2.0]);
    assert!(matches!(
        sample_covariance(&mat),
        Err(QuantRiskError::InvalidArgument(_))
    ));
}

#[test]
fn sample_correlation_ok() {
    let mat = demo_returns();
    let corr = sample_correlation(&mat).unwrap();

    // Perfect co-movement in the demo panel → correlation matrix is all ones.
    assert!((corr[(0, 0)] - 1.0).abs() < 1e-12);
    assert!((corr[(1, 1)] - 1.0).abs() < 1e-12);
    assert!((corr[(0, 1)] - 1.0).abs() < 1e-12);
    assert!((corr[(1, 0)] - 1.0).abs() < 1e-12);
}

#[test]
fn ledoit_wolf_covariance_ok_shape_and_symmetry() {
    let mat = demo_returns();
    let lw = ledoit_wolf_covariance(&mat).unwrap();

    assert_eq!(lw.nrows(), 2);
    assert_eq!(lw.ncols(), 2);
    // Shrinkage result must stay symmetric.
    assert!((lw[(0, 1)] - lw[(1, 0)]).abs() < 1e-12);
    assert!(lw.iter().all(|x| x.is_finite()));
}

#[test]
fn ledoit_wolf_covariance_too_few_rows_err() {
    let mat = MatF::from_row_slice(1, 2, &[1.0, 2.0]);
    assert!(matches!(
        ledoit_wolf_covariance(&mat),
        Err(QuantRiskError::InvalidArgument(_))
    ));
}
