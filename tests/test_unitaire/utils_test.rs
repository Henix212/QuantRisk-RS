//! Tests for `quantrisk_rs::utils` helpers.

use quantrisk_rs::constant::TRADING_DAYS;
use quantrisk_rs::types::{MatF, QuantRiskError, VecF};
use quantrisk_rs::utils::{
    annualize_cov, annualize_mean, annualize_vol, check_confidence_level, check_len,
    check_non_empty_mat, check_non_empty_vec, check_positive_usize, check_square,
    check_weights_cov, ensure_spd,
};

#[test]
fn check_weights_cov_ok() {
    let mat = MatF::zeros(3, 3);
    let w = VecF::zeros(3);
    assert!(matches!(check_weights_cov(&w, &mat), Ok(())));
}

#[test]
fn check_weights_cov_mismatch() {
    let mat = MatF::zeros(3, 3);
    let w = VecF::zeros(5);
    assert!(matches!(
        check_weights_cov(&w, &mat),
        Err(QuantRiskError::DimensionMismatch { .. })
    ));
}

#[test]
fn check_square_ok() {
    let mat = MatF::zeros(3, 3);
    assert!(matches!(check_square(&mat), Ok(())));
}

#[test]
fn check_square_mismatch() {
    let mat = MatF::zeros(4, 3);
    assert!(matches!(
        check_square(&mat),
        Err(QuantRiskError::DimensionMismatch { .. })
    ));
}

#[test]
fn check_len_ok() {
    let w = VecF::zeros(5);
    let n = 5;
    assert!(matches!(check_len(&w, n, "test1"), Ok(())));
}

#[test]
fn check_len_mismatch() {
    let w = VecF::zeros(5);
    let n = 4;
    assert!(matches!(
        check_len(&w, n, "test1"),
        Err(QuantRiskError::DimensionMismatch { .. })
    ));
}

#[test]
fn ensure_spd_ok() {
    let mat = MatF::from_row_slice(2, 2, &[2.0, 1.0, 1.0, 2.0]);
    assert!(matches!(ensure_spd(&mat), Ok(())));
}

#[test]
fn ensure_spd_mismatch() {
    let not_symmetric = MatF::from_row_slice(2, 2, &[1.0, 2.0, 0.0, 1.0]);
    assert!(matches!(
        ensure_spd(&not_symmetric),
        Err(QuantRiskError::NotSpd)
    ));

    let not_pd = MatF::from_row_slice(2, 2, &[1.0, 0.0, 0.0, -1.0]);
    assert!(matches!(ensure_spd(&not_pd), Err(QuantRiskError::NotSpd)));
}

#[test]
fn check_confidence_level_ok() {
    assert!(matches!(check_confidence_level(0.95), Ok(())));
}

#[test]
fn check_confidence_level_mismatch() {
    for level in [0.0, 1.0, -0.1, 1.1] {
        assert!(matches!(
            check_confidence_level(level),
            Err(QuantRiskError::InvalidArgument(_))
        ));
    }
}

#[test]
fn check_non_empty_vec_ok() {
    let w = VecF::zeros(1);
    assert!(matches!(check_non_empty_vec(&w), Ok(())));
}

#[test]
fn check_non_empty_vec_mismatch() {
    let w = VecF::zeros(0);
    assert!(matches!(
        check_non_empty_vec(&w),
        Err(QuantRiskError::InvalidArgument(_))
    ));
}

#[test]
fn check_non_empty_mat_ok() {
    let m = MatF::zeros(2, 3);
    assert!(matches!(check_non_empty_mat(&m, "returns"), Ok(())));
}

#[test]
fn check_non_empty_mat_zero_rows() {
    let m = MatF::zeros(0, 3);
    assert!(matches!(
        check_non_empty_mat(&m, "returns"),
        Err(QuantRiskError::InvalidArgument(_))
    ));
}

#[test]
fn check_non_empty_mat_zero_cols() {
    let m = MatF::zeros(3, 0);
    assert!(matches!(
        check_non_empty_mat(&m, "returns"),
        Err(QuantRiskError::InvalidArgument(_))
    ));
}

#[test]
fn check_positive_usize_ok() {
    assert!(matches!(check_positive_usize(1), Ok(())));
}

#[test]
fn check_positive_usize_mismatch() {
    assert!(matches!(
        check_positive_usize(0),
        Err(QuantRiskError::InvalidArgument(_))
    ));
}

#[test]
fn annualize_mean_ok() {
    let daily = 0.001;
    let expected = (1.0_f64 + daily).powi(TRADING_DAYS as i32) - 1.0;
    let got = annualize_mean(daily).unwrap();
    assert!((got - expected).abs() < 1e-12);
}

#[test]
fn annualize_vol_ok() {
    let vol = 0.01;
    let expected = vol * (TRADING_DAYS as f64).sqrt();
    let got = annualize_vol(vol).unwrap();
    assert!((got - expected).abs() < 1e-12);
}

#[test]
fn annualize_cov_ok() {
    let cov = MatF::from_row_slice(2, 2, &[0.01, 0.002, 0.002, 0.02]);
    let got = annualize_cov(&cov).unwrap();
    let scale = TRADING_DAYS as f64;
    assert_eq!(got.nrows(), 2);
    assert_eq!(got.ncols(), 2);
    assert!((got[(0, 0)] - 0.01 * scale).abs() < 1e-12);
    assert!((got[(0, 1)] - 0.002 * scale).abs() < 1e-12);
    assert!((got[(1, 0)] - 0.002 * scale).abs() < 1e-12);
    assert!((got[(1, 1)] - 0.02 * scale).abs() < 1e-12);
}

#[test]
fn annualize_cov_mismatch() {
    let cov = MatF::zeros(2, 3);
    assert!(matches!(
        annualize_cov(&cov),
        Err(QuantRiskError::DimensionMismatch { .. })
    ));
}
