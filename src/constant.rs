//! Shared numeric constants used across the crate.

/// Relative / absolute tolerance for floating-point symmetry checks (e.g. SPD).
pub const DEFAULT_TOLERANCE: f64 = 1e-9;

/// Conventional number of trading days per year for equity-style annualization.
pub const TRADING_DAYS: u8 = 252;
