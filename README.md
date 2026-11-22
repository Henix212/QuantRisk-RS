# QuantRisk-RS

**High-performance risk, research & execution engine — Rust core, Python API.**

![Rust](https://img.shields.io/badge/rust-1.75%2B-orange)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![PyO3](https://img.shields.io/badge/bindings-PyO3-red)
![License](https://img.shields.io/badge/license-MIT-green)

**QuantRisk-RS** is a native **Rust** numerical engine for **quant trading / quant research** desks: book risk, stress testing, walk-forward research, and execution-aware sizing — exposed to Python via **PyO3**.

It is the **decision / analytics layer**, not a market-connectivity stack. Order-book / HFT path tooling lives elsewhere (e.g. a C++ feed & matching prototype); this crate is the fast substrate under research notebooks and desk risk workflows.

| Layer | Role |
| :--- | :--- |
| **Rust core** | Risk, Monte Carlo, backtest, execution costs, portfolio solvers |
| **Python bindings** | Research & pipeline API (`maturin` wheel) |
| **Out of scope** | Market data feeds, order gateways, matching engines, RL allocators |

---

## Positioning

- **Quant research** — auditable risk metrics, Monte Carlo stress, walk-forward backtests with costs.
- **Quant trading** — book-level VaR/CVaR & risk decomposition; execution-aware schedules (impact / Almgren–Chriss style).
- **Performance first** — `rayon` + tight linear algebra where NumPy/`SciPy` bottlenecks show up.
- **Complements HFT infra** — systems latency stack ≠ research/risk engine; both belong on a QT/QR profile.

---

## Key Features

### 1. Risk & performance (desk)
* **Tail risk:** Parametric, Historical, and Monte Carlo VaR / CVaR (Expected Shortfall) at 95% and 99%.
* **Risk decomposition:** Marginal (MCTR) and Component (CCTR) contributions.
* **Performance ratios:** Annualized return, volatility, Sharpe, Sortino, Calmar, Maximum Drawdown.

### 2. Simulation & research backtesting
* **Parallel Monte Carlo:** Multi-threaded GBM and Cholesky-correlated paths (`rayon`).
* **Walk-forward backtester:** Rolling rebalance with transaction costs, slippage, and turnover.

### 3. Execution-aware layer
* **Cost / impact models:** Linear and square-root style temporary impact.
* **Optimal execution:** Almgren–Chriss-type schedules bridging research signals and tradable trajectories.

### 4. Portfolio construction toolkit
* **Mean-Variance (MVO):** GMV, Max Sharpe, efficient frontier (long-only / box constraints).
* **Risk Parity (ERC):** Equal risk contribution solve.
* **Black-Litterman:** Equilibrium returns + investor views.
* **Min-CVaR:** Rockafellar–Uryasev tail-risk minimization.

---

## Mathematical Foundations

### Risk contribution
Portfolio volatility \(\sigma(w) = \sqrt{w^T \Sigma w}\). Asset \(i\) risk contribution:

\[RC_i = w_i \frac{(\Sigma w)_i}{\sqrt{w^T \Sigma w}}\]

### Conditional Value-at-Risk (CVaR)
For loss \(L(w, X)\) and level \(\alpha \in (0,1)\):

\[\text{CVaR}_\alpha(w) = \mathbb{E}[L(w, X) \mid L(w, X) \ge \text{VaR}_\alpha(w)]\]

### Almgren–Chriss (sketch)
Trade trajectory \(x_t\) minimizing expected cost + \(\lambda \cdot\) variance of cost (permanent + temporary impact). Used as the execution bridge between a target book and a fill schedule.

---

## Performance Benchmarks

*Targets* on Apple M1 Pro (8 cores: 6P+2E) vs NumPy + SciPy (to be locked in `benches/`):

| Task | Scale | Python | QuantRisk-RS | Speedup |
| :--- | :--- | :---: | :---: | :---: |
| Monte Carlo VaR / CVaR | 10 assets, 1M paths | 1,820 ms | 24.5 ms | **~74×** |
| Walk-Forward Backtest | 10Y daily (2,520 days) | 410.5 ms | 6.1 ms | **~67×** |
| Risk Parity | 50 assets | 145.2 ms | 1.8 ms | **~80×** |

---

## Architecture

```text
.
├── Cargo.toml
├── pyproject.toml           # Maturin Python wheel
├── main.ipynb               # Empirical / desk-style case study
├── src/
│   ├── lib.rs               # PyO3 entrypoint
│   ├── stats.rs             # Covariance (Sample, Ledoit-Wolf)
│   ├── risk.rs              # VaR, CVaR, drawdown, MCTR / CCTR
│   ├── monte_carlo.rs       # Cholesky + parallel stress paths
│   ├── backtest.rs          # Walk-forward + costs / slippage
│   ├── execution.rs         # Impact models + Almgren–Chriss
│   └── portfolio/
│       ├── markowitz.rs
│       ├── risk_parity.rs
│       ├── black_litterman.rs
│       └── min_cvar.rs
├── benches/
│   └── benchmarks.rs
└── tests/
    └── integration_tests.rs
```

---

## Quickstart

### Prerequisites
* [Rust toolchain](https://rustup.rs/) (`rustc` & `cargo`)
* Python 3.10+ and [uv](https://docs.astral.sh/uv/)

### Build & install
```bash
git clone https://github.com/<your-org>/quantrisk-rs
cd quantrisk-rs
uv venv
uv pip install maturin
uv run maturin develop --release
```

### Python usage
```python
import numpy as np
import quantrisk_rs as qr

returns = np.random.normal(0.0004, 0.012, size=(2520, 10))
cov = np.cov(returns, rowvar=False) * 252
mu = np.mean(returns, axis=0) * 252
weights = np.full(10, 0.1)

mc = qr.monte_carlo_risk(
    weights=weights,
    expected_returns=mu,
    cov_matrix=cov,
    n_simulations=1_000_000,
    confidence_level=0.99,
)
print(f"99% 1Y VaR:  {mc['var']:.2%}")
print(f"99% 1Y CVaR: {mc['cvar']:.2%}")

# Execution schedule toward a target book (API target)
# schedule = qr.optimal_execution(target=weights, adv=..., risk_aversion=1e-6)
```

---

## References
* Rockafellar, R. T., & Uryasev, S. (2000). *Optimization of Conditional Value-at-Risk*. Journal of Risk.
* Almgren, R., & Chriss, N. (2000). *Optimal Execution of Portfolio Transactions*. Journal of Risk.
* Maillard, S., Roncalli, T., & Teïletche, J. (2010). *The Properties of Equally Weighted Risk Contribution Portfolios*. The Journal of Portfolio Management.
* Markowitz, H. (1952). *Portfolio Selection*. The Journal of Finance.
* Black, F., & Litterman, R. (1992). *Global Portfolio Optimization*. Financial Analysts Journal.

---

## License
MIT — see `LICENSE`.
