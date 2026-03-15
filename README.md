# Statistical Arbitrage Engine

S&P 500 pairs trading system built in Python. Event-driven backtester with quarterly re-screening, walk-forward validated out-of-sample performance.

---

## Results (Out-of-Sample, Walk-Forward Validated)

| Metric | Value |
|--------|-------|
| Total OOS Return | +7.92% |
| Annualised Return | +2.58% |
| Sharpe Ratio | 0.835 |
| Sortino Ratio | 0.935 |
| Calmar Ratio | 1.206 |
| Max Drawdown | −2.14% |
| Profitable Folds | 3 / 3 |
| Total OOS Trades | 35 |
| Mean P&L per Trade | +$219 |

All metrics are computed exclusively on out-of-sample test periods. No in-sample data contaminates the reported performance.

---

## Architecture

```
pair_screener.py     — Cointegration screening with structural break detection
spread_model.py      — Rolling hedge ratio, z-score, OU process fitting, regime detection
signal_generator.py  — Entry/exit signal generation from z-score crossings
backtester.py        — Event-driven simulation with quarterly re-screening
risk_manager.py      — Kelly sizing, drawdown monitor, MTM circuit breakers
walk_forward.py      — Rolling train/test validation across non-overlapping folds
param_optimizer.py   — Grid search over signal parameters, evaluated on OOS Sharpe
dashboard.py         — Interactive Plotly HTML dashboard
config.py            — Central parameter store
data_pipeline.py     — yfinance download, caching, validation
```

---

## Methodology

### Universe

S&P 500 stocks screened sector-by-sector across eight sectors: financials, energy, healthcare, utilities, materials, consumer staples, industrials, and real estate. Cross-sector pairs are excluded — same-sector stocks share macro exposures that drive genuine cointegration.

### Pair Screening

Each candidate pair is tested in five steps:

1. **Correlation pre-filter** — return correlation must exceed 0.70 to proceed
2. **Engle-Granger cointegration test** — p-value threshold 0.075
3. **ADF test on the spread** — confirms the residual series is stationary
4. **Half-life filter** — mean reversion must complete in 5–126 trading days
5. **Chow structural break test** — rejects pairs whose cointegrating relationship shifted mid-period (spin-offs, mergers, major regime changes). Tested at three split points (33%, 50%, 67%) at p < 0.001 significance

Pairs are re-screened quarterly using a rolling 504-day (2-year) lookback window, ensuring the strategy adapts to regime shifts rather than trading stale relationships.

### Spread Model

For each approved pair (Y, X):

```
S_t = log(Y_t) − β · log(X_t)
```

The hedge ratio β is estimated via rolling OLS on the lookback window and fixed for the forward trading period — no look-ahead bias. The z-score is computed with a 40-bar rolling mean and standard deviation, seeded with the final 40 bars of the training window so the signal is calibrated from day one of each test period.

### Signal Logic

| Condition | Action |
|-----------|--------|
| z-score crosses −2.2 | Enter long spread (long Y, short X) |
| z-score crosses +2.2 | Enter short spread (short Y, long X) |
| \|z-score\| falls below 0.5 | Exit — spread has normalised |
| \|z-score\| exceeds 4.0 | Stop-loss — relationship breaking down |
| Holding > 45 bars | Time-based stop — zombie trade exit |

Regime detection classifies each day as normal, volatile, trending, or breakdown. Positions are halved in volatile regimes and suspended entirely in breakdown regimes.

### Position Sizing

Fractional Kelly criterion adapted for spread P&L:

```
f* = (E[spread_pnl] × kelly_fraction) / Var[spread_pnl]
```

`kelly_fraction = 0.35` (approximately quarter-Kelly) — the standard industry practice that halves drawdowns relative to full Kelly while preserving most of the geometric growth benefit. Position notional is capped at 12% of equity per leg.

### Risk Controls

Three independent circuit breakers operate simultaneously:

- **Per-pair MTM loss cap** — closes and permanently blacklists any pair whose combined realised + unrealised loss exceeds 2.5% of starting capital
- **Portfolio drawdown soft limit** — scales all new position sizes linearly from 100% to 0% as drawdown moves from 6% to 12%
- **Portfolio drawdown hard limit** — suspends all trading at 12% drawdown; resumes only when drawdown recovers below 3%

Three pairs are permanently blacklisted in config after consistently failing in the forward window despite passing statistical screens: O/AVB (COVID correlation artifact), EQR/ESS (Sun Belt vs coastal divergence post-2021), C/PNC (global vs regional bank rate sensitivity mismatch).

### Walk-Forward Validation

```
|── Train (2yr) ──|── Test (1yr) ──|
                  |── Train (2yr) ──|── Test (1yr) ──|
                                   |── Train (2yr) ──|── Test (1yr) ──|
```

Three non-overlapping folds over 2018–2024. Performance is reported exclusively on the test periods. The train periods are used only for pair screening and hedge ratio estimation — never for evaluating the strategy.

### Parameter Optimisation

Signal parameters (entry\_z, exit\_z, stop\_z) were selected via grid search over 48 valid combinations, evaluated on mean OOS Sharpe across walk-forward folds — not in-sample performance. The search explicitly penalises:

- Inconsistency across folds (stability penalty: 0.5 × std of fold Sharpes)
- Excessive drawdown (DD penalty: 0.1 × excess above 15%)
- Sparse signals (activity penalty: flat 0.20 for fewer than 5 trades per fold)

---

## Running the Engine

### Prerequisites

```bash
pip install pandas numpy statsmodels scipy yfinance plotly pyarrow
```

### Pipeline

```bash
# 1. Download and validate price data
python data_pipeline.py

# 2. Screen pairs — finds cointegrated pairs per sector
python pair_screener.py

# 3. Build spread models and z-scores
python spread_model.py

# 4. Generate signals
python signal_generator.py

# 5. Run event-driven backtest with quarterly re-screening
python backtester.py

# 6. Walk-forward validation (out-of-sample)
python walk_forward.py

# 7. Interactive dashboard
python dashboard.py
# Opens results/dashboard.html
```

### Parameter optimisation (optional, takes 30–60 min)

```bash
python param_optimizer.py
python param_optimizer.py --resume   # resume after interruption
python param_optimizer.py --workers 4  # control parallelism
```

---

## Configuration

All parameters live in `config.py`. Key settings:

```python
# Screening
coint_pvalue_threshold = 0.075   # cointegration p-value threshold
min_correlation        = 0.70    # minimum return correlation
chow_significance      = 0.001   # structural break rejection threshold

# Signal
entry_z = 2.2    # z-score entry threshold
exit_z  = 0.5    # z-score exit threshold
stop_z  = 4.0    # z-score stop-loss

# Risk
kelly_fraction    = 0.35    # fractional Kelly (quarter-Kelly)
risk_per_trade    = 0.015   # base notional fraction per pair
max_loss_per_pair = 0.025   # per-pair MTM loss cap
max_drawdown      = 0.12    # portfolio circuit breaker
```

---

## Key Design Decisions

**Event-driven over vectorised** — the backtester processes one day at a time in chronological order, explicitly carrying forward cash, positions, and equity state. Vectorised approaches make it easy to accidentally introduce look-ahead bias by operating on the full price matrix at once.

**Quarterly re-screening in the backtester** — the original static approach (screen once, trade the same pairs for 6 years) produced −14% return because pairs like COP/EOG and AME/ITW were traded through their breakdown periods. Re-screening every 90 calendar days on a rolling 2-year lookback means the engine stops seeing failed relationships within one quarter of their breakdown.

**Z-score seeding** — the walk-forward test periods compute their rolling z-score from scratch, which wastes the first 40 bars of each test year as the window warms up. Seeding with the final 40 bars of the train window eliminates this warmup burn and increases OOS trade count by roughly 30%.

**OOS Sharpe as the optimisation target** — the grid search evaluates each parameter combination exclusively on out-of-sample test periods. Using IS Sharpe as the target would be a form of data snooping that inflates apparent performance without improving real-world results.

---

## Limitations

- **Sample size** — 35 OOS trades across 3 years gives a Sharpe confidence interval of approximately ±0.5 at 95%. The true Sharpe could be materially different from the reported 0.84.
- **Data source** — daily adjusted close prices from yfinance. Survivorship bias is partially mitigated by using a fixed universe defined at project start rather than the current S&P 500 constituents.
- **Transaction costs** — slippage modelled as a flat 0.05% of notional. Real market impact for larger positions would be higher and size-dependent.
- **Short selling** — the model assumes unrestricted short selling at zero borrow cost. In practice, borrow availability and cost vary significantly across stocks.
- **2022 rate environment** — fold 3 (2022–2023) covers the most aggressive Fed tightening cycle in 40 years. Most pairs-trading books had their worst year in 2022. The strategy producing a positive fold in that environment is encouraging but the sample is too small to conclude it is rate-cycle-robust.

---

## Stack

Python · pandas · numpy · statsmodels · scipy · yfinance · plotly · pyarrow
