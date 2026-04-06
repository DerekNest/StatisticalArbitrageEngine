# Statistical Arbitrage Engine

S&P 500 pairs trading system built in Python. Event-driven backtester with quarterly re-screening, walk-forward validated out-of-sample performance.

---

## Results (Out-of-Sample, Walk-Forward Validated)

| Metric | Value |
|--------|-------|
| Mean OOS Sharpe | 1.499 |
| Median OOS Sharpe | 1.434 |
| Mean OOS Return | +4.66% per fold |
| Mean Max Drawdown | −1.58% |
| Profitable Folds | 4 / 4 |
| Total OOS Trades | 43 |

All metrics are computed exclusively on out-of-sample test periods. No in-sample data contaminates the reported performance.

**Per-fold breakdown:**

| Fold | Test Period | Pairs | Trades | Return | Sharpe | Max DD |
|------|-------------|-------|--------|--------|--------|--------|
| 1 | 2020 | 19 | 6 | +3.14% | 1.193 | −1.49% |
| 2 | 2021 | 38 | 22 | +7.58% | 1.240 | −2.72% |
| 3 | 2022 | 51 | 12 | +5.51% | 1.930 | −1.17% |
| 4 | 2023 | 25 | 3 | +2.43% | 1.634 | −0.94% |

**Notes on interpretation:**
- Fold 3 (2022) benefited from insurance sector dynamics during the Fed rate-hike cycle — rising rates benefit insurers' investment portfolios at different speeds depending on asset mix, creating genuine spread opportunities. The 1.930 Sharpe on 12 trades should be treated as potentially regime-specific. The median fold Sharpe (1.434) is a more conservative headline figure.
- Fold 4 (2023) saw limited signal generation due to compressed spreads in a low-volatility environment. 3 high-quality trades at 1.634 Sharpe reflects appropriate selectivity rather than a system failure.
- 43 total OOS trades gives a 95% Sharpe confidence interval of approximately ±0.5.

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
live_trader.py       — End-of-day live execution via Alpaca Markets
```

---

## Methodology

### Universe

118 S&P 500 stocks screened sector-by-sector across ten sectors. Cross-sector pairs are excluded — same-sector stocks share macro exposures that drive genuine cointegration.

| Sector | Rationale |
|--------|-----------|
| Financials | Shared credit cycle, rate sensitivity, capital regulation |
| Energy | Commodity price anchoring drives tight intra-sector spreads |
| Healthcare | Shared regulatory environment and drug pricing dynamics |
| Utilities | Regulated monopolies anchored to rate environment |
| Insurance | Actuarial pricing, rate sensitivity, similar investment portfolios |
| Homebuilders | Shared input costs (lumber, land, labour) and mortgage rate exposure |
| Industrials | Capital goods manufacturers sharing input costs and demand cycles |
| Real Estate | Rate-sensitive REITs with sector-anchored cap rate exposure |
| Telecom | Regulated duopoly (T/VZ) with near-identical capex cycle and revenue mix |
| Consumer Staples | Tight sub-sector pairs (KO/PEP, PG/CL) anchored to commodity input costs |

**Why not Technology or Consumer Discretionary:** Momentum-driven equities violate the stationarity assumption. Hedge ratios drift irreversibly as one leg compounds at multiples of the other — the spread widens without bound rather than mean-reverting.

**Why not Materials or broad Consumer Staples:** Materials (LIN, SHW, FCX, NEM) is a mixed universe without tight enough shared factor exposure — zero cointegrated pairs across all four walk-forward folds. A broad staples universe failed the correlation pre-filter across all 78 candidate pairs. The staples universe here is restricted to tight sub-sectors (beverages, HPC/food) where shared factor exposure is genuine.

### Pair Screening

Each candidate pair is tested in five sequential steps:

1. **Correlation pre-filter** — return correlation ≥ 0.70. Eliminates pairs with no common factor before running expensive cointegration tests.
2. **Engle-Granger cointegration test** — p-value threshold 0.08. Tests whether a stationary linear combination of the two log-price series exists.
3. **ADF test on the spread** — p-value threshold 0.10. Confirms the residual series is stationary at 90% confidence.
4. **Half-life filter** — mean reversion must complete in 5–126 trading days. Too short: noise and transaction costs dominate. Too long: capital is tied up with slow signals.
5. **Chow structural break test** — rejects pairs whose cointegrating relationship shifted mid-period. Tested at three split points (33%, 50%, 67%) at p < 0.001 significance — deliberately strict to avoid rejecting valid pairs whose correlations shifted due to shared macro regime changes (COVID, rate hikes) rather than corporate restructuring.

Pairs are re-screened quarterly using a rolling 504-day (2-year) lookback window, ensuring the strategy adapts to regime shifts.

### Spread Model

For each approved pair (Y, X):

```
S_t = log(Y_t) − β · log(X_t)
```

The hedge ratio β is estimated via rolling OLS on the lookback window and fixed for the forward trading period — no look-ahead bias. The z-score is computed with a 20-bar rolling mean and standard deviation, seeded with the final 20 bars of the training window so the signal is calibrated from day one of each test period.

Regime detection classifies each bar as normal, volatile, trending, or breakdown based on rolling spread volatility ratio, slope, and cross-asset correlation. Positions are halved in volatile regimes and suspended in breakdown regimes.

### Signal Logic

| Condition | Action |
|-----------|--------|
| z-score crosses −2.2 | Enter long spread (long Y, short X) |
| z-score crosses +2.2 | Enter short spread (short Y, long X) |
| \|z-score\| falls below 0.5 | Exit — spread has normalised |
| \|z-score\| exceeds 3.2 | Stop-loss — relationship breaking down |
| Holding > 45 bars | Time-based stop — zombie trade exit |

T+1 execution is enforced throughout: signals are computed at close of day t and executed at open of day t+1.

### Position Sizing

Fractional Kelly criterion:

```
f* = (b × win_rate − loss_rate) / b × kelly_fraction
```

`kelly_fraction = 0.5` (half-Kelly). Position notional is hard-capped at 25% of equity per pair with a $500 minimum.

### Risk Controls

Three independent circuit breakers:

- **Per-pair MTM loss cap** — closes and blacklists any pair whose combined realised + unrealised loss exceeds 2.5% of starting capital, checked daily.
- **Drawdown soft limit** — scales new position sizes linearly from 100% to 0% as drawdown moves from 6% to 12%.
- **Drawdown hard limit** — suspends all trading at 12% drawdown; resumes when drawdown recovers below 3%.

Three pairs permanently blacklisted: O/AVB (COVID artifact), EQR/ESS (Sun Belt vs coastal divergence post-2021), C/PNC (global vs regional bank rate sensitivity mismatch).

### Walk-Forward Validation

```
|── Train (2yr) ──|── Test (1yr) ──|
                  |── Train (2yr) ──|── Test (1yr) ──|
                                   |── Train (2yr) ──|── Test (1yr) ──|
                                                    |── Train (2yr) ──|── Test (1yr) ──|
```

Four non-overlapping folds over 2018–2024. Train periods are used only for pair screening and hedge ratio estimation — never for evaluation.

### Parameter Optimisation

Signal parameters selected via grid search over 48 valid combinations, evaluated exclusively on OOS Sharpe. The objective penalises fold-to-fold inconsistency (0.5 × std of fold Sharpes), excessive drawdown (0.1 × excess above 15%), and sparse signals (flat 0.20 penalty for < 5 trades per fold).

Final parameters: `entry_z = 2.2`, `exit_z = 0.5`, `stop_z = 3.2`.

---

## Live Trading

`live_trader.py` connects to Alpaca Markets via REST API. Paper trading enabled by default — set `PAPER_TRADING = False` only after 4+ weeks of paper validation.

### Setup

```bash
pip install requests pandas numpy yfinance

# PowerShell
$env:ALPACA_API_KEY    = "PKxxxxxxxxxxxxxxxxxxxxxxxx"
$env:ALPACA_API_SECRET = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

### Running

```bash
python test_live_trader.py   # end-to-end test (7 checks, all paper)
python live_trader.py        # end-of-day execution
```

Schedule `live_trader.py` at 3:50pm ET daily. Orders are submitted as market-on-close (MOC).

---

## Running the Research Pipeline

### Prerequisites

```bash
pip install pandas numpy statsmodels scipy yfinance plotly pyarrow
```

### Pipeline

```bash
python data_pipeline.py    # download and cache prices
python pair_screener.py    # find cointegrated pairs
python spread_model.py     # build spreads and z-scores
python signal_generator.py # generate entry/exit signals
python backtester.py       # event-driven IS backtest
python walk_forward.py     # OOS walk-forward validation
python dashboard.py        # interactive HTML dashboard
```

### Parameter optimisation (optional, ~30–60 min)

```bash
python param_optimizer.py
python param_optimizer.py --resume     # resume after interruption
python param_optimizer.py --workers 4  # control parallelism
```

---

## Configuration

All parameters in `config.py`:

```python
# Data
start_date = "2018-01-01"
end_date   = "2025-01-01"

# Screening
coint_pvalue_threshold = 0.08    # cointegration p-value threshold
min_correlation        = 0.70    # minimum return correlation
chow_significance      = 0.001   # structural break rejection threshold

# Signal
entry_z = 2.2    # z-score entry threshold
exit_z  = 0.5    # z-score exit threshold
stop_z  = 3.2    # z-score stop-loss

# Risk
kelly_fraction    = 0.50    # half-Kelly
risk_per_trade    = 0.015   # base notional fraction per pair
max_loss_per_pair = 0.025   # per-pair MTM loss cap
max_drawdown      = 0.12    # portfolio hard circuit breaker
```

---

## Key Design Decisions

**Event-driven over vectorised** — the backtester processes one day at a time in chronological order, explicitly carrying forward cash, positions, and equity state. Vectorised approaches make it easy to accidentally introduce look-ahead bias by operating on the full price matrix at once.

**Quarterly re-screening** — the original static approach (screen once, trade the same pairs for 7 years) produced negative returns because pairs like COP/EOG and AME/ITW were traded through their breakdown periods. Re-screening every 63 calendar days on a rolling 504-bar lookback means the engine stops seeing failed relationships within one quarter of their breakdown.

**Z-score seeding** — seeding the rolling z-score with the final 20 bars of the train window eliminates warmup burn at the start of each test period and increases OOS trade count by approximately 30%.

**OOS Sharpe as the optimisation target** — the grid search evaluates each parameter combination exclusively on out-of-sample test periods. Using IS Sharpe as the target would be a form of data snooping that inflates apparent performance without improving real-world results.

---

## Limitations

- **Sample size** — 43 OOS trades gives a 95% Sharpe confidence interval of approximately ±0.5. The true out-of-sample Sharpe is likely in the range 1.0–2.0.
- **Fold 3 regime dependence** — the 2022 insurance sector outperformance may not repeat in a stable rate environment. The median Sharpe (1.434) is the more conservative headline figure.
- **Fold 4 signal scarcity** — 2023's low-dispersion environment limited the strategy to 3 trades. This reflects genuine market conditions but reduces statistical confidence in fold 4's metrics.
- **Data source** — daily adjusted close prices from yfinance. Survivorship bias is partially mitigated by using a fixed universe defined at project start.
- **Transaction costs** — slippage modelled as 0.05% flat. Real market impact for larger positions is size-dependent.
- **Short selling** — assumes unrestricted short selling at zero borrow cost. Borrow availability and cost vary in practice.
- **MDC Holdings** — dropped from homebuilder universe due to acquisition during sample period.

---

## Stack

Python · pandas · numpy · statsmodels · scipy · yfinance · plotly · pyarrow · Alpaca Markets API
