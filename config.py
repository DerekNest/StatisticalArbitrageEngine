"""
config.py — Central parameter store for the StatArb engine.

All magic numbers live here. Never hardcode values in other modules.
Change something once, it propagates everywhere.
"""

from dataclasses import dataclass, field
from typing import List


# ---------------------------------------------------------------------------
# Universe definition
# ---------------------------------------------------------------------------
# We screen S&P 500 stocks sector-by-sector.
# Pairs within the same sector share macro exposures → better cointegration.
# ---------------------------------------------------------------------------
# WHY THESE SECTORS:
#   Utilities, Materials, Consumer Staples, Industrials, and Real Estate all
#   share a key property: pricing is anchored to physical costs (fuel, commodites,
#   land, regulation) rather than growth narratives. That anchoring creates the
#   stable long-run equilibria that cointegration requires.
#
# WHY NOT Technology / Consumer Discretionary:
#   Momentum-driven equities violate the stationarity assumption. A pair like
#   NVDA/LRCX may pass a short-window Engle-Granger test during correlated sector
#   rotation, but the hedge ratio drifts irreversibly as one leg compounds at 3x
#   the other. The spread widens without bound — the exact opposite of what we need.
# ---------------------------------------------------------------------------
SECTORS = {
    # Original core — proven cointegration in backtests
    "financials":   ["JPM",  "BAC",  "WFC",  "GS",   "MS",   "C",    "BLK",
                     "SCHW", "USB",  "PNC",  "AXP",  "COF",  "TFC",  "FITB"],
    "energy":       ["XOM",  "CVX",  "COP",  "SLB",  "EOG",  "MPC",
                     "PSX",  "VLO",  "OXY",  "HAL",  "DVN",  "BKR"],
    "healthcare":   ["JNJ",  "UNH",  "PFE",  "ABBV", "MRK",  "TMO",  "ABT",
                     "DHR",  "BMY",  "AMGN", "GILD", "ISRG", "VRTX", "REGN"],

    # New: Utilities — regulated monopolies, anchored to rate environment.
    # Intra-sector pairs (e.g. AEP/EXC, NEE/DUK) are historically very stable.
    "utilities":    ["NEE",  "DUK",  "SO",   "D",    "AEP",  "EXC",  "SRE",
                     "PEG",  "XEL",  "WEC",  "ES",   "ETR",  "PPL",  "EIX"],

    # New: Materials — commodity price anchoring drives tight inter-stock spreads.
    # Miners, chemicals, and packaging companies move together on input costs.
    "insurance":    ["TRV",  "CB",   "ALL",  "PGR",  "MET",  "PRU",
                 "AFL",  "AIG",  "HIG",  "UNM"],
    
    # New: Consumer Staples — NOT consumer discretionary. Staples (food, household
    # products, tobacco) are slow-moving, defensive businesses with predictable
    # cash flows. PG/CL and KO/PEP are the classic pairs-trading textbook examples.
    "homebuilders": ["DHI",  "LEN",  "PHM",  "NVR",  "TOL",  "MDC",
                 "KBH",  "MHO",  "TPH",  "TMHC"],
    
    # New: Industrials — capital goods manufacturers share input costs and
    # demand cycles. HON/EMR, GE/MMM have historically cointegrated well.
    "industrials":  ["HON",  "MMM",  "GE",   "EMR",  "ETN",  "PH",   "ROK",
                     "IR",   "FTV",  "AME",  "ITW",  "DOV",  "XYL",  "GWW"],

    # New: Real Estate — REITs are rate-sensitive, sector-anchored, and
    # capital-constrained. Sub-sector pairs (office/office, retail/retail)
    # share tenant-mix and cap rate exposure.
    "realestate":   ["PLD",  "AMT",  "EQIX", "PSA",  "SPG",  "O",    "VICI",
                     "WY",   "EQR",  "AVB",   "ESS",  "MAA",  "UDR"],
}

ALL_TICKERS: List[str] = [t for tickers in SECTORS.values() for t in tickers]


# ---------------------------------------------------------------------------
# Data mode
# ---------------------------------------------------------------------------
# The engine supports two data resolutions:
#
#   DAILY  ("1d") — Full history back to 2018. Supports 3-fold walk-forward.
#                   12 trades/6yr on core sectors; ~50-80 with expanded universe.
#                   Recommended for primary research and WF validation.
#
#   HOURLY ("1h") — yfinance hard limit: last 730 calendar days only (~2 years).
#                   Cannot be extended via yfinance — use Polygon.io or Alpaca
#                   for longer hourly history. Captures intraday mean-reversion
#                   events invisible to daily close prices.
#                   zscore_window and half-life thresholds auto-scale to hours.
#
# Set DATA_MODE here. All downstream modules read this and adapt automatically.
# ---------------------------------------------------------------------------
DATA_MODE = "1d"   # "1d" | "1h"

# Per-mode defaults — automatically selected in DataConfig below
_DAILY_DEFAULTS  = dict(start="2018-01-01", end="2024-01-01",
                         min_history=252,   interval="1d")
_HOURLY_DEFAULTS = dict(start=None,         end=None,
                         min_history=390*6, interval="1h")
# 390 trading hours/quarter × 6 quarters = minimum 6 months of hourly bars


# ---------------------------------------------------------------------------
# Data parameters
# ---------------------------------------------------------------------------
@dataclass
class DataConfig:
    # Full history for cointegration screening — the longer the better.
    # Screening on 6yr catches regime shifts the Chow test will filter.
    start_date:     str   = "2018-01-01"
    end_date:       str   = "2025-01-01"
    interval:       str   = "1d"
    price_col:      str   = "Adj Close"
    min_history:    int   = 252           # bars needed to approve a ticker
    cache_dir:      str   = "data/"
    session_filter: bool  = True          # 1h only: drop pre/post-market bars

    # --- Rolling production window ---
    # This is the window used by the backtester and live engine.
    # Screening still uses the full history above.
    # In production you re-run pair_screener.py every rescreen_days
    # on the trailing screen_lookback_days of data.
    screen_lookback_days: int = 0       # 0 = full history (best for initial screen)
                                          # Set to 504 only for live quarterly re-screens
    rescreen_every_days:  int = 63       # re-screen quarterly in live mode


# ---------------------------------------------------------------------------
# Pair screening parameters
# ---------------------------------------------------------------------------
@dataclass
class ScreenConfig:
    # Cointegration
    coint_pvalue_threshold: float = 0.08  # Relaxed from 0.05 to account for lower test power on N=504
    min_half_life:          int   = 5      # days
    max_half_life:          int   = 126    # 126 days
    min_correlation:        float = 0.7   # Lowered from 0.65 to capture valid lagged sector pairings

    # Stationarity of the spread
    adf_pvalue_threshold:   float = 0.10   # Relaxed from 0.05 to align with standard 90% CI trading thresholds

    # Structural break detection
    chow_test_enabled:      bool  = True    
    chow_split_ratio:       float = 0.5    
    chow_significance:      float = 0.001  

    # Universe: only test same-sector pairs
    cross_sector:           bool  = False


# ---------------------------------------------------------------------------
# Signal / spread parameters
# ---------------------------------------------------------------------------
@dataclass
class SignalConfig:
    # --- Core thresholds (mode-invariant: 2.0σ entry is a statistical requirement,
    #     not a parameter to tune. The data resolution changes, the math does not.) ---
    entry_z:            float = 2.2    # 2σ structural dislocation — DO NOT LOWER
    exit_z:             float = 0.5   # exit at spread mean — full round-trip capture
    stop_z:             float = 3.2    # stop-loss: beyond 4σ the relationship may be breaking
    hedge_method:       str   = "ols"  # rolling OLS — more stable than Kalman on value stocks

    # --- Resolution-dependent parameters ---
    # Daily (1d): 40-bar window ≈ 2 trading months
    # Hourly (1h): 120-bar window ≈ 3 trading weeks of hours (390 hrs/5wks ÷ ~3)
    #   Rationale: intraday spread oscillations are faster; a 40-hour window is too
    #   short (captures only 1 week) and a 252-hour window is too slow. 120 hours
    #   (~3 trading weeks) matches the typical intraday mean-reversion half-life.
    zscore_window:      int   = 40     # DAILY default — overridden for hourly below
    min_hold_bars:      int   = 1      # DAILY: 1 day minimum hold
                                       # HOURLY: will be set to 2 (2-hour minimum)


# ---------------------------------------------------------------------------
# Risk / position sizing parameters
# ---------------------------------------------------------------------------
@dataclass
class RiskConfig:
    capital:            float = 100_000.0
    max_pairs:          int   = 20
    risk_per_trade:     float = 0.015
    max_drawdown:       float = 0.12
    kelly_fraction:     float = .5   # Kelly scaling factor (1.0 = full Kelly, 0.5 = half-Kelly)
    max_loss_per_pair:  float = 0.025
    commission_pct:     float = 0.000
    slippage_pct:       float = 0.0005

    # Pairs permanently banned from trading regardless of screening results.
    # Add any pair that consistently loses across multiple rescreens —
    # these are structurally unstable relationships that pass statistical
    # tests but fail in the forward window due to regime-specific cointegration.
    pair_blacklist: tuple = (
        "O/AVB",     # net lease vs residential REIT — COVID artifact
        "EQR/ESS",   # Sun Belt vs coastal diverged post-2021 migration
        "C/PNC",     # global SIFI vs regional — different rate sensitivity
    )
    


# ---------------------------------------------------------------------------
# Backtest parameters
# ---------------------------------------------------------------------------
@dataclass
class BacktestConfig:
    # Walk-forward splits
    train_years:        int   = 2       # in-sample window
    test_years:         int   = 1       # out-of-sample window
    n_splits:           int   = 4       # how many WF folds to run

    # Re-screening frequency (how often we re-find pairs in-sample)
    rescreen_days:      int   = 63      # quarterly


# ---------------------------------------------------------------------------
# Master config: one object to pass around
# ---------------------------------------------------------------------------
@dataclass
class Config:
    data:     DataConfig     = field(default_factory=DataConfig)
    screen:   ScreenConfig   = field(default_factory=ScreenConfig)
    signal:   SignalConfig   = field(default_factory=SignalConfig)
    risk:     RiskConfig     = field(default_factory=RiskConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    sectors:  dict           = field(default_factory=lambda: SECTORS)


# ---------------------------------------------------------------------------
# Auto-configure based on DATA_MODE
# ---------------------------------------------------------------------------
def _build_config() -> "Config":
    """
    Build the master Config object, applying DATA_MODE overrides automatically.
    Downstream modules import CFG and never need to check DATA_MODE directly.
    """
    cfg = Config()

    if DATA_MODE == "1h":
        from datetime import datetime, timedelta
        # yfinance 1h limit: 730 calendar days back from today
        end   = datetime.today().strftime("%Y-%m-%d")
        start = (datetime.today() - timedelta(days=729)).strftime("%Y-%m-%d")
        cfg.data.interval    = "1h"
        cfg.data.start_date  = start
        cfg.data.end_date    = end
        cfg.data.min_history = 390 * 3    # 3 months of hourly bars minimum
        cfg.signal.zscore_window = 120    # ~3 trading weeks of hours
        cfg.signal.min_hold_bars = 2      # minimum 2-bar (2-hour) hold
        # Tighten half-life bounds for intraday: 2hr minimum, 2 trading days max
        cfg.screen.min_half_life = 2      # bars (hours)
        cfg.screen.max_half_life = 16     # bars (hours) — ~2 trading days
    else:
        cfg.data.interval    = "1d"
        cfg.signal.zscore_window = 20
        cfg.signal.min_hold_bars = 1
        cfg.screen.min_half_life = 5      # days
        cfg.screen.max_half_life = 126    # days

    return cfg


# Convenience singleton — import this in other modules
CFG = _build_config()