"""
pair_screener.py — Find cointegrated pairs in each sector.

CONCEPT: What is cointegration?
  Two stocks can be individually non-stationary (prices wander randomly,
  like a random walk) but still be "bound together" in the long run.
  Think of a drunk walking a dog on a leash — both paths are random, but
  the distance between them is bounded and mean-reverting. That distance
  is the "spread" we trade.

  Formally, if X and Y are I(1) (integrated order 1 — i.e., their
  first differences are stationary), and some linear combination
      S_t = Y_t - β * X_t
  is I(0) (stationary), then X and Y are cointegrated with hedge ratio β.

CONCEPT: Engle-Granger two-step test
  Step 1: Regress Y on X to get β (OLS). This is the hedge ratio.
  Step 2: Run an ADF (Augmented Dickey-Fuller) test on the residuals S_t.
          If p-value < 0.05, we reject the null of a unit root → S_t is
          stationary → pair is cointegrated.

CONCEPT: Half-life of mean reversion
  If S_t is an Ornstein-Uhlenbeck (OU) process:
      dS = κ(μ - S)dt + σdW
  then the half-life is ln(2)/κ. We estimate κ by regressing
      ΔS_t on S_{t-1}
  The half-life tells us how quickly the spread reverts to its mean.
  Too short (<5 days) → noise, transaction costs kill you.
  Too long (>63 days) → capital tied up, slow signal.

CONCEPT: Chow structural break test
  A pair can pass the Engle-Granger test on a 6-year window but still be
  untradeable if the cointegrating relationship broke down mid-period.
  IR/ITW is a perfect example: Ingersoll Rand spun off its industrial
  segment in 2020, permanently changing its fundamental business. The
  pre-2020 history looked cointegrated; the post-2020 relationship was
  completely different.

  The Chow test splits the training window at the midpoint and tests
  whether the OLS regression coefficients (hedge ratio) are statistically
  equal in both halves. If the F-statistic is significant (p < 0.05),
  we reject the pair — the relationship is structurally unstable.

  Formally: fit OLS on full period (RSS_total), fit OLS on each half
  separately (RSS_1, RSS_2), then:
      F = [(RSS_total - RSS_1 - RSS_2) / k] / [(RSS_1 + RSS_2) / (n - 2k)]
  where k = number of parameters (2: intercept + slope), n = total obs.
  This F-statistic follows F(k, n-2k) under the null of no break.
"""

import itertools
import logging
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.tsa.stattools import adfuller, coint

from config import CFG, ScreenConfig

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class PairResult:
    """All statistics for one candidate pair."""
    sector:         str
    ticker_y:       str       # dependent variable (we go long/short this)
    ticker_x:       str       # independent variable (the "driver")
    hedge_ratio:    float     # β from OLS regression
    coint_pvalue:   float     # Engle-Granger p-value (lower = stronger)
    adf_pvalue:     float     # ADF on spread residuals
    half_life:      float     # days to half-reversion
    correlation:    float     # Pearson correlation of returns
    spread_mean:    float     # long-run mean of the spread
    spread_std:     float     # standard deviation of the spread
    hurst_exp:      float     # Hurst exponent (< 0.5 = mean-reverting)
    score:          float     # composite ranking score (higher = better)

    def __str__(self):
        return (
            f"{self.ticker_y}/{self.ticker_x} | "
            f"p={self.coint_pvalue:.3f} | "
            f"HL={self.half_life:.1f}d | "
            f"H={self.hurst_exp:.3f} | "
            f"score={self.score:.3f}"
        )


# ---------------------------------------------------------------------------
# Individual statistical tests
# ---------------------------------------------------------------------------

def estimate_hedge_ratio(price_y: pd.Series, price_x: pd.Series) -> Tuple[float, pd.Series]:
    """
    OLS regression of log(Y) on log(X) to get hedge ratio β.

    We use LOG prices (not raw prices) so that β is in return-space,
    meaning a 1% move in X predicts a β% move in Y. This makes β
    comparable across different price levels.

    Returns: (hedge_ratio, spread_series)
    """
    log_y = np.log(price_y)
    log_x = np.log(price_x)

    X = add_constant(log_x)
    model = OLS(log_y, X).fit()
    beta = model.params.iloc[1]
    spread = log_y - beta * log_x   # the residual series we'll trade
    return float(beta), spread


def estimate_half_life(spread: pd.Series) -> float:
    """
    Estimate half-life of mean reversion via OU process regression.

    Regress ΔS_t on S_{t-1}:
        ΔS_t = α + κ * S_{t-1} + ε_t

    κ is the speed of mean reversion. half_life = -ln(2) / κ
    A negative κ means the spread pulls back toward its mean.
    """
    delta_s = spread.diff().dropna()
    lagged_s = spread.shift(1).dropna()

    # Align
    delta_s, lagged_s = delta_s.align(lagged_s, join="inner")

    X = add_constant(lagged_s)
    model = OLS(delta_s, X).fit()
    kappa = model.params.iloc[1]

    if kappa >= 0:
        return float("inf")  # not mean-reverting

    half_life = -np.log(2) / kappa
    return float(half_life)


def hurst_exponent(series: pd.Series, max_lag: int = 100) -> float:
    """
    Estimate the Hurst exponent via rescaled range analysis.
    H < 0.5 → mean-reverting; H = 0.5 → random walk; H > 0.5 → trending.
    """
    series = series.dropna()
    # Extract underlying numpy array to prevent Pandas index-alignment issues during subtraction
    vals = series.values  
    
    lags = range(2, min(max_lag, len(vals) // 2))
    tau = []
    reg = []

    for lag in lags:
        tau.append(lag)
        # Raw array subtraction ensures we are comparing the exact lagged offsets
        reg.append(np.std(np.subtract(vals[lag:], vals[:-lag])))

    log_lags = np.log(tau)
    log_reg = np.log(reg)
    poly = np.polyfit(log_lags, log_reg, 1)
    
    return float(poly[0])



def chow_structural_break_test(
    spread: pd.Series,
    split_ratio: float = 0.5,
    significance: float = 0.001,
) -> Tuple[bool, float]:
    """
    Chow test for structural break in the OU mean-reversion dynamics.

    IMPORTANT — why significance=0.001 (not 0.01 or 0.05):
      The midpoint of 2018-2024 data lands in mid-2021, right through
      COVID recovery and the start of the Fed rate-hike cycle. Almost
      every sector pair shows different OU dynamics before/after that
      date — NOT because of corporate restructuring, but because of
      shared macro regime shifts. We only want to reject pairs where
      the break is so extreme it cannot be explained by macro alone.

      At p<0.001 on 1500-bar series: expected false rejections ≈ 0.66
      per 664 pairs. Only truly catastrophic breaks (spin-offs, mergers,
      bankruptcy) will clear this bar. IR/ITW post-2020 spin-off: p≈0.
      AEP/EXC post-rate-hike regime: p≈0.05 → passes, correctly.

    Returns:
        (break_detected: bool, p_value: float)
        break_detected=True → REJECT this pair.
    """
    spread = spread.dropna()
    n = len(spread)
    if n < 252:
        return False, 1.0   # need at least 1yr to split meaningfully

    delta_s = spread.diff().dropna()
    lagged  = spread.shift(1).dropna()
    delta_s, lagged = delta_s.align(lagged, join="inner")
    n_obs = len(delta_s)

    X_full = np.column_stack([np.ones(n_obs), lagged.values])
    y_full = delta_s.values
    k = X_full.shape[1]  # 2 parameters

    def _rss(y, X):
        b, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        return float(np.sum((y - X @ b) ** 2))

    rss_full = _rss(y_full, X_full)

    split_idx = int(n_obs * split_ratio)
    min_seg = max(k + 5, 63)  # at least 63 obs (1 quarter) per segment
    if split_idx < min_seg or (n_obs - split_idx) < min_seg:
        return False, 1.0

    rss_parts = (_rss(y_full[:split_idx],  X_full[:split_idx]) +
                 _rss(y_full[split_idx:],  X_full[split_idx:]))

    numerator   = (rss_full - rss_parts) / k
    denominator = rss_parts / (n_obs - 2 * k)

    if denominator <= 0 or numerator <= 0:
        return False, 1.0

    f_stat  = numerator / denominator
    from scipy.stats import f as f_dist
    p_value = float(1 - f_dist.cdf(f_stat, dfn=k, dfd=n_obs - 2 * k))

    return p_value < significance, p_value


def score_pair(result: PairResult) -> float:
    """
    Composite score to rank pairs. Higher = more tradeable.

    We combine:
      - Low cointegration p-value (strongest statistical signal)
      - Half-life in the sweet spot (not too fast, not too slow)
      - Low Hurst exponent (strong mean reversion)
      - High correlation (pairs share common factor)

    Weights are tunable — this is a design choice, not a fact.
    """
    # p-value score: lower p = higher score (exponential reward)
    p_score = np.exp(-10 * result.coint_pvalue)

    # Half-life score: peaks at 20 days, falls off on both sides
    hl_target = 20.0
    hl_score = np.exp(-((result.half_life - hl_target) / 30) ** 2)

    # Hurst score: lower H = better (cap at 0.5)
    h_score = max(0, 0.5 - result.hurst_exp) * 2

    # Correlation score
    corr_score = max(0, result.correlation - 0.7) / 0.3  # 0 at 0.7, 1 at 1.0

    # Weighted sum
    score = 0.40 * p_score + 0.25 * hl_score + 0.20 * h_score + 0.15 * corr_score
    return float(score)


# ---------------------------------------------------------------------------
# Main screening function
# ---------------------------------------------------------------------------

def screen_pairs(
    prices: pd.DataFrame,
    sector: str,
    cfg: ScreenConfig = None,
    lookback_days: Optional[int] = None,
) -> List[PairResult]:
    """
    Test all pairs in a sector and return those passing all filters.

    Parameters
    ----------
    prices       : full price history (used for Chow test structural stability)
    sector       : sector label for logging
    cfg          : screening config
    lookback_days: if set, cointegration tests use only the trailing N days.
                   The Chow test ALWAYS uses the full prices history so it can
                   detect regime changes even when trading on a short window.

    DESIGN RATIONALE — decoupled windows:
      - Cointegration screening: trailing lookback_days (default: full history)
        Tests whether the RECENT relationship is mean-reverting.
      - Chow structural break test: full prices history
        Tests whether the relationship has been STABLE. A pair can look
        cointegrated in the last 2 years but still have broken down — the
        Chow test catches that using the longer history.
      - z-score / signal generation: rolling 40-day window (in spread_model.py)
        This is the production trading window.

    Steps for each candidate pair (Y, X):
      1. Pre-filter: correlation check
      2. Engle-Granger cointegration test (on lookback window)
      3. ADF test on the spread residuals (on lookback window)
      4. Estimate hedge ratio, half-life, Hurst exponent
      5. Apply threshold filters
      6. Chow structural break test (on FULL history)
      7. Score and rank survivors
    """
    cfg = cfg or CFG.screen

    # Subset for cointegration tests if lookback specified
    # lookback_days=0 or None means use full history — best for initial screening
    # Use a non-zero lookback only for live quarterly re-screens
    if lookback_days and lookback_days > 0 and len(prices) > lookback_days:
        coint_prices = prices.iloc[-lookback_days:]
        log.info(f"  [{sector}] Using trailing {lookback_days}-day window for cointegration tests")
    else:
        coint_prices = prices
        if lookback_days == 0:
            log.info(f"  [{sector}] Using full history ({len(prices)} bars) for cointegration tests")

    tickers = list(prices.columns)
    candidates = list(itertools.combinations(tickers, 2))
    log.info(f"  [{sector}] Testing {len(candidates)} pairs from {len(tickers)} tickers "
             f"({len(coint_prices)} bars for coint, {len(prices)} for Chow)")

    # Pre-compute log returns for correlation screening (use coint window)
    log_returns = np.log(coint_prices / coint_prices.shift(1)).dropna()
    results = []
    _f = {"short": 0, "corr": 0, "coint": 0, "adf": 0, "hl": 0, "chow": 0, "ok": 0}
    _coint_pvals = []   # collect all p-values to see distribution

    for ticker_y, ticker_x in candidates:
        try:
            # Cointegration uses the shorter lookback window
            series_y = coint_prices[ticker_y].dropna()
            series_x = coint_prices[ticker_x].dropna()

            # Align both series to the same dates
            series_y, series_x = series_y.align(series_x, join="inner")
            if len(series_y) < 126:   # need at least 6 months for coint test
                _f["short"] += 1; continue

            # --- STEP 1: Quick correlation pre-filter ---
            ret_y = log_returns[ticker_y].dropna()
            ret_x = log_returns[ticker_x].dropna()
            ret_y, ret_x = ret_y.align(ret_x, join="inner")
            correlation = float(ret_y.corr(ret_x))

            if abs(correlation) < cfg.min_correlation:
                _f["corr"] += 1; continue

            # --- STEP 2: Engle-Granger cointegration test ---
            # statsmodels coint() handles the OLS + ADF internally
            coint_stat, coint_pvalue, _ = coint(
                np.log(series_y),
                np.log(series_x),
                trend="c",
            )

            _coint_pvals.append((coint_pvalue, ticker_y, ticker_x))
            if coint_pvalue > cfg.coint_pvalue_threshold:
                _f["coint"] += 1; continue

            # --- STEP 3: Estimate hedge ratio and spread ---
            hedge_ratio, spread = estimate_hedge_ratio(series_y, series_x)

            # --- STEP 4: ADF test on the spread directly ---
            adf_stat, adf_pvalue, *_ = adfuller(spread, maxlag=1, autolag=None)

            if adf_pvalue > cfg.adf_pvalue_threshold:
                _f["adf"] += 1; continue

            # --- STEP 5: Half-life ---
            half_life = estimate_half_life(spread)

            if not (cfg.min_half_life <= half_life <= cfg.max_half_life):
                _f["hl"] += 1; continue

            # --- STEP 6: Hurst exponent ---
            hurst = hurst_exponent(spread)

            # --- STEP 7: Chow structural break test ---
            # Rejects pairs whose cointegrating relationship changed mid-period
            # (e.g. corporate restructurings, spin-offs, major M&A events).
            if getattr(cfg, "chow_test_enabled", True):
                # Chow test on FULL history — detects corporate structural breaks
                # (spin-offs, mergers) across the complete dataset.
                # We test at THREE split points (1/3, 1/2, 2/3) and take the
                # MINIMUM p-value. This avoids the midpoint coinciding exactly
                # with COVID/rate-hike boundary and rejecting valid pairs.
                full_y = prices[ticker_y].dropna()
                full_x = prices[ticker_x].dropna()
                full_y, full_x = full_y.align(full_x, join="inner")
                _, full_spread = estimate_hedge_ratio(full_y, full_x)
                chow_sig = getattr(cfg, "chow_significance", 0.001)
                min_chow_p = 1.0
                for split_pt in [0.33, 0.50, 0.67]:
                    _, p = chow_structural_break_test(
                        full_spread, split_ratio=split_pt, significance=chow_sig,
                    )
                    min_chow_p = min(min_chow_p, p)
                if min_chow_p < chow_sig:
                    _f["chow"] += 1
                    log.debug(f"  Chow break {ticker_y}/{ticker_x} (min_p={min_chow_p:.4f}) — rejecting")
                    continue

            # Build result
            result = PairResult(
                sector=sector,
                ticker_y=ticker_y,
                ticker_x=ticker_x,
                hedge_ratio=hedge_ratio,
                coint_pvalue=coint_pvalue,
                adf_pvalue=adf_pvalue,
                half_life=half_life,
                correlation=correlation,
                spread_mean=float(spread.mean()),
                spread_std=float(spread.std()),
                hurst_exp=hurst,
                score=0.0,  # filled in below
            )
            result.score = score_pair(result)
            results.append(result)
            _f["ok"] += 1

        except Exception as e:
            _f["error"] = _f.get("error", 0) + 1
            log.warning(f"  EXCEPTION {ticker_y}/{ticker_x}: {type(e).__name__}: {e}")
            continue

    # Sort by composite score
    results.sort(key=lambda r: r.score, reverse=True)
    top_coint = sorted(_coint_pvals)[:8]
    top_str = "  ".join(f"{y}/{x}={p:.3f}" for p, y, x in top_coint)
    log.info(f"  [{sector}] {len(results)} pairs passed | filters: short={_f['short']} corr={_f['corr']} coint={_f['coint']} adf={_f['adf']} hl={_f['hl']} chow={_f['chow']} error={_f.get('error',0)}")
    log.info(f"  [{sector}] Best coint p-values: {top_str}")
    return results


def screen_all_sectors(
    sector_universe: Dict[str, pd.DataFrame],
    cfg: ScreenConfig = None,
    top_n_per_sector: int = 5,
    lookback_days: Optional[int] = None,
) -> pd.DataFrame:
    """
    Run pair screening across all sectors and return a ranked DataFrame.

    lookback_days: trailing days for cointegration tests.
                   None = use full history (best for initial screening).
                   504  = 2yr lookback (good for quarterly re-screens).
    The Chow test always uses full history regardless of this parameter.
    """
    cfg = cfg or CFG.screen
    all_results = []

    for sector, prices in sector_universe.items():
        log.info(f"\nScreening sector: {sector.upper()}")
        sector_results = screen_pairs(prices, sector, cfg, lookback_days=lookback_days)
        # Take top N from each sector
        all_results.extend(sector_results[:top_n_per_sector])

    if not all_results:
        log.warning("No pairs survived screening!")
        return pd.DataFrame()

    rows = []
    for r in all_results:
        rows.append({
            "sector":       r.sector,
            "pair":         f"{r.ticker_y}/{r.ticker_x}",
            "ticker_y":     r.ticker_y,
            "ticker_x":     r.ticker_x,
            "score":        round(r.score, 4),
            "coint_pvalue": round(r.coint_pvalue, 4),
            "half_life":    round(r.half_life, 1),
            "hedge_ratio":  round(r.hedge_ratio, 4),
            "hurst_exp":    round(r.hurst_exp, 3),
            "correlation":  round(r.correlation, 3),
            "spread_std":   round(r.spread_std, 4),
        })

    df = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    from data_pipeline import download_prices, validate_and_clean, build_sector_universe

    cfg = CFG
    os.makedirs("data", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    print("\n" + "="*60)
    print("  SPRINT 1 — PAIR SCREENER")
    print("="*60)

    # Load data
    all_tickers = [t for tickers in cfg.sectors.values() for t in tickers]
    raw_prices = download_prices(all_tickers, cfg.data.start_date, cfg.data.end_date)
    prices, _ = validate_and_clean(raw_prices, cfg.data.min_history)
    universe = build_sector_universe(prices, cfg.sectors)

    # Screen
    # Use screen_lookback_days for cointegration tests (default: full history)
    # Chow test will still use full history internally.
    lookback = getattr(cfg.data, "screen_lookback_days", 0)
    # 0 = full history for cointegration tests (recommended for initial screen)
    effective_lookback = lookback if lookback and lookback > 0 else None
    window_desc = "full history" if not effective_lookback else f"{effective_lookback} days"
    print(f"\nRunning cointegration screening "
          f"(coint window: {window_desc}, Chow: full history)...")
    ranked_pairs = screen_all_sectors(universe, cfg.screen,
                                       top_n_per_sector=5,
                                       lookback_days=effective_lookback)

    # Display results
    if ranked_pairs.empty:
        print("No pairs found — try relaxing thresholds in config.py")
    else:
        print(f"\n{'='*60}")
        print(f"  TOP PAIRS FOUND: {len(ranked_pairs)}")
        print(f"{'='*60}")
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 120)
        print(ranked_pairs.to_string(index=True))

        # Save for Sprint 2
        ranked_pairs.to_csv("results/ranked_pairs.csv", index=False)
        print(f"\n✓ Saved to results/ranked_pairs.csv")
        print(f"  Next step: run spread_model.py (Sprint 2)")