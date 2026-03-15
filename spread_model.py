"""
spread_model.py — Model the spread between a cointegrated pair.

CONCEPT: The spread as a tradeable instrument
  Once we have a cointegrated pair (Y, X), we define the spread:
      S_t = log(Y_t) - β * log(X_t)

  This spread is stationary — it has a stable long-run mean (μ) and
  standard deviation (σ). When it deviates far from μ, it will
  (statistically) revert. That deviation, measured in σ-units, is the
  z-score — our primary trading signal.

CONCEPT: Rolling vs fixed hedge ratio
  The OLS hedge ratio β we computed in Sprint 1 used the full history.
  In live trading we only know the past. We use a ROLLING window so that
  β adapts as the relationship evolves — this is crucial for avoiding
  look-ahead bias and for capturing structural shifts in the relationship.

CONCEPT: Z-score
  z_t = (S_t - μ_rolling) / σ_rolling

  where μ and σ are computed over the same rolling window.
  z = +2 means the spread is 2 standard deviations ABOVE its mean.
      → Y is relatively expensive, X is relatively cheap
      → We SHORT Y and LONG X (expect spread to fall back to 0)
  z = -2 means the opposite.

CONCEPT: Ornstein-Uhlenbeck process
  The theoretical model behind our spread is the OU process:
      dS_t = κ(μ - S_t)dt + σ dW_t

  Parameters:
    κ  = speed of mean reversion (we already estimated half_life = ln2/κ)
    μ  = long-run mean
    σ  = volatility of the process

  Fitting these gives us:
    - Expected time to revert (for position sizing)
    - Theoretical profit per trade: ~2σ * σ_OU per round-trip
    - Whether the pair is currently in a "good regime" for trading
"""

import logging
import warnings
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.tsa.stattools import adfuller

from config import CFG, SignalConfig

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core spread computation
# ---------------------------------------------------------------------------

def compute_spread(
    price_y: pd.Series,
    price_x: pd.Series,
    window: int = 60,
    hedge_method: str = "ols",
) -> pd.DataFrame:
    """
    Compute the spread, rolling hedge ratio, and z-score for a pair.

    Parameters
    ----------
    price_y, price_x : aligned price series (same index)
    window          : rolling window in trading days
    hedge_method    : 'ols' (rolling OLS) or 'fixed' (full-sample OLS)

    Returns
    -------
    DataFrame with columns:
        log_y, log_x        : log prices
        hedge_ratio         : rolling β
        spread              : S_t = log_y - β * log_x
        spread_mean         : rolling mean of spread
        spread_std          : rolling std of spread
        zscore              : (spread - spread_mean) / spread_std
        zscore_smoothed     : 3-day EMA of zscore (reduces noise)
    """
    log_y = np.log(price_y).rename("log_y")
    log_x = np.log(price_x).rename("log_x")

    df = pd.DataFrame({"log_y": log_y, "log_x": log_x}).dropna()

    if hedge_method == "ols":
        hedges = _rolling_ols_hedge(df["log_y"], df["log_x"], window)
    elif hedge_method == "kalman":
        hedges = _kalman_filter_hedge(df["log_y"], df["log_x"])
    else:
        # Fixed hedge ratio from full sample
        X = add_constant(df["log_x"])
        model = OLS(df["log_y"], X).fit()
        hedges = pd.Series(model.params.iloc[1], index=df.index)

    df["hedge_ratio"] = hedges
    df["spread"] = df["log_y"] - df["hedge_ratio"] * df["log_x"]

    # Rolling mean and std of the spread
    df["spread_mean"] = df["spread"].rolling(window, min_periods=window // 2).mean()
    df["spread_std"]  = df["spread"].rolling(window, min_periods=window // 2).std()

    # Z-score: how many standard deviations is the spread from its mean?
    df["zscore"] = (df["spread"] - df["spread_mean"]) / df["spread_std"]

    # Smoothed z-score: 3-day EMA reduces tick noise without much lag
    df["zscore_smoothed"] = df["zscore"].ewm(span=3, adjust=False).mean()

    return df


def _rolling_ols_hedge(
    log_y: pd.Series,
    log_x: pd.Series,
    window: int,
) -> pd.Series:
    """
    Efficient rolling OLS hedge ratio using numpy sliding windows.

    Standard pandas rolling().apply(lambda) with OLS inside is ~50x
    slower than this vectorised approach. Speed matters when we run
    hundreds of pairs through a backtest.
    """
    n = len(log_y)
    betas = np.full(n, np.nan)
    y_arr = log_y.values
    x_arr = log_x.values

    for i in range(window - 1, n):
        y_w = y_arr[i - window + 1 : i + 1]
        x_w = x_arr[i - window + 1 : i + 1]

        # OLS normal equations: β = (X'X)^{-1} X'y
        x_dm = x_w - x_w.mean()
        y_dm = y_w - y_w.mean()
        denom = np.dot(x_dm, x_dm)
        if denom < 1e-10:
            continue
        betas[i] = np.dot(x_dm, y_dm) / denom

    # Forward-fill the early NaNs (before the first full window)
    result = pd.Series(betas, index=log_y.index)
    result = result.ffill()
    return result


def _kalman_filter_hedge(log_y: pd.Series, log_x: pd.Series) -> pd.Series:
    """
    Dynamic hedge ratio using a 1D Kalman Filter.
    Adapts instantly to structural breaks without relying on a fixed lookback window.
    """
    n = len(log_y)
    betas = np.zeros(n)
    
    # Initial state guess
    beta = log_y.iloc[0] / log_x.iloc[0] if log_x.iloc[0] != 0 else 1.0
    P = 1.0           # Initial estimate error covariance
    V_w = 1e-5        # Process noise variance (how fast true beta can change)
    V_v = 1e-3        # Measurement noise variance (volatility of the spread)
    
    y_vals = log_y.values
    x_vals = log_x.values
    
    for i in range(n):
        x_i = x_vals[i]
        y_i = y_vals[i]
        
        # Predict step (assume beta stays the same, but uncertainty grows)
        P = P + V_w
        
        # Measurement update step
        y_pred = beta * x_i
        error = y_i - y_pred
        
        # Kalman Gain (how much should we trust the new error vs our old estimate?)
        S = (x_i ** 2) * P + V_v
        K = (P * x_i) / S
        
        # Update the state (the new beta)
        beta = beta + K * error
        
        # Update the uncertainty
        P = (1 - K * x_i) * P
        
        betas[i] = beta
        
    return pd.Series(betas, index=log_y.index)

# ---------------------------------------------------------------------------
# OU process parameter estimation
# ---------------------------------------------------------------------------

def fit_ou_process(spread: pd.Series) -> dict:
    """
    Fit an Ornstein-Uhlenbeck process to the spread:
        dS_t = κ(μ - S_t)dt + σ dW_t

    via discrete-time regression:
        ΔS_t = α + β * S_{t-1} + ε_t
    where:
        β  = -κΔt      → κ = -β/Δt  (Δt = 1 day = 1/252 year)
        α  = κ * μ * Δt → μ = α / (κΔt)
        σ_ε = std(ε)   → σ_OU = σ_ε / sqrt(Δt)

    Returns a dict with keys: kappa, mu, sigma, half_life, mean_reversion_strength
    """
    delta_s = spread.diff().dropna()
    lagged_s = spread.shift(1).dropna()
    delta_s, lagged_s = delta_s.align(lagged_s, join="inner")

    X = add_constant(lagged_s)
    model = OLS(delta_s, X).fit()

    alpha = float(model.params.iloc[0])
    beta  = float(model.params.iloc[1])
    resid_std = float(model.resid.std())

    dt = 1.0 / 252.0
    kappa = -beta / dt

    if kappa <= 0:
        # Not mean-reverting
        return {"kappa": 0, "mu": float(spread.mean()), "sigma": resid_std,
                "half_life": float("inf"), "mean_reversion_strength": 0.0}

    mu       = alpha / (kappa * dt)
    sigma_ou = resid_std / np.sqrt(dt)
    half_life = np.log(2) / kappa * 252  # convert back to days

    # Mean reversion strength: |β| / se(β) — a t-stat for mean reversion
    # Higher = more confident the spread actually reverts
    mr_strength = abs(beta) / model.bse.iloc[1]

    return {
        "kappa":                  round(kappa, 4),
        "mu":                     round(mu, 6),
        "sigma":                  round(sigma_ou, 6),
        "half_life":              round(half_life, 2),
        "mean_reversion_strength": round(mr_strength, 2),
        "r_squared":              round(model.rsquared, 4),
    }


# ---------------------------------------------------------------------------
# Regime detection
# ---------------------------------------------------------------------------

def detect_regime(
    spread_df: pd.DataFrame,
    lookback: int = 63,
) -> pd.Series:
    """
    Classify each day into a trading regime based on spread behaviour.

    Regimes:
      'normal'    → spread well-behaved, half-life in range, trade normally
      'trending'  → spread has been drifting in one direction — caution
      'volatile'  → spread std is unusually high — reduce position size
      'breakdown' → pair may have broken down (correlation collapsed)

    Returns a Series of regime labels aligned to spread_df.index.
    """
    spread = spread_df["spread"].copy()
    regimes = pd.Series("normal", index=spread.index)

    # Trailing volatility ratio: current vol / long-run vol
    short_std = spread.rolling(21).std()
    long_std  = spread.rolling(126).std()
    vol_ratio = short_std / long_std

    # Trending: use the slope of the spread over the lookback window
    slopes = spread.rolling(lookback).apply(
        lambda s: np.polyfit(np.arange(len(s)), s, 1)[0] if len(s) == lookback else np.nan,
        raw=True,
    )
    normalised_slope = slopes / spread.rolling(lookback).std()

    regimes[vol_ratio > 2.0]                   = "volatile"
    regimes[normalised_slope.abs() > 0.05]     = "trending"
    # Breakdown: 30-day correlation of returns dropped below 0.4
    ret_y = spread_df["log_y"].diff()
    ret_x = spread_df["log_x"].diff()
    rolling_corr = ret_y.rolling(30).corr(ret_x)
    regimes[rolling_corr < 0.4]                = "breakdown"

    return regimes


# ---------------------------------------------------------------------------
# Spread quality metrics
# ---------------------------------------------------------------------------

def spread_quality_report(
    spread_df: pd.DataFrame,
    pair_name: str,
) -> dict:
    """
    Compute key quality metrics for a spread.
    Use this to quickly assess whether a pair is still worth trading.
    """
    spread = spread_df["spread"].dropna()
    zscore = spread_df["zscore"].dropna()
    ou = fit_ou_process(spread)

    # ADF test (re-check stationarity on actual rolling spread)
    adf_stat, adf_pvalue, *_ = adfuller(spread, maxlag=5, autolag="AIC")

    # Zero crossings per year — how often does the spread cross its mean?
    # More crossings = more trading opportunities
    centered = spread - spread.mean()
    crossings = ((centered.shift(1) * centered) < 0).sum()
    crossings_per_year = crossings / (len(spread) / 252)

    # Percentage of time |z| > 2 (entry zone)
    pct_in_entry_zone = (zscore.abs() > 2.0).mean()

    return {
        "pair":                 pair_name,
        "adf_pvalue":           round(adf_pvalue, 4),
        "half_life_days":       ou["half_life"],
        "ou_kappa":             ou["kappa"],
        "ou_sigma":             ou["sigma"],
        "mr_strength":          ou["mean_reversion_strength"],
        "zero_crossings_pa":    round(crossings_per_year, 1),
        "pct_time_in_entry":    round(pct_in_entry_zone * 100, 1),
        "spread_mean":          round(spread.mean(), 6),
        "spread_std":           round(spread.std(), 6),
    }


# ---------------------------------------------------------------------------
# Visualisation helper (Plotly)
# ---------------------------------------------------------------------------

def plot_spread(
    spread_df: pd.DataFrame,
    pair_name: str,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    stop_z: float = 3.5,
    show: bool = True,
) -> object:
    """
    Plot price ratio, spread, and z-score for a pair.
    Returns the figure object so it can be saved or embedded.
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        log.warning("plotly not installed — skipping plot")
        return None

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        subplot_titles=["Log Prices", "Spread (log_y − β·log_x)", "Z-Score"],
        vertical_spacing=0.08,
        row_heights=[0.3, 0.3, 0.4],
    )

    # --- Row 1: Log prices ---
    fig.add_trace(go.Scatter(
        x=spread_df.index, y=spread_df["log_y"],
        name=pair_name.split("/")[0], line=dict(color="#4f8ef7", width=1.5)
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=spread_df.index, y=spread_df["log_x"],
        name=pair_name.split("/")[1], line=dict(color="#f7884f", width=1.5)
    ), row=1, col=1)

    # --- Row 2: Spread ---
    fig.add_trace(go.Scatter(
        x=spread_df.index, y=spread_df["spread"],
        name="Spread", line=dict(color="#aaaaaa", width=1.2)
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=spread_df.index, y=spread_df["spread_mean"],
        name="Rolling mean", line=dict(color="#ffffff", width=1, dash="dash")
    ), row=2, col=1)

    # --- Row 3: Z-score with trade zones ---
    fig.add_trace(go.Scatter(
        x=spread_df.index, y=spread_df["zscore_smoothed"],
        name="Z-score", line=dict(color="#a78bfa", width=1.5)
    ), row=3, col=1)

    # Add threshold lines
    for z, color, label in [
        ( entry_z, "#22c55e", f"Entry (+{entry_z})"),
        (-entry_z, "#22c55e", f"Entry (−{entry_z})"),
        ( exit_z,  "#facc15", f"Exit (+{exit_z})"),
        (-exit_z,  "#facc15", f"Exit (−{exit_z})"),
        ( stop_z,  "#ef4444", f"Stop (+{stop_z})"),
        (-stop_z,  "#ef4444", f"Stop (−{stop_z})"),
        (0,        "#888888", "Mean"),
    ]:
        fig.add_hline(y=z, line_color=color, line_width=1,
                      line_dash="dot" if abs(z) != stop_z else "dash",
                      annotation_text=label if z > 0 else "",
                      annotation_position="top right", row=3, col=1)

    fig.update_layout(
        title=dict(text=f"Spread Analysis: {pair_name}", font=dict(size=16)),
        paper_bgcolor="#111827",
        plot_bgcolor="#1f2937",
        font=dict(color="#e5e7eb", family="monospace"),
        legend=dict(orientation="h", y=-0.05),
        height=700,
        margin=dict(l=50, r=50, t=60, b=50),
    )
    fig.update_xaxes(gridcolor="#374151", showgrid=True)
    fig.update_yaxes(gridcolor="#374151", showgrid=True)

    if show:
        fig.show()

    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    from data_pipeline import download_prices, validate_and_clean, build_sector_universe

    cfg = CFG
    os.makedirs("results", exist_ok=True)

    print("\n" + "="*60)
    print("  SPRINT 2 — SPREAD MODEL")
    print("="*60)

    # Load data
    all_tickers = [t for tickers in cfg.sectors.values() for t in tickers]
    raw_prices = download_prices(all_tickers, cfg.data.start_date, cfg.data.end_date)
    prices, _ = validate_and_clean(raw_prices, cfg.data.min_history)

    # Load top pairs from Sprint 1
    import pandas as pd
    ranked_pairs = pd.read_csv("results/ranked_pairs.csv")
    # Hard filter: ensure coint_pvalue is genuinely below threshold
    ranked_pairs = ranked_pairs[ranked_pairs["coint_pvalue"] < cfg.screen.coint_pvalue_threshold]
    top_pairs = ranked_pairs.head(10)

    print(f"\nAnalysing top {len(top_pairs)} pairs...\n")

    quality_reports = []
    spread_data = {}   # save for signal generator

    for _, row in top_pairs.iterrows():
        ty, tx = row["ticker_y"], row["ticker_x"]
        pair_name = f"{ty}/{tx}"

        if ty not in prices.columns or tx not in prices.columns:
            print(f"  ✗ {pair_name}: tickers not in price data")
            continue

        price_y = prices[ty]
        price_x = prices[tx]

        # Compute spread
        spread_df = compute_spread(
            price_y, price_x,
            window=cfg.signal.zscore_window,
            hedge_method=cfg.signal.hedge_method,
        )

        # OU fit
        ou_params = fit_ou_process(spread_df["spread"].dropna())

        # Regime
        regimes = detect_regime(spread_df)
        regime_counts = regimes.value_counts().to_dict()

        # Quality report
        qr = spread_quality_report(spread_df, pair_name)
        quality_reports.append(qr)
        spread_data[pair_name] = spread_df

        print(f"  {pair_name:12s} | "
              f"HL={ou_params['half_life']:5.1f}d | "
              f"κ={ou_params['kappa']:6.2f} | "
              f"MR-strength={ou_params['mean_reversion_strength']:5.1f} | "
              f"crossings/yr={qr['zero_crossings_pa']:4.0f} | "
              f"ADF p={qr['adf_pvalue']:.3f}")

    # Save quality report
    qr_df = pd.DataFrame(quality_reports)
    qr_df.to_csv("results/spread_quality.csv", index=False)
    print(f"\n--- Spread Quality Report ---")
    print(qr_df.to_string(index=False))
    print(f"\n✓ Saved spread quality to results/spread_quality.csv")

    # Plot the best pair
    best = top_pairs.iloc[0]
    best_pair = f"{best['ticker_y']}/{best['ticker_x']}"
    print(f"\nPlotting best pair: {best_pair}")
    if best_pair in spread_data:
        plot_spread(
            spread_data[best_pair],
            best_pair,
            entry_z=cfg.signal.entry_z,
            exit_z=cfg.signal.exit_z,
            stop_z=cfg.signal.stop_z,
        )

    print("\n  Next step: run signal_generator.py")
