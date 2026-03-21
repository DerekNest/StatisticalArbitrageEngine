"""
walk_forward.py — Out-of-sample validation via rolling walk-forward analysis.

CONCEPT: Why walk-forward validation?
  In-sample backtesting is almost always optimistic. The strategy was
  designed looking at the same data it's being tested on — even without
  deliberate overfitting, researcher bias leaks in.

  Walk-forward analysis mimics real deployment:

    |--- Train 1 ---|--- Test 1 ---|
                    |--- Train 2 ---|--- Test 2 ---|
                                   |--- Train 3 ---|--- Test 3 ---|

  We ONLY report performance on the TEST periods. The train periods are
  used to re-screen pairs and re-fit parameters. This is what separates
  a publishable backtest from a toy one.

CONCEPT: What we check in each fold
  1. Re-screen pairs on train window (fresh cointegration test)
  2. Re-fit OU parameters (half-life, hedge ratio)
  3. Generate signals on test window using train-period parameters
  4. Run backtester on test window
  5. Record metrics

  If the strategy is real, OOS Sharpe should be meaningfully positive
  across MOST folds — not just one lucky period.

CONCEPT: Interpreting walk-forward results
  Good signs:
    - Positive OOS Sharpe in 3+ of 4 folds
    - OOS Sharpe > 0.3 on average (lower than IS but consistently positive)
    - Similar pairs surviving screening in each fold

  Red flags:
    - OOS Sharpe strongly negative in most folds
    - IS Sharpe >> OOS Sharpe by a large factor (overfit)
    - Different pairs dominating in each fold (unstable universe)
"""

import logging
import warnings
from typing import List, Tuple

import numpy as np
import pandas as pd

from config import CFG, BacktestConfig
from data_pipeline import compute_returns
from pair_screener import screen_all_sectors, screen_pairs
from spread_model import compute_spread, detect_regime, fit_ou_process
from signal_generator import generate_signals, extract_trades, signal_stats
from backtester import PairsBacktester
from risk_manager import portfolio_risk_report

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Walk-forward split generator
# ---------------------------------------------------------------------------

def generate_wf_splits(
    index: pd.DatetimeIndex,
    train_years: int = 2,
    test_years: int = 1,
) -> List[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    """
    Generate rolling train/test splits from a date index.

    Anchored expanding window: train window slides forward by test_years
    each fold. Returns list of (train_dates, test_dates) tuples.
    """
    splits = []
    train_days = int(train_years * 252)
    test_days  = int(test_years  * 252)

    start = 0
    while True:
        train_end = start + train_days
        test_end  = train_end + test_days

        if test_end > len(index):
            break

        train_idx = index[start:train_end]
        test_idx  = index[train_end:test_end]
        splits.append((train_idx, test_idx))
        start += test_days   # slide forward by one test window

    log.info(f"Generated {len(splits)} walk-forward folds")
    for i, (tr, te) in enumerate(splits):
        log.info(f"  Fold {i+1}: train {tr[0].date()}→{tr[-1].date()} | "
                 f"test {te[0].date()}→{te[-1].date()}")
    return splits


# ---------------------------------------------------------------------------
# Single fold runner
# ---------------------------------------------------------------------------

def run_fold(
    fold_idx:    int,
    prices:      pd.DataFrame,
    train_idx:   pd.DatetimeIndex,
    test_idx:    pd.DatetimeIndex,
    cfg = None,
) -> dict:
    """
    Run one walk-forward fold:
      1. Screen pairs on TRAIN data
      2. Build signals using train-period hedge ratios on TEST data
      3. Backtest TEST period
      4. Return fold metrics
    """
    cfg = cfg or CFG
    train_prices = prices.loc[train_idx]
    test_prices  = prices.loc[test_idx]

    print(f"\n  {'─'*54}")
    print(f"  Fold {fold_idx+1} | "
          f"Train: {train_idx[0].date()}→{train_idx[-1].date()} | "
          f"Test: {test_idx[0].date()}→{test_idx[-1].date()}")
    print(f"  {'─'*54}")

    # ------------------------------------------------------------------
    # Step 1: Screen pairs on TRAIN window
    # ------------------------------------------------------------------
    # We pass ALL data up to the end of the train window (train_prices)
    # for cointegration tests. The Chow test uses the same train window
    # internally — within a walk-forward fold this is appropriate since
    # the "full history" available at that point is the train window.
    # Cointegration lookback = full train window (no subset).
    from data_pipeline import build_sector_universe
    train_universe = build_sector_universe(train_prices, cfg.sectors)

    ranked_pairs = screen_all_sectors(
        train_universe,
        cfg.screen,
        top_n_per_sector=8,   # keep more candidates at this stage since some may fail quality gate
        lookback_days=None,   # use full train window for cointegration
    )

    # Re-apply pvalue filter — guard against empty DataFrame from regime shifts
    if not ranked_pairs.empty:
        ranked_pairs = ranked_pairs[
            ranked_pairs["coint_pvalue"] < cfg.screen.coint_pvalue_threshold
        ].reset_index(drop=True)

    if ranked_pairs.empty:
        print(f"  Fold {fold_idx+1}: No pairs survived screening")
        return {"fold": fold_idx + 1,
                "train_start": str(train_idx[0].date()), "train_end": str(train_idx[-1].date()),
                "test_start": str(test_idx[0].date()), "test_end": str(test_idx[-1].date()),
                "n_pairs": 0, "n_trades": 0,
                "total_return_pct": 0.0, "ann_return_pct": 0.0, "ann_vol_pct": 0.0,
                "sharpe": np.nan, "sortino": np.nan, "calmar": np.nan,
                "max_dd_pct": 0.0, "win_days_pct": 0.0}

    print(f"  Screened {len(ranked_pairs)} pairs: {list(ranked_pairs['pair'])}")
    # After line 161 — print(f"  Screened {len(ranked_pairs)} pairs: ...")
    # Add this:
    if fold_idx == 3:  # fold 4 only
        print(f"  [FOLD 4 DEBUG] Pairs entering quality gate:")
        for _, r in ranked_pairs.iterrows():
            print(f"    {r['pair']:12s} coint_p={r['coint_pvalue']:.4f} hl={r['half_life']:.1f} hurst={r['hurst_exp']:.3f}")
    # ------------------------------------------------------------------
    # Step 2: Build signals on TEST window using TRAIN hedge ratios
    # ------------------------------------------------------------------
    # KEY: hedge ratio estimated on train, applied to test prices
    # This is what prevents look-ahead bias in the signal
    signal_data  = {}
    signal_stats_list = []
    spread_quality_list = []

    for _, row in ranked_pairs.iterrows():
        ty, tx = row["ticker_y"], row["ticker_x"]
        pair_name = f"{ty}/{tx}"

        if ty not in prices.columns or tx not in prices.columns:
            continue

        # Fit hedge ratio on TRAIN
        train_spread_df = compute_spread(
            train_prices[ty], train_prices[tx],
            window=cfg.signal.zscore_window,
            hedge_method=cfg.signal.hedge_method,
        )
        if train_spread_df.empty or train_spread_df["hedge_ratio"].isna().all():
            continue

        # Use final train-period hedge ratio for the test period
        fixed_hedge = float(train_spread_df["hedge_ratio"].dropna().iloc[-1])

        # Apply to TEST prices — but seed the rolling z-score with the
        # final `w` bars of the TRAIN window so the mean/std are already
        # calibrated on day 1 of the test period.
        #
        # Without seeding, the rolling window needs w bars to warm up,
        # so z-scores are noisy or NaN for the first ~40 test-period bars.
        # That kills trade frequency: a 252-bar test year loses ~16% of
        # its tradeable days before the signal is reliable.
        #
        # With seeding: we concatenate train_tail + test prices, compute
        # rolling stats on the full sequence, then slice back to test-only
        # rows. No look-ahead — train_tail prices are all in the past.
        w = cfg.signal.zscore_window
        seed_bars = w  # use exactly one window of train history as warmup

        train_tail_y = np.log(train_prices[ty].iloc[-seed_bars:])
        train_tail_x = np.log(train_prices[tx].iloc[-seed_bars:])
        test_log_y   = np.log(test_prices[ty])
        test_log_x   = np.log(test_prices[tx])

        combined_log_y = pd.concat([train_tail_y, test_log_y])
        combined_log_x = pd.concat([train_tail_x, test_log_x])
        combined_spread = combined_log_y - fixed_hedge * combined_log_x

        spread_mean_full = combined_spread.rolling(w, min_periods=w // 2).mean()
        spread_std_full  = (combined_spread.rolling(w, min_periods=w // 2)
                            .std().replace(0, np.nan))
        zscore_full      = (combined_spread - spread_mean_full) / spread_std_full
        zscore_sm_full   = zscore_full.ewm(span=3, adjust=False).mean()

        # Slice back to test-period rows only
        test_idx_set = set(test_prices.index)
        mask = combined_spread.index.isin(test_idx_set)

        test_spread   = combined_spread[mask]
        spread_mean   = spread_mean_full[mask]
        spread_std    = spread_std_full[mask]
        zscore        = zscore_full[mask]
        zscore_sm     = zscore_sm_full[mask]

        test_spread_df = pd.DataFrame({
            "log_y":           combined_log_y[mask],
            "log_x":           combined_log_x[mask],
            "hedge_ratio":     fixed_hedge,
            "spread":          test_spread,
            "spread_mean":     spread_mean,
            "spread_std":      spread_std,
            "zscore":          zscore,
            "zscore_smoothed": zscore_sm,
        }).dropna(subset=["zscore_smoothed"])

        if len(test_spread_df) < 20:
            continue
            # After the `if len(test_spread_df) < 20: continue` line (~line 230):
        # Detect regime on test period
        regimes = detect_regime(test_spread_df)

        # Generate signals
        sig_df = generate_signals(test_spread_df, cfg.signal, regimes)

        # Extract trades for signal stats
        trades_df = extract_trades(sig_df, pair_name, test_prices[ty], test_prices[tx])
        st = signal_stats(sig_df, trades_df, pair_name)

        signal_data[pair_name] = sig_df
        signal_stats_list.append(st)

        ou = fit_ou_process(test_spread_df["spread"].dropna())
        from statsmodels.tsa.stattools import adfuller
        adf_p = adfuller(test_spread_df["spread"].dropna(), maxlag=5, autolag="AIC")[1]
        spread_quality_list.append({
            "pair": pair_name,
            "adf_pvalue": adf_p,
            "half_life_days": ou["half_life"],
        })

    if not signal_data:
        print(f"  Fold {fold_idx+1}: No signals generated")
        return {"fold": fold_idx + 1,
                "train_start": str(train_idx[0].date()), "train_end": str(train_idx[-1].date()),
                "test_start": str(test_idx[0].date()), "test_end": str(test_idx[-1].date()),
                "n_pairs": 0, "n_trades": 0,
                "total_return_pct": 0.0, "ann_return_pct": 0.0, "ann_vol_pct": 0.0,
                "sharpe": np.nan, "sortino": np.nan, "calmar": np.nan,
                "max_dd_pct": 0.0, "win_days_pct": 0.0}

    signal_stats_df  = pd.DataFrame(signal_stats_list)
    spread_quality_df = pd.DataFrame(spread_quality_list)

    # ------------------------------------------------------------------
    # Step 3: Backtest the TEST window
    # ------------------------------------------------------------------
    bt = PairsBacktester(
        prices=test_prices,
        signal_data=signal_data,
        ranked_pairs=ranked_pairs,
        signal_stats=signal_stats_df,
        spread_quality=spread_quality_df,
        cfg=cfg.risk,
    )
    results = bt.run()

    if not results or results["equity_curve"].empty:
        return {"fold": fold_idx + 1,
                "train_start": str(train_idx[0].date()), "train_end": str(train_idx[-1].date()),
                "test_start": str(test_idx[0].date()), "test_end": str(test_idx[-1].date()),
                "n_pairs": 0, "n_trades": 0,
                "total_return_pct": 0.0, "ann_return_pct": 0.0, "ann_vol_pct": 0.0,
                "sharpe": np.nan, "sortino": np.nan, "calmar": np.nan,
                "max_dd_pct": 0.0, "win_days_pct": 0.0}

    m = results["metrics"]
    n_trades = len(results["trades"])

    print(f"  Result: return={m['total_return_pct']:+.2f}% | "
          f"Sharpe={m['sharpe_ratio']:.3f} | "
          f"MaxDD={m['max_drawdown_pct']:.2f}% | "
          f"trades={n_trades}")

    return {
        "fold":              fold_idx + 1,
        "train_start":       str(train_idx[0].date()),
        "train_end":         str(train_idx[-1].date()),
        "test_start":        str(test_idx[0].date()),
        "test_end":          str(test_idx[-1].date()),
        "n_pairs":           len(signal_data),
        "n_trades":          n_trades,
        "total_return_pct":  m["total_return_pct"],
        "ann_return_pct":    m["ann_return_pct"],
        "ann_vol_pct":       m["ann_vol_pct"],
        "sharpe":            m["sharpe_ratio"],
        "sortino":           m["sortino_ratio"],
        "calmar":            m["calmar_ratio"],
        "max_dd_pct":        m["max_drawdown_pct"],
        "win_days_pct":      m["win_days_pct"],
        "equity_curve":      results["equity_curve"],
        "trades":            results["trades"],
    }


# ---------------------------------------------------------------------------
# Full walk-forward runner
# ---------------------------------------------------------------------------

def run_walk_forward(
    prices:      pd.DataFrame,
    cfg = None,
) -> dict:
    """
    Run all walk-forward folds and aggregate results.
    """
    cfg = cfg or CFG

    splits = generate_wf_splits(
        prices.index,
        train_years=cfg.backtest.train_years,
        test_years=cfg.backtest.test_years,
    )

    if not splits:
        print("Not enough data for walk-forward splits.")
        print(f"Need at least {cfg.backtest.train_years + cfg.backtest.test_years} years.")
        return {}

    fold_results = []
    equity_curves = []

    for i, (train_idx, test_idx) in enumerate(splits):
        result = run_fold(i, prices, train_idx, test_idx, cfg)
        fold_results.append(result)

        if "equity_curve" in result:
            eq = result.pop("equity_curve")
            eq["fold"] = i + 1
            equity_curves.append(eq)

    summary_df = pd.DataFrame([
        {k: v for k, v in r.items() if k != "trades"} for r in fold_results
    ])

    # Stitch equity curves together for a continuous OOS equity curve
    combined_equity = None
    if equity_curves:
        combined_equity = _stitch_equity_curves(equity_curves)

    # Aggregate stats across folds
    valid = summary_df[summary_df["n_trades"] > 0]
    agg = {
        "n_folds":            len(summary_df),
        "n_folds_profitable": int((valid["total_return_pct"] > 0).sum()),
        "n_folds_positive_sharpe": int((valid["sharpe"] > 0).sum()),
        "mean_sharpe":        round(valid["sharpe"].mean(), 3) if not valid.empty else np.nan,
        "mean_return_pct":    round(valid["total_return_pct"].mean(), 2) if not valid.empty else 0,
        "mean_max_dd_pct":    round(valid["max_dd_pct"].mean(), 2) if not valid.empty else 0,
        "total_oos_trades":   int(valid["n_trades"].sum()),
    }

    return {
        "fold_summary":      summary_df,
        "aggregate":         agg,
        "combined_equity":   combined_equity,
        "fold_trades":       [r.get("trades", pd.DataFrame()) for r in fold_results],
    }


def _stitch_equity_curves(equity_curves: list) -> pd.Series:
    """
    Join per-fold equity curves into a single continuous OOS equity curve.
    Chain-links folds so each fold starts at the ending equity of the previous.
    Returns a pd.Series (not DataFrame) of equity values indexed by date.
    """
    pieces = []
    running_equity = CFG.risk.capital

    for eq_df in equity_curves:
        series = eq_df["equity"].copy()
        start_val = float(series.iloc[0])
        if start_val <= 0:
            continue
        # Scale fold so it starts at running_equity
        scaled = (series / start_val) * running_equity
        running_equity = float(scaled.iloc[-1])
        pieces.append(scaled)

    if not pieces:
        return pd.Series(dtype=float)

    # Concatenate in fold order — do NOT sort_index as that scrambles
    # the chain-linking. Folds are already in chronological order.
    combined = pd.concat(pieces)
    # Drop exact duplicate timestamps only (keep last)
    combined = combined[~combined.index.duplicated(keep="last")]
    return combined


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    from data_pipeline import download_prices, validate_and_clean

    cfg = CFG
    os.makedirs("results", exist_ok=True)

    print("\n" + "="*60)
    print("  SPRINT 4 — WALK-FORWARD VALIDATION")
    print("="*60)

    # Load data — use extended window for meaningful WF splits
    all_tickers = [t for tickers in cfg.sectors.values() for t in tickers]
    raw_prices  = download_prices(all_tickers, cfg.data.start_date, cfg.data.end_date)
    prices, _   = validate_and_clean(raw_prices, cfg.data.min_history)

    wf_results = run_walk_forward(prices, cfg)

    if not wf_results:
        print("\nNot enough data for walk-forward. Increase date range in config.py:")
        print("  start_date = '2018-01-01'")
        print("  end_date   = '2024-01-01'")
    else:
        summary = wf_results["fold_summary"]
        agg     = wf_results["aggregate"]

        print(f"\n{'='*60}")
        print(f"  WALK-FORWARD SUMMARY ({agg['n_folds']} folds)")
        print(f"{'='*60}")
        print(f"  Profitable folds:     {agg['n_folds_profitable']} / {agg['n_folds']}")
        print(f"  Positive Sharpe folds:{agg['n_folds_positive_sharpe']} / {agg['n_folds']}")
        print(f"  Mean OOS Sharpe:      {agg['mean_sharpe']:.3f}")
        print(f"  Mean OOS return:      {agg['mean_return_pct']:+.2f}%")
        print(f"  Mean max drawdown:    {agg['mean_max_dd_pct']:.2f}%")
        print(f"  Total OOS trades:     {agg['total_oos_trades']}")
        print(f"{'='*60}")

        print(f"\n--- Per-Fold Results (OOS only) ---")
        display_cols = ["fold", "test_start", "test_end", "n_pairs",
                        "n_trades", "total_return_pct", "sharpe", "max_dd_pct"]
        print(summary[display_cols].to_string(index=False))

        summary.to_csv("results/wf_fold_summary.csv", index=False)
        if wf_results["combined_equity"] is not None:
            wf_results["combined_equity"].to_csv("results/wf_equity_curve.csv")

        print(f"\n✓ Walk-forward results saved to results/")
        print(f"  Next step: run dashboard.py")