"""
signal_generator.py — Convert z-scores into trade signals and track positions.

CONCEPT: The classic pairs trade
  When z > +entry_z (spread too wide, Y expensive vs X):
      SHORT Y, LONG X  →  bet the spread narrows

  When z < -entry_z (spread too wide the other way, Y cheap vs X):
      LONG Y, SHORT X  →  bet the spread widens back

  Exit when |z| < exit_z (spread has normalised)
  Stop-loss when |z| > stop_z (spread is breaking down, get out)

CONCEPT: Dollar-neutral positioning
  We want the trade to be neutral to overall market moves (beta-neutral).
  For every $1 short in Y, we go long $β in X. The hedge ratio β ensures
  that a 1% market move affects both legs equally, leaving only the
  spread move as our P&L driver.

CONCEPT: Signal states
  A pair can be in one of four states:
    FLAT     → no position
    LONG     → long spread (long Y, short X)
    SHORT    → short spread (short Y, long X)
    CLOSING  → position opened, waiting for exit signal

  We track state per-pair per-day. This prevents double-entry and ensures
  we only exit when the spread has actually reverted.

CONCEPT: Look-ahead bias
  This is the #1 mistake in backtesting. We NEVER use today's signal to
  take today's price. Signal is computed at close of day t, trade executes
  at open (approximately close) of day t+1. We enforce this with a 1-day
  shift throughout.
"""

import logging
import warnings
from enum import Enum
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from config import CFG, SignalConfig

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------

class Position(Enum):
    FLAT    = 0
    LONG    = 1   # long spread: long Y, short X
    SHORT   = -1  # short spread: short Y, long X


# ---------------------------------------------------------------------------
# Signal generation
# ---------------------------------------------------------------------------

def generate_signals(
    spread_df: pd.DataFrame,
    cfg: SignalConfig = None,
    regime_series: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    Generate entry/exit signals from the z-score series.

    Logic:
      - Entry LONG spread:  z crosses below -entry_z  (spread too low)
      - Entry SHORT spread: z crosses above +entry_z  (spread too high)
      - Exit:               |z| drops below exit_z    (spread normalised)
      - Stop-loss:          |z| exceeds stop_z        (relationship breaking)

    We enforce a minimum holding period to avoid whipsawing in and out
    on consecutive days (transaction costs would kill us).

    Returns spread_df augmented with:
        signal          : +1 (long), -1 (short), 0 (flat)
        signal_shifted  : signal shifted 1 day (no look-ahead!)
        position        : current active position (+1/-1/0)
        trade_open      : True on the day we enter
        trade_close     : True on the day we exit
        bars_held       : consecutive bars in current position
    """
    cfg = cfg or CFG.signal
    df = spread_df.copy()
    z = df["zscore_smoothed"].fillna(0)

    n = len(df)
    position   = np.zeros(n, dtype=int)
    trade_open  = np.zeros(n, dtype=bool)
    trade_close = np.zeros(n, dtype=bool)
    bars_held  = np.zeros(n, dtype=int)

    # Minimum holding period: half the half-life (don't close before the spread
    # has had a chance to revert even partially).
    # Guard: rolling(60).std().mean() returns NaN when fewer than 60 valid
    # zscore bars exist (e.g. short forward windows after a rescreen).
    # Fall back to 3 bars in that case.
    _raw_min_hold = spread_df["zscore"].rolling(60).std().mean()
    min_hold = 3 if (pd.isna(_raw_min_hold) or _raw_min_hold == 0) \
               else max(3, int(_raw_min_hold * 2))

    current_pos = 0
    hold_count  = 0

    for i in range(1, n):
        z_prev = z.iloc[i - 1]  # use PREVIOUS day's z (no look-ahead)
        regime = regime_series.iloc[i] if regime_series is not None else "normal"

        # Don't trade in breakdown regime — pair relationship unreliable
        if regime == "breakdown":
            if current_pos != 0:
                position[i]    = 0
                trade_close[i] = True
                current_pos = 0
                hold_count  = 0
            continue

        # Position sizing modifier for volatile regime
        vol_mult = 0.5 if regime == "volatile" else 1.0

        if current_pos == 0:
            # --- ENTRY LOGIC ---
            if z_prev <= -cfg.entry_z:
                # Spread too low: LONG spread (long Y, short X)
                current_pos    = int(1 * vol_mult) or 1  # at least 1 if vol regime
                trade_open[i]  = True
                hold_count     = 1
            elif z_prev >= cfg.entry_z:
                # Spread too high: SHORT spread (short Y, long X)
                current_pos    = int(-1 * vol_mult) or -1
                trade_open[i]  = True
                hold_count     = 1

        else:
            # --- EXIT / STOP LOGIC ---
            hold_count += 1

            # 1. Stop-loss: spread moving further against us (Structural Break)
            if abs(z_prev) >= cfg.stop_z:
                position[i]    = 0
                trade_close[i] = True
                current_pos = 0
                hold_count  = 0
                continue

            # 2. NEW: Time-based stop (Zombie Trade)
            # Exit if the trade persists too long (e.g., 45 trading days)
            # Replace 45 with half_life * 2 for a dynamic approach
            if hold_count > 45:
                position[i]    = 0
                trade_close[i] = True
                current_pos = 0
                hold_count  = 0
                continue

            # 3. Normal exit: spread reverted sufficiently
            if hold_count >= min_hold and abs(z_prev) <= cfg.exit_z:
                position[i]    = 0
                trade_close[i] = True
                current_pos = 0
                hold_count  = 0
                continue

        position[i]   = current_pos
        bars_held[i]  = hold_count

    df["position"]    = position
    df["trade_open"]  = trade_open
    df["trade_close"] = trade_close
    df["bars_held"]   = bars_held

    # Signal is the INTENDED position for the next day
    # (what we'd put on tomorrow based on today's z-score)
    df["signal"] = df["position"].shift(1).fillna(0).astype(int)

    return df


# ---------------------------------------------------------------------------
# Trade log
# ---------------------------------------------------------------------------

def extract_trades(
    signal_df: pd.DataFrame,
    pair_name: str,
    price_y: pd.Series,
    price_x: pd.Series,
) -> pd.DataFrame:
    """
    Extract individual trade records from the signal series.

    Each trade is a round-trip: entry → exit. We compute:
      - Entry / exit dates and z-scores
      - Raw spread P&L (in log-spread units, before costs)
      - Duration in calendar days
      - Whether it was a winning or losing trade
    """
    df = signal_df.copy()
    df["price_y"] = price_y
    df["price_x"] = price_x

    trades = []
    in_trade = False
    entry_date = entry_z = entry_spread = entry_pos = None

    for date, row in df.iterrows():
        if not in_trade and row["trade_open"]:
            in_trade    = True
            entry_date  = date
            entry_z     = row["zscore_smoothed"]
            entry_spread = row["spread"]
            entry_pos   = row["position"]

        elif in_trade and row["trade_close"]:
            exit_date  = date
            exit_z     = row["zscore_smoothed"]
            exit_spread = row["spread"]

            # P&L: we're long/short the spread, so:
            # LONG spread (pos=+1):  profit if spread rose     → exit - entry
            # SHORT spread (pos=-1): profit if spread fell     → entry - exit
            spread_pnl = entry_pos * (exit_spread - entry_spread)

            duration = (exit_date - entry_date).days

            trades.append({
                "pair":        pair_name,
                "direction":   "LONG" if entry_pos > 0 else "SHORT",
                "entry_date":  entry_date.date(),
                "exit_date":   exit_date.date(),
                "entry_z":     round(entry_z, 3),
                "exit_z":      round(exit_z, 3),
                "spread_pnl":  round(spread_pnl, 5),
                "duration_d":  duration,
                "stop_hit":    abs(exit_z) >= CFG.signal.stop_z,
            })

            in_trade = False

    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        trades_df["win"] = trades_df["spread_pnl"] > 0
    return trades_df


# ---------------------------------------------------------------------------
# Signal summary statistics
# ---------------------------------------------------------------------------

def signal_stats(signal_df: pd.DataFrame, trades_df: pd.DataFrame, pair_name: str) -> dict:
    """
    Summary statistics for the signal — sanity-check before backtesting.
    """
    if trades_df.empty:
        # Return full zero-filled dict so downstream code never hits KeyError
        return {
            "pair": pair_name, "n_trades": 0, "win_rate": 0.0,
            "avg_win": 0.0, "avg_loss": 0.0, "profit_factor": 0.0,
            "avg_duration_d": 0.0, "n_stops_hit": 0, "pct_in_market": 0.0,
        }

    win_rate = trades_df["win"].mean()
    avg_win  = trades_df.loc[trades_df["win"],  "spread_pnl"].mean()
    avg_loss = trades_df.loc[~trades_df["win"], "spread_pnl"].mean()
    profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")

    days_in_market = (signal_df["position"] != 0).sum()
    pct_in_market  = days_in_market / len(signal_df)

    return {
        "pair":             pair_name,
        "n_trades":         len(trades_df),
        "win_rate":         round(win_rate * 100, 1),
        "avg_win":          round(avg_win, 5),
        "avg_loss":         round(avg_loss, 5),
        "profit_factor":    round(profit_factor, 2),
        "avg_duration_d":   round(trades_df["duration_d"].mean(), 1),
        "n_stops_hit":      int(trades_df["stop_hit"].sum()),
        "pct_in_market":    round(pct_in_market * 100, 1),
    }


# ---------------------------------------------------------------------------
# Multi-pair signal runner
# ---------------------------------------------------------------------------

def run_all_signals(
    prices: pd.DataFrame,
    ranked_pairs: pd.DataFrame,
    cfg: SignalConfig = None,
    top_n: int = 10,
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    """
    Run signal generation for the top N pairs.

    Returns:
        signal_data  : dict of {pair_name → signal_df}
        all_trades   : combined trade log DataFrame
        stats_df     : summary statistics per pair
    """
    from spread_model import compute_spread, detect_regime

    cfg = cfg or CFG.signal
    pairs = ranked_pairs[
        ranked_pairs["coint_pvalue"] < CFG.screen.coint_pvalue_threshold
    ].head(top_n)

    signal_data = {}
    all_trades  = []
    stats       = []

    for _, row in pairs.iterrows():
        ty, tx = row["ticker_y"], row["ticker_x"]
        pair_name = f"{ty}/{tx}"

        if ty not in prices.columns or tx not in prices.columns:
            continue

        # Build spread
        spread_df = compute_spread(
            prices[ty], prices[tx],
            window=CFG.signal.zscore_window,
            hedge_method=CFG.signal.hedge_method,
        )
        spread_df = spread_df.dropna(subset=["zscore_smoothed"])

        # Detect regime
        regimes = detect_regime(spread_df)

        # Generate signals
        sig_df = generate_signals(spread_df, cfg, regimes)

        # Extract trades
        trades_df = extract_trades(sig_df, pair_name, prices[ty], prices[tx])

        # Stats
        st = signal_stats(sig_df, trades_df, pair_name)
        stats.append(st)
        signal_data[pair_name] = sig_df

        if not trades_df.empty:
            all_trades.append(trades_df)

        print(f"  {pair_name:12s} | "
              f"trades={st['n_trades']:3d} | "
              f"win%={st['win_rate']:5.1f} | "
              f"PF={st['profit_factor']:5.2f} | "
              f"in-mkt={st['pct_in_market']:4.1f}%")

    all_trades_df = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    stats_df = pd.DataFrame(stats)
    return signal_data, all_trades_df, stats_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    from data_pipeline import download_prices, validate_and_clean

    cfg = CFG
    os.makedirs("results", exist_ok=True)

    print("\n" + "="*60)
    print("  SPRINT 2 — SIGNAL GENERATOR")
    print("="*60)

    # Load data + pairs
    all_tickers = [t for tickers in cfg.sectors.values() for t in tickers]
    raw_prices  = download_prices(all_tickers, cfg.data.start_date, cfg.data.end_date)
    prices, _   = validate_and_clean(raw_prices, cfg.data.min_history)
    ranked_pairs = pd.read_csv("results/ranked_pairs.csv")

    print("\nGenerating signals for top 10 pairs...\n")
    signal_data, all_trades, stats_df = run_all_signals(
        prices, ranked_pairs, cfg.signal, top_n=10
    )

    print(f"\n--- Signal Summary ---")
    print(stats_df.to_string(index=False))

    all_trades.to_csv("results/trade_log.csv", index=False)
    stats_df.to_csv("results/signal_stats.csv", index=False)
    print(f"\n✓ {len(all_trades)} trades logged to results/trade_log.csv")
    print(f"  Next step: run backtester.py (Sprint 3)")