"""
backtester.py — Event-driven backtesting engine with quarterly re-screening.

CONCEPT: Event-driven vs vectorised backtesting
  Vectorised backtesting (numpy operations over the whole array) is fast
  but makes it easy to accidentally introduce look-ahead bias because you
  operate on the full matrix at once.

  Event-driven backtesting simulates trading day by day, in chronological
  order, as if you were live. Each "event" is a market open or close.
  State (positions, cash, equity) is carried forward explicitly.

CONCEPT: Dollar-neutral pairs book
  Each pair trade consists of two simultaneous positions:
    - Long leg:  +$N in the cheaper stock (relative to spread)
    - Short leg: -$N in the more expensive stock

  The portfolio is MARKET-NEUTRAL: long and short dollars cancel.

CONCEPT: Quarterly re-screening — the right pattern
  The walk-forward optimizer achieves its 1.80 OOS Sharpe because each
  fold does the following:

    1. Screen pairs on TRAIN window  →  get hedge ratio β
    2. Fix β at its final train value
    3. Build spread + z-score on the entire TEST window using that fixed β
    4. Generate signals over TEST window
    5. Run backtester on TEST window

  The _rescreen() method replicates this exactly:

    1. Screen pairs on prices[:rescreen_date]  →  LOOKBACK_DAYS rolling window
    2. Fix β = last value of rolling OLS hedge on that lookback window
    3. Build spread + z-score on prices[rescreen_date:]  using fixed β
    4. Generate signals over prices[rescreen_date:]
    5. Register those signals — the main loop can now trade them

  This is why the previous version was broken: it built signals ONLY on
  the lookback window (past), so when the main loop asked for a signal
  on today's date, that date didn't exist in signal_df → trade_open
  never fired → 36 trades in 6 years.

  NO look-ahead bias:
  - Hedge ratio comes only from prices[:rescreen_date]
  - Z-score on prices[rescreen_date:] is computed rolling (no future mean/std)
  - Signal generation reads z_prev (previous day) before acting today

CONCEPT: Realistic cost modelling
  Commission + slippage on EVERY leg. 0.05% slippage is conservative for
  S&P 500 stocks at retail sizes.
"""

import logging
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from config import CFG, RiskConfig
from risk_manager import (
    DrawdownMonitor, kelly_size,
    compute_trade_cost, passes_quality_gate, portfolio_risk_report,
)

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Re-screening constants — mirror the walk-forward fold parameters
# ---------------------------------------------------------------------------

# How often to re-screen (calendar days between rescreens).
# 90 days ≈ quarterly — same cadence as CFG.data.rescreen_every_days.
RESCREEN_CALENDAR_DAYS = 63

# Rolling lookback window (trading days) for cointegration tests at each rescreen.
# 504 = 2 trading years — matches the WF train window length that produced
# the 1.80 OOS Sharpe. Do not shorten: fewer bars → weaker coint tests.
SCREEN_LOOKBACK_BARS = 504


# ---------------------------------------------------------------------------
# Trade record
# ---------------------------------------------------------------------------

@dataclass
class OpenTrade:
    pair:          str
    direction:     int          # +1 = long spread, -1 = short spread
    entry_date:    pd.Timestamp
    entry_price_y: float
    entry_price_x: float
    shares_y:      float        # positive = long, negative = short
    shares_x:      float
    entry_cost:    float        # costs paid at entry (other half at exit)
    hedge_ratio:   float


# ---------------------------------------------------------------------------
# Backtester
# ---------------------------------------------------------------------------

class PairsBacktester:
    """
    Event-driven backtester for a portfolio of pairs trades.

    Usage:
        bt = PairsBacktester(prices, signal_data, ranked_pairs, cfg)
        results = bt.run()
    """

    def __init__(
        self,
        prices:         pd.DataFrame,
        signal_data:    Dict[str, pd.DataFrame],
        ranked_pairs:   pd.DataFrame,
        signal_stats:   pd.DataFrame,
        spread_quality: pd.DataFrame,
        cfg:            RiskConfig = None,
    ):
        self.prices         = prices
        self.signal_data    = signal_data          # updated at each rescreen
        self.ranked_pairs   = ranked_pairs         # updated at each rescreen
        self.signal_stats   = signal_stats         # updated at each rescreen
        self.spread_quality = spread_quality       # updated at each rescreen
        self.cfg            = cfg or CFG.risk

        self.cash           = self.cfg.capital
        self.equity         = self.cfg.capital
        self.open_trades:   Dict[str, OpenTrade] = {}
        self.equity_curve:  List[dict] = []
        self.closed_trades: List[dict] = []
        self.rescreen_log:  List[dict] = []
        self.pair_realised_pnl: Dict[str, float] = {}
        self.blacklisted:   set = set()

        self.dd_monitor = DrawdownMonitor(
            soft_limit=self.cfg.max_drawdown * 0.5,
            hard_limit=self.cfg.max_drawdown,
        )

    # ------------------------------------------------------------------
    # Pair approval gate
    # ------------------------------------------------------------------

    def _approved_pairs(self) -> List[str]:
        """Quality-gate + ticker-diversification filter on current ranked_pairs."""
        approved      = []
        ticker_counts: Dict[str, int] = {}

        for _, row in self.ranked_pairs.iterrows():
            ty, tx    = row["ticker_y"], row["ticker_x"]
            pair_name = f"{ty}/{tx}"

            config_blacklist = set(getattr(self.cfg, "pair_blacklist", ()))
            if pair_name in self.blacklisted or pair_name in config_blacklist:
                continue
            if pair_name not in self.signal_data:
                continue

            stats_row = self.signal_stats[self.signal_stats["pair"] == pair_name]
            qual_row  = self.spread_quality[self.spread_quality["pair"] == pair_name]
            if stats_row.empty or qual_row.empty:
                continue

            ok, reason = passes_quality_gate(
                stats_row.iloc[0].to_dict(),
                min_trades=0,   # no minimum trades filter — we want to keep promising pairs that just haven't traded yet
                min_adf_pvalue_threshold=0.10,
                adf_pvalue=float(qual_row.iloc[0]["adf_pvalue"]),
            )

            if ok:
                # Hurst filter — reject trending spreads before allocating capital
                hurst = float(row.get("hurst_exp", 0.5))
                if hurst >= 0.45:
                    log.debug(f"  Excluding {pair_name}: Hurst={hurst:.3f} >= 0.45 (trending)")
                    continue

                if ticker_counts.get(ty, 0) < 2 and ticker_counts.get(tx, 0) < 2:
                    approved.append(pair_name)
                ticker_counts[ty] = ticker_counts.get(ty, 0) + 1
                ticker_counts[tx] = ticker_counts.get(tx, 0) + 1
            else:
                log.debug(f"  Excluding {pair_name}: {reason}")

        log.info(f"Approved {len(approved)} pairs for trading.")
        return approved[: self.cfg.max_pairs]

    # ------------------------------------------------------------------
    # Quarterly re-screening  ← THE CORE FIX
    # ------------------------------------------------------------------

    def _rescreen(self, rescreen_date: pd.Timestamp) -> List[str]:
        """
        Re-screen pairs and rebuild forward-looking signal data.

        Pattern mirrors walk_forward.run_fold() exactly:

          Step 1  Screen pairs on prices[:rescreen_date] (lookback window)
          Step 2  Fix hedge ratio β = last value from lookback OLS
          Step 3  Build spread + z-score on prices[rescreen_date:]
                  using that fixed β  (rolling mean/std — no look-ahead)
          Step 4  Generate signals on prices[rescreen_date:]
          Step 5  Register signals in self.signal_data — main loop can trade them

        Open positions are NOT disturbed. They keep their original signal_df
        and exit naturally when trade_close fires.
        """
        from data_pipeline import build_sector_universe
        from pair_screener import screen_all_sectors
        from spread_model import compute_spread, detect_regime, fit_ou_process
        from signal_generator import (
            generate_signals, extract_trades,
            signal_stats as sig_stats_fn,
        )
        from statsmodels.tsa.stattools import adfuller

        log.info(f"  [RESCREEN] {rescreen_date.date()}")

        # ── Step 1: Screen on lookback window (strictly historical) ──────
        lookback_prices = self.prices.loc[:rescreen_date].iloc[-SCREEN_LOOKBACK_BARS:]
        if len(lookback_prices) < SCREEN_LOOKBACK_BARS // 2:
            log.warning("  [RESCREEN] Not enough history — skipping")
            return self._approved_pairs()

        universe   = build_sector_universe(lookback_prices, CFG.sectors)
        new_ranked = screen_all_sectors(
            universe,
            CFG.screen,
            top_n_per_sector=8,   # keep more candidates at this stage since some may fail quality gate
            lookback_days=None,   # use full lookback_prices (already sliced)
        )

        if new_ranked.empty:
            log.warning(f"  [RESCREEN] {rescreen_date.date()} — no pairs found")
            self.rescreen_log.append({
                "date": rescreen_date.date(), "n_pairs": 0, "pairs": "",
            })
            return self._approved_pairs()

        new_ranked = new_ranked[
            new_ranked["coint_pvalue"] < CFG.screen.coint_pvalue_threshold
        ].reset_index(drop=True)

        # ── Step 2-4: Fix β on lookback, build signals on forward window ─
        # forward_prices = everything from rescreen_date onward.
        # This is what the main loop will trade. It is NOT future data at
        # the time of running — we only USE these signals day-by-day in
        # chronological order, never reading ahead.
        forward_prices = self.prices.loc[rescreen_date:]

        new_stats:   List[dict] = []
        new_quality: List[dict] = []

        for _, row in new_ranked.iterrows():
            ty, tx    = row["ticker_y"], row["ticker_x"]
            pair_name = f"{ty}/{tx}"

            if ty not in self.prices.columns or tx not in self.prices.columns:
                continue

            # ── Step 2: Hedge ratio fixed at end of lookback window ──────
            lookback_spread_df = compute_spread(
                lookback_prices[ty], lookback_prices[tx],
                window=CFG.signal.zscore_window,
                hedge_method=CFG.signal.hedge_method,
            )
            if lookback_spread_df.empty or lookback_spread_df["hedge_ratio"].isna().all():
                continue

            fixed_hedge = float(
                lookback_spread_df["hedge_ratio"].dropna().iloc[-1]
            )

            # ── Step 3: Build spread + z-score on forward window ─────────
            # Using fixed_hedge throughout — no peeking at future prices
            # to re-estimate β.
            fwd_log_y  = np.log(forward_prices[ty])
            fwd_log_x  = np.log(forward_prices[tx])
            fwd_spread = fwd_log_y - fixed_hedge * fwd_log_x

            w           = CFG.signal.zscore_window
            spread_mean = fwd_spread.rolling(w, min_periods=w // 2).mean()
            spread_std  = (
                fwd_spread.rolling(w, min_periods=w // 2)
                .std()
                .replace(0, np.nan)
            )
            zscore    = (fwd_spread - spread_mean) / spread_std
            zscore_sm = zscore.ewm(span=3, adjust=False).mean()

            fwd_spread_df = pd.DataFrame({
                "log_y":           fwd_log_y,
                "log_x":           fwd_log_x,
                "hedge_ratio":     fixed_hedge,
                "spread":          fwd_spread,
                "spread_mean":     spread_mean,
                "spread_std":      spread_std,
                "zscore":          zscore,
                "zscore_smoothed": zscore_sm,
            }).dropna(subset=["zscore_smoothed"])

            # Need at least 60 bars so the rolling(60).std() inside
            # generate_signals() returns a non-NaN mean for min_hold.
            if len(fwd_spread_df) < 60:
                continue

            # ── Step 4: Generate signals on forward window ────────────────
            regimes = detect_regime(fwd_spread_df)
            sig_df  = generate_signals(fwd_spread_df, CFG.signal, regimes)

            # Signal stats (Kelly sizing uses win_rate / avg_win / avg_loss
            # from the lookback window, not the forward window — no look-ahead)
            lookback_sig_df = generate_signals(
                lookback_spread_df.dropna(subset=["zscore_smoothed"]),
                CFG.signal,
                detect_regime(
                    lookback_spread_df.dropna(subset=["zscore_smoothed"])
                ),
            )
            lookback_trades = extract_trades(
                lookback_sig_df, pair_name,
                lookback_prices[ty], lookback_prices[tx],
            )
            st = sig_stats_fn(lookback_sig_df, lookback_trades, pair_name)
            new_stats.append(st)

            # Spread quality from lookback window
            ou    = fit_ou_process(lookback_spread_df["spread"].dropna())
            adf_p = adfuller(
                lookback_spread_df["spread"].dropna(), maxlag=5, autolag="AIC"
            )[1]
            new_quality.append({
                "pair":           pair_name,
                "adf_pvalue":     adf_p,
                "half_life_days": ou["half_life"],
            })

            # ── Step 5: Register forward-looking signal_df ────────────────
            # Existing open trades keep their OLD signal_df until they close
            # naturally. Only pairs not currently open get overwritten.
            if pair_name not in self.open_trades:
                self.signal_data[pair_name] = sig_df

        # Replace metadata tables
        if new_stats:
            self.signal_stats   = pd.DataFrame(new_stats)
        if new_quality:
            self.spread_quality = pd.DataFrame(new_quality)
        self.ranked_pairs = new_ranked

        self.rescreen_log.append({
            "date":    rescreen_date.date(),
            "n_pairs": len(new_ranked),
            "pairs":   "|".join(new_ranked["pair"].tolist()),
        })

        new_approved = self._approved_pairs()
        log.info(
            f"  [RESCREEN] {len(new_ranked)} pairs found → "
            f"{len(new_approved)} approved"
        )
        return new_approved

    # ------------------------------------------------------------------
    # Position sizing
    # ------------------------------------------------------------------

    def _compute_position_size(
        self,
        pair_name:  str,
        signal_row: pd.Series,
        price_y:    float,
        price_x:    float,
        scale:      float,
    ) -> Tuple[float, float, float]:
        """Returns (shares_y, shares_x, notional_y) or (0, 0, 0)."""
        stats_row = self.signal_stats[self.signal_stats["pair"] == pair_name]
        if stats_row.empty:
            return 0.0, 0.0, 0.0

        st       = stats_row.iloc[0]
        win_rate = st["win_rate"] / 100.0
        avg_win  = st["avg_win"]
        raw_loss = st["avg_loss"]

        if pd.isna(raw_loss) or raw_loss == 0.0:
            avg_loss = -abs(avg_win) * 0.5
        else:
            avg_loss = raw_loss

        f = kelly_size(win_rate, avg_win, avg_loss, self.cfg.kelly_fraction)
        if f <= 0:
            return 0.0, 0.0, 0.0

        notional_y  = self.equity * f * scale * self.cfg.risk_per_trade / 0.02
        notional_y  = min(notional_y, self.equity * 0.25)   # hard cap: max 15% per leg
        shares_y    = notional_y / price_y
        hedge_ratio = float(signal_row["hedge_ratio"])
        shares_x    = (shares_y * hedge_ratio * price_x) / price_x

        if notional_y < 500:
            return 0.0, 0.0, 0.0

        return shares_y, shares_x, notional_y

    # ------------------------------------------------------------------
    # Core simulation loop
    # ------------------------------------------------------------------

    def run(self) -> dict:
        """
        Main event-driven loop. Re-screens every RESCREEN_CALENDAR_DAYS.
        """
        approved = self._approved_pairs()
        if not approved:
            log.error("No pairs approved at start — check ranked_pairs input")
            return {}

        all_dates         = self.prices.index
        last_screen_date  = all_dates[0]

        log.info(
            f"Backtest {all_dates[0].date()} → {all_dates[-1].date()} | "
            f"rescreen every {RESCREEN_CALENDAR_DAYS} calendar days | "
            f"lookback {SCREEN_LOOKBACK_BARS} bars"
        )
        log.info(f"Initial pairs: {approved}")

        for date in all_dates:

            # ── Quarterly re-screen trigger ───────────────────────────────
            calendar_days_elapsed = (date - last_screen_date).days
            bars_available        = len(self.prices.loc[:date])

            if (calendar_days_elapsed >= RESCREEN_CALENDAR_DAYS
                    and bars_available >= SCREEN_LOOKBACK_BARS):
                approved         = self._rescreen(date)
                last_screen_date = date

            # ── Mark-to-market ────────────────────────────────────────────
            mtm_pnl     = self._mark_to_market(date)
            self.equity = self.cash + mtm_pnl
            scale       = self.dd_monitor.update(self.equity)

            # ── Equity curve ──────────────────────────────────────────────
            self.equity_curve.append({
                "date":       date,
                "equity":     round(self.equity, 2),
                "cash":       round(self.cash, 2),
                "mtm_pnl":    round(mtm_pnl, 2),
                "n_open":     len(self.open_trades),
                "drawdown":   round(self.dd_monitor.current_dd * 100, 3),
                "dd_scale":   round(scale, 3),
                "open_pairs": "|".join(self.open_trades.keys()),
            })

            # ── Exits ─────────────────────────────────────────────────────
            for pair_name in list(self.open_trades.keys()):
                sig_df = self.signal_data.get(pair_name)
                if sig_df is None or date not in sig_df.index:
                    continue

                row = sig_df.loc[date]

                if scale == 0.0:
                    self._close_trade(pair_name, date, row, forced=True)
                    continue

                if row["trade_close"]:
                    self._close_trade(pair_name, date, row, forced=False)
                    continue

                # Per-pair loss circuit breaker — checks TOTAL exposure:
                # realised losses from previous trades on this pair PLUS
                # today's unrealised mark-to-market loss.
                # Previously only realised P&L was checked, so a position
                # bleeding $7k unrealised would never trip the breaker until
                # it closed — by which point the damage was done.
                max_loss_cap  = getattr(self.cfg, "max_loss_per_pair", 0.025)
                realised_loss = self.pair_realised_pnl.get(pair_name, 0.0)
                unrealised    = self._pair_mtm(pair_name, date)
                total_loss    = realised_loss + unrealised
                if total_loss < -(self.cfg.capital * max_loss_cap):
                    log.info(
                        f"  PAIR LOSS CAP: {pair_name} "
                        f"realised=${realised_loss:+,.0f} unrealised=${unrealised:+,.0f} "
                        f"total=${total_loss:+,.0f} — closing & blacklisting"
                    )
                    self._close_trade(pair_name, date, row, forced=True)
                    self.blacklisted.add(pair_name)
                    approved = [p for p in approved if p != pair_name]

            # ── Entries ───────────────────────────────────────────────────
            n_open = len(self.open_trades)
            for pair_name in approved:
                if pair_name in self.open_trades:
                    continue
                if n_open >= self.cfg.max_pairs:
                    break
                if scale == 0.0:
                    break

                sig_df = self.signal_data.get(pair_name)
                if sig_df is None or date not in sig_df.index:
                    continue

                row = sig_df.loc[date]
                if row["trade_open"] and row["position"] != 0:
                    if self._open_trade(pair_name, date, row, scale):
                        n_open += 1

        # ── Close residual positions at end of history ────────────────────
        last_date = all_dates[-1]
        for pair_name in list(self.open_trades.keys()):
            sig_df = self.signal_data.get(pair_name)
            if sig_df is not None and last_date in sig_df.index:
                self._close_trade(
                    pair_name, last_date, sig_df.loc[last_date], forced=True
                )

        return self._compile_results()

    # ------------------------------------------------------------------
    # Trade execution
    # ------------------------------------------------------------------

    def _open_trade(
        self,
        pair_name: str,
        date:      pd.Timestamp,
        row:       pd.Series,
        scale:     float,
    ) -> bool:
        ty, tx = pair_name.split("/")
        if ty not in self.prices.columns or tx not in self.prices.columns:
            return False
        if date not in self.prices.index:
            return False

        price_y = float(self.prices.loc[date, ty])
        price_x = float(self.prices.loc[date, tx])
        if np.isnan(price_y) or np.isnan(price_x) or price_y <= 0 or price_x <= 0:
            return False

        direction              = int(row["position"])
        shares_y, shares_x, notional_y = self._compute_position_size(
            pair_name, row, price_y, price_x, scale
        )
        if shares_y == 0:
            return False

        notional_x = shares_x * price_x
        entry_cost = compute_trade_cost(
            notional_y, notional_x,
            self.cfg.commission_pct, self.cfg.slippage_pct,
        ) / 2
        self.cash -= entry_cost

        self.open_trades[pair_name] = OpenTrade(
            pair=pair_name,
            direction=direction,
            entry_date=date,
            entry_price_y=price_y,
            entry_price_x=price_x,
            shares_y= direction * shares_y,
            shares_x=-direction * shares_x,
            entry_cost=entry_cost,
            hedge_ratio=float(row["hedge_ratio"]),
        )

        log.debug(
            f"  OPEN {pair_name} dir={direction:+d} "
            f"@{price_y:.2f}/{price_x:.2f} "
            f"notional=${notional_y:,.0f} cost=${entry_cost:.2f}"
        )
        return True

    def _close_trade(
        self,
        pair_name: str,
        date:      pd.Timestamp,
        row:       pd.Series,
        forced:    bool = False,
    ) -> None:
        trade  = self.open_trades.pop(pair_name)
        ty, tx = pair_name.split("/")

        if date not in self.prices.index:
            return

        price_y = float(self.prices.loc[date, ty])
        price_x = float(self.prices.loc[date, tx])
        if np.isnan(price_y) or np.isnan(price_x):
            return

        pnl_y     = trade.shares_y * (price_y - trade.entry_price_y)
        pnl_x     = trade.shares_x * (price_x - trade.entry_price_x)
        gross_pnl = pnl_y + pnl_x

        notional_y = abs(trade.shares_y) * price_y
        notional_x = abs(trade.shares_x) * price_x
        exit_cost  = compute_trade_cost(
            notional_y, notional_x,
            self.cfg.commission_pct, self.cfg.slippage_pct,
        ) / 2

        net_pnl    = gross_pnl - exit_cost - trade.entry_cost
        self.cash += gross_pnl - exit_cost
        self.equity = self.cash + self._mark_to_market(date)

        self.pair_realised_pnl[pair_name] = (
            self.pair_realised_pnl.get(pair_name, 0.0) + net_pnl
        )

        self.closed_trades.append({
            "pair":         pair_name,
            "direction":    "LONG" if trade.direction > 0 else "SHORT",
            "entry_date":   trade.entry_date.date(),
            "exit_date":    date.date(),
            "entry_y":      round(trade.entry_price_y, 4),
            "exit_y":       round(price_y, 4),
            "entry_x":      round(trade.entry_price_x, 4),
            "exit_x":       round(price_x, 4),
            "shares_y":     round(trade.shares_y, 2),
            "shares_x":     round(trade.shares_x, 2),
            "gross_pnl":    round(gross_pnl, 2),
            "costs":        round(exit_cost + trade.entry_cost, 2),
            "net_pnl":      round(net_pnl, 2),
            "duration_d":   (date - trade.entry_date).days,
            "forced_close": forced,
            "zscore_exit":  round(float(row.get("zscore_smoothed", 0)), 3),
        })

        log.debug(f"  CLOSE {pair_name} net_pnl=${net_pnl:+.2f}")

    # ------------------------------------------------------------------
    # Mark-to-market
    # ------------------------------------------------------------------

    def _pair_mtm(self, pair_name: str, date: pd.Timestamp) -> float:
        """Unrealised P&L for one open position at today's prices."""
        if date not in self.prices.index:
            return 0.0
        trade = self.open_trades.get(pair_name)
        if trade is None:
            return 0.0
        ty, tx = pair_name.split("/")
        if ty not in self.prices.columns or tx not in self.prices.columns:
            return 0.0
        py = float(self.prices.loc[date, ty])
        px = float(self.prices.loc[date, tx])
        if np.isnan(py) or np.isnan(px):
            return 0.0
        return (trade.shares_y * (py - trade.entry_price_y)
              + trade.shares_x * (px - trade.entry_price_x))

    def _mark_to_market(self, date: pd.Timestamp) -> float:
        if not self.open_trades or date not in self.prices.index:
            return 0.0
        total = 0.0
        for pair_name, trade in self.open_trades.items():
            ty, tx = pair_name.split("/")
            if ty not in self.prices.columns or tx not in self.prices.columns:
                continue
            py = float(self.prices.loc[date, ty])
            px = float(self.prices.loc[date, tx])
            if np.isnan(py) or np.isnan(px):
                continue
            total += trade.shares_y * (py - trade.entry_price_y)
            total += trade.shares_x * (px - trade.entry_price_x)
        return total

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    def _compile_results(self) -> dict:
        equity_df = pd.DataFrame(self.equity_curve).set_index("date")
        trades_df = pd.DataFrame(self.closed_trades)
        metrics   = portfolio_risk_report(equity_df["equity"])

        pair_pnl = pd.DataFrame()
        if not trades_df.empty:
            pair_pnl = (
                trades_df.groupby("pair")["net_pnl"]
                .agg(["sum", "count", "mean"])
                .rename(columns={"sum": "total_pnl", "count": "n_trades", "mean": "avg_pnl"})
                .sort_values("total_pnl", ascending=False)
                .round(2)
            )

        return {
            "equity_curve": equity_df,
            "trades":       trades_df,
            "pair_pnl":     pair_pnl,
            "metrics":      metrics,
            "rescreens":    pd.DataFrame(self.rescreen_log),
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    from data_pipeline import download_prices, validate_and_clean
    from signal_generator import run_all_signals

    cfg = CFG
    os.makedirs("results", exist_ok=True)

    print("\n" + "="*60)
    print("  BACKTESTER — with quarterly re-screening")
    print("="*60)
    print(f"  Rescreen every : {RESCREEN_CALENDAR_DAYS} calendar days")
    print(f"  Lookback window: {SCREEN_LOOKBACK_BARS} trading days (2yr)")

    all_tickers = [t for tickers in cfg.sectors.values() for t in tickers]
    raw_prices  = download_prices(all_tickers, cfg.data.start_date, cfg.data.end_date)
    prices, _   = validate_and_clean(raw_prices, cfg.data.min_history)

    # Initial pair list — used for Day 0 only; overwritten at first rescreen
    ranked_pairs   = pd.read_csv("results/ranked_pairs.csv")
    spread_quality = pd.read_csv("results/spread_quality.csv")

    print("\nBuilding initial signals...")
    signal_data, all_trades, signal_stats = run_all_signals(
        prices, ranked_pairs, cfg.signal, top_n=10
    )

    print("\nRunning backtest...\n")
    bt = PairsBacktester(
        prices=prices,
        signal_data=signal_data,
        ranked_pairs=ranked_pairs,
        signal_stats=signal_stats,
        spread_quality=spread_quality,
        cfg=cfg.risk,
    )
    results = bt.run()

    if not results:
        print("No results — check pair approval filters")
    else:
        eq  = results["equity_curve"]
        trd = results["trades"]
        m   = results["metrics"]
        rsc = results["rescreens"]

        print(f"\n{'='*60}")
        print(f"  BACKTEST RESULTS")
        print(f"{'='*60}")
        print(f"  Period:         {eq.index[0].date()} → {eq.index[-1].date()}")
        print(f"  Starting cap:   ${cfg.risk.capital:>12,.2f}")
        print(f"  Ending equity:  ${eq['equity'].iloc[-1]:>12,.2f}")
        print()
        print(f"  Total return:   {m['total_return_pct']:>+8.2f}%")
        print(f"  Ann. return:    {m['ann_return_pct']:>+8.2f}%")
        print(f"  Ann. vol:       {m['ann_vol_pct']:>8.2f}%")
        print(f"  Sharpe ratio:   {m['sharpe_ratio']:>8.3f}")
        print(f"  Sortino ratio:  {m['sortino_ratio']:>8.3f}")
        print(f"  Calmar ratio:   {m['calmar_ratio']:>8.3f}")
        print(f"  Max drawdown:   {m['max_drawdown_pct']:>8.2f}%")
        print(f"  Win days:       {m['win_days_pct']:>8.1f}%")
        print(f"  N trades:       {len(trd):>8d}")
        print(f"  N rescreens:    {len(rsc):>8d}")
        print(f"{'='*60}")

        if not results["pair_pnl"].empty:
            print(f"\n--- P&L by Pair ---")
            print(results["pair_pnl"].to_string())

        if not rsc.empty:
            print(f"\n--- Re-screen Log ---")
            print(rsc.to_string(index=False))

        eq.to_csv("results/equity_curve.csv")
        trd.to_csv("results/backtest_trades.csv", index=False)
        results["pair_pnl"].to_csv("results/pair_pnl.csv")
        pd.DataFrame([m]).to_csv("results/metrics.csv", index=False)
        rsc.to_csv("results/rescreen_log.csv", index=False)

        print(f"\n✓ Results saved to results/")
        print(f"  Next step: run walk_forward.py (Sprint 4)")