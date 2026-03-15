"""
risk_manager.py — Position sizing and portfolio risk controls.

CONCEPT: Kelly Criterion
  The Kelly formula tells you the optimal fraction of capital to bet
  given your edge and odds. For a bet with win probability p, win
  return b, and loss return 1:

      f* = (bp - q) / b      where q = 1 - p

  Full Kelly maximises long-run wealth but produces large drawdowns.
  We use HALF Kelly (kelly_fraction=0.5) which halves the position
  but dramatically smooths the equity curve — a standard industry practice.

  For a pairs trade we adapt Kelly to spread P&L:
      f* = (E[spread_pnl] * kelly_fraction) / Var[spread_pnl]

CONCEPT: Volatility targeting
  Instead of fixed-dollar positions, we size so that EACH pair
  contributes equal daily vol to the portfolio. A pair with tight
  spread (low σ) gets a bigger notional than a pair with a wide,
  noisy spread — because both should feel the same to the portfolio.

  Target vol per pair = (total_portfolio_vol_target) / sqrt(n_pairs)
  Dollar size = target_vol_per_pair / spread_daily_vol

CONCEPT: Drawdown control
  We monitor rolling drawdown on the live equity curve.
  When drawdown exceeds a soft limit (50% of max_DD), we scale all
  new positions by 50%. When it exceeds the hard limit, we go flat
  on ALL positions and cease trading until drawdown recovers.
  This is called a "circuit breaker" — essential for real capital.
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from config import CFG, RiskConfig

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Kelly position sizing
# ---------------------------------------------------------------------------

def kelly_size(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    kelly_fraction: float = 0.5,
) -> float:
    """
    Fractional Kelly for spread trades.

    Parameters
    ----------
    win_rate      : historical win rate (0-1)
    avg_win       : average spread P&L on winning trades (positive)
    avg_loss      : average spread P&L on losing trades (negative)
    kelly_fraction: scaling factor (0.5 = half-Kelly)

    Returns
    -------
    f : fraction of capital to allocate (capped at 0.25 per pair)
    """
    if avg_loss >= 0 or avg_win <= 0 or win_rate <= 0:
        return 0.0

    q = 1.0 - win_rate
    b = abs(avg_win / avg_loss)   # win/loss ratio
    f_full = (b * win_rate - q) / b
    f_kelly = max(0.0, f_full * kelly_fraction)

    # Hard cap: never bet more than 25% of capital on one pair
    return min(f_kelly, 0.25)


def volatility_target_size(
    capital: float,
    spread_daily_vol: float,
    portfolio_vol_target: float = 0.10,
    n_pairs: int = 5,
    price_y: float = 100.0,
) -> int:
    """
    Size a pair position so that its daily vol contribution equals
    portfolio_vol_target / sqrt(n_pairs).

    Returns number of shares of the Y leg (X leg is hedge_ratio * shares_y).
    """
    per_pair_vol_target = (portfolio_vol_target / np.sqrt(n_pairs)) / np.sqrt(252)
    # Dollar target such that dollar * spread_daily_vol = per_pair_vol_target * capital
    dollar_target = (per_pair_vol_target * capital) / spread_daily_vol
    shares = max(1, int(dollar_target / price_y))
    return shares


# ---------------------------------------------------------------------------
# Transaction cost model
# ---------------------------------------------------------------------------

def compute_trade_cost(
    notional_y: float,
    notional_x: float,
    commission_pct: float = 0.001,
    slippage_pct: float = 0.0005,
) -> float:
    """
    Estimate total round-trip cost for one pair trade.

    We trade TWO legs (Y and X), each entry and exit → 4 fills total.
    Cost = (commission + slippage) * notional * 4 fills
    """
    total_notional = notional_y + notional_x
    cost_per_fill  = (commission_pct + slippage_pct) * total_notional
    return cost_per_fill * 4   # 4 fills per round-trip


# ---------------------------------------------------------------------------
# Drawdown monitor
# ---------------------------------------------------------------------------

class DrawdownMonitor:
    """
    Tracks peak equity and current drawdown in real time.
    Returns a 'scale' factor (0-1) that the backtester multiplies
    position sizes by — providing smooth de-risking rather than binary on/off.
    """

    def __init__(self, soft_limit: float = 0.10, hard_limit: float = 0.20):
        self.soft_limit = soft_limit
        self.hard_limit = hard_limit
        self.peak_equity = None
        self.current_dd  = 0.0
        self.halted      = False

    def update(self, equity: float) -> float:
        """
        Update with current equity. Returns position scale factor (0-1).
        """
        if self.peak_equity is None:
            self.peak_equity = equity
        else:
            self.peak_equity = max(self.peak_equity, equity)

        self.current_dd = (self.peak_equity - equity) / self.peak_equity

        if self.current_dd >= self.hard_limit:
            self.halted = True
            return 0.0   # go flat

        # Recovery: only resume when DD < soft_limit / 2
        if self.halted and self.current_dd < self.soft_limit / 2:
            self.halted = False

        if self.halted:
            return 0.0

        if self.current_dd >= self.soft_limit:
            # Linear scale: 1.0 at soft_limit, 0.0 at hard_limit
            scale = 1.0 - (self.current_dd - self.soft_limit) / (self.hard_limit - self.soft_limit)
            return max(0.0, scale)

        return 1.0   # normal sizing

    @property
    def summary(self) -> str:
        return f"DD={self.current_dd:.1%} | Peak=${self.peak_equity:,.0f} | Halted={self.halted}"


# ---------------------------------------------------------------------------
# Pair quality gate
# ---------------------------------------------------------------------------

def passes_quality_gate(
    pair_stats: dict,
    min_trades: int = 5,
    min_adf_pvalue_threshold: float = 0.05,
    adf_pvalue: float = 1.0,
) -> Tuple[bool, str]:
    """
    Hard filters applied before allocating capital to a pair.

    Returns (passes: bool, reason: str)
    """
    n = pair_stats.get("n_trades", 0)
    pf = pair_stats.get("profit_factor", 0)

    if n < min_trades:
        return False, f"too few trades ({n} < {min_trades})"

    if adf_pvalue > min_adf_pvalue_threshold:
        return False, f"ADF p-value too high ({adf_pvalue:.3f})"

    # NOTE: pf can be NaN/inf for 100% win-rate pairs (no losing trades).
    # We do NOT reject these — a pair with zero losses is a valid signal.
    # We only reject if pf is a real negative number (returns are consistently negative).
    if pf is not None and not np.isnan(pf) and not np.isinf(pf) and pf < 0:
        return False, f"negative profit factor ({pf:.2f})"

    return True, "ok"


# ---------------------------------------------------------------------------
# Portfolio-level risk summary
# ---------------------------------------------------------------------------

def portfolio_risk_report(equity_curve: pd.Series) -> dict:
    """
    Compute standard performance and risk metrics from an equity curve.

    Metrics:
      Sharpe ratio    : annualised return / annualised vol (risk-free = 0)
      Sortino ratio   : annualised return / downside deviation
      Calmar ratio    : annualised return / max drawdown
      Max drawdown    : largest peak-to-trough decline
      Win rate        : % of days with positive P&L
    """
    returns = equity_curve.pct_change().dropna()
    n_years = len(returns) / 252

    total_return   = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
    ann_return     = (1 + total_return) ** (1 / n_years) - 1
    ann_vol        = returns.std() * np.sqrt(252)
    sharpe         = ann_return / ann_vol if ann_vol > 0 else 0

    # Sortino: only penalise downside volatility
    downside_ret   = returns[returns < 0]
    sortino_denom  = downside_ret.std() * np.sqrt(252) if len(downside_ret) > 0 else 1e-9
    sortino        = ann_return / sortino_denom

    # Max drawdown
    rolling_max    = equity_curve.cummax()
    drawdown       = (equity_curve - rolling_max) / rolling_max
    max_dd         = float(drawdown.min())

    calmar         = ann_return / abs(max_dd) if max_dd != 0 else 0
    win_days       = (returns > 0).mean()

    return {
        "total_return_pct":   round(total_return * 100, 2),
        "ann_return_pct":     round(ann_return * 100, 2),
        "ann_vol_pct":        round(ann_vol * 100, 2),
        "sharpe_ratio":       round(sharpe, 3),
        "sortino_ratio":      round(sortino, 3),
        "calmar_ratio":       round(calmar, 3),
        "max_drawdown_pct":   round(max_dd * 100, 2),
        "win_days_pct":       round(win_days * 100, 1),
        "n_trading_days":     len(returns),
    }
