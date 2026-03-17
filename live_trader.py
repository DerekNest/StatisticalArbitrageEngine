"""
live_trader.py — End-of-day live execution via Interactive Brokers.

ARCHITECTURE: End-of-Day Signal → MOC Order
  At 3:50pm ET each trading day:
    1. Download today's prices (yfinance intraday snapshot or Alpaca)
    2. Run pair screener on trailing 504-day window (if rescreen day)
    3. Compute spread + z-score for all approved pairs
    4. Generate signals using the same logic as the backtester
    5. Compare today's signals against open IBKR positions
    6. Submit Market-On-Close (MOC) orders for any entries or exits
    7. Log everything and send a Discord/Telegram alert

PAPER TRADING FIRST:
  Set PAPER_TRADING = True (default) to route all orders to IBKR's
  paper trading gateway (port 7497) instead of live (port 7496).
  Run in paper mode for at least 4 weeks before going live.

PREREQUISITES:
  pip install ib_insync yfinance pandas numpy

  1. Download and install TWS (Trader Workstation) or IB Gateway
     https://www.interactivebrokers.com/en/trading/tws.php
  2. In TWS: File → Global Configuration → API → Settings
     - Enable ActiveX and Socket Clients
     - Socket port: 7497 (paper) or 7496 (live)
     - Uncheck "Read-Only API"
  3. Keep TWS running while this script executes

SCHEDULING:
  Run this script at 3:50pm ET every trading day.
  On Windows: Task Scheduler
  On Mac/Linux: cron → `50 15 * * 1-5 /path/to/python /path/to/live_trader.py`
  On cloud: GitHub Actions with a scheduled workflow (see bottom of file)

RESUME FRAMING:
  "Deployed statistical arbitrage engine to live paper trading via IBKR API.
   Daily EOD signal generation, automated MOC order execution, position
   monitoring with MTM circuit breakers and Discord trade alerts."
"""

import json
import logging
import os
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PAPER_TRADING   = True          # ALWAYS start True — switch to False only after 4+ weeks paper
IBKR_HOST       = "127.0.0.1"
IBKR_PORT       = 7497 if PAPER_TRADING else 7496
IBKR_CLIENT_ID  = 1

STATE_FILE      = "live_state.json"     # persists open positions across runs
LOG_DIR         = Path("live_logs")

# Discord webhook URL for trade alerts (optional — set to None to disable)
# Get from: Discord server → channel settings → Integrations → Webhooks
DISCORD_WEBHOOK = os.environ.get("DISCORD_WEBHOOK_URL", None)

# How many calendar days between full pair rescreens
RESCREEN_EVERY_DAYS = 90


# ---------------------------------------------------------------------------
# Live state persistence
# ---------------------------------------------------------------------------

@dataclass
class LivePosition:
    """Tracks one open pairs position in the live portfolio."""
    pair:          str
    direction:     int        # +1 = long Y / short X, -1 = short Y / long X
    entry_date:    str        # ISO date string
    entry_price_y: float
    entry_price_x: float
    shares_y:      float      # signed (positive = long, negative = short)
    shares_x:      float      # signed
    hedge_ratio:   float
    entry_zscore:  float


def load_state(path: str = STATE_FILE) -> dict:
    """Load persisted live state from disk."""
    if not Path(path).exists():
        return {
            "positions":        {},       # pair_name → LivePosition dict
            "last_rescreen":    None,     # ISO date string
            "equity":           None,     # last known equity
            "blacklisted":      [],       # pairs permanently banned
            "trade_log":        [],       # list of closed trade dicts
        }
    with open(path) as f:
        return json.load(f)


def save_state(state: dict, path: str = STATE_FILE) -> None:
    """Persist live state to disk (called after every trade decision)."""
    with open(path, "w") as f:
        json.dump(state, f, indent=2, default=str)
    log.debug(f"State saved → {path}")


# ---------------------------------------------------------------------------
# Price fetching
# ---------------------------------------------------------------------------

def fetch_prices_today(tickers: List[str], lookback_days: int = 520) -> pd.DataFrame:
    """
    Download daily prices for the trailing lookback window.
    Uses yfinance — free, no API key, but has rate limits.
    For production use Alpaca or Polygon.io for reliability.
    """
    end   = datetime.today().strftime("%Y-%m-%d")
    start = (datetime.today() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

    log.info(f"Fetching prices for {len(tickers)} tickers ({start} → {end})...")
    raw = yf.download(tickers, start=start, end=end,
                      auto_adjust=True, progress=False, threads=True)

    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})

    prices.index = pd.to_datetime(prices.index)
    log.info(f"Prices loaded: {prices.shape[1]} tickers × {len(prices)} bars")
    return prices.ffill(limit=3).dropna(how="all")


# ---------------------------------------------------------------------------
# Signal generation for today
# ---------------------------------------------------------------------------

def generate_today_signals(
    prices:       pd.DataFrame,
    ranked_pairs: pd.DataFrame,
    state:        dict,
) -> Dict[str, dict]:
    """
    Run the full signal pipeline on today's prices.
    Returns a dict of {pair_name → signal_dict} for pairs with actionable signals.

    signal_dict keys:
        action       : "enter_long" | "enter_short" | "exit" | "hold" | "stop"
        zscore       : today's smoothed z-score
        hedge_ratio  : β used for this signal
        direction    : +1 or -1 (for entries)
    """
    from config import CFG
    from spread_model import compute_spread, detect_regime
    from signal_generator import generate_signals, signal_stats, extract_trades

    signals = {}
    blacklisted = set(state.get("blacklisted", []))
    open_positions = state.get("positions", {})

    for _, row in ranked_pairs.iterrows():
        ty, tx = row["ticker_y"], row["ticker_x"]
        pair_name = f"{ty}/{tx}"

        if pair_name in blacklisted:
            continue
        if ty not in prices.columns or tx not in prices.columns:
            continue

        # Use trailing 504 bars for spread computation
        lookback = prices.iloc[-504:]

        spread_df = compute_spread(
            lookback[ty], lookback[tx],
            window=CFG.signal.zscore_window,
            hedge_method=CFG.signal.hedge_method,
        )
        spread_df = spread_df.dropna(subset=["zscore_smoothed"])
        if len(spread_df) < 60:
            continue

        regimes = detect_regime(spread_df)
        sig_df  = generate_signals(spread_df, CFG.signal, regimes)

        if sig_df.empty:
            continue

        # Today's signal is the LAST row
        today_row  = sig_df.iloc[-1]
        today_z    = float(today_row["zscore_smoothed"])
        hedge      = float(today_row["hedge_ratio"])
        position   = int(today_row["position"])
        trade_open = bool(today_row["trade_open"])
        trade_close= bool(today_row["trade_close"])

        currently_open = pair_name in open_positions

        if trade_open and not currently_open and position != 0:
            action = "enter_long" if position > 0 else "enter_short"
        elif trade_close and currently_open:
            # Check stop condition
            if abs(today_z) >= CFG.signal.stop_z:
                action = "stop"
            else:
                action = "exit"
        elif currently_open:
            action = "hold"
        else:
            continue   # flat and no signal — skip

        signals[pair_name] = {
            "action":      action,
            "zscore":      round(today_z, 3),
            "hedge_ratio": round(hedge, 4),
            "direction":   position,
            "ticker_y":    ty,
            "ticker_x":    tx,
        }

    return signals


# ---------------------------------------------------------------------------
# Position sizing (mirrors backtester logic)
# ---------------------------------------------------------------------------

def compute_live_position_size(
    equity:      float,
    price_y:     float,
    price_x:     float,
    hedge_ratio: float,
    win_rate:    float = 0.60,   # conservative default if no history
    avg_win:     float = 0.002,
    avg_loss:    float = -0.001,
) -> Tuple[float, float]:
    """
    Returns (shares_y, shares_x) for a dollar-neutral pairs position.
    Mirrors _compute_position_size() in backtester.py exactly.
    """
    from config import CFG
    from risk_manager import kelly_size

    f = kelly_size(win_rate, avg_win, avg_loss, CFG.risk.kelly_fraction)
    if f <= 0:
        return 0.0, 0.0

    notional_y = equity * f * CFG.risk.risk_per_trade / 0.02
    notional_y = min(notional_y, equity * 0.12)   # 12% cap per leg

    if notional_y < 500:
        return 0.0, 0.0

    shares_y = notional_y / price_y
    shares_x = (shares_y * hedge_ratio * price_x) / price_x

    return round(shares_y, 2), round(shares_x, 2)


# ---------------------------------------------------------------------------
# IBKR execution
# ---------------------------------------------------------------------------

def connect_ibkr():
    """
    Connect to IBKR TWS or IB Gateway.
    Returns an ib_insync IB instance, or None if connection fails.
    """
    try:
        from ib_insync import IB
        ib = IB()
        ib.connect(IBKR_HOST, IBKR_PORT, clientId=IBKR_CLIENT_ID, timeout=15)
        log.info(f"Connected to IBKR {'PAPER' if PAPER_TRADING else 'LIVE'} "
                 f"({IBKR_HOST}:{IBKR_PORT})")
        return ib
    except Exception as e:
        log.error(f"IBKR connection failed: {e}")
        log.error("Is TWS / IB Gateway running? Check API settings.")
        return None


def get_ibkr_equity(ib) -> Optional[float]:
    """Fetch current net liquidation value from IBKR account."""
    try:
        account_values = ib.accountValues()
        for av in account_values:
            if av.tag == "NetLiquidation" and av.currency == "USD":
                return float(av.value)
    except Exception as e:
        log.warning(f"Could not fetch equity from IBKR: {e}")
    return None


def get_current_price(ib, ticker: str) -> Optional[float]:
    """Fetch last trade price for a ticker from IBKR."""
    try:
        from ib_insync import Stock
        contract = Stock(ticker, "SMART", "USD")
        ib.qualifyContracts(contract)
        ticker_data = ib.reqMktData(contract, "", False, False)
        ib.sleep(1.0)   # allow data to arrive
        price = ticker_data.last or ticker_data.close
        ib.cancelMktData(contract)
        return float(price) if price and price > 0 else None
    except Exception as e:
        log.warning(f"Could not fetch price for {ticker}: {e}")
        return None


def submit_moc_order(
    ib,
    ticker:    str,
    shares:    float,
    direction: str,   # "BUY" or "SELL"
    dry_run:   bool = False,
) -> Optional[str]:
    """
    Submit a Market-On-Close order for one leg of a pairs trade.

    MOC orders execute at the official closing price — zero slippage
    vs the close price, which is what our backtester assumed.
    Must be submitted before 3:50pm ET (NYSE MOC cutoff).

    Returns order ID string, or None if submission failed.
    """
    try:
        from ib_insync import Stock, Order

        contract = Stock(ticker, "SMART", "USD")
        ib.qualifyContracts(contract)

        order = Order()
        order.action         = direction          # "BUY" or "SELL"
        order.orderType      = "MOC"              # Market-On-Close
        order.totalQuantity  = abs(round(shares))
        order.transmit       = not dry_run        # False = stage but don't send

        if dry_run:
            log.info(f"  [DRY RUN] {direction} {abs(round(shares))} {ticker} MOC")
            return "DRY_RUN"

        trade = ib.placeOrder(contract, order)
        ib.sleep(0.5)
        order_id = str(trade.order.orderId)
        log.info(f"  ORDER: {direction} {abs(round(shares))} {ticker} MOC → id={order_id}")
        return order_id

    except Exception as e:
        log.error(f"Order submission failed for {ticker}: {e}")
        return None


# ---------------------------------------------------------------------------
# Trade execution logic
# ---------------------------------------------------------------------------

def execute_entry(
    ib,
    pair_name:   str,
    signal:      dict,
    prices:      pd.DataFrame,
    state:       dict,
    equity:      float,
    dry_run:     bool = False,
) -> bool:
    """
    Enter a new pairs position — submit two MOC orders (one per leg).
    Returns True if both orders were submitted successfully.
    """
    ty = signal["ticker_y"]
    tx = signal["ticker_x"]
    direction   = signal["direction"]   # +1 or -1
    hedge_ratio = signal["hedge_ratio"]

    # Get current prices for sizing
    price_y = float(prices[ty].iloc[-1])
    price_x = float(prices[tx].iloc[-1])

    shares_y, shares_x = compute_live_position_size(
        equity, price_y, price_x, hedge_ratio
    )

    if shares_y == 0:
        log.warning(f"  {pair_name}: position too small, skipping")
        return False

    # Dollar-neutral: direction=+1 → long Y, short X
    #                 direction=-1 → short Y, long X
    dir_y = "BUY"  if direction > 0 else "SELL"
    dir_x = "SELL" if direction > 0 else "BUY"

    log.info(f"  ENTER {pair_name} dir={direction:+d} "
             f"| {dir_y} {shares_y:.0f} {ty} @ ~${price_y:.2f} "
             f"| {dir_x} {shares_x:.0f} {tx} @ ~${price_x:.2f}")

    id_y = submit_moc_order(ib, ty, shares_y, dir_y, dry_run=dry_run)
    id_x = submit_moc_order(ib, tx, shares_x, dir_x, dry_run=dry_run)

    if id_y is None or id_x is None:
        log.error(f"  {pair_name}: one or both orders failed — check IBKR")
        return False

    # Record position in state
    state["positions"][pair_name] = {
        "pair":          pair_name,
        "direction":     direction,
        "entry_date":    date.today().isoformat(),
        "entry_price_y": price_y,
        "entry_price_x": price_x,
        "shares_y":      direction * shares_y,
        "shares_x":      -direction * shares_x,
        "hedge_ratio":   hedge_ratio,
        "entry_zscore":  signal["zscore"],
        "order_id_y":    id_y,
        "order_id_x":    id_x,
    }
    return True


def execute_exit(
    ib,
    pair_name: str,
    prices:    pd.DataFrame,
    state:     dict,
    forced:    bool = False,
    dry_run:   bool = False,
) -> bool:
    """
    Close an existing pairs position — reverse both legs.
    Returns True if both orders were submitted successfully.
    """
    pos = state["positions"].get(pair_name)
    if pos is None:
        log.warning(f"  {pair_name}: no open position to close")
        return False

    ty, tx = pair_name.split("/")
    shares_y = pos["shares_y"]   # signed
    shares_x = pos["shares_x"]   # signed

    # Closing = reverse the opening direction
    dir_y = "SELL" if shares_y > 0 else "BUY"
    dir_x = "SELL" if shares_x > 0 else "BUY"

    price_y = float(prices[ty].iloc[-1])
    price_x = float(prices[tx].iloc[-1])

    reason = "STOP" if forced else "EXIT"
    log.info(f"  {reason} {pair_name} "
             f"| {dir_y} {abs(shares_y):.0f} {ty} @ ~${price_y:.2f} "
             f"| {dir_x} {abs(shares_x):.0f} {tx} @ ~${price_x:.2f}")

    id_y = submit_moc_order(ib, ty, abs(shares_y), dir_y, dry_run=dry_run)
    id_x = submit_moc_order(ib, tx, abs(shares_x), dir_x, dry_run=dry_run)

    if id_y is None or id_x is None:
        log.error(f"  {pair_name}: exit orders failed — check IBKR urgently")
        return False

    # Compute approximate realised P&L
    entry_py = pos["entry_price_y"]
    entry_px = pos["entry_price_x"]
    pnl_y = shares_y * (price_y - entry_py)
    pnl_x = shares_x * (price_x - entry_px)
    net_pnl = round(pnl_y + pnl_x, 2)

    # Log the closed trade
    state["trade_log"].append({
        "pair":        pair_name,
        "entry_date":  pos["entry_date"],
        "exit_date":   date.today().isoformat(),
        "direction":   pos["direction"],
        "net_pnl":     net_pnl,
        "forced":      forced,
    })

    del state["positions"][pair_name]
    log.info(f"  {pair_name} closed | net P&L ≈ ${net_pnl:+,.2f}")
    return True


# ---------------------------------------------------------------------------
# MTM circuit breaker check
# ---------------------------------------------------------------------------

def check_circuit_breakers(
    prices: pd.DataFrame,
    state:  dict,
    equity: float,
) -> List[str]:
    """
    Check all open positions for MTM loss cap breaches.
    Returns list of pair names that should be force-closed.
    """
    from config import CFG

    to_close = []
    max_loss = CFG.risk.capital * CFG.risk.max_loss_per_pair

    for pair_name, pos in state["positions"].items():
        ty, tx = pair_name.split("/")
        if ty not in prices.columns or tx not in prices.columns:
            continue

        price_y = float(prices[ty].iloc[-1])
        price_x = float(prices[tx].iloc[-1])

        pnl_y = pos["shares_y"] * (price_y - pos["entry_price_y"])
        pnl_x = pos["shares_x"] * (price_x - pos["entry_price_x"])
        mtm   = pnl_y + pnl_x

        if mtm < -max_loss:
            log.warning(f"  CIRCUIT BREAKER: {pair_name} MTM=${mtm:+,.0f} "
                        f"(limit=${-max_loss:,.0f})")
            to_close.append(pair_name)
            state["blacklisted"].append(pair_name)

    return to_close


# ---------------------------------------------------------------------------
# Discord alerts
# ---------------------------------------------------------------------------

def send_discord_alert(message: str) -> None:
    """Post a message to a Discord channel via webhook."""
    if not DISCORD_WEBHOOK:
        return
    try:
        import urllib.request
        data = json.dumps({"content": message}).encode()
        req  = urllib.request.Request(
            DISCORD_WEBHOOK,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=5)
    except Exception as e:
        log.warning(f"Discord alert failed: {e}")


def format_daily_alert(signals: dict, state: dict, equity: float) -> str:
    """Format the end-of-day trade summary for Discord."""
    lines = [
        f"**StatArb EOD — {date.today().isoformat()}**",
        f"Equity: ${equity:,.0f}",
        f"Open positions: {len(state['positions'])}",
        "",
    ]

    entries = {k: v for k, v in signals.items() if "enter" in v["action"]}
    exits   = {k: v for k, v in signals.items() if v["action"] in ("exit", "stop")}

    if entries:
        lines.append("**Entries:**")
        for pair, sig in entries.items():
            lines.append(f"  {sig['action'].upper()} {pair} | z={sig['zscore']:+.2f}")

    if exits:
        lines.append("**Exits:**")
        for pair, sig in exits.items():
            lines.append(f"  {sig['action'].upper()} {pair} | z={sig['zscore']:+.2f}")

    if not entries and not exits:
        lines.append("No trades today.")

    open_pos = list(state["positions"].keys())
    if open_pos:
        lines.append(f"\nHolding: {', '.join(open_pos)}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main daily run
# ---------------------------------------------------------------------------

def run_daily() -> None:
    """
    Main entry point — run once at 3:50pm ET on each trading day.

    Full flow:
      1. Load persisted state
      2. Fetch today's prices
      3. Rescreen pairs if due
      4. Generate signals
      5. Check circuit breakers
      6. Connect to IBKR
      7. Execute exits, then entries
      8. Send Discord alert
      9. Save state
    """
    LOG_DIR.mkdir(exist_ok=True)
    log.info("=" * 60)
    log.info(f"  StatArb Live Trader — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info(f"  Mode: {'PAPER' if PAPER_TRADING else '*** LIVE ***'}")
    log.info("=" * 60)

    # ── 1. Load state ─────────────────────────────────────────────────────
    state = load_state()

    # ── 2. Fetch prices ───────────────────────────────────────────────────
    from config import CFG
    all_tickers = [t for tickers in CFG.sectors.values() for t in tickers]
    prices = fetch_prices_today(all_tickers, lookback_days=520)

    if prices.empty:
        log.error("No price data — aborting")
        return

    # ── 3. Rescreen pairs if due ──────────────────────────────────────────
    last_rescreen = state.get("last_rescreen")
    days_since    = 999
    if last_rescreen:
        days_since = (date.today() - date.fromisoformat(last_rescreen)).days

    if days_since >= RESCREEN_EVERY_DAYS:
        log.info(f"Rescreening pairs (last: {last_rescreen or 'never'})...")
        from data_pipeline import build_sector_universe, validate_and_clean
        from pair_screener import screen_all_sectors

        clean_prices, _ = validate_and_clean(prices, CFG.data.min_history)
        lookback_prices = clean_prices.iloc[-504:]
        universe        = build_sector_universe(lookback_prices, CFG.sectors)
        ranked_pairs    = screen_all_sectors(
            universe, CFG.screen, top_n_per_sector=6, lookback_days=None
        )
        if not ranked_pairs.empty:
            ranked_pairs = ranked_pairs[
                ranked_pairs["coint_pvalue"] < CFG.screen.coint_pvalue_threshold
            ]
            ranked_pairs.to_csv("live_ranked_pairs.csv", index=False)
            state["last_rescreen"] = date.today().isoformat()
            log.info(f"Rescreen complete: {len(ranked_pairs)} pairs")
        else:
            log.warning("No pairs survived rescreen — using previous list")
    else:
        log.info(f"No rescreen needed ({days_since} days since last)")

    # Load current ranked pairs
    ranked_pairs_path = Path("live_ranked_pairs.csv")
    if not ranked_pairs_path.exists():
        log.error("No ranked_pairs file — run a rescreen first")
        return
    ranked_pairs = pd.read_csv(ranked_pairs_path)

    # ── 4. Generate signals ───────────────────────────────────────────────
    log.info("Generating signals...")
    signals = generate_today_signals(prices, ranked_pairs, state)

    if signals:
        log.info(f"Signals: {len(signals)} actionable")
        for pair, sig in signals.items():
            log.info(f"  {pair}: {sig['action']} | z={sig['zscore']:+.2f}")
    else:
        log.info("No actionable signals today")

    # ── 5. Circuit breakers ───────────────────────────────────────────────
    to_force_close = check_circuit_breakers(prices, state, state.get("equity", CFG.risk.capital))

    # ── 6. Connect to IBKR ────────────────────────────────────────────────
    ib = connect_ibkr()
    if ib is None:
        log.error("Cannot connect to IBKR — saving state and exiting")
        save_state(state)
        return

    try:
        # Get live equity
        live_equity = get_ibkr_equity(ib) or state.get("equity") or CFG.risk.capital
        state["equity"] = live_equity
        log.info(f"Account equity: ${live_equity:,.2f}")

        # ── 7a. Force-close circuit breaker hits ──────────────────────────
        for pair_name in to_force_close:
            execute_exit(ib, pair_name, prices, state, forced=True,
                         dry_run=not PAPER_TRADING or True)  # always dry_run in paper

        # ── 7b. Normal exits ──────────────────────────────────────────────
        for pair_name, sig in signals.items():
            if sig["action"] in ("exit", "stop") and pair_name not in to_force_close:
                execute_exit(ib, pair_name, prices, state, forced=(sig["action"] == "stop"),
                             dry_run=False)

        # ── 7c. New entries ───────────────────────────────────────────────
        n_open = len(state["positions"])
        for pair_name, sig in signals.items():
            if sig["action"] not in ("enter_long", "enter_short"):
                continue
            if n_open >= CFG.risk.max_pairs:
                log.info("Portfolio full — no new entries")
                break
            if execute_entry(ib, pair_name, sig, prices, state, live_equity, dry_run=False):
                n_open += 1

    finally:
        ib.disconnect()
        log.info("Disconnected from IBKR")

    # ── 8. Discord alert ──────────────────────────────────────────────────
    alert = format_daily_alert(signals, state, state.get("equity", 0))
    send_discord_alert(alert)
    log.info("Alert sent")

    # ── 9. Save state ─────────────────────────────────────────────────────
    save_state(state)

    # Append to daily log file
    log_path = LOG_DIR / f"{date.today().isoformat()}.json"
    with open(log_path, "w") as f:
        json.dump({"date": date.today().isoformat(), "signals": signals,
                   "equity": state.get("equity"), "n_open": len(state["positions"])},
                  f, indent=2)

    log.info(f"Done. Open positions: {len(state['positions'])}")
    log.info("=" * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_daily()


# ---------------------------------------------------------------------------
# GitHub Actions workflow (save as .github/workflows/daily_trade.yml)
# ---------------------------------------------------------------------------
"""
name: Daily Trade Signal

on:
  schedule:
    - cron: '50 19 * * 1-5'   # 3:50pm ET = 19:50 UTC (adjust for DST)
  workflow_dispatch:            # allow manual trigger

jobs:
  run:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - run: pip install ib_insync yfinance pandas numpy statsmodels scipy
      - run: python live_trader.py
        env:
          DISCORD_WEBHOOK_URL: ${{ secrets.DISCORD_WEBHOOK_URL }}

NOTE: GitHub Actions cannot connect to your local IBKR TWS instance.
For cloud execution you need IBKR's server-based IB Gateway running on
a cloud VM (AWS/GCP t2.micro ~$8/month), or use Alpaca's API instead
(no local software required, has a paper trading mode).
"""
