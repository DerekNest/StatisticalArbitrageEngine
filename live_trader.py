"""
live_trader.py — End-of-day live execution via Alpaca Markets.

WHY ALPACA OVER IBKR:
  - No TWS desktop app required — pure REST API
  - Paper trading built in, free, instant signup at alpaca.markets
  - Works on Python 3.14+ with zero compatibility issues
  - Commission-free US equities, fractional shares supported
  - Short selling available on margin accounts

SETUP (5 minutes):
  1. Sign up at alpaca.markets — free, no deposit needed for paper trading
  2. Switch to Paper Trading mode in the dashboard
  3. Go to API Keys → Generate New Key
  4. Set environment variables in PowerShell before running:

     $env:ALPACA_API_KEY    = "PKxxxxxxxxxxxxxxxxxxxxxxxx"
     $env:ALPACA_API_SECRET = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

INSTALL:
  pip install requests pandas numpy yfinance

SCHEDULING (Windows):
  Task Scheduler → Create Basic Task → Daily → 3:50pm → python live_trader.py

PAPER vs LIVE:
  PAPER_TRADING = True  → paper endpoint, fake money (default, always start here)
  PAPER_TRADING = False → live endpoint, real money (do NOT touch for weeks)
"""

import json
import logging
import os
import warnings
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
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

PAPER_TRADING     = True   # ALWAYS start True — change only after 4+ weeks paper

ALPACA_API_KEY    = os.environ.get("ALPACA_API_KEY",    "YOUR_KEY_HERE")
ALPACA_API_SECRET = os.environ.get("ALPACA_API_SECRET", "YOUR_SECRET_HERE")

BASE_URL = (
    "https://paper-api.alpaca.markets"
    if PAPER_TRADING else
    "https://api.alpaca.markets"
)
DATA_URL = "https://data.alpaca.markets"

DISCORD_WEBHOOK      = os.environ.get("DISCORD_WEBHOOK_URL", None)
STATE_FILE           = "live_state.json"
LOG_DIR              = Path("live_logs")
RESCREEN_EVERY_DAYS  = 90


# ---------------------------------------------------------------------------
# Alpaca REST client — no SDK, plain requests, works on any Python version
# ---------------------------------------------------------------------------

class AlpacaClient:
    def __init__(self, key: str, secret: str, base_url: str):
        self.base = base_url.rstrip("/")
        self.h    = {
            "APCA-API-KEY-ID":     key,
            "APCA-API-SECRET-KEY": secret,
            "Content-Type":        "application/json",
        }

    def _get(self, path: str, params: dict = None) -> dict:
        r = requests.get(f"{self.base}{path}", headers=self.h,
                         params=params, timeout=10)
        r.raise_for_status()
        return r.json()

    def _post(self, path: str, body: dict) -> dict:
        r = requests.post(f"{self.base}{path}", headers=self.h,
                          json=body, timeout=10)
        r.raise_for_status()
        return r.json()

    def _delete(self, path: str) -> None:
        requests.delete(f"{self.base}{path}", headers=self.h,
                        timeout=10).raise_for_status()

    def get_account(self) -> dict:
        return self._get("/v2/account")

    def get_equity(self) -> float:
        return float(self.get_account()["equity"])

    def get_clock(self) -> dict:
        return self._get("/v2/clock")

    def is_market_open(self) -> bool:
        return self.get_clock().get("is_open", False)

    def get_positions(self) -> List[dict]:
        return self._get("/v2/positions")

    def close_all_positions(self) -> None:
        self._delete("/v2/positions")

    def is_shortable(self, symbol: str) -> bool:
        try:
            a = self._get(f"/v2/assets/{symbol}")
            return a.get("shortable", False) and a.get("easy_to_borrow", False)
        except Exception:
            return False

    def get_latest_price(self, symbol: str) -> Optional[float]:
        try:
            r = requests.get(
                f"{DATA_URL}/v2/stocks/{symbol}/trades/latest",
                headers=self.h, timeout=10,
            )
            r.raise_for_status()
            return float(r.json()["trade"]["p"])
        except Exception:
            return None

    def submit_moc_order(self, symbol: str, qty: float, side: str) -> dict:
        """Market-on-close order — fills at official closing price."""
        body = {
            "symbol":        symbol,
            "qty":           str(int(abs(qty))),
            "side":          side,            # "buy" or "sell"
            "type":          "market",
            "time_in_force": "cls",           # cls = closing session (MOC)
        }
        log.info(f"  ORDER: {side.upper()} {int(abs(qty))} {symbol} MOC")
        return self._post("/v2/orders", body)

    def cancel_all_orders(self) -> None:
        self._delete("/v2/orders")


# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------

def load_state() -> dict:
    if not Path(STATE_FILE).exists():
        return {
            "positions":     {},
            "last_rescreen": None,
            "equity":        None,
            "blacklisted":   [],
            "trade_log":     [],
        }
    with open(STATE_FILE) as f:
        return json.load(f)


def save_state(state: dict) -> None:
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Prices
# ---------------------------------------------------------------------------

def fetch_prices(tickers: List[str], lookback_days: int = 520) -> pd.DataFrame:
    end   = datetime.today().strftime("%Y-%m-%d")
    start = (datetime.today() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    log.info(f"Fetching prices for {len(tickers)} tickers...")
    raw = yf.download(tickers, start=start, end=end,
                      auto_adjust=True, progress=False, threads=True)
    prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else \
             raw[["Close"]].rename(columns={"Close": tickers[0]})
    prices.index = pd.to_datetime(prices.index)
    return prices.ffill(limit=3).dropna(how="all")


# ---------------------------------------------------------------------------
# Signal generation
# ---------------------------------------------------------------------------

def generate_today_signals(
    prices:       pd.DataFrame,
    ranked_pairs: pd.DataFrame,
    state:        dict,
) -> Dict[str, dict]:
    from config import CFG
    from spread_model import compute_spread, detect_regime
    from signal_generator import generate_signals

    signals     = {}
    blacklisted = set(state.get("blacklisted", []))
    open_pos    = state.get("positions", {})

    for _, row in ranked_pairs.iterrows():
        ty, tx    = row["ticker_y"], row["ticker_x"]
        pair_name = f"{ty}/{tx}"

        if pair_name in blacklisted:
            continue
        if ty not in prices.columns or tx not in prices.columns:
            continue

        spread_df = compute_spread(
            prices.iloc[-504:][ty], prices.iloc[-504:][tx],
            window=CFG.signal.zscore_window,
            hedge_method=CFG.signal.hedge_method,
        ).dropna(subset=["zscore_smoothed"])

        if len(spread_df) < 60:
            continue

        sig_df = generate_signals(spread_df, CFG.signal, detect_regime(spread_df))
        if sig_df.empty:
            continue

        today       = sig_df.iloc[-1]
        zscore      = float(today["zscore_smoothed"])
        hedge       = float(today["hedge_ratio"])
        position    = int(today["position"])
        in_position = pair_name in open_pos

        if bool(today["trade_open"]) and not in_position and position != 0:
            action = "enter_long" if position > 0 else "enter_short"
        elif bool(today["trade_close"]) and in_position:
            action = "stop" if abs(zscore) >= CFG.signal.stop_z else "exit"
        elif in_position:
            action = "hold"
        else:
            continue

        signals[pair_name] = {
            "action":      action,
            "zscore":      round(zscore, 3),
            "hedge_ratio": round(hedge, 4),
            "direction":   position,
            "ticker_y":    ty,
            "ticker_x":    tx,
        }

    return signals


# ---------------------------------------------------------------------------
# Position sizing
# ---------------------------------------------------------------------------

def compute_position_size(
    equity:      float,
    price_y:     float,
    price_x:     float,
    hedge_ratio: float,
) -> Tuple[float, float]:
    from config import CFG
    from risk_manager import kelly_size

    f = kelly_size(0.60, 0.002, -0.001, CFG.risk.kelly_fraction)
    if f <= 0:
        return 0.0, 0.0

    notional_y = min(equity * f * CFG.risk.risk_per_trade / 0.02, equity * 0.12)
    if notional_y < 500:
        return 0.0, 0.0

    shares_y = notional_y / price_y
    shares_x = shares_y * hedge_ratio * price_x / price_x
    return round(shares_y, 2), round(shares_x, 2)


# ---------------------------------------------------------------------------
# Trade execution
# ---------------------------------------------------------------------------

def execute_entry(
    client:    AlpacaClient,
    pair_name: str,
    signal:    dict,
    prices:    pd.DataFrame,
    state:     dict,
    equity:    float,
) -> bool:
    ty, tx    = signal["ticker_y"], signal["ticker_x"]
    direction = signal["direction"]
    hedge     = signal["hedge_ratio"]

    price_y = float(prices[ty].iloc[-1])
    price_x = float(prices[tx].iloc[-1])

    shares_y, shares_x = compute_position_size(equity, price_y, price_x, hedge)
    if shares_y == 0:
        log.warning(f"  {pair_name}: position too small")
        return False

    short_leg = tx if direction > 0 else ty
    if not client.is_shortable(short_leg):
        log.warning(f"  {pair_name}: {short_leg} not shortable — skipping")
        return False

    side_y = "buy"  if direction > 0 else "sell"
    side_x = "sell" if direction > 0 else "buy"

    log.info(
        f"  ENTER {pair_name} | "
        f"{side_y.upper()} {shares_y:.0f} {ty} @ ~${price_y:.2f} | "
        f"{side_x.upper()} {shares_x:.0f} {tx} @ ~${price_x:.2f}"
    )

    try:
        oy = client.submit_moc_order(ty, shares_y, side_y)
        ox = client.submit_moc_order(tx, shares_x, side_x)
    except Exception as e:
        log.error(f"  {pair_name}: entry failed — {e}")
        return False

    state["positions"][pair_name] = {
        "pair":          pair_name,
        "direction":     direction,
        "entry_date":    date.today().isoformat(),
        "entry_price_y": price_y,
        "entry_price_x": price_x,
        "shares_y":       direction * shares_y,
        "shares_x":      -direction * shares_x,
        "hedge_ratio":   hedge,
        "entry_zscore":  signal["zscore"],
        "order_id_y":    oy.get("id"),
        "order_id_x":    ox.get("id"),
    }
    return True


def execute_exit(
    client:    AlpacaClient,
    pair_name: str,
    prices:    pd.DataFrame,
    state:     dict,
    forced:    bool = False,
) -> bool:
    pos = state["positions"].get(pair_name)
    if not pos:
        return False

    ty, tx   = pair_name.split("/")
    shares_y = pos["shares_y"]
    shares_x = pos["shares_x"]

    side_y = "sell" if shares_y > 0 else "buy"
    side_x = "sell" if shares_x > 0 else "buy"

    log.info(
        f"  {'STOP' if forced else 'EXIT'} {pair_name} | "
        f"{side_y.upper()} {abs(shares_y):.0f} {ty} | "
        f"{side_x.upper()} {abs(shares_x):.0f} {tx}"
    )

    try:
        client.submit_moc_order(ty, abs(shares_y), side_y)
        client.submit_moc_order(tx, abs(shares_x), side_x)
    except Exception as e:
        log.error(f"  {pair_name}: exit failed — {e}")
        return False

    pnl = (shares_y * (float(prices[ty].iloc[-1]) - pos["entry_price_y"])
         + shares_x * (float(prices[tx].iloc[-1]) - pos["entry_price_x"]))

    state["trade_log"].append({
        "pair":       pair_name,
        "entry_date": pos["entry_date"],
        "exit_date":  date.today().isoformat(),
        "net_pnl":    round(pnl, 2),
        "forced":     forced,
    })
    del state["positions"][pair_name]
    log.info(f"  {pair_name} closed | approx P&L ${pnl:+,.2f}")

    if forced:
        state["blacklisted"].append(pair_name)
    return True


# ---------------------------------------------------------------------------
# Circuit breakers
# ---------------------------------------------------------------------------

def check_circuit_breakers(
    prices: pd.DataFrame,
    state:  dict,
) -> List[str]:
    from config import CFG
    max_loss = CFG.risk.capital * CFG.risk.max_loss_per_pair
    to_close = []

    for pair_name, pos in state["positions"].items():
        ty, tx = pair_name.split("/")
        if ty not in prices.columns or tx not in prices.columns:
            continue
        mtm = (pos["shares_y"] * (float(prices[ty].iloc[-1]) - pos["entry_price_y"])
             + pos["shares_x"] * (float(prices[tx].iloc[-1]) - pos["entry_price_x"]))
        if mtm < -max_loss:
            log.warning(f"  CIRCUIT BREAKER: {pair_name} MTM=${mtm:+,.0f}")
            to_close.append(pair_name)

    return to_close


# ---------------------------------------------------------------------------
# Discord
# ---------------------------------------------------------------------------

def send_discord_alert(message: str) -> None:
    if not DISCORD_WEBHOOK:
        return
    try:
        requests.post(DISCORD_WEBHOOK, json={"content": message}, timeout=5)
    except Exception as e:
        log.warning(f"Discord failed: {e}")


def format_alert(signals: dict, state: dict, equity: float) -> str:
    lines = [
        f"**StatArb EOD — {date.today().isoformat()}**",
        f"Equity: ${equity:,.0f} | Open positions: {len(state['positions'])}",
        "",
    ]
    entries = {k: v for k, v in signals.items() if "enter" in v["action"]}
    exits   = {k: v for k, v in signals.items() if v["action"] in ("exit", "stop")}

    if entries:
        lines.append("**Entries:**")
        for p, s in entries.items():
            lines.append(f"  {s['action'].upper()} {p} | z={s['zscore']:+.2f}")
    if exits:
        lines.append("**Exits:**")
        for p, s in exits.items():
            lines.append(f"  {s['action'].upper()} {p} | z={s['zscore']:+.2f}")
    if not entries and not exits:
        lines.append("No trades today.")

    holding = list(state["positions"].keys())
    if holding:
        lines.append(f"\nHolding: {', '.join(holding)}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_daily() -> None:
    LOG_DIR.mkdir(exist_ok=True)
    log.info("=" * 60)
    log.info(f"  StatArb Live Trader — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    log.info(f"  Mode: {'PAPER' if PAPER_TRADING else '*** LIVE MONEY ***'}")
    log.info("=" * 60)

    if ALPACA_API_KEY == "YOUR_KEY_HERE":
        log.error("Set ALPACA_API_KEY and ALPACA_API_SECRET first")
        log.error("  PowerShell: $env:ALPACA_API_KEY = 'your_key'")
        return

    # Connect
    client = AlpacaClient(ALPACA_API_KEY, ALPACA_API_SECRET, BASE_URL)
    try:
        account = client.get_account()
        equity  = float(account["equity"])
        log.info(f"Connected | Equity: ${equity:,.2f}")
    except Exception as e:
        log.error(f"Alpaca connection failed: {e}")
        return

    state         = load_state()
    state["equity"] = equity

    # Prices
    from config import CFG
    all_tickers = [t for tickers in CFG.sectors.values() for t in tickers]
    prices = fetch_prices(all_tickers)
    if prices.empty:
        log.error("No price data")
        return

    # Rescreen if due
    last_rs    = state.get("last_rescreen")
    days_since = 999 if not last_rs else \
                 (date.today() - date.fromisoformat(last_rs)).days

    if days_since >= RESCREEN_EVERY_DAYS:
        log.info("Rescreening pairs...")
        from data_pipeline import build_sector_universe, validate_and_clean
        from pair_screener import screen_all_sectors

        clean, _ = validate_and_clean(prices, CFG.data.min_history)
        universe  = build_sector_universe(clean.iloc[-504:], CFG.sectors)
        new_pairs = screen_all_sectors(
            universe, CFG.screen, top_n_per_sector=8, lookback_days=None
        )
        if not new_pairs.empty:
            new_pairs = new_pairs[
                new_pairs["coint_pvalue"] < CFG.screen.coint_pvalue_threshold
            ]
            new_pairs.to_csv("live_ranked_pairs.csv", index=False)
            state["last_rescreen"] = date.today().isoformat()
            log.info(f"Rescreen done: {len(new_pairs)} pairs")

    if not Path("live_ranked_pairs.csv").exists():
        log.error("No live_ranked_pairs.csv — first run will generate it on rescreen day")
        save_state(state)
        return

    ranked_pairs = pd.read_csv("live_ranked_pairs.csv")

    # Signals
    log.info("Generating signals...")
    signals = generate_today_signals(prices, ranked_pairs, state)
    for pair, sig in signals.items():
        log.info(f"  {pair}: {sig['action']} | z={sig['zscore']:+.2f}")
    if not signals:
        log.info("No actionable signals today")

    # Circuit breakers
    to_force_close = check_circuit_breakers(prices, state)

    # Market clock
    clock = client.get_clock()
    log.info(f"Market: {'OPEN' if clock['is_open'] else 'CLOSED'}")

    # Exits first, then entries
    for p in to_force_close:
        execute_exit(client, p, prices, state, forced=True)

    for p, sig in signals.items():
        if sig["action"] in ("exit", "stop") and p not in to_force_close:
            execute_exit(client, p, prices, state, forced=(sig["action"] == "stop"))

    n_open = len(state["positions"])
    for p, sig in signals.items():
        if sig["action"] not in ("enter_long", "enter_short"):
            continue
        if n_open >= CFG.risk.max_pairs:
            break
        if execute_entry(client, p, sig, prices, state, equity):
            n_open += 1

    # Alert + save
    alert = format_alert(signals, state, equity)
    send_discord_alert(alert)
    log.info(f"\n{alert}")

    save_state(state)

    log_file = LOG_DIR / f"{date.today().isoformat()}.json"
    with open(log_file, "w") as f:
        json.dump({
            "date":     date.today().isoformat(),
            "equity":   equity,
            "signals":  signals,
            "n_open":   len(state["positions"]),
            "holdings": list(state["positions"].keys()),
        }, f, indent=2)

    log.info(f"Done | Open: {len(state['positions'])} positions")
    log.info("=" * 60)


if __name__ == "__main__":
    run_daily()