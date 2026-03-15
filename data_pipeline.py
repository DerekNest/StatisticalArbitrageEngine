"""
data_pipeline.py — Download, cache, and validate price data.
Supports both daily (1d) and hourly (1h) resolutions via DATA_MODE in config.py.

CONCEPT: Why adjusted close prices?
  Raw close prices contain gaps from dividends and stock splits. Adjusted
  close backtracks these so our returns are economically accurate.

CONCEPT: The 730-day yfinance hourly limit
  yfinance enforces a hard server-side limit: interval="1h" returns at most
  730 calendar days of history, regardless of the start_date you request.
  Requesting further back silently returns the same 730-day window.
  This means hourly walk-forward validation is limited to 1 fold maximum
  (2yr train + 1yr test = 3yr needed; yfinance provides ~2yr).
  For longer hourly history use Polygon.io, Alpaca Markets, or EODHD.

CONCEPT: Intraday boundary filtering
  Hourly bars include a 09:30 open bar and a 16:00 close bar each day.
  Some brokers also return pre-market (04:00-09:30) and post-market
  (16:00-20:00) bars. These must be removed because:
    1. Spreads are artificially wide outside regular hours (thin liquidity)
    2. Including them inflates apparent volatility in our z-score window
    3. Our execution model assumes regular-session fills only

CONCEPT: Why Parquet caching?
  yfinance has rate limits. Parquet is a columnar binary format read/write
  is ~10-50x faster than CSV for numerical data and the files are smaller.
  Cache keys encode date range AND interval so daily and hourly caches
  never collide on disk.
"""

import datetime
import logging
import warnings
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from config import CFG, SECTORS, DATA_MODE

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

_SESSION_START = datetime.time(9, 30)
_SESSION_END   = datetime.time(16, 0)


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_prices(
    tickers: list,
    start: str,
    end: str,
    interval: str = "1d",
    cache_dir: str = "data/",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Download prices for a list of tickers at the given interval.

    Cache key includes interval so 1d and 1h caches never cross-contaminate.

    For interval="1h":
      - yfinance returns at most 730 calendar days regardless of start_date.
      - Index is timezone-aware (America/New_York); we strip tz after filtering.
    """
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    safe_start = start.replace("-", "") if start else "auto"
    safe_end   = end.replace("-", "")   if end   else "auto"
    cache_file = Path(cache_dir) / f"prices_{interval}_{safe_start}_{safe_end}.parquet"

    if cache_file.exists() and not force_refresh:
        log.info(f"Loading cached prices from {cache_file}")
        prices = pd.read_parquet(cache_file)
        log.info(f"Loaded {prices.shape[1]} tickers, {len(prices)} bars ({interval})")
        return prices

    if interval == "1h":
        log.warning(
            "Hourly download: yfinance hard limit = 730 calendar days. "
            "start_date is advisory only. "
            "For longer history use Polygon.io or Alpaca Markets."
        )

    log.info(f"Downloading {len(tickers)} tickers "
             f"({interval}) {start or 'max'} -> {end or 'today'}...")

    kwargs = dict(auto_adjust=True, progress=True, threads=True)
    if start: kwargs["start"] = start
    if end:   kwargs["end"]   = end

    raw = yf.download(tickers, interval=interval, **kwargs)

    prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else \
             raw[["Close"]].rename(columns={"Close": tickers[0]})

    prices.index = pd.to_datetime(prices.index)

    # Strip timezone — keep tz-naive for consistency with daily data
    if prices.index.tz is not None:
        prices.index = prices.index.tz_convert("America/New_York").tz_localize(None)

    if interval == "1h":
        prices = _filter_regular_session(prices)

    prices.to_parquet(cache_file)
    log.info(f"Saved {prices.shape[1]} tickers, {len(prices)} bars to cache")
    return prices


def _filter_regular_session(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only bars within the regular trading session (09:30-16:00 ET).

    Removes pre/post-market bars which have artificially wide spreads and
    inflated volatility — both would corrupt our z-score and position sizing.
    """
    times = pd.Series(prices.index.time, index=prices.index)
    mask  = (times >= _SESSION_START) & (times <= _SESSION_END)
    filtered = prices[mask.values]
    removed  = len(prices) - len(filtered)
    if removed > 0:
        log.info(f"Session filter: removed {removed} pre/post-market bars "
                 f"({removed/len(prices):.1%})")
    return filtered


# ---------------------------------------------------------------------------
# Quality validation
# ---------------------------------------------------------------------------

def validate_and_clean(
    prices: pd.DataFrame,
    min_history: int = 252,
    max_missing_pct: float = 0.02,
    interval: str = "1d",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Remove tickers that do not meet data quality standards.

    For hourly mode min_history is measured in bars (hours), not days.
    The config auto-sets min_history=390*3 for hourly (3 months of hours).
    """
    ffill_limit = 3 if interval == "1h" else 5
    report      = []
    total_rows  = len(prices)

    for ticker in prices.columns:
        series      = prices[ticker]
        n_valid     = series.notna().sum()
        missing_pct = series.isna().sum() / total_rows if total_rows else 1.0

        status = "ok"
        if n_valid < min_history:
            status = f"drop: only {n_valid} bars (need {min_history})"
        elif missing_pct > max_missing_pct:
            status = f"drop: {missing_pct:.1%} missing"

        report.append({"ticker": ticker, "valid_bars": n_valid,
                        "missing_pct": round(missing_pct * 100, 2),
                        "status": status})

    report_df    = pd.DataFrame(report)
    good_tickers = report_df.loc[report_df["status"] == "ok", "ticker"].tolist()
    dropped      = report_df.loc[report_df["status"] != "ok"]

    if len(dropped) > 0:
        log.warning(f"Dropping {len(dropped)} tickers:\n{dropped.to_string(index=False)}")

    clean = prices[good_tickers].ffill(limit=ffill_limit).dropna()
    unit  = "bars" if interval == "1h" else "trading days"
    log.info(f"Clean universe: {len(good_tickers)} tickers, {len(clean)} {unit}")
    return clean, report_df


# ---------------------------------------------------------------------------
# Returns
# ---------------------------------------------------------------------------

def compute_returns(prices: pd.DataFrame, log_returns: bool = True) -> pd.DataFrame:
    """
    Compute bar-to-bar returns.

    HOURLY EDGE CASE: naive pct_change() across the full index produces a
    spurious overnight return between 16:00 Friday and 09:30 Monday.
    We zero out the first bar of each trading day (09:30) so only
    true intraday bar-to-bar moves contribute to returns and volatility.
    """
    if log_returns:
        raw = np.log(prices / prices.shift(1))
    else:
        raw = prices.pct_change()

    # Zero overnight gaps for hourly data
    if hasattr(prices.index, "time"):
        first_bars = pd.Series(prices.index.time, index=prices.index) == _SESSION_START
        raw[first_bars] = np.nan

    return raw.dropna(how="all")


# ---------------------------------------------------------------------------
# Universe builder
# ---------------------------------------------------------------------------

def build_sector_universe(
    prices: pd.DataFrame,
    sectors: dict = None,
) -> Dict[str, pd.DataFrame]:
    """
    Split cleaned prices into sector sub-frames.

    Pairs are searched WITHIN sectors only — stocks in the same sector
    share macro exposures (interest rates, commodity prices, regulation)
    which is what drives cointegration. Cross-sector pairs require
    explicit factor neutralisation to be reliable.
    """
    sectors  = sectors or SECTORS
    universe = {}

    for sector, tickers in sectors.items():
        available = [t for t in tickers if t in prices.columns]
        if len(available) < 2:
            log.warning(f"Sector '{sector}' has <2 tickers — skipping")
            continue
        n_pairs = len(available) * (len(available) - 1) // 2
        universe[sector] = prices[available]
        log.info(f"  {sector:15s}: {len(available):3d} tickers → {n_pairs:4d} candidate pairs")

    total = sum(len(v.columns)*(len(v.columns)-1)//2 for v in universe.values())
    log.info(f"Total candidate pairs: {total}")
    return universe


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def price_summary(prices: pd.DataFrame, interval: str = "1d") -> pd.DataFrame:
    """Quick sanity-check summary."""
    hrs_per_year = 252 * 6.5 if interval == "1h" else 252
    unit = "bars" if interval == "1h" else "days"
    return pd.DataFrame({
        "start":        prices.apply(lambda s: s.dropna().index[0]),
        "end":          prices.apply(lambda s: s.dropna().index[-1]),
        f"n_{unit}":    prices.notna().sum(),
        "first_px":     prices.apply(lambda s: round(s.dropna().iloc[0], 2)),
        "last_px":      prices.apply(lambda s: round(s.dropna().iloc[-1], 2)),
        "ann_vol":      (compute_returns(prices).std() * np.sqrt(hrs_per_year)).round(3),
    })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    cfg = CFG
    os.makedirs(cfg.data.cache_dir, exist_ok=True)
    os.makedirs("results", exist_ok=True)

    print("\n" + "="*60)
    print(f"  DATA PIPELINE  [mode={DATA_MODE}  interval={cfg.data.interval}]")
    print("="*60)

    if DATA_MODE == "1h":
        print(f"\n  ⚠  Hourly mode: yfinance limit = last 730 days")
        print(f"     For multi-year hourly history: Polygon.io / Alpaca / EODHD\n")

    all_tickers = [t for tickers in cfg.sectors.values() for t in tickers]
    print(f"Universe: {len(all_tickers)} tickers across {len(cfg.sectors)} sectors")
    print(f"Sectors:  {list(cfg.sectors.keys())}\n")

    raw_prices = download_prices(
        tickers=all_tickers,
        start=cfg.data.start_date,
        end=cfg.data.end_date,
        interval=cfg.data.interval,
        cache_dir=cfg.data.cache_dir,
    )

    prices, _ = validate_and_clean(
        raw_prices,
        min_history=cfg.data.min_history,
        interval=cfg.data.interval,
    )

    print("\n--- Sector universe ---")
    universe = build_sector_universe(prices, cfg.sectors)

    print("\n--- Price summary (first 5) ---")
    print(price_summary(prices, cfg.data.interval).head(5).to_string())

    total_pairs = sum(len(v.columns)*(len(v.columns)-1)//2 for v in universe.values())
    print(f"\n✓ Pipeline ready")
    print(f"  Tickers:         {len(prices.columns)}")
    print(f"  Bars:            {len(prices)}")
    print(f"  Candidate pairs: {total_pairs}")
    print(f"  Next step:       python pair_screener.py")
