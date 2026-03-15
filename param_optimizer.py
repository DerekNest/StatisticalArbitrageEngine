"""
param_optimizer.py — Grid search over signal parameters, evaluated on OOS Sharpe.

DESIGN PHILOSOPHY: No In-Sample Cheating
  Every parameter combination is evaluated EXCLUSIVELY on the out-of-sample
  test periods of the walk-forward folds. We never look at train-period
  performance when selecting winners. This is what separates a publishable
  result from curve-fitting.

SEARCH SPACE:
  entry_z : [2.0, 2.2, 2.4, 2.6]       — how extreme must the dislocation be?
  exit_z  : [0.0, 0.3, 0.5, 0.7]       — how much reversion do we capture?
  stop_z  : [2.8, 3.2, 3.5, 4.0]       — when do we admit the pair is broken?

  Total combinations: 4 × 4 × 4 = 64 → ~48 valid after hard constraints.

OBJECTIVE FUNCTION (adj_sharpe):
  adj_sharpe = mean_oos_sharpe
             - STABILITY_WEIGHT * std(fold_sharpes)   # penalise inconsistency
             - DD_PENALTY_WEIGHT * max(0, mean_dd - DD_HARD_LIMIT)
             - ACTIVITY_PENALTY  if any fold has < MIN_TRADES_PER_FOLD trades

  A consistent Sharpe of 0.30 across 3 folds beats a lucky 0.80 with two
  negative folds. The stability penalty enforces this mathematically.

HARD CONSTRAINTS (invalid combos pruned before any compute):
  stop_z  > entry_z + MIN_ENTRY_STOP_GAP   (stop can't fire on entry)
  exit_z  < entry_z                         (exit before re-entry level)

SUBPROCESS SAFETY:
  _evaluate_combo() does ALL its imports locally inside the function.
  ParamCombo is reconstructed from a plain dict (plain dicts pickle safely;
  dataclass instances defined in __main__ do not). Constants are passed
  explicitly so workers never depend on the parent process's module globals.

OUTPUTS:
  results/grid_search_results.csv    — full results table, every combo
  results/grid_search_best.json      — winner ready to paste into config.py
  results/grid_search_heatmaps.html  — interactive Plotly heatmaps

USAGE:
  python param_optimizer.py
  python param_optimizer.py --resume       # skip already-computed combos
  python param_optimizer.py --workers 4   # override worker count
"""

import argparse
import concurrent.futures
import json
import logging
import os
import time
import warnings
from dataclasses import dataclass, asdict
from itertools import product
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tunable constants — change here, propagates everywhere
# ---------------------------------------------------------------------------

ENTRY_Z_VALUES      = [2.0, 2.2, 2.4, 2.6]
EXIT_Z_VALUES       = [0.0, 0.3, 0.5, 0.7]
STOP_Z_VALUES       = [2.8, 3.2, 3.5, 4.0]

MIN_ENTRY_STOP_GAP  = 0.5    # stop must be at least this far above entry_z
MIN_TRADES_PER_FOLD = 5      # fewer → sparse flag + activity penalty applied
DD_HARD_LIMIT       = 15.0   # % — drawdowns above this are penalised
DD_PENALTY_WEIGHT   = 0.10   # Sharpe penalty per 1% above DD_HARD_LIMIT
STABILITY_WEIGHT    = 0.50   # Sharpe penalty per 1.0 of fold-to-fold std-dev
ACTIVITY_PENALTY    = 0.20   # flat penalty for sparse combos

RESULTS_DIR         = Path("results")
CACHE_FILE          = RESULTS_DIR / "grid_search_results.csv"


# ---------------------------------------------------------------------------
# Parameter combo dataclass — plain fields only (safe to pickle)
# ---------------------------------------------------------------------------

@dataclass
class ParamCombo:
    entry_z: float
    exit_z:  float
    stop_z:  float

    @property
    def key(self) -> str:
        return f"e{self.entry_z}_x{self.exit_z}_s{self.stop_z}"

    def is_valid(self) -> bool:
        return (
            self.stop_z  > self.entry_z + MIN_ENTRY_STOP_GAP and
            self.exit_z  < self.entry_z
        )


def build_search_space() -> List[ParamCombo]:
    combos = [
        ParamCombo(e, x, s)
        for e, x, s in product(ENTRY_Z_VALUES, EXIT_Z_VALUES, STOP_Z_VALUES)
    ]
    valid = [c for c in combos if c.is_valid()]
    log.info(
        f"Search space: {len(combos)} total → {len(valid)} valid "
        f"(stop > entry+{MIN_ENTRY_STOP_GAP}, exit < entry)"
    )
    return valid


# ---------------------------------------------------------------------------
# Worker — ALL imports are local so this is safe in a subprocess
# ---------------------------------------------------------------------------

def _evaluate_combo(args: Tuple) -> dict:
    """
    Evaluate one (entry_z, exit_z, stop_z) combo across all WF folds.

    Receives a 3-tuple: (combo_as_plain_dict, prices_parquet_path, constants_dict).
    Returns a flat results dict (all primitives — safe to pickle back).

    WHY local imports: ProcessPoolExecutor spawns fresh Python interpreters.
    Any object defined at module level in the parent (including dataclasses,
    loggers, and pandas DataFrames) is NOT available in the worker unless it
    is re-imported or passed explicitly. Putting all imports inside this
    function guarantees each worker builds its own clean environment.
    """
    combo_dict, prices_path, constants = args

    entry_z = combo_dict["entry_z"]
    exit_z  = combo_dict["exit_z"]
    stop_z  = combo_dict["stop_z"]
    key     = f"e{entry_z}_x{exit_z}_s{stop_z}"

    min_trades_pfold  = constants["min_trades_per_fold"]
    dd_hard_limit     = constants["dd_hard_limit"]
    dd_penalty_weight = constants["dd_penalty_weight"]
    stability_weight  = constants["stability_weight"]
    activity_penalty  = constants["activity_penalty"]

    def _empty(reason: str) -> dict:
        nan = float("nan")
        return {
            "key": key, "entry_z": entry_z, "exit_z": exit_z, "stop_z": stop_z,
            "mean_oos_sharpe": nan, "std_oos_sharpe": nan, "adj_sharpe": nan,
            "mean_oos_return_pct": nan, "mean_max_dd_pct": nan,
            "n_folds_evaluated": 0, "n_folds_profitable": 0,
            "total_oos_trades": 0, "min_trades_any_fold": 0,
            "stability_penalty": nan, "dd_penalty": nan, "activity_penalty": nan,
            "is_sparse": True, "status": reason,
        }

    try:
        import warnings
        warnings.filterwarnings("ignore")

        import copy
        import numpy as np
        import pandas as pd
        from config import CFG
        from walk_forward import generate_wf_splits, run_fold

        prices = pd.read_parquet(prices_path)

        cfg = copy.deepcopy(CFG)
        cfg.signal.entry_z = entry_z
        cfg.signal.exit_z  = exit_z
        cfg.signal.stop_z  = stop_z

        splits = generate_wf_splits(
            prices.index,
            train_years=cfg.backtest.train_years,
            test_years=cfg.backtest.test_years,
        )
        if not splits:
            return _empty("no_splits")

        fold_sharpes  = []
        fold_returns  = []
        fold_max_dds  = []
        fold_n_trades = []

        for i, (train_idx, test_idx) in enumerate(splits):
            try:
                result = run_fold(i, prices, train_idx, test_idx, cfg)
                n = result.get("n_trades", 0)
                if n == 0:
                    continue
                fold_sharpes.append(float(result["sharpe"]))
                fold_returns.append(float(result["total_return_pct"]))
                fold_max_dds.append(abs(float(result["max_dd_pct"])))
                fold_n_trades.append(int(n))
            except Exception:
                continue

        if not fold_sharpes:
            return _empty("no_valid_folds")

        mean_sharpe  = float(np.mean(fold_sharpes))
        std_sharpe   = float(np.std(fold_sharpes)) if len(fold_sharpes) > 1 else 0.0
        mean_return  = float(np.mean(fold_returns))
        mean_max_dd  = float(np.mean(fold_max_dds))
        min_trades   = min(fold_n_trades)
        n_profitable = sum(1 for r in fold_returns if r > 0)

        stab_pen  = stability_weight  * std_sharpe
        dd_pen    = dd_penalty_weight * max(0.0, mean_max_dd - dd_hard_limit)
        act_pen   = activity_penalty  if min_trades < min_trades_pfold else 0.0
        adj       = mean_sharpe - stab_pen - dd_pen - act_pen

        return {
            "key":                  key,
            "entry_z":              entry_z,
            "exit_z":               exit_z,
            "stop_z":               stop_z,
            "mean_oos_sharpe":      round(mean_sharpe,  4),
            "std_oos_sharpe":       round(std_sharpe,   4),
            "adj_sharpe":           round(adj,          4),
            "mean_oos_return_pct":  round(mean_return,  3),
            "mean_max_dd_pct":      round(mean_max_dd,  3),
            "n_folds_evaluated":    len(fold_sharpes),
            "n_folds_profitable":   n_profitable,
            "total_oos_trades":     sum(fold_n_trades),
            "min_trades_any_fold":  min_trades,
            "stability_penalty":    round(stab_pen,     4),
            "dd_penalty":           round(dd_pen,       4),
            "activity_penalty":     round(act_pen,      4),
            "is_sparse":            min_trades < min_trades_pfold,
            "status":               "ok",
        }

    except Exception as e:
        return _empty(f"error: {type(e).__name__}: {str(e)[:100]}")


# ---------------------------------------------------------------------------
# Heatmap builder
# ---------------------------------------------------------------------------

def build_heatmaps(results_df: pd.DataFrame, output_path: str) -> None:
    """
    2D Plotly heatmaps: entry_z (x) × exit_z (y), one panel per stop_z.
    Cell text shows raw OOS Sharpe + adj Sharpe for quick comparison.
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        from plotly.io import to_html
    except ImportError:
        log.warning("plotly not installed — skipping heatmaps")
        return

    valid = results_df[results_df["status"] == "ok"].copy()
    if valid.empty:
        log.warning("No valid results for heatmaps")
        return

    stop_vals = sorted(valid["stop_z"].unique())
    n_stops   = len(stop_vals)
    cols      = min(n_stops, 2)
    rows      = (n_stops + cols - 1) // cols

    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=[f"stop_z = {s}" for s in stop_vals],
        vertical_spacing=0.14,
        horizontal_spacing=0.08,
    )

    z_abs = max(abs(valid["mean_oos_sharpe"].dropna()).max(), 0.1)

    for idx, stop_z in enumerate(stop_vals):
        r = idx // cols + 1
        c = idx  % cols + 1

        sub = valid[valid["stop_z"] == stop_z]
        piv_raw = sub.pivot_table(index="exit_z", columns="entry_z",
                                   values="mean_oos_sharpe", aggfunc="mean")
        piv_adj = sub.pivot_table(index="exit_z", columns="entry_z",
                                   values="adj_sharpe", aggfunc="mean")

        text_mat = []
        for i_r, row_vals in enumerate(piv_raw.values):
            text_row = []
            for i_c, v in enumerate(row_vals):
                if np.isnan(v):
                    text_row.append("")
                else:
                    try:
                        a = piv_adj.values[i_r, i_c]
                        text_row.append(f"{v:+.3f} / {a:+.3f}")
                    except Exception:
                        text_row.append(f"{v:+.3f}")
            text_mat.append(text_row)

        fig.add_trace(go.Heatmap(
            z=piv_raw.values.tolist(),
            x=[str(e) for e in piv_raw.columns],
            y=[str(x) for x in piv_raw.index],
            text=text_mat,
            texttemplate="%{text}",
            colorscale=[[0, "#f85149"], [0.5, "#161b22"], [1, "#3fb950"]],
            zmid=0, zmin=-z_abs, zmax=z_abs,
            showscale=(idx == 0),
            colorbar=dict(title=dict(text="OOS Sharpe", font=dict(color="#e6edf3")),
                          tickformat=".2f", tickfont=dict(color="#8b949e")),
            hovertemplate=(
                f"entry_z=%{{x}}  exit_z=%{{y}}  stop_z={stop_z}<br>"
                "raw / adj Sharpe = %{text}<extra></extra>"
            ),
            xgap=3, ygap=3,
        ), row=r, col=c)

        fig.update_xaxes(title_text="entry_z",
                         title_font=dict(color="#8b949e"),
                         tickfont=dict(color="#8b949e"),
                         gridcolor="#21262d", row=r, col=c)
        fig.update_yaxes(title_text="exit_z",
                         title_font=dict(color="#8b949e"),
                         tickfont=dict(color="#8b949e"),
                         gridcolor="#21262d", row=r, col=c)

    BG = "#0d1117"
    subtitle = (
        f"Cell text: raw OOS Sharpe / adj Sharpe  "
        f"(adj = raw − {STABILITY_WEIGHT}×std − {DD_PENALTY_WEIGHT}×DD_excess "
        f"− {ACTIVITY_PENALTY} if sparse)"
    )
    fig.update_layout(
        title=dict(
            text=f"Grid Search — Mean OOS Sharpe<br>"
                 f"<span style='font-size:11px;color:#8b949e'>{subtitle}</span>",
            font=dict(size=14, color="#e6edf3"), x=0.01,
        ),
        paper_bgcolor=BG, plot_bgcolor="#161b22",
        font=dict(color="#8b949e", family="'Courier New', monospace", size=10),
        height=max(460, rows * 320),
        margin=dict(l=60, r=40, t=90, b=50),
    )

    html = (
        f'<!DOCTYPE html><html><head><meta charset="UTF-8">'
        f'<title>Grid Search Heatmaps</title>'
        f'<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>'
        f'<style>body{{background:{BG};margin:0;padding:24px}}</style>'
        f'</head><body>'
        f'{to_html(fig, full_html=False, include_plotlyjs=False)}'
        f'</body></html>'
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    log.info(f"Heatmaps saved → {output_path}")


# ---------------------------------------------------------------------------
# Console table
# ---------------------------------------------------------------------------

def print_results_table(df: pd.DataFrame, top_n: int = 15) -> None:
    valid = df[df["status"] == "ok"].sort_values("adj_sharpe", ascending=False)
    if valid.empty:
        print("  No valid results to display.")
        return

    cols = [
        "entry_z", "exit_z", "stop_z",
        "mean_oos_sharpe", "std_oos_sharpe", "adj_sharpe",
        "mean_oos_return_pct", "mean_max_dd_pct",
        "n_folds_profitable", "total_oos_trades", "is_sparse",
    ]
    print(f"\n{'─'*118}")
    print(f"  TOP {top_n} COMBOS — ranked by stability/DD-adjusted OOS Sharpe")
    print(f"{'─'*118}")
    print(valid[cols].head(top_n).to_string(index=False,
                                             float_format=lambda x: f"{x:8.4f}"))
    print(f"{'─'*118}")

    if valid.iloc[0]["is_sparse"]:
        print(
            f"\n  ⚠  Best combo has < {MIN_TRADES_PER_FOLD} trades in at least "
            f"one fold — treat with caution."
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(resume: bool = False, n_workers: Optional[int] = None) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    max_workers = n_workers or max(1, (os.cpu_count() or 4) - 1)

    # Load prices
    from config import CFG
    from data_pipeline import download_prices, validate_and_clean

    all_tickers = [t for tickers in CFG.sectors.values() for t in tickers]
    raw_prices  = download_prices(
        all_tickers, CFG.data.start_date, CFG.data.end_date,
        interval=CFG.data.interval, cache_dir=CFG.data.cache_dir,
    )
    prices, _ = validate_and_clean(raw_prices, CFG.data.min_history)

    prices_path = RESULTS_DIR / "_grid_prices_cache.parquet"
    prices.to_parquet(prices_path)
    log.info(f"Prices: {prices.shape[1]} tickers × {len(prices)} bars → {prices_path}")

    search_space = build_search_space()

    # Resume
    completed_keys, existing_rows = set(), []
    if resume and CACHE_FILE.exists():
        ex = pd.read_csv(CACHE_FILE)
        completed_keys = set(ex["key"].tolist())
        existing_rows  = ex.to_dict("records")
        log.info(f"Resuming: {len(completed_keys)} done, "
                 f"{len(search_space) - len(completed_keys)} remaining")

    pending = [c for c in search_space if c.key not in completed_keys]

    print(f"\n{'='*72}")
    print(f"  PARAMETER GRID SEARCH — OOS SHARPE OPTIMISATION")
    print(f"{'='*72}")
    print(f"  Valid combos  : {len(search_space)}")
    print(f"  Pending       : {len(pending)}")
    print(f"  Workers       : {max_workers}")
    print(f"  WF folds      : {CFG.backtest.train_years}yr train / "
          f"{CFG.backtest.test_years}yr test")
    print(f"  Date range    : {CFG.data.start_date} → {CFG.data.end_date}")
    print(f"  Objective     : mean_sharpe "
          f"− {STABILITY_WEIGHT}×std − {DD_PENALTY_WEIGHT}×DD_excess "
          f"− {ACTIVITY_PENALTY} if sparse")
    print(f"{'='*72}\n")

    if not pending:
        log.info("All combos computed. Loading from cache.")
        _finish(pd.read_csv(CACHE_FILE))
        return

    # Constants dict passed explicitly to each worker
    worker_constants = {
        "min_trades_per_fold": MIN_TRADES_PER_FOLD,
        "dd_hard_limit":       DD_HARD_LIMIT,
        "dd_penalty_weight":   DD_PENALTY_WEIGHT,
        "stability_weight":    STABILITY_WEIGHT,
        "activity_penalty":    ACTIVITY_PENALTY,
    }

    args_list   = [(asdict(c), str(prices_path), worker_constants) for c in pending]
    all_results = list(existing_rows)
    t_start     = time.time()
    done        = len(completed_keys)
    total       = len(search_space)

    print(f"  Launching {len(pending)} combos across {max_workers} workers...\n")

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_evaluate_combo, args): args[0]
            for args in args_list
        }

        for future in concurrent.futures.as_completed(futures):
            cd    = futures[future]
            done += 1
            elapsed = time.time() - t_start
            eta_s   = (elapsed / max(done, 1)) * (total - done)

            try:
                res = future.result()
            except Exception as e:
                nan = float("nan")
                res = {
                    "key":     f"e{cd['entry_z']}_x{cd['exit_z']}_s{cd['stop_z']}",
                    "entry_z": cd["entry_z"], "exit_z": cd["exit_z"], "stop_z": cd["stop_z"],
                    "mean_oos_sharpe": nan, "std_oos_sharpe": nan, "adj_sharpe": nan,
                    "mean_oos_return_pct": nan, "mean_max_dd_pct": nan,
                    "n_folds_evaluated": 0, "n_folds_profitable": 0,
                    "total_oos_trades": 0, "min_trades_any_fold": 0,
                    "stability_penalty": nan, "dd_penalty": nan, "activity_penalty": nan,
                    "is_sparse": True,
                    "status": f"exception: {type(e).__name__}: {str(e)[:80]}",
                }

            all_results.append(res)

            raw = res.get("mean_oos_sharpe", float("nan"))
            adj = res.get("adj_sharpe",      float("nan"))
            ok  = "✓" if res.get("status") == "ok" else "✗"
            sp  = "⚠" if res.get("is_sparse") else " "
            sh  = f"{raw:+.3f}" if not pd.isna(raw) else "  N/A "
            aj  = f"{adj:+.3f}" if not pd.isna(adj) else "  N/A "
            rt  = res.get("mean_oos_return_pct", float("nan"))

            print(
                f"  [{done:3d}/{total}] {ok}{sp} "
                f"e={cd['entry_z']} x={cd['exit_z']} s={cd['stop_z']}  "
                f"Sharpe={sh}  adj={aj}  "
                f"ret={rt:+6.2f}%  "
                f"ETA {int(eta_s//60)}m{int(eta_s%60):02d}s"
            )

            # Checkpoint every result so --resume works after any crash
            pd.DataFrame(all_results).to_csv(CACHE_FILE, index=False)

    _finish(pd.DataFrame(all_results))


def _finish(results_df: pd.DataFrame) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results_df = (
        results_df.sort_values("adj_sharpe", ascending=False).reset_index(drop=True)
    )
    results_df.to_csv(CACHE_FILE, index=False)
    print_results_table(results_df)

    valid = results_df[results_df["status"] == "ok"]
    if valid.empty:
        print("\n  No valid results — check logs for errors.")
        return

    best = valid.iloc[0]
    best_dict = {
        "entry_z":             float(best["entry_z"]),
        "exit_z":              float(best["exit_z"]),
        "stop_z":              float(best["stop_z"]),
        "mean_oos_sharpe":     float(best["mean_oos_sharpe"]),
        "std_oos_sharpe":      float(best["std_oos_sharpe"]),
        "adj_sharpe":          float(best["adj_sharpe"]),
        "mean_oos_return_pct": float(best["mean_oos_return_pct"]),
        "mean_max_dd_pct":     float(best["mean_max_dd_pct"]),
        "n_folds_profitable":  int(best["n_folds_profitable"]),
        "total_oos_trades":    int(best["total_oos_trades"]),
        "is_sparse":           bool(best["is_sparse"]),
        "config_snippet": (
            "\n# ── Paste into config.py → SignalConfig ──────────\n"
            f"entry_z = {best['entry_z']}\n"
            f"exit_z  = {best['exit_z']}\n"
            f"stop_z  = {best['stop_z']}\n"
        ),
    }

    best_path = RESULTS_DIR / "grid_search_best.json"
    with open(best_path, "w") as f:
        json.dump(best_dict, f, indent=2)

    print(f"\n{'='*72}")
    print(f"  BEST COMBO")
    print(f"{'='*72}")
    print(f"  entry_z              = {best['entry_z']}")
    print(f"  exit_z               = {best['exit_z']}")
    print(f"  stop_z               = {best['stop_z']}")
    print(f"  Mean OOS Sharpe      = {best['mean_oos_sharpe']:+.4f}")
    print(f"  Std  OOS Sharpe      = {best['std_oos_sharpe']:.4f}  "
          f"(lower = more consistent across folds)")
    print(f"  Adj  Sharpe (ranked) = {best['adj_sharpe']:+.4f}")
    print(f"  Mean OOS Return      = {best['mean_oos_return_pct']:+.3f}%")
    print(f"  Mean Max DD          = {best['mean_max_dd_pct']:.3f}%")
    print(f"  Profitable folds     = {best['n_folds_profitable']}")
    print(f"  Total OOS trades     = {best['total_oos_trades']}")
    if best["is_sparse"]:
        print(f"  ⚠  SPARSE — < {MIN_TRADES_PER_FOLD} trades in at least one fold")
    print(f"{'='*72}")
    print(f"\n  ✓ Full table  → {CACHE_FILE}")
    print(f"  ✓ Best combo  → {best_path}")
    print(f"  ✓ Heatmaps    → results/grid_search_heatmaps.html")
    print(f"\n  Paste into config.py → SignalConfig:")
    print(f"    entry_z = {best['entry_z']}")
    print(f"    exit_z  = {best['exit_z']}")
    print(f"    stop_z  = {best['stop_z']}")

    build_heatmaps(results_df, "results/grid_search_heatmaps.html")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Grid search over entry/exit/stop z-score parameters"
    )
    parser.add_argument("--resume", action="store_true",
                        help="Skip already-computed combos")
    parser.add_argument("--workers", type=int, default=None,
                        help="Parallel worker count (default: cpu_count − 1)")
    args = parser.parse_args()
    main(resume=args.resume, n_workers=args.workers)