"""
dashboard.py — Interactive Plotly performance dashboard.

Generates a single self-contained HTML file with all charts.
Open it in any browser — no server required.

Charts:
  1. Equity curve (IS + OOS if available) vs buy-and-hold SPY
  2. Drawdown profile
  3. Per-pair P&L waterfall
  4. Trade P&L distribution (histogram)
  5. Rolling Sharpe ratio (63-day window)
  6. Spread + z-score for the best pair
  7. Walk-forward fold performance (if available)
  8. Monthly returns heatmap
"""

import logging
import os
import warnings
from pathlib import Path
from typing import Optional
from config import CFG

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Colour palette — dark terminal aesthetic
# ---------------------------------------------------------------------------
BG_DARK    = "#0d1117"
BG_PANEL   = "#161b22"
BG_BORDER  = "#30363d"
TEXT_PRI   = "#e6edf3"
TEXT_MUT   = "#8b949e"
GREEN      = "#3fb950"
RED        = "#f85149"
BLUE       = "#58a6ff"
PURPLE     = "#bc8cff"
AMBER      = "#e3b341"
TEAL       = "#39d353"
GRID       = "#21262d"


def _layout_defaults(title: str = "") -> dict:
    return dict(
        title=dict(text=title, font=dict(size=14, color=TEXT_PRI), x=0.01),
        paper_bgcolor=BG_DARK,
        plot_bgcolor=BG_PANEL,
        font=dict(color=TEXT_MUT, family="'Courier New', monospace", size=11),
        legend=dict(
            bgcolor=BG_PANEL, bordercolor=BG_BORDER, borderwidth=1,
            font=dict(size=10, color=TEXT_PRI),
        ),
        margin=dict(l=55, r=20, t=45, b=45),
        xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, showgrid=True),
        yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, showgrid=True),
    )


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------

def chart_equity_curve(
    equity_curve: pd.DataFrame,
    wf_equity: Optional[pd.Series] = None,
    initial_capital: float = 100_000,
) -> go.Figure:
    """Equity curve — always shows OOS stitched curve rebased to $100k start."""
    fig = go.Figure()
    eq = equity_curve["equity"]

    # Rebase so curve always starts exactly at initial_capital
    eq_rebased = (eq / float(eq.iloc[0])) * initial_capital

    fig.add_trace(go.Scatter(
        x=eq_rebased.index, y=eq_rebased,
        name="Strategy (OOS)",
        line=dict(color=GREEN, width=2.5),
        hovertemplate="$%{y:,.0f}<extra>OOS Walk-Forward</extra>",
    ))

    fig.add_hline(
        y=initial_capital, line_color=BG_BORDER, line_width=1, line_dash="dash",
        annotation_text="Starting capital", annotation_font_color=TEXT_MUT,
    )

    y_min = float(eq_rebased.min())
    y_max = float(eq_rebased.max())
    pad   = max((y_max - y_min) * 0.15, initial_capital * 0.05)

    fig.update_layout(**_layout_defaults("Equity Curve (OOS Walk-Forward)"))
    fig.update_layout(yaxis=dict(
        tickformat="$,.0f", gridcolor=GRID,
        range=[min(y_min, initial_capital) - pad,
               max(y_max, initial_capital) + pad],
        title=dict(text="Portfolio Value", font=dict(color=TEXT_MUT)),
    ))
    return fig


def chart_drawdown(equity_curve: pd.DataFrame) -> go.Figure:
    """Underwater equity chart."""
    eq = equity_curve["equity"]
    # Rebase to 100k so fold-boundary discontinuities don't corrupt cummax
    eq = eq / float(eq.iloc[0]) * 100_000
    rolling_max = eq.cummax()
    dd = ((eq - rolling_max) / rolling_max * 100).clip(lower=-100)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dd.index, y=dd, fill="tozeroy", name="Drawdown",
                             line=dict(color=RED, width=1), fillcolor="rgba(248,81,73,0.15)",
                             hovertemplate="%{y:.2f}%<extra>Drawdown</extra>"))
    max_dd = dd.min()  # most negative value
    y_floor = min(max_dd * 1.2, max_dd - 2)  # 20% headroom below worst DD
    fig.update_layout(**_layout_defaults("Drawdown Profile"))
    fig.update_layout(yaxis=dict(
        ticksuffix="%", gridcolor=GRID,
        range=[y_floor, 0.5],            # cap at 0, show floor
        title=dict(text="Drawdown %", font=dict(color=TEXT_MUT)),
    ))
    return fig


def chart_pair_pnl(pair_pnl: pd.DataFrame) -> go.Figure:
    """Net P&L per pair — horizontal bar chart (easier to read long ticker names)."""
    if pair_pnl.empty:
        return go.Figure().update_layout(**_layout_defaults("Net P&L by Pair"))

    df = pair_pnl.reset_index().sort_values("total_pnl")   # ascending = worst at top
    colours   = [GREEN if v >= 0 else RED for v in df["total_pnl"]]
    bar_text  = df["total_pnl"].apply(lambda v: f"${v:+,.0f}")

    fig = go.Figure(go.Bar(
        x=df["total_pnl"],
        y=df["pair"],
        orientation="h",
        marker_color=colours,
        text=bar_text,
        textposition="outside",
        textfont=dict(size=11, color=TEXT_PRI),
        hovertemplate="%{y}: $%{x:+,.0f}<br>Trades: %{customdata}<extra></extra>",
        customdata=df["n_trades"],
        width=0.55,
    ))

    fig.add_vline(x=0, line_color=BG_BORDER, line_width=1)

    x_abs = max(abs(df["total_pnl"].max()), abs(df["total_pnl"].min()), 500)
    fig.update_layout(**_layout_defaults("Net P&L by Pair"), showlegend=False)
    fig.update_layout(
        xaxis=dict(
            tickformat="$,.0f", gridcolor=GRID,
            range=[-(x_abs * 1.5), x_abs * 1.6],   # extra right room for outside labels
            title=dict(text="Net P&L ($)", font=dict(color=TEXT_MUT)),
        ),
        yaxis=dict(gridcolor=GRID, automargin=True),   # automargin prevents label clipping
        margin=dict(l=90, r=30, t=45, b=45),           # explicit left margin for pair names
    )
    return fig


def chart_trade_distribution(trades: pd.DataFrame) -> go.Figure:
    """
    Trade P&L histogram with win/loss split colouring.

    Edge case: when all trades cluster tightly (e.g. $795-$799) the x-axis shows
    labels like "$795", "$796" which look broken. Fix: compute bin size from the
    actual range and enforce a minimum of $50/bin so the axis always looks sensible.
    """
    if trades.empty:
        return go.Figure().update_layout(**_layout_defaults("Trade P&L Distribution"))

    pnl = trades["net_pnl"]
    rng = pnl.max() - pnl.min()
    # Aim for ~20 bins but never smaller than $50 wide
    bin_size = max(50.0, rng / 20) if rng > 0 else 100.0

    wins   = pnl[pnl >= 0]
    losses = pnl[pnl <  0]

    fig = go.Figure()
    if not wins.empty:
        fig.add_trace(go.Histogram(
            x=wins, name="Win", xbins=dict(size=bin_size),
            marker_color=GREEN, opacity=0.75,
        ))
    if not losses.empty:
        fig.add_trace(go.Histogram(
            x=losses, name="Loss", xbins=dict(size=bin_size),
            marker_color=RED, opacity=0.75,
        ))

    mean_pnl = float(pnl.mean())
    fig.add_vline(x=mean_pnl, line_color=AMBER, line_width=1.5, line_dash="dash",
                  annotation_text=f"Mean {mean_pnl:+,.0f}",
                  annotation_font_color=AMBER,
                  annotation_position="top right")

    fig.update_layout(
        **_layout_defaults("Trade P&L Distribution"),
        barmode="overlay",
        showlegend=True,
    )
    # Enforce a minimum x-axis span of $500 so tightly clustered trades
    # don't compress the axis into an unreadable single column
    x_span = max(500.0, rng * 1.5)
    x_center = float(pnl.mean())
    fig.update_layout(
        yaxis=dict(gridcolor=GRID, title=dict(text="# Trades", font=dict(color=TEXT_MUT))),
        xaxis=dict(
            tickprefix="$", tickformat=",.0f", gridcolor=GRID,
            range=[x_center - x_span / 2, x_center + x_span / 2],
            title=dict(text="Net P&L per Trade ($)", font=dict(color=TEXT_MUT)),
        ),
    )
    return fig


def chart_rolling_sharpe(equity_curve: pd.DataFrame, window: int = 63) -> go.Figure:
    """Rolling Sharpe ratio."""
    returns = equity_curve["equity"].pct_change().dropna()
    # Replace 0 std (cash days) with 1e-9 so we get 0 Sharpe instead of NaN/inf
    roll_std = returns.rolling(window).std().replace(0, 1e-9)
    roll_sharpe = (returns.rolling(window).mean() / roll_std) * np.sqrt(252)
    # Clamp to [-5, 5] and fill any residual NaNs (start of window) with 0
    roll_sharpe = roll_sharpe.clip(-5, 5).fillna(0)
    fig = go.Figure()
    fig.add_hline(y=0,    line_color=BG_BORDER, line_width=1)
    fig.add_hline(y=1.0,  line_color=GREEN, line_width=1, line_dash="dot",
                  annotation_text="Sharpe = 1.0", annotation_font_color=GREEN)
    fig.add_hline(y=-1.0, line_color=RED,   line_width=1, line_dash="dot")
    fig.add_trace(go.Scatter(x=roll_sharpe.index, y=roll_sharpe,
                             name=f"Rolling {window}d Sharpe",
                             line=dict(color=PURPLE, width=1.5),
                             hovertemplate="Sharpe: %{y:.2f}<extra></extra>"))
    fig.update_layout(**_layout_defaults(f"Rolling {window}-Day Sharpe Ratio"))
    fig.update_layout(yaxis=dict(
        gridcolor=GRID, range=[-3, 3],
        title=dict(text="Sharpe", font=dict(color=TEXT_MUT)),
    ))
    return fig

def chart_spread_zscore(
    spread_df: pd.DataFrame,
    pair_name: str,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    stop_z: float = 3.5,
    signal_df: Optional[pd.DataFrame] = None,
) -> go.Figure:
    """Spread z-score chart with trade markers."""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.4, 0.6],
        subplot_titles=["Log Prices", f"Z-Score — {pair_name}"],
        vertical_spacing=0.08,
    )

    ty, tx = pair_name.split("/")

    # Log prices
    fig.add_trace(go.Scatter(
        x=spread_df.index, y=spread_df["log_y"],
        name=ty, line=dict(color=BLUE, width=1.3),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=spread_df.index, y=spread_df["log_x"],
        name=tx, line=dict(color=AMBER, width=1.3),
    ), row=1, col=1)

    # Z-score — clamp display to [-6, 6] so non-stationary pairs don't destroy the axis
    z_clamped = spread_df["zscore_smoothed"].clip(-6, 6)
    fig.add_trace(go.Scatter(
        x=spread_df.index, y=z_clamped,
        name="Z-score", line=dict(color=PURPLE, width=1.5),
    ), row=2, col=1)

    # Threshold bands (filled)
    for z_val, col in [(entry_z, "rgba(63,185,80,0.08)"), (stop_z, "rgba(248,81,73,0.06)")]:
        fig.add_hrect(y0=exit_z,  y1=z_val,  fillcolor=col, line_width=0, row=2, col=1)
        fig.add_hrect(y0=-z_val, y1=-exit_z, fillcolor=col, line_width=0, row=2, col=1)

    # Threshold lines
    for z_val, color in [(entry_z, GREEN), (-entry_z, GREEN), (0, TEXT_MUT),
                          (exit_z, AMBER), (-exit_z, AMBER),
                          (stop_z, RED), (-stop_z, RED)]:
        fig.add_hline(y=z_val, line_color=color, line_width=0.8,
                      line_dash="dot", row=2, col=1)

    # Trade entry/exit markers (use clamped z-score for y position)
    if signal_df is not None:
        opens  = signal_df[signal_df["trade_open"]].copy()
        closes = signal_df[signal_df["trade_close"]].copy()
        opens["z_plot"]  = opens["zscore_smoothed"].clip(-6, 6)
        closes["z_plot"] = closes["zscore_smoothed"].clip(-6, 6)

        if not opens.empty:
            fig.add_trace(go.Scatter(
                x=opens.index, y=opens["z_plot"],
                mode="markers",
                marker=dict(symbol="triangle-up", size=10, color=GREEN, line_width=0),
                name="Entry", showlegend=True,
            ), row=2, col=1)

        if not closes.empty:
            fig.add_trace(go.Scatter(
                x=closes.index, y=closes["z_plot"],
                mode="markers",
                marker=dict(symbol="triangle-down", size=10, color=RED, line_width=0),
                name="Exit", showlegend=True,
            ), row=2, col=1)

    fig.update_layout(
        paper_bgcolor=BG_DARK, plot_bgcolor=BG_PANEL,
        font=dict(color=TEXT_MUT, family="'Courier New', monospace", size=11),
        legend=dict(bgcolor=BG_PANEL, bordercolor=BG_BORDER, borderwidth=1,
                    font=dict(size=10, color=TEXT_PRI)),
        margin=dict(l=55, r=20, t=45, b=45),
        height=520,
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID)
    # Clamp z-score subplot to [-6, 6]
    fig.update_yaxes(range=[-6, 6], row=2, col=1)
    return fig


def chart_wf_folds(fold_summary: pd.DataFrame) -> go.Figure:
    """
    Walk-forward OOS return per fold with Sharpe annotation.

    Single-fold edge case: a single bar with no neighbours renders as a hairline.
    Fix: set bargap=0.4 and enforce a minimum bar width via width param.
    """
    if fold_summary.empty or fold_summary["n_trades"].sum() == 0:
        return go.Figure().update_layout(**_layout_defaults("Walk-Forward OOS Returns by Fold"))

    valid = fold_summary[fold_summary["n_trades"] > 0].copy()
    valid["fold_label"] = valid.apply(
        lambda r: f"Fold {int(r['fold'])}<br>{r['test_start'][:7]} → {r['test_end'][:7]}", axis=1
    )
    colours = [GREEN if v >= 0 else RED for v in valid["total_return_pct"]]
    bar_text = valid.apply(
        lambda r: f"{r['total_return_pct']:+.2f}%<br>Sharpe {r['sharpe']:.2f}", axis=1
    )

    fig = go.Figure(go.Bar(
        x=valid["fold_label"],
        y=valid["total_return_pct"],
        marker_color=colours,
        text=bar_text,
        texttemplate="%{text}",      # renders bar_text on the bar
        textposition="outside",
        textfont=dict(size=10, color=TEXT_PRI),
        width=0.5,
        customdata=valid[["sharpe", "max_dd_pct", "n_trades"]].values,
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Return: %{y:+.2f}%<br>"
            "Sharpe: %{customdata[0]:.3f}<br>"
            "Max DD: %{customdata[1]:.2f}%<br>"
            "Trades: %{customdata[2]}<extra></extra>"
        ),
    ))

    fig.add_hline(y=0, line_color=BG_BORDER, line_width=1)

    # Shade the zero line region so positive/negative is immediately obvious
    y_abs = max(abs(valid["total_return_pct"].max()), abs(valid["total_return_pct"].min()), 1.0)
    fig.update_layout(**_layout_defaults("Walk-Forward OOS Returns by Fold"), showlegend=False)
    fig.update_layout(
        bargap=0.4,
        yaxis=dict(
            ticksuffix="%", gridcolor=GRID,
            range=[-(y_abs * 1.5), y_abs * 1.8],   # room for outside labels
            title=dict(text="OOS Return %", font=dict(color=TEXT_MUT)),
        ),
    )
    return fig

def chart_monthly_returns(equity_curve: pd.DataFrame) -> go.Figure:
    """
    Monthly returns heatmap — Year × Month grid with Annual totals column.

    Final rendering fix: go.Heatmap with a categorical y-axis silently drops
    np.nan cells — the entire row becomes invisible because Plotly cannot map
    NaN through its categorical colour lookup. Replacing np.nan with Python
    None tells Plotly to render the cell as transparent (no colour, no text)
    which is exactly the correct behaviour for months with no data.

    The text array uses "" for empty cells so texttemplate="%{text}" renders
    nothing rather than "None" or "nan".
    """
    monthly = equity_curve["equity"].resample("ME").last().pct_change().dropna() * 100
    monthly.index = pd.to_datetime(monthly.index)

    if monthly.empty:
        return go.Figure().update_layout(**_layout_defaults("Monthly Returns Heatmap"))

    col_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec", "Ann."]

    years   = sorted(monthly.index.year.unique())
    n_years = len(years)

    # Build as plain Python lists of None (not np.nan) for Plotly compatibility
    matrix = [[None] * 13 for _ in range(n_years)]
    text   = [[""]   * 13 for _ in range(n_years)]

    all_vals = []   # collect for colorscale calibration

    for i, year in enumerate(years):
        year_vals = []
        for j in range(12):
            mask = (monthly.index.year == year) & (monthly.index.month == j + 1)
            if mask.any():
                v = float(monthly[mask].iloc[0])
                matrix[i][j] = v
                text[i][j]   = f"{v:+.1f}%"
                year_vals.append(v)
                all_vals.append(v)
        if year_vals:
            ann = (np.prod([1 + r / 100 for r in year_vals]) - 1) * 100
            matrix[i][12] = ann
            text[i][12]   = f"{ann:+.1f}%"
            all_vals.append(ann)

    # Cap colorscale at ±5% so a single outlier month doesn't flatten everything else
    z_max = min(5.0, max(abs(v) for v in all_vals)) if all_vals else 5.0

    year_labels = [str(y) for y in years]

    fig = go.Figure(go.Heatmap(
        z=matrix,                           # Python list with None for missing cells
        x=col_labels,
        y=year_labels,
        text=text,
        texttemplate="%{text}",             # renders "" as blank, not "None"
        colorscale=[[0.0, RED], [0.5, BG_PANEL], [1.0, GREEN]],
        zmid=0, zmin=-z_max, zmax=z_max,
        showscale=False,
        hovertemplate="%{y} %{x}: %{text}<extra></extra>",
        textfont=dict(size=11, color=TEXT_PRI),
        xgap=2, ygap=2,
    ))

    # Vertical separator before the Ann. column
    fig.add_shape(type="line",
        x0=11.5, x1=11.5, y0=-0.5, y1=n_years - 0.5,
        line=dict(color=BG_BORDER, width=2),
    )

    fig.update_layout(
        **_layout_defaults("Monthly Returns Heatmap"),
        height=max(160, n_years * 48 + 90),
    )
    fig.update_layout(
        yaxis=dict(
            type="category",
            categoryorder="array",
            categoryarray=year_labels,
            autorange="reversed",       # most recent year at top (Bloomberg style)
            gridcolor=GRID,
            title=dict(text="Year", font=dict(color=TEXT_MUT)),
        ),
        xaxis=dict(
            type="category",            # force string labels — prevents numeric index bug
            categoryorder="array",
            categoryarray=col_labels,   # explicit order locks Jan–Dec–Ann. sequence
            gridcolor=GRID,
            side="top",
        ),
    )
    return fig


# ---------------------------------------------------------------------------
# Metrics summary card (HTML)
# ---------------------------------------------------------------------------

def _metrics_card(metrics: dict, n_trades: int) -> str:
    def fmt(k, v):
        if "pct" in k or "return" in k.lower() or "vol" in k.lower() or "dd" in k.lower():
            color = GREEN if v >= 0 else RED
            return f'<span style="color:{color}">{v:+.2f}%</span>'
        elif "ratio" in k or "sharpe" in k or "sortino" in k or "calmar" in k:
            color = GREEN if v >= 0.5 else (AMBER if v >= 0 else RED)
            return f'<span style="color:{color}">{v:.3f}</span>'
        else:
            return f'<span style="color:{TEXT_PRI}">{v}</span>'

    rows = [
        ("Total Return",    metrics.get("total_return_pct",  0), "total_return_pct"),
        ("Ann. Return",     metrics.get("ann_return_pct",    0), "ann_return_pct"),
        ("Ann. Volatility", metrics.get("ann_vol_pct",       0), "ann_vol_pct"),
        ("Sharpe Ratio",    metrics.get("sharpe_ratio",      0), "sharpe_ratio"),
        ("Sortino Ratio",   metrics.get("sortino_ratio",     0), "sortino_ratio"),
        ("Calmar Ratio",    metrics.get("calmar_ratio",      0), "calmar_ratio"),
        ("Max Drawdown",    metrics.get("max_drawdown_pct",  0), "max_drawdown_pct"),
        ("Win Days",        metrics.get("win_days_pct",      0), "win_days_pct"),
        ("Total Trades",    n_trades,                            "n"),
    ]

    html_rows = "\n".join(
        f'<tr><td style="color:{TEXT_MUT};padding:4px 12px 4px 0">{label}</td>'
        f'<td style="text-align:right;padding:4px 0">{fmt(key, val)}</td></tr>'
        for label, val, key in rows
    )

    return f"""
<div style="background:{BG_PANEL};border:1px solid {BG_BORDER};border-radius:6px;
            padding:16px 20px;font-family:'Courier New',monospace;
            display:inline-block;min-width:280px">
  <div style="color:{TEXT_PRI};font-size:13px;font-weight:bold;margin-bottom:10px;
              border-bottom:1px solid {BG_BORDER};padding-bottom:6px">
    PERFORMANCE SUMMARY
  </div>
  <table style="font-size:12px;border-collapse:collapse;width:100%">
    {html_rows}
  </table>
</div>"""


# ---------------------------------------------------------------------------
# Master dashboard builder
# ---------------------------------------------------------------------------

def build_dashboard(
    equity_curve:     pd.DataFrame,
    trades:           pd.DataFrame,
    pair_pnl:         pd.DataFrame,
    metrics:          dict,
    spread_data:      dict,
    signal_data:      dict,
    fold_summary:     Optional[pd.DataFrame] = None,
    wf_equity:        Optional[pd.Series]    = None,
    output_path:      str = "results/dashboard.html",
) -> str:
    """
    Build the full dashboard and save as a self-contained HTML file.
    Returns the output path.
    """
    from plotly.io import to_html

    # Pick best pair for spread chart
    best_pair = pair_pnl.index[0] if not pair_pnl.empty else (
        list(spread_data.keys())[0] if spread_data else None
    )

    charts = {}

    charts["equity"]      = chart_equity_curve(equity_curve)
    charts["drawdown"]    = chart_drawdown(equity_curve)      # IS: continuous curve needed
    charts["pair_pnl"]    = chart_pair_pnl(pair_pnl)
    charts["trade_dist"]  = chart_trade_distribution(trades)
    charts["monthly"]     = chart_monthly_returns(equity_curve)  # IS: full date range

    if best_pair and best_pair in spread_data:
        charts["spread"]  = chart_spread_zscore(
            spread_data[best_pair],
            best_pair,
            signal_df=signal_data.get(best_pair),
        )

    if fold_summary is not None and not fold_summary.empty:
        charts["wf_folds"] = chart_wf_folds(fold_summary)

    # Convert charts to HTML div strings
    def to_div(fig, height=380):
        fig.update_layout(height=height)
        return to_html(fig, full_html=False, include_plotlyjs=False, config={
            "displayModeBar": True,
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
            "displaylogo": False,
        })

    n_trades_display = metrics.get("n_oos_trades", len(trades))
    metrics_html = _metrics_card(metrics, n_trades_display)

    # Layout
    panel_css = (
        f"background:{BG_PANEL};border:1px solid {BG_BORDER};"
        "border-radius:6px;padding:12px;margin-bottom:16px"
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>StatArb Engine — Performance Dashboard</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: {BG_DARK}; color: {TEXT_PRI}; font-family: 'Courier New', monospace;
         font-size: 13px; padding: 24px; }}
  h1 {{ font-size: 18px; color: {TEXT_PRI}; margin-bottom: 4px; font-weight: bold; }}
  .subtitle {{ color: {TEXT_MUT}; font-size: 11px; margin-bottom: 24px; }}
  .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
  .grid-3 {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; }}
  .panel {{ {panel_css} }}
  .full {{ grid-column: 1 / -1; }}
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 3px;
            font-size: 10px; font-weight: bold; background: {BG_BORDER}; color: {TEXT_MUT}; }}
  .badge.green {{ background: rgba(63,185,80,0.15); color: {GREEN}; }}
  .badge.red   {{ background: rgba(248,81,73,0.15); color: {RED}; }}
</style>
</head>
<body>
<h1>STATISTICAL ARBITRAGE ENGINE</h1>
<div class="subtitle">
  S&amp;P 500 Pairs Trading · Event-Driven Backtest · Walk-Forward Validated
</div>

<div class="grid-2" style="margin-bottom:16px">
  <div>{metrics_html}</div>
  <div class="panel" style="align-self:start">
    <div style="color:{TEXT_PRI};font-size:12px;font-weight:bold;margin-bottom:8px">ACTIVE PAIRS</div>
    {''.join(f'<div style="color:{TEXT_MUT};font-size:11px;padding:2px 0">{p}</div>' for p in pair_pnl.index.tolist()) if not pair_pnl.empty else '<div style="color:{TEXT_MUT}">No data</div>'}
  </div>
</div>

<div class="grid-2">
  <div class="panel full">{to_div(charts['equity'], 320)}</div>
  <div class="panel full">{to_div(charts['drawdown'], 220)}</div>

  <div class="panel">{to_div(charts['pair_pnl'], 300)}</div>
  <div class="panel">{to_div(charts['trade_dist'], 300)}</div>

  {'<div class="panel full">' + to_div(charts['spread'], 520) + '</div>' if 'spread' in charts else ''}
  {'<div class="panel full">' + to_div(charts['wf_folds'], 280) + '</div>' if 'wf_folds' in charts else ''}

  <div class="panel full">{to_div(charts['monthly'], 220)}</div>
</div>

<div style="color:{TEXT_MUT};font-size:10px;margin-top:16px;text-align:center">
  Statistical Arbitrage Engine · Built with Python, statsmodels, Plotly
</div>
</body>
</html>"""

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    log.info(f"Dashboard saved to {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Main — wire everything together
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from data_pipeline import download_prices, validate_and_clean
    from signal_generator import run_all_signals
    from backtester import PairsBacktester
    from walk_forward import run_walk_forward
    from spread_model import compute_spread
    from risk_manager import portfolio_risk_report

    cfg = CFG
    os.makedirs("results", exist_ok=True)

    print("\n" + "="*60)
    print("  DASHBOARD — Walk-Forward OOS Primary Display")
    print("="*60)

    # Load prices
    all_tickers = [t for tickers in cfg.sectors.values() for t in tickers]
    raw_prices  = download_prices(all_tickers, cfg.data.start_date, cfg.data.end_date)
    prices, _   = validate_and_clean(raw_prices, cfg.data.min_history)

    # Load pre-saved IS results from disk (avoids re-running the slow backtester)
    # If files don't exist, run a quick IS backtest to generate spread/signal data
    ranked_pairs   = pd.read_csv("results/ranked_pairs.csv")
    spread_quality = pd.read_csv("results/spread_quality.csv")

    print("\nBuilding signals for spread chart...")
    signal_data, _, signal_stats_df = run_all_signals(
        prices, ranked_pairs, cfg.signal, top_n=10
    )

    # Try loading pre-saved IS backtest results; regenerate if missing
    try:
        is_equity   = pd.read_csv("results/equity_curve.csv", index_col=0, parse_dates=True)
        is_trades   = pd.read_csv("results/backtest_trades.csv")
        is_pair_pnl = pd.read_csv("results/pair_pnl.csv", index_col=0)
        is_metrics  = pd.read_csv("results/metrics.csv").iloc[0].to_dict()
        print("Loaded IS results from disk.")
    except FileNotFoundError:
        print("IS results not found — running IS backtest...")
        bt = PairsBacktester(
            prices=prices,
            signal_data=signal_data,
            ranked_pairs=ranked_pairs,
            signal_stats=signal_stats_df,
            spread_quality=spread_quality,
            cfg=cfg.risk,
        )
        is_res      = bt.run()
        is_equity   = is_res["equity_curve"]
        is_trades   = is_res["trades"]
        is_pair_pnl = is_res["pair_pnl"]
        is_metrics  = is_res["metrics"]

    # Walk-forward — primary OOS results
    print("\nRunning walk-forward validation...")
    wf_results   = run_walk_forward(prices, cfg)
    fold_summary = wf_results.get("fold_summary")
    wf_equity    = wf_results.get("combined_equity")    # pd.Series, OOS only
    wf_agg       = wf_results.get("aggregate", {})

    if wf_agg:
        print(f"\nWalk-Forward Summary:")
        print(f"  Profitable folds : {wf_agg['n_folds_profitable']} / {wf_agg['n_folds']}")
        print(f"  Mean OOS Sharpe  : {wf_agg['mean_sharpe']:.3f}")
        print(f"  Mean OOS return  : {wf_agg['mean_return_pct']:+.2f}%")
        print(f"  Total OOS trades : {wf_agg['total_oos_trades']}")

    # Determine if we have usable OOS data
    has_oos = (
        wf_equity is not None
        and len(wf_equity) > 10
        and wf_agg.get("n_folds_profitable", 0) > 0
    )

    if has_oos:
        # OOS equity — Series → DataFrame with "equity" column
        oos_equity_df = wf_equity.rename("equity").to_frame()

        # Combine fold trades into one DataFrame
        fold_trade_list = [
            t for t in wf_results.get("fold_trades", [])
            if isinstance(t, pd.DataFrame) and not t.empty
        ]
        all_fold_trades = (
            pd.concat(fold_trade_list, ignore_index=True)
            if fold_trade_list else pd.DataFrame()
        )

        # OOS pair P&L
        if not all_fold_trades.empty and "net_pnl" in all_fold_trades.columns:
            oos_pair_pnl = (
                all_fold_trades.groupby("pair")["net_pnl"]
                .agg(["sum", "count", "mean"])
                .rename(columns={"sum": "total_pnl", "count": "n_trades", "mean": "avg_pnl"})
                .sort_values("total_pnl", ascending=False)
                .round(2)
            )
        else:
            oos_pair_pnl = is_pair_pnl

        # Compute OOS metrics from the stitched OOS equity curve
        oos_metrics = portfolio_risk_report(oos_equity_df["equity"])
        # Override n_trades with actual OOS trade count (more meaningful)
        n_oos_trades = int(wf_agg.get("total_oos_trades", len(all_fold_trades)))

        # The equity chart shows IS curve as context + OOS as the headline overlay
        # Pass IS equity as equity_curve (shown in blue), OOS as wf_equity (dotted green)
        display_equity   = oos_equity_df      # IS: full 6-year context
        display_trades   = all_fold_trades if not all_fold_trades.empty else is_trades
        display_pair_pnl = oos_pair_pnl
        display_metrics  = oos_metrics    # HEADLINE: OOS metrics in summary card
        display_n_trades = n_oos_trades
        display_label    = "OOS Walk-Forward"
        print("\nDashboard: headline = OOS metrics | equity chart = IS(blue) + OOS(green)")
    else:
        display_equity   = is_equity
        display_trades   = is_trades
        display_pair_pnl = is_pair_pnl
        display_metrics  = is_metrics
        display_n_trades = len(is_trades)
        display_label    = "IS In-Sample (no OOS data)"
        print("\nWARNING: No OOS equity — showing IS results")

    # Spread data for best-pair chart
    spread_data = {}
    for _, row in ranked_pairs.head(5).iterrows():
        ty, tx = row["ticker_y"], row["ticker_x"]
        pn = f"{ty}/{tx}"
        if ty in prices.columns and tx in prices.columns:
            spread_data[pn] = compute_spread(
                prices[ty], prices[tx],
                window=cfg.signal.zscore_window,
            )

    # Inject OOS trade count directly into metrics so the summary card
    # shows the correct number without any proxy tricks.
    if has_oos:
        display_metrics["n_oos_trades"] = display_n_trades

    print(f"\nBuilding dashboard ({display_label})...")
    output_path = build_dashboard(
        equity_curve  = display_equity,
        trades        = display_trades,
        pair_pnl      = display_pair_pnl,
        metrics       = display_metrics,
        spread_data   = spread_data,
        signal_data   = signal_data,
        fold_summary  = fold_summary,
        wf_equity     = wf_equity if has_oos else None,
        output_path   = "results/dashboard.html",
    )

    print(f"\n" + "="*60)
    print(f"  Dashboard saved : {output_path}")
    print(f"  Headline source : {display_label}")
    if has_oos:
        print(f"  OOS Sharpe      : {oos_metrics['sharpe_ratio']:.3f}")
        print(f"  OOS total return: {oos_metrics['total_return_pct']:+.2f}%")
        print(f"  OOS trades      : {n_oos_trades}")
    print("  Open results/dashboard.html in your browser")
    print("="*60)