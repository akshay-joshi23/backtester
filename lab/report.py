"""Generate a self-contained HTML report for a single backtest run.

Charts are matplotlib PNGs, base64-encoded so the HTML has no external file
dependencies. Open the file in any browser to view.

Layout:
    Title + strategy name
    Metrics table (Sharpe, CAGR, max DD, etc.)
    Equity curve + drawdown timeline
    Realized-weights heatmap (asset × time)
    Monthly returns calendar (heatmap)
    Strategy source code (collapsible <details>)
    Original prompt
"""

from __future__ import annotations

import base64
import html
import io
import logging
import textwrap
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt  # noqa: E402

logger = logging.getLogger(__name__)


def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _equity_and_drawdown(equity: pd.Series) -> str:
    cum = equity / equity.iloc[0]
    cummax = cum.cummax()
    dd = (cum / cummax - 1.0) * 100.0
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True,
                                    gridspec_kw={"height_ratios": [3, 1]})
    ax1.plot(cum.index, cum.values, color="#1f77b4", linewidth=1.2)
    ax1.set_ylabel("Cumulative return (×)")
    ax1.set_title("Equity curve")
    ax1.grid(alpha=0.3)
    ax2.fill_between(dd.index, dd.values, 0.0, color="#d62728", alpha=0.5)
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("Date")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    return _fig_to_b64(fig)


def _weights_heatmap(weights: pd.DataFrame) -> str:
    if weights.empty:
        return ""
    # Resample to monthly average for readability.
    monthly = weights.resample("ME").mean()
    fig, ax = plt.subplots(figsize=(10, max(2.5, 0.35 * len(monthly.columns))))
    cmap = plt.get_cmap("viridis")
    im = ax.imshow(
        monthly.T.values, aspect="auto", origin="lower",
        cmap=cmap, vmin=0.0, vmax=max(0.1, float(monthly.values.max())),
    )
    ax.set_yticks(range(len(monthly.columns)))
    ax.set_yticklabels(monthly.columns)
    # X axis with year-month ticks
    n = len(monthly)
    tick_idx = list(range(0, n, max(1, n // 12)))
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([monthly.index[i].strftime("%Y-%m") for i in tick_idx],
                       rotation=45, ha="right")
    ax.set_title("Average monthly weights")
    fig.colorbar(im, ax=ax, label="weight")
    fig.tight_layout()
    return _fig_to_b64(fig)


def _monthly_calendar(equity: pd.Series) -> str:
    rets = equity.pct_change().dropna()
    if rets.empty:
        return ""
    monthly = (1.0 + rets).resample("ME").prod() - 1.0
    df = pd.DataFrame({
        "year": monthly.index.year,
        "month": monthly.index.month,
        "ret": monthly.values * 100.0,
    })
    pivot = df.pivot(index="year", columns="month", values="ret")
    pivot = pivot.reindex(columns=range(1, 13))
    fig, ax = plt.subplots(figsize=(10, max(2.2, 0.35 * len(pivot.index))))
    vmax = float(np.nanmax(np.abs(pivot.values))) if pivot.size else 5.0
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=-vmax, vmax=vmax)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xticks(range(12))
    ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    # Annotate cells
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.iat[i, j]
            if pd.notna(v):
                ax.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=7,
                        color="black")
    ax.set_title("Monthly returns (%)")
    fig.colorbar(im, ax=ax, label="%")
    fig.tight_layout()
    return _fig_to_b64(fig)


def _metrics_table_html(metrics: dict) -> str:
    rows = []
    order = [
        ("Sharpe", "sharpe", "{:.3f}"),
        ("Sortino", "sortino", "{:.3f}"),
        ("CAGR", "cagr", "{:.2%}"),
        ("Annualized vol", "ann_vol", "{:.2%}"),
        ("Max drawdown", "max_drawdown", "{:.2%}"),
        ("Calmar", "calmar", "{:.3f}"),
        ("Final NAV", "final_nav", "{:.3f}"),
        ("Annual turnover", "annual_turnover", "{:.2f}"),
        ("Total TX cost", "total_transaction_cost", "{:.4f}"),
        ("Trading days", "n_obs", "{:d}"),
    ]
    for label, key, fmt in order:
        v = metrics.get(key)
        if v is None:
            continue
        try:
            cell = fmt.format(int(v) if isinstance(v, float) and fmt == "{:d}" else v)
        except Exception:
            cell = str(v)
        rows.append(f"<tr><th>{html.escape(label)}</th><td>{cell}</td></tr>")
    return f"<table class='metrics'>{''.join(rows)}</table>"


HTML_TEMPLATE = """\
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <style>
    body {{ font-family: -apple-system, system-ui, "SF Pro", sans-serif;
           max-width: 1100px; margin: 32px auto; padding: 0 24px; color: #222; }}
    h1 {{ margin-bottom: 4px; }}
    h2 {{ margin-top: 28px; padding-bottom: 4px; border-bottom: 1px solid #ddd; }}
    table.metrics {{ border-collapse: collapse; margin-top: 8px; }}
    table.metrics th, table.metrics td {{
      padding: 6px 14px; border-bottom: 1px solid #eee; text-align: left;
    }}
    table.metrics th {{ font-weight: 600; color: #555; }}
    img {{ display: block; max-width: 100%; margin-top: 12px; }}
    pre {{ background: #f6f8fa; padding: 12px; border-radius: 6px;
           overflow-x: auto; font-size: 13px; line-height: 1.4; }}
    .meta {{ color: #666; font-size: 14px; margin-bottom: 18px; }}
    details summary {{ cursor: pointer; user-select: none; }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <div class="meta">
    Run <code>{run_id}</code> · universe <code>{universe}</code> ·
    train end {train_end} · rebalance every {rebalance_freq} days
  </div>

  <h2>Metrics</h2>
  {metrics_table}

  <h2>Equity curve and drawdown</h2>
  <img src="data:image/png;base64,{equity_img}" alt="equity curve">

  <h2>Average monthly weights</h2>
  {weights_img_block}

  <h2>Monthly returns calendar</h2>
  {monthly_img_block}

  <h2>Prompt</h2>
  <pre>{prompt}</pre>

  <h2>Strategy source</h2>
  <details><summary>Show / hide</summary><pre>{strategy_code}</pre></details>
</body>
</html>
"""


def render_run_report(run_dict: dict, *, output_path: Path | None = None) -> Path:
    """Write an HTML report for a single run. Returns the path to the HTML."""
    run_dir = Path(run_dict["run_dir"])
    cfg = run_dict["config"]
    metrics = run_dict["metrics"]
    equity = run_dict["equity"]
    weights = run_dict.get("weights")
    prompt = run_dict.get("prompt", "")
    strategy_code = run_dict.get("strategy_code", "")

    equity_img = _equity_and_drawdown(equity)
    weights_img = _weights_heatmap(weights) if weights is not None else ""
    monthly_img = _monthly_calendar(equity)

    def _img_block(b64: str, alt: str) -> str:
        if not b64:
            return "<p><em>(no data)</em></p>"
        return f'<img src="data:image/png;base64,{b64}" alt="{alt}">'

    output_path = output_path or (run_dir / "report.html")
    html_text = HTML_TEMPLATE.format(
        title=html.escape(f"{cfg.get('strategy_name', 'Strategy')} — {run_dict['run_id']}"),
        run_id=html.escape(run_dict["run_id"]),
        universe=html.escape(", ".join(cfg.get("universe", []))),
        train_end=html.escape(cfg.get("train_end", "?")),
        rebalance_freq=html.escape(str(cfg.get("rebalance_freq", "?"))),
        metrics_table=_metrics_table_html(metrics),
        equity_img=equity_img,
        weights_img_block=_img_block(weights_img, "weights heatmap"),
        monthly_img_block=_img_block(monthly_img, "monthly returns calendar"),
        prompt=html.escape(prompt) or "(no prompt — reference strategy)",
        strategy_code=html.escape(strategy_code),
    )
    output_path.write_text(html_text)
    logger.info("HTML report written: %s", output_path)
    return output_path
