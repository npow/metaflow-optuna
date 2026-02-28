"""
Study card rendering.

render_study_card(study) — appends components to current.card (Metaflow blank card)
render_study_html(study) — returns a standalone HTML string with Plotly interactivity
"""
from __future__ import annotations

import io
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import optuna


# ---------------------------------------------------------------------------
# Metaflow card rendering  (@card(type="blank") + current.card.append(...))
# ---------------------------------------------------------------------------

def render_study_card(study: "optuna.Study") -> None:
    """
    Call this inside a @card(type="blank") step body.
    Appends best-trial summary, optimization history chart, parameter
    importance chart, and trials table to current.card.
    """
    from metaflow import current
    from metaflow.cards import Image, Markdown, Table

    complete = [t for t in study.trials if t.state.name == "COMPLETE"]
    failed   = [t for t in study.trials if t.state.name == "FAIL"]
    total    = len(study.trials)

    direction_arrow = "↓" if study.direction.name == "MINIMIZE" else "↑"

    # ---- Best trial hero ----
    try:
        best = study.best_trial
        best_val = f"{best.value:.6g}"
        param_rows = "\n".join(
            f"| `{k}` | `{v}` |" for k, v in best.params.items()
        )
        hero_md = f"""## Best Trial — #{best.number} of {total}

| Metric | Value |
|--------|-------|
| **Objective** | **{best_val}** {direction_arrow} |
| Complete | {len(complete)} / {total} |
| Failed | {len(failed)} |

### Best Parameters

| Parameter | Value |
|-----------|-------|
{param_rows}
"""
    except Exception:
        hero_md = f"## Study complete — {len(complete)}/{total} trials\n\n_No best trial available._\n"

    current.card.append(Markdown(hero_md))

    # ---- Optimization history (matplotlib) ----
    if len(complete) >= 2:
        try:
            current.card.append(Image.from_matplotlib(_plot_history(study), "Optimization History"))
            current.card.append(Image.from_matplotlib(_plot_importance(study), "Parameter Importance"))
        except Exception as e:
            current.card.append(Markdown(f"_Chart rendering failed: {e}_"))

    # ---- Trials table ----
    if complete:
        try:
            param_names = list(complete[0].params.keys())
            headers = ["#", *param_names, "Objective", "Duration (s)"]

            rows = []
            best_val_num = study.best_value if complete else None
            for t in sorted(complete, key=lambda t: t.value if study.direction.name == "MINIMIZE" else -t.value):
                marker = " ★" if best_val_num is not None and abs(t.value - best_val_num) < 1e-12 else ""
                row = [
                    str(t.number),
                    *[f"{t.params[p]:.4g}" if isinstance(t.params[p], float) else str(t.params[p])
                      for p in param_names],
                    f"{t.value:.6g}{marker}",
                    "—",
                ]
                rows.append(row)

            current.card.append(Markdown("## All Trials"))
            current.card.append(Table(data=rows, headers=headers))
        except Exception as e:
            current.card.append(Markdown(f"_Table rendering failed: {e}_"))

    # ---- Failed trials note ----
    if failed:
        current.card.append(Markdown(f"> ⚠️ {len(failed)} trial(s) failed. Check train step logs."))


# ---------------------------------------------------------------------------
# Standalone HTML export (Plotly interactive)
# ---------------------------------------------------------------------------

def render_study_html(study: "optuna.Study") -> str:
    """
    Returns a self-contained HTML string with interactive Plotly charts
    and a styled trials table.  Save as self.study_html or write to disk.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    complete = [t for t in study.trials if t.state.name == "COMPLETE"]
    if not complete:
        return "<html><body><h1>No completed trials</h1></body></html>"

    direction_minimize = study.direction.name == "MINIMIZE"
    best = study.best_trial

    # ---- 1. Optimization history ----
    nums   = [t.number for t in complete]
    vals   = [t.value  for t in complete]
    best_so_far = []
    running_best = float("inf") if direction_minimize else float("-inf")
    for v in vals:
        running_best = min(running_best, v) if direction_minimize else max(running_best, v)
        best_so_far.append(running_best)

    hist_fig = go.Figure()
    hist_fig.add_trace(go.Scatter(x=nums, y=vals, mode="markers", name="Trial value",
                                  marker=dict(color="steelblue", size=6, opacity=0.7)))
    hist_fig.add_trace(go.Scatter(x=nums, y=best_so_far, mode="lines", name="Best so far",
                                  line=dict(color="tomato", width=2)))
    hist_fig.update_layout(title="Optimization History", xaxis_title="Trial #",
                           yaxis_title="Objective", template="plotly_white", height=350)

    # ---- 2. Parameter importance (fANOVA via optuna) ----
    imp_html = ""
    try:
        import optuna
        importances = optuna.importance.get_param_importances(study)
        params_sorted = list(importances.keys())
        imps_sorted   = [importances[p] for p in params_sorted]
        imp_fig = go.Figure(go.Bar(x=imps_sorted, y=params_sorted, orientation="h",
                                   marker_color="mediumpurple"))
        imp_fig.update_layout(title="Parameter Importance (fANOVA)", template="plotly_white", height=300)
        imp_html = imp_fig.to_html(full_html=False, include_plotlyjs=False)
    except Exception:
        imp_html = "<p><em>Importance not available (need ≥ 2 complete trials with varied params)</em></p>"

    # ---- 3. Parallel coordinates ----
    param_names = list(complete[0].params.keys())
    par_data = {p: [t.params[p] for t in complete] for p in param_names}
    par_data["objective"] = vals

    dimensions = []
    for p in param_names:
        vals_p = par_data[p]
        # Treat as categorical if any value is non-numeric (str, None, bool)
        is_categorical = any(isinstance(v, (str, bool)) or v is None for v in vals_p)
        if is_categorical:
            str_vals = [str(v) for v in vals_p]  # None → "None", bool → "True"/"False"
            unique = sorted(set(str_vals))
            dims = dict(label=p, values=[unique.index(v) for v in str_vals],
                        tickvals=list(range(len(unique))), ticktext=unique)
        else:
            dims = dict(label=p, values=vals_p)
        dimensions.append(dims)
    dimensions.append(dict(label="objective", values=vals))

    colorscale = "RdYlGn_r" if direction_minimize else "RdYlGn"
    par_fig = go.Figure(go.Parcoords(
        line=dict(color=vals, colorscale=colorscale, showscale=True),
        dimensions=dimensions,
    ))
    par_fig.update_layout(title="Parallel Coordinates", template="plotly_white", height=350)

    # ---- 4. Trials table HTML ----
    sorted_trials = sorted(complete, key=lambda t: t.value if direction_minimize else -t.value)
    table_rows = ""
    for i, t in enumerate(sorted_trials):
        bg = "#e8f5e9" if i == 0 else ("#fff" if i % 2 == 0 else "#f9f9f9")
        star = " ★" if i == 0 else ""
        param_cells = "".join(
            "<td>{}</td>".format(
                f"{t.params[p]:.4g}" if isinstance(t.params[p], float) else t.params[p]
            )
            for p in param_names
        )
        table_rows += (
            f'<tr style="background:{bg}">'
            f"<td>{t.number}</td>{param_cells}"
            f"<td><strong>{t.value:.6g}{star}</strong></td>"
            f"</tr>\n"
        )
    param_headers = "".join(f"<th>{p}</th>" for p in param_names)
    table_html = f"""
<table style="width:100%;border-collapse:collapse;font-size:13px">
  <thead>
    <tr style="background:#2c3e50;color:white">
      <th>#</th>{param_headers}<th>Objective</th>
    </tr>
  </thead>
  <tbody>{table_rows}</tbody>
</table>"""

    # ---- Assemble ----
    direction_label = "↓ minimize" if direction_minimize else "↑ maximize"
    best_params_html = " &nbsp;·&nbsp; ".join(
        "<code>{}</code>: <strong>{}</strong>".format(
            k, f"{v:.4g}" if isinstance(v, float) else v
        )
        for k, v in best.params.items()
    )

    hist_html = hist_fig.to_html(full_html=False, include_plotlyjs=False)
    par_html  = par_fig.to_html(full_html=False, include_plotlyjs=False)
    from plotly.offline import get_plotlyjs
    plotly_js = f"<script>{get_plotlyjs()}</script>"

    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>metaflow-optuna study</title>
  {plotly_js}
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
           margin: 0; padding: 24px; background: #f5f5f5; color: #2c3e50; }}
    .hero {{ background: white; border-radius: 8px; padding: 24px; margin-bottom: 24px;
             box-shadow: 0 2px 8px rgba(0,0,0,.08); }}
    .hero h1 {{ margin: 0 0 8px; font-size: 22px; }}
    .metric {{ font-size: 36px; font-weight: 700; color: #27ae60; }}
    .direction {{ color: #888; font-size: 14px; margin-left: 8px; }}
    .params {{ margin-top: 12px; font-size: 14px; color: #555; }}
    .charts {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 24px; }}
    .card {{ background: white; border-radius: 8px; padding: 16px;
             box-shadow: 0 2px 8px rgba(0,0,0,.08); }}
    .card.full {{ grid-column: 1/-1; }}
    h2 {{ margin: 0 0 12px; font-size: 16px; color: #2c3e50; }}
    th {{ padding: 8px 12px; text-align: left; }}
    td {{ padding: 6px 12px; border-top: 1px solid #eee; }}
    .failed-note {{ background: #fff3cd; border: 1px solid #ffc107; border-radius: 4px;
                    padding: 12px; margin-top: 16px; font-size: 13px; }}
  </style>
</head>
<body>
  <div class="hero">
    <h1>Best Trial — #{best.number} of {len(study.trials)}
      <span class="direction">({direction_label})</span></h1>
    <div class="metric">{best.value:.6g}</div>
    <div class="params">{best_params_html}</div>
  </div>

  <div class="charts">
    <div class="card">{hist_html}</div>
    <div class="card">{imp_html}</div>
    <div class="card full">{par_html}</div>
    <div class="card full">
      <h2>All Trials ({len(complete)} complete)</h2>
      {table_html}
    </div>
  </div>
</body>
</html>"""

    return html


# ---------------------------------------------------------------------------
# matplotlib helpers (used by render_study_card)
# ---------------------------------------------------------------------------

def _plot_history(study: "optuna.Study"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    complete = [t for t in study.trials if t.state.name == "COMPLETE"]
    minimize = study.direction.name == "MINIMIZE"
    nums = [t.number for t in complete]
    vals = [t.value  for t in complete]
    best_so_far, cur = [], float("inf") if minimize else float("-inf")
    for v in vals:
        cur = min(cur, v) if minimize else max(cur, v)
        best_so_far.append(cur)

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.scatter(nums, vals, s=25, alpha=0.6, color="#5b9bd5", label="Trial value", zorder=3)
    ax.plot(nums, best_so_far, color="#ed7d31", linewidth=2, label="Best so far")
    ax.set_xlabel("Trial #")
    ax.set_ylabel("Objective")
    ax.set_title("Optimization History")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def _plot_importance(study: "optuna.Study"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        import optuna
        importances = optuna.importance.get_param_importances(study)
    except Exception:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "Insufficient data for importance", ha="center", va="center",
                transform=ax.transAxes, color="gray")
        ax.axis("off")
        return fig

    params   = list(reversed(list(importances.keys())))
    imps     = [importances[p] for p in params]
    colors   = ["#70ad47" if i == imps.index(max(imps)) else "#a9d18e" for i in range(len(imps))]

    fig, ax = plt.subplots(figsize=(6, max(2.5, len(params) * 0.45)))
    bars = ax.barh(params, imps, color=colors, edgecolor="white")
    ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
    ax.set_xlabel("Importance")
    ax.set_title("Parameter Importance (fANOVA)")
    ax.set_xlim(0, max(imps) * 1.25)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return fig
