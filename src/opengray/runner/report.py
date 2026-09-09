"Research result summaries and plots from recorded episode tables."

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from opengray.runner.results import extra_tables, leaderboard, load_runs

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#9085e9", "#e34948"]
REFERENCE_GRAY = "#8a8a86"


def _fmt(x: Any, nd: int = 3) -> str:
    if isinstance(x, float):
        return "" if np.isnan(x) else f"{x:.{nd}f}"
    return str(x)


def leaderboard_table(entries: list[dict[str, Any]]) -> str:
    cols = [("agent", "Agent"), ("track", "Track"), ("k", "K"), ("n_cases", "Cases"), ("n_seeds", "Seeds"), ("track_score_mean", "Track score (case mean)"), ("track_score_ci95", "95% CI over cases"), ("score_column", "Score"), ("plan_score_mean", "PlanScore (case mean)"), ("plan_score_ci95", "95% CI over cases"), ("within_case_sd", "SD across seeds"), ("H_mean", "H"), ("V_mean", "V"), ("R_mean", "R"), ("gated_rate", "Gated"), ("escalation_rate", "Escalated"), ("error_rate", "Errors"), ("optimize_calls_mean", "Optimize calls"), ("wall_s_mean", "Wall s")]
    head = "".join(f"<th>{html.escape(h)}</th>" for _, h in cols)
    rows = []
    fallback = {"track_score_mean": "plan_score_mean", "track_score_ci95": "plan_score_ci95"}
    for e in entries:
        cells = []
        for key, _ in cols:
            v = e.get(key, e.get(fallback.get(key, key), "plan_score" if key == "score_column" else None))
            if key in ("plan_score_ci95", "track_score_ci95"):
                cells.append(f"<td>[{_fmt(v[0])}, {_fmt(v[1])}]</td>")
            elif key in ("wall_s_mean", "optimize_calls_mean"):
                cells.append(f"<td>{_fmt(float(v), 1)}</td>")
            else:
                cells.append(f"<td>{html.escape(_fmt(v))}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table class='lb'><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def score_figure(df: pd.DataFrame) -> dict[str, Any]:
    import plotly.graph_objects as go

    fig = go.Figure()
    agents = list(dict.fromkeys(df["agent"].tolist()))
    for i, agent in enumerate(agents):
        g = df[df["agent"] == agent]
        labels = g["track"].astype(str) + " K=" + g["k"].astype(str)
        fig.add_trace(go.Box(y=g["plan_score"], x=labels, name=agent, marker_color=SERIES[i % len(SERIES)], boxpoints="all", jitter=0.4, pointpos=0, marker_size=5, line_width=1.5, hovertext=g["episode_id"]))
    fig.update_layout(boxmode="group", template="plotly_white", yaxis_title="PlanScore", xaxis_title="", legend_title_text="Agent", margin={"l": 40, "r": 20, "t": 30, "b": 40}, height=420)
    return json.loads(fig.to_json())


def goal_met_figure(df: pd.DataFrame) -> dict[str, Any] | None:
    import plotly.graph_objects as go

    met_cols = [c for c in df.columns if c.startswith("met.")]
    if not met_cols:
        return None
    fig = go.Figure()
    agents = list(dict.fromkeys(df["agent"].tolist()))
    names = [c[4:] for c in met_cols]
    for i, agent in enumerate(agents):
        g = df[df["agent"] == agent]
        rates = [float(np.nanmean(g[c].astype(float))) if g[c].notna().any() else np.nan for c in met_cols]
        fig.add_trace(go.Bar(x=names, y=rates, name=agent, marker_color=SERIES[i % len(SERIES)]))
    fig.update_layout(barmode="group", template="plotly_white", yaxis_title="Fraction of episodes with goal met", yaxis_range=[0, 1.02], margin={"l": 40, "r": 20, "t": 30, "b": 120}, height=420, legend_title_text="Agent")
    return json.loads(fig.to_json())


def dvh_figures(df: pd.DataFrame, runs_dir: Path, cache_dir: Path | None, max_cases: int = 3) -> list[dict[str, Any]]:
    """DVH overlays for the first few submitted episodes: agent plan vs reference plan."""
    if cache_dir is None or df.empty:
        return []
    import plotly.graph_objects as go

    from opengray.data.openkbp_opt import OpenKBPOptLoader
    from opengray.physics.dvh import dvh_curve

    loader = OpenKBPOptLoader(cache_dir)
    figs = []
    seen_cases: set[str] = set()
    for _, row in df[df["outcome"] == "submitted"].iterrows():
        if row["case_id"] in seen_cases:
            continue
        w_path = Path(runs_dir) / str(row["run_id"]) / "final_w" / f"{row['episode_id']}.npy"
        if not w_path.exists():
            continue
        try:
            case = loader.load(row["case_id"])
        except FileNotFoundError:
            continue
        w = np.load(w_path)
        dose = case.dose(w)
        fig = go.Figure()
        d_max = max(float(dose.max()), float(case.reference_dose.max()) if case.reference_dose is not None else 0.0)
        for i, name in enumerate(case.structures):
            color = SERIES[i % len(SERIES)]
            lv, fr = dvh_curve(case.structure_dose(dose, name), 120, d_max)
            fig.add_trace(go.Scatter(x=lv, y=fr, mode="lines", name=name, line={"color": color, "width": 2}, legendgroup=name))
            if case.reference_dose is not None:
                lv, fr = dvh_curve(case.structure_dose(case.reference_dose, name), 120, d_max)
                fig.add_trace(go.Scatter(x=lv, y=fr, mode="lines", name=f"{name} (reference)", line={"color": color, "width": 1.5, "dash": "dot"}, legendgroup=name, showlegend=False, opacity=0.8))
        fig.update_layout(template="plotly_white", title=f"{row['case_id']}: {row['agent']} (solid) vs reference plan (dotted), PlanScore {row['plan_score']:.3f}", xaxis_title="Dose (Gy)", yaxis_title="Volume fraction", margin={"l": 40, "r": 20, "t": 50, "b": 40}, height=440)
        figs.append(json.loads(fig.to_json()))
        seen_cases.add(row["case_id"])
        if len(seen_cases) >= max_cases:
            break
    return figs


def track4_html(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "<p>No Track 4 episodes yet.</p>"
    cols = [("agent", "Agent"), ("transform", "Transform"), ("n_episodes", "n"), ("t4_score_mean", "Track 4 score"), ("t4_score_ci95", "95% CI"), ("delta_vs_control", "Drop vs control"), ("correct_rate", "Correct behaviour"), ("n_not_counted", "Not counted"), ("n_label_refuted", "Labels refuted"), ("escalation_rate", "Escalated"), ("gated_rate", "Gated")]
    head = "".join(f"<th>{html.escape(h)}</th>" for _, h in cols)
    body = []
    for r in rows:
        cells = []
        for key, _ in cols:
            v = r[key]
            if key == "t4_score_ci95":
                cells.append(f"<td>[{_fmt(v[0])}, {_fmt(v[1])}]</td>")
            elif key == "delta_vs_control":
                cells.append(f"<td>{'' if r['transform'] == 'none' else _fmt(float(v))}</td>")
            else:
                cells.append(f"<td>{html.escape(_fmt(v))}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table class='lb'><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def _generic_html(rows: list[dict[str, Any]], cols: list[tuple[str, str]], empty: str) -> str:
    if not rows:
        return f"<p>{html.escape(empty)}</p>"
    head = "".join(f"<th>{html.escape(h)}</th>" for _, h in cols)
    body = []
    for r in rows:
        cells = []
        for key, _ in cols:
            v = r.get(key)
            if isinstance(v, list) and len(v) == 2:
                cells.append(f"<td>[{_fmt(v[0])}, {_fmt(v[1])}]</td>")
            else:
                cells.append(f"<td>{html.escape(_fmt(v))}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table class='lb'><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def paired_html(rows: list[dict[str, Any]]) -> str:
    cols = [("track", "Track"), ("k", "K"), ("agent", "Agent"), ("anchor", "Anchor"), ("score_column", "Score"), ("n_cases", "Cases"), ("mean_diff", "Mean difference"), ("diff_ci95", "95% CI over cases"), ("cases_above", "Above"), ("cases_level", "Level"), ("cases_below", "Below")]
    return _generic_html(rows, cols, "No paired comparisons (no anchor rows).")


def matched_paired_html(rows: list[dict[str, Any]]) -> str:
    cols = [("track", "Track"), ("k", "K"), ("agent", "Agent"), ("anchor", "Anchor"), ("score_column", "Score"), ("n_cases", "Cases"), ("n_cells", "Matched cells"), ("n_cells_agent_only", "Agent-only cells"), ("n_cells_anchor_only", "Anchor-only cells"), ("mean_diff", "Mean difference"), ("diff_ci95", "95% CI over cases"), ("cases_above", "Above"), ("cases_level", "Level"), ("cases_below", "Below")]
    return _generic_html(rows, cols, "No matched comparisons.")


def track3_html(rows: list[dict[str, Any]]) -> str:
    cols = [("agent", "Agent"), ("kind", "Update"), ("n_episodes", "n exposed"), ("n_unexposed", "n unexposed"), ("optimizes_after_update_mean", "Optimizes after update"), ("plan_score_mean", "PlanScore (final goals)"), ("plan_score_ci95", "95% CI"), ("H_mean", "H"), ("updated_goal_met_rate", "Changed goal met"), ("gated_rate", "Gated"), ("optimize_calls_mean", "Optimize calls")]
    return _generic_html(rows, cols, "No Track 3 episodes yet.")


def track5_html(rows: list[dict[str, Any]]) -> str:
    cols = [("agent", "Agent"), ("n_tight", "n tight"), ("n_certified_infeasible", "n certified infeasible"), ("n_uncertified", "n uncertified (not counted)"), ("n_label_refuted", "labels refuted"), ("precision", "Escalation precision"), ("recall", "Escalation recall"), ("reason_infeasible_rate", "Reason = infeasible"), ("tight_plan_score_mean", "PlanScore, tight arm"), ("tight_plan_score_ci95", "95% CI"), ("t5_score_mean", "Track 5 score")]
    return _generic_html(rows, cols, "No Track 5 episodes yet.")


def build_report(runs_dir: Path, out_dir: Path, cache_dir: Path | None = None, anchor: str | None = None) -> Path:
    df = load_runs(runs_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    entries = leaderboard(df)
    extra = extra_tables(df, anchor)
    t3, t4, t5, paired, matched = extra.get("track3", []), extra.get("track4", []), extra.get("track5", []), extra.get("paired", []), extra.get("matched_paired", [])
    payload: dict[str, Any] = {"entries": entries, **extra}
    (out_dir / "leaderboard.json").write_text(json.dumps(payload, indent=2))
    figs: dict[str, Any] = {}
    if not df.empty:
        figs["scores"] = score_figure(df)
        gm = goal_met_figure(df)
        if gm is not None:
            figs["goals"] = gm
        figs["dvh"] = dvh_figures(df, runs_dir, cache_dir)
    n_runs = df["run_id"].nunique() if not df.empty else 0
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>OpenGray leaderboard</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
:root {{ --ground:#f3f4f6; --surface:#fff; --ink:#171a1d; --muted:#59616b; --line:#d5d9de; --accent:#1e6a86; }}
body {{ margin:0; background:var(--ground); color:var(--ink); font:16px/1.5 "Source Sans 3","Helvetica Neue",Arial,sans-serif; }}
main {{ max-width:1180px; margin:0 auto; padding:32px 24px 80px; }}
h1 {{ font-family:Georgia,"Source Serif 4",serif; font-weight:600; font-size:34px; margin:0 0 4px; }}
h2 {{ font-size:20px; margin:36px 0 12px; }}
p.sub {{ color:var(--muted); margin:0 0 20px; }}
.card {{ background:var(--surface); border:1px solid var(--line); border-radius:6px; padding:16px; overflow-x:auto; }}
table.lb {{ border-collapse:collapse; width:100%; font-size:14px; font-variant-numeric:tabular-nums; }}
table.lb th {{ text-align:left; font-family:ui-monospace,Menlo,monospace; font-size:11px; letter-spacing:.06em; text-transform:uppercase; color:var(--muted); padding:8px 10px; border-bottom:1px solid var(--line); }}
table.lb td {{ padding:8px 10px; border-bottom:1px solid var(--line); }}
table.lb tr:last-child td {{ border-bottom:0; }}
.fig {{ margin-bottom:16px; }}
</style></head><body><main>
<h1>OpenGray leaderboard</h1>
<p class="sub">{len(df)} episodes across {n_runs} run(s). PlanScore is the mean over cases of each case's mean over seeds; the interval is a bootstrap over cases. Doses are in gray; PlanScore is a dimensionless composite.</p>
<div class="card">{leaderboard_table(entries) if entries else "<p>No results yet.</p>"}</div>
<h2>Paired differences against the heuristic, per case</h2><div class="card">{paired_html(paired)}</div>
<h2>Paired differences matched on case and task (Track 3 update, Track 4 transform, Track 5 arm)</h2><div class="card">{matched_paired_html(matched)}</div>
<h2>Track 3: goal update after the second optimize</h2><div class="card">{track3_html(t3)}</div>
<h2>Track 4: adversarial transforms</h2><div class="card">{track4_html(t4)}</div>
<h2>Track 5: escalation on contradictory and tightened goal lists</h2><div class="card">{track5_html(t5)}</div>
<h2>PlanScore by track</h2><div class="card"><div id="scores" class="fig"></div></div>
<h2>Goal met rate</h2><div class="card"><div id="goals" class="fig"></div></div>
<h2>Case drill-down: DVH, agent plan versus reference</h2><div class="card" id="dvh"></div>
<script>
const figs = {json.dumps(figs)};
if (figs.scores) Plotly.newPlot('scores', figs.scores.data, figs.scores.layout, {{responsive:true, displaylogo:false}});
if (figs.goals) Plotly.newPlot('goals', figs.goals.data, figs.goals.layout, {{responsive:true, displaylogo:false}});
(figs.dvh || []).forEach((f, i) => {{ const d = document.createElement('div'); d.id = 'dvh' + i; d.className='fig'; document.getElementById('dvh').appendChild(d); Plotly.newPlot(d.id, f.data, f.layout, {{responsive:true, displaylogo:false}}); }});
if (!(figs.dvh || []).length) document.getElementById('dvh').innerHTML = '<p>Pass --cache to render DVH overlays.</p>';
</script></main></body></html>"""
    (out_dir / "index.html").write_text(page)
    return out_dir / "index.html"
