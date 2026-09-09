"""OpenGray command line: download | ingest | validate | splits (more arrive with later milestones)."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="OpenGray: an open environment and benchmark for agentic radiotherapy planning.")
console = Console()

DEFAULT_CACHE = Path(os.environ.get("OPENGRAY_CACHE", Path.home() / ".opengray"))

OPENKBP_OPT_LINKS = {
    "core-data.zip (reference-plans, paper-predictions; 10.19 GB)": "https://1drv.ms/u/c/2150c5a213e729e3/EeMp5xOixVAggCFvAAAAAAABEDPNyGWc32_OuGeTHUFZkw?e=x3V3fq",
    "experiments-data.zip (paper-plans, results-data, results; 13.08 GB, optional)": "https://1drv.ms/u/c/2150c5a213e729e3/EeMp5xOixVAggCFwAAAAAAABgPUYh9eHaIT0pv-w-8yF6A?e=2QDrwB",
}


@app.command()
def download(cohort: str = typer.Argument("openkbp-opt")) -> None:
    """Print where to fetch a cohort. OpenGray never redistributes data; see data/LICENSES.md."""
    if cohort.replace("_", "-") != "openkbp-opt":
        raise typer.BadParameter(f"unknown cohort {cohort!r}; v1 supports openkbp-opt")
    console.print("[bold]OpenKBP-Opt[/bold] is hosted on the authors' OneDrive and has no programmatic downloader.")
    console.print("Download in a browser (links from github.com/ababier/open-kbp-opt README), then run:")
    console.print("  opengray ingest /path/to/core-data.zip --cache <cache_dir>\n")
    for label, url in OPENKBP_OPT_LINKS.items():
        console.print(f"  {label}\n    {url}")
    console.print("\nCite: Babier A, et al. Phys Med Biol 67, 185012 (2022). Upstream software: MIT; consult the source-data terms. See data/LICENSES.md.")


@app.command()
def ingest(
    archive: Annotated[Path, typer.Argument(help="Path to core-data.zip", exists=True, readable=True)],
    cache: Annotated[Path, typer.Option(help="Cache directory")] = DEFAULT_CACHE,
    patients: Annotated[str | None, typer.Option(help="Comma list or range, e.g. pt_241-pt_260")] = None,
    no_resume: Annotated[bool, typer.Option("--no-resume", help="Re-ingest cases already cached")] = False,
    grid: Annotated[str, typer.Option(help="Source grid shape (advanced; tests use small grids)")] = "128,128,128",
) -> None:
    """Convert OpenKBP-Opt patients from the downloaded archive into the package cache."""
    from opengray.data.openkbp_opt import ingest as _ingest
    from opengray.data.openkbp_opt import list_patients_in_zip

    selected = _parse_patients(patients, list_patients_in_zip(archive)) if patients else None
    t0 = time.perf_counter()
    n_done = 0

    def progress(pt: str, entry: dict[str, Any]) -> None:
        nonlocal n_done
        if entry.get("skipped"):
            console.print(f"  {pt}: cached, skipped")
            return
        n_done += 1
        s = entry["stats"]
        w = len(entry["warnings"])
        console.print(
            f"  {pt}: {s['n_feasible']} voxels, {s['n_beamlets']} beamlets, nnz {s['nnz']:,}, "
            f"{s['n_structures']} structures, {s['load_s']} s" + (f", {w} warning(s)" if w else "")
        )

    grid_shape = tuple(int(x) for x in grid.split(","))
    if len(grid_shape) != 3:
        raise typer.BadParameter("--grid must be three comma-separated integers")
    manifest = _ingest(
        archive, cache, patients=selected, resume=not no_resume, progress=progress, grid_shape=grid_shape  # type: ignore[arg-type]
    )
    console.print(
        f"Ingested {n_done} case(s) in {time.perf_counter() - t0:.0f} s; "
        f"{len(manifest['cases'])} case(s) now in {cache / 'openkbp_opt'}"
    )


@app.command()
def validate(
    cache: Annotated[Path, typer.Option(help="Cache directory")] = DEFAULT_CACHE,
    n: Annotated[int, typer.Option(help="Number of cases to load (0 = all)")] = 0,
    markdown: Annotated[Path | None, typer.Option(help="Also write a markdown summary here")] = None,
) -> None:
    "Load cached cases, check invariants, and print a cohort table."
    from opengray.data.openkbp_opt import OpenKBPOptLoader

    loader = OpenKBPOptLoader(cache)
    ids = loader.list_cases()
    if not ids:
        console.print(f"[red]No cached cases in {cache}. Run `opengray ingest` first.[/red]")
        raise typer.Exit(code=1)
    if n:
        ids = ids[:n]

    rows: list[dict[str, Any]] = []
    errors: list[tuple[str, str]] = []
    t0 = time.perf_counter()
    for cid in ids:
        try:
            t = time.perf_counter()
            case = loader.load(cid)
            load_s = time.perf_counter() - t
            w = np.ones(case.n_beamlets)
            d = case.dose(w)
            checks = {
                "D_nonneg": bool(case.D.data.min() >= 0),
                "dose_unit_w_nonzero_rows": int(np.count_nonzero(d)),
                "ref_in_mask": bool(case.reference_dose is not None and case.reference_dose.max() > 0),
            }
            if not checks["D_nonneg"]:
                errors.append((cid, "negative entries in D"))
            rows.append(
                {
                    "case": cid,
                    "voxels": case.n_feasible,
                    "beamlets": case.n_beamlets,
                    "nnz": int(case.D.nnz),
                    "D_mb": round((case.D.data.nbytes + case.D.indices.nbytes + case.D.indptr.nbytes) / 2**20, 1),
                    "targets": len(case.prescriptions),
                    "structures": len(case.structures),
                    "missing": ",".join(case.provenance.get("missing_structures", [])) or "-",
                    "ref_max_gy": round(float(case.reference_dose.max()), 1) if case.reference_dose is not None else float("nan"),
                    "unit_w_max_gy": round(float(d.max()), 1),
                    "warnings": len(case.provenance.get("warnings", [])),
                    "load_s": round(load_s, 2),
                }
            )
        except Exception as e:  # noqa: BLE001 - report every failure, keep going
            errors.append((cid, f"{type(e).__name__}: {e}"))

    table = Table(title=f"OpenKBP-Opt cache: {len(rows)} loaded, {len(errors)} errors")
    for col in rows[0].keys() if rows else []:
        table.add_column(col, justify="right" if col not in ("case", "missing") else "left")
    for r in rows:
        table.add_row(*[str(v) for v in r.values()])
    console.print(table)

    if rows:
        summary = _summarize(rows)
        console.print(summary_table(summary))
    for cid, msg in errors:
        console.print(f"[red]{cid}: {msg}[/red]")
    console.print(f"Validated in {time.perf_counter() - t0:.1f} s")

    if markdown and rows:
        markdown.parent.mkdir(parents=True, exist_ok=True)
        markdown.write_text(_markdown_report(rows, errors))
        console.print(f"Wrote {markdown}")
    if errors:
        raise typer.Exit(code=1)


@app.command("validate-solver")
def validate_solver(
    cache: Annotated[Path, typer.Option(help="Cache directory")] = DEFAULT_CACHE,
    n: Annotated[int, typer.Option(help="Number of cases (0 = all)")] = 10,
    cases: Annotated[str | None, typer.Option(help="Comma list of case ids (overrides --n)")] = None,
    experiments: Annotated[Path | None, typer.Option(help="experiments-data.zip for the matrix-read check")] = None,
    model: Annotated[str | None, typer.Option(help="paper-plans model to use (default: first)")] = None,
    n_sub: Annotated[int, typer.Option(help="Voxels per structure for the L-BFGS-B comparison")] = 200,
    markdown: Annotated[Path | None, typer.Option(help="Also write a markdown report here")] = None,
) -> None:
    "Check matrix reads, independent-solver agreement, and timings."
    from opengray.data.openkbp_opt import OpenKBPOptLoader
    from opengray.physics.validation import (
        default_objective,
        independent_solver_check,
        list_paper_plan_models,
        matrix_read_check,
        read_paper_plan,
        timing_run,
    )

    loader = OpenKBPOptLoader(cache)
    ids = loader.list_cases()
    if not ids:
        console.print(f"[red]No cached cases in {cache}. Run `opengray ingest` first.[/red]")
        raise typer.Exit(code=1)
    if cases:
        ids = [c.strip() for c in cases.split(",")]
    elif n:
        ids = ids[:n]
    if experiments is not None:
        models = list_paper_plan_models(experiments)
        if not models:
            console.print("[red]No paper-plans found in the experiments archive[/red]")
            raise typer.Exit(code=1)
        model = model or models[0]
        console.print(f"Matrix-read check against paper-plans model {model!r} ({len(models)} models available)")

    read_rows, solver_rows, timing_rows = [], [], []
    failures: list[str] = []
    for cid in ids:
        case = loader.load(cid)
        objective = default_objective(case)
        if experiments is not None:
            try:
                fl, d_idx, d_val = read_paper_plan(experiments, model, cid)  # type: ignore[arg-type]
                r = matrix_read_check(case, fl, d_idx, d_val)
                read_rows.append({"case": cid, **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}})
                if r["rel_max_diff"] > 1e-3:
                    failures.append(f"{cid}: matrix-read rel max diff {r['rel_max_diff']:.2e}")
            except FileNotFoundError as e:
                read_rows.append({"case": cid, "error": str(e)})
        r = independent_solver_check(case, objective, n_per_structure=n_sub)
        solver_rows.append({"case": cid, **{k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()}})
        # The objectives end near zero on a flat valley of near-equivalent plans, so the relative
        # test is complemented by the gap as a fraction of the starting objective: both solvers
        # must have removed the same fraction of the initial penalty to within 1e-4.
        gap_frac = (r["f_fista"] - r["f_lbfgsb"]) / max(r["f0"], 1e-9)
        if abs(r["rel_diff"]) > 1e-2 and gap_frac > 1e-4:
            failures.append(f"{cid}: FISTA vs L-BFGS-B rel diff {r['rel_diff']:.2e}, gap {gap_frac:.1e} of f0")
        r = timing_run(case, objective)
        timing_rows.append({"case": cid, **{k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()}})
        console.print(f"  {cid}: cold {timing_rows[-1]['cold_s']} s / {timing_rows[-1]['cold_iters']} it, warm {timing_rows[-1]['warm_s']} s / {timing_rows[-1]['warm_iters']} it, solver rel diff {solver_rows[-1]['rel_diff']:.1e}")

    for title, rows in (("Matrix-read check", read_rows), ("FISTA vs L-BFGS-B (subsampled)", solver_rows), ("Timing", timing_rows)):
        if not rows:
            continue
        table = Table(title=title)
        cols = list(rows[0].keys())
        for c in cols:
            table.add_column(c, justify="left" if c == "case" else "right")
        for r in rows:
            table.add_row(*[str(r.get(c, "")) for c in cols])
        console.print(table)
    if timing_rows:
        cold = np.array([r["cold_s"] for r in timing_rows], dtype=float)
        console.print(f"Cold solve wall time: min {cold.min():.1f} s, median {np.median(cold):.1f} s, max {cold.max():.1f} s (target: 60 s on CPU)")
    for f in failures:
        console.print(f"[red]{f}[/red]")
    if markdown:
        markdown.parent.mkdir(parents=True, exist_ok=True)
        lines = []
        for title, rows in (("Matrix-read check", read_rows), ("FISTA vs L-BFGS-B (subsampled)", solver_rows), ("Timing", timing_rows)):
            if not rows:
                continue
            cols = list(rows[0].keys())
            lines += [f"### {title}", "", "| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
            lines += ["| " + " | ".join(str(r.get(c, "")) for c in cols) + " |" for r in rows]
            lines.append("")
        lines += [f"- FAIL {f}" for f in failures]
        with open(markdown, "a") as fh:
            fh.write("\n".join(lines) + "\n")
        console.print(f"Appended to {markdown}")
    if failures:
        raise typer.Exit(code=1)


@app.command("score-reference")
def score_reference(
    cache: Annotated[Path, typer.Option(help="Cache directory")] = DEFAULT_CACHE,
    n: Annotated[int, typer.Option(help="Number of cases (0 = all)")] = 0,
    out: Annotated[Path, typer.Option(help="Output directory for the table and plot")] = Path("data/reference_scores"),
    no_merge: Annotated[bool, typer.Option("--no-merge", help="Evaluate targets without merging higher-dose PTVs")] = False,
) -> None:
    "Score reference plans with the default goal list."
    import json as _json

    from opengray.data.openkbp_opt import OpenKBPOptLoader
    from opengray.goals.defaults import openkbp_default_goals
    from opengray.goals.scoring import plan_score

    loader = OpenKBPOptLoader(cache)
    ids = loader.list_cases()
    if not ids:
        console.print(f"[red]No cached cases in {cache}. Run `opengray ingest` first.[/red]")
        raise typer.Exit(code=1)
    if n:
        ids = ids[:n]
    goals = openkbp_default_goals()
    rows = []
    for cid in ids:
        case = loader.load(cid)
        if case.reference_dose is None:
            continue
        bd = plan_score(case, case.reference_dose, goals, merge_targets=not no_merge)
        row = {"case": cid, **bd.as_row(), "hard_met": sum(g.met for g in bd.goals if g.kind == "hard"), "n_hard": bd.n_hard, "soft_met": sum(g.met for g in bd.goals if g.kind == "soft"), "n_soft": bd.n_soft}
        row["unmet"] = ";".join(f"{g.structure}.{g.metric}={g.achieved:.1f}" for g in bd.goals if not g.met)
        rows.append(row)
        console.print(f"  {cid}: score {bd.plan_score:.3f} H {bd.H:.2f} V {bd.V:.3f} R {bd.R:.2f}" + (f" GATED ({bd.gate_reason})" if bd.gated else "") + (f" unmet: {row['unmet']}" if row["unmet"] else ""))
    if not rows:
        console.print("[red]No reference plans found[/red]")
        raise typer.Exit(code=1)
    out.mkdir(parents=True, exist_ok=True)
    (out / "reference_scores.json").write_text(_json.dumps(rows, indent=1))
    scores = np.array([r["plan_score"] for r in rows])
    H = np.array([r["H"] for r in rows])
    V = np.array([r["V"] for r in rows])
    gated = [r for r in rows if r["gated"]]
    all_hard = sum(1 for r in rows if r["hard_met"] == r["n_hard"])
    summary = [
        f"Reference plans scored: {len(rows)} (merge_targets={not no_merge})",
        f"PlanScore mean {scores.mean():.3f}, median {np.median(scores):.3f}, min {scores.min():.3f}, max {scores.max():.3f}",
        f"H = 1 (all hard goals met) in {all_hard} of {len(rows)}; mean H {H.mean():.3f}; mean V {V.mean():.3f}",
        f"Gated: {len(gated)}" + (": " + ", ".join(r["case"] + " (" + r["gate_reason"] + ")" for r in gated) if gated else ""),
    ]
    unmet_counts: dict[str, int] = {}
    for r in rows:
        for item in r["unmet"].split(";") if r["unmet"] else []:
            k = item.split("=")[0]
            unmet_counts[k] = unmet_counts.get(k, 0) + 1
    lines = ["# Reference plan scores", ""] + summary + ["", "Goals unmet (count of cases):", ""]
    lines += [f"- {k}: {v}" for k, v in sorted(unmet_counts.items(), key=lambda kv: -kv[1])] or ["- none"]
    lines += ["", "| case | PlanScore | H | V | R | gated | hard met | soft met | unmet |", "|---|---:|---:|---:|---:|---|---:|---:|---|"]
    lines += [f"| {r['case']} | {r['plan_score']:.3f} | {r['H']:.2f} | {r['V']:.3f} | {r['R']:.2f} | {'yes' if r['gated'] else ''} | {r['hard_met']}/{r['n_hard']} | {r['soft_met']}/{r['n_soft']} | {r['unmet']} |" for r in rows]
    (out / "reference_scores.md").write_text("\n".join(lines) + "\n")
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        fig = make_subplots(rows=1, cols=3, subplot_titles=("PlanScore", "H (hard goals met)", "V (soft violation)"))
        fig.add_trace(go.Histogram(x=scores, nbinsx=20), row=1, col=1)
        fig.add_trace(go.Histogram(x=H, nbinsx=10), row=1, col=2)
        fig.add_trace(go.Histogram(x=V, nbinsx=20), row=1, col=3)
        fig.update_layout(showlegend=False, title=f"OpenKBP-Opt reference plans, n={len(rows)}", template="plotly_white")
        fig.write_html(out / "reference_scores.html", include_plotlyjs="cdn")
    except Exception as e:  # noqa: BLE001
        console.print(f"[yellow]plot skipped: {e}[/yellow]")
    for line in summary:
        console.print(line)
    console.print(f"Wrote {out / 'reference_scores.md'}")


@app.command()
def run(
    track: Annotated[str, typer.Option(help="T1 to T5")] = "T2",
    k: Annotated[int | None, typer.Option(help="optimize budget (T2: 3 or 5; fixed on the other tracks)")] = None,
    agent: Annotated[str, typer.Option(help="heuristic | controller | preflight | random | optuna | llm | interpreter (LLM reads, controller plans)")] = "heuristic",
    model: Annotated[str | None, typer.Option(help="llm agent: provider model id, e.g. openai/gpt-oss-120b:free")] = None,
    base_url: Annotated[str, typer.Option(help="llm agent: OpenAI-compatible endpoint")] = "https://openrouter.ai/api/v1",
    key_file: Annotated[Path, typer.Option(help="llm agent: file holding the API key (or set OPENROUTER_API_KEY)")] = Path(".secrets/openrouter.key"),
    temperature: Annotated[float, typer.Option(help="llm agent: sampling temperature")] = 0.0,
    max_model_calls: Annotated[int | None, typer.Option(help="llm agent: cap on model calls per episode (default 2 x tool cap)")] = None,
    llm_cache: Annotated[Path | None, typer.Option(help="llm agent: on-disk reply cache directory")] = Path("runs/llm_cache"),
    max_tokens: Annotated[int, typer.Option(help="llm agent: max completion tokens per model call")] = 8192,
    resume: Annotated[bool, typer.Option("--resume", help="skip episodes that already have a successful row for this agent under --out")] = False,
    strict_resume: Annotated[bool, typer.Option("--strict-resume", help="like --resume, but only rows under the current protocol id count; rows from an earlier presentation or grader are rerun and superseded")] = False,
    levels: Annotated[bool, typer.Option("--levels", help="optuna agent: also search the dose levels of upper-limit terms (0.90 to 1.0 x limit)")] = False,
    transforms: Annotated[str | None, typer.Option(help="T4: comma list of transforms (default all eight incl. the clean control); T5: arms (unplannable, tight)")] = None,
    full_factorial: Annotated[bool, typer.Option("--full-factorial", help="T4/T5: every transform or arm on every (case, seed) instead of rotating one per episode")] = False,
    feasibility: Annotated[Path, typer.Option(help="T4/T5: feasibility floors JSON (computed and cached on first use)")] = Path("data/feasibility/openkbp_opt_v1.json"),
    split: Annotated[str, typer.Option(help="train | validation | test")] = "validation",
    seeds: Annotated[int, typer.Option(help="number of seeds (0..n-1)")] = 5,
    seed_list: Annotated[str | None, typer.Option(help="explicit comma list of seeds (overrides --seeds)")] = None,
    cache: Annotated[Path, typer.Option(help="Cache directory")] = DEFAULT_CACHE,
    split_file: Annotated[Path, typer.Option(help="Split file")] = Path("data/splits/openkbp_opt_v1.json"),
    out: Annotated[Path, typer.Option(help="Runs directory")] = Path("runs"),
    concurrency: int = 1,
    cases: Annotated[str | None, typer.Option(help="Comma list of case ids to run instead of the split")] = None,
    no_merge: Annotated[bool, typer.Option("--no-merge", help="Score targets without merging higher-dose PTVs")] = False,
    no_rules: Annotated[bool, typer.Option("--no-rules", help="Disclosure experiment: withhold the acceptability rules from the case summary and the prompt (label carries rules=off)")] = False,
    max_iter: Annotated[int | None, typer.Option(help="Solver study: iteration cap per optimize (default 500); use a separate --out")] = None,
    rel_tol: Annotated[float | None, typer.Option(help="Solver study: relative tolerance per optimize (default 1e-4); use a separate --out")] = None,
    deliverable: Annotated[bool, typer.Option("--deliverable", help="Fluence-complexity experiment targeting weights <=15 and SPG <=65; check final limits after normalization. Does not certify clinical deliverability. Use a separate --out (label carries solver=deliverable)")] = False,
    priorities: Annotated[str | None, typer.Option(help="heuristic | controller | preflight: starting priorities, flat (every term 1, the default) or hard-first (hard goals 100, soft goals 1)")] = None,
    checks: Annotated[str | None, typer.Option(help="preflight: 'arithmetic' adds the two summary-only consistency checks (coverage against the External cap; an organ limit against the coverage its voxels inside a target must receive)")] = None,
    summary: Annotated[str | None, typer.Option(help="interpreter: what the model reads, 'reduced' (names and volumes; the default) or 'full' (every case-summary field: voxel counts, overlap fractions, distances, beam geometry). The prompt is the same under both")] = None,
) -> None:
    """Run a track with a reference agent; writes manifest, JSONL log, results, and leaderboard."""
    from opengray.agents.base import AgentSpec
    from opengray.data.openkbp_opt import OpenKBPOptLoader
    from opengray.env.tracks import track_config, with_score
    from opengray.goals.defaults import openkbp_default_goals
    from opengray.runner.results import write_leaderboard
    from opengray.runner.run import RunConfig
    from opengray.runner.run import run as _run

    params: dict[str, Any] = {}
    options: dict[str, Any] = {}
    if agent == "optuna":
        if levels:
            params["levels"] = True
    if agent in ("llm", "interpreter"):
        if not model:
            raise typer.BadParameter(f"--agent {agent} needs --model")
        params["model"] = model
        if temperature:
            params["temperature"] = temperature
        options.update({"base_url": base_url, "key_file": key_file, "max_model_calls": max_model_calls, "cache_dir": llm_cache, "max_tokens": max_tokens})
    trk = track_config(track, k)
    if agent == "optuna":
        trk = with_score(trk)
    if no_rules:
        from opengray.env.tracks import without_rules

        trk = without_rules(trk)
        params["rules"] = "off"  # in the label, so rows under the two conditions never dedupe against each other
    if priorities is not None:
        if agent not in ("heuristic", "controller", "preflight"):
            raise typer.BadParameter("--priorities applies to the heuristic, controller and preflight agents")
        if priorities not in ("flat", "hard-first"):
            raise typer.BadParameter("--priorities must be flat or hard-first")
        if priorities != "flat":
            params["priorities"] = priorities  # flat is the default and keeps the label unchanged
    if checks is not None:
        if agent != "preflight" or checks != "arithmetic":
            raise typer.BadParameter("--checks arithmetic applies to the preflight agent")
        params["checks"] = "arithmetic"
    if summary is not None:
        if agent != "interpreter" or summary not in ("reduced", "full"):
            raise typer.BadParameter("--summary reduced|full applies to the interpreter agent")
        if summary != "reduced":
            params["summary"] = summary  # reduced is the default and keeps the label unchanged
    solver = None
    if max_iter is not None or rel_tol is not None or deliverable:
        from opengray.physics.solver import SolverConfig, deliverable_config

        overrides = {**({"max_iter": max_iter} if max_iter is not None else {}), **({"rel_tol": rel_tol} if rel_tol is not None else {})}
        solver = deliverable_config(**overrides) if deliverable else SolverConfig(**overrides)
        if deliverable:
            params["solver"] = "deliverable"
        if Path(out).resolve() == Path("runs").resolve():
            raise typer.BadParameter("a solver study must write to its own --out (its rows carry another protocol id and must not enter the leaderboard's runs directory)")
    cfg = RunConfig(
        track=trk,
        agent=AgentSpec(agent, params),
        split=split,
        seeds=[int(x) for x in seed_list.split(",")] if seed_list else list(range(seeds)),
        out_dir=out,
        split_file=split_file,
        concurrency=concurrency,
        merge_targets=not no_merge,
        case_ids=[c.strip() for c in cases.split(",")] if cases else None,
        agent_options=options,
        resume=resume or strict_resume,
        strict_resume=strict_resume,
        transforms=[t.strip() for t in transforms.split(",")] if transforms else None,
        rotate_transforms=not full_factorial,
        feasibility_file=feasibility,
        solver=solver,
    )
    loader = OpenKBPOptLoader(cache)
    goals = openkbp_default_goals()
    t0 = time.perf_counter()

    def progress(row: dict[str, Any]) -> None:
        sc = row["plan_score"]
        t4 = f" t4 {row['t4_score']:.3f} {'ok' if row.get('t4_correct') else 'wrong'}" if row.get("transform") and row["track"] == "T4" else ""
        if row.get("t5_arm"):
            t4 = f" t5 {row['t5_arm']} {row['t5_score']:.3f} {'ok' if row.get('t5_correct') else 'wrong'}"
        if row.get("t3_update"):
            t4 = f" t3 [{row['t3_update']}]"
        console.print(f"  {row['episode_id']}: {row['outcome']} score {sc:.3f} H {row['H']:.2f} V {row['V']:.3f}{t4} ({row['optimize_calls']} opt, {row['wall_s']} s)" + (f" [red]{row['error'].splitlines()[0]}[/red]" if row["error"] else ""))

    res = _run(cfg, loader, goals, progress=progress)
    if res.skipped:
        console.print(f"  resumed: skipped {len(res.skipped)} episodes already completed")
    if not res.rows:
        console.print(f"Run {res.run_id}: nothing to do")
        return
    import pandas as pd

    df = pd.DataFrame(res.rows)
    write_leaderboard(df, res.run_dir / "leaderboard.json")
    scores = df["plan_score"].to_numpy(dtype=float)
    console.print(f"Run {res.run_id}: {len(df)} episodes in {time.perf_counter() - t0:.0f} s; mean PlanScore {np.nanmean(scores):.3f}; {res.run_dir}")


@app.command()
def models(
    base_url: Annotated[str, typer.Option(help="OpenAI-compatible endpoint")] = "https://openrouter.ai/api/v1",
    free: Annotated[bool, typer.Option("--free", help="only ids ending in :free")] = False,
    tools: Annotated[bool, typer.Option("--tools/--no-tools", help="only models that advertise tool calling")] = True,
    limit: int = 60,
) -> None:
    """List models on the endpoint (OpenRouter format), optionally only free ones with tool calling."""
    import httpx

    r = httpx.get(base_url.rstrip("/") + "/models", timeout=30)
    r.raise_for_status()
    rows = []
    for m in r.json().get("data", []):
        mid = m.get("id", "")
        params = m.get("supported_parameters") or []
        if free and not mid.endswith(":free"):
            continue
        if tools and "tools" not in params:
            continue
        pricing = m.get("pricing") or {}
        rows.append((mid, m.get("context_length"), pricing.get("prompt"), pricing.get("completion")))
    rows.sort()
    table = Table(title=f"{len(rows)} models" + (" (free, tools)" if free and tools else ""))
    for col in ("id", "context", "$/prompt tok", "$/completion tok"):
        table.add_column(col)
    for mid, ctx, pp, pc in rows[:limit]:
        table.add_row(mid, str(ctx), str(pp), str(pc))
    console.print(table)


@app.command()
def serve(
    transport: Annotated[str, typer.Option(help="stdio | http (Streamable HTTP at /mcp)")] = "stdio",
    host: str = "127.0.0.1",
    port: int = 8765,
    cache: Annotated[Path, typer.Option(help="Cache directory")] = DEFAULT_CACHE,
    track: Annotated[str, typer.Option(help="default track for start_episode")] = "T2",
    k: Annotated[int, typer.Option(help="default optimize budget for T2")] = 3,
    log_dir: Annotated[Path | None, typer.Option(help="Where episode logs and results go (default runs/mcp)")] = None,
) -> None:
    """Serve the nine planning tools over MCP (stdio for Claude Code and similar; http for networked clients)."""
    from opengray.data.openkbp_opt import OpenKBPOptLoader
    from opengray.env.mcp_server import serve as _serve
    from opengray.goals.defaults import openkbp_default_goals

    loader = OpenKBPOptLoader(cache)
    if not loader.list_cases():
        console.print(f"[red]No cached cases under {cache}; run `opengray ingest` first.[/red]")
        raise typer.Exit(code=1)
    if transport == "http":
        console.print(f"OpenGray MCP server on http://{host}:{port}/mcp ({len(loader.list_cases())} cases)")
    _serve(loader, openkbp_default_goals(), transport=transport, host=host, port=port, log_dir=log_dir or Path("runs/mcp"), default_track=track, default_k=k)


@app.command()
def rescore(
    runs: Annotated[Path, typer.Option(help="Runs directory")] = Path("runs"),
    cache: Annotated[Path, typer.Option(help="Cache directory")] = DEFAULT_CACHE,
    cases: Annotated[str | None, typer.Option(help="Comma list of case ids to do in this call (resumable)")] = None,
    quiet: bool = False,
) -> None:
    """Re-score every stored episode from its saved beam weights under the current scoring rules."""
    from opengray.data.openkbp_opt import OpenKBPOptLoader
    from opengray.goals.defaults import openkbp_default_goals
    from opengray.runner.rescore import rescore_all

    case_ids = [c.strip() for c in cases.split(",")] if cases else None
    report_rows = rescore_all(runs, OpenKBPOptLoader(cache), openkbp_default_goals(), progress=None if quiet else lambda s: console.print("  " + s), cases=case_ids)
    n_re = sum(1 for r in report_rows if r["status"] == "rescored")
    n_part = sum(1 for r in report_rows if r["status"] == "partial")
    n_ch = sum(r.get("changed", 0) for r in report_rows)
    remaining = sorted({c for r in report_rows for c in r.get("remaining_cases", [])})
    console.print(f"Rescored {n_re} run(s) ({n_ch} episode score(s) changed), {n_part} partial, {sum(1 for r in report_rows if r['status'] == 'current')} already current." + (f" Remaining cases: {', '.join(remaining)}" if remaining else ""))


@app.command()
def feasibility(
    cache: Annotated[Path, typer.Option(help="Cache directory")] = DEFAULT_CACHE,
    split: Annotated[str, typer.Option(help="train | validation | test")] = "validation",
    split_file: Annotated[Path, typer.Option(help="Split file")] = Path("data/splits/openkbp_opt_v1.json"),
    out: Annotated[Path, typer.Option(help="Floors JSON (resumable per case)")] = Path("data/feasibility/openkbp_opt_v1.json"),
    cases: Annotated[str | None, typer.Option(help="Comma list of case ids instead of the split")] = None,
    structure: str = "SpinalCord",
    force: Annotated[bool, typer.Option("--force", help="Recompute cases already in the floors file (drops their certificate; run opengray certify again)")] = False,
) -> None:
    """Feasibility solve per case: the lowest serial-organ dose compatible with target coverage (Track 4 infeasible_goals, Track 5)."""
    from opengray.data.openkbp_opt import OpenKBPOptLoader
    from opengray.env.tracks import load_split
    from opengray.goals.defaults import openkbp_default_goals
    from opengray.physics.feasibility import (
        coverage_floor,
        load_floors,
        save_floors,
        summarize,
        tightened_limit,
    )

    loader = OpenKBPOptLoader(cache)
    goals = openkbp_default_goals()
    ids = [c.strip() for c in cases.split(",")] if cases else load_split(split_file, split)
    floors = load_floors(out)
    for cid in ids:
        if structure in floors.get(cid, {}) and not force:
            rec = floors[cid][structure]
            console.print(f"  {cid}: {structure} floor {rec['floor_gy']:.2f} Gy (cached)")
            continue
        case = loader.load(cid)
        t0 = time.perf_counter()
        rec = coverage_floor(case, goals, structure=structure)
        if force and structure in floors.get(cid, {}):
            rec["previous"] = {k: v for k, v in floors[cid][structure].items() if k in ("floor_gy", "best_achieved_gy", "bracket_low_gy", "solves", "witness_file", "certified_below_gy")}
        floors.setdefault(cid, {})[structure] = rec
        save_floors(out, floors)
        tight = tightened_limit(rec["floor_gy"], rec["original_gy"]) if rec["original_gy"] else None
        console.print(f"  {cid}: {structure} floor {rec['floor_gy']:.2f} Gy (limit {rec['original_gy']}, others met {rec['others_met']}, max {rec['global_max_gy']:.1f} Gy), infeasible arm {0.7 * rec['floor_gy']:.1f}, tightened {tight}, {rec['solves']} solves, {time.perf_counter() - t0:.1f} s")
        del case
    console.print(json.dumps(summarize(floors)))


@app.command()
def sensitivity(
    runs: Annotated[Path, typer.Option(help="Runs directory")] = Path("runs"),
    cache: Annotated[Path, typer.Option(help="Cache directory")] = DEFAULT_CACHE,
    out: Annotated[Path, typer.Option(help="Per-case score tables (resumable)")] = Path("runs/_analysis/sensitivity"),
    summary: Annotated[Path | None, typer.Option(help="Write the summary JSON here (default <out>/summary.json)")] = None,
    cases: Annotated[str | None, typer.Option(help="Comma list of case ids to do in this call")] = None,
    quiet: bool = False,
) -> None:
    """E5: score every stored plan under the alternative weightings and report rank stability."""
    from opengray.data.openkbp_opt import OpenKBPOptLoader
    from opengray.goals.defaults import openkbp_default_goals
    from opengray.runner.sensitivity import sensitivity_scores, sensitivity_summary

    case_ids = [c.strip() for c in cases.split(",")] if cases else None
    files = sensitivity_scores(runs, OpenKBPOptLoader(cache), openkbp_default_goals(), out, cases=case_ids, progress=None if quiet else lambda s: console.print("  " + s))
    rep = sensitivity_summary(out)
    path = summary or (out / "summary.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rep, indent=1))
    console.print(f"{len(files)} case table(s), {rep['n_episodes']} episodes; summary at {path}")
    for track, t in rep["tracks"].items():
        taus = next(v for k, v in t.items() if k.startswith("kendall_tau_vs_"))
        console.print(f"  {track}: rank stability vs v1 " + ", ".join(f"{k} {v:.2f}" for k, v in taus.items() if k != "v1"))


@app.command()
def spg(
    cache: Annotated[Path, typer.Option(help="Cache directory")] = DEFAULT_CACHE,
    runs: Annotated[Path, typer.Option(help="Runs directory")] = Path("runs"),
    out: Annotated[Path, typer.Option(help="Output CSV (per plan) and JSON summary beside it")] = Path("data/results/complexity_v1.csv"),
    experiments: Annotated[Path | None, typer.Option(help="experiments-data.zip: also measure the published paper plans")] = None,
    paper_model: Annotated[str, typer.Option(help="Paper-plan model/set to measure")] = "absolute_max/set_1",
    cases: Annotated[str | None, typer.Option(help="Comma list of case ids (default: every case with stored plans)")] = None,
) -> None:
    """Fluence complexity (OpenKBP-Opt's sum of positive gradients) of every stored plan and of the published plans."""
    from opengray.data.openkbp_opt import OpenKBPOptLoader
    from opengray.physics.complexity import SPG_LIMIT_OPENKBP_OPT, complexity_report
    from opengray.runner.results import load_runs

    loader = OpenKBPOptLoader(cache)
    df = load_runs(runs)
    df = df[(df["outcome"] == "submitted") & df["error"].isna()] if not df.empty else df
    wanted = {c.strip() for c in cases.split(",")} if cases else None
    rows: list[dict[str, Any]] = []
    case_ids = sorted(df["case_id"].unique()) if not df.empty else []
    for cid in case_ids:
        if wanted and cid not in wanted:
            continue
        case = loader.load(cid)
        sub = df[df["case_id"] == cid]
        for r in sub.itertuples(index=False):
            wp = Path(runs) / r.run_id / "final_w" / f"{r.episode_id}.npy"
            if not wp.exists():
                continue
            w = np.load(wp)
            rep = complexity_report(case.beamlets, w, case.dose(w))
            rows.append({"case_id": cid, "agent": r.agent, "track": r.track, "k": int(r.k), "episode_id": r.episode_id, "seed": int(r.seed), "plan_score": float(r.plan_score), **rep})
        if experiments is not None:
            from opengray.physics.validation import read_paper_plan

            try:
                fl, _di, _dv = read_paper_plan(experiments, paper_model, cid)
                w = np.zeros(case.n_beamlets)
                w[fl[:, 0].astype(int)] = fl[:, 1]
                rep = complexity_report(case.beamlets, w, case.dose(w))
                rows.append({"case_id": cid, "agent": f"paper_plan:{paper_model}", "track": "published", "k": 0, "episode_id": f"paper-{cid}", "seed": 0, "plan_score": float("nan"), **rep})
            except FileNotFoundError:
                console.print(f"  no paper plan for {cid} under {paper_model}")
        console.print(f"  {cid}: {len(sub)} stored plans measured")
        del case
    import pandas as pd

    out.parent.mkdir(parents=True, exist_ok=True)
    tab = pd.DataFrame(rows)
    tab.to_csv(out, index=False)
    summary: dict[str, Any] = {"spg_limit_openkbp_opt": SPG_LIMIT_OPENKBP_OPT, "n_plans": int(len(tab)), "by_agent": {}}
    if not tab.empty:
        for (agent, track), g in tab.groupby(["agent", "track"]):
            summary["by_agent"][f"{agent}|{track}"] = {"n": int(len(g)), "spg_median": float(g["spg"].median()), "spg_max": float(g["spg"].max()), "within_65": float(g["spg_within_limit"].mean()), "w_max_median": float(g["w_max"].median()), "w_within_15": float(g["w_within_limit"].mean()), "dose_within_82": float(g["dose_within_limit"].mean()) if "dose_within_limit" in g else None}
            console.print(f"  {agent} {track}: n={len(g)} SPG median {g['spg'].median():.1f} max {g['spg'].max():.1f}, within 65: {g['spg_within_limit'].mean():.0%}; w_max median {g['w_max'].median():.2f}, within 15: {g['w_within_limit'].mean():.0%}")
    out.with_suffix(".json").write_text(json.dumps(summary, indent=1))
    console.print(f"Wrote {out} and {out.with_suffix('.json')}")


@app.command()
def certify(
    cache: Annotated[Path, typer.Option(help="Cache directory")] = DEFAULT_CACHE,
    floors: Annotated[Path, typer.Option(help="Floors JSON to read and update (opengray feasibility)")] = Path("data/feasibility/openkbp_opt_v1.json"),
    cases: Annotated[str | None, typer.Option(help="Comma list of case ids (default: every case in the floors file)")] = None,
    structure: str = "SpinalCord",
    target_cap: Annotated[int, typer.Option(help="Target voxels nearest the organ carried into the tail constraints")] = 2000,
    target_levels: Annotated[str, typer.Option(help="Lower-tail levels as multiple:scope pairs (scope near = the target_cap voxels nearest the organ, all = every merged-target voxel)")] = "2:near,10:all",
    tail_multiples: Annotated[str, typer.Option(help="Upper-tail sizes as multiples of the D<v>cc voxel count")] = "2,4,10,30,100,300",
    method: Annotated[str, typer.Option(help="scipy linprog method: highs, highs-ds, highs-ipm")] = "highs-ipm",
    time_limit: Annotated[float | None, typer.Option(help="Seconds per LP (HiGHS time limit)")] = None,
    no_repair: Annotated[bool, typer.Option("--no-repair", help="Skip the witness repair sweep from the LP fluence")] = False,
    force: Annotated[bool, typer.Option("--force", help="Recompute cases that already carry a certificate")] = False,
) -> None:
    """LP feasibility certificates: a lower bound on the serial-organ D<v>cc that every plan meeting the other hard goals must respect; fills certified_below_gy in the floors file."""
    from opengray.data.openkbp_opt import OpenKBPOptLoader
    from opengray.goals.defaults import openkbp_default_goals
    from opengray.physics.certificate import certify as certify_case
    from opengray.physics.certificate import repair_witness
    from opengray.physics.feasibility import (
        feasibility_status,
        infeasible_limit,
        load_floors,
        save_floors,
        tightened_limit,
        witness_dir,
    )

    loader = OpenKBPOptLoader(cache)
    goals = openkbp_default_goals()
    recs = load_floors(floors)
    if not recs:
        console.print(f"[red]no floors at {floors}; run opengray feasibility first[/red]")
        raise typer.Exit(1)
    ids = [c.strip() for c in cases.split(",")] if cases else sorted(recs)
    tlev = tuple((int(x.split(":")[0]), x.split(":")[1]) for x in target_levels.split(","))
    kmult = tuple(int(x) for x in tail_multiples.split(","))
    for cid in ids:
        rec = recs.get(cid, {}).get(structure)
        if rec is None:
            console.print(f"  {cid}: no {structure} floor record, skipped")
            continue
        if rec.get("certificate") and not force:
            console.print(f"  {cid}: certified below {rec.get('certified_below_gy')} Gy (cached)")
            continue
        case = loader.load(cid)
        cert = certify_case(case, goals, structure=structure, metric=rec.get("metric", "D0.1cc"), tail_multiples=kmult, target_cap=target_cap, target_levels=tlev, method=method, time_limit=time_limit, keep_w=True)
        best_lp_w = cert.pop("best_lp_w", None)
        cert.pop("tail_w", None)
        best = float(rec.get("best_achieved_gy", rec.get("floor_gy")))
        rec.setdefault("sweep_best_achieved_gy", best)  # the solver sweep's own witness, kept
        # A better witness: an LP fluence that meets every other hard goal, or the repair sweep
        # from the LP fluence with the lowest organ dose. The old witness file is kept as .sweep.
        new_w, source, new_best = None, None, best
        lp_ok = [p for p in cert.get("lp_plans", []) if p["others_met"] and p["organ_gy"] < best - 1e-6]
        if lp_ok and best_lp_w is not None:
            chk = min(lp_ok, key=lambda p: p["organ_gy"])
            new_w, source, new_best = best_lp_w, "lp", float(chk["organ_gy"])
        elif best_lp_w is not None and not no_repair and not cert.get("relaxation_infeasible"):
            rep = repair_witness(case, goals, best_lp_w, target_gy=best, structure=structure, metric=rec.get("metric", "D0.1cc"))
            cert["repair"] = {"sweep": rep["sweep"], "found_gy": rep["best"]["organ_gy"] if rep["best"] else None}
            if rep["best"] and rep["best"]["organ_gy"] < best - 1e-6:
                new_w, source, new_best = rep["best"]["w"], "repair", float(rep["best"]["organ_gy"])
        if new_w is not None:
            wd = witness_dir(floors)
            old = wd / f"{cid}.{structure}.npy"
            if old.exists() and not (wd / f"{cid}.{structure}.sweep.npy").exists():
                old.rename(wd / f"{cid}.{structure}.sweep.npy")
            rec["witness_w"] = new_w
            rec["witness_source"] = source
            rec["best_achieved_gy"] = round(new_best, 3)
            rec["floor_gy"] = rec["best_achieved_gy"]
            best = rec["best_achieved_gy"]
        conflict = cert.get("lower_bound_gy") is not None and cert["lower_bound_gy"] > best + 1e-6
        cert["conflicts_with_witness"] = bool(conflict)
        rec["certificate"] = cert
        rec["certified_below_gy"] = None if conflict else cert.get("certified_below_gy")
        recs[cid][structure] = rec
        save_floors(floors, recs)
        limit = infeasible_limit(best, 0.7, rec["certified_below_gy"])
        status = feasibility_status(limit, best, rec["certified_below_gy"])
        tight = tightened_limit(best, float(rec["original_gy"])) if rec.get("original_gy") else None
        if new_w is not None:
            console.print(f"  {cid}: witness improved by the {source} plan: {rec['sweep_best_achieved_gy']:.2f} -> {best:.2f} Gy (tight arm now {tight})")
        if conflict:
            console.print(f"  [red]{cid}: certificate {cert['lower_bound_gy']:.3f} Gy exceeds the witness {best:.3f} Gy: a bug in one of them; certificate withheld[/red]")
        elif cert.get("reason"):
            console.print(f"  {cid}: no certificate ({cert['reason']})")
        else:
            console.print(f"  {cid}: bound {cert['lower_bound_gy']} Gy, certified below {rec['certified_below_gy']} Gy (witness {best:.2f}); unplannable arm {limit} Gy is {status}; LP {cert['lp_rows']} x {cert['lp_cols']}, nnz {cert['lp_nnz']}, {cert['seconds']} s")
        del case
    n_cert = sum(1 for r in recs.values() if r.get(structure, {}).get("certified_below_gy") is not None)
    console.print(f"{n_cert} of {len(recs)} cases carry a certificate; floors at {floors}")


@app.command("regrade")
def regrade_cmd(
    runs: Annotated[Path, typer.Option(help="Runs directory")] = Path("runs"),
    track: Annotated[str, typer.Option(help="T4, T5 or both")] = "both",
) -> None:
    """Re-grade stored Track 4 and 5 rows under the current graders from the episode logs (no rerun, no solve)."""
    from opengray.runner.regrade import regrade_track4, regrade_track5

    for name, fn in (("T4", regrade_track4), ("T5", regrade_track5)):
        if track not in ("both", name):
            continue
        rows = fn(runs, progress=lambda s: console.print("  " + s))
        n = sum(1 for r in rows if r["status"] == "regraded")
        console.print(f"{name}: {n} runs re-graded, {sum(1 for r in rows if r['status'] == 'current')} already current")


@app.command("mark-provider-failures")
def mark_provider_failures_cmd(
    runs: Annotated[Path, typer.Option(help="Runs directory")] = Path("runs"),
) -> None:
    "Convert legacy provider-error automatic submissions into error rows eligible for resume."
    from opengray.runner.regrade import mark_provider_failures

    rows = mark_provider_failures(runs, progress=lambda s: console.print("  " + s))
    console.print(f"{sum(r['n'] for r in rows)} rows marked in {sum(1 for r in rows if r['status'] == 'marked')} runs")


@app.command("regrade-track5", hidden=True)
def regrade_track5_cmd(
    runs: Annotated[Path, typer.Option(help="Runs directory")] = Path("runs"),
) -> None:
    """Alias of ``regrade --track T5``."""
    regrade_cmd(runs, "T5")


@app.command("solver-budget")
def solver_budget(
    runs: Annotated[Path, typer.Option(help="Runs directory holding the episodes to replay")] = Path("runs"),
    cache: Annotated[Path, typer.Option(help="Cache directory")] = DEFAULT_CACHE,
    out: Annotated[Path, typer.Option(help="Per-(episode, arm) records, resumable")] = Path("runs/_analysis/solver_budget"),
    agents: Annotated[str, typer.Option(help="Comma list of agent labels to replay")] = "heuristic",
    tracks: Annotated[str, typer.Option(help="Comma list of tracks")] = "T1,T2",
    cases: Annotated[str | None, typer.Option(help="Comma list of case ids to do in this call")] = None,
    all_seeds: Annotated[bool, typer.Option("--all-seeds", help="Replay every seed (default: one per case, for deterministic agents)")] = False,
    summary: Annotated[Path | None, typer.Option(help="Write the summary JSON here too")] = None,
    redo: Annotated[str | None, typer.Option(help="Comma list of arms to recompute even when their records exist (e.g. cap500 after a change to the repeat controls)")] = None,
) -> None:
    """Solver budget study: replay stored objective sequences at other iteration caps and score the submitted plans."""
    from opengray.data.openkbp_opt import OpenKBPOptLoader
    from opengray.goals.defaults import openkbp_default_goals
    from opengray.runner.solver_budget import run_study, summarize

    case_ids = [c.strip() for c in cases.split(",")] if cases else None
    table = run_study(runs, OpenKBPOptLoader(cache), openkbp_default_goals(), out, agents=tuple(a.strip() for a in agents.split(",")), tracks=tuple(t.strip() for t in tracks.split(",")), cases=case_ids, one_seed_per_case=not all_seeds, progress=lambda s: console.print("  " + s), redo_arms=tuple(x.strip() for x in redo.split(",")) if redo else ())
    rep = summarize(out)
    if summary:
        summary.parent.mkdir(parents=True, exist_ok=True)
        summary.write_text(json.dumps(rep, indent=1))
    console.print(f"{rep.get('n_episodes', 0)} episode(s) replayed; table at {table}, summary at {out / 'summary.json'}")
    for track, t in rep.get("tracks", {}).items():
        for arm, a in t["arms"].items():
            console.print(f"  {track} {arm}: score {a['plan_score_mean']:.3f} ({a['plan_score_mean_vs_reference']:+.3f} vs reference, max |change| {a['plan_score_max_abs_change']:.3f}), metric max change {a['metric_max_abs_change_gy']:.2f} Gy, gate flips {a['n_gate_flips']}, hard-count changes {a['n_hard_count_changed']}, iterations {a['iterations_mean']:.0f}, residual median {a['grad_residual_median']:.2e}")
            if a.get("repeat"):
                r = a["repeat"]
                console.print(f"    repeat solve from the submitted plan ({r.get('n_submitted_normalized', '?')} normalized): score change mean {r['plan_score_mean_change']:+.3f} max |{r['plan_score_max_abs_change']:.3f}|, metric max change {r['metric_max_abs_change_gy']:.2f} Gy (median {r['metric_median_max_abs_change_gy']:.2f}), gate flips {r['n_gate_flips']}")
            if a.get("repeat_pre_normalize"):
                r = a["repeat_pre_normalize"]
                console.print(f"    repeat solve from the last optimize's own result: score change mean {r['plan_score_mean_change']:+.3f} max |{r['plan_score_max_abs_change']:.3f}|, metric max change {r['metric_max_abs_change_gy']:.2f} Gy (median {r['metric_median_max_abs_change_gy']:.2f}), gate flips {r['n_gate_flips']}, converged {r['fraction_converged']:.2f}")


@app.command()
def disclosure(
    runs: Annotated[Path, typer.Option(help="Runs directory of the experiment (both conditions)")] = Path("runs/_experiments/disclosure_v1"),
    out: Annotated[Path, typer.Option(help="Summary JSON")] = Path("data/results/disclosure_v1.json"),
    track: str = "T1",
) -> None:
    """Rule-disclosure experiment: per model, case-level means under both conditions, the paired difference, and the gate, normalize and escalation rates."""
    from opengray.runner.disclosure import disclosure_table

    rep = disclosure_table(runs, track=track)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=1))
    console.print(f"{rep['n_rows']} rows on {track}; summary at {out}")
    def f3(x: Any) -> str:
        return "n/a" if x is None else f"{x:.3f}"

    for model, m in rep["models"].items():
        for cond, c in m["conditions"].items():
            ci = c["plan_score_ci95"]
            console.print(f"  {model} {cond}: {f3(c['plan_score_case_mean'])} [{f3(ci[0])}, {f3(ci[1])}] over {c['n_cases']} cases; gated {c['n_gated']} (case-mean rate {f3(c['gate_rate_case_mean'])}), normalize rate {f3(c['normalize_rate'])}, escalated {c['n_escalated']}, auto-submitted {c['n_auto_submitted']}, errors {c['n_errors']}")
        pd_ = m.get("paired_disclosed_minus_withheld")
        if pd_:
            console.print(f"    paired disclosed minus withheld: {pd_['mean_diff']:+.3f} [{f3(pd_['diff_ci95'][0])}, {f3(pd_['diff_ci95'][1])}] ({pd_['cases_above']} above / {pd_['cases_level']} level / {pd_['cases_below']} below)")


@app.command()
def contradictions(
    cache: Annotated[Path, typer.Option(help="Cache directory")] = DEFAULT_CACHE,
    split: Annotated[str, typer.Option(help="train | validation | test")] = "validation",
    split_file: Annotated[Path, typer.Option(help="Split file")] = Path("data/splits/openkbp_opt_v1.json"),
    cases: Annotated[str | None, typer.Option(help="Comma list of case ids instead of the split")] = None,
    out: Annotated[Path | None, typer.Option(help="Write the per-case table here as JSON")] = None,
) -> None:
    """Track 5 constructibility per case: the overlap and coverage_cap contradictions the contours and goals certify, and the limits each arm would present."""
    from opengray.data.openkbp_opt import OpenKBPOptLoader
    from opengray.env.tracks import load_split
    from opengray.goals.defaults import openkbp_default_goals
    from opengray.goals.scoring import GLOBAL_MAX_FACTOR
    from opengray.physics.contradictions import (
        coverage_cap_contradiction,
        overlap_contradictions,
        presented_coverage,
        presented_overlap_limit,
    )

    loader = OpenKBPOptLoader(cache)
    goals = openkbp_default_goals()
    ids = [c.strip() for c in cases.split(",")] if cases else load_split(split_file, split)
    table: dict[str, Any] = {}
    for cid in ids:
        case = loader.load(cid)
        opts = overlap_contradictions(case, goals)
        chosen = None
        for o in opts:
            lim = presented_overlap_limit(o["certified_below_gy"], o["organ_limit_gy"])
            if lim is not None:
                chosen = {**o, "presented_limit_gy": lim}
                break
        cc = coverage_cap_contradiction(case, goals)
        cov = None
        if cc is not None:
            v = presented_coverage(cc["certified_above_gy"], case.prescriptions[cc["target"]], GLOBAL_MAX_FACTOR)
            cov = {**cc, "presented_coverage_gy": v}
        table[cid] = {"overlap_options": opts, "overlap": chosen, "coverage_cap": cov}
        o_txt = f"overlap: {chosen['organ']} in {chosen['target']} ({chosen['overlap_voxels']} voxels vs allowance {chosen['cold_allowance']} + {chosen['organ_allowance']}), limit {chosen['organ_limit_gy']:g} -> {chosen['presented_limit_gy']:g} Gy (certified below {chosen['certified_below_gy']:g})" if chosen else f"overlap: none ({len(opts)} pairs, none with a usable limit)"
        c_txt = f"coverage_cap: {cov['target']} D99 {cov['published_gy']:g} -> {cov['presented_coverage_gy']} Gy (certified above {cov['certified_above_gy']:g}; N {cov['n_total']}, m {cov['cold_allowance']})" if cov else "coverage_cap: none"
        console.print(f"  {cid}: {o_txt}; {c_txt}")
        del case
    n_o = sum(1 for v in table.values() if v["overlap"])
    n_c = sum(1 for v in table.values() if v["coverage_cap"] and v["coverage_cap"]["presented_coverage_gy"] is not None)
    console.print(f"{n_o} of {len(table)} cases carry the overlap arm, {n_c} the coverage_cap arm")
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(table, indent=1))


@app.command()
def campaign(
    name: Annotated[str, typer.Argument(help="Campaign name (output directory under --out)")],
    runs: Annotated[Path, typer.Option(help="Runs directory")] = Path("runs"),
    out: Annotated[Path, typer.Option(help="Where campaign directories go")] = Path("data/results/campaigns"),
    run_ids: Annotated[Path | None, typer.Option(help="File with one run id per line (default: every run under --runs)")] = None,
    allow_mixed: Annotated[bool, typer.Option("--allow-mixed", help="Build even when a cell mixes protocol ids (development only)")] = False,
) -> None:
    """Table of record from an explicit list of runs: one protocol per cell, attempts and failures counted, manifest written."""
    from opengray.runner.campaign import CampaignError, write_campaign

    ids = [x.strip() for x in run_ids.read_text().splitlines() if x.strip() and not x.startswith("#")] if run_ids else None
    try:
        path = write_campaign(runs, out / name, ids, name=name, allow_mixed=allow_mixed)
    except CampaignError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1) from None
    manifest = json.loads(path.read_text())
    console.print(f"Campaign {name}: {len(manifest['runs'])} runs, {len(manifest['cells'])} cells; {path}")
    for cell in manifest["cells"]:
        console.print(f"  {cell['agent']} {cell['track']} k={cell['k']}: {cell['n_episodes']} episodes from {cell['n_attempts']} attempts ({cell['n_error_attempts']} failed, {cell['n_auto_submitted']} auto-submitted, {cell['n_escalated']} escalated); protocol {','.join(cell['protocol_ids'])}")
    if manifest["mixed_protocols"]:
        console.print(f"[yellow]mixed protocols: {manifest['mixed_protocols']}[/yellow]")


@app.command()
def report(
    runs: Annotated[Path, typer.Option(help="Runs directory")] = Path("runs"),
    out: Annotated[Path, typer.Option(help="Site directory")] = Path("site"),
    cache: Annotated[Path | None, typer.Option(help="Cache directory (enables DVH drill-down)")] = None,
    anchor: Annotated[str | None, typer.Option(help="Anchor agent label for the paired table (default: heuristic, or its tagged label inside an experiment directory)")] = None,
) -> None:
    """Build the static dashboard and leaderboard.json from every run under --runs."""
    from opengray.runner.report import build_report

    page = build_report(runs, out, cache, anchor=anchor)
    console.print(f"Wrote {page} and {out / 'leaderboard.json'}")


@app.command("validate-metrics")
def validate_metrics(
    experiments: Annotated[Path, typer.Argument(help="experiments-data.zip", exists=True)],
    cache: Annotated[Path, typer.Option(help="Cache directory")] = DEFAULT_CACHE,
    cases: Annotated[str | None, typer.Option(help="Comma list of case ids (default all)")] = None,
    tol: Annotated[float, typer.Option(help="Max absolute difference tolerated, Gy")] = 1e-3,
) -> None:
    """M2 evidence: reproduce the authors' reference-plan metric tables (disjoint and merged targets)."""
    from opengray.data.openkbp_opt import OpenKBPOptLoader
    from opengray.goals.reference_check import compare_reference_metrics

    loader = OpenKBPOptLoader(cache)
    ids = [c.strip() for c in cases.split(",")] if cases else None
    failed = False
    for merged in (False, True):
        r = compare_reference_metrics(loader, experiments, merged=merged, case_ids=ids)
        table = Table(title=f"{r['table']} ({'merged' if merged else 'disjoint'} targets): {r['n_compared']} values, {r['n_cases']} cases, {r['n_blank']} blank, {r['n_missing']} not comparable")
        for col in ("metric", "n", "max abs diff Gy", "mean abs diff Gy"):
            table.add_column(col, justify="right" if col != "metric" else "left")
        for m, v in r["per_metric"].items():
            table.add_row(m, str(v["n"]), f"{v['max_abs_diff']:.2e}", f"{v['mean_abs_diff']:.2e}")
        console.print(table)
        bad = [w for w in r["worst"] if w["diff"] > tol]
        for w in bad[:5]:
            console.print(f"[red]{w['case']} {w['structure']} {w['metric']}: theirs {w['theirs']:.4f} ours {w['ours']:.4f}[/red]")
        failed = failed or bool(bad)
    if failed:
        raise typer.Exit(code=1)
    console.print("[green]All reference metrics reproduced within tolerance.[/green]")


@app.command()
def splits(
    cache: Annotated[Path, typer.Option(help="Cache directory")] = DEFAULT_CACHE,
    out: Annotated[Path, typer.Option(help="Split file to write")] = Path("data/splits/openkbp_opt_v1.json"),
    seed: int = 20260903,
    train: int = 60,
    validation: int = 10,
    test: int = 30,
) -> None:
    """Write the fixed, seeded 60/10/30 split stratified by number of targets."""
    from opengray.data.openkbp_opt import OpenKBPOptLoader

    manifest = OpenKBPOptLoader(cache).manifest()
    cases = manifest.get("cases", {})
    if len(cases) != train + validation + test:
        console.print(f"[red]Manifest has {len(cases)} cases; expected {train + validation + test}.[/red]")
        raise typer.Exit(code=1)
    strata: dict[int, list[str]] = {}
    for cid, entry in cases.items():
        strata.setdefault(int(entry["stats"]["n_targets"]), []).append(cid)
    rng = np.random.default_rng(seed)
    result: dict[str, list[str]] = {"train": [], "validation": [], "test": []}
    fractions = {"train": train, "validation": validation, "test": test}
    total = train + validation + test
    for k in sorted(strata):
        ids = sorted(strata[k], key=lambda s: int(s.split("_")[1]))
        rng.shuffle(ids)
        n_k = len(ids)
        counts = {name: int(round(n_k * f / total)) for name, f in fractions.items()}
        # Fix rounding so the stratum is fully assigned.
        diff = n_k - sum(counts.values())
        counts["train"] += diff
        i = 0
        for name in ("train", "validation", "test"):
            result[name].extend(ids[i : i + counts[name]])
            i += counts[name]
    # Rebalance across strata to hit the exact global sizes, moving from train.
    for name in ("validation", "test"):
        while len(result[name]) < fractions[name]:
            result[name].append(result["train"].pop())
        while len(result[name]) > fractions[name]:
            result["train"].append(result[name].pop())
    for name in result:
        result[name] = sorted(result[name], key=lambda s: int(s.split("_")[1]))
    payload = {
        "cohort": "openkbp_opt",
        "version": "v1",
        "seed": seed,
        "stratify_by": "n_targets",
        "strata_sizes": {str(k): len(v) for k, v in sorted(strata.items())},
        "sizes": {k: len(v) for k, v in result.items()},
        **result,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    console.print(f"Wrote {out}: " + ", ".join(f"{k} {len(v)}" for k, v in result.items()))


def _parse_patients(spec: str, available: list[str]) -> list[str]:
    out: list[str] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part and part.count("pt_") == 2:
            a, b = part.split("-")
            lo, hi = int(a.split("_")[1]), int(b.split("_")[1])
            out.extend(p for p in available if lo <= int(p.split("_")[1]) <= hi)
        elif part:
            out.append(part)
    missing = [p for p in out if p not in available]
    if missing:
        raise typer.BadParameter(f"not in archive: {missing[:5]}")
    return out


def _summarize(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    keys = ["voxels", "beamlets", "nnz", "D_mb", "load_s", "unit_w_max_gy", "ref_max_gy"]
    out: dict[str, dict[str, float]] = {}
    for k in keys:
        vals = np.array([r[k] for r in rows], dtype=np.float64)
        out[k] = {"min": float(np.nanmin(vals)), "median": float(np.nanmedian(vals)), "max": float(np.nanmax(vals))}
    return out


def summary_table(summary: dict[str, dict[str, float]]) -> Table:
    t = Table(title="Cohort summary")
    t.add_column("quantity")
    for c in ("min", "median", "max"):
        t.add_column(c, justify="right")
    for k, v in summary.items():
        t.add_row(k, *[f"{v[c]:,.2f}" if k in ("D_mb", "load_s") else f"{v[c]:,.0f}" for c in ("min", "median", "max")])
    return t


def _markdown_report(rows: list[dict[str, Any]], errors: list[tuple[str, str]]) -> str:
    summary = _summarize(rows)
    lines = [f"Cases loaded: {len(rows)}; load errors: {len(errors)}", "", "| quantity | min | median | max |", "|---|---:|---:|---:|"]
    for k, v in summary.items():
        fmt = (lambda x: f"{x:,.2f}") if k in ("D_mb", "load_s") else (lambda x: f"{x:,.0f}")
        lines.append(f"| {k} | {fmt(v['min'])} | {fmt(v['median'])} | {fmt(v['max'])} |")
    present: dict[str, int] = {}
    for r in rows:
        for m in (r["missing"].split(",") if r["missing"] != "-" else []):
            present[m] = present.get(m, 0) + 1
    if present:
        lines += ["", "Structures absent (count of cases):", ""]
        lines += [f"- {k}: {v}" for k, v in sorted(present.items())]
    lines += ["", "| case | voxels | beamlets | nnz | D MB | targets | structures | missing | ref max Gy | warnings |", "|---|---:|---:|---:|---:|---:|---:|---|---:|---:|"]
    for r in rows:
        lines.append(
            f"| {r['case']} | {r['voxels']:,} | {r['beamlets']:,} | {r['nnz']:,} | {r['D_mb']} | {r['targets']} | {r['structures']} | {r['missing']} | {r['ref_max_gy']} | {r['warnings']} |"
        )
    for cid, msg in errors:
        lines.append(f"- ERROR {cid}: {msg}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    app()
