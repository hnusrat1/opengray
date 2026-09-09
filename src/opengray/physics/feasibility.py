"Feasibility sweeps and explicitly qualified task labels.\n\nA sweep records achieved values under finite optimization; it does not prove that a lower value is impossible. Certified infeasibility requires an independent valid bound. Feasible witness values and uncertified estimates are kept distinct."

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from opengray.data.base import Case
from opengray.goals.schema import GoalList
from opengray.physics.dvh import evaluate
from opengray.physics.objectives import PRIORITY_MAX, Objective, ObjectiveTerm
from opengray.physics.solver import CaseSolverContext, SolverConfig, solve

DEFAULT_STRUCTURE = "SpinalCord"
DEFAULT_METRIC = "D0.1cc"
COMPANION_MAX_FACTOR = 1.07
SHRINK = 0.75
UNREACHED = 1.05  # the organ level was not reached: the solver is at its floor
MARGINS = (0.95, 0.90)  # upper-limit terms sit below their goals, as the agents that meet them do


def _template(case: Case, goals: GoalList, organ: str, organ_level: float, margin: float) -> Objective:
    terms: list[ObjectiveTerm] = []
    for g in goals.goals:
        pri = PRIORITY_MAX if g.kind == "hard" else 1.0
        if g.structure == organ:
            terms.append(ObjectiveTerm(structure=organ, term="max_dose", level=organ_level, priority=PRIORITY_MAX))
            continue
        if g.structure in case.prescriptions and g.op == ">=" and g.metric.startswith("D"):
            rx = case.prescriptions[g.structure]
            terms.append(ObjectiveTerm(structure=g.structure, term="min_dose", level=rx, priority=pri))
            # A hot-spot ceiling only on the top target: a lower-dose PTV overlaps the higher one,
            # and a ceiling there at 1.07 x its own prescription fights the higher target's coverage.
            if rx >= max(case.prescriptions.values()):
                terms.append(ObjectiveTerm(structure=g.structure, term="max_dose", level=max(rx, rx * COMPANION_MAX_FACTOR * margin), priority=pri))
        elif g.metric == "Dmean":
            terms.append(ObjectiveTerm(structure=g.structure, term="mean_dose", level=g.value * margin, priority=pri))
        elif g.op == "<=" and (g.metric.startswith("D") or g.metric == "Dmax"):
            top = max(case.prescriptions.values()) if case.prescriptions else 0.0
            terms.append(ObjectiveTerm(structure=g.structure, term="max_dose", level=max(top, g.value * margin) if g.structure == "External" else g.value * margin, priority=pri))
    return Objective(terms=terms)


def coverage_floor(case: Case, goals: GoalList, structure: str = DEFAULT_STRUCTURE, metric: str = DEFAULT_METRIC, context: CaseSolverContext | None = None, config: SolverConfig | None = None, max_solves: int = 12, bisect: int = 3) -> dict[str, Any]:
    """Lowest ``structure`` ``metric`` (Gy) this solver reaches while every other hard goal is met."""
    if structure not in case.structures:
        raise KeyError(f"{structure} not in {case.case_id}")
    gl = goals.for_case(case).to_gy()
    organ_goal = next((g for g in gl.goals if g.structure == structure and g.is_upper), None)
    others = [g for g in gl.goals if g.kind == "hard" and g.structure != structure]
    if organ_goal is None or not others:
        raise ValueError(f"need an upper-limit goal on {structure} and at least one other hard goal")
    ctx = context or CaseSolverContext(case)
    cfg = config or SolverConfig()
    level = float(organ_goal.value)
    w0 = None
    best: dict[str, Any] | None = None
    sweep: list[dict[str, Any]] = []
    margins = list(MARGINS)
    margin = margins.pop(0)
    failed_level: float | None = None  # highest level shown not to meet the other hard goals

    def run(level: float) -> tuple[float, bool]:
        nonlocal w0, best
        res = solve(case, _template(case, gl, structure, level, margin), w0=w0, config=cfg, context=ctx)
        w0 = res.w
        dose = res.dose
        achieved = float(evaluate(case, dose, structure, metric, merge_targets=True))
        others_met = all(g.is_met(float(evaluate(case, dose, g.structure, g.spec, merge_targets=True))) for g in others)
        sweep.append({"level": round(level, 3), "margin": margin, "achieved": round(achieved, 3), "others_met": others_met, "iterations": res.iterations, "global_max_gy": round(float(dose.max()), 3)})
        if others_met and (best is None or achieved < best["floor_gy"]):
            best = {"floor_gy": round(achieved, 3), "global_max_gy": round(float(dose.max()), 3), "iterations": res.iterations, "witness_w": res.w.copy()}
        return achieved, others_met

    for _ in range(max_solves):
        achieved, others_met = run(level)
        if not others_met:
            if best is None and margins:
                margin = margins.pop(0)  # the first template missed a hard goal: retry with more margin
                continue
            failed_level = level
            break
        if achieved > UNREACHED * level:
            break  # the solver could not bring the organ down to the level: this is the floor
        level = SHRINK * achieved
        if level < 0.5:
            break
    if best is not None and failed_level is not None:
        # Bisect between the floor found and the level that failed: each solve halves the bracket.
        lo, hi = failed_level, best["floor_gy"]
        for _ in range(bisect):
            if hi - lo < 0.5:
                break
            mid = 0.5 * (lo + hi)
            achieved, others_met = run(mid)
            if others_met:
                hi = min(hi, achieved)
            else:
                lo = mid
        failed_level = lo
    if best is None:
        # The published objective itself cannot meet the other hard goals here: report the first
        # solve's organ metric with a flag; there is no witness, so no limit is feasible_witness.
        best = {"floor_gy": sweep[0]["achieved"], "global_max_gy": sweep[0]["global_max_gy"], "iterations": sweep[0]["iterations"], "witness_w": None}
    return {
        "structure": structure,
        "metric": metric,
        "best_achieved_gy": best["floor_gy"],
        "floor_gy": best["floor_gy"],  # deprecated alias of best_achieved_gy
        "certified_below_gy": None,
        "bracket_low_gy": round(failed_level, 3) if failed_level is not None else None,
        "original_gy": float(organ_goal.value),
        "others_met": any(s["others_met"] for s in sweep),
        "global_max_gy": best["global_max_gy"],
        "iterations": best["iterations"],
        "solves": len(sweep),
        "sweep": sweep,
        "witness_w": best.get("witness_w"),  # ndarray or None; the caller stores it as .npy
    }


def load_floors(path: Path) -> dict[str, dict[str, Any]]:
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else {}


def witness_dir(path: Path) -> Path:
    return Path(path).with_suffix(".witness")


def save_floors(path: Path, floors: dict[str, dict[str, Any]]) -> None:
    """Write the floors JSON; any ``witness_w`` array in a record goes to ``<path>.witness/`` as
    ``<case>.<structure>.npy`` and the record keeps the file name."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    out: dict[str, dict[str, Any]] = {}
    for cid, recs in floors.items():
        out[cid] = {}
        for structure, rec in recs.items():
            rec = dict(rec)
            w = rec.pop("witness_w", None)
            if w is not None:
                wd = witness_dir(p)
                wd.mkdir(parents=True, exist_ok=True)
                fname = f"{cid}.{structure}.npy"
                np.save(wd / fname, np.asarray(w, dtype=np.float64))
                rec["witness_file"] = fname
            out[cid][structure] = rec
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(out, indent=1, sort_keys=True))
    tmp.replace(p)


def floor_for(case: Case, goals: GoalList, path: Path | None, structure: str = DEFAULT_STRUCTURE, verify_witness: bool = False) -> dict[str, dict[str, Any]]:
    """The {structure: record} map a presentation needs (best achieved dose, certificate if any),
    computed and cached on first use."""
    floors = load_floors(path) if path else {}
    rec = floors.get(case.case_id, {}).get(structure)
    if rec is None:
        rec = coverage_floor(case, goals, structure=structure)
        if path:
            floors.setdefault(case.case_id, {})[structure] = rec
            save_floors(path, floors)
    result = {"best_achieved_gy": float(rec.get("best_achieved_gy", rec.get("floor_gy"))), "certified_below_gy": rec.get("certified_below_gy"), "original_gy": rec.get("original_gy")}
    if verify_witness:
        audit = check_tight_witness(case, goals, rec, path, structure)
        result["witness_verified"] = audit["valid"]
        result["witness_audit"] = audit
    return {structure: result}


def check_tight_witness(case: Case, goals: GoalList, record: dict[str, Any], path: Path | None, structure: str = DEFAULT_STRUCTURE) -> dict[str, Any]:
    """Recalculate every final hard goal from saved fluence before labelling a control feasible."""
    w = record.get("witness_w")
    if w is None and path and record.get("witness_file"):
        fname = record["witness_file"]
        if Path(fname).name != fname:
            return {"valid": False, "detail": "invalid witness filename"}
        file = witness_dir(path) / fname
        if file.is_file():
            w = np.load(file, allow_pickle=False)
    if w is None:
        return {"valid": False, "detail": "no saved witness fluence"}
    w = np.asarray(w)
    if w.shape != (case.n_beamlets,) or not np.all(np.isfinite(w)) or np.any(w < 0):
        return {"valid": False, "detail": "invalid witness fluence"}
    gl = goals.for_case(case).to_gy()
    organ = next((g for g in gl.goals if g.structure == structure and g.is_upper), None)
    if organ is None:
        return {"valid": False, "detail": "no organ goal"}
    best = float(record.get("best_achieved_gy", record.get("floor_gy")))
    tightened = tightened_limit(best, organ.value)
    limit = organ.value if tightened is None else tightened
    dose = case.dose(w)
    checks = []
    for original in gl.goals:
        if original.kind != "hard":
            continue
        g = original.with_value(limit) if original is organ else original
        value = float(evaluate(case, dose, g.structure, g.spec, merge_targets=True))
        checks.append({"goal": g.label(), "achieved_gy": value, "met": bool(g.is_met(value))})
    return {"valid": bool(checks) and all(g["met"] for g in checks), "presented_limit_gy": limit, "goals": checks}


def feasibility_record(feasibility: dict[str, Any], structure: str) -> tuple[float, float | None]:
    """(best achieved dose, certified-infeasible-below limit or None) from a map that may hold a
    plain float (older callers) or a record."""
    v = feasibility[structure]
    if isinstance(v, dict):
        return float(v.get("best_achieved_gy", v.get("floor_gy"))), (float(v["certified_below_gy"]) if v.get("certified_below_gy") is not None else None)
    return float(v), None


def feasibility_status(limit_gy: float, best_achieved_gy: float, certified_below_gy: float | None, tol: float = 0.05) -> str:
    """``feasible_witness`` when a stored plan meets the limit, ``certified_infeasible`` when a
    valid lower bound puts the limit out of reach, else ``uncertified``."""
    if limit_gy >= best_achieved_gy - tol:
        return "feasible_witness"
    if certified_below_gy is not None and limit_gy < certified_below_gy:
        return "certified_infeasible"
    return "uncertified"


def tightened_limit(floor_gy: float, original_gy: float) -> float | None:
    """Track 5's plannable-but-tight limit: the midpoint between the floor and the published limit,
    or None when the floor already exceeds the published limit (the case is unplannable as given)."""
    if floor_gy >= original_gy:
        return None
    return round(floor_gy + 0.5 * (original_gy - floor_gy), 1)


def infeasible_limit(floor_gy: float, fraction: float = 0.7, certified_below_gy: float | None = None) -> float:
    """The unplannable arm's presented limit: ``fraction`` x the best achieved dose, lowered to a
    tenth of a gray below ``certified_below_gy`` when a certificate exists and the fraction is
    not itself certified, so the label carries the certificate wherever one can be had."""
    limit = round(fraction * floor_gy, 1)
    if certified_below_gy is not None and limit >= certified_below_gy:
        limit = math.floor((certified_below_gy - 0.05) * 10.0 + 1e-9) / 10.0
    return limit


def summarize(floors: dict[str, dict[str, Any]]) -> dict[str, Any]:
    vals = [rec["floor_gy"] for recs in floors.values() for rec in recs.values()]
    return {"n_cases": len(floors), "floor_gy_min": float(np.min(vals)) if vals else None, "floor_gy_median": float(np.median(vals)) if vals else None, "floor_gy_max": float(np.max(vals)) if vals else None}
