"Counting-based certificates for coverage-cap and target-organ-overlap contradictions."

from __future__ import annotations

from typing import Any

import numpy as np

from opengray.data.base import Case
from opengray.goals.schema import DOSE_TOL_GY, GoalList
from opengray.physics.certificate import ROBUST_EXTRA_VOXELS, cold_voxel_allowance, d_cc_voxels
from opengray.physics.dvh import merged_target_indices

SERIAL_ORGANS = ("SpinalCord", "Brainstem", "Bone_Mandible")


def overlap_contradictions(case: Case, goals: GoalList, tol: float = DOSE_TOL_GY) -> list[dict[str, Any]]:
    """Every (organ goal, target goal) pair whose overlap exceeds the allowance, with the limit
    below which the organ goal is certified infeasible. Sorted by certified limit, highest
    first, then by overlap margin."""
    gl = goals.for_case(case).to_gy()
    targets = [g for g in gl.hard if g.structure in case.prescriptions and g.op == ">=" and g.spec.kind == "Dpercent"]
    organs = [g for g in gl.hard if g.structure not in case.prescriptions and g.structure != "External" and g.is_upper and g.spec.kind in ("Dcc", "Dmax")]
    out: list[dict[str, Any]] = []
    for og in organs:
        o_idx = case.structures[og.structure].mask_idx
        n = (d_cc_voxels(case, og.structure, float(og.spec.param)) if og.spec.kind == "Dcc" else 0) + ROBUST_EXTRA_VOXELS
        for tg in targets:
            t_idx, n_out = merged_target_indices(case, tg.structure)
            n_total = int(t_idx.size) + int(n_out)
            m = cold_voxel_allowance(n_total, float(tg.spec.param))
            overlap = int(np.intersect1d(o_idx, t_idx, assume_unique=False).size)
            if overlap > m + n:
                c = float(tg.value)
                out.append({"organ": og.structure, "organ_goal": og.label(), "organ_limit_gy": float(og.value), "target": tg.structure, "target_goal": tg.label(), "c_gy": c, "overlap_voxels": overlap, "cold_allowance": m, "organ_allowance": n, "margin_voxels": overlap - m - n, "certified_below_gy": round(c - 2.0 * tol - 1e-6, 4)})
    out.sort(key=lambda r: (-r["certified_below_gy"], -r["margin_voxels"]))
    return out


def coverage_cap_contradiction(case: Case, goals: GoalList, target: str | None = None, tol: float = DOSE_TOL_GY) -> dict[str, Any] | None:
    """The coverage value above which a target's D_p goal contradicts the External cap."""
    gl = goals.for_case(case).to_gy()
    ext = next((g for g in gl.hard if g.structure == "External" and g.metric == "Dmax" and g.is_upper), None)
    if ext is None or not case.prescriptions:
        return None
    tname = target or max(case.prescriptions, key=case.prescriptions.get)
    tg = next((g for g in gl.hard if g.structure == tname and g.op == ">=" and g.spec.kind == "Dpercent"), None)
    if tg is None:
        return None
    t_idx, n_out = merged_target_indices(case, tname)
    n_total = int(t_idx.size) + int(n_out)
    m = cold_voxel_allowance(n_total, float(tg.spec.param))
    e_eff = float(ext.value) + tol
    certified_above = n_total * e_eff / (n_total - m) + tol
    return {"target": tname, "target_goal": tg.label(), "metric": tg.metric, "published_gy": float(tg.value), "external_goal": ext.label(), "external_gy": float(ext.value), "n_total": n_total, "cold_allowance": m, "certified_above_gy": round(certified_above + 1e-6, 4)}


def presented_overlap_limit(certified_below_gy: float, published_gy: float, fraction: float = 0.9) -> float | None:
    """A limit that is certified infeasible and reads as a tightened clinical limit: ``fraction``
    of the certified threshold, rounded to a tenth, and below the published limit. None when the
    published limit is itself certified infeasible (the case's own goal list is unmeetable; the
    arm cannot be built on it)."""
    if published_gy < certified_below_gy:
        return None
    limit = float(np.floor(fraction * certified_below_gy * 10.0) / 10.0)
    return min(limit, round(published_gy - 0.1, 1))


def presented_coverage(certified_above_gy: float, rx: float, gate_factor: float = 1.15) -> float | None:
    """A coverage goal that is certified infeasible against the External cap and stays under the
    task's own rejection line (so the contradiction is with the goal list, not the rules): the
    certified value rounded up to the next half gray. None if that lands at or above the line."""
    v = float(np.ceil((certified_above_gy + 0.05) * 2.0) / 2.0)
    return v if v < gate_factor * rx else None
