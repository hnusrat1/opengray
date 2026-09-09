"Feasibility certificates from dose-influence constraints and dual bounds.\n\nA feasible witness or a low value in an optimization sweep does not prove infeasibility. Only a valid bound sufficient to exclude the presented limit supports a certified label; other cases remain uncertified."

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

from opengray.data.base import Case
from opengray.goals.schema import DOSE_TOL_GY, GoalList
from opengray.physics.dvh import merged_target_indices, parse_metric

TAIL_MULTIPLES = (2, 4, 10, 30, 100, 300)  # k' as multiples of the D<v>cc voxel count
TARGET_LEVELS: tuple[tuple[int, str], ...] = ((2, "near"), (10, "all"))  # (j as a multiple of the cold allowance, voxel scope)
TARGET_CAP = 2000  # merged-target voxels nearest the organ that carry the "near" levels
DEFAULT_METHOD = "highs-ipm"  # pt_339 (3.8M nonzeros): interior point 70 s where dual simplex passed 110 s
ROBUST_EXTRA_VOXELS = 1  # one extra voxel in each count, against percentile rounding
NUMERICAL_MARGIN_GY = 1e-3  # subtracted from the bound for the LP's own tolerance


def d_cc_voxels(case: Case, structure: str, v_cc: float) -> int:
    """The voxel count D<v>cc uses, computed as ``physics.dvh.evaluate`` does."""
    idx = case.structures[structure].mask_idx
    voxel_cc = float(np.mean(case.voxel_volume_cc[idx])) if idx.size else float(np.mean(case.voxel_volume_cc))
    return max(1, round(v_cc / voxel_cc))


def cold_voxel_allowance(n_total: int, p: float) -> int:
    """Voxels a merged target may hold below c while D_p >= c still holds (one extra for safety)."""
    q = (1.0 - p / 100.0) * (n_total - 1)
    return int(math.ceil(q - 1e-9)) + ROBUST_EXTRA_VOXELS


def _coords(case: Case, idx: np.ndarray) -> np.ndarray | None:
    if case.grid_shape is None:
        return None
    ijk = np.stack(np.unravel_index(case.feasible_idx[idx], case.grid_shape), axis=1).astype(np.float64)
    if case.voxel_size_mm is not None:
        ijk *= np.asarray(case.voxel_size_mm, dtype=np.float64)
    return ijk


def nearest_subset(case: Case, target_idx: np.ndarray, organ_idx: np.ndarray, cap: int, seed: int = 0) -> np.ndarray:
    """Up to ``cap`` target voxels nearest to the organ (by grid distance), or a seeded random
    subset when the case has no grid."""
    if target_idx.size <= cap:
        return target_idx
    tc, oc = _coords(case, target_idx), _coords(case, organ_idx)
    if tc is None or oc is None:
        rng = np.random.default_rng(seed)
        return np.sort(rng.choice(target_idx, size=cap, replace=False))
    from scipy.spatial import cKDTree

    dist, _ = cKDTree(oc).query(tc, k=1)
    order = np.argsort(dist, kind="stable")[:cap]
    return np.sort(target_idx[order])


def build_relaxation(case: Case, goals: GoalList, structure: str = "SpinalCord", metric: str = "D0.1cc", target_cap: int = TARGET_CAP, target_levels: tuple[tuple[int, str], ...] = TARGET_LEVELS, tol: float = DOSE_TOL_GY) -> dict[str, Any]:
    """The LP's constraint matrix and the numbers the bound needs. Returns a dict with ``A``
    (csr), ``b``, ``bounds``, the variable layout, and ``reason`` when no certificate is possible."""
    gl = goals.for_case(case).to_gy()
    organ = next((g for g in gl.goals if g.structure == structure and g.is_upper and g.metric == metric), None)
    if organ is None:
        return {"reason": f"no upper-limit {metric} goal on {structure}"}
    spec = parse_metric(metric)
    if spec.kind == "Dcc":
        n_top = d_cc_voxels(case, structure, float(spec.param)) + ROBUST_EXTRA_VOXELS
    elif spec.kind == "Dmax":
        n_top = 0
    else:
        return {"reason": f"certificates cover D<v>cc and Dmax, not {metric}"}
    ext = next((g for g in gl.hard if g.structure == "External" and g.metric == "Dmax" and g.is_upper), None)
    if ext is None:
        return {"reason": "no hard External Dmax goal: no per-voxel dose cap"}
    e_eff = float(ext.value) + tol
    ext_idx = case.structures["External"].mask_idx
    organ_idx = case.structures[structure].mask_idx
    if organ_idx.size == 0:
        return {"reason": f"{structure} has no feasible voxels"}
    if not np.all(np.isin(organ_idx, ext_idx, assume_unique=False)):
        return {"reason": f"{structure} has voxels outside External: no per-voxel cap on them"}
    targets = [g for g in gl.hard if g.structure in case.prescriptions and g.op == ">=" and g.spec.kind == "Dpercent"]
    if not targets:
        return {"reason": "no hard target D_p goal to relax"}

    # Columns: only beamlets that reach a cord voxel or a merged-target voxel. The others enter
    # no constraint and no objective term, so dropping them changes nothing in the LP and keeps
    # the returned fluence free of arbitrary weights on beamlets the LP never saw.
    target_union = np.unique(np.concatenate([merged_target_indices(case, g.structure)[0] for g in targets]))
    seen = np.unique(np.concatenate([organ_idx, target_union]))
    active = np.unique(sp.csr_matrix(case.D[seen]).indices)
    n_b = int(active.size)
    n_c = int(organ_idx.size)

    def rows_of(idx: np.ndarray) -> sp.csr_matrix:
        return sp.csr_matrix(case.D[idx][:, active])

    Dc = rows_of(organ_idx)
    # Variable layout: w (active beamlets) | t | s (n_c) | per target and level: tau, u, [u_out]
    n_var = n_b + 1 + n_c
    blocks: list[sp.spmatrix] = []
    rhs: list[np.ndarray] = []
    # Cord tail: D_c w - t - s <= 0
    blocks.append(sp.hstack([Dc, sp.csr_matrix(-np.ones((n_c, 1))), -sp.identity(n_c, format="csr")], format="csr"))
    rhs.append(np.zeros(n_c))
    target_info: list[dict[str, Any]] = []
    tail_blocks: list[tuple[sp.spmatrix, np.ndarray, int]] = []  # (block over its own aux columns, rhs, n_aux)
    for g in targets:
        idx, n_out = merged_target_indices(case, g.structure)
        n_total = int(idx.size) + int(n_out)
        m = cold_voxel_allowance(n_total, float(g.spec.param))
        c_eff = float(g.value) - tol
        if n_out > m:
            return {"reason": f"{g.structure}: {n_out} merged voxels outside the feasible mask exceed the {m} the goal allows below {c_eff:g} Gy; the goal is unmeetable as stated"}
        near = nearest_subset(case, idx, organ_idx, target_cap)
        info = {"structure": g.structure, "goal": g.label(), "n_merged": n_total, "n_outside": int(n_out), "cold_allowance": m, "c_eff_gy": c_eff, "n_near": int(near.size), "levels": []}
        target_info.append(info)
        # Whole-target mean: -(1^T D_t) w <= -(N - m) c'. The column sums over the merged
        # target's rows come from one transposed matvec with an indicator, without slicing D.
        ind = np.zeros(case.n_feasible)
        ind[idx] = 1.0
        mean_row = sp.csr_matrix(-np.asarray(case.D.T @ ind).ravel()[active][None, :])
        blocks.append(sp.hstack([mean_row, sp.csr_matrix((1, 1 + n_c))], format="csr"))
        rhs.append(np.array([-(n_total - m) * c_eff]))
        for mult, scope in target_levels:
            j = mult * m
            vox = idx if scope == "all" else near
            n_s = int(vox.size)
            if not (m < j <= n_s + n_out):
                continue
            info["levels"].append({"j": j, "scope": scope, "n_voxels": n_s})
            Ds = rows_of(vox)
            has_out = n_out > 0
            n_aux = 1 + n_s + (1 if has_out else 0)
            # rows: -D_i w + tau - u_i <= 0 (n_s); tau - u_out <= 0 (if out); -j tau + sum u + n_out u_out <= -(j - m) c'
            r1 = sp.hstack([-Ds, sp.csr_matrix((n_s, 1 + n_c)), sp.csr_matrix(np.ones((n_s, 1))), -sp.identity(n_s, format="csr")] + ([sp.csr_matrix((n_s, 1))] if has_out else []), format="csr")
            rows = [r1]
            b = [np.zeros(n_s)]
            if has_out:
                rows.append(sp.csr_matrix(np.concatenate([np.zeros(n_var), [1.0], np.zeros(n_s), [-1.0]])[None, :]))
                b.append(np.zeros(1))
            last = np.concatenate([np.zeros(n_var), [-float(j)], np.ones(n_s), [float(n_out)] if has_out else []])
            rows.append(sp.csr_matrix(last[None, :]))
            b.append(np.array([-(j - m) * c_eff]))
            tail_blocks.append((sp.vstack(rows, format="csr"), np.concatenate(b), n_aux))
    # Per-voxel cap from the External goal on the cord and on every merged-target voxel.
    cap_idx = np.unique(np.concatenate([organ_idx, target_union]))
    blocks.append(sp.hstack([rows_of(cap_idx), sp.csr_matrix((cap_idx.size, 1 + n_c))], format="csr"))
    rhs.append(np.full(cap_idx.size, e_eff))
    # Assemble: base blocks span the first n_var columns; each tail block adds its own columns.
    n_aux_total = sum(n for _b, _r, n in tail_blocks)
    A_parts = [sp.hstack([blk, sp.csr_matrix((blk.shape[0], n_aux_total))], format="csr") for blk in blocks]
    b_parts = list(rhs)
    offset = 0
    for blk, b, n_aux in tail_blocks:
        left = blk[:, :n_var]
        aux = blk[:, n_var:]
        A_parts.append(sp.hstack([left, sp.csr_matrix((blk.shape[0], offset)), aux, sp.csr_matrix((blk.shape[0], n_aux_total - offset - n_aux))], format="csr"))
        b_parts.append(b)
        offset += n_aux
    A = sp.vstack(A_parts, format="csr")
    b = np.concatenate(b_parts)
    lo = np.zeros(n_var + n_aux_total)
    hi = np.full(n_var + n_aux_total, np.inf)
    lo[n_b] = -np.inf  # t is free
    off = n_var
    for _blk, _r, n_aux in tail_blocks:
        lo[off] = -np.inf  # tau is free
        off += n_aux
    return {
        "A": A, "b": b, "bounds": np.stack([lo, hi], axis=1), "n_b": n_b, "n_c": n_c, "n_var": n_var, "n_aux": n_aux_total, "active": active, "n_beamlets": case.n_beamlets,
        "n_top": n_top, "e_eff_gy": e_eff, "organ_goal": organ.label(), "external_goal": ext.label(), "targets": target_info,
        "nnz": int(A.nnz), "n_rows": int(A.shape[0]), "n_cols": int(A.shape[1]),
    }


def tail_bound(relax: dict[str, Any], k: int, method: str = DEFAULT_METHOD, time_limit: float | None = None) -> dict[str, Any]:
    """Minimize the mean of the k highest organ voxel doses over the relaxation."""
    n_b, n_c, n_cols = relax["n_b"], relax["n_c"], relax["n_cols"]
    c = np.zeros(n_cols)
    c[n_b] = 1.0
    c[n_b + 1 : n_b + 1 + n_c] = 1.0 / k
    opts: dict[str, Any] = {"presolve": True}
    if time_limit is not None:
        opts["time_limit"] = float(time_limit)
    t0 = time.perf_counter()
    res = linprog(c, A_ub=relax["A"], b_ub=relax["b"], bounds=relax["bounds"], method=method, options=opts)
    out = {"k": int(k), "status": int(res.status), "message": str(res.message), "seconds": round(time.perf_counter() - t0, 2)}
    if res.status == 0:
        out["tail_mean_gy"] = float(res.fun)
        w = np.zeros(relax["n_beamlets"])
        w[relax["active"]] = np.asarray(res.x[:n_b])
        out["w"] = w
    return out


def certificate_from_tails(tails: list[dict[str, Any]], n_top: int, e_eff: float, tol: float = DOSE_TOL_GY) -> dict[str, Any]:
    """Turn per-k tail-mean minima into the D<v>cc lower bound and the certified-below limit."""
    per_k = []
    best = None
    for t in tails:
        if t.get("status") != 0 or t["k"] <= n_top:
            continue
        x = (t["k"] * t["tail_mean_gy"] - n_top * e_eff) / (t["k"] - n_top)
        per_k.append({"k": t["k"], "tail_mean_gy": round(t["tail_mean_gy"], 4), "d_bound_gy": round(x, 4)})
        if best is None or x > best:
            best = x
    if best is None:
        return {"per_k": per_k, "lower_bound_gy": None, "certified_below_gy": None}
    lb = best - NUMERICAL_MARGIN_GY
    cert = lb - tol
    return {"per_k": per_k, "lower_bound_gy": round(lb, 4), "certified_below_gy": round(cert, 4) if cert > 0 else None}


def certify(case: Case, goals: GoalList, structure: str = "SpinalCord", metric: str = "D0.1cc", tail_multiples: tuple[int, ...] = TAIL_MULTIPLES, target_cap: int = TARGET_CAP, target_levels: tuple[tuple[int, str], ...] = TARGET_LEVELS, method: str = DEFAULT_METHOD, time_limit: float | None = None, tol: float = DOSE_TOL_GY, keep_w: bool = False) -> dict[str, Any]:
    """The certificate record for one case: the relaxation's size, each LP's optimum, the
    D<v>cc lower bound, and ``certified_below_gy`` (None when the bound does not reach above zero
    or the relaxation could not be built)."""
    t0 = time.perf_counter()
    relax = build_relaxation(case, goals, structure, metric, target_cap=target_cap, target_levels=target_levels, tol=tol)
    rec: dict[str, Any] = {"structure": structure, "metric": metric, "method": method, "target_cap": target_cap, "target_levels": [list(x) for x in target_levels], "tail_multiples": list(tail_multiples), "tol_gy": tol, "robust_extra_voxels": ROBUST_EXTRA_VOXELS}
    if "reason" in relax:
        rec.update({"reason": relax["reason"], "lower_bound_gy": None, "certified_below_gy": None, "seconds": round(time.perf_counter() - t0, 2)})
        return rec
    n_top = relax["n_top"]
    ks = sorted({max(n_top + 1, mult * max(n_top, 1)) for mult in tail_multiples if mult * max(n_top, 1) <= relax["n_c"]})
    if not ks and relax["n_c"] > n_top:
        ks = [relax["n_c"]]
    tails = []
    lp_plans = []
    best_lp: tuple[float, np.ndarray] | None = None
    for k in ks:
        t = tail_bound(relax, k, method=method, time_limit=time_limit)
        w = t.pop("w", None)
        if w is not None:
            # The LP's own fluence, evaluated on the whole case against every other hard goal: a
            # relaxation's optimum is not a plan, but when it happens to meet the goals it is a
            # witness, and otherwise it is the warm start for ``repair_witness``.
            chk = witness_check(case, goals, w, structure, metric)
            lp_plans.append({"k": t["k"], **{k2: v for k2, v in chk.items() if k2 != "goals"}, "n_unmet": sum(1 for g in chk["goals"].values() if not g["met"])})
            if best_lp is None or chk["organ_gy"] < best_lp[0]:
                best_lp = (chk["organ_gy"], w)
        if keep_w and w is not None:
            t["w"] = w
        tails.append(t)
        if t["status"] == 2:  # infeasible relaxation: the other hard goals cannot all be met
            break
    cert = certificate_from_tails(tails, n_top, relax["e_eff_gy"], tol=tol)
    rec.update({
        "n_top": n_top, "e_eff_gy": relax["e_eff_gy"], "organ_goal": relax["organ_goal"], "external_goal": relax["external_goal"], "targets": relax["targets"],
        "lp_rows": relax["n_rows"], "lp_cols": relax["n_cols"], "lp_nnz": relax["nnz"], "tails": [{k2: v for k2, v in t.items() if k2 != "w"} for t in tails],
        "relaxation_infeasible": any(t["status"] == 2 for t in tails), "lp_plans": lp_plans, **cert, "seconds": round(time.perf_counter() - t0, 2),
    })
    if keep_w:
        rec["tail_w"] = {t["k"]: t["w"] for t in tails if "w" in t}
        rec["best_lp_w"] = best_lp[1] if best_lp else None
    return rec


def other_hard_goals(case: Case, goals: GoalList, structure: str) -> list:
    gl = goals.for_case(case).to_gy()
    return [g for g in gl.hard if g.structure != structure]


def witness_check(case: Case, goals: GoalList, w: np.ndarray, structure: str = "SpinalCord", metric: str = "D0.1cc", merge_targets: bool = True) -> dict[str, Any]:
    """Evaluate a fluence against every hard goal except the organ's: whether it is a witness
    (all met at the reporting tolerance), the organ metric it reaches, and each goal's value."""
    from opengray.physics.dvh import evaluate

    dose = case.dose(w)
    per = {}
    ok = True
    for g in other_hard_goals(case, goals, structure):
        v = float(evaluate(case, dose, g.structure, g.spec, merge_targets=merge_targets))
        met = g.is_met(v)
        ok = ok and met
        per[g.label()] = {"achieved": round(v, 4), "met": met}
    organ = float(evaluate(case, dose, structure, metric, merge_targets=merge_targets))
    return {"others_met": ok, "organ_gy": round(organ, 4), "global_max_gy": round(float(dose.max()), 3), "goals": per}


def repair_witness(case: Case, goals: GoalList, w0: np.ndarray, target_gy: float, structure: str = "SpinalCord", metric: str = "D0.1cc", priorities: tuple[float, ...] = (1000.0, 100.0, 10.0), levels: tuple[float, ...] | None = None, config: Any = None, context: Any = None) -> dict[str, Any]:
    """Try to turn the LP's fluence into a real witness below ``target_gy``: warm-start the
    environment's solver from it with the organ capped at each level and the targets' coverage
    terms at each priority (the caps stay at the maximum priority), and keep the plan with the
    lowest organ metric that meets every other hard goal. Returns that plan's record (or None)
    and the sweep."""
    from opengray.physics.objectives import PRIORITY_MAX, Objective, ObjectiveTerm
    from opengray.physics.solver import CaseSolverContext, SolverConfig, solve

    gl = goals.for_case(case).to_gy()
    ctx = context or CaseSolverContext(case)
    cfg = config or SolverConfig()
    top = max(case.prescriptions.values()) if case.prescriptions else 0.0
    best: dict[str, Any] | None = None
    sweep: list[dict[str, Any]] = []
    lv = levels or (max(0.5, 0.9 * target_gy), max(0.5, 0.7 * target_gy))
    for level in lv:
        for pri in priorities:
            terms: list[ObjectiveTerm] = []
            for g in gl.goals:
                if g.structure == structure:
                    terms.append(ObjectiveTerm(structure=structure, term="max_dose", level=level, priority=PRIORITY_MAX))
                elif g.structure in case.prescriptions and g.op == ">=" and g.metric.startswith("D"):
                    rx = case.prescriptions[g.structure]
                    terms.append(ObjectiveTerm(structure=g.structure, term="min_dose", level=rx, priority=pri))
                    if rx >= top:
                        terms.append(ObjectiveTerm(structure=g.structure, term="max_dose", level=rx * 1.07 * 0.95, priority=pri))
                elif g.metric == "Dmean":
                    terms.append(ObjectiveTerm(structure=g.structure, term="mean_dose", level=g.value * 0.95, priority=1.0))
                elif g.op == "<=":
                    lvl = max(top, g.value * 0.95) if g.structure == "External" else g.value * 0.95
                    terms.append(ObjectiveTerm(structure=g.structure, term="max_dose", level=lvl, priority=PRIORITY_MAX if g.kind == "hard" else 1.0))
            res = solve(case, Objective(terms=terms), w0=w0, config=cfg, context=ctx)
            chk = witness_check(case, goals, res.w, structure, metric)
            sweep.append({"level": round(level, 3), "target_priority": pri, "organ_gy": chk["organ_gy"], "others_met": chk["others_met"], "iterations": res.iterations, "global_max_gy": chk["global_max_gy"]})
            if chk["others_met"] and (best is None or chk["organ_gy"] < best["organ_gy"]):
                best = {"organ_gy": chk["organ_gy"], "level": level, "target_priority": pri, "w": res.w.copy(), "global_max_gy": chk["global_max_gy"]}
    return {"best": best, "sweep": sweep}
