"Matrix-read consistency, independent-solver comparisons, and timing diagnostics."

from __future__ import annotations

import re
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import scipy.optimize as sopt
import scipy.sparse as sp

from opengray.data.base import Case, Structure
from opengray.physics.objectives import AssembledObjective, Objective, ObjectiveTerm
from opengray.physics.solver import CaseSolverContext, SolverConfig, solve

PAPER_PLANS_PREFIX = "open-kbp-opt-data/paper-plans/"


def default_objective(case: Case) -> Objective:
    """A plausible starting objective from the case's structures (used for timing only)."""
    terms: list[ObjectiveTerm] = []
    for name, rx in case.prescriptions.items():
        terms.append(ObjectiveTerm(structure=name, term="min_dose", level=rx * 0.95, priority=10))
        terms.append(ObjectiveTerm(structure=name, term="max_dose", level=rx * 1.07, priority=5))
    serial = {"Brainstem": 50.0, "SpinalCord": 45.0, "Bone_Mandible": 73.5}
    parallel = {"Parotid_L": 26.0, "Parotid_R": 26.0, "Esophagus": 45.0, "Larynx": 45.0}
    for name, lim in serial.items():
        if name in case.structures:
            terms.append(ObjectiveTerm(structure=name, term="max_dose", level=lim, priority=20))
    for name, lim in parallel.items():
        if name in case.structures:
            terms.append(ObjectiveTerm(structure=name, term="mean_dose", level=lim, priority=5))
    return Objective(terms=terms)


def list_paper_plan_models(zip_path: Path) -> list[str]:
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
    models = set()
    for n in names:
        m = re.match(re.escape(PAPER_PLANS_PREFIX) + r"([^/]+)/plan-fluence/(set_\d+)/pt_\d+\.csv$", n)
        if m:
            models.add(f"{m.group(1)}/{m.group(2)}")
    return sorted(models, key=lambda s: (s.split("/")[0], int(s.split("_")[-1])))


def read_paper_plan(zip_path: Path, model: str, case_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    "Read a published fluence and dose pair from paper-plans/<optimization model>/plan-fluence/<prediction set>/pt_<n>.csv and its corresponding dose file."
    from opengray.data.openkbp_opt import read_sparse_csv

    opt_model, pred_set = model.split("/", 1)
    with zipfile.ZipFile(zip_path) as z, tempfile.TemporaryDirectory(prefix="opengray_pp_") as tmp:
        f_name = f"{PAPER_PLANS_PREFIX}{opt_model}/plan-fluence/{pred_set}/{case_id}.csv"
        d_name = f"{PAPER_PLANS_PREFIX}{opt_model}/plan-dose/{pred_set}/{case_id}.csv"
        for n in (f_name, d_name):
            if n not in z.namelist():
                raise FileNotFoundError(n)
        z.extract(f_name, tmp)
        z.extract(d_name, tmp)
        f_idx, f_val = read_sparse_csv(Path(tmp) / f_name)
        d_idx, d_val = read_sparse_csv(Path(tmp) / d_name)
    if f_val is None or d_val is None:
        raise ValueError("paper plan files must carry a data column")
    return np.stack([f_idx, f_val], axis=1), d_idx, d_val


def matrix_read_check(case: Case, fluence: np.ndarray, dose_idx: np.ndarray, dose_val: np.ndarray) -> dict[str, Any]:
    """Compare ``D @ w`` with a published dose. ``fluence`` is (index, value) pairs."""
    w = np.zeros(case.n_beamlets)
    idx = fluence[:, 0].astype(np.int64)
    if idx.min() < 0 or idx.max() >= case.n_beamlets:
        raise ValueError(f"fluence indices {idx.min()}..{idx.max()} outside 0..{case.n_beamlets - 1}")
    w[idx] = fluence[:, 1]
    d_ours = case.dose(w)
    # Published dose in feasible space; anything outside the mask is reported, not ignored.
    pos = np.searchsorted(case.feasible_idx, dose_idx)
    inside = (pos < case.feasible_idx.size) & (case.feasible_idx[np.minimum(pos, case.feasible_idx.size - 1)] == dose_idx)
    d_pub = np.zeros(case.n_feasible)
    d_pub[pos[inside]] = dose_val[inside]
    diff = d_ours - d_pub
    scale = max(float(d_pub.max()), 1e-9)
    return {
        "n_pub_outside_mask": int((~inside).sum()),
        "pub_max_gy": float(d_pub.max()),
        "ours_max_gy": float(d_ours.max()),
        "max_abs_diff_gy": float(np.abs(diff).max()),
        "mean_abs_diff_gy": float(np.abs(diff).mean()),
        "rel_max_diff": float(np.abs(diff).max() / scale),
        "w_max": float(w.max()),
        "w_nonzero": int(np.count_nonzero(w)),
    }


def subsample_case(case: Case, n_per_structure: int, seed: int = 0) -> Case:
    """Keep up to ``n_per_structure`` random voxels of each structure (a smaller, equivalent problem)."""
    rng = np.random.default_rng(seed)
    keep: list[np.ndarray] = []
    for s in case.structures.values():
        if s.mask_idx.size > n_per_structure:
            keep.append(rng.choice(s.mask_idx, n_per_structure, replace=False))
        else:
            keep.append(s.mask_idx)
    rows = np.unique(np.concatenate(keep)) if keep else np.array([], dtype=np.int64)
    remap = {int(r): i for i, r in enumerate(rows)}
    structures = {}
    for name, s in case.structures.items():
        sel = np.array([remap[int(v)] for v in s.mask_idx if int(v) in remap], dtype=np.int32)
        # Dropped voxels leave the problem entirely (they are not zero-dose voxels), so the
        # subsampled objective is a genuine smaller instance rather than one with a large constant.
        structures[name] = Structure(name=name, raw_name=s.raw_name, mask_idx=sel, n_outside=s.n_outside, volume_cc=s.volume_cc)
    return Case(
        case_id=case.case_id + "_sub",
        cohort=case.cohort,
        feasible_idx=case.feasible_idx[rows],
        voxel_volume_cc=case.voxel_volume_cc[rows],
        D=sp.csr_matrix(case.D[rows, :]),
        structures=structures,
        prescriptions=dict(case.prescriptions),
        beamlets=case.beamlets,
        beam_geometry=case.beam_geometry,
    )


def independent_solver_check(case: Case, objective: Objective, n_per_structure: int = 200, seed: int = 0, w_max: float | None = None, complexity_lambda: float = 0.0, complexity_limit: float = 65.0) -> dict[str, Any]:
    """FISTA against L-BFGS-B on the same subsampled objective. By default the unconstrained
    problem (the optimizer check); with ``w_max`` and ``complexity_lambda`` both solvers see the
    box and the same fixed-weight SPG penalty, which checks the penalized solve too (the
    continuation itself is a schedule over such solves)."""
    sub = subsample_case(case, n_per_structure, seed)
    # Subsampled voxels count as "outside" so the objective keeps its scale; for the comparison
    # both solvers see exactly the same function, which is all that matters here.
    obj = AssembledObjective(sub, objective)
    term = None
    if complexity_lambda > 0 and sub.beamlets is not None:
        from opengray.physics.complexity import ComplexityTerm

        term = ComplexityTerm(sub.beamlets, limit=complexity_limit)

    def fg(w: np.ndarray) -> tuple[float, np.ndarray]:
        f, g = obj.value_and_grad(w)
        if term is not None:
            fp, gp = term.value_and_grad(w, complexity_lambda)
            return f + fp, g + gp
        return f, g

    t = time.perf_counter()
    ref = sopt.minimize(
        fg,
        np.zeros(sub.n_beamlets),
        jac=True,
        method="L-BFGS-B",
        bounds=[(0, w_max)] * sub.n_beamlets,
        options={"maxiter": 5000, "maxfun": 10000, "ftol": 1e-14, "gtol": 1e-8},
    )
    t_ref = time.perf_counter() - t
    if term is None:
        res = solve(sub, objective, config=SolverConfig(rel_tol=1e-8, max_iter=5000, patience=10, w_max=w_max, complexity_limit=None))
        f_ours = float(res.objective_value)
    else:
        from opengray.physics.solver import CaseSolverContext, _fista

        res = _fista(sub, obj, term, complexity_lambda, None, SolverConfig(rel_tol=1e-8, max_iter=5000, patience=10, w_max=w_max, complexity_limit=None), CaseSolverContext(sub))
        f_ours = float(res.objective_value + res.penalty_value)
    f_ref = float(ref.fun)
    denom = max(abs(f_ref), 1e-9)
    d_ref = sub.dose(np.asarray(ref.x))
    d_diff = np.abs(res.dose - d_ref)
    f0 = obj.value(np.zeros(sub.n_beamlets))
    return {
        "n_sub_voxels": sub.n_feasible,
        "f0": f0,
        "f_lbfgsb": f_ref,
        "f_fista": f_ours,
        "rel_diff": (f_ours - f_ref) / denom,
        "dose_max_abs_diff_gy": float(d_diff.max()),
        "dose_mean_abs_diff_gy": float(d_diff.mean()),
        "lbfgsb_s": t_ref,
        "fista_s": res.wall_s,
        "fista_iters": res.iterations,
        "lbfgsb_converged": bool(ref.success),
    }


def timing_run(case: Case, objective: Objective, context: CaseSolverContext | None = None) -> dict[str, Any]:
    ctx = context or CaseSolverContext(case)
    t = time.perf_counter()
    ctx.lipschitz_bound(AssembledObjective(case, objective))
    t_norms = time.perf_counter() - t
    cold = solve(case, objective, context=ctx)
    perturbed = Objective(
        terms=[t.model_copy(update={"priority": min(1000.0, t.priority * 1.5)}) if t.term == "max_dose" else t for t in objective.terms],
        smoothness_lambda=objective.smoothness_lambda,
    )
    warm = solve(case, perturbed, w0=cold.w, context=ctx)
    return {
        "norms_s": t_norms,
        "cold_s": cold.wall_s,
        "cold_iters": cold.iterations,
        "cold_converged": cold.converged,
        "cold_backtracks": cold.n_backtracks,
        "warm_s": warm.wall_s,
        "warm_iters": warm.iterations,
        "warm_converged": warm.converged,
        "cold_f": cold.objective_value,
        "cold_dose_max_gy": float(cold.dose.max()),
    }
