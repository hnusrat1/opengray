"Projected accelerated-gradient fluence optimization with backtracking and adaptive restart.\n\nThe default solver enforces nonnegative weights, with 500 iterations and a relative objective tolerance of 1e-4 over ten iterations. Stopping reason and numerical diagnostics are returned. An optional beamlet cap and smooth SPG penalty support complexity experiments. Finite penalty optimization and post-solve normalization do not guarantee final constraint satisfaction or clinical deliverability."

from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.sparse as sp

from opengray.data.base import Case
from opengray.physics.complexity import SPG_LIMIT_OPENKBP_OPT, W_MAX_OPENKBP_OPT
from opengray.physics.objectives import AssembledObjective, Objective, smoothness_operator


@dataclass
class SolverConfig:
    max_iter: int = 500
    rel_tol: float = 1e-4
    abs_tol: float = 1e-8
    patience: int = 10





    w_max: float | None = None
    complexity_limit: float | None = None
    complexity_tol: float = 0.02
    complexity_rounds: int = 15
    complexity_inner_iter: int = 200
    complexity_tau: float = 0.5
    complexity_delta: float = 0.5
    step_growth: float = 1.05
    step_shrink: float = 0.5
    restart: bool = True
    record_history: bool = False


def deliverable_config(**overrides: Any) -> SolverConfig:
    "Target reference-style fluence complexity: weights at most 15 and SPG at most 65. Check final values explicitly; these settings do not certify clinical deliverability."
    return SolverConfig(**{"w_max": W_MAX_OPENKBP_OPT, "complexity_limit": SPG_LIMIT_OPENKBP_OPT, **overrides})


def fingerprint(cfg: SolverConfig | None) -> dict[str, Any]:
    """The configuration as the protocol hashes it. The complexity fields are dropped when no
    limit is set, so the default regime hashes as it did before they existed and every stored
    run keeps its protocol id."""
    d = dataclasses.asdict(cfg or SolverConfig())
    if d.get("complexity_limit") is None:
        d = {k: v for k, v in d.items() if not k.startswith("complexity_")}
    return d


@dataclass
class SolveResult:
    w: np.ndarray
    dose: np.ndarray
    objective_value: float
    iterations: int
    converged: bool
    wall_s: float
    n_backtracks: int
    n_restarts: int
    step_final: float
    history: list[float] = field(default_factory=list)
    stop_reason: str = ""
    # First-order stationarity at the returned point: the largest entry of w - P(w - grad f(w)),
    # zero at a minimizer over w >= 0, and the same scaled by the gradient's largest entry.
    grad_residual: float = 0.0
    grad_residual_rel: float = 0.0
    # Fluence complexity at the returned point (None when the case has no beamlet table), the
    # number of penalty rounds the continuation needed, the final penalty weight, and the
    # penalty's value at the returned point (objective_value excludes it).
    spg: float | None = None
    complexity_rounds: int = 0
    complexity_lambda: float = 0.0  # the augmented Lagrangian's final rho
    complexity_multiplier: float = 0.0  # its final multiplier mu
    penalty_value: float = 0.0


def operator_norm(M: sp.spmatrix, n_iter: int = 50, seed: int = 0) -> float:
    """Largest singular value of a sparse matrix by power iteration on M^T M."""
    M = sp.csr_matrix(M)
    if M.shape[0] == 0 or M.shape[1] == 0 or M.nnz == 0:
        return 0.0
    if min(M.shape) <= 2:
        return float(np.linalg.norm(M.toarray(), 2))
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(M.shape[1])
    v /= np.linalg.norm(v)
    MT = M.T.tocsr()
    s = 0.0
    for _ in range(n_iter):
        u = MT @ (M @ v)
        s_new = float(np.linalg.norm(u))
        if s_new == 0.0:
            return 0.0
        v = u / s_new
        if abs(s_new - s) <= 1e-9 * max(s_new, 1.0):
            s = s_new
            break
        s = s_new
    return float(np.sqrt(s))


class CaseSolverContext:
    """Per-case quantities cached across solves: operator norms of D, of each D_s, and of L."""

    def __init__(self, case: Case):
        self.case = case
        self._D_norm: float | None = None
        self._structure_norms: dict[str, float] = {}
        self._L_norm: float | None = None

    @property
    def D_norm(self) -> float:
        if self._D_norm is None:
            self._D_norm = operator_norm(self.case.D)
        return self._D_norm

    def structure_norm(self, name: str) -> float:
        if name not in self._structure_norms:
            idx = self.case.structures[name].mask_idx
            self._structure_norms[name] = operator_norm(self.case.D[idx, :]) if idx.size else 0.0
        return self._structure_norms[name]

    @property
    def L_norm(self) -> float:
        if self._L_norm is None:
            if self.case.beamlets is None:
                self._L_norm = 0.0
            else:
                self._L_norm = operator_norm(smoothness_operator(self.case.beamlets))
        return self._L_norm

    def lipschitz_bound(self, obj: AssembledObjective) -> float:
        norms = {t.structure: self.structure_norm(t.structure) for t in obj.objective.terms}
        return obj.lipschitz_bound(norms, self.D_norm, self.L_norm)

    def complexity_term(self, limit: float, tau: float, delta: float = 0.5):
        key = (float(limit), float(tau), float(delta))
        cache = getattr(self, "_complexity", {})
        if key not in cache:
            from opengray.physics.complexity import ComplexityTerm

            cache[key] = ComplexityTerm(self.case.beamlets, limit=limit, tau=tau, delta=delta) if self.case.beamlets is not None else None
            self._complexity = cache
        return cache[key]


def _project(w: np.ndarray, w_max: float | None) -> np.ndarray:
    np.maximum(w, 0.0, out=w)
    if w_max is not None:
        np.minimum(w, w_max, out=w)
    return w


def solve(
    case: Case,
    objective: Objective,
    w0: np.ndarray | None = None,
    config: SolverConfig | None = None,
    context: CaseSolverContext | None = None,
) -> SolveResult:
    """Minimize the objective over 0 <= w <= w_max starting from ``w0`` (zeros when None), then,
    when the case carries a beamlet table and ``complexity_limit`` is set, continue with an
    augmented Lagrangian on the plan's SPG (rounds of ``complexity_inner_iter`` iterations from
    the previous result) until the exact SPG is within tolerance of the limit or the rounds are
    spent. ``objective_value`` is the agent's objective at the returned point; ``penalty_value``
    the augmented term's."""
    cfg = config or SolverConfig()
    ctx = context or CaseSolverContext(case)
    obj = AssembledObjective(case, objective)
    term = ctx.complexity_term(cfg.complexity_limit, cfg.complexity_tau, cfg.complexity_delta) if cfg.complexity_limit is not None else None
    res = _fista(case, obj, None, 0.0, w0, cfg, ctx)
    if term is None:
        if case.beamlets is not None:
            from opengray.physics.complexity import spg as spg_of

            res.spg = float(spg_of(case.beamlets, res.w))
        return res
    spg = term.spg(res.w)
    rounds = 0
    iterations, wall, backtracks, restarts = res.iterations, res.wall_s, res.n_backtracks, res.n_restarts
    target = float(cfg.complexity_limit)
    # Augmented Lagrangian on the smoothed SPG: minimize f(w) + (rho / 2) max(0, mu / rho + S(w)
    # minus limit)^2, then mu <- max(0, mu + rho (S minus limit)); rho grows when a round fails to
    # shrink the violation. The smoothed limit sits below the target by the gap between the
    # exact and smoothed SPG at the current point, so the exact value is what lands under it.
    rho, mu = 0.0, 0.0
    prev_viol = spg - target
    inner = SolverConfig(**{**dataclasses.asdict(cfg), "max_iter": cfg.complexity_inner_iter, "complexity_limit": None})
    while spg > target * (1.0 + cfg.complexity_tol) and rounds < cfg.complexity_rounds:
        rounds += 1
        rs, _ = term.row_sums(res.w, smooth=True)
        smooth, _ = term.smooth_spg(rs)
        slimit = target - max(0.0, spg - smooth) - 0.01 * target
        if rho == 0.0:
            rho = 20.0 * max(res.objective_value, 1e-6) / max(spg - target, 1e-6) ** 2
        term.limit = slimit - mu / rho
        res = _fista(case, obj, term, rho / 2.0, res.w, inner, ctx)
        spg = term.spg(res.w)
        iterations += res.iterations
        wall += res.wall_s
        backtracks += res.n_backtracks
        restarts += res.n_restarts
        viol = spg - target
        mu = max(0.0, mu + rho * (spg - slimit))
        if viol > 0.7 * prev_viol:
            rho *= 3.0
        prev_viol = viol
    term.limit = target
    res.spg = spg
    res.complexity_rounds = rounds
    res.complexity_lambda = rho
    res.complexity_multiplier = mu
    res.iterations, res.wall_s, res.n_backtracks, res.n_restarts = iterations, wall, backtracks, restarts
    return res


def _fista(case: Case, obj: AssembledObjective, term, lam: float, w0: np.ndarray | None, cfg: SolverConfig, ctx: CaseSolverContext) -> SolveResult:
    n = case.n_beamlets
    t0 = time.perf_counter()

    def value_and_grad(x: np.ndarray) -> tuple[float, np.ndarray]:
        f, g = obj.value_and_grad(x)
        if term is not None and lam > 0.0:
            fp, gp = term.value_and_grad(x, lam)
            return f + fp, g + gp
        return f, g

    def value(x: np.ndarray) -> float:
        return value_and_grad(x)[0]

    w = np.zeros(n) if w0 is None else np.array(w0, dtype=np.float64, copy=True)
    _project(w, cfg.w_max)
    L_bound = ctx.lipschitz_bound(obj)
    # The bound is safe but often loose; start optimistic and let backtracking correct it.
    L = max(L_bound / 8.0, 1e-12)

    y = w.copy()
    t = 1.0
    f_prev = value(w)
    history = [f_prev] if cfg.record_history else []
    recent: list[float] = [f_prev]
    n_backtracks = 0
    n_restarts = 0
    converged = False
    stop_reason = "max_iter"
    k = 0
    for k in range(1, cfg.max_iter + 1):  # noqa: B007 - k is reported after the loop
        f_y, g_y = value_and_grad(y)
        # Backtracking on the quadratic upper bound at y.
        backtracked = False
        while True:
            w_new = _project(y - g_y / L, cfg.w_max)
            diff = w_new - y
            f_new = value(w_new)
            q = f_y + float(g_y @ diff) + 0.5 * L * float(diff @ diff)
            if f_new <= q + 1e-12 * max(1.0, abs(f_y)):
                break
            L /= cfg.step_shrink
            n_backtracks += 1
            backtracked = True
            if L > 1e30:
                raise RuntimeError("FISTA backtracking diverged")
        # Adaptive restart when momentum points uphill.
        if cfg.restart and float((y - w_new) @ (w_new - w)) > 0:
            t = 1.0
            n_restarts += 1
            y = w_new.copy()
        else:
            t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
            y = w_new + ((t - 1.0) / t_new) * (w_new - w)
            t = t_new
        w = w_new
        # Let the step grow slowly after a clean iteration; backtracking will pull it back.
        if not backtracked:
            L = max(L / cfg.step_growth, L_bound / 64.0)
        if cfg.record_history:
            history.append(f_new)
        recent.append(f_new)
        if len(recent) > cfg.patience + 1:
            recent.pop(0)
        if len(recent) == cfg.patience + 1:
            change = abs(recent[0] - recent[-1])
            if change / max(abs(recent[0]), 1e-12) < cfg.rel_tol:
                converged = True
                stop_reason = "rel_tol"
                break
            if change < cfg.abs_tol:
                converged = True
                stop_reason = "abs_tol"
                break
        if f_new <= cfg.abs_tol:
            converged = True
            stop_reason = "zero_objective"
            break

    f_total, g_final = value_and_grad(w)
    f_base = obj.value(w) if (term is not None and lam > 0.0) else f_total
    resid = float(np.max(np.abs(w - _project(w - g_final, cfg.w_max)))) if n else 0.0
    g_scale = float(np.max(np.abs(g_final))) if n else 0.0
    return SolveResult(
        w=w,
        dose=obj.dose(w),
        objective_value=f_base,
        penalty_value=f_total - f_base,
        iterations=k,
        converged=converged,
        wall_s=time.perf_counter() - t0,
        n_backtracks=n_backtracks,
        n_restarts=n_restarts,
        step_final=1.0 / L,
        history=history,
        stop_reason=stop_reason,
        grad_residual=resid,
        grad_residual_rel=resid / max(g_scale, 1e-12),
    )
