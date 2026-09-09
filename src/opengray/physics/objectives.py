"Fluence objective terms, analytic gradients, and smoothness operators."

from __future__ import annotations

import hashlib
import json
from typing import Literal

import numpy as np
import scipy.sparse as sp
from pydantic import BaseModel, Field, field_validator, model_validator

from opengray.data.base import BeamletTable, Case
from opengray.physics.linalg import BlockedCSR

TermType = Literal["min_dose", "max_dose", "mean_dose", "uniform_dose", "dvh_max"]

PRIORITY_MAX = 1000.0
DEFAULT_SMOOTHNESS_LAMBDA = 1e-4


class ObjectiveTerm(BaseModel):
    structure: str
    term: TermType
    level: float = Field(ge=0.0, description="Gy")
    priority: float = Field(ge=0.0, le=PRIORITY_MAX)
    volume_fraction: float | None = Field(
        default=None, gt=0.0, lt=1.0, description="dvh_max only: allowed fraction above level"
    )

    @model_validator(mode="after")
    def _check_dvh(self) -> ObjectiveTerm:
        if self.term == "dvh_max" and self.volume_fraction is None:
            raise ValueError("dvh_max requires volume_fraction")
        if self.term != "dvh_max" and self.volume_fraction is not None:
            raise ValueError("volume_fraction is only valid for dvh_max")
        return self


class Objective(BaseModel):
    terms: list[ObjectiveTerm] = Field(min_length=1)
    smoothness_lambda: float = Field(default=DEFAULT_SMOOTHNESS_LAMBDA, ge=0.0)

    @field_validator("terms")
    @classmethod
    def _nonempty(cls, v: list[ObjectiveTerm]) -> list[ObjectiveTerm]:
        if not v:
            raise ValueError("an objective needs at least one term")
        return v

    def validate_against(self, case: Case) -> list[str]:
        """Return problems that make this objective unusable on ``case`` (empty list if fine)."""
        problems = []
        for i, t in enumerate(self.terms):
            if t.structure not in case.structures:
                problems.append(f"term {i}: structure {t.structure!r} not in case")
        return problems

    def hash(self) -> str:
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


def smoothness_operator(beamlets: BeamletTable) -> sp.csr_matrix:
    """First-difference operator between fluence-map neighbours within each beam.

    Rows are (beam, row, col) to (beam, row, col + 1) and (beam, row + 1, col) pairs that both
    exist; each row of L has +1 and -1. ``||L w||^2`` is the sum of squared neighbour differences.
    """
    key = {}
    for j in range(beamlets.n_beamlets):
        key[(int(beamlets.beam[j]), int(beamlets.row[j]), int(beamlets.col[j]))] = j
    rows, cols, vals = [], [], []
    r = 0
    for (b, i, k), j in key.items():
        for nb in ((b, i, k + 1), (b, i + 1, k)):
            j2 = key.get(nb)
            if j2 is not None:
                rows += [r, r]
                cols += [j, j2]
                vals += [1.0, -1.0]
                r += 1
    if r == 0:
        return sp.csr_matrix((0, beamlets.n_beamlets))
    return sp.csr_matrix((vals, (rows, cols)), shape=(r, beamlets.n_beamlets))


class AssembledObjective:
    """Value and gradient of an :class:`Objective` on a :class:`Case`, as a function of w.

    Structure doses are gathered once per evaluation from the feasible-space dose; voxels
    outside the feasible mask are appended as zeros (they have no D rows, so they contribute
    to counts and means but never to the gradient).
    """

    def __init__(self, case: Case, objective: Objective):
        problems = objective.validate_against(case)
        if problems:
            raise ValueError("; ".join(problems))
        self.case = case
        self.objective = objective
        self.D = case.D
        self.DT = case.D.T.tocsr()
        # Threaded products (physics.linalg): bit-identical to D @ w and D^T @ g, faster.
        self._Db = BlockedCSR(self.D)
        self._DTb = BlockedCSR(self.DT)
        self.n_beamlets = case.n_beamlets
        self._terms = []
        for t in objective.terms:
            s = case.structures[t.structure]
            self._terms.append((t, s.mask_idx, int(s.n_outside)))
        self.L = None
        self.LtL = None
        if objective.smoothness_lambda > 0 and case.beamlets is not None:
            L = smoothness_operator(case.beamlets)
            if L.shape[0] > 0:
                self.L = L
                self.LtL = (L.T @ L).tocsr()

    def dose(self, w: np.ndarray) -> np.ndarray:
        return self._Db.matvec(w)

    def value_and_grad(self, w: np.ndarray) -> tuple[float, np.ndarray]:
        return self._evaluate(w, need_grad=True)

    def _evaluate(self, w: np.ndarray, need_grad: bool) -> tuple[float, np.ndarray]:
        d = self.dose(w)
        f = 0.0
        g_d = np.zeros_like(d) if need_grad else None
        for t, idx, n_out in self._terms:
            ds = d[idx]
            n = idx.size + n_out
            p = t.priority
            if p == 0 or n == 0:
                continue
            if t.term == "min_dose":
                under = np.maximum(0.0, t.level - ds)
                # Voxels outside the mask have zero dose and contribute (level)^2 each.
                f += (p / n) * (float(under @ under) + n_out * t.level**2)
                if need_grad:
                    g_d[idx] += (p / n) * (-2.0) * under
            elif t.term == "max_dose":
                over = np.maximum(0.0, ds - t.level)
                f += (p / n) * float(over @ over)
                if need_grad:
                    g_d[idx] += (p / n) * 2.0 * over
            elif t.term == "mean_dose":
                mean = float(ds.sum()) / n
                excess = max(0.0, mean - t.level)
                f += p * excess**2
                if need_grad and excess > 0:
                    g_d[idx] += p * 2.0 * excess / n
            elif t.term == "uniform_dose":
                diff = ds - t.level
                f += (p / n) * (float(diff @ diff) + n_out * t.level**2)
                if need_grad:
                    g_d[idx] += (p / n) * 2.0 * diff
            elif t.term == "dvh_max":
                assert t.volume_fraction is not None
                # Dose at the allowed volume fraction: the (1 - vf) quantile of the full structure
                # (zeros included). Voxels between level and that dose are pushed down.
                full = ds if n_out == 0 else np.concatenate([ds, np.zeros(n_out)])
                d_v = float(np.quantile(full, 1.0 - t.volume_fraction))
                if d_v > t.level:
                    sel = (ds > t.level) & (ds <= d_v)
                    over = np.where(sel, ds - t.level, 0.0)
                    f += (p / n) * float(over @ over)
                    if need_grad:
                        g_d[idx] += (p / n) * 2.0 * over
        lam = self.objective.smoothness_lambda
        if not need_grad:
            if self.LtL is not None and lam > 0:
                Lw = self.L @ w
                f += lam * float(Lw @ Lw)
            return float(f), np.empty(0)
        g = self._DTb.matvec(g_d)
        if self.LtL is not None and lam > 0:
            Lw = self.L @ w
            f += lam * float(Lw @ Lw)
            g += 2.0 * lam * np.asarray(self.LtL @ w).ravel()
        return float(f), g

    def value(self, w: np.ndarray) -> float:
        """The objective alone: one dose product, no gradient (the backtracking test's cost)."""
        return self._evaluate(w, need_grad=False)[0]

    def lipschitz_bound(self, structure_norms: dict[str, float], D_norm: float, L_norm: float = 0.0) -> float:
        """Upper bound on the gradient Lipschitz constant.

        Each quadratic-type term has Hessian D_s^T C D_s with C <= (2 p / n_s) I, so its
        contribution is at most 2 p / n_s * ||D_s||_2^2; the mean term satisfies the same bound.
        ``structure_norms`` holds ||D_s||_2 per structure (falling back to ||D||_2).
        """
        total = 0.0
        for t, idx, n_out in self._terms:
            n = idx.size + n_out
            if n == 0 or t.priority == 0:
                continue
            ns = structure_norms.get(t.structure, D_norm)
            total += 2.0 * t.priority / n * ns**2
        if self.LtL is not None:
            total += 2.0 * self.objective.smoothness_lambda * L_norm**2
        return float(total)
