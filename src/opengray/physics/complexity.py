"Fluence sum-of-positive-gradients metrics and smooth optimization penalties.\n\nSPG uses the maximum row sum of positive gradients for each beam. A smooth penalty approximates the constraint during optimization. Final fluence must be checked explicitly because finite optimization and subsequent normalization can violate the intended limit. These measures do not establish clinical deliverability."

from __future__ import annotations

from typing import Any

import numpy as np

from opengray.data.base import BeamletTable

SPG_LIMIT_OPENKBP_OPT = 65.0
W_MAX_OPENKBP_OPT = 15.0
DOSE_MAX_OPENKBP_OPT = 82.0


def row_spg_terms(beamlets: BeamletTable, w: np.ndarray) -> dict[tuple[int, int], float]:
    """Sum of positive one-sided column differences per (beam, row), the OpenKBP-Opt way."""
    w = np.asarray(w, dtype=np.float64)
    key: dict[tuple[int, int, int], int] = {}
    for j in range(beamlets.n_beamlets):
        key[(int(beamlets.beam[j]), int(beamlets.row[j]), int(beamlets.col[j]))] = j
    rows: dict[tuple[int, int], float] = {}
    for (b, r, c), j in key.items():
        j2 = key.get((b, r, c + 1))
        neighbour = w[j2] if j2 is not None else 0.0
        rows[(b, r)] = rows.get((b, r), 0.0) + max(0.0, float(w[j]) - float(neighbour))
    return rows


def spg(beamlets: BeamletTable, w: np.ndarray) -> float:
    """Plan SPG: the sum over beams of the largest row sum of positive gradients in the beam."""
    rows = row_spg_terms(beamlets, w)
    per_beam: dict[int, float] = {}
    for (b, _r), v in rows.items():
        per_beam[b] = max(per_beam.get(b, 0.0), v)
    return float(sum(per_beam.values()))


def complexity_report(beamlets: BeamletTable, w: np.ndarray, dose: np.ndarray | None = None) -> dict[str, Any]:
    """SPG plus the other two OpenKBP-Opt restrictions, and whether the plan satisfies each."""
    w = np.asarray(w, dtype=np.float64)
    rows = row_spg_terms(beamlets, w)
    per_beam: dict[int, float] = {}
    for (b, _r), v in rows.items():
        per_beam[b] = max(per_beam.get(b, 0.0), v)
    total = float(sum(per_beam.values()))
    out: dict[str, Any] = {
        "spg": total,
        "spg_per_beam": [round(per_beam.get(b, 0.0), 3) for b in sorted(per_beam)],
        "spg_within_limit": total <= SPG_LIMIT_OPENKBP_OPT,
        "w_max": float(w.max()) if w.size else 0.0,
        "w_within_limit": bool(w.size == 0 or w.max() <= W_MAX_OPENKBP_OPT),
        "n_nonzero": int((w > 1e-9).sum()),
    }
    if dose is not None:
        out["dose_max_gy"] = float(np.max(dose))
        out["dose_within_limit"] = bool(np.max(dose) <= DOSE_MAX_OPENKBP_OPT)
    return out


class ComplexityTerm:
    "Smooth SPG penalty using log-sum-exp over each beam's row gradients. The approximation is at least the exact maximum and at most the maximum plus tau times the log of the row count. Excess above the limit is squared and weighted by lam."

    def __init__(self, beamlets: BeamletTable, limit: float = SPG_LIMIT_OPENKBP_OPT, tau: float = 0.5, delta: float = 0.5):
        self.limit = float(limit)
        self.tau = float(tau)
        self.delta = float(delta)  # Huber width of the positive part: smooth gradients for the solver
        n = beamlets.n_beamlets
        key = {(int(beamlets.beam[j]), int(beamlets.row[j]), int(beamlets.col[j])): j for j in range(n)}
        self.nbr = np.full(n, -1, dtype=np.int64)
        rows: dict[tuple[int, int], int] = {}
        self.row_id = np.zeros(n, dtype=np.int64)
        beam_of_row: list[int] = []
        for j in range(n):
            b, r, c = int(beamlets.beam[j]), int(beamlets.row[j]), int(beamlets.col[j])
            j2 = key.get((b, r, c + 1))
            self.nbr[j] = -1 if j2 is None else j2
            rid = rows.get((b, r))
            if rid is None:
                rid = len(rows)
                rows[(b, r)] = rid
                beam_of_row.append(b)
            self.row_id[j] = rid
        self.n_rows = len(rows)
        self.beam_of_row = np.asarray(beam_of_row, dtype=np.int64)
        beams = np.unique(self.beam_of_row)
        self.beam_index = np.searchsorted(beams, self.beam_of_row)
        self.n_beams = int(beams.size)
        self.has_nbr = self.nbr >= 0
        self.nbr_safe = np.where(self.has_nbr, self.nbr, 0)

    def row_sums(self, w: np.ndarray, smooth: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """Per-row sum of positive gradients and the per-beamlet difference to the neighbour.
        With ``smooth`` the positive part is its Huber version (x^2 / 2 delta below delta,
        x minus delta / 2 above), which the solver's penalty uses; the exact SPG never is."""
        w = np.asarray(w, dtype=np.float64)
        diff = w - np.where(self.has_nbr, w[self.nbr_safe], 0.0)
        if smooth and self.delta > 0:
            pos = np.where(diff <= 0.0, 0.0, np.where(diff < self.delta, diff * diff / (2.0 * self.delta), diff - 0.5 * self.delta))
        else:
            pos = np.maximum(diff, 0.0)
        return np.bincount(self.row_id, weights=pos, minlength=self.n_rows), diff

    def spg(self, w: np.ndarray) -> float:
        rs, _ = self.row_sums(w)
        per_beam = np.zeros(self.n_beams)
        np.maximum.at(per_beam, self.beam_index, rs)
        return float(per_beam.sum())

    def smooth_spg(self, rs: np.ndarray) -> tuple[float, np.ndarray]:
        """Log-sum-exp per beam (at least the max) and its gradient with respect to the rows."""
        out = 0.0
        grad = np.zeros_like(rs)
        for b in range(self.n_beams):
            sel = self.beam_index == b
            x = rs[sel] / self.tau
            m = float(x.max())
            e = np.exp(x - m)
            z = float(e.sum())
            out += self.tau * (m + np.log(z))
            grad[sel] = e / z
        return out, grad

    def value_and_grad(self, w: np.ndarray, lam: float) -> tuple[float, np.ndarray]:
        """lam x max(0, smooth SPG minus limit)^2 and its gradient in w, with the Huber positive
        part and the log-sum-exp row maximum, so the gradient is continuous everywhere."""
        rs, diff = self.row_sums(w, smooth=True)
        sm, drows = self.smooth_spg(rs)
        excess = sm - self.limit
        if lam <= 0.0 or excess <= 0.0:
            return 0.0, np.zeros_like(w, dtype=np.float64)
        f = lam * excess * excess
        # d(pos)/d(diff): 0 below zero, diff / delta on the Huber ramp, 1 above; the difference is
        # +1 in w_j and -1 in its neighbour, weighted by the row's share of the beam's maximum.
        if self.delta > 0:
            dpos = np.clip(diff / self.delta, 0.0, 1.0)
        else:
            dpos = (diff > 0.0).astype(np.float64)
        coef = 2.0 * lam * excess * drows[self.row_id] * dpos
        g = coef.copy()
        sel = self.has_nbr & (coef != 0.0)
        np.add.at(g, self.nbr_safe[sel], -coef[sel])
        return float(f), g
