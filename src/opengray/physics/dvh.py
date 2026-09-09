"""DVH curves and dose metrics, matching the OpenKBP reference implementation where one exists.

Reference (`dose_evaluation_class.py` in open-kbp-opt):

* ``D_p`` (dose received by p percent of the volume) is ``np.percentile(dose, 100 - p)``,
  so D99 is the first percentile and D1 the 99th, numpy linear interpolation.
* ``D_{v}cc`` uses ``n = max(1, round(v_mm3 / voxel_mm3))`` voxels and is
  ``np.percentile(dose, 100 - n / n_voxels * 100)``.
* ``Dmean`` is the arithmetic mean over all structure voxels (zero-dose voxels outside the
  feasible mask included).

Additional metrics (not in the reference): ``Dmax`` (maximum voxel dose), ``V_xGy`` (fraction
of the volume receiving at least x Gy), Paddick conformity index, and the homogeneity index
D5 / D95. All doses are in Gy.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

import numpy as np

from opengray.data.base import Case

MetricKind = Literal["Dmean", "Dmax", "Dpercent", "Dcc", "Vgy", "CI", "HI"]

_METRIC_RE = re.compile(
    r"^(?:(?P<mean>Dmean)|(?P<max>Dmax)|(?P<ci>CI)|(?P<hi>HI)|D(?P<cc>\d+(?:\.\d+)?)cc|D(?P<pct>\d+(?:\.\d+)?)|V(?P<gy>\d+(?:\.\d+)?)Gy)$"
)


@dataclass(frozen=True)
class MetricSpec:
    kind: MetricKind
    param: float | None = None

    @property
    def name(self) -> str:
        if self.kind == "Dpercent":
            return f"D{_fmt(self.param)}"
        if self.kind == "Dcc":
            return f"D{_fmt(self.param)}cc"
        if self.kind == "Vgy":
            return f"V{_fmt(self.param)}Gy"
        return self.kind

    @property
    def targets_only(self) -> bool:
        return self.kind in ("CI", "HI")


def _fmt(x: float | None) -> str:
    if x is None:
        return ""
    return f"{x:g}"


def parse_metric(text: str) -> MetricSpec:
    m = _METRIC_RE.match(text.strip())
    if not m:
        raise ValueError(f"unknown metric {text!r}; expected Dmean, Dmax, D<p>, D<v>cc, V<x>Gy, CI, or HI")
    if m.group("mean"):
        return MetricSpec("Dmean")
    if m.group("max"):
        return MetricSpec("Dmax")
    if m.group("ci"):
        return MetricSpec("CI")
    if m.group("hi"):
        return MetricSpec("HI")
    if m.group("cc") is not None:
        v = float(m.group("cc"))
        if v <= 0:
            raise ValueError("D<v>cc needs v > 0")
        return MetricSpec("Dcc", v)
    if m.group("pct") is not None:
        p = float(m.group("pct"))
        if not 0 < p <= 100:
            raise ValueError("D<p> needs 0 < p <= 100")
        return MetricSpec("Dpercent", p)
    x = float(m.group("gy"))
    if x < 0:
        raise ValueError("V<x>Gy needs x >= 0")
    return MetricSpec("Vgy", x)


def d_percent(dose: np.ndarray, p: float) -> float:
    return float(np.percentile(dose, 100.0 - p))


def d_cc(dose: np.ndarray, v_cc: float, voxel_cc: float) -> float:
    n_voxels = max(1, round(v_cc / voxel_cc))
    return float(np.percentile(dose, 100.0 - n_voxels / dose.size * 100.0))


def d_mean(dose: np.ndarray) -> float:
    return float(dose.mean())


def d_max(dose: np.ndarray) -> float:
    return float(dose.max())


def v_gy(dose: np.ndarray, x: float) -> float:
    return float((dose >= x).mean())


def homogeneity_index(dose: np.ndarray) -> float:
    d95 = d_percent(dose, 95)
    return float(d_percent(dose, 5) / d95) if d95 > 0 else float("inf")


def paddick_ci(target_dose: np.ndarray, all_dose: np.ndarray, prescription: float) -> float:
    """Paddick conformity index: TV_PIV^2 / (TV * PIV), PIV over the feasible region."""
    tv = target_dose.size
    tv_piv = int((target_dose >= prescription).sum())
    piv = int((all_dose >= prescription).sum())
    if tv == 0 or piv == 0:
        return 0.0
    return float(tv_piv**2 / (tv * piv))


def dvh_curve(dose: np.ndarray, n_points: int = 50, d_max: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Cumulative DVH: dose levels and the fraction of the volume receiving at least that dose."""
    top = float(dose.max()) if d_max is None else d_max
    levels = np.linspace(0.0, max(top, 1e-9), n_points)
    sorted_dose = np.sort(dose)
    frac = 1.0 - np.searchsorted(sorted_dose, levels, side="left") / dose.size
    return levels, frac


def merged_target_indices(case: Case, target: str) -> tuple[np.ndarray, int]:
    """Union of a target with all higher-prescription targets (OpenKBP's criteria convention)."""
    rx = case.prescriptions[target]
    idx = [case.structures[target].mask_idx]
    n_out = case.structures[target].n_outside
    for other, rx_o in case.prescriptions.items():
        if other != target and rx_o > rx:
            idx.append(case.structures[other].mask_idx)
            n_out += case.structures[other].n_outside
    return np.unique(np.concatenate(idx)), n_out


def structure_dose(case: Case, dose: np.ndarray, structure: str, merge_targets: bool) -> np.ndarray:
    if merge_targets and structure in case.prescriptions:
        idx, n_out = merged_target_indices(case, structure)
        d = dose[idx]
        return np.concatenate([d, np.zeros(n_out)]) if n_out else d
    return case.structure_dose(dose, structure)


def evaluate(case: Case, dose: np.ndarray, structure: str, metric: MetricSpec | str, merge_targets: bool = True) -> float:
    """Evaluate one metric of one structure on a feasible-space dose vector."""
    spec = parse_metric(metric) if isinstance(metric, str) else metric
    if structure not in case.structures:
        raise KeyError(structure)
    if spec.targets_only and structure not in case.prescriptions:
        raise ValueError(f"{spec.name} is defined for targets only")
    ds = structure_dose(case, dose, structure, merge_targets)
    if ds.size == 0:
        return float("nan")
    if spec.kind == "Dmean":
        return d_mean(ds)
    if spec.kind == "Dmax":
        return d_max(ds)
    if spec.kind == "Dpercent":
        return d_percent(ds, float(spec.param))  # type: ignore[arg-type]
    if spec.kind == "Dcc":
        voxel_cc = float(np.mean(case.voxel_volume_cc[case.structures[structure].mask_idx])) if case.structures[structure].mask_idx.size else float(np.mean(case.voxel_volume_cc))
        return d_cc(ds, float(spec.param), voxel_cc)  # type: ignore[arg-type]
    if spec.kind == "Vgy":
        return v_gy(ds, float(spec.param))  # type: ignore[arg-type]
    if spec.kind == "HI":
        return homogeneity_index(ds)
    if spec.kind == "CI":
        return paddick_ci(ds, dose, case.prescriptions[structure])
    raise AssertionError(spec.kind)
