"Shared case, structure, and beamlet data objects.\n\nDose influence rows represent feasible-dose voxels. Structure indices address that space; n_outside retains zero-dose voxels outside it. Every voxel carries a volume in cubic centimetres. Doses and prescriptions use gray, and source conversions belong in provenance."

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np
import scipy.sparse as sp
from pydantic import BaseModel, ConfigDict, Field, field_validator


class Structure(BaseModel):
    """A labeled region given as indices into the case's feasible-voxel space."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = Field(description="TG-263 canonical name after alias resolution")
    raw_name: str = Field(description="Name as found in the source dataset")
    mask_idx: np.ndarray = Field(description="int32 indices into the feasible-voxel space")
    n_outside: int = Field(
        default=0, ge=0, description="Structure voxels outside the feasible mask (zero dose)"
    )
    volume_cc: float = Field(ge=0.0)

    @field_validator("mask_idx")
    @classmethod
    def _as_int32(cls, v: np.ndarray) -> np.ndarray:
        arr = np.asarray(v)
        if arr.ndim != 1:
            raise ValueError("mask_idx must be one-dimensional")
        return np.ascontiguousarray(arr, dtype=np.int32)

    @property
    def n_voxels(self) -> int:
        """Total voxel count, including voxels outside the feasible mask."""
        return int(self.mask_idx.size) + int(self.n_outside)

    @property
    def is_target(self) -> bool:
        return self.name.upper().startswith("PTV")


class BeamletTable(BaseModel):
    """Per-beamlet fluence-map coordinates, one entry per column of ``D``."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    row: np.ndarray
    col: np.ndarray
    beam: np.ndarray = Field(description="Beam index, zero-based, one per gantry angle")

    @property
    def n_beamlets(self) -> int:
        return int(self.row.size)

    @property
    def n_beams(self) -> int:
        return int(np.unique(self.beam).size)


class Case(BaseModel):
    """One planning case: anatomy, influence matrix, prescriptions, optional reference plan."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    case_id: str
    cohort: str
    grid_shape: tuple[int, int, int] | None = Field(
        default=None, description="Source grid shape when the cohort has a regular grid"
    )
    voxel_size_mm: tuple[float, float, float] | None = None
    feasible_idx: np.ndarray = Field(
        description="int64 flat source-grid index of each feasible voxel (row of D), sorted"
    )
    voxel_volume_cc: np.ndarray = Field(description="Volume of each feasible voxel, cm3")
    D: sp.csr_matrix = Field(description="Sparse (n_feasible x n_beamlets) dose per unit intensity, Gy")
    structures: dict[str, Structure] = Field(default_factory=dict)
    prescriptions: dict[str, float] = Field(default_factory=dict, description="Gy, keyed by target name")
    reference_dose: np.ndarray | None = Field(
        default=None, description="Feasible-space dose of the clinical reference plan, Gy"
    )
    beamlets: BeamletTable | None = None
    beam_geometry: dict[str, Any] = Field(default_factory=dict)
    provenance: dict[str, Any] = Field(
        default_factory=dict, description="Source files, checksums, conversions, and load warnings"
    )

    @field_validator("D")
    @classmethod
    def _as_csr(cls, v: sp.spmatrix) -> sp.csr_matrix:
        return sp.csr_matrix(v)

    @property
    def n_feasible(self) -> int:
        return int(self.D.shape[0])

    @property
    def n_beamlets(self) -> int:
        return int(self.D.shape[1])

    @property
    def target_names(self) -> list[str]:
        return [n for n, s in self.structures.items() if s.is_target]

    @property
    def oar_names(self) -> list[str]:
        return [n for n, s in self.structures.items() if not s.is_target]

    def dose(self, w: np.ndarray) -> np.ndarray:
        """Feasible-space dose for beamlet intensities ``w`` (Gy)."""
        w = np.asarray(w, dtype=np.float64)
        if w.shape != (self.n_beamlets,):
            raise ValueError(f"w must have shape ({self.n_beamlets},), got {w.shape}")
        return np.asarray(self.D @ w).ravel()

    def structure_dose(self, dose: np.ndarray, name: str) -> np.ndarray:
        """Dose values for every voxel of a structure, including zero-dose voxels outside the mask."""
        s = self.structures[name]
        d = dose[s.mask_idx]
        if s.n_outside:
            d = np.concatenate([d, np.zeros(s.n_outside, dtype=d.dtype)])
        return d

    def summary(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "cohort": self.cohort,
            "n_feasible": self.n_feasible,
            "n_beamlets": self.n_beamlets,
            "nnz": int(self.D.nnz),
            "structures": {n: s.n_voxels for n, s in self.structures.items()},
            "prescriptions": dict(self.prescriptions),
            "has_reference": self.reference_dose is not None,
        }


@runtime_checkable
class CohortLoader(Protocol):
    """Every cohort exposes the same two calls; everything else is cohort-specific."""

    cohort: str

    def list_cases(self) -> list[str]: ...

    def load(self, case_id: str) -> Case: ...
