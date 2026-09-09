from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from opengray.data.cache import read_case, write_case
from opengray.data.openkbp_opt import (
    COHORT,
    OpenKBPOptLoader,
    ingest,
    list_patients_in_zip,
    load_patient_dir,
    read_sparse_csv,
)


def test_read_sparse_csv_distinguishes_masks_from_data(synthetic_patient: dict) -> None:
    pdir: Path = synthetic_patient["dir"]
    idx, vals = read_sparse_csv(pdir / "possible_dose_mask.csv")
    assert vals is None
    assert np.array_equal(idx, synthetic_patient["feasible"])
    idx, vals = read_sparse_csv(pdir / "dose.csv")
    assert vals is not None and vals.min() > 0


def test_load_patient_dir_restricts_to_feasible_rows(synthetic_patient: dict) -> None:
    case, rep = load_patient_dir(synthetic_patient["dir"], grid_shape=synthetic_patient["grid"])
    feasible = synthetic_patient["feasible"]
    assert case.cohort == COHORT
    assert case.grid_shape == synthetic_patient["grid"]
    assert case.voxel_size_mm == pytest.approx(synthetic_patient["voxel_size"])
    assert np.array_equal(case.feasible_idx, feasible)
    assert case.D.shape == (feasible.size, synthetic_patient["n_beamlets"])
    assert case.D.indices.dtype == np.int32


    D_full = synthetic_patient["D_full"]
    w = np.ones(case.n_beamlets)
    full = np.asarray(D_full @ w).ravel()
    assert set(np.flatnonzero(full)).issubset(set(feasible))
    assert np.allclose(case.dose(w), full[feasible])

    # Reference dose reproduces the synthetic plan in feasible space (3-decimal rounding).
    assert np.allclose(case.reference_dose, synthetic_patient["dose_full"][feasible], atol=6e-4)

    # Scaling and zero invariants.
    assert np.allclose(case.dose(2.0 * w), 2.0 * case.dose(w))
    assert np.all(case.dose(np.zeros(case.n_beamlets)) == 0)


def test_structures_resolve_to_tg263_and_count_outside_voxels(synthetic_patient: dict) -> None:
    case, rep = load_patient_dir(synthetic_patient["dir"], grid_shape=synthetic_patient["grid"])
    assert set(case.structures) == {"PTV_7000", "PTV_5600", "SpinalCord", "Parotid_L", "External"}
    assert case.structures["External"].n_voxels == case.n_feasible
    assert case.structures["Parotid_L"].raw_name == "LeftParotid"
    cord = case.structures["SpinalCord"]
    assert cord.n_outside == 5
    assert cord.n_voxels == 15
    vox_cc = float(np.prod(synthetic_patient["voxel_size"])) / 1000.0
    assert cord.volume_cc == pytest.approx(15 * vox_cc)
    # Zero-dose voxels outside the mask are included in structure dose.
    d = case.structure_dose(case.dose(np.ones(case.n_beamlets)), "SpinalCord")
    assert d.size == 15 and (d[-5:] == 0).all()
    assert case.prescriptions == {"PTV_7000": 70.0, "PTV_5600": 56.0}
    assert rep.missing_structures == ["Brainstem", "RightParotid", "Esophagus", "Larynx", "Mandible", "PTV63"]
    assert any("SpinalCord: 5 of 15" in w for w in rep.warnings)
    assert case.target_names == ["PTV_7000", "PTV_5600"] or set(case.target_names) == {"PTV_7000", "PTV_5600"}


def test_beamlet_table(synthetic_patient: dict) -> None:
    case, _ = load_patient_dir(synthetic_patient["dir"], grid_shape=synthetic_patient["grid"])
    assert case.beamlets is not None
    assert case.beamlets.n_beamlets == case.n_beamlets
    assert case.beamlets.beam.min() == 0 and case.beamlets.beam.max() == 8
    assert case.beam_geometry["n_beams"] == 9


def test_cache_roundtrip(synthetic_patient: dict, tmp_path: Path) -> None:
    case, _ = load_patient_dir(synthetic_patient["dir"], grid_shape=synthetic_patient["grid"])
    cache = tmp_path / "cache"
    write_case(case, cache)
    back = read_case(cache, COHORT, case.case_id)
    assert back.case_id == case.case_id
    assert np.array_equal(back.feasible_idx, case.feasible_idx)
    assert (back.D != case.D).nnz == 0
    assert np.array_equal(back.reference_dose, case.reference_dose)
    assert set(back.structures) | {"External"} == set(case.structures)
    for n, s in case.structures.items():
        if n == "External":
            continue
        assert np.array_equal(back.structures[n].mask_idx, s.mask_idx)
        assert back.structures[n].n_outside == s.n_outside
        assert back.structures[n].raw_name == s.raw_name
    assert back.prescriptions == case.prescriptions
    assert back.beamlets is not None and np.array_equal(back.beamlets.beam, case.beamlets.beam)
    assert back.provenance["conversions"] == case.provenance["conversions"]


def test_ingest_archive_is_resumable(synthetic_archive: Path, tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    assert list_patients_in_zip(synthetic_archive) == ["pt_1", "pt_2", "pt_3", "pt_4"]
    seen: list[str] = []
    manifest = ingest(synthetic_archive, cache, patients=["pt_1", "pt_2"], progress=lambda p, e: seen.append(p), grid_shape=(8, 8, 8))
    assert seen == ["pt_1", "pt_2"]
    assert set(manifest["cases"]) == {"pt_1", "pt_2"}
    assert "dij.npz" in manifest["cases"]["pt_1"]["source_sha256"]

    skipped: list[str] = []
    manifest = ingest(synthetic_archive, cache, progress=lambda p, e: skipped.append(p) if e.get("skipped") else None, grid_shape=(8, 8, 8))
    assert skipped == ["pt_1", "pt_2"]
    assert set(manifest["cases"]) == {"pt_1", "pt_2", "pt_3", "pt_4"}

    loader = OpenKBPOptLoader(cache)
    assert loader.list_cases() == ["pt_1", "pt_2", "pt_3", "pt_4"]
    case = loader.load("pt_3")
    assert case.n_beamlets == 12
    assert case.provenance["source_sha256"]["dij.npz"] == manifest["cases"]["pt_3"]["source_sha256"]["dij.npz"]
