from __future__ import annotations

import pytest

from opengray.physics.nomenclature import (
    ALIASES,
    CANONICAL_OARS,
    CANONICAL_TARGETS,
    is_target_name,
    resolve,
    resolve_relative_target,
)


@pytest.mark.parametrize(
    ("raw", "canonical"),
    [
        ("LeftParotid", "Parotid_L"),
        ("Parotid_L", "Parotid_L"),
        ("parotid l", "Parotid_L"),
        ("Lt_Parotid", "Parotid_L"),
        ("RightParotid", "Parotid_R"),
        ("Cord", "SpinalCord"),
        ("spinal cord", "SpinalCord"),
        ("Mandible", "Bone_Mandible"),
        ("PTV70", "PTV_7000"),
        ("ptv_70", "PTV_7000"),
        ("PTV56", "PTV_5600"),
        ("Brainstem", "Brainstem"),
    ],
)
def test_resolve_known_aliases(raw: str, canonical: str) -> None:
    r = resolve(raw)
    assert r.known and r.canonical == canonical


def test_unknown_names_are_returned_unchanged() -> None:
    r = resolve("Lens_L")
    assert not r.known and r.canonical == "Lens_L" and not r.changed


def test_every_canonical_name_resolves_to_itself() -> None:
    for name in CANONICAL_OARS + CANONICAL_TARGETS:
        assert resolve(name).canonical == name
        assert name in ALIASES


def test_target_detection() -> None:
    assert is_target_name("PTV_7000") and is_target_name("ptv56")
    assert not is_target_name("SpinalCord")


def test_relative_targets() -> None:
    three = ["PTV_5600", "PTV_6300", "PTV_7000"]
    assert resolve_relative_target("PTV_High", three) == "PTV_7000"
    assert resolve_relative_target("PTV_Mid", three) == "PTV_6300"
    assert resolve_relative_target("PTV_Low", three) == "PTV_5600"
    two = ["PTV_7000", "PTV_5600"]
    assert resolve_relative_target("PTV_Low", two) == "PTV_5600"
    assert resolve_relative_target("PTV_Mid", two) is None
    assert resolve_relative_target("PTV_Low", ["PTV_7000"]) is None
    assert resolve_relative_target("PTV_High", []) is None
