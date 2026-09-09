"Canonical structure naming and explicit aliases for supported source data."

from __future__ import annotations

import re
from dataclasses import dataclass

CANONICAL_OARS: tuple[str, ...] = (
    "External",
    "Brainstem",
    "SpinalCord",
    "Parotid_L",
    "Parotid_R",
    "Esophagus",
    "Larynx",
    "Bone_Mandible",
)

CANONICAL_TARGETS: tuple[str, ...] = ("PTV_7000", "PTV_6300", "PTV_5600")

ALIASES: dict[str, tuple[str, ...]] = {
    "External": ("External", "Body", "BODY", "Skin", "Patient", "possible_dose_mask"),
    "Brainstem": ("Brainstem", "BrainStem", "Brain_Stem", "brain stem", "BS"),
    "SpinalCord": ("SpinalCord", "Spinal_Cord", "spinal cord", "Cord", "SC"),
    "Parotid_L": ("Parotid_L", "LeftParotid", "Parotid_Lt", "Lt_Parotid", "L_Parotid", "LParotid", "Parotid Left", "Parotid_Left"),
    "Parotid_R": ("Parotid_R", "RightParotid", "Parotid_Rt", "Rt_Parotid", "R_Parotid", "RParotid", "Parotid Right", "Parotid_Right"),
    "Esophagus": ("Esophagus", "Oesophagus", "Esoph"),
    "Larynx": ("Larynx",),
    "Bone_Mandible": ("Bone_Mandible", "Mandible", "Mandibula", "Bone_Mand"),
    "PTV_7000": ("PTV_7000", "PTV70", "PTV_70", "PTV7000", "PTV70Gy", "PTV_70Gy"),
    "PTV_6300": ("PTV_6300", "PTV63", "PTV_63", "PTV6300", "PTV63Gy", "PTV_63Gy"),
    "PTV_5600": ("PTV_5600", "PTV56", "PTV_56", "PTV5600", "PTV56Gy", "PTV_56Gy"),
}

# Prescription in Gy implied by each canonical target name.
TARGET_PRESCRIPTION_GY: dict[str, float] = {"PTV_7000": 70.0, "PTV_6300": 63.0, "PTV_5600": 56.0}


def _key(name: str) -> str:
    """Matching key: lowercase with separators and whitespace removed."""
    return re.sub(r"[\s_\-\.]+", "", name).lower()


_LOOKUP: dict[str, str] = {}
for _canon, _aliases in ALIASES.items():
    for _a in _aliases:
        _LOOKUP.setdefault(_key(_a), _canon)


@dataclass(frozen=True)
class ResolvedName:
    canonical: str
    raw: str
    known: bool

    @property
    def changed(self) -> bool:
        return self.canonical != self.raw


def resolve(raw_name: str) -> ResolvedName:
    """Resolve a raw structure name to its TG-263 canonical form."""
    canon = _LOOKUP.get(_key(raw_name))
    if canon is None:
        return ResolvedName(canonical=raw_name, raw=raw_name, known=False)
    return ResolvedName(canonical=canon, raw=raw_name, known=True)


def is_target_name(name: str) -> bool:
    return _key(name).startswith("ptv")


def resolve_relative_target(label: str, targets_present: list[str]) -> str | None:
    """Map ``PTV_High`` / ``PTV_Mid`` / ``PTV_Low`` onto the case's canonical targets.

    ``PTV_High`` is the highest prescription present, ``PTV_Low`` the lowest. ``PTV_Mid`` exists
    only when three targets are present. Returns ``None`` when the label cannot be resolved.
    """
    ranked = sorted(
        (t for t in targets_present if t in TARGET_PRESCRIPTION_GY),
        key=lambda t: TARGET_PRESCRIPTION_GY[t],
        reverse=True,
    )
    k = _key(label)
    if not ranked:
        return None
    if k in ("ptvhigh", "ptvhi"):
        return ranked[0]
    if k in ("ptvlow", "ptvlo"):
        return ranked[-1] if len(ranked) > 1 else None
    if k == "ptvmid":
        return ranked[1] if len(ranked) == 3 else None
    return None
