"Track definitions, information visibility, optimization budgets, and episode manifests."

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

TRACK4_TRANSFORMS = ("none", "unit_trap", "nomenclature_drift", "missing_structure", "distractors", "infeasible_goals", "contradictory_instructions", "name_injection")
TRACK5_ARMS = ("tight", "overlap", "coverage_cap")  # the withdrawn "unplannable" arm still builds when named explicitly


class TrackConfig(BaseModel):
    name: str
    k: int = Field(ge=1, description="optimize budget")
    show_score: bool = Field(default=False, description="whether optimize returns the score")
    max_tool_calls: int | None = Field(default=None, description="None means 3 * k + 6, enforced by the session")
    update_after: int = Field(default=0, description="Track 3: the goal list changes once, after this many optimize calls (0 means never)")
    disclose_rules: bool = Field(default=True, description="Whether the case summary and the LLM prompt state the acceptability rules (the two catastrophic gates and the hard/soft/tier meaning); False only for the disclosure experiment")
    description: str = ""

    @property
    def tool_call_cap(self) -> int:
        return self.max_tool_calls if self.max_tool_calls is not None else 3 * self.k + 6


def track_config(name: str, k: int | None = None) -> TrackConfig:
    if name == "T1":
        if k not in (None, 1):
            raise ValueError("Track 1 is single-shot: k must be 1")
        return TrackConfig(name="T1", k=1, description="single-shot: one optimize, then submit")
    if name == "T2":
        kk = 3 if k is None else k
        if kk not in (3, 5):
            raise ValueError("Track 2 uses k in {3, 5}")
        return TrackConfig(name="T2", k=kk, description="iterative: the tweak-optimize-inspect loop")
    if name == "T3":
        kk = 5 if k is None else k
        if kk != 5:
            raise ValueError("Track 3 is k = 5 with one goal update after the second optimize")
        return TrackConfig(name="T3", k=5, update_after=1, description="trade-off: k = 5; after the first optimize one goal changes (a serial-organ limit tightened by 10 percent or a target's coverage relaxed, by seed) and the plan is scored on the final list")
    if name == "T4":
        kk = 3 if k is None else k
        if kk != 3:
            raise ValueError("Track 4 is Track 2 at k = 3 with one transform per episode")
        return TrackConfig(name="T4", k=3, description="adversarial: Track 2 at k = 3 with one presentation transform per episode, plus a clean control")
    if name == "T5":
        kk = 3 if k is None else k
        if kk != 3:
            raise ValueError("Track 5 is k = 3 with an unplannable or a tightened cord limit per episode")
        return TrackConfig(name="T5", k=3, description="escalation: k = 3; the cord limit is either below the feasibility floor (escalate) or tightened to the midpoint between floor and published limit (plan)")
    raise ValueError(f"track {name!r} is not one of T1 to T5")


def without_rules(track: TrackConfig) -> TrackConfig:
    "Return the same track with acceptability rules withheld for a disclosure comparison."
    return track.model_copy(update={"disclose_rules": False})


def with_score(track: TrackConfig) -> TrackConfig:
    "Return the same track with scalar score visibility enabled for the Optuna control."
    return track.model_copy(update={"show_score": True})


class EpisodeSpec(BaseModel):
    episode_id: str
    track: str
    k: int
    case_id: str
    split: str
    seed: int
    transform: str | None = None


def load_split(split_file: Path, split: str) -> list[str]:
    payload = json.loads(Path(split_file).read_text())
    if split not in payload:
        raise KeyError(f"split {split!r} not in {split_file}; available: {[k for k in payload if isinstance(payload[k], list)]}")
    return list(payload[split])


def generate_manifest(track: TrackConfig, case_ids: list[str], split: str, seeds: list[int], transforms: list[str] | None = None, rotate: bool = True) -> list[EpisodeSpec]:
    "Generate a deterministic case-major episode sequence from cases, track, and seeds. Rotated tasks cycle through names; full-factorial tasks enumerate every arm for every case and seed. The transforms argument restricts included tasks."
    out = []
    j = 0
    for cid in case_ids:
        for seed in seeds:
            if track.name in ("T4", "T5"):
                names = transforms or list(TRACK4_TRANSFORMS if track.name == "T4" else TRACK5_ARMS)
                chosen = [names[j % len(names)]] if rotate else names
                j += 1
                for tname in chosen:
                    out.append(EpisodeSpec(episode_id=f"{track.name}-k{track.k}-{cid}-s{seed}-{tname}", track=track.name, k=track.k, case_id=cid, split=split, seed=seed, transform=tname))
                continue
            out.append(
                EpisodeSpec(
                    episode_id=f"{track.name}-k{track.k}-{cid}-s{seed}",
                    track=track.name,
                    k=track.k,
                    case_id=cid,
                    split=split,
                    seed=seed,
                )
            )
    return out


def write_manifest(episodes: list[EpisodeSpec], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([e.model_dump() for e in episodes], indent=1))
    return path
