"""Compare OpenGray's metric implementation with the OpenKBP-Opt authors' own numbers.

``experiments-data.zip`` ships ``results-data/reference_metrics.csv`` (per-structure D_0.1_cc,
mean, D_99, D_95, D_1 of every reference plan, disjoint PTV masks) and
``results-data/reference_criteria.csv`` (the same metrics evaluated for the clinical criteria,
where each PTV is merged with all higher-dose PTVs). Reproducing both tables to numerical
precision is the M2 evidence that the DVH code matches the reference implementation.
"""

from __future__ import annotations

import csv
import io
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from opengray.data.base import CohortLoader
from opengray.physics.dvh import evaluate
from opengray.physics.nomenclature import resolve

METRIC_MAP = {"D_0.1_cc": "D0.1cc", "mean": "Dmean", "D_99": "D99", "D_95": "D95", "D_1": "D1"}
RESULTS_PREFIX = "open-kbp-opt-data/results-data/"


def read_reference_table(zip_path: Path, name: str) -> list[dict[str, Any]]:
    """Rows of ``results-data/<name>`` as dicts with case, metric, structure, value (None if blank)."""
    with zipfile.ZipFile(zip_path) as z:
        text = z.read(f"{RESULTS_PREFIX}{name}").decode()
    rows = []
    for rec in csv.reader(io.StringIO(text)):
        if len(rec) < 5 or rec[0] != "Reference":
            continue
        val = rec[4].strip()
        rows.append({"case": rec[1], "metric": rec[2], "structure": rec[3], "value": float(val) if val else None})
    return rows


def compare_reference_metrics(loader: CohortLoader, zip_path: Path, merged: bool, case_ids: list[str] | None = None) -> dict[str, Any]:
    name = "reference_criteria.csv" if merged else "reference_metrics.csv"
    rows = read_reference_table(zip_path, name)
    wanted = set(case_ids) if case_ids else None
    diffs: dict[str, list[float]] = {}
    n_compared = n_missing = n_blank = 0
    worst: list[tuple[float, str, str, str, float, float]] = []
    cases_seen: set[str] = set()
    by_case: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        if wanted is not None and r["case"] not in wanted:
            continue
        if r["value"] is None:
            n_blank += 1
            continue
        by_case.setdefault(r["case"], []).append(r)
    # One case in memory at a time: the cohort does not fit in a small machine's RAM.
    for case_id, case_rows in by_case.items():
        case = loader.load(case_id)
        cases_seen.add(case_id)
        for r in case_rows:
            canon = resolve(r["structure"]).canonical
            metric = METRIC_MAP.get(r["metric"])
            if metric is None or canon not in case.structures or case.reference_dose is None:
                n_missing += 1
                continue
            ours = evaluate(case, case.reference_dose, canon, metric, merge_targets=merged)
            d = abs(ours - r["value"])
            diffs.setdefault(metric, []).append(d)
            n_compared += 1
            worst.append((d, case_id, canon, metric, r["value"], ours))
        del case
    worst.sort(reverse=True)
    summary = {m: {"n": len(v), "max_abs_diff": float(np.max(v)), "mean_abs_diff": float(np.mean(v))} for m, v in diffs.items()}
    return {
        "table": name,
        "merged": merged,
        "n_cases": len(cases_seen),
        "n_compared": n_compared,
        "n_missing": n_missing,
        "n_blank": n_blank,
        "per_metric": summary,
        "worst": [{"diff": d, "case": c, "structure": s, "metric": m, "theirs": t, "ours": o} for d, c, s, m, t, o in worst[:10]],
    }
