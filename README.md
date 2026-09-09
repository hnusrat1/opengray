# OpenGray

**An open research environment for evaluating AI agents in radiotherapy treatment planning.**

[![Tests](https://github.com/hnusrat1/opengray/actions/workflows/ci.yml/badge.svg)](https://github.com/hnusrat1/opengray/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

An agent can produce a plausible treatment plan while misreading a goal, overlooking missing information, or failing to recognize contradictory instructions. OpenGray makes these behaviors testable alongside the dose distribution the agent produces.

The environment gives scripted and language-model agents a common set of planning tools and a bounded optimization budget. Agents can inspect a case, set objectives, optimize, review dose metrics, revise a plan, and either submit it or escalate a problem. Dose is computed from a supplied sparse dose-influence matrix. The environment records tool calls, final fluence, protocol settings, and separate dosimetric and behavioral outcomes.

OpenGray is under active development. It is research software for controlled experiments, not a clinical treatment-planning system. Independent physicist review of the development experiments is pending.

## What is implemented

- A shared planning-session engine with typed tool requests and responses, budget enforcement, structured errors, and event logs.
- Sparse dose calculation, dose-volume metrics, explicit goal scoring, and a projected-gradient fluence optimizer.
- Scripted heuristic and controller baselines, random and Optuna search, preflight checks, a language-model interpreter followed by a controller, and full tool-using language-model agents.
- Five evaluation tracks covering one-shot planning, iterative planning, changed goals, altered task presentation, and constructed contradictions with feasible controls.
- Protocol fingerprints, saved submitted fluence, campaign manifests, case-weighted comparisons, and sensitivity-analysis utilities.
- An OpenKBP-Opt importer and an MCP interface for Tracks T1–T3.

The batch runner supports all five tracks. The current release does not include a Gymnasium interface, a trained planning policy, or a hosted clinical service.

## Try it without patient data or an API key

Python 3.11 or later is required. From a clone of this repository:

```bash
git clone https://github.com/hnusrat1/opengray.git
cd opengray
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python examples/synthetic_demo.py
python -m pytest -m 'not data'
```

On Windows, activate the environment with `.venv\Scripts\activate`.

The example constructs a small fictional dose-influence matrix, runs a scripted planning episode, and prints its outcome, goal results, and optimization use. It is a software demonstration, not a patient result or a clinical plan-quality test. The automated tests use synthetic fixtures and mocked model responses; they do not call a paid model API.

## Use OpenKBP-Opt data

Obtain the original archives from the [OpenKBP-Opt authors](https://github.com/ababier/open-kbp-opt). This repository contains no patient-data archives or cached dose matrices. See [data access and attribution](data/LICENSES.md).

```bash
opengray download
opengray ingest /path/to/core-data.zip --cache ./data/cache
opengray validate --cache ./data/cache
opengray run --track T2 --k 3 --agent controller --cases pt_289 \
  --seeds 1 --cache ./data/cache --out ./runs/controller-demo
```

The last command uses an explicitly named source case and a scripted controller. Language-model agents require provider credentials and can incur API costs. They are optional; inspect `opengray run --help` before using them. Keep credentials in environment variables or an ignored local key file.

## Research questions and current evidence

The development experiments examine how measured performance changes when rejection rules are disclosed, anatomical information is available, scripted checks are added, or scoring and solver assumptions change. These experiments use ten development cases from one head-and-neck resource. They motivate evaluation-method questions rather than a general model ranking.

The study is in preparation. Detailed study results and independent review findings are not part of this initial software release. The implemented evaluation methods can be inspected in the code and exercised with locally obtained data. This repository alone does not reproduce the complete study's original trajectories.

Several limits matter when interpreting an experiment:

- Repeated runs are not additional patients. Match agents on the same patient and task before aggregating differences.
- PlanScore is a research scoring definition. A higher score does not establish a better clinical treatment plan.
- Logged escalation is distinct from correctly recognizing and explaining a problem. Behavioral grades require independent review.
- The default optimizer does not enforce clinical machine-delivery constraints. The optional `--deliverable` flag is a historical name for a fluence-complexity experiment; final limits can be violated, including after normalization.
- Clinical deliverability, prospective workflow benefit, external validation, and patient outcomes have not been demonstrated.

See the [technical overview](docs/architecture.md) for the tool flow, tracks, scoring, and evidence boundaries.

## Code map

| Directory | Purpose |
|---|---|
| [`src/opengray/env`](src/opengray/env) | Planning sessions, tools, tasks, and MCP |
| [`src/opengray/agents`](src/opengray/agents) | Scripted and language-model agents |
| [`src/opengray/physics`](src/opengray/physics) | Dose, objectives, optimization, and certificates |
| [`src/opengray/goals`](src/opengray/goals) | Goal definitions and scoring |
| [`src/opengray/runner`](src/opengray/runner) | Execution, provenance, aggregation, and audit utilities |
| [`tests`](tests) | Synthetic numerical, contract, and regression tests |

## Attribution and contributions

OpenGray is developed by Humza Nusrat. The software is licensed under [Apache 2.0](LICENSE). Upstream data and software retain their own terms; see [third-party notices](THIRD_PARTY_NOTICES.md).

Please cite OpenKBP and OpenKBP-Opt when using those data:

1. Babier A, et al. OpenKBP: The open-access knowledge-based planning grand challenge and dataset. *Medical Physics*. 2021;48:5549–5561. [DOI](https://doi.org/10.1002/mp.14845).
2. Babier A, et al. OpenKBP-Opt: an international and reproducible evaluation of 76 knowledge-based planning pipelines. *Physics in Medicine & Biology*. 2022;67:185012. [DOI](https://doi.org/10.1088/1361-6560/ac8044).

Issues and focused pull requests are welcome. Include a synthetic reproducer when possible. Do not upload patient information, credentials, or restricted datasets.
