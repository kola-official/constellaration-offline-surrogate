# Offline Surrogate-Guided Search for ConStellaration Problem 2

**Author: Shengnian Liu**

[English](README.md) | [中文](README_zh.md)

This repository studies **offline surrogate-assisted optimization** for
ConStellaration Problem 2: discovering simple-to-build quasi-isodynamic (QI)
stellarator boundary candidates when high-fidelity VMEC++ evaluations are
expensive. The public release includes methodology, executable experiment
pipelines, configurations, a Chinese technical report, an interactive HTML
presentation, and three final figures.

## Work summary

### Problem

ConStellaration Problem 2 is an approximately **80-dimensional constrained black-box**
optimization task. The design variables are stellarator-symmetric Fourier
boundary coefficients (`Nfp=3`, low-order modes). The objective is to maximize
coil manufacturability (`L_gradB`). Official feasibility requires five constraint
groups to hold simultaneously (aspect ratio, rotational transform, QI residual
`log10(qi)`, magnetic mirror ratio, elongation). The official score is
`L_gradB / 20` only if all constraints pass; otherwise the score is **zero**.

VMEC++ equilibrium solves are too costly to place inside a large free search
loop. The official Hugging Face dataset provides a large QI-like corpus, but the
**error-free Nfp=3 subset used here (68,191 rows) contains zero samples that meet
all strict Problem 2 constraints at once**. Feasibility is therefore sparse in
the training support. Any “feasibility” signal learned by a model is at best
an interpolation or extrapolation of continuous violations. The model has not
seen true feasible equilibria in this subset.

### Goal of this project

Build a full **offline** loop:

```text
official data
  -> deterministic filtering and splits
  -> multi-task surrogate ensemble
  -> surrogate-only candidate search
  -> fixed-budget VMEC++ / official audit
  -> prediction-versus-physics diagnostics
```

Hard protocol: VMEC++ and the official forward model are **absent** from
training, model selection, search objectives, and reranking within one offline
run. Physics is used only as a predetermined post-search audit.

### What was built

| Layer | Content |
|---|---|
| **Core model** | Spectral multi-task deep ensemble MLP on Fourier coefficients |
| **Search tools** | CMA-ES / NGOpt on surrogate scores; PCA–GMM near relaxed seeds; trust regions; surrogate ALM-style prescreen |
| **Mainline stages** | Stage 1 E0–E3 → Stage 2 latent → Stage 3 trust-region → Stage 4A ALM prescreen + random control |
| **Model upgrades** | Scheme A (15 metrics), B-small (wout24 auxiliary heads), C (cross-Nfp pretrain → Nfp=3 finetune) |
| **Diagnostics** | Distance–bias curves, constraint floors, VMEC success rates, random-prescreen percentiles |
| **Communication** | Chinese report, HTML deck, three curated figures |

### Main conclusions (scoped)

Under the stated offline protocol and the used Nfp=3 subset:

1. **In-distribution learning works.** Surrogates fit Problem 2 metrics and
   continuous violations well on held-out splits drawn from the same pool.
2. **Optimization pressure exposes two failure modes.** Unconstrained surrogate
   search tends to **exploit optimistic model error** (high predicted score, poor
   audited physics). Hard trust-region limits tend to **collapse back to the
   database neighborhood** without crossing the official feasible line.
3. **Under matched budgets in Stage 4A, surrogate-assisted ALM-style prescreen
   stays inside the random audit-control distribution.** Candidate-pool
   construction may help slightly. Ranking credit alone is hard to attribute to
   the surrogate.
4. **Representation upgrades help supervision, not automatic feasibility.**
   15-metric multitask, wout intermediate heads, and cross-Nfp pretraining improve
   prediction metrics (and sometimes VMEC runnability), but audited runs remain
   at **zero official feasible candidates** in the reported protocols.
5. **Primary product is diagnostic evidence** about offline surrogate reliability.
   Crossing feasibility likely needs feasible-side or near-feasible-side
   high-fidelity labels (hybrid active learning).

Detailed numbers, ablations, and wording boundaries are in the Chinese report.

## Methods overview

### Offline protocol

A run is offline only when VMEC++ is not used for training, hyperparameter
selection, CMA-ES/NGOpt objectives, penalty tuning, or reranking. After search,
a **fixed attempted audit budget** evaluates selected candidates; failures count
in the denominator. Audit labels must not re-enter the same offline run.

### Surrogate

- **Input:** free Fourier boundary coefficients in a fixed `(m,n)` order,
  standardized on the training split.
- **Heads (depending on track):** six Problem 2 physics metrics; continuous
  `max_normalized_violation`; optional extra default metrics (15-label scheme);
  optional 24-D wout-derived auxiliary labels (training only).
- **Architecture:** residual MLP ensemble (4 members), multitask regression,
  ensemble disagreement as uncertainty.
- **Why continuous violations:** with zero official-feasible positives, binary
  feasibility classification collapses. Continuous violation regression supplies
  a usable ranking signal from available labels.
- **Wout rule:** wout-derived quantities are **training labels only**. Using them
  at inference would require VMEC++ first and remove the intended speedup.

### Candidate generation (Stage 1 methods E0–E3)

| ID | Idea |
|---|---|
| **E0** | Static database ranking (no model search) |
| **E1** | Local generation near **relaxed55** seeds (PCA + GMM + surrogate rank) |
| **E2** | Surrogate-only CMA-ES / high-objective search |
| **E3** | Conservative search: objective + violation, uncertainty, and support penalties |

**Relaxed55** seeds are near-feasible samples under **paper-style relaxed
thresholds**. They are **not** official feasible points and are held out of all
supervised train/val/test splits; they serve only as generation priors.

### Stage progression

| Stage | Question |
|---|---|
| **1** | Can surrogates and basic search produce audited candidates under a fixed top-K VMEC budget? |
| **2** | Does latent / geometry-aware search improve VMEC success versus blind surrogate search? |
| **3** | Is the bottleneck geometry, surrogate optimism, or data support? Trust-region diagnostics. |
| **4A** | If surrogate ALM/NGOpt-style search is used as a prescreener, does it beat random draw from the same pool? |

### Model schemes beyond the 6-metric mainline

| Scheme | Change | Role |
|---|---|---|
| **A – 15 metrics** | Multitask on extra default physics labels; scoring still uses Problem 2 metrics | Stronger supervised representation |
| **B-small – wout24** | 24-D intermediate physics heads from equilibrium wout parts | Auxiliary supervision only |
| **C – cross-Nfp** | Pretrain on non-Nfp=3 rows, finetune on Nfp=3 | Transfer initialization |

### Optimizers and diagnostics (short)

- **CMA-ES / NGOpt:** derivative-free continuous optimizers querying the surrogate.
- **PCA / GMM:** keep candidates on data-supported geometry near relaxed seeds.
- **Trust region:** ensemble uncertainty + train/seed distances + spectral checks.
- **Surrogate arbitrage:** optimizers mining systematic optimism at the edge of
  training support. This project measures that behavior with audit diagnostics.

## Report, presentation, and figures

- Chinese technical report:
  [`presentations/advisor_report/report_cn.md`](presentations/advisor_report/report_cn.md)
- Interactive HTML presentation:
  [`presentations/advisor_report/advisor_report_deck.html`](presentations/advisor_report/advisor_report_deck.html)

Open the HTML file in a browser and use arrow keys to navigate.

### Final figures

![Surrogate validity boundary](figures/final-negative-result/fig1_surrogate_validity_boundary.png)

![Constraint floor](figures/final-negative-result/fig2_constraint_floor_positive_violation.png)

![Model scheme comparison](figures/scheme-comparison/fig3_model_scheme_comparison.png)

## Relation to ConStellaration and the official leaderboard

This repository builds on the official ConStellaration ecosystem.

| Resource | Role here |
|---|---|
| [Dataset](https://huggingface.co/datasets/proxima-fusion/constellaration) | Offline training and diagnostics (published low-order QI-like boundaries) |
| [Code / forward model](https://github.com/proximafusion/constellaration) | Problem definition, scoring, audit APIs |
| [Design leaderboard](https://huggingface.co/spaces/proxima-fusion/constellaration-bench) | Public ranking of audited final boundaries and scores |
| [Public result files](https://huggingface.co/datasets/proxima-fusion/constellaration-bench-results) | Stored `boundary_json` for submitted surfaces |
| Paper ALM-NGOpt baseline | Online physics optimization reference in the paper’s low-order setting |

### Fourier dimension: two evaluation tracks

The official dataset and the paper’s Problem 2 baseline use a **low-order**
stellarator-symmetric Fourier boundary with poloidal/toroidal mode cutoffs
`m,n ≤ 4`. Free design dimension is about **80**. This repository’s mainline
surrogate training, E0–E3 search, Stages 2–4A, and schemes A/B/C all stay on
that **official-space** support.

Public leaderboard result files for `simple_to_build` (Problem 2) show that many
**high-score** final boundaries use **expanded** Fourier arrays:

| Observed `r_cos` / `z_sin` shape | Mode cutoffs (approx.) | Role on the public board |
|---|---|---|
| `(5, 9)` | `m,n ≤ 4` | Same low-order class as the published dataset and this repo’s mainline |
| `(8, 15)` | `m,n ≤ 7` | Expanded-space submissions; several strong scores |
| `(11, 21)` | `m,n ≤ 10` | Further expanded-space submissions among top scores |

Measured from public rows in
`proxima-fusion/constellaration-bench-results` (July 2026 snapshot). Example:
top `simple_to_build` scores in that sample sit on expanded shapes such as
`(8, 15)` or `(11, 21)`, while feasible low-order `(5, 9)` scores also appear at
lower ranks. Expanded modes are accepted by the official evaluator when the
submitted boundary is a valid Fourier surface; they define a **different design
space** from the published 80-D training corpus.

### How this project sits relative to the board

- Official evaluation remains the score authority. Surrogate outputs are
  predictions; VMEC++ / forward-model outputs are audited physics quantities.
- Leaderboard rows publish the **final audited boundary and score**. Search
  history and VMEC++ call counts are not published with those rows.
- Paper ALM-NGOpt reports a feasible Problem 2 score under **online** physics
  optimization in the paper’s low-order Fourier setting, with a large compute
  budget.
- This repository studies **fixed-budget offline surrogate search on the
  published low-order Nfp=3 data**. Its diagnostics (validity boundary,
  constraint floor, random-prescreen controls) apply to that track.
- Expanded-mode leaderboard practice is an **independent track**: higher-mode
  final boundaries are not mixed into this repo’s official-space method
  comparisons. Matching or ranking against expanded-space board scores requires
  an expanded-space experiment design of its own.

**Leaderboard question:** among submitted (and audited) boundaries, which score
is highest under the official metric?  
**This repository’s question:** on the published low-order data alone, with a
fixed offline audit budget, what can a surrogate pipeline learn, and where does
it fail?

## Repository layout

```text
.
├── experiments/
│   ├── official_space/
│   │   ├── stage1_base/                # base 6/15-metric pipeline (E0–E3)
│   │   ├── stage2_latent/              # latent-feasibility search
│   │   ├── stage3_trust_region/        # arbitrage / trust-region diagnostics
│   │   └── stage4_alm_prescreen/       # surrogate ALM/NGOpt prescreen + control
│   ├── auxiliary_supervision/wout24/   # scheme B-small
│   ├── transfer/cross_nfp/
│   │   ├── pretrain/                   # scheme C pretrain / finetune
│   │   └── downstream/                 # stages 1–4 with cross-Nfp surrogate
│   ├── hf_config_screening/
│   └── wout_download_estimate/
├── presentations/advisor_report/
├── figures/
├── docs/                               # methodology, catalog, references
├── tests/
├── requirements.txt
├── CITATION.cff
└── LICENSE
```

Generated `outputs*` stay local via `.gitignore`. Script roles:
[`docs/SCRIPT_CATALOG.md`](docs/SCRIPT_CATALOG.md). Layout notes:
[`docs/REPOSITORY_LAYOUT.md`](docs/REPOSITORY_LAYOUT.md). Methodology detail:
[`docs/METHODOLOGY.md`](docs/METHODOLOGY.md).

## Installation

Python 3.10 matches the principal experiment environment.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Requirements pin the ConStellaration revision used in recorded runs. Install a
CUDA-matched PyTorch wheel before other dependencies when using GPU.

## Data access

```python
from datasets import load_dataset

dataset = load_dataset(
    "proxima-fusion/constellaration",
    "default",
    split="train",
)
```

Caches, Parquet splits, wout files, checkpoints, candidates, and row-level
audits remain local and are not published in this repository.

Wout24 requires an explicit local parts directory:

```bash
python experiments/auxiliary_supervision/wout24/01_build_wout24_labels.py \
  --filtered-parts-dir /path/to/filtered/wout/parquet/parts
```

## How to run (base pipeline)

```bash
cd experiments/official_space/stage1_base
python 00_check_environment.py --config configs/quick.yaml
python 01_prepare_dataset.py --config configs/quick.yaml
python 02_train_surrogate.py --config configs/quick.yaml
python 03_generate_relaxed_seed_candidates.py --config configs/quick.yaml
python 04_run_conservative_cmaes.py --config configs/quick.yaml
python 05_vmec_audit_candidates.py --config configs/quick.yaml
python 06_analyze_results.py --config configs/quick.yaml
```

Later stages consume earlier local outputs. Full stage/script map:
[`docs/SCRIPT_CATALOG.md`](docs/SCRIPT_CATALOG.md).

## Methodological tracks (directory map)

- **Official-space:** low-order Fourier benchmark space and matched audit budget.
- **Latent / trust-region:** PCA support, distances, uncertainty, spectral checks.
- **Auxiliary supervision:** extra metrics and wout-derived training labels.
- **Cross-Nfp transfer:** pretrain on non-target periods, finetune on Nfp=3.
- **Sparse-feedback extension (future hybrid):** batched VMEC audits + retrain.

## Future directions

1. Hybrid active learning: propose with the surrogate, audit with VMEC++, retrain;
   plot best official score vs. cumulative physics calls.
2. Acquire feasible-side or near-feasible-side high-fidelity labels.
3. Use intermediate physics as ranking **filters**, with Fourier coefficients still
   as inference input and Problem 2 metrics for scoring.
4. Broader transfer with aligned Problem 2 vacuum definitions and scoring.
5. Boundary representation research (near-axis, learned latents, spectral bands)
   is left for separate work beyond this release.
6. Engineering: shared package, unit tests (feature order, constraint
   normalization, boundary round-trip, audit budget), clean-environment smoke runs.

See report Section 7 and [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md).

## Reproducibility rules

1. Record the upstream ConStellaration commit and environment versions.
2. Keep splits deterministic; keep relaxed seeds out of supervised training.
3. Keep VMEC++ audit labels outside model selection for each offline run.
4. Count every attempted VMEC++ audit slot, including solver failures.
5. Report surrogate predictions and official physics in separate fields.
6. Compare methods in the same Fourier space under the same audit budget.
7. Report expanded Fourier parameterizations as an independent track.

## Citation

See [`CITATION.cff`](CITATION.cff). Benchmark and VMEC++ BibTeX:
[`docs/REFERENCES.md`](docs/REFERENCES.md). Upstream notices:
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## License

Copyright © 2026 Shengnian Liu. Original code, documentation, report, HTML
presentation, and curated figures are under the MIT License. Upstream software
and datasets retain their own terms. See [`LICENSE_SCOPE.md`](LICENSE_SCOPE.md).
