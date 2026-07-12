# Offline Surrogate-Guided Search for ConStellaration Problem 2

**Author: Shengnian Liu**

[English](README.md) | [中文](README_zh.md)

This repository presents an offline surrogate-guided workflow for discovering
quasi-isodynamic stellarator boundary candidates in ConStellaration Problem 2.
The release includes methodology, executable scripts, configurations, a Chinese
technical report, an HTML presentation, and three final figures.

## Project overview

ConStellaration Problem 2 optimizes an approximately 80-dimensional Fourier
representation of a stellarator plasma boundary under five groups of physics
and geometry constraints. The project studies candidate discovery from an
offline training subset with zero samples satisfying all strict Problem 2
constraints simultaneously.

The workflow uses boundary Fourier coefficients as model inputs, predicts the
Problem 2 metrics and normalized constraint violations, searches the surrogate
model, and sends a fixed candidate batch to the official VMEC++ forward model
for high-fidelity audit.

The central research outputs are:

- a simulation-free candidate-search inner loop;
- constraint-aware surrogate training and ranking;
- PCA, GMM, ensemble uncertainty, and trust-region diagnostics;
- auxiliary supervision from additional metrics and wout-derived labels;
- cross-Nfp pretraining and target-subset finetuning;
- a quantitative analysis of surrogate extrapolation and constraint floors;
- a sparse-VMEC++ active-learning roadmap evaluated by official score versus
  cumulative physics calls.

Official physical feasibility is determined by the ConStellaration forward
model. Surrogate values are reported as predictions and VMEC++ values are
reported as audited physics quantities. Public leaderboard records establish
the final audited boundary and score; the candidate-generation history and the
number of VMEC++ calls used by each submitter remain outside the public record.

## Report and HTML presentation

- Chinese technical report:
  [`presentations/advisor_report/report_cn.md`](presentations/advisor_report/report_cn.md)
- Interactive HTML presentation:
  [`presentations/advisor_report/advisor_report_deck.html`](presentations/advisor_report/advisor_report_deck.html)

The HTML presentation uses repository-local CSS and JavaScript assets. Open the
HTML file directly in a browser and use the arrow keys to navigate.

### Final figures

![Surrogate validity boundary](figures/final-negative-result/fig1_surrogate_validity_boundary.png)

![Constraint floor](figures/final-negative-result/fig2_constraint_floor_positive_violation.png)

![Model scheme comparison](figures/scheme-comparison/fig3_model_scheme_comparison.png)

## Repository layout

Experiment directories are named by scientific role (track and stage), not by the
machine that ran them:

```text
.
├── experiments/
│   ├── official_space/
│   │   ├── stage1_base/                # base 6/15-metric surrogate pipeline
│   │   ├── stage2_latent/              # latent-feasibility search
│   │   ├── stage3_trust_region/        # surrogate-arbitrage and trust-region diagnostics
│   │   └── stage4_alm_prescreen/       # surrogate-assisted ALM/NGOpt prescreening
│   ├── auxiliary_supervision/
│   │   └── wout24/                     # wout-derived auxiliary supervision
│   ├── transfer/
│   │   └── cross_nfp/
│   │       ├── pretrain/               # cross-Nfp pretraining and finetuning
│   │       └── downstream/             # Stage 1–4 with cross-Nfp surrogates
│   ├── hf_config_screening/            # dataset configuration probes
│   └── wout_download_estimate/         # wout download / stream utilities
├── presentations/advisor_report/       # public report and HTML presentation
├── figures/                            # three curated final figures
├── docs/                               # methodology, script catalog, and layout notes
├── requirements.txt
├── CITATION.cff
└── LICENSE
```

Generated `outputs*` directories remain local through `.gitignore`. Script
responsibilities are listed in [`docs/SCRIPT_CATALOG.md`](docs/SCRIPT_CATALOG.md).
Layout notes and a planned package extraction are described in
[`docs/REPOSITORY_LAYOUT.md`](docs/REPOSITORY_LAYOUT.md).

## Installation

Python 3.10 matches the principal experiment environment and the tested
ConStellaration stack.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The requirements pin the ConStellaration source revision used by the recorded
experiments. GPU systems can install the PyTorch wheel that matches their CUDA
runtime before installing the remaining dependencies.

## Data access

The scripts load ConStellaration from its official Hugging Face repository:

```python
from datasets import load_dataset

dataset = load_dataset(
    "proxima-fusion/constellaration",
    "default",
    split="train",
)
```

Dataset caches, local Parquet splits, VMEC++ wout files, trained checkpoints,
candidate boundaries, and row-level audit records remain in local experiment
storage.

The wout24 track accepts the local wout parts directory explicitly:

```bash
python experiments/auxiliary_supervision/wout24/01_build_wout24_labels.py \
  --filtered-parts-dir /path/to/filtered/wout/parquet/parts
```

## Experiment flow

```text
official dataset
  -> deterministic filtering and split construction
  -> surrogate ensemble training
  -> relaxed-seed construction
  -> surrogate-guided candidate search
  -> fixed-budget VMEC++ audit
  -> prediction-versus-physics diagnostics
```

A typical base run is:

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

Later experiment directories consume the local outputs of earlier stages. Their
input contracts and responsibilities are listed in
[`docs/SCRIPT_CATALOG.md`](docs/SCRIPT_CATALOG.md).

## Methodological tracks

- **Official-space track:** the benchmark low-order Fourier parameterization and
  an aligned VMEC++ audit budget.
- **Latent and trust-region track:** PCA support, neighborhood distances,
  ensemble uncertainty, and spectral geometry checks.
- **Auxiliary-supervision track:** additional default metrics and wout-derived
  low-dimensional training labels with Fourier coefficients as inference input.
- **Cross-Nfp track:** representation pretraining on non-target field periods
  followed by finetuning on the Nfp=3 subset.
- **Sparse-feedback extension:** batched VMEC++ audits, label acquisition, and
  surrogate retraining across active-learning rounds.

## Relation to ConStellaration and the official leaderboard

This repository builds on the official ConStellaration ecosystem rather than
replacing it:

| Resource | Role in this project |
|---|---|
| [Dataset](https://huggingface.co/datasets/proxima-fusion/constellaration) | Offline training and diagnostic source |
| [Code / forward model](https://github.com/proximafusion/constellaration) | Problem definition, scoring, and VMEC++ audit entry points |
| [Design leaderboard](https://huggingface.co/spaces/proxima-fusion/constellaration-bench) | Public ranking of audited Problem 2 boundaries and scores |
| Paper ALM-NGOpt baseline | External online-physics reference for Problem 2 |

How the pieces connect:

- **Official evaluation is authoritative.** Feasibility and score are defined by
  the ConStellaration forward model. Surrogate outputs are predictions only;
  VMEC++ / official evaluator outputs are audited physics quantities.
- **Leaderboard entries establish the final audited boundary and score.** The
  public table does not disclose each submitter’s candidate-generation history
  or the number of VMEC++ calls used during search. This project therefore does
  not reconstruct or rank against private search trajectories.
- **Paper ALM-NGOpt is an online baseline.** It optimizes with repeated physics
  evaluations in the loop and reports a feasible Problem 2 score under a large
  compute budget. This repository studies a different protocol: offline
  surrogate search plus a **fixed local audit budget**. The two settings are not
  equal-budget comparisons.
- **What this project adds.** Quantitative diagnostics of surrogate validity
  boundaries, constraint floors, trust-region collapse, and random-prescreen
  controls under zero official-feasible training labels in the used Nfp=3
  subset. The main product is method-level evidence about when offline
  surrogate ranking is reliable, not a claim of a new top leaderboard entry.

In short: the leaderboard answers “what audited boundary scores best under the
official metric?”; this repository answers “what can a fixed-budget offline
surrogate pipeline learn and fail to learn from the published data alone?”

## Future directions

Priority extensions follow the technical report and keep evaluation rules
unchanged (official score vs. cumulative high-fidelity calls):

1. **Sparse-feedback / hybrid active learning.** Use the surrogate only to
   propose batches; audit with VMEC++; add labels; retrain; plot best official
   score against cumulative physics calls. This is hybrid work, not pure offline
   search.
2. **Feasible-side or near-feasible-side data acquisition.** The current gap is
   coverage of official-feasible or near-feasible labels, not merely larger
   model capacity on the existing infeasible-heavy pool.
3. **Richer scoring with intermediate physics.** Keep Fourier coefficients as
   inference input, but use wout-derived or multi-metric heads as filters for
   VMEC stability / conservative ranking rather than as sole optimization
   objectives.
4. **Cross-domain transfer with careful label alignment.** Cross-Nfp
   pretraining already helps supervised metrics; broader transfer (e.g. related
   QA/QH corpora) must preserve Problem 2 vacuum definitions and official
   scoring.
5. **Boundary representation research.** Near-axis expansions, learned latent
   boundaries, or spectral-band decompositions may reshape the feasible set and
   are intentionally out of scope for this release.
6. **Engineering consolidation.** Shared package extraction, stronger unit
   tests (feature order, constraint normalization, boundary round-trip, audit
   budget accounting), and clean-environment smoke runs.

Details and evidence for completed stages versus open directions are in the
Chinese report
([`presentations/advisor_report/report_cn.md`](presentations/advisor_report/report_cn.md),
Section 7) and [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md).

## Reproducibility rules

1. Record the upstream ConStellaration commit and environment versions.
2. Keep dataset splits deterministic and separate relaxed seeds from training.
3. Keep VMEC++ audit labels outside model selection for each offline run.
4. Count every attempted VMEC++ audit slot, including solver failures.
5. Report surrogate predictions and official physics evaluations in separate
   fields and tables.
6. Compare methods in the same Fourier space under the same audit budget.
7. Report expanded Fourier parameterizations as an independent track.

## Citation

Citation metadata is provided in [`CITATION.cff`](CITATION.cff). Benchmark and
VMEC++ BibTeX entries are available in
[`docs/REFERENCES.md`](docs/REFERENCES.md). Upstream attribution appears in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## License

Copyright © 2026 Shengnian Liu. Original code, documentation, report, HTML
presentation, and curated figures are released under the MIT License. Upstream
software and datasets retain their original terms. See
[`LICENSE_SCOPE.md`](LICENSE_SCOPE.md).
