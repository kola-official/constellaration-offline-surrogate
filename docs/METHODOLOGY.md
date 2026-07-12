# Methodology

## 1. Problem definition

The public code targets ConStellaration Problem 2 in the benchmark's official
low-order boundary representation. The design vector is formed from the free
stellarator-symmetric Fourier coefficients of `r_cos` and `z_sin`, with the
major-radius coefficient fixed by the benchmark definition.

The objective is evaluated only for candidates satisfying all official
constraints. Consequently, candidate ranking is a constrained discovery problem,
not ordinary regression on the objective alone.

## 2. Offline protocol

A run is considered **offline** only when VMEC++ and the official forward model
are absent from:

- surrogate training;
- hyperparameter and checkpoint selection;
- candidate generation;
- CMA-ES/NGOpt objective evaluations;
- penalty-weight tuning;
- candidate reranking.

VMEC++ may be called afterward under a predetermined audit budget. Audit outputs
must not flow back into the same offline run.

## 3. Data protocol

The base scripts:

1. load the official dataset through Hugging Face;
2. retain samples with the required boundary and physics fields;
3. select the target field-period subset;
4. compute the benchmark-aligned target transformations and normalized
   constraint violations;
5. construct deterministic train, validation, test, and optimization-validation
   splits;
6. keep relaxed near-feasible seeds outside all supervised splits.

No local split or source row is distributed in this repository.

## 4. Surrogate model

The core model is an ensemble of multilayer perceptrons. Its inference input is
always the Fourier boundary vector. Depending on the track, the supervised heads
include:

- the six metrics required for Problem 2 scoring;
- a continuous maximum normalized constraint-violation target;
- additional default-dataset physics metrics;
- optional low-dimensional targets derived from wout files.

Wout-derived values are auxiliary training labels, not inference-time inputs.
Using a wout field as an inference input would require running VMEC++ first and
would invalidate the intended acceleration mechanism.

## 5. Candidate generation

The scripts implement several candidate sources and ablations:

- dataset or relaxed-seed candidates;
- a GMM fitted near relaxed seeds;
- direct CMA-ES search of the surrogate objective;
- conservative constraint-aware ranking using ensemble uncertainty;
- PCA latent-space search;
- distance- and uncertainty-based trust regions;
- surrogate-assisted augmented-Lagrangian/NGOpt prescreening;
- cross-Nfp representation pretraining followed by target-subset finetuning.

All candidate search scores are surrogate quantities until audited.

## 6. Official audit

The audit stage reconstructs each boundary, calls the official forward model,
computes the objective and all official constraints, and assigns a nonzero score
only if every constraint is satisfied. Failed solver calls count against the
attempted audit budget.

The primary comparison metric for future mixed offline/online work should be:

```text
best official score versus cumulative VMEC++ calls
```

For runs with no feasible candidate, report the best positive maximum normalized
constraint violation and its component breakdown, without calling it an official
score.

## 7. Interpretation rules

- A high surrogate score is not evidence of physical feasibility.
- A lower audited violation is not evidence of superiority unless the search
  space and VMEC++ budget are matched.
- Additional Fourier modes define an expanded-space track and must not be mixed
  with official-space baseline comparisons.
- Ensemble variance and nearest-neighbor distance are diagnostics, not formal
  guarantees of in-distribution behavior.
- Negative results should identify the evaluated model class, data support, and
  audit budget. The interpretation remains scoped to those conditions.

## 8. Recommended extension

The most informative next experiment is sparse-feedback active learning:

1. train an ensemble on the offline data;
2. generate a large candidate pool without VMEC++;
3. select a small batch by conservative feasibility, objective, diversity, and
   uncertainty;
4. audit the batch with VMEC++;
5. add the new labels and retrain;
6. repeat while plotting best official score against cumulative calls.

This extension is hybrid or active-learning work.
