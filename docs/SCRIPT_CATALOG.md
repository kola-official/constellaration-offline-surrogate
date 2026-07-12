# Script catalog

Scripts use numbered prefixes and verb-object names that match the results
pipeline: prepare → train → candidates → audit → analyze / summarize.

## Official-space track

### Stage 1: base surrogate pipeline

Directory: `experiments/official_space/stage1_base/`

| Script | Responsibility | Main outputs (local only) |
|---|---|---|
| `00_check_environment.py` | Verify Python, GPU, solver, and package availability | Environment record |
| `01_prepare_dataset.py` | Load/filter official data, compute labels, create deterministic splits and relaxed seeds | Parquet splits and manifests |
| `02_train_surrogate.py` | Train the MLP ensemble and write validation/test predictions | Checkpoints, scalers, metrics |
| `03_generate_relaxed_seed_candidates.py` | Fit/sample the relaxed-seed generator | Candidate JSON/JSONL |
| `04_run_conservative_cmaes.py` | Run surrogate-only candidate search and conservative ranking | Ranked candidates |
| `05_vmec_audit_candidates.py` | Evaluate a fixed top-K set with the official forward model | VMEC++ audit records |
| `06_analyze_results.py` | Compare methods and separate predictions from audited values | Local tables and report |
| `08_benchmark_surrogate_3000.py` | Benchmark batched surrogate throughput and search behavior | Benchmark summaries |

### Stage 2: latent feasibility

Directory: `experiments/official_space/stage2_latent/`

| Script | Responsibility |
|---|---|
| `00_link_or_copy_inputs.md` | Documents Stage 1 outputs consumed as inputs |
| `01_run_latent_feasibility_cmaes.py` | PCA/latent-space constrained surrogate search |
| `02_vmec_audit_stage2_candidates.py` | Fixed Stage 2 VMEC++ audit |
| `03_compare_stage1_stage2.py` | Align Stage 1 and Stage 2 under the same interpretation rules |

### Stage 3: surrogate-arbitrage diagnosis

Directory: `experiments/official_space/stage3_trust_region/`

| Script | Responsibility |
|---|---|
| `01_diagnose_surrogate_arbitrage.py` | Join surrogate and audit quantities; measure prediction optimism vs support distance |
| `02_generate_trust_region_candidates.py` | Distance-, uncertainty-, and geometry-constrained candidate pool |
| `03_vmec_audit_stage3_candidates.py` | Fixed Stage 3 audit |
| `04_analyze_stage3.py` | Report whether the trust region improves audited quantities without overstating feasibility |

### Stage 4: surrogate-assisted ALM/NGOpt

Directory: `experiments/official_space/stage4_alm_prescreen/`

| Script | Responsibility |
|---|---|
| `01_run_surrogate_alm_ngopt.py` | Augmented-Lagrangian-style surrogate objective and candidate prescreen |
| `02_vmec_audit_stage4a_candidates.py` | Fixed per-group selections and random prescreen control |
| `03_analyze_stage4a.py` | Audit-efficiency and random-control comparisons |

## Auxiliary-supervision track (wout24)

Directory: `experiments/auxiliary_supervision/wout24/`

| Script | Responsibility |
|---|---|
| `01_build_wout24_labels.py` | Derive 24 low-dimensional auxiliary targets from locally supplied wout Parquet parts |
| `02_train_wout24_multitask.py` | Train a multitask model with the auxiliary labels |
| `03_generate_relaxed_seed_candidates.py` | Reuse the base candidate protocol with the multitask surrogate |
| `04_run_conservative_cmaes.py` | Surrogate search and conservative ranking |
| `05_vmec_audit_candidates.py` | Independent fixed-budget audit |

The wout parts path must be supplied explicitly with `--filtered-parts-dir`.

## Cross-Nfp transfer track

### Pretrain / finetune

Directory: `experiments/transfer/cross_nfp/pretrain/`

| Script | Responsibility |
|---|---|
| `00_check_environment.py` | Environment and dependency check |
| `01_prepare_pretrain_dataset.py` | Create non-target-Nfp pretraining splits |
| `02_train_15metric.py` | Baseline, pretraining, and finetuning modes |
| `03_summarize_matrix.py` | Matrix comparison tables |
| `04_run_full_matrix.sh` | Resumable full single-seed matrix runner |
| `05_run_smoke_matrix.sh` | Wiring smoke test |
| `06_gate_results.py` | Gate checks on transfer metrics |
| `08_run_multiseed_matrix.sh` | Multi-seed matrix runner |
| `09_aggregate_multiseed.py` | Aggregate multi-seed summaries |
| `10_status.py` | Progress / marker status |
| `11_make_server_bundle.sh` | Pack the pretrain tree for server transfer |

### Downstream evaluation

Directories under `experiments/transfer/cross_nfp/downstream/` mirror the
official-space Stage 1–4 pipeline while loading the selected cross-Nfp-finetuned
surrogate:

| Directory | Role |
|---|---|
| `stage1/` | Base candidate search and audit |
| `stage2_latent/` | Latent-feasibility search |
| `stage3_trust_region/` | Trust-region diagnostics |
| `stage4_alm_prescreen/` | ALM/NGOpt prescreen |

Script names match the corresponding official-space stage.

## Utility experiments

- `experiments/hf_config_screening/`: inspect official dataset configurations.
  Generated screening outputs are excluded.
- `experiments/wout_download_estimate/`: estimate and stream wout artifacts
  (`run_full_wout_pipeline.sh`, `run_stream_test.sh`). Downloaded or transformed
  data are excluded.
