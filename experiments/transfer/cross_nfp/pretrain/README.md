# Cross-Nfp Pretrain Then Finetune

This directory implements the two-stage transfer test:

```text
90k non-Nfp3 rows -> 15-label pretrain -> 68k Nfp=3 finetune
```

For server execution details, see `RUNBOOK.md`.

The implementation keeps the two scaler corrections in the protocol:

- `Nfp` is appended as a fixed condition (`nfp / 3.0`) and is not included in the ordinary geometry scaler.
- Geometry `x_*` features always use the Nfp=3 train split scaler, including during non-Nfp3 pretraining.
- Target scalers are stage-local: pretraining targets use the pretrain split scaler; finetuning and baseline targets use the 68k Nfp=3 train split scaler.

The 15 labels are the verified default set:

```text
L_gradB
aspect_ratio
abs_edge_iota_over_nfp
log10_qi
edge_magnetic_mirror_ratio
max_elongation
aspect_ratio_over_edge_rotational_transform
average_triangularity
axis_magnetic_mirror_ratio
axis_rotational_transform_over_n_field_periods
edge_rotational_transform_over_n_field_periods
flux_compression_in_regions_of_bad_curvature
minimum_normalized_magnetic_gradient_scale_length
qi
vacuum_well
```

The continuous `max_normalized_violation` head is retained to match the existing default surrogate training protocol.

## Run order

First run a small smoke test:

```bash
PYTHON_BIN=python \
  bash 05_run_smoke_matrix.sh
```

The smoke gate only verifies the data, scaler, checkpoint loading, and metric-writing chain; do not interpret its scores as transfer evidence.

Then run the full matrix:

```bash
PYTHON_BIN=python \
  bash 04_run_full_matrix.sh
```

For the seed-variance check, run the 3-seed matrix:

```bash
PYTHON_BIN=python \
  bash 08_run_multiseed_matrix.sh
```

All runners are resumable. A completed step writes a marker under `outputs_cross_nfp/run_markers/`; rerunning the same script skips marked steps. Set `FORCE=1` to rerun from scratch or after deleting selected marker files.

To inspect progress:

```bash
python 10_status.py
```

For unattended runs:

```bash
mkdir -p outputs_cross_nfp/run_logs
nohup env PYTHON_BIN=python \
  bash 04_run_full_matrix.sh > outputs_cross_nfp/run_logs/full_matrix.nohup.log 2>&1 &
```

The matrix runs:

- `baseline_random_15metric_nfp`: random init on Nfp=3, with the constant Nfp condition.
- `pretrain_90k_15metric`: non-Nfp3 pretraining with 15 labels.
- `finetune_low_lr_15metric`: pretrained backbone, Nfp=3 finetune at `3e-4`.
- `finetune_default_lr_15metric`: pretrained backbone, Nfp=3 finetune at `1e-3`.

By default finetuning loads only the backbone. To also load the regression and constraint heads, pass `--load-heads` to `02_train_15metric.py` for an additional diagnostic run.

## Outputs

Primary outputs are written under:

```text
experiments/transfer/cross_nfp/pretrain/outputs_cross_nfp/
```

Key files:

- `dataset_pretrain/pretrain_manifest.json`
- `models/<run_name>/metrics.json`
- `models/<run_name>/predictions_test.parquet`
- `models/<run_name>/predictions_optimization_validation.parquet`
- `run_summary/pretrain_finetune_comparison.md`
- `run_summary/pretrain_finetune_gate.md`
- `run_summary/pretrain_finetune_multiseed_summary.md`
- `run_logs/*.log`
- `run_markers/**/*.done`
