# Cross-Nfp Pretrain Runbook

Minimal server-side sequence for the cross-Nfp pretrain experiment.

## 1. Put code on the server

If using the bundle:

```bash
cd /path/to/constellaration
tar -xzf /path/to/cross_nfp_pretrain_bundle_YYYYMMDD_HHMMSS.tar.gz
cd experiments/transfer/cross_nfp/pretrain
```

If the directory already exists in the repo:

```bash
cd /path/to/constellaration/experiments/transfer/cross_nfp/pretrain
```

## 2. Smoke test

```bash
PYTHON_BIN=python \
  bash 05_run_smoke_matrix.sh
```

Check:

```bash
python 10_status.py
cat outputs_cross_nfp/run_summary/pretrain_finetune_gate.md
```

The smoke test only validates wiring; do not interpret its scores.

## 3. Full 3-seed run

```bash
PYTHON_BIN=python \
  bash 08_run_multiseed_matrix.sh
```

For unattended execution:

```bash
mkdir -p outputs_cross_nfp/run_logs_multiseed
nohup env PYTHON_BIN=python \
  bash 08_run_multiseed_matrix.sh > outputs_cross_nfp/run_logs_multiseed/nohup.log 2>&1 &
```

## 4. Resume or force rerun

All runners write done markers under `outputs_cross_nfp/run_markers/`.

Resume:

```bash
PYTHON_BIN=python \
  bash 08_run_multiseed_matrix.sh
```

Force all steps:

```bash
FORCE=1 PYTHON_BIN=python \
  bash 08_run_multiseed_matrix.sh
```

Force one step by deleting its marker:

```bash
rm outputs_cross_nfp/run_markers/multiseed/seed1_04_finetune_low_lr_15metric.done
```

## 5. Final evidence

Primary decision file:

```bash
cat outputs_cross_nfp/run_summary/pretrain_finetune_multiseed_summary.md
```

Useful diagnostics:

```bash
python 10_status.py
ls outputs_cross_nfp/run_logs_multiseed
cat outputs_cross_nfp/run_summary/pretrain_finetune_multiseed_summary.json
```

The result is actionable only after `pretrain_finetune_multiseed_summary.md` exists and all expected seed model metrics exist.
