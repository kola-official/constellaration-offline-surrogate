# Wout24 Scheme B-small

This experiment keeps online inputs unchanged:

```text
80 Fourier boundary coefficients -> surrogate -> Problem 2 metrics
```

The additional `vmecpp_wout` information is used only as training supervision.

## Label Set

Regression heads:

- `problem2`: 6 official Problem 2 labels.
- `default_aux`: 9 auxiliary default labels from the 15-label scheme.
- `wout`: 24 low-dimensional VMEC++/wout-derived labels.

The continuous `max_normalized_violation` target remains a separate constraint head.

## Why This Differs From The 15-label MLP

The 15-label model uses one regression head for all labels. This scheme uses a shared encoder with separate heads:

```text
shared encoder
  -> problem2_head
  -> default_aux_head
  -> wout_head
  -> constraint_head
```

The wout loss is label-wise masked, so rows missing `vmecpp_wout` labels still train the Problem 2/default/constraint heads.

## Run Order

On the machine with the filtered wout parquet files:

```bash
python \
  01_build_wout24_labels.py --config configs/quick_wout24.yaml

python \
  02_train_wout24_multitask.py --config configs/quick_wout24.yaml
```

`01_build_wout24_labels.py` reuses an existing
`outputs_wout24/dataset/wout24_pca_models.joblib` when
`wout.reuse_pca_models: true`, and parallelizes the wout transform by parquet
part using `wout.transform_workers`.

## Primary Comparison

Compare against the existing 15-label model on:

- test and optimization-validation `log10_qi` MAE/R2;
- continuous `max_normalized_violation` MAE/R2;
- `wout` head masked validation loss and per-label R2;
- later, audited candidate optimism: predicted vs VMEC++ `log10_qi` and positive max violation.
