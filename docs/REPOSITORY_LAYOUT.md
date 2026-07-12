# Repository layout

## Release contents

The public release contains:

- executable Python and shell scripts;
- YAML configurations with machine-independent defaults;
- README and methodology documentation;
- dependency, citation, license, and third-party notice files;
- the Chinese technical report, HTML presentation assets, and three final figures.

It does not contain data, checkpoints, candidates, audits, result tables,
server logs, environment dumps, PID/marker files, or bundles.

## Current layout (science-role names)

Experiment directories are grouped by scientific role rather than by the machine
that ran them:

```text
experiments/
  official_space/
    stage1_base/              # base 6/15-metric surrogate pipeline
    stage2_latent/            # PCA / latent-feasibility search
    stage3_trust_region/      # surrogate-arbitrage and trust-region diagnostics
    stage4_alm_prescreen/     # surrogate-assisted ALM/NGOpt prescreening
  auxiliary_supervision/
    wout24/                   # wout-derived auxiliary multitask supervision
  transfer/
    cross_nfp/
      pretrain/               # non-target-Nfp pretraining and finetuning
      downstream/
        stage1/               # Stage 1 pipeline with cross-Nfp surrogate
        stage2_latent/
        stage3_trust_region/
        stage4_alm_prescreen/
  hf_config_screening/        # official dataset configuration probes
  wout_download_estimate/     # wout download / stream utilities
```

Sibling stages within a track resolve peer directories by relative path. Cross-
track references (for example, pretrain → official-space Stage 1) use repository
root discovery via `requirements.txt`, `LICENSE`, and `CITATION.cff` markers.

## Historical name map (retired)

Machine-prefixed names used during development were renamed for the public
release. They are not live entry points:

| Retired directory | Current directory |
|---|---|
| `rtx3090_simple_qi` | `experiments/official_space/stage1_base` |
| `rtx3090_simple_qi_stage2` | `experiments/official_space/stage2_latent` |
| `rtx3090_simple_qi_stage3` | `experiments/official_space/stage3_trust_region` |
| `rtx3090_simple_qi_stage4` | `experiments/official_space/stage4_alm_prescreen` |
| `rtx3090_simple_qi_wout24` | `experiments/auxiliary_supervision/wout24` |
| `rtx3090_cross_nfp_pretrain` | `experiments/transfer/cross_nfp/pretrain` |
| `rtx3090_cross_nfp_downstream` | `experiments/transfer/cross_nfp/downstream/stage1` |
| `rtx3090_cross_nfp_downstream_stage2` | `experiments/transfer/cross_nfp/downstream/stage2_latent` |
| `rtx3090_cross_nfp_downstream_stage3` | `experiments/transfer/cross_nfp/downstream/stage3_trust_region` |
| `rtx3090_cross_nfp_downstream_stage4` | `experiments/transfer/cross_nfp/downstream/stage4_alm_prescreen` |

## Planned package extraction (post-v0.1)

After unit tests land, reusable modules can move into an importable package:

```text
src/constellaration_offline_search/
  data.py
  labels.py
  surrogate.py
  candidates.py
  audit.py

configs/
docs/
tests/
```

That extraction is intentionally deferred so stage-to-stage reproduction paths
stay intact for the first public release.
