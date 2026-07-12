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

## Current layout

Experiment directories are grouped by scientific role:

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

Sibling stages within a track resolve peer directories by relative path.
Cross-track references use repository-root discovery via `requirements.txt`,
`LICENSE`, and `CITATION.cff`.

## Planned package extraction

After broader unit tests land, reusable modules can move into an importable
package:

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

That extraction is deferred so stage-to-stage reproduction paths remain
intact for the current release.
