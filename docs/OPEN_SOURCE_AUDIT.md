# Open-source release notes

Date: 2026-07-12

## Scope

This repository is a **methodology, scripts, and curated communication**
release for offline surrogate-guided search on ConStellaration Problem 2.

Public contents:

- executable Python and shell scripts;
- machine-independent YAML configurations;
- README (English and Chinese), methodology notes, and script catalog;
- dependency and citation metadata;
- MIT license, license-scope statement, and third-party notices;
- Chinese technical report and portable HTML presentation;
- exactly three curated PNG figures.

Excluded from the public tree:

- all `outputs*` directories;
- Hugging Face dataset caches and local Parquet splits;
- VMEC++ wout files;
- model weights and other serialized training artifacts;
- candidate boundaries, audit JSON/JSONL, and row-level result tables;
- private planning materials and interview documents.

## Authorship and citation

- Sole author of original repository content: **Shengnian Liu**
- Citation metadata: [`CITATION.cff`](../CITATION.cff)
- Public repository URL is recorded in `CITATION.cff` (`url` and
  `repository-code`).

## Evaluation language

Official physical feasibility is determined only by the ConStellaration
forward model. Surrogate outputs are reported as predictions; VMEC++ outputs
are reported as audited physics quantities. Expanded Fourier-space experiments
are reported as a separate track from the official low-order parameterization.

## License

Original code, documentation, report, presentation-specific content, and the
three curated figures are released under the MIT License. Upstream
ConStellaration, VMEC++, and HTML PPT Studio retain their original terms. See
[`LICENSE_SCOPE.md`](../LICENSE_SCOPE.md) and
[`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md).

## Known limitations

1. Automated unit tests currently cover science-role path wiring; broader
   physics-label and audit-budget tests remain planned for a later release.
2. Stage directories still share logic through sibling imports. A shared-package
   extraction is deferred.
3. Platform-specific wheels may be required for PyTorch and the pinned
   ConStellaration stack.
4. Public figures and report tables summarize local experiments; row-level
   provenance remains private.

## Public entry points

- English README: [`README.md`](../README.md)
- Chinese README: [`README_zh.md`](../README_zh.md)
- Chinese report: [`presentations/advisor_report/report_cn.md`](../presentations/advisor_report/report_cn.md)
- HTML presentation: [`presentations/advisor_report/advisor_report_deck.html`](../presentations/advisor_report/advisor_report_deck.html)
- Script catalog: [`SCRIPT_CATALOG.md`](SCRIPT_CATALOG.md)
- Layout notes: [`REPOSITORY_LAYOUT.md`](REPOSITORY_LAYOUT.md)
