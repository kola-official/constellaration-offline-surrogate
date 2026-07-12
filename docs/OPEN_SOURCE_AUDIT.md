# Open-source release audit

Audit date: 2026-07-12

## Decision

Release as a **methodology, scripts, and curated communication repository**.
The public tree contains executable experiment logic, machine-independent
configuration, documentation, one Chinese technical report, one portable HTML
presentation, and three selected figures. Dataset rows and generated experiment
artifacts remain outside Git.

## Upload allowlist

- Python and shell source files;
- machine-independent YAML configurations;
- root README and methodology documents;
- dependency and citation metadata;
- MIT license, license-scope statement, and third-party notices;
- `presentations/advisor_report/report_cn.md`;
- the portable HTML presentation, its project CSS, and vendored HTML runtime
  assets with the upstream license;
- exactly three curated PNG figures listed in the root README.

## Upload denylist

- all `outputs*` directories and symlinked output directories;
- Hugging Face dataset caches and locally generated Parquet splits;
- VMEC++ wout files;
- `.pt`, `.pth`, `.ckpt`, and `.npz` model artifacts;
- candidate boundary JSON/JSONL files;
- row-level audit logs, result serializations, CSV tables, and manifests;
- source data and generation scripts for figures;
- duplicated presentation assets, rendered pages, QA files, build scripts, and
  PPTX exports;
- server logs, PIDs, completion markers, environment dumps, and bundles;
- internal plans, task packets, interview material, and local paper copies.

The denylist and narrow public allowlists are implemented in `.gitignore`.
Before publishing, verify with:

```bash
git status --short --ignored
git ls-files --others --exclude-standard
```

The second command should show source, configuration, documentation, the report,
the HTML presentation and runtime, and exactly three PNG files.

## Findings

### Resolved for the initial release

1. **Repository metadata:** added `README.md`, `LICENSE`,
   `LICENSE_SCOPE.md`, `THIRD_PARTY_NOTICES.md`, `CITATION.cff`, and
   `requirements.txt`.
2. **Large and sensitive generated artifacts:** excluded output trees,
   checkpoints, Parquet files, candidates, logs, serialized results, and
   internal documents.
3. **Curated communication artifacts:** allowlisted the Chinese report, portable
   HTML presentation, required runtime assets, and exactly three final PNG
   figures. Duplicate and source assets remain local.
4. **HTML portability and licensing:** replaced personal absolute paths with
   repository-relative resources and included the upstream HTML PPT Studio MIT
   license beside the vendored files.
5. **Machine-specific public configuration:** removed personal absolute paths
   from public scripts and YAML files. The wout parts directory is an explicit
   CLI/config input.
6. **Baseline reference:** replaced the placeholder in
   `07_write_paper_reference.py` with the table references and values stated in
   the ConStellaration paper.
7. **Claim scope:** the root README separates surrogate ranking from official
   physical verification, states the zero-feasible-sample scope precisely, and
   reports fixed-dimension and expanded-dimension work as distinct tracks.
8. **Authorship:** license and citation metadata identify Shengnian Liu as the
   sole author of the repository content.

### Remaining release risks

1. **No automated tests (high):** scripts compile, but there is no unit test for
   feature ordering, constraint normalization, boundary round-tripping, or
   audit-budget accounting. Add tests before a stable `v1.0` release.
2. **Directory coupling (medium):** later stages still import sibling directories
   via `sys.path` inserts. Experiment trees now use science-role paths under
   `experiments/`, and peer constants resolve the renamed siblings; a shared
   package extraction remains deferred until unit tests exist.
3. **Code duplication (medium):** base, cross-Nfp, and stage-specific modules
   duplicate candidate, common, and audit logic. A shared package should replace
   copies after reproducibility smoke tests are available.
4. **Platform-specific environment (medium):** `torch==2.6.0` and the pinned
   ConStellaration stack may require platform-specific wheels. The README warns
   users to install the appropriate PyTorch build.
5. **Repository URL pending (low):** add the final repository URL to
   `CITATION.cff` after the GitHub repository is created.
6. **Dataset redistribution policy (medium if policy changes):** the official
   Hugging Face dataset card currently declares MIT. This release still excludes
   dataset rows, local splits, and row-level derivatives to keep the repository
   focused on methods and to prevent accidental publication of bulky local
   experiment state. Recheck the dataset card and attribution requirements before
   changing this policy.
7. **Result provenance (medium):** the public report and figures summarize local
   experiments while row-level data remain private. Preserve internal hashes,
   command logs, and manifests so reported values can be audited without
   publishing restricted or bulky artifacts.

## README and paper-style assessment

The public README follows a software-paper pattern:

1. title, author, and one-sentence purpose;
2. research question and contribution;
3. benchmark context and evaluation protocol;
4. repository map and public-artifact links;
5. installation and upstream data acquisition;
6. executable pipeline and experiment tracks;
7. reproducibility and local-artifact rules;
8. references, citation, attribution, and license.

The Chinese report provides the detailed experimental narrative. The README
remains the concise project entry point and links to the report and interactive
presentation.

## License decision

Use MIT for original code, documentation, report, presentation-specific content,
and the three curated figures because:

- the upstream ConStellaration and VMEC++ software are MIT-licensed;
- MIT supports reuse of the repository's scripts and documentation with a clear,
  permissive notice;
- the repository distributes no dataset rows or trained artifacts that require a
  separate data or model license;
- the vendored HTML PPT Studio files carry their upstream MIT license in their
  own directory.

Preserve upstream notices for copied or substantially adapted source. The local
MIT license does not relicense the ConStellaration dataset, VMEC++, HTML PPT
Studio, or other dependencies.

## Pre-publication checklist

- [x] Record Shengnian Liu as sole author in `LICENSE` and `CITATION.cff`.
- [x] Define a narrow allowlist for the report, HTML presentation, and three
      figures.
- [x] Vendor the HTML runtime with its upstream MIT license.
- [ ] Add the final repository URL to `CITATION.cff` after creation.
- [ ] Review the unignored-file list manually.
- [ ] Run Python compilation, shell syntax, YAML parsing, link checks, and
      private-path scans.
- [ ] Create at least one clean-environment smoke run from downloaded upstream
      data without using ignored local outputs.
- [ ] Confirm no candidate boundary, dataset row, checkpoint, or row-level
      result serialization is staged.
