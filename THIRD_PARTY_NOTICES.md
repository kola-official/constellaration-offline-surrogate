# Third-party notices

## ConStellaration

This project builds on the ConStellaration benchmark, evaluator, and Python
package developed by Proxima Fusion GmbH and collaborators.

- Upstream repository: <https://github.com/proximafusion/constellaration>
- Dataset: <https://huggingface.co/datasets/proxima-fusion/constellaration>
- Paper: <https://openreview.net/forum?id=NQSbGKlCpx>
- Upstream software license: MIT
- Dataset card license: MIT (verified 2026-07-12)
- Reproducibility reference used by these experiments: upstream commit
  `112b20ae07193910d467d26033fe51022e641b9f`

Copyright notices and the upstream MIT license must be preserved for any copied
or substantially adapted upstream source code.

## VMEC++

VMEC++ is developed by Proxima Fusion and collaborators and is distributed
under the MIT License. This repository calls VMEC++ through ConStellaration for
high-fidelity audit runs; it does not vendor VMEC++ source code.

- Project: <https://github.com/proximafusion/vmecpp>
- Documentation: <https://proximafusion.github.io/vmecpp/>
- Paper: <https://arxiv.org/abs/2502.04374>

## Python dependencies

Python dependencies are installed from their original distributions and retain
their own licenses. `requirements.txt` is an environment specification, not a
redistribution of dependency source code.

## HTML PPT Studio assets

The HTML presentation includes local copies of the runtime and style assets from
HTML PPT Studio (`html-ppt-skill`), created by Lewis (`sudolewis@gmail.com`) and
released under the MIT License.

- Project: <https://github.com/lewislulu/html-ppt-skill>
- Vendored files and upstream license: `presentations/advisor_report/vendor/html-ppt/`

The vendored files retain their upstream MIT terms. The presentation-specific
content and styling authored for this project are copyright © 2026 Shengnian
Liu.
