"""Measure stellarator-symmetric Fourier mode cutoffs from r_cos / z_sin shapes.

Used by tests that document public leaderboard boundary dimensions relative to
the official low-order (~80-DOF, m,n ≤ 4) training support.
"""

from __future__ import annotations

from typing import Any


def mode_cutoffs_from_array_shape(
    r_cos_shape: list[int] | tuple[int, ...],
    z_sin_shape: list[int] | tuple[int, ...] | None = None,
) -> dict[str, Any]:
    """Map 2-D Fourier coefficient array shapes to mode cutoffs.

    For a stellarator-symmetric ``SurfaceRZFourier``-style table:
    - rows index poloidal modes ``m = 0 .. mpol``
    - columns index toroidal modes ``n = -ntor .. +ntor``

    The official ConStellaration low-order setting uses ``mpol = ntor = 4``,
    which yields shapes ``(5, 9)`` for both ``r_cos`` and ``z_sin``.
    """
    if len(r_cos_shape) != 2:
        raise ValueError(f"expected 2-D r_cos shape, got {r_cos_shape!r}")
    m_rows, n_cols = int(r_cos_shape[0]), int(r_cos_shape[1])
    if n_cols % 2 == 0:
        raise ValueError(f"expected odd n-columns for n=-N..N, got {n_cols}")
    mpol = m_rows - 1
    ntor = (n_cols - 1) // 2
    if z_sin_shape is not None and tuple(z_sin_shape) != (m_rows, n_cols):
        raise ValueError(
            f"r_cos shape {tuple(r_cos_shape)} != z_sin shape {tuple(z_sin_shape)}"
        )
    return {
        "mpol_max": mpol,
        "ntor_max": ntor,
        "r_cos_shape": (m_rows, n_cols),
        "is_low_order_4": mpol == 4 and ntor == 4,
        "is_expanded": mpol > 4 or ntor > 4,
    }
