"""Public leaderboard Fourier shapes vs this repo's official-space mainline."""

from __future__ import annotations

import json
import re
from pathlib import Path

from fourier_mode_shapes import mode_cutoffs_from_array_shape

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "leaderboard_boundary_shapes.json"


def test_low_order_dataset_shape_is_m_n_le_4() -> None:
    info = mode_cutoffs_from_array_shape([5, 9], [5, 9])
    assert info["mpol_max"] == 4
    assert info["ntor_max"] == 4
    assert info["is_low_order_4"] is True
    assert info["is_expanded"] is False


def test_public_high_score_shapes_include_expanded_modes() -> None:
    payload = json.loads(FIXTURE.read_text())
    rows = payload["rows"]
    assert rows, "fixture must contain public leaderboard shape rows"

    expanded = []
    low_order = []
    for row in rows:
        info = mode_cutoffs_from_array_shape(row["r_cos_shape"], row["z_sin_shape"])
        assert info["r_cos_shape"] == tuple(row["r_cos_shape"])
        if info["is_expanded"]:
            expanded.append((row["user"], row["score"], info))
        if info["is_low_order_4"]:
            low_order.append((row["user"], row["score"], info))

    assert low_order, "fixture should retain at least one low-order (5,9) row"
    assert expanded, "fixture should retain expanded-mode public board rows"
    # High simple_to_build scores in the fixture use expanded cutoffs.
    stb_expanded = [
        row
        for row in rows
        if row["problem_type"] == "simple_to_build"
        and mode_cutoffs_from_array_shape(row["r_cos_shape"], row["z_sin_shape"])[
            "is_expanded"
        ]
    ]
    assert max(r["score"] for r in stb_expanded) > 0.5
    assert any(
        mode_cutoffs_from_array_shape(r["r_cos_shape"])["mpol_max"] >= 7
        for r in stb_expanded
    )


def _relation_section(text: str, heading: str) -> str:
    parts = text.split(heading, 1)
    assert len(parts) == 2, f"missing heading {heading!r}"
    body = parts[1]
    # next level-2 heading ends the section
    m = re.search(r"\n## ", body)
    return body if m is None else body[: m.start()]


def test_readme_en_states_expanded_vs_official_space() -> None:
    text = (ROOT / "README.md").read_text()
    section = _relation_section(
        text, "## Relation to ConStellaration and the official leaderboard"
    )
    assert "official-space" in section or "low-order" in section
    assert "(5, 9)" in section
    assert "(8, 15)" in section or "m,n ≤ 7" in section or "m,n <= 7" in section
    assert "expanded" in section.lower()
    assert "80" in section
    # plain prose: no contrastive framing in this section
    for bad in ("rather than", "not a claim", "does not replace", "not X"):
        assert bad not in section.lower()


def test_readme_zh_states_expanded_vs_official_space() -> None:
    text = (ROOT / "README_zh.md").read_text()
    section = _relation_section(text, "## 与 ConStellaration 及官方排行榜的关系")
    assert "官方空间" in section or "低阶" in section
    assert "(5, 9)" in section
    assert "(8, 15)" in section or "m,n ≤ 7" in section
    assert "扩展" in section
    assert "80" in section
    for bad in ("而不是", "而非", "并非"):
        assert bad not in section
