from __future__ import annotations

from common import OUTPUT_DIR, ensure_output_dirs, parse_args, write_json


def main() -> None:
    _ = parse_args()
    ensure_output_dirs()
    write_json(
        OUTPUT_DIR / "baselines" / "paper_alm_ngopt_reference.json",
        {
            "baseline_id": "R0_paper_ALM_NGOpt",
            "problem": "SimpleToBuildQIStellarator",
            "evidence_type": "paper_reference",
            "paper_reported_score": 0.431,
            "paper_reported_metrics": {},
            "source": {
                "paper": "ConStellaration",
                "table_or_section": "Table 3 (score) and Table 4 (objective/constraint violation)",
                "quote_or_note": "Simple-to-build ALM-NGOpt: score 0.431; objective 8.61; normalized constraint violation 0.009.",
            },
            "rerun_on_gpu_server": False,
            "reason_not_rerun": "time_and_compute_budget",
            "comparison_rule": "external_reference_only_not_equal_budget_audit",
        },
    )
    print("Wrote paper ALM-NGOpt reference")


if __name__ == "__main__":
    main()
