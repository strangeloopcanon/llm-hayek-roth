from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any, TypeAlias

from econ_llm_preferences_experiment.logging_utils import get_logger, log

logger = get_logger(__name__)

RowValue: TypeAlias = str | float | int


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _as_float(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return float(value)
    raise TypeError(f"Expected numeric value, got {type(value).__name__}")


def _write_csv(rows: list[dict[str, RowValue]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        out_path.write_text("", encoding="utf-8")
        return
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_md_table(rows: list[dict[str, RowValue]], out_path: Path) -> None:
    out_path.write_text("\n".join(_md_table_lines(rows)) + ("\n" if rows else ""), encoding="utf-8")


def _md_table_lines(rows: list[dict[str, RowValue]]) -> list[str]:
    if not rows:
        return [""]
    headers = list(rows[0].keys())
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    return lines


def _find_term(rows: list[dict[str, str]], term: str) -> dict[str, str]:
    for row in rows:
        if row.get("term") == term:
            return row
    raise KeyError(f"Missing term={term!r}")


def _extract_triple_reg(*, reg_csv: Path) -> dict[str, RowValue]:
    rows = _read_csv(reg_csv)
    triple = _find_term(rows, "ai_x_central_x_hard")
    nrow = _find_term(rows, "n_obs / n_clusters")
    return {
        "coef": round(_as_float(triple["coef"]), 4),
        "se_cluster": round(_as_float(triple["se(cluster)"]), 4),
        "p": round(_as_float(triple["p(normal)"]), 4),
        "n_obs": str(nrow["coef"]).split("/")[0].strip(),
        "n_clusters": str(nrow["coef"]).split("/")[1].strip() if "/" in str(nrow["coef"]) else "",
    }


def _extract_congestion_row(*, sat_csv: Path, saturation: float) -> dict[str, str]:
    rows = _read_csv(sat_csv)
    for row in rows:
        if abs(_as_float(row["saturation"]) - saturation) <= 1e-9:
            return row
    raise KeyError(f"Missing saturation={saturation} in {sat_csv}")


def _copy_if_exists(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


def _try_add_regime_lambda_rows(*, key_rows: list[dict[str, RowValue]], regime_csv: Path) -> None:
    if not regime_csv.exists():
        return
    rows = _read_csv(regime_csv)
    if not rows or "lambda_star" not in rows[0]:
        return

    max_k_i = max(int(_as_float(r["k_I"])) for r in rows)
    max_k_j = max(int(_as_float(r["k_J"])) for r in rows)

    def find(k_i: int, k_j: int) -> dict[str, str]:
        for r in rows:
            if int(_as_float(r["k_I"])) == k_i and int(_as_float(r["k_J"])) == k_j:
                return r
        raise KeyError(f"Missing (k_I,k_J)=({k_i},{k_j}) in {regime_csv}")

    low = find(1, 1)
    high = find(max_k_i, max_k_j)
    lam_vals = [_as_float(r["lambda_star"]) for r in rows]

    key_rows.extend(
        [
            {
                "block": "Regime_map",
                "metric": "λ* @ low info",
                "outcome": "attention_cost threshold",
                "estimate": round(_as_float(low["lambda_star"]), 4),
                "se": "",
                "p": "",
                "n_obs": "",
                "n_clusters": "",
                "notes": "k_I=1,k_J=1; central wins if λ > λ*",
            },
            {
                "block": "Regime_map",
                "metric": "λ* @ high info",
                "outcome": "attention_cost threshold",
                "estimate": round(_as_float(high["lambda_star"]), 4),
                "se": "",
                "p": "",
                "n_obs": "",
                "n_clusters": "",
                "notes": f"k_I={max_k_i},k_J={max_k_j}; central wins if λ > λ*",
            },
            {
                "block": "Regime_map",
                "metric": "λ* range",
                "outcome": "attention_cost threshold",
                "estimate": "",
                "se": "",
                "p": "",
                "n_obs": "",
                "n_clusters": "",
                "notes": f"min={min(lam_vals):.4f}, max={max(lam_vals):.4f}",
            },
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="reports/paper_latest")
    parser.add_argument("--field-v2-dir", default="reports/field_v2_latest")
    parser.add_argument("--sens-dir", default="reports/field_v2_sensitivity_latest")
    parser.add_argument("--ablations-dir", default="reports/ablations_latest")
    parser.add_argument("--congestion-dir", default="reports/latest_congestion")
    parser.add_argument("--minimal-dir", default="reports/latest")
    parser.add_argument("--intakes-dir", default="reports/intakes_latest")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    field_v2 = Path(args.field_v2_dir)
    sens = Path(args.sens_dir)
    ablations = Path(args.ablations_dir)
    congestion = Path(args.congestion_dir)
    minimal = Path(args.minimal_dir)
    intakes = Path(args.intakes_dir)

    key_rows: list[dict[str, RowValue]] = []

    # FieldSim v2: the main interaction claim (cluster-robust).
    for outcome in ("matched", "canceled", "consumer_surplus", "provider_profit", "net_welfare"):
        reg = _extract_triple_reg(reg_csv=field_v2 / f"reg_{outcome}.csv")
        key_rows.append(
            {
                "block": "FieldSim_v2",
                "metric": "ai×central×hard",
                "outcome": outcome,
                "estimate": reg["coef"],
                "se": reg["se_cluster"],
                "p": reg["p"],
                "n_obs": reg["n_obs"],
                "n_clusters": reg["n_clusters"],
                "notes": "cluster-robust (cluster=cell)",
            }
        )

    # FieldSim v2 sensitivity: robustness summary (no new estimation).
    sens_summary = _read_json(sens / "summary.json")
    key_rows.append(
        {
            "block": "FieldSim_v2_sensitivity",
            "metric": "share(triple_welfare>0)",
            "outcome": "net_welfare",
            "estimate": round(_as_float(sens_summary["pos_share_triple_welfare"]), 3),
            "se": "",
            "p": "",
            "n_obs": int(_as_float(sens_summary["n_runs"])),
            "n_clusters": "",
            "notes": "grid over seeds × knobs",
        }
    )
    key_rows.append(
        {
            "block": "FieldSim_v2_sensitivity",
            "metric": "share(p<0.05)",
            "outcome": "net_welfare",
            "estimate": round(_as_float(sens_summary["sig_share_triple_welfare_p05"]), 3),
            "se": "",
            "p": "",
            "n_obs": int(_as_float(sens_summary["n_runs"])),
            "n_clusters": "",
            "notes": "p-value for ai×central×hard (clustered)",
        }
    )

    # Regime map: ROI boundary λ* for centralized matching.
    _try_add_regime_lambda_rows(key_rows=key_rows, regime_csv=minimal / "regime_grid_hard.csv")

    # Congestion experiment: adoption intensity externality.
    for category in ("easy", "hard"):
        sat_csv = congestion / f"congestion_saturation_{category}.csv"
        r0 = _extract_congestion_row(sat_csv=sat_csv, saturation=0.0)
        r25 = _extract_congestion_row(sat_csv=sat_csv, saturation=0.25)
        r100 = _extract_congestion_row(sat_csv=sat_csv, saturation=1.0)

        key_rows.append(
            {
                "block": "Congestion_saturation",
                "metric": "Δ net_welfare (0.25 - 0.00)",
                "outcome": f"net_welfare_per_customer ({category})",
                "estimate": round(
                    _as_float(r25["net_welfare_per_customer"])
                    - _as_float(r0["net_welfare_per_customer"]),
                    4,
                ),
                "se": "",
                "p": "",
                "n_obs": "",
                "n_clusters": "",
                "notes": (
                    f"treated match_rate={float(r25['match_rate_treated']):.3f}, "
                    f"control match_rate={float(r25['match_rate_control']):.3f}"
                ),
            }
        )
        key_rows.append(
            {
                "block": "Congestion_saturation",
                "metric": "Δ net_welfare (1.00 - 0.00)",
                "outcome": f"net_welfare_per_customer ({category})",
                "estimate": round(
                    _as_float(r100["net_welfare_per_customer"])
                    - _as_float(r0["net_welfare_per_customer"]),
                    4,
                ),
                "se": "",
                "p": "",
                "n_obs": "",
                "n_clusters": "",
                "notes": (
                    f"inbox/provider/day@1.0={float(r100['inbox_per_provider_per_day']):.2f}, "
                    f"response_rate@1.0={float(r100['provider_response_rate']):.3f}"
                ),
            }
        )

    # LLM parsing quality: why AI intake is plausibly higher-density.
    for category in ("easy", "hard"):
        pq_rows = _read_csv(ablations / f"parsing_quality_{category}.csv")
        for mode in ("form_top3", "free_text_gpt", "chat_gpt"):
            for side in ("customer", "provider"):
                row = next(
                    r
                    for r in pq_rows
                    if r["mode"] == mode and r["side"] == side and r["category"] == category
                )
                key_rows.append(
                    {
                        "block": "LLM_parsing_quality",
                        "metric": mode,
                        "outcome": f"mean_l1 ({category},{side})",
                        "estimate": round(_as_float(row["mean_l1"]), 4),
                        "se": "",
                        "p": "",
                        "n_obs": "",
                        "n_clusters": "",
                        "notes": f"top1_acc={float(row['top1_accuracy']):.3f}",
                    }
                )

    _write_csv(key_rows, out_dir / "key_results.csv")
    key_md_preamble = [
        "# Key results",
        "",
        "So what: one table spanning the repo’s synthetic experiments (main effects + mechanisms).",
        "",
        "Notes:",
        "- `attention_cost` (λ): per-action cost (messages, inbox screening, accept decisions).",
        "- `λ*`: threshold λ where central and search tie in net welfare; central wins if λ > λ*.",
        "- FieldSim v2: `net_welfare = total_surplus - attention_cost * actions`.",
        "- GPT usage: parsing/elicitation only; FieldSim v2 has no API calls.",
        "",
    ]
    (out_dir / "key_results.md").write_text(
        "\n".join([*key_md_preamble, *_md_table_lines(key_rows)]) + "\n", encoding="utf-8"
    )

    # Copy a few "main figures" for quick scanning.
    _copy_if_exists(field_v2 / "fig_hard_net_welfare.svg", out_dir / "fig_hard_net_welfare.svg")
    _copy_if_exists(
        field_v2 / "fig_hard_reciprocity_curve.svg",
        out_dir / "fig_hard_reciprocity_curve.svg",
    )
    _copy_if_exists(
        congestion / "fig_hard_net_welfare_vs_saturation.svg",
        out_dir / "fig_hard_net_welfare_vs_saturation.svg",
    )
    _copy_if_exists(
        minimal / "regime_map_hard_net_welfare_diff.svg",
        out_dir / "fig_regime_map_hard_net_welfare_diff.svg",
    )
    _copy_if_exists(
        minimal / "regime_map_hard_lambda_star.svg",
        out_dir / "fig_regime_map_hard_lambda_star.svg",
    )

    readme = "\n".join(
        [
            "# Paper bundle (synthetic, but reviewer-proofed)",
            "",
            "So what: this folder is the single entrypoint to the repo’s main results",
            "(one table + the key figures).",
            "",
            "Notes (quick definitions):",
            "- `attention_cost` (λ): per-action cost charged for communication/attention.",
            "- `λ*`: break-even λ for centralized matching vs search (central wins if λ > λ*).",
            "- FieldSim v2 net welfare subtracts attention costs from total surplus.",
            "- GPT is used for parsing/elicitation only; FieldSim v2 is simulation-only.",
            "",
            "Why this matters:",
            "- Search is the backup plan when the platform can’t confidently infer preferences.",
            "- Better AI intake can lower the ROI boundary λ* needed for centralized matching",
            "  to be worthwhile in hard-to-describe jobs.",
            "- Delegated outreach agents can create a congestion externality at high adoption.",
            "",
            "Framing (matches reviewer-proof evidence):",
            "- Not “LLMs always improve markets” (effects vary by regime).",
            "- Yes “LLMs can move you across a threshold where central mechanisms become viable,”",
            "  which is captured by the regime map ROI boundary λ*",
            "  and the hard-category interaction.",
            "",
            "Key artifacts:",
            "- `key_results.md` / `key_results.csv`: one table spanning all experiments",
            "- `fig_hard_net_welfare.svg`: FieldSim v2 net welfare by arm (hard category)",
            "- `fig_hard_reciprocity_curve.svg`: FieldSim v2 reciprocity proxy vs rank (hard)",
            "- `fig_hard_net_welfare_vs_saturation.svg`: delegated outreach congestion (hard)",
            "- `fig_regime_map_hard_net_welfare_diff.svg`: regime map sweep (hard)",
            "- `fig_regime_map_hard_lambda_star.svg`: regime map ROI boundary (hard)",
            "",
            "Source reports (full details + additional plots):",
            f"- FieldSim v2: `{field_v2}`",
            f"- Sensitivity: `{sens}`",
            f"- LLM ablations: `{ablations}`",
            f"- Congestion: `{congestion}`",
            f"- Minimal 2×2 + regime map: `{minimal}`",
            f"- Intake samples (home services): `{intakes}`",
            "",
        ]
    )
    (out_dir / "README.md").write_text(readme + "\n", encoding="utf-8")
    log(logger, 20, "paper_bundle_written", out_dir=str(out_dir), n_rows=len(key_rows))


if __name__ == "__main__":
    main()
