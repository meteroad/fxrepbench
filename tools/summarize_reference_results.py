#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "reference_results/counterfx-200/candidate_budget"
RUN_ROOT = BASE / "pedalboard_test200_seed20270724"


PATHS = {
    ("RelFx", "random", "known"): RUN_ROOT / "relativefx_random_known/budget_curve.csv",
    ("RelFx", "random", "hidden"): RUN_ROOT / "relativefx_random_unknown/budget_curve.csv",
    ("RelFx", "head", "known"): RUN_ROOT / "relativefx_head_known/budget_curve.csv",
    ("RelFx", "head", "hidden"): RUN_ROOT / "relativefx_head_unknown/budget_curve.csv",
    ("Fx-Encoder++", "random", "known"): RUN_ROOT / "fxencoderpp_random_known/budget_curve.csv",
    ("Fx-Encoder++", "random", "hidden"): BASE / "fxencoderpp_random_unknown/budget_curve.csv",
    ("Fx-Encoder++", "head", "known"): BASE / "fxencoderpp_head_known/budget_curve.csv",
    ("Fx-Encoder++", "head", "hidden"): RUN_ROOT / "fxencoderpp_head_unknown/budget_curve.csv",
    ("AFx-Rep", "random", "known"): RUN_ROOT / "afxrep_random_known/budget_curve.csv",
    ("AFx-Rep", "random", "hidden"): BASE / "afxrep_random_unknown/budget_curve.csv",
    ("AFx-Rep", "head", "known"): BASE / "afxrep_head_known/budget_curve.csv",
    ("AFx-Rep", "head", "hidden"): RUN_ROOT / "afxrep_head_unknown/budget_curve.csv",
}


def budget_row(path: Path, budget: int) -> dict[str, str]:
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if int(row["budget"]) == budget:
                return row
    raise KeyError(f"budget {budget} is absent from {path}")


def values(encoder: str, proposal: str, budget: int) -> list[str]:
    known = budget_row(PATHS[(encoder, proposal, "known")], budget)
    hidden = budget_row(PATHS[(encoder, proposal, "hidden")], budget)
    return [
        encoder,
        "Global random" if proposal == "random" else ("Head@1" if budget == 1 else "Head-centered"),
        str(budget),
        f"{float(known['selected_ld_mean']):.3f}",
        f"{float(known['oracle_ld_mean']):.3f}",
        f"{float(hidden['selected_ld_mean']):.3f}",
        f"{float(hidden['oracle_ld_mean']):.3f}",
        f"{float(hidden['selected_fx_set_f1_mean']):.3f}",
    ]


def main() -> None:
    header = [
        "Verifier",
        "Proposal",
        "N",
        "Known selected",
        "Known oracle",
        "Hidden selected",
        "Hidden oracle",
        "FX-set F1",
    ]
    rows = [
        *(values(encoder, "random", 512) for encoder in ("RelFx", "Fx-Encoder++", "AFx-Rep")),
        values("RelFx", "head", 1),
        values("RelFx", "head", 16),
        values("RelFx", "head", 512),
        values("Fx-Encoder++", "head", 512),
        values("AFx-Rep", "head", 512),
    ]
    widths = [max(len(row[index]) for row in [header, *rows]) for index in range(len(header))]
    print(" | ".join(value.ljust(widths[index]) for index, value in enumerate(header)))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(" | ".join(value.ljust(widths[index]) for index, value in enumerate(row)))


if __name__ == "__main__":
    main()
