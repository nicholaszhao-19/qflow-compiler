#!/usr/bin/env python3
"""Recompute qubit-budget and Pareto comparisons from measured circuit tables.

Run from the repository root:
    python scripts/analyze_crossover.py --results-dir results

No benchmark numbers are hard-coded.  Both baseline families and SelectSwap
block sizes are reconsidered independently at each qubit cap and T model.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform


CAPS = (128, 256, 512)
MODELS = {
    "slope3": "t_count_estimate",
    "slope4": "t_count_estimate_slope4",
}
CONTEXT_FIELDS = (
    "experiment", "dataset", "kind", "dimension", "grid", "rank",
    "beta", "time", "tolerance", "seed",
)
NUMBER_FIELDS = {"dimension", "grid", "rank", "beta", "time", "tolerance", "seed"}


def number(row: dict, field: str, default=None):
    value = row.get(field)
    return default if value is None or value == "" else float(value)


def context(row: dict) -> dict:
    return {
        field: number(row, field) if field in NUMBER_FIELDS else row.get(field, "")
        for field in CONTEXT_FIELDS
    }


def key(row: dict) -> tuple:
    return tuple(context(row).values())


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def compact(row: dict) -> dict:
    fields = (
        "logical_qubits", "ancilla_qubits", "t_count_estimate",
        "t_count_estimate_slope4", "t_depth_macro_estimate", "native_depth",
        "cnot_decomposed", "ccx", "arbitrary_rotations", "operator_error",
        "block_bits", "phase_bits", "core_fraction_bits", "validated_inputs",
    )
    item = {"method": row["method"]}
    for field in fields:
        value = number(row, field)
        if value is not None:
            item[field] = int(value) if field not in {"operator_error"} else value
    item["exhaustive_gate_validation"] = row.get("exhaustive", "").lower() == "true"
    return item


def choose_baseline(rows: list[dict], cap: int | None, metric: str) -> dict | None:
    feasible = [row for row in rows if cap is None or number(row, "logical_qubits") <= cap]
    if not feasible:
        return None
    return min(feasible, key=lambda row: (
        number(row, metric), number(row, "logical_qubits"),
        number(row, "native_depth"), row["method"], number(row, "block_bits", 0),
    ))


def compare(tt: dict, baselines: list[dict], cap: int | None, metric: str) -> dict:
    feasible = cap is None or number(tt, "logical_qubits") <= cap
    baseline = choose_baseline(baselines, cap, metric)
    answer = {
        "qubit_cap": cap,
        "tt_feasible": feasible,
        "baseline_feasible": baseline is not None,
        "eligible_baseline_count": sum(cap is None or number(row, "logical_qubits") <= cap for row in baselines),
        "best_baseline": compact(baseline) if baseline is not None else None,
    }
    if not feasible:
        answer["outcome"] = "tt_infeasible"
        return answer
    if baseline is None:
        answer["outcome"] = "no_feasible_baseline"
        return answer
    a, b = number(tt, metric), number(baseline, metric)
    answer["outcome"] = "tt_win" if a < b else "baseline_win" if a > b else "tie"
    answer["tt_t_count"] = int(a)
    answer["baseline_t_count"] = int(b)
    answer["tt_over_baseline_t_ratio"] = a / b if b else None
    answer["tt_relative_t_reduction"] = 1 - a / b if b else None
    return answer


def dominates(a: dict, b: dict, metric: str) -> bool:
    qa, qb = number(a, "logical_qubits"), number(b, "logical_qubits")
    ta, tb = number(a, metric), number(b, metric)
    return qa <= qb and ta <= tb and (qa < qb or ta < tb)


def pareto_frontier(rows: list[dict], metric: str) -> list[dict]:
    frontier = [row for row in rows if not any(dominates(other, row, metric) for other in rows)]
    # Equivalent candidate points need not be printed multiple times.
    seen = set()
    result = []
    for row in sorted(frontier, key=lambda row: (number(row, "logical_qubits"), number(row, metric))):
        identity = (number(row, "logical_qubits"), number(row, metric))
        if identity not in seen:
            result.append(compact(row))
            seen.add(identity)
    return result


def analyze_case(tt: dict, baselines: list[dict]) -> dict:
    tolerance = number(tt, "tolerance")
    if tolerance is None or tolerance <= 0:
        raise ValueError("Every matched case must have a positive target tolerance")
    approximation_budget = tolerance / 2
    for row in [tt] + baselines:
        if number(row, "operator_error", float("inf")) > approximation_budget + 1e-10:
            raise ValueError(f"Input case exceeds its matched approximation budget: {context(row)}")
        if abs(number(row, "rotation_error_budget", -1) - tolerance / 2) > 1e-14:
            raise ValueError("Input rows do not share the same synthesis-error allocation")
        if any(number(row, metric) is None for metric in MODELS.values()):
            raise ValueError("Input rows lack a required T-count sensitivity estimate")
    answer = {"context": context(tt), "tt": compact(tt), "baseline_candidate_count": len(baselines), "models": {}}
    for model, metric in MODELS.items():
        dominators = [compact(row) for row in baselines if dominates(row, tt, metric)]
        comparisons = {str(cap): compare(tt, baselines, cap, metric) for cap in CAPS}
        comparisons["tt_width"] = compare(tt, baselines, int(number(tt, "logical_qubits")), metric)
        comparisons["compiled_portfolio"] = compare(tt, baselines, None, metric)
        win_interval = None
        if comparisons["tt_width"]["outcome"] == "tt_win":
            matching_cost_widths = [int(number(row, "logical_qubits")) for row in baselines
                                    if number(row, metric) <= number(tt, metric)]
            win_interval = {
                "minimum_cap_inclusive": int(number(tt, "logical_qubits")),
                "maximum_cap_inclusive": min(matching_cost_widths) - 1 if matching_cost_widths else None,
                "scope": "all integer caps against the actually compiled baseline portfolio",
            }
        answer["models"][model] = {
            "metric": metric,
            "comparisons": comparisons,
            "tt_strict_win_cap_interval": win_interval,
            "tt_pareto_nondominated": not dominators,
            "baseline_dominators": dominators,
            "combined_pareto_frontier": pareto_frontier([tt] + baselines, metric),
        }
    return answer


def summarize(cases: list[dict]) -> dict:
    summary = {}
    scopes = {"all": cases, "learned": [c for c in cases if c["context"]["kind"] == "learned"],
              "synthetic": [c for c in cases if c["context"]["kind"] == "synthetic"]}
    for scope, selected in scopes.items():
        summary[scope] = {"case_count": len(selected), "models": {}}
        for model in MODELS:
            model_summary = {}
            for cap in [str(n) for n in CAPS] + ["tt_width", "compiled_portfolio"]:
                comparisons = [(case, case["models"][model]["comparisons"][cap]) for case in selected]
                outcomes = [row["outcome"] for _, row in comparisons]
                model_summary[cap] = {
                    "tt_feasible_cases": sum(row["tt_feasible"] for _, row in comparisons),
                    "tt_infeasible_cases": outcomes.count("tt_infeasible"),
                    "tt_wins": outcomes.count("tt_win"),
                    "baseline_wins": outcomes.count("baseline_win"),
                    "ties": outcomes.count("tie"),
                    "no_feasible_baseline": outcomes.count("no_feasible_baseline"),
                    "tt_winning_cases": [case["context"] for case, row in comparisons if row["outcome"] == "tt_win"],
                }
            model_summary["tt_pareto_nondominated_cases"] = sum(case["models"][model]["tt_pareto_nondominated"] for case in selected)
            summary[scope]["models"][model] = model_summary
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path(__file__).resolve().parents[1] / "results")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.results_dir.resolve()
    output = args.output or root / "crossover_analysis.json"
    inputs = [root / name for name in ("matched.csv", "baseline_portfolio.csv", "scaling.csv", "scaling_portfolio.csv")]
    all_cases = []
    for case_file, portfolio_file in ((inputs[0], inputs[1]), (inputs[2], inputs[3])):
        portfolios = {}
        for row in read_csv(portfolio_file):
            portfolios.setdefault(key(row), []).append(row)
        tt_rows = [row for row in read_csv(case_file) if row["method"] == "reversible_tt"]
        if len({key(row) for row in tt_rows}) != len(tt_rows):
            raise ValueError(f"Duplicate case identities in {case_file}")
        for row in tt_rows:
            baseline_rows = portfolios.get(key(row))
            if not baseline_rows:
                raise ValueError(f"Missing complete baseline portfolio for {context(row)}")
            all_cases.append(analyze_case(row, baseline_rows))
    config_file = root / "config.json"
    config = json.loads(config_file.read_text()) if config_file.exists() else {}
    if config_file.exists():
        inputs.append(config_file)
    result = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "source_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs},
        "baseline_portfolio_qubit_limits": {
            group: config.get(group, {}).get("baseline_max_qubits") for group in ("oracle", "scaling")
        },
        "interpretation": {
            "primary_comparison": "Minimum modelled T count among all implemented baseline candidates satisfying the same qubit cap and operator-error budget.",
            "compiled_portfolio": "No additional qubit cap beyond the configured, actually compiled baseline portfolio; not an unrestricted optimization over every possible implementation.",
            "tt_width": "Baseline candidates may use up to the exact TT circuit width; both sides satisfy that same cap.",
            "pareto": "Two objectives: allocated logical qubits and modelled T count. Circuit depth, CNOTs, compilation time and physical overhead are not Pareto objectives here.",
            "cap_intervals": "A strict-win interval starts at the TT width and ends just before the first compiled baseline with no greater T cost becomes feasible. SelectSwap block sizes are powers of two, so other lookup variants could fill gaps.",
            "infeasible": "A TT circuit wider than the fixed cap is excluded from both wins and losses at that cap.",
            "accuracy": "Each method must have measured finite-grid potential-phase error at most tolerance/2; the remaining tolerance/2 is a prospective rotation-synthesis budget.",
            "model": "Primary ceil(3 log2(1/delta)+10) and sensitivity ceil(4 log2(1/delta)+10) per arbitrary rotation; exact CCX costs 7T.",
            "claim_scope": "Logical circuit tradeoffs for the implemented coherent circuits. No physical fault-tolerant execution, global synthesis optimality, new algorithm theorem or computational quantum advantage is established.",
        },
        "summary": summarize(all_cases),
        "cases": all_cases,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Wrote {output} from {len(all_cases)} complete matched cases")
    for scope, info in result["summary"].items():
        primary = info["models"]["slope3"]
        text = ", ".join(f"{cap}: {primary[cap]['tt_wins']}/{primary[cap]['tt_feasible_cases']} TT wins" for cap in [str(n) for n in CAPS] + ["tt_width", "compiled_portfolio"])
        print(f"{scope} ({info['case_count']} cases): {text}")


if __name__ == "__main__":
    main()
