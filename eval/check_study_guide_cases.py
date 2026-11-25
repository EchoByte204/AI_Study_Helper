"""Check arithmetic in the hand-labeled study-guide regression examples."""

import json
import math
from pathlib import Path
from typing import Any


def calculate(case: dict[str, Any]) -> float:
    data = case["calculation"]
    kind = data["kind"]
    if kind == "ratio":
        return float(data["numerator"]) / float(data["denominator"])
    if kind == "amdahl":
        fraction = float(data["fraction"])
        part_speedup = float(data["part_speedup"])
        return 1 / ((1 - fraction) + fraction / part_speedup)
    if kind == "solve_part_speedup":
        fraction = float(data["fraction"])
        overall_speedup = float(data["overall_speedup"])
        denominator = 1 / overall_speedup - (1 - fraction)
        if denominator <= 0:
            raise ValueError(f"{case['id']}: target speedup is impossible for this fraction")
        return fraction / denominator
    raise ValueError(f"{case['id']}: unknown calculation kind {kind!r}")


def main() -> None:
    dataset_path = Path(__file__).with_name("study_guide_cases.json")
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    failed = []
    for case in dataset["cases"]:
        actual = calculate(case)
        expected = float(case["calculation"]["expected"])
        passed = math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-6)
        print(f"{'PASS' if passed else 'FAIL'} {case['id']}: expected {expected:.6f}, got {actual:.6f}")
        if not passed:
            failed.append(case["id"])
    print(f"\n{len(dataset['cases']) - len(failed)}/{len(dataset['cases'])} arithmetic cases passed.")
    print(f"Instructor review status: {dataset['review_status']}.")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
