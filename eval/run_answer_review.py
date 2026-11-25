"""Collect grounded answers for manual, side-by-side model review."""

import argparse
import csv
import json
import math
import re
import urllib.request
from pathlib import Path
from typing import Any


def ask(api: str, question: str) -> dict[str, Any]:
    request = urllib.request.Request(
        f"{api.rstrip('/')}/ask",
        data=json.dumps({"question": question, "top_k": 12, "explanation_level": "detailed"}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.loads(response.read())


def flatten_answer(result: dict[str, Any]) -> str:
    structured = result.get("structured_answer", {})
    pieces = [structured.get("direct_answer", {}).get("text", result.get("answer", ""))]
    for section in structured.get("sections", []):
        pieces.append(f"\n{section.get('title', '')}")
        pieces.extend(item.get("text", "") for item in section.get("content", []))
    pieces.extend(structured.get("additional_context", []))
    return "\n\n".join(piece for piece in pieces if piece)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("eval/study_guide_cases.json"))
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--model-name", required=True, help="Label the model/configuration used for this run")
    parser.add_argument("--output", type=Path, required=True, help="CSV file to review and retain")
    args = parser.parse_args()
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))

    fields = [
        "model_name", "case_id", "topic", "question", "expected_answer", "generated_answer", "retrieved_sources",
        "expected_numeric_result", "numeric_result_present", "factual_accuracy_0_2", "slide_coverage_0_2", "formula_math_0_2", "explanation_quality_0_2", "reviewer", "notes",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8-sig") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields)
        writer.writeheader()
        for case in dataset["cases"]:
            result = ask(args.api, case["question"])
            sources = "; ".join(
                f"{item.get('filename', '')} page/slide {item.get('page', '')}"
                for item in result.get("citations", [])
            )
            answer_text = flatten_answer(result)
            calculation = case.get("calculation", {})
            expected_numeric = calculation.get("expected")
            numeric_present = ""
            if expected_numeric is not None:
                values = re.findall(r"(?<![A-Za-z])[-+]?(?:\d+(?:,\d{3})*(?:\.\d+)?|\.\d+)(?:e[-+]?\d+)?", answer_text, re.IGNORECASE)
                numeric_present = str(any(
                    math.isclose(float(value.replace(",", "")), float(expected_numeric), rel_tol=0.01, abs_tol=0.005)
                    for value in values
                )).lower()
            writer.writerow({
                "model_name": args.model_name,
                "case_id": case["id"],
                "topic": case["topic"],
                "question": case["question"],
                "expected_answer": case["expected_answer"],
                "generated_answer": answer_text,
                "retrieved_sources": sources,
                "expected_numeric_result": expected_numeric if expected_numeric is not None else "",
                "numeric_result_present": numeric_present,
                "factual_accuracy_0_2": "",
                "slide_coverage_0_2": "",
                "formula_math_0_2": "",
                "explanation_quality_0_2": "",
                "reviewer": "",
                "notes": "",
            })
            print(f"Collected {case['id']}")
    print(f"Review sheet written to {args.output}")
    print("Have an instructor fill the score and reviewer columns before treating this run as approved.")


if __name__ == "__main__":
    main()
