"""Evaluate retrieval and answer evidence against a small human-labeled JSON dataset."""

import argparse
import json
import os
import re
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def post_json(url: str, body: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.loads(response.read())


def evaluate(dataset_path: Path, api: str, k: int) -> dict[str, Any]:
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not dataset:
        raise ValueError("Evaluation dataset is empty")
    recalls: list[float] = []
    reciprocal_ranks: list[float] = []
    citation_correct: list[float] = []
    groundedness: list[float] = []
    latencies: list[float] = []
    total_input_tokens = 0
    failures = 0

    for row in dataset:
        question = str(row["question"])
        gold = {(str(item["filename"]), int(item["page"])) for item in row.get("expected_sources", [])}
        start = time.perf_counter()
        try:
            result = post_json(f"{api.rstrip('/')}/ask", {"question": question, "top_k": k})
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            failures += 1
            print(f"Request failed for {question!r}: {exc}")
            continue
        latency = time.perf_counter() - start
        latencies.append(latency)
        citations = result.get("citations", [])[:k]
        predicted = [(item.get("filename", ""), int(item.get("page", -1))) for item in citations]
        hits = [index for index, item in enumerate(predicted, start=1) if item in gold]
        recalls.append(len(set(predicted) & gold) / max(len(gold), 1))
        reciprocal_ranks.append(1 / min(hits) if hits else 0.0)
        citation_correct.append(sum(item in gold for item in predicted) / max(len(predicted), 1))
        evidence = " ".join(str(item.get("excerpt", "")) for item in citations)
        answer_terms = tokens(str(result.get("answer", ""))) - {"the", "and", "for", "with", "that", "this", "from", "are", "was", "is"}
        evidence_terms = tokens(evidence)
        groundedness.append(len(answer_terms & evidence_terms) / max(len(answer_terms), 1))
        total_input_tokens += max(1, (len(question) + len(evidence)) // 4)

    count = len(recalls)
    if not count:
        raise RuntimeError("No evaluation requests succeeded")
    cost_rate = os.getenv("COST_PER_1K_INPUT_TOKENS")
    cost = total_input_tokens / 1000 * float(cost_rate) if cost_rate else None
    return {
        "examples": count,
        "failed_requests": failures,
        f"recall@{k}": round(statistics.mean(recalls), 4),
        "mrr": round(statistics.mean(reciprocal_ranks), 4),
        "citation_correctness": round(statistics.mean(citation_correct), 4),
        "groundedness_lexical_proxy": round(statistics.mean(groundedness), 4),
        "latency_seconds_mean": round(statistics.mean(latencies), 3),
        "latency_seconds_p95": round(sorted(latencies)[min(len(latencies) - 1, int(0.95 * len(latencies)))], 3),
        "estimated_input_cost": round(cost, 6) if cost is not None else "not configured",
        "cost_note": "Input tokens and configured rate are estimates; provider usage accounting is not included.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("eval/dataset.json"))
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.dataset, args.api, args.k), indent=2))


if __name__ == "__main__":
    main()
