#!/usr/bin/env python3
"""
PLM RAG Evaluation Pipeline
Runs test cases against the live chat API and logs metrics to MLflow.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
import mlflow

EVAL_DIR = Path(__file__).parent
DATASET_PATH = EVAL_DIR / "dataset.json"
RESULTS_PATH = EVAL_DIR / "results.json"
DEFAULT_API_URL = "http://localhost:8889"
MLFLOW_TRACKING_URI = "http://localhost:5000"
EXPERIMENT_NAME = "plm-rag-eval"


def load_dataset(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def call_api(api_url: str, query: str, speaker: str | None, timeout: int = 30) -> dict | None:
    """Call the chat API. Returns response dict or None on error."""
    payload = {"query": query, "speaker": speaker or ""}
    try:
        resp = requests.post(
            f"{api_url}/api/chat",
            json=payload,
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        print(f"  API error: {e}", file=sys.stderr)
        return None


def compute_case_metrics(case: dict, response: dict | None) -> dict:
    """Compute per-case metrics from the API response."""
    result = {
        "id": case["id"],
        "category": case["category"],
        "query": case["query"],
        "expected_episode_ids": case["expected_episode_ids"],
        "expected_speaker": case.get("expected_speaker"),
        "expected_relevant": case["expected_relevant"],
    }

    if response is None:
        result.update({
            "hit_at_5": 0, "hit_at_1": 0, "speaker_match": None,
            "relevance_score": 0.0, "source_correct": 0, "source": None,
            "returned_episodes": [], "first_rank": None, "error": True,
        })
        return result

    chunks = response.get("chunks", [])
    source = response.get("source", "")
    expected_ids = set(case["expected_episode_ids"])
    is_negative = not case["expected_relevant"]

    # Episode IDs returned in order
    returned_episode_ids = [c.get("youtube_id") for c in chunks]
    result["returned_episodes"] = returned_episode_ids
    result["source"] = source
    result["error"] = False

    # hit@5: any of top-5 chunks from expected episode
    top5_ids = set(returned_episode_ids[:5])
    result["hit_at_5"] = 1 if (expected_ids & top5_ids) else 0

    # hit@1: top chunk from expected episode
    result["hit_at_1"] = 1 if (returned_episode_ids and returned_episode_ids[0] in expected_ids) else 0

    # First rank of correct episode (for MRR)
    first_rank = None
    for i, eid in enumerate(returned_episode_ids[:5], 1):
        if eid in expected_ids:
            first_rank = i
            break
    result["first_rank"] = first_rank

    # Speaker match
    expected_speaker = case.get("expected_speaker")
    if expected_speaker:
        chunk_speakers = [c.get("speaker", "") for c in chunks if c.get("speaker")]
        if chunk_speakers:
            match_count = sum(1 for s in chunk_speakers if expected_speaker.lower() in s.lower())
            result["speaker_match"] = 1 if match_count > 0 else 0
        else:
            result["speaker_match"] = 0
    else:
        result["speaker_match"] = None

    # Relevance score (avg similarity from API)
    scores = [c.get("score", 0.0) for c in chunks]
    result["relevance_score"] = sum(scores) / len(scores) if scores else 0.0

    # Source correct: chunks = pgvector (good), episodes = fallback
    if is_negative:
        # For negative cases, low relevance is good regardless of source
        result["source_correct"] = 1
    else:
        result["source_correct"] = 1 if source == "chunks" else 0

    return result


def compute_aggregate_metrics(results: list[dict]) -> dict:
    """Compute aggregate metrics across all test cases."""
    # Filter by category for relevant metrics
    relevant_cases = [r for r in results if r["expected_relevant"] and not r.get("error")]
    speaker_cases = [r for r in results if r["category"] == "speaker_attribution" and not r.get("error")]
    negative_cases = [r for r in results if r["category"] == "negative" and not r.get("error")]
    all_valid = [r for r in results if not r.get("error")]

    metrics = {}

    # hit_rate@5
    if relevant_cases:
        metrics["hit_rate_at_5"] = sum(r["hit_at_5"] for r in relevant_cases) / len(relevant_cases)
    else:
        metrics["hit_rate_at_5"] = 0.0

    # hit_rate@1
    if relevant_cases:
        metrics["hit_rate_at_1"] = sum(r["hit_at_1"] for r in relevant_cases) / len(relevant_cases)
    else:
        metrics["hit_rate_at_1"] = 0.0

    # MRR
    if relevant_cases:
        reciprocal_ranks = []
        for r in relevant_cases:
            if r["first_rank"] is not None:
                reciprocal_ranks.append(1.0 / r["first_rank"])
            else:
                reciprocal_ranks.append(0.0)
        metrics["mrr"] = sum(reciprocal_ranks) / len(reciprocal_ranks)
    else:
        metrics["mrr"] = 0.0

    # Speaker accuracy
    if speaker_cases:
        speaker_with_match = [r for r in speaker_cases if r["speaker_match"] is not None]
        if speaker_with_match:
            metrics["speaker_accuracy"] = sum(r["speaker_match"] for r in speaker_with_match) / len(speaker_with_match)
        else:
            metrics["speaker_accuracy"] = 0.0
    else:
        metrics["speaker_accuracy"] = 0.0

    # Negative rejection rate (avg score < 0.3 = correctly rejected)
    if negative_cases:
        rejected = sum(1 for r in negative_cases if r["relevance_score"] < 0.3)
        metrics["negative_rejection_rate"] = rejected / len(negative_cases)
    else:
        metrics["negative_rejection_rate"] = 0.0

    # Avg relevance
    if all_valid:
        metrics["avg_relevance"] = sum(r["relevance_score"] for r in all_valid) / len(all_valid)
    else:
        metrics["avg_relevance"] = 0.0

    # pgvector usage rate
    if all_valid:
        pgvector_count = sum(1 for r in all_valid if r["source"] == "chunks")
        metrics["pgvector_usage_rate"] = pgvector_count / len(all_valid)
    else:
        metrics["pgvector_usage_rate"] = 0.0

    return metrics


def print_summary(results: list[dict], metrics: dict):
    """Print a formatted summary table to stdout."""
    print("\n" + "=" * 100)
    print("PLM RAG EVALUATION RESULTS")
    print("=" * 100)

    # Per-case table
    header = f"{'ID':<18} {'Category':<22} {'Hit@5':>5} {'Hit@1':>5} {'Spkr':>5} {'Relev':>6} {'Src':>8} {'Query':<30}"
    print(header)
    print("-" * 100)

    for r in results:
        if r.get("error"):
            status = "ERROR"
        else:
            status = ""
        spkr = str(r["speaker_match"]) if r["speaker_match"] is not None else "-"
        src = r.get("source", "?") or "?"
        query_short = r["query"][:28] + ".." if len(r["query"]) > 30 else r["query"]
        print(f"{r['id']:<18} {r['category']:<22} {r['hit_at_5']:>5} {r['hit_at_1']:>5} {spkr:>5} {r['relevance_score']:>6.3f} {src:>8} {query_short:<30} {status}")

    # Aggregate metrics
    print("\n" + "=" * 100)
    print("AGGREGATE METRICS")
    print("=" * 100)
    for key, value in sorted(metrics.items()):
        print(f"  {key:<30} {value:.4f}")
    print("=" * 100 + "\n")


def log_to_mlflow(dataset: dict, results: list[dict], metrics: dict, api_url: str):
    """Log everything to MLflow."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=f"eval-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"):
        # Log parameters
        mlflow.log_param("dataset_version", dataset.get("version", "unknown"))
        mlflow.log_param("num_test_cases", len(dataset["test_cases"]))
        mlflow.log_param("api_url", api_url)
        mlflow.log_param("timestamp", datetime.now(timezone.utc).isoformat())

        # Log aggregate metrics
        for key, value in metrics.items():
            mlflow.log_metric(key, value)

        # Log dataset as artifact
        mlflow.log_artifact(str(DATASET_PATH), artifact_path="eval")

        # Log per-case results as artifact
        with open(RESULTS_PATH, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        mlflow.log_artifact(str(RESULTS_PATH), artifact_path="eval")

        # Log per-case results as table (dict of lists format)
        table_data = {
            "id": [], "category": [], "query": [], "hit_at_5": [],
            "hit_at_1": [], "speaker_match": [], "relevance_score": [],
            "source": [], "source_correct": [], "returned_episodes": [],
            "expected_episodes": [], "error": [],
        }
        for r in results:
            table_data["id"].append(r["id"])
            table_data["category"].append(r["category"])
            table_data["query"].append(r["query"])
            table_data["hit_at_5"].append(r["hit_at_5"])
            table_data["hit_at_1"].append(r["hit_at_1"])
            table_data["speaker_match"].append(r["speaker_match"] if r["speaker_match"] is not None else -1)
            table_data["relevance_score"].append(round(r["relevance_score"], 4))
            table_data["source"].append(r.get("source", ""))
            table_data["source_correct"].append(r["source_correct"])
            table_data["returned_episodes"].append(",".join(r.get("returned_episodes", [])[:3]))
            table_data["expected_episodes"].append(",".join(r.get("expected_episode_ids", [])))
            table_data["error"].append(r.get("error", False))
        mlflow.log_table(data=table_data, artifact_file="eval/per_case_results.json")

        run_id = mlflow.active_run().info.run_id
        print(f"MLflow run logged: {run_id}")
        print(f"View at: {MLFLOW_TRACKING_URI}/#/experiments/")


def main():
    parser = argparse.ArgumentParser(description="PLM RAG Evaluation Pipeline")
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help="Chat API base URL")
    parser.add_argument("--dry-run", action="store_true", help="Show test cases without calling the API")
    parser.add_argument("--dataset", default=str(DATASET_PATH), help="Path to dataset JSON")
    parser.add_argument("--no-mlflow", action="store_true", help="Skip MLflow logging")
    args = parser.parse_args()

    dataset = load_dataset(Path(args.dataset))
    test_cases = dataset["test_cases"]
    print(f"Loaded {len(test_cases)} test cases (dataset v{dataset.get('version', '?')})")

    if args.dry_run:
        print("\n[DRY RUN] Test cases that would be evaluated:\n")
        for tc in test_cases:
            speaker_info = f" [speaker={tc['expected_speaker']}]" if tc.get("expected_speaker") else ""
            episodes = ", ".join(tc["expected_episode_ids"]) or "(none)"
            print(f"  {tc['id']:<18} {tc['category']:<22} episodes={episodes}{speaker_info}")
            print(f"    query: {tc['query']}")
        print(f"\nTotal: {len(test_cases)} cases")
        return

    # Run evaluation
    results = []
    total = len(test_cases)
    start_time = time.time()

    for i, case in enumerate(test_cases, 1):
        print(f"[{i:>2}/{total}] {case['id']}: {case['query'][:60]}...", end="", flush=True)
        response = call_api(args.api_url, case["query"], case.get("expected_speaker"))
        case_result = compute_case_metrics(case, response)
        results.append(case_result)

        status = "OK" if not case_result.get("error") else "ERR"
        hit = "HIT" if case_result["hit_at_5"] else ("---" if case["expected_relevant"] else "NEG")
        print(f" [{status}] {hit} (score={case_result['relevance_score']:.3f})")

        # Small delay to avoid hammering the API
        time.sleep(0.2)

    elapsed = time.time() - start_time
    print(f"\nCompleted {total} cases in {elapsed:.1f}s")

    # Compute aggregate metrics
    metrics = compute_aggregate_metrics(results)

    # Print summary
    print_summary(results, metrics)

    # Log to MLflow
    if not args.no_mlflow:
        print("Logging to MLflow...")
        log_to_mlflow(dataset, results, metrics, args.api_url)
    else:
        # Still save results locally
        with open(RESULTS_PATH, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
