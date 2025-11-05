#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def run_tests(stage_path: Path, output_file: Path, timeout: int = 300) -> bool:
    cmd = [
        "uv",
        "run",
        "pytest",
        str(stage_path),
        "--json-report",
        f"--json-report-file={output_file}",
        "--json-report-indent=2",
        f"--timeout={timeout}",
        "-v",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode == 0
    except subprocess.CalledProcessError:
        return False


def grade_stage(results_file: Path) -> dict[str, Any]:
    if not results_file.exists():
        return {
            "stage": "unknown",
            "score": 0.0,
            "passed": 0,
            "total": 0,
            "failed": 0,
            "skipped": 0,
            "error": "Results file not found",
        }

    with open(results_file) as f:
        data = json.load(f)

    summary = data.get("summary", {})
    total = summary.get("total", 0)
    passed = summary.get("passed", 0)
    failed = summary.get("failed", 0)
    skipped = summary.get("skipped", 0)

    score = (passed / total * 100) if total > 0 else 0.0

    return {
        "stage": data.get("root", "unknown"),
        "score": round(score, 2),
        "passed": passed,
        "total": total,
        "failed": failed,
        "skipped": skipped,
        "duration": round(data.get("duration", 0.0), 2),
    }


def print_report(grade_data: dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print(f"AUTOGRADER REPORT: {grade_data['stage']}")
    print("=" * 60)
    print(f"Score:    {grade_data['score']:.2f}%")
    print(f"Passed:   {grade_data['passed']}/{grade_data['total']}")
    print(f"Failed:   {grade_data['failed']}")
    print(f"Skipped:  {grade_data['skipped']}")
    print(f"Duration: {grade_data['duration']:.2f}s")
    print("=" * 60)

    if grade_data["score"] == 100.0:
        print("🎉 Perfect score! All tests passed!")
    elif grade_data["score"] >= 80.0:
        print("✅ Great work! Most tests passed.")
    elif grade_data["score"] >= 50.0:
        print("⚠️  Keep going! You're halfway there.")
    else:
        print("❌ Keep working on it. Review the test output above.")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description="Autograde a curriculum stage")
    parser.add_argument(
        "stage",
        type=str,
        help="Stage to grade (e.g., s01_number_systems_and_bits or stages/s01_*)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=".grading_results.json",
        help="Output file for JSON results (default: .grading_results.json)",
    )
    parser.add_argument(
        "--timeout",
        "-t",
        type=int,
        default=300,
        help="Timeout per test in seconds (default: 300)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output only JSON (no formatted report)",
    )

    args = parser.parse_args()

    stage_path = Path(args.stage)
    if not stage_path.exists():
        stage_path = Path("stages") / args.stage
    if not stage_path.exists():
        print(f"Error: Stage path '{args.stage}' not found", file=sys.stderr)
        return 1

    output_file = Path(args.output)

    if not args.json:
        print(f"Running tests for {stage_path}...")

    run_tests(stage_path, output_file, args.timeout)
    grade_data = grade_stage(output_file)

    if args.json:
        print(json.dumps(grade_data, indent=2))
    else:
        print_report(grade_data)

    return 0 if grade_data["score"] == 100.0 else 1


if __name__ == "__main__":
    sys.exit(main())
