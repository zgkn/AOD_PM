#!/usr/bin/env python3
"""Append the per-year download summary as a markdown table to the GH Actions job summary."""
import argparse
import json
import os
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--title", default="CAMS EAC4 AOD download summary")
    parser.add_argument("--summary-path", default="output/summary.json")
    args = parser.parse_args()

    summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_file:
        print("GITHUB_STEP_SUMMARY not set; skipping.", file=sys.stderr)
        return 0

    lines = [
        f"## {args.title}",
        "",
        "| Year | Status |",
        "|------|--------|",
    ]

    summary_json = Path(args.summary_path)
    if summary_json.exists():
        data = json.loads(summary_json.read_text())
        for year, status in sorted(data.items(), key=lambda kv: int(kv[0])):
            lines.append(f"| {year} | {status} |")
    else:
        lines.append("| - | download script produced no summary.json (see logs) |")

    with open(summary_file, "a") as f:
        f.write("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
