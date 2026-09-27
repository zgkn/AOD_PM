#!/usr/bin/env python3
"""Append the per-year download summary as a markdown table to the GH Actions job summary."""
import json
import os
import sys
from pathlib import Path


def main() -> int:
    summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_file:
        print("GITHUB_STEP_SUMMARY not set; skipping.", file=sys.stderr)
        return 0

    lines = [
        "## CAMS EAC4 AOD download summary",
        "",
        "| Year | Status |",
        "|------|--------|",
    ]

    summary_json = Path("output/summary.json")
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
