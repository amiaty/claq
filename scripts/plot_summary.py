#!/usr/bin/env python3
"""Plot multi-seed CLAQ trade-offs from ``summary.json``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    payload = json.loads(args.summary.read_text(encoding="utf-8"))
    rows = []
    for key, metrics in payload["by_lambda"].items():
        rows.append((float(key), metrics))
    rows.sort(key=lambda item: item[0])
    if not rows:
        raise ValueError("No lambda summaries were found")

    query_metric = "designated_query_rate"
    x = [row[1][query_metric]["mean"] for row in rows]
    xerr = [row[1][query_metric]["std"] for row in rows]
    y = [row[1]["accuracy"]["mean"] for row in rows]
    yerr = [row[1]["accuracy"]["std"] for row in rows]

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.errorbar(x, y, xerr=xerr, yerr=yerr, marker="o", capsize=3)
    for (lambda_value, _), x_value, y_value in zip(rows, x, y, strict=True):
        ax.annotate(f"$\\lambda_s={lambda_value:g}$", (x_value, y_value),
                    textcoords="offset points", xytext=(4, 6))
    ax.set_xlabel("Designated-query rate")
    ax.set_ylabel("Target accuracy")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220, bbox_inches="tight")
    print(args.output)


if __name__ == "__main__":
    main()
