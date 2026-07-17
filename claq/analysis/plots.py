"""Plot helpers for rollout comparisons and fixed-history summaries."""

from __future__ import annotations

from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
import numpy as np

from claq.analysis.rollouts import format_stop_sequence


def plot_fixed_history_eval_summary(
    summary_rows: list[dict],
    output_path: str | Path,
    hparam_name: str = "lambda_s",
    hparam_label: str | None = None,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not summary_rows:
        raise ValueError("summary_rows must not be empty")

    hparam_label = hparam_label or hparam_name.replace("_", " ")
    rows = sorted(
        summary_rows,
        key=lambda row: (float("-inf") if row[hparam_name] is None else row[hparam_name], row["run_name"]),
    )
    x = [row[hparam_name] for row in rows]
    mean_acc = [row["mean_acc"] for row in rows]
    std_acc = [row["std_acc"] for row in rows]
    mean_sens = [row["mean_sensitive_query_rate"] for row in rows]
    std_sens = [row["std_sensitive_query_rate"] for row in rows]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    axes[0].errorbar(x, mean_acc, yerr=std_acc, marker="o", linewidth=2, capsize=4, color="#1f77b4")
    axes[0].set_title("Fixed-history accuracy")
    axes[0].set_xlabel(hparam_label)
    axes[0].set_ylabel("Accuracy")
    axes[0].grid(alpha=0.25)

    axes[1].errorbar(x, mean_sens, yerr=std_sens, marker="o", linewidth=2, capsize=4, color="#d62728")
    axes[1].set_title("Fixed-history sensitive query rate")
    axes[1].set_xlabel(hparam_label)
    axes[1].set_ylabel("Sensitive query rate")
    axes[1].grid(alpha=0.25)

    axes[2].errorbar(
        mean_sens,
        mean_acc,
        xerr=std_sens,
        yerr=std_acc,
        fmt="o",
        linewidth=1.5,
        capsize=4,
        color="#2e8b57",
    )
    axes[2].set_title("Fixed-history trade-off")
    axes[2].set_xlabel("Sensitive query rate")
    axes[2].set_ylabel("Accuracy")
    axes[2].grid(alpha=0.25)
    for row in rows:
        axes[2].annotate(
            f"{row[hparam_name]:.2f}",
            (row["mean_sensitive_query_rate"], row["mean_acc"]),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            fontsize=9,
        )

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_lambda_tradeoff_summary(
    summary_rows: list[dict],
    output_path: str | Path,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not summary_rows:
        raise ValueError("summary_rows must not be empty")

    rows = sorted(summary_rows, key=lambda row: float(row["lambda_s"]))
    validation_metrics = "validation_acc" in rows[0]
    acc_key = "validation_acc" if validation_metrics else "test_acc"
    sens_key = "validation_sens_q_rate" if validation_metrics else "test_sens_q_rate"
    acc_ci_key = "validation_acc_ci95" if validation_metrics else "test_acc_ci95"
    sens_ci_key = (
        "validation_sens_q_rate_ci95" if validation_metrics else "test_sens_q_rate_ci95"
    )
    lambdas = [float(row["lambda_s"]) for row in rows]
    acc = [float(row[acc_key]) for row in rows]
    sens = [float(row[sens_key]) for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))

    acc_ci = [float(row.get(acc_ci_key, 0.0)) for row in rows]
    sens_ci = [float(row.get(sens_ci_key, 0.0)) for row in rows]

    axes[0].errorbar(
        sens,
        acc,
        xerr=sens_ci,
        yerr=acc_ci,
        marker="o",
        linewidth=2.2,
        capsize=3,
        color="#3a6ea5",
    )
    for row in rows:
        label = "baseline" if float(row["lambda_s"]) == 0.0 else f"lambda_s={row['lambda_s']:.2f}"
        axes[0].annotate(
            label,
            (float(row[sens_key]), float(row[acc_key])),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=9.5,
        )
    axes[0].set_title("Utility-sensitivity trade-off")
    axes[0].set_xlabel("Sensitive query rate")
    axes[0].set_ylabel("Accuracy")
    axes[0].grid(alpha=0.25)

    axes[1].errorbar(
        lambdas,
        acc,
        yerr=acc_ci,
        marker="o",
        linewidth=2.2,
        capsize=3,
        label="Accuracy",
        color="#3a6ea5",
    )
    axes[1].errorbar(
        lambdas,
        sens,
        yerr=sens_ci,
        marker="s",
        linewidth=2.2,
        capsize=3,
        label="Sensitive query rate",
        color="#b58b00",
    )
    axes[1].set_title("Lambda sweep")
    axes[1].set_xlabel("lambda_s")
    axes[1].set_ylabel("Metric value")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)

    plt.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _format_metric_path(
    row: dict,
    stop: dict,
    *,
    value_key: str,
    empty_key: str,
    initial_key: str,
    prefix: str,
    max_items: int,
    wrap_width: int,
) -> str:
    parts = []
    if empty_key in stop:
        parts.append(f"empty:{stop[empty_key]:.2f}")
    if row["initial_history_size"] > 0 and initial_key in stop:
        parts.append(f"init:{stop[initial_key]:.2f}")

    query_states = stop["states"][1:]
    for state in query_states[:max_items]:
        parts.append(f"q{state['after_queries']}:{state[value_key]:.2f}")
    if len(query_states) > max_items:
        parts.append("...")

    return textwrap.fill(
        f"{prefix}: " + " -> ".join(parts),
        width=wrap_width,
        subsequent_indent="    ",
    )


def _wrap_block(label: str, row: dict, key: str, wrap_width: int = 64, seq_items: int = 6, conf_items: int = 8) -> str:
    stop = row[key]
    target_name = stop.get("positive_class_name")
    if target_name is None:
        metric_lines = [
            _format_metric_path(
                row,
                stop,
                value_key="confidence",
                empty_key="empty_confidence",
                initial_key="initial_confidence",
                prefix="conf path",
                max_items=conf_items,
                wrap_width=wrap_width,
            )
        ]
    else:
        metric_lines = [
            _format_metric_path(
                row,
                stop,
                value_key="positive_prob",
                empty_key="empty_positive_prob",
                initial_key="initial_positive_prob",
                prefix=f"p({target_name}) path",
                max_items=conf_items,
                wrap_width=wrap_width,
            ),
            _format_metric_path(
                row,
                stop,
                value_key="confidence",
                empty_key="empty_confidence",
                initial_key="initial_confidence",
                prefix="conf path",
                max_items=conf_items,
                wrap_width=wrap_width,
            ),
        ]
    lines = [
        label,
        (
            f"q={stop['queries_asked']} | sens={stop['sensitive_steps']} | "
            f"stop={stop['stop_reason']} | pred={stop['final_pred_name']}"
        ),
        *metric_lines,
        textwrap.fill(
            f"path: {format_stop_sequence(stop['sequence'], max_items=seq_items)}",
            width=wrap_width,
            subsequent_indent="    ",
        ),
    ]
    return "\n".join(lines)


def plot_rollout_comparisons(
    records: list[dict],
    raw_dataset,
    output_path: str | Path,
    title_prefix: str,
    box_fontsize: float = 16,
    title_fontsize: float = 16,
    text_wrap_width: int = 72,
    path_items: int = 10,
    confidence_items: int = 10,
    column_wspace: float = 0.16,
    left_margin: float = 0.012,
    right_margin: float = 0.999,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        raise ValueError("records must not be empty")
    _ = title_prefix  # Kept for backward compatibility; not shown in the figure.

    text_blocks = [
        (
            _wrap_block(
                "Baseline",
                row,
                "baseline",
                wrap_width=text_wrap_width,
                seq_items=path_items,
                conf_items=confidence_items,
            ),
            _wrap_block(
                "CLAQ",
                row,
                "claq",
                wrap_width=text_wrap_width,
                seq_items=path_items,
                conf_items=confidence_items,
            ),
        )
        for row in records
    ]
    # A fixed row height causes long confidence and query paths to extend into
    # the next case. Size each row from its rendered line count instead.
    line_height_inches = box_fontsize * 1.28 / 72.0
    row_heights = [
        max(3.8, 1.0 + max(base.count("\n"), claq.count("\n")) * line_height_inches)
        for base, claq in text_blocks
    ]
    fig, axes = plt.subplots(
        len(records),
        3,
        figsize=(25.0, sum(row_heights)),
        gridspec_kw={
            "width_ratios": [0.72, 2.55, 2.55],
            "height_ratios": row_heights,
        },
    )
    if len(records) == 1:
        axes = np.array([axes])

    for (ax_img, ax_base, ax_claq), row, (baseline_text, claq_text) in zip(
        axes, records, text_blocks
    ):
        image, _ = raw_dataset[row["sample_idx"]]
        ax_img.imshow(image)
        ax_img.axis("off")
        start_text = ", ".join(row["initial_history"]) if row["initial_history"] else "(empty)"
        ax_img.set_title(
            f"sample {row['sample_idx']}\ntrue: {row['label_name']}\n"
            f"initial knowledge ({row['initial_history_size']}): {start_text}",
            fontsize=title_fontsize,
            pad=8,
        )

        for ax in (ax_base, ax_claq):
            ax.axis("off")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)

        ax_base.text(
            0.0,
            0.98,
            baseline_text,
            fontsize=box_fontsize,
            va="top",
            ha="left",
            linespacing=1.28,
            family="monospace",
            transform=ax_base.transAxes,
            bbox=dict(boxstyle="round,pad=0.40", facecolor="#fff8dc", edgecolor="#b58b00", linewidth=1.1),
        )
        ax_claq.text(
            0.0,
            0.98,
            claq_text,
            fontsize=box_fontsize,
            va="top",
            ha="left",
            linespacing=1.28,
            family="monospace",
            transform=ax_claq.transAxes,
            bbox=dict(boxstyle="round,pad=0.40", facecolor="#eef5ff", edgecolor="#3a6ea5", linewidth=1.1),
        )

    plt.subplots_adjust(
        left=left_margin,
        right=right_margin,
        top=0.98,
        bottom=0.02,
        wspace=column_wspace,
        hspace=0.32,
    )
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path
