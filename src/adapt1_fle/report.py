"""Machine-readable and human-readable benchmark reporting."""

from __future__ import annotations

import json
import os
from pathlib import Path

from adapt1_fle.evaluation import BenchmarkSummary


def write_report(
    summary: BenchmarkSummary,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    json_path = destination / "benchmark.json"
    markdown_path = destination / "benchmark.md"
    _atomic_write(
        json_path,
        json.dumps(summary.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
    )
    _atomic_write(markdown_path, render_markdown(summary))
    return json_path, markdown_path


def render_markdown(summary: BenchmarkSummary) -> str:
    lines = [
        "# Adapt-1 x FLE benchmark",
        "",
        f"Runs analyzed: **{summary.total_runs}**",
        "",
        "| Arm | N | Pass rate (95% CI) | Final score | Automated score | "
        "Execution errors | Adapt selections | Fallbacks |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm, metrics in summary.arms.items():
        lines.append(
            f"| {arm} | {metrics.sample_count} | "
            f"{metrics.pass_rate:.1%} "
            f"({metrics.pass_rate_ci95_low:.1%}-{metrics.pass_rate_ci95_high:.1%}) | "
            f"{metrics.mean_final_score:.3f} | "
            f"{metrics.mean_automated_score:.3f} | "
            f"{metrics.execution_error_rate:.1%} | "
            f"{metrics.adapt_selection_rate:.1%} | "
            f"{metrics.fallback_rate:.1%} |"
        )

    lines.extend(
        [
            "",
            "## Integrity notes",
            "",
            f"- Results aggregate immutable run ledgers under `{summary.generated_from}`.",
            "- A fallback is application/controller behavior, not learned Adapt-1 behavior.",
            "- Adapt confidence and support are not interpreted as calibrated correctness.",
            "- Exposure, held-out status, model, FLE version, Domain revision, and raw "
            "interaction evidence remain in each run manifest and event stream.",
            "- This report does not assert state of the art. Such a claim requires matched "
            "information conditions, repeated trials, and external baseline results.",
            "",
        ]
    )
    return "\n".join(lines)


def _atomic_write(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
