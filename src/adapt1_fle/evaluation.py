"""Aggregate controlled benchmark results from immutable run ledgers."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from statistics import fmean
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from adapt1_fle.ledger import RunLedger, summarize_run
from adapt1_fle.models import RunMetrics


class ArmMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    arm: str
    sample_count: int = Field(ge=0)
    tasks: list[str]
    success_count: int = Field(ge=0)
    pass_rate: float = Field(ge=0, le=1)
    pass_rate_ci95_low: float = Field(ge=0, le=1)
    pass_rate_ci95_high: float = Field(ge=0, le=1)
    mean_final_score: float
    mean_automated_score: float
    mean_score_auc: float
    mean_steps: float
    execution_error_rate: float = Field(ge=0, le=1)
    adapt_selection_rate: float = Field(ge=0, le=1)
    fallback_rate: float = Field(ge=0, le=1)
    abstention_rate: float = Field(ge=0, le=1)
    operational_failure_count: int = Field(ge=0)
    operational_failure_rate: float = Field(ge=0, le=1)
    ambiguous_write_count: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    total_model_latency_seconds: float = Field(ge=0)
    total_adapt_latency_seconds: float = Field(ge=0)


class BenchmarkSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    generated_from: str
    experiment_id: str | None
    comparison_fingerprint: str | None
    total_runs: int = Field(ge=0)
    arms: dict[str, ArmMetrics]
    task_arms: dict[str, dict[str, ArmMetrics]]
    runs: list[RunMetrics]


def build_benchmark_summary(
    ledger_root: str | Path,
    *,
    experiment_id: str | None = None,
) -> BenchmarkSummary:
    root = Path(ledger_root)
    grouped: defaultdict[str, list[tuple[RunMetrics, dict[str, Any]]]] = defaultdict(list)
    task_grouped: defaultdict[tuple[str, str], list[tuple[RunMetrics, dict[str, Any]]]] = (
        defaultdict(list)
    )
    all_metrics: list[RunMetrics] = []
    experiment_ids: set[str] = set()
    fingerprints: set[str] = set()
    if root.exists():
        for run_dir in sorted(root.iterdir()):
            if not run_dir.is_dir() or not (run_dir / "manifest.json").exists():
                continue
            manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
            manifest_experiment = manifest.get("experiment_id")
            arm_value = manifest.get("benchmark_arm")
            if not isinstance(manifest_experiment, str) or not isinstance(arm_value, str):
                continue
            if experiment_id is not None and manifest_experiment != experiment_id:
                continue
            fingerprint = manifest.get("comparison_fingerprint")
            if not isinstance(fingerprint, str):
                raise ValueError(f"benchmark run lacks comparison_fingerprint: {run_dir}")
            ledger = RunLedger.open(run_dir)
            metrics = summarize_run(ledger)
            arm = arm_value
            grouped[arm].append((metrics, manifest))
            task_grouped[(str(manifest.get("env_id", "unknown")), arm)].append((metrics, manifest))
            all_metrics.append(metrics)
            experiment_ids.add(manifest_experiment)
            fingerprints.add(fingerprint)

    if experiment_id is None and len(experiment_ids) > 1:
        raise ValueError(
            "ledger root contains multiple experiments; select one with --experiment-id"
        )
    if len(fingerprints) > 1:
        raise ValueError("experiment mixes incompatible comparison fingerprints")
    resolved_experiment = experiment_id or next(iter(experiment_ids), None)
    resolved_fingerprint = next(iter(fingerprints), None)

    arms = {arm: _summarize_arm(arm, values) for arm, values in sorted(grouped.items())}
    task_arms: defaultdict[str, dict[str, ArmMetrics]] = defaultdict(dict)
    for (task, arm), values in sorted(task_grouped.items()):
        task_arms[task][arm] = _summarize_arm(arm, values)
    return BenchmarkSummary(
        generated_from=str(root),
        experiment_id=resolved_experiment,
        comparison_fingerprint=resolved_fingerprint,
        total_runs=len(all_metrics),
        arms=arms,
        task_arms=dict(task_arms),
        runs=all_metrics,
    )


def _summarize_arm(
    arm: str,
    values: list[tuple[RunMetrics, dict[str, Any]]],
) -> ArmMetrics:
    metrics = [item[0] for item in values]
    manifests = [item[1] for item in values]
    sample_count = len(metrics)
    success_count = sum(item.success for item in metrics)
    pass_rate = success_count / sample_count if sample_count else 0.0
    ci_low, ci_high = wilson_interval(success_count, sample_count)
    total_steps = sum(item.steps for item in metrics)
    tasks = sorted(
        {
            str(manifest.get("env_id"))
            for manifest in manifests
            if manifest.get("env_id") is not None
        }
    )
    return ArmMetrics(
        arm=arm,
        sample_count=sample_count,
        tasks=tasks,
        success_count=success_count,
        pass_rate=pass_rate,
        pass_rate_ci95_low=ci_low,
        pass_rate_ci95_high=ci_high,
        mean_final_score=_mean(item.final_score for item in metrics),
        mean_automated_score=_mean(item.final_automated_score for item in metrics),
        mean_score_auc=_mean(item.score_auc for item in metrics),
        mean_steps=_mean(float(item.steps) for item in metrics),
        execution_error_rate=(
            sum(item.execution_error_count for item in metrics) / total_steps
            if total_steps
            else 0.0
        ),
        adapt_selection_rate=(
            sum(item.adapt_selection_count for item in metrics) / total_steps
            if total_steps
            else 0.0
        ),
        fallback_rate=(
            sum(item.fallback_count for item in metrics) / total_steps if total_steps else 0.0
        ),
        abstention_rate=(
            sum(item.abstention_count for item in metrics) / total_steps if total_steps else 0.0
        ),
        operational_failure_count=sum(item.operational_failure for item in metrics),
        operational_failure_rate=(
            sum(item.operational_failure for item in metrics) / sample_count
            if sample_count
            else 0.0
        ),
        ambiguous_write_count=sum(item.ambiguous_write_count for item in metrics),
        total_tokens=sum(item.token_count for item in metrics),
        total_model_latency_seconds=sum(item.model_latency_seconds for item in metrics),
        total_adapt_latency_seconds=sum(item.adapt_latency_seconds for item in metrics),
    )


def wilson_interval(successes: int, total: int, *, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return 0.0, 0.0
    proportion = successes / total
    denominator = 1 + z**2 / total
    center = (proportion + z**2 / (2 * total)) / denominator
    margin = (
        z * math.sqrt(proportion * (1 - proportion) / total + z**2 / (4 * total**2)) / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def _mean(values: Iterable[float]) -> float:
    materialized = list(values)
    return fmean(materialized) if materialized else 0.0
