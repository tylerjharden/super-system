from pathlib import Path

import pytest
from fle.env.gym_env.registry import list_available_environments

from adapt1_fle.curriculum import (
    BlockedCurriculumError,
    load_curriculum,
    pending_curriculum_jobs,
)
from adapt1_fle.evaluation import (
    ExperimentCell,
    ExperimentPlan,
    build_benchmark_summary,
    wilson_interval,
    write_experiment_plan,
)
from adapt1_fle.ledger import RunLedger
from adapt1_fle.models import RunCompletion
from adapt1_fle.report import write_report


def test_curriculum_resume_skips_completed_jobs(tmp_path: Path) -> None:
    curriculum_path = tmp_path / "curriculum.yaml"
    curriculum_path.write_text(
        """
revision: "test"
description: small
stages:
  - name: basics
    episodes_per_task: 2
    tasks: [iron_ore_throughput]
held_out_tasks: [iron_plate_throughput]
transfer_tasks: []
benchmark_arms: [baseline, warm_frozen]
""",
        encoding="utf-8",
    )
    curriculum = load_curriculum(curriculum_path)
    first = curriculum.jobs()[0]
    ledger = RunLedger.create(
        tmp_path / "runs",
        "run-1",
        {
            "run_id": "run-1",
            "mode": "train",
            "curriculum_job_id": first.job_id,
        },
    )
    ledger.append(
        RunCompletion(
            run_id="run-1",
            episode_id="episode-1",
            status="trajectory_limit",
            steps_completed=1,
            success=False,
            final_score=1,
            final_automated_score=1,
        )
    )

    pending = pending_curriculum_jobs(curriculum, tmp_path / "runs")

    assert len(curriculum.jobs()) == 2
    assert [job.episode_index for job in pending] == [1]


def test_curriculum_validates_task_registry() -> None:
    loaded = load_curriculum(
        "configs/curriculum.v1.yaml",
        available_tasks=list_available_environments(),
    )
    assert "sufuric_acid_throughput" in loaded.held_out_tasks

    with pytest.raises(ValueError, match="unknown FLE tasks"):
        load_curriculum(
            "configs/curriculum.v1.yaml",
            available_tasks={"iron_ore_throughput"},
        )


def test_benchmark_summary_and_report_keep_arms_separate(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    _experiment_plan(runs)
    _completed_run(runs, "baseline-1", "baseline", success=False, score=2)
    _completed_run(runs, "warm-1", "warm_frozen", success=True, score=16)

    summary = build_benchmark_summary(runs)
    json_path, markdown_path = write_report(summary, tmp_path / "report")

    assert summary.total_runs == 2
    assert summary.arms["baseline"].pass_rate == 0
    assert summary.arms["warm_frozen"].pass_rate == 1
    assert summary.arms["warm_frozen"].mean_final_score == 16
    assert summary.task_arms["iron_plate_throughput"]["warm_frozen"].pass_rate == 1
    assert json_path.exists()
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "| baseline |" in markdown
    assert "| warm_frozen |" in markdown
    assert "## Per-task results" in markdown
    assert "does not assert state of the art" in markdown


def test_report_rejects_mixed_comparison_protocols(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    _experiment_plan(runs)
    _completed_run(runs, "baseline-1", "baseline", success=False, score=2)
    _completed_run(runs, "warm-1", "warm_frozen", success=True, score=16)
    manifest = runs / "warm-1" / "manifest.json"
    changed = manifest.read_text(encoding="utf-8").replace(
        "fingerprint-1",
        "different-fingerprint",
    )
    manifest.write_text(changed, encoding="utf-8")

    with pytest.raises(ValueError, match="incompatible comparison fingerprints"):
        build_benchmark_summary(runs)


def test_report_marks_missing_planned_cells_incomplete(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    _experiment_plan(runs)
    _completed_run(runs, "baseline-1", "baseline", success=False, score=2)

    summary = build_benchmark_summary(runs)

    assert summary.experiment_complete is False
    assert summary.missing_cells == ["warm_frozen/iron_plate_throughput/0"]


def test_curriculum_refuses_automatic_retry_after_failure(tmp_path: Path) -> None:
    curriculum = load_curriculum("configs/curriculum.v1.yaml")
    first = curriculum.jobs()[0]
    ledger = RunLedger.create(
        tmp_path,
        "failed-run",
        {
            "run_id": "failed-run",
            "mode": "train",
            "curriculum_job_id": first.job_id,
        },
    )
    ledger.append(
        RunCompletion(
            run_id="failed-run",
            episode_id="episode-1",
            status="failed",
            steps_completed=1,
            success=False,
            final_score=0,
            final_automated_score=0,
            error="ambiguous feedback",
        )
    )

    with pytest.raises(BlockedCurriculumError, match=first.job_id):
        pending_curriculum_jobs(curriculum, tmp_path)


def test_curriculum_blocks_incomplete_wal_run(tmp_path: Path) -> None:
    curriculum = load_curriculum("configs/curriculum.v1.yaml")
    first = curriculum.jobs()[0]
    ledger = RunLedger.create(
        tmp_path,
        "crashed-run",
        {
            "run_id": "crashed-run",
            "mode": "train",
            "curriculum_job_id": first.job_id,
        },
    )
    ledger.append(
        {
            "kind": "decision_started",
            "ids": {"interaction_id": "possibly-mutating"},
        }
    )

    with pytest.raises(BlockedCurriculumError, match=first.job_id):
        pending_curriculum_jobs(curriculum, tmp_path)


def test_wilson_interval_is_bounded() -> None:
    low, high = wilson_interval(8, 10)

    assert 0 <= low < 0.8 < high <= 1
    assert wilson_interval(0, 0) == (0.0, 0.0)


def _completed_run(
    root: Path,
    run_id: str,
    arm: str,
    *,
    success: bool,
    score: float,
) -> None:
    ledger = RunLedger.create(
        root,
        run_id,
        {
            "run_id": run_id,
            "mode": "frozen" if arm != "baseline" else "baseline",
            "benchmark_arm": arm,
            "experiment_id": "experiment-1",
            "comparison_fingerprint": "fingerprint-1",
            "env_id": "iron_plate_throughput",
            "evaluation_episode": 0,
        },
    )
    ledger.append(
        RunCompletion(
            run_id=run_id,
            episode_id=f"{run_id}-episode",
            status="success" if success else "trajectory_limit",
            steps_completed=0,
            success=success,
            final_score=score,
            final_automated_score=score,
        )
    )


def _experiment_plan(root: Path) -> None:
    write_experiment_plan(
        root,
        ExperimentPlan(
            experiment_id="experiment-1",
            comparison_fingerprint="fingerprint-1",
            cells=[
                ExperimentCell(
                    arm="baseline",
                    env_id="iron_plate_throughput",
                    episode=0,
                ),
                ExperimentCell(
                    arm="warm_frozen",
                    env_id="iron_plate_throughput",
                    episode=0,
                ),
            ],
        ),
    )
