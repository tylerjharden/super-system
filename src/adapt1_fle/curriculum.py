"""Versioned curriculum loading and resumable job planning."""

from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class BenchmarkArm(StrEnum):
    BASELINE = "baseline"
    COLD_ONLINE = "cold_online"
    WARM_FROZEN = "warm_frozen"
    DOMAIN_ONLY = "domain_only"
    MEMORY_ONLY = "memory_only"


class CurriculumStage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    episodes_per_task: int = Field(ge=1, le=100)
    tasks: list[str] = Field(min_length=1)


class Curriculum(BaseModel):
    model_config = ConfigDict(extra="forbid")

    revision: str
    description: str
    stages: list[CurriculumStage] = Field(min_length=1)
    held_out_tasks: list[str] = Field(default_factory=list)
    transfer_tasks: list[str] = Field(default_factory=list)
    benchmark_arms: list[BenchmarkArm] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_unique_tasks(self) -> Curriculum:
        training = [task for stage in self.stages for task in stage.tasks]
        if len(training) != len(set(training)):
            raise ValueError("training tasks must appear in exactly one curriculum stage")
        overlap = set(training) & set(self.held_out_tasks)
        if overlap:
            raise ValueError(f"held-out tasks also appear in training: {sorted(overlap)}")
        return self

    def jobs(self) -> list[CurriculumJob]:
        result: list[CurriculumJob] = []
        ordinal = 0
        for stage in self.stages:
            for task in stage.tasks:
                for episode_index in range(stage.episodes_per_task):
                    result.append(
                        CurriculumJob(
                            job_id=(
                                f"curriculum-v{self.revision}-{ordinal:04d}-"
                                f"{task}-episode-{episode_index:02d}"
                            ),
                            ordinal=ordinal,
                            stage=stage.name,
                            env_id=task,
                            episode_index=episode_index,
                        )
                    )
                    ordinal += 1
        return result


class CurriculumJob(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    job_id: str
    ordinal: int = Field(ge=0)
    stage: str
    env_id: str
    episode_index: int = Field(ge=0)


def load_curriculum(path: str | Path) -> Curriculum:
    parsed = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise ValueError(f"curriculum must be a mapping: {path}")
    return Curriculum.model_validate(parsed)


def completed_curriculum_jobs(ledger_root: str | Path) -> set[str]:
    """Find jobs with a successful or trajectory-limit completion event."""

    completed: set[str] = set()
    root = Path(ledger_root)
    if not root.exists():
        return completed
    for run_dir in root.iterdir():
        if not run_dir.is_dir():
            continue
        manifest_path = run_dir / "manifest.json"
        events_path = run_dir / "events.jsonl"
        if not manifest_path.exists() or not events_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        job_id = manifest.get("curriculum_job_id")
        if not isinstance(job_id, str):
            continue
        if _has_completion(events_path):
            completed.add(job_id)
    return completed


def pending_curriculum_jobs(
    curriculum: Curriculum,
    ledger_root: str | Path,
) -> list[CurriculumJob]:
    completed = completed_curriculum_jobs(ledger_root)
    return [job for job in curriculum.jobs() if job.job_id not in completed]


def _has_completion(events_path: Path) -> bool:
    with events_path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            envelope = json.loads(line)
            event = envelope.get("event", {})
            if isinstance(event, dict) and event.get("kind") == "completion":
                return event.get("status") in {"success", "trajectory_limit"}
    return False
