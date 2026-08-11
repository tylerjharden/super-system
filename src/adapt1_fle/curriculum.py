"""Versioned curriculum loading and resumable job planning."""

from __future__ import annotations

import json
from collections.abc import Collection
from enum import StrEnum
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class BlockedCurriculumError(RuntimeError):
    """A prior failed/ambiguous job requires operator reconciliation."""


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


def load_curriculum(
    path: str | Path,
    *,
    available_tasks: Collection[str] | None = None,
) -> Curriculum:
    parsed = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise ValueError(f"curriculum must be a mapping: {path}")
    curriculum = Curriculum.model_validate(parsed)
    if available_tasks is not None:
        configured = {
            *(task for stage in curriculum.stages for task in stage.tasks),
            *curriculum.held_out_tasks,
            *curriculum.transfer_tasks,
        }
        missing = sorted(configured - set(available_tasks))
        if missing:
            raise ValueError(f"curriculum contains unknown FLE tasks: {missing}")
    return curriculum


def completed_curriculum_jobs(ledger_root: str | Path) -> set[str]:
    """Find jobs with a successful or trajectory-limit completion event."""

    return {
        job_id
        for job_id, state in curriculum_job_states(ledger_root).items()
        if state == "completed"
    }


def curriculum_job_states(ledger_root: str | Path) -> dict[str, str]:
    states: dict[str, str] = {}
    root = Path(ledger_root)
    if not root.exists():
        return states
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
        run_state = _run_state(events_path)
        previous = states.get(job_id)
        if run_state == "completed" or previous is None:
            states[job_id] = run_state
    return states


def pending_curriculum_jobs(
    curriculum: Curriculum,
    ledger_root: str | Path,
) -> list[CurriculumJob]:
    states = curriculum_job_states(ledger_root)
    blocked = sorted(job.job_id for job in curriculum.jobs() if states.get(job.job_id) == "blocked")
    if blocked:
        raise BlockedCurriculumError(
            "curriculum contains failed or ambiguous jobs requiring reconciliation: "
            + ", ".join(blocked)
        )
    return [job for job in curriculum.jobs() if states.get(job.job_id) != "completed"]


def _run_state(events_path: Path) -> str:
    state = "blocked"
    with events_path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            envelope = json.loads(line)
            event = envelope.get("event", {})
            if not isinstance(event, dict):
                continue
            if event.get("kind") == "interaction" and _interaction_is_ambiguous(event):
                state = "blocked"
            if event.get("kind") == "completion":
                if event.get("status") in {"success", "trajectory_limit"}:
                    return "completed"
                if event.get("status") == "failed":
                    state = "blocked"
    return state


def _interaction_is_ambiguous(event: dict[str, object]) -> bool:
    for key in ("feedback_exchange", "memory_write_exchange"):
        exchange = event.get(key)
        if isinstance(exchange, dict) and exchange.get("ambiguous") is True:
            return True
    return False
