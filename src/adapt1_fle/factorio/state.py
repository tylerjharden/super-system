"""Deterministic mapping from rich FLE observations to public learner state."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from fle.env.gym_env.observation import Observation

from adapt1_fle.models import CompactState

WORKING_STATUSES = {"working", "normal", "full_output"}


def compact_state(
    observation: Observation,
    *,
    step: int,
    trajectory_length: int,
    quota: float | None = None,
    production_score: float | None = None,
    automated_production_score: float | None = None,
    last_execution_error: bool = False,
    last_output: str = "",
) -> CompactState:
    """Create a compact pre-consequence state from public FLE fields."""

    task_key = observation.task_info.task_key if observation.task_info else "open_play"
    goal = (
        observation.task_info.goal_description if observation.task_info else "Maximize production"
    )
    score = observation.score if production_score is None else production_score
    automated_score = (
        observation.automated_score
        if automated_production_score is None
        else automated_production_score
    )
    progress = min(max(score / quota, 0.0), 1.0) if quota and quota > 0 else None
    inventory = {
        str(name): int(quantity)
        for name, quantity in sorted(dict(observation.inventory).items())
        if quantity > 0
    }
    entity_counts, status_counts = _entity_summaries(observation.entities)
    flows = observation.flows.to_dict()
    flow_inputs = _number_mapping(flows.get("input"))
    flow_outputs = _number_mapping(flows.get("output"))
    harvested = _number_mapping(flows.get("harvested"))
    crafted = _crafted_mapping(flows.get("crafted"))
    researched_count = sum(
        1 for technology in observation.research.technologies.values() if technology.researched
    )
    error_category = _error_category(last_output) if last_execution_error else None
    target_item = _target_item(task_key)
    phase = infer_phase(
        target_item=target_item,
        progress=progress,
        entity_counts=entity_counts,
        status_counts=status_counts,
        last_execution_error=last_execution_error,
        task_success=bool(observation.task_verification and observation.task_verification.success),
    )

    return CompactState(
        task_key=task_key,
        goal=goal,
        target_item=target_item,
        phase=phase,
        step=step,
        trajectory_length=trajectory_length,
        tick=max(int(observation.game_info.tick), 0),
        elapsed_seconds=max(float(observation.game_info.time), 0.0),
        score=float(score),
        automated_score=float(automated_score),
        quota=quota,
        progress=progress,
        inventory=inventory,
        entity_counts=entity_counts,
        entity_status_counts=status_counts,
        flow_inputs=flow_inputs,
        flow_outputs=flow_outputs,
        crafted=crafted,
        harvested=harvested,
        researched_count=researched_count,
        last_action_error=last_execution_error,
        last_error_category=error_category,
    )


def infer_phase(
    *,
    target_item: str | None,
    progress: float | None,
    entity_counts: dict[str, int],
    status_counts: dict[str, int],
    last_execution_error: bool,
    task_success: bool,
) -> str:
    """Infer only coarse public phase information for strategy routing."""

    if last_execution_error:
        return "debugging"
    non_working = sum(
        count for status, count in status_counts.items() if status not in WORKING_STATUSES
    )
    if non_working > 0 and entity_counts:
        return "debugging"
    if task_success or (progress is not None and progress >= 1.0):
        return "verification"
    if not entity_counts:
        return "bootstrap"
    if not _contains_any(entity_counts, ("mining-drill", "pumpjack")):
        return "extraction"
    if target_item and target_item.endswith("-ore"):
        return "optimization" if progress and progress >= 0.5 else "verification"
    if not _contains_any(entity_counts, ("furnace", "chemical-plant", "oil-refinery")):
        return "smelting"
    if target_item in {"iron-plate", "steel-plate", "plastic-bar", "sulfur", "petroleum-gas"}:
        return "optimization" if progress and progress >= 0.5 else "logistics"
    if not _contains_any(entity_counts, ("assembling-machine",)):
        return "assembly"
    if progress and progress >= 0.5:
        return "optimization"
    return "logistics"


def _entity_summaries(entities: list[dict[str, Any]]) -> tuple[dict[str, int], dict[str, int]]:
    names: Counter[str] = Counter()
    statuses: Counter[str] = Counter()
    for entity in entities:
        name = _entity_value(entity, "name")
        if name:
            names[name] += 1
        status = _entity_value(entity, "status")
        if status:
            statuses[status.lower()] += 1
    return dict(sorted(names.items())), dict(sorted(statuses.items()))


def _entity_value(entity: dict[str, Any], field: str) -> str | None:
    value = entity.get(field, entity.get(f"_{field}"))
    if value is None:
        return None
    enum_name = getattr(value, "name", None)
    if isinstance(enum_name, str):
        return enum_name.replace("_", "-").lower()
    text = str(value)
    if "." in text and text.split(".")[-1]:
        text = text.split(".")[-1]
    return text.strip("'\"<> ").replace("_", "-").lower() or None


def _number_mapping(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, float] = {}
    for key, amount in value.items():
        if isinstance(amount, int | float) and not isinstance(amount, bool):
            result[str(key)] = float(amount)
    return dict(sorted(result.items()))


def _crafted_mapping(value: Any) -> dict[str, float]:
    if not isinstance(value, list):
        return {}
    result: defaultdict[str, float] = defaultdict(float)
    for item in value:
        if not isinstance(item, dict):
            continue
        crafted_count = item.get("crafted_count", 1)
        count = (
            float(crafted_count)
            if isinstance(crafted_count, int | float) and not isinstance(crafted_count, bool)
            else 1.0
        )
        outputs = item.get("outputs")
        if not isinstance(outputs, dict):
            continue
        for name, amount in outputs.items():
            if isinstance(amount, int | float) and not isinstance(amount, bool):
                result[str(name)] += count * float(amount)
    return dict(sorted(result.items()))


def _target_item(task_key: str) -> str | None:
    suffix = "_throughput"
    if task_key.endswith(suffix):
        return task_key[: -len(suffix)].replace("_", "-")
    return None


def _contains_any(entity_counts: dict[str, int], fragments: tuple[str, ...]) -> bool:
    return any(fragment in name for name in entity_counts for fragment in fragments)


def _error_category(output: str) -> str:
    lowered = output.lower()
    categories = (
        "syntaxerror",
        "typeerror",
        "attributeerror",
        "nameerror",
        "valueerror",
        "runtimeerror",
        "assertionerror",
        "timeout",
    )
    for category in categories:
        if category in lowered:
            return category
    return "execution_error"
