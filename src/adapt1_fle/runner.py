"""Owned FLE trajectory loop with explicit Adapt boundaries."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, Protocol, cast

import gym
from fle.env.gym_env.action import Action
from fle.env.gym_env.observation import Observation
from fle.env.gym_env.observation_formatter import TreeObservationFormatter
from fle.eval.tasks.task_definitions.lab_play.throughput_tasks import THROUGHPUT_TASKS

from adapt1_fle.agent.controller import AdaptiveController
from adapt1_fle.factorio.state import compact_state
from adapt1_fle.models import (
    ExecutionResult,
    InteractionIds,
    JsonObject,
    RunCompletion,
)


class FactorioEnvironment(Protocol):
    """Minimal Gym interface required by the runner."""

    def reset(self) -> Any:
        """Reset and return an initial observation."""

    def step(self, action: Action) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Execute one FLE action."""

    def close(self) -> None:
        """Release local environment resources."""


class TrajectoryRunner:
    """Run one ordered Factorio episode."""

    def __init__(
        self,
        *,
        env_id: str,
        trajectory_length: int,
        controller: AdaptiveController,
        run_id: str,
        episode_id: str,
        run_idx: int = 0,
        environment: FactorioEnvironment | None = None,
    ) -> None:
        self.env_id = env_id
        self.trajectory_length = trajectory_length
        self.controller = controller
        self.run_id = run_id
        self.episode_id = episode_id
        self._owns_environment = environment is None
        self.environment = environment or cast(
            FactorioEnvironment, gym.make(env_id, run_idx=run_idx)
        )
        self.formatter = TreeObservationFormatter(
            include_research=False,
            include_flows=False,
        )

    async def run(self) -> RunCompletion:
        steps_completed = 0
        final_score = 0.0
        final_automated_score = 0.0
        success = False
        completion_written = False
        last_execution_error = False
        last_output = ""

        try:
            reset_result = self.environment.reset()
            observation = _observation_from_dict(_reset_observation(reset_result))

            for step in range(self.trajectory_length):
                before = compact_state(
                    observation,
                    step=step,
                    trajectory_length=self.trajectory_length,
                    quota=task_quota(self.env_id),
                    production_score=final_score,
                    automated_production_score=final_automated_score,
                    last_execution_error=last_execution_error,
                    last_output=last_output,
                )
                ids = interaction_ids(
                    run_id=self.run_id,
                    episode_id=self.episode_id,
                    step=step,
                )
                detailed = self.formatter.format(observation).raw_str
                pending = await self.controller.decide(
                    ids=ids,
                    state=before,
                    detailed_observation=detailed,
                )

                obs_dict, raw_reward, terminated, truncated, info = self.environment.step(
                    Action(agent_idx=0, code=pending.generated_policy.code)
                )
                after_observation = _observation_from_dict(obs_dict)
                last_output = str(info.get("result", obs_dict.get("raw_text", "")))
                last_execution_error = bool(info.get("error_occurred", False))
                final_score = float(info.get("production_score", raw_reward))
                final_automated_score = float(info.get("automated_production_score", 0.0))
                execution = ExecutionResult(
                    reward=float(raw_reward),
                    production_score=final_score,
                    automated_production_score=final_automated_score,
                    terminated=bool(terminated),
                    truncated=bool(truncated),
                    error_occurred=last_execution_error,
                    output=last_output,
                    ticks=max(int(info.get("ticks", 0)), 0),
                    policy_execution_seconds=max(
                        float(info.get("policy_execution_time", 0.0)), 0.0
                    ),
                    achievements=_json_object(info.get("achievements", {})),
                )
                after = compact_state(
                    after_observation,
                    step=step + 1,
                    trajectory_length=self.trajectory_length,
                    quota=task_quota(self.env_id),
                    production_score=final_score,
                    automated_production_score=final_automated_score,
                    last_execution_error=last_execution_error,
                    last_output=last_output,
                )
                await self.controller.observe(
                    pending=pending,
                    execution=execution,
                    after_state=after,
                    episode_end=bool(terminated or truncated or step + 1 == self.trajectory_length),
                )
                steps_completed = step + 1
                observation = after_observation

                if terminated or truncated:
                    success = bool(terminated)
                    break

            completion = RunCompletion(
                run_id=self.run_id,
                episode_id=self.episode_id,
                status="success" if success else "trajectory_limit",
                steps_completed=steps_completed,
                success=success,
                final_score=final_score,
                final_automated_score=final_automated_score,
            )
            self.controller.ledger.append(completion)
            completion_written = True
            return completion
        except BaseException as error:
            completion = RunCompletion(
                run_id=self.run_id,
                episode_id=self.episode_id,
                status="failed",
                steps_completed=steps_completed,
                success=False,
                final_score=final_score,
                final_automated_score=final_automated_score,
                error=f"{type(error).__name__}: {error}",
            )
            if not completion_written:
                self.controller.ledger.append(completion)
            raise
        finally:
            if self._owns_environment:
                self.environment.close()


def task_quota(env_id: str) -> float | None:
    task = THROUGHPUT_TASKS.get(env_id)
    if task is None:
        return None
    return float(task.quota)


def interaction_ids(*, run_id: str, episode_id: str, step: int) -> InteractionIds:
    ordinal = f"{step:05d}"
    return InteractionIds(
        run_id=run_id,
        episode_id=episode_id,
        interaction_id=f"{episode_id}-interaction-{ordinal}",
        event_id=f"{episode_id}-event-{ordinal}",
        trial_id=episode_id,
        step=step,
    )


def _reset_observation(value: Any) -> dict[str, Any]:
    observation = value[0] if isinstance(value, tuple) and value else value
    if not isinstance(observation, dict):
        raise TypeError("FLE reset did not return an observation mapping")
    return observation


def _observation_from_dict(value: dict[str, Any]) -> Observation:
    """Normalize known FLE 0.4.x serializer/deserializer disagreements."""

    normalized = dict(value)
    research = normalized.get("research")
    if isinstance(research, Mapping):
        normalized_research = dict(research)
        progress = normalized_research.get("progress")
        if progress is None or progress == "None":
            normalized_research["progress"] = []
        elif isinstance(progress, Mapping):
            normalized_research["progress"] = [
                {"name": str(name), "value": amount} for name, amount in progress.items()
            ]
        if normalized_research.get("current_research") in {None, "", "None"}:
            normalized_research["current_research"] = None
        normalized["research"] = normalized_research
    return Observation.from_dict(normalized)


def _json_object(value: Any) -> JsonObject:
    if isinstance(value, Mapping):
        serialized = json.loads(json.dumps(dict(value), default=str))
        if isinstance(serialized, dict):
            return serialized
    return {"value": str(value)}
