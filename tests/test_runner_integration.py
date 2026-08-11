from pathlib import Path
from typing import Any

from fle.env.gym_env.action import Action

from adapt1_fle.agent.controller import AdaptiveController
from adapt1_fle.agent.model import StaticPolicyGenerator
from adapt1_fle.agent.prompt import ConversationWindow
from adapt1_fle.config import RunMode
from adapt1_fle.ledger import RunLedger, summarize_run
from adapt1_fle.runner import TrajectoryRunner


class FakeEnvironment:
    def __init__(self) -> None:
        self.actions: list[Action] = []
        self.closed = False

    def reset(self) -> tuple[dict[str, Any], dict[str, Any]]:
        return observation(success=False, score=0), {}

    def step(self, action: Action) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        self.actions.append(action)
        return (
            observation(success=True, score=16),
            16.0,
            True,
            False,
            {
                "result": "Inventory({'iron-ore': 16})",
                "error_occurred": False,
                "production_score": 16.0,
                "automated_production_score": 16.0,
                "ticks": 3_600,
                "policy_execution_time": 0.01,
                "achievements": {"iron-ore": 16},
            },
        )

    def close(self) -> None:
        self.closed = True


def observation(*, success: bool, score: float) -> dict[str, Any]:
    return {
        "raw_text": "success" if success else "",
        "map_image": "",
        "entities": [],
        "inventory": [{"type": "iron-ore", "quantity": int(score)}],
        "research": {
            "technologies": [],
            "current_research": "None",
            "research_progress": 0,
            "research_queue": [],
            "progress": "None",
        },
        "game_info": {"tick": int(score * 225), "time": score * 3.75, "speed": 1},
        "score": score,
        "automated_score": score,
        "flows": {
            "input": [],
            "output": [{"type": "iron-ore", "rate": score}],
            "crafted": [],
            "harvested": [],
            "price_list": [],
            "static_items": [],
        },
        "task_verification": {"success": int(success), "meta": []},
        "messages": [],
        "serialized_functions": [],
        "task_info": {
            "goal_description": "Produce 16 iron ore",
            "agent_instructions": "",
            "task_key": "iron_ore_throughput",
            "trajectory_length": 4,
        },
        "character_positions": [{"agent_idx": 0, "x": 0, "y": 0}],
    }


async def test_baseline_runner_executes_and_records_complete_transition(
    tmp_path: Path,
) -> None:
    ledger = RunLedger.create(
        tmp_path,
        "run-1",
        {"run_id": "run-1", "mode": "baseline"},
    )
    controller = AdaptiveController(
        mode=RunMode.BASELINE,
        run_id="run-1",
        domain_revision="1",
        generator=StaticPolicyGenerator(),
        conversation=ConversationWindow("system", max_messages=5),
        ledger=ledger,
    )
    environment = FakeEnvironment()
    runner = TrajectoryRunner(
        env_id="iron_ore_throughput",
        trajectory_length=4,
        controller=controller,
        run_id="run-1",
        episode_id="episode-1",
        environment=environment,
    )

    completion = await runner.run()
    metrics = summarize_run(ledger)
    events = list(ledger.read_events())

    assert completion.success is True
    assert completion.steps_completed == 1
    assert len(environment.actions) == 1
    assert environment.actions[0].code == "print(inspect_inventory())"
    assert environment.closed is False
    assert len(events) == 2
    assert events[0]["kind"] == "interaction"
    assert events[0]["reward"]["normalized_reward"] == 1.0
    assert events[1]["kind"] == "completion"
    assert metrics.success is True
    assert metrics.final_score == 16
    assert metrics.steps == 1
    assert metrics.adapt_selection_count == 0
