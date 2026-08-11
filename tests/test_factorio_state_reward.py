import pytest
from fle.env.gym_env.observation import Observation

from adapt1_fle.factorio.reward import calculate_reward
from adapt1_fle.factorio.state import compact_state, infer_phase
from adapt1_fle.models import CompactState


def observation() -> Observation:
    return Observation.from_dict(
        {
            "raw_text": "",
            "map_image": "",
            "entities": [
                {"name": "electric-mining-drill", "status": "WORKING"},
                {"name": "stone-furnace", "status": "WORKING"},
            ],
            "inventory": [
                {"type": "iron-ore", "quantity": 20},
                {"type": "coal", "quantity": 5},
            ],
            "research": {
                "technologies": [],
                "current_research": None,
                "research_progress": 0,
                "research_queue": [],
                "progress": [],
            },
            "game_info": {"tick": 600, "time": 10, "speed": 1},
            "score": 2,
            "automated_score": 1,
            "flows": {
                "input": [{"type": "coal", "rate": 1}],
                "output": [{"type": "iron-plate", "rate": 2}],
                "crafted": [],
                "harvested": [],
                "price_list": [],
                "static_items": [],
            },
            "task_verification": {"success": 0, "meta": []},
            "messages": [],
            "serialized_functions": [],
            "task_info": {
                "goal_description": "Produce 16 iron plates",
                "agent_instructions": "",
                "task_key": "iron_plate_throughput",
                "trajectory_length": 64,
            },
            "character_positions": [{"agent_idx": 0, "x": 0, "y": 0}],
        }
    )


def compact(*, score: float, automated: float, quota: float | None = 16) -> CompactState:
    return CompactState(
        task_key="iron_plate_throughput",
        goal="Produce plates",
        target_item="iron-plate",
        phase="smelting",
        step=1,
        trajectory_length=64,
        tick=0,
        elapsed_seconds=0,
        score=score,
        automated_score=automated,
        quota=quota,
    )


def test_compact_state_uses_absolute_score_overrides() -> None:
    result = compact_state(
        observation(),
        step=2,
        trajectory_length=64,
        quota=16,
        production_score=8,
        automated_production_score=6,
    )

    assert result.target_item == "iron-plate"
    assert result.score == 8
    assert result.automated_score == 6
    assert result.progress == 0.5
    assert result.inventory == {"coal": 5, "iron-ore": 20}
    assert result.entity_counts == {"electric-mining-drill": 1, "stone-furnace": 1}
    assert result.flow_outputs == {"iron-plate": 2.0}


def test_phase_prioritizes_execution_error() -> None:
    phase = infer_phase(
        target_item="iron-plate",
        progress=0.5,
        entity_counts={"stone-furnace": 1},
        status_counts={"working": 1},
        last_execution_error=True,
        task_success=False,
    )

    assert phase == "debugging"


def test_throughput_reward_is_monotonic_and_bounded() -> None:
    low = calculate_reward(
        before=compact(score=0, automated=0),
        after=compact(score=4, automated=2),
        raw_reward=4,
        terminal_success=False,
        execution_error=False,
    )
    high = calculate_reward(
        before=compact(score=0, automated=0),
        after=compact(score=32, automated=10),
        raw_reward=32,
        terminal_success=False,
        execution_error=False,
    )

    assert low.normalized_reward == 0.25
    assert high.normalized_reward == 1.0
    assert high.normalized_reward > low.normalized_reward


def test_no_progress_is_neutral_and_error_is_negative() -> None:
    neutral = calculate_reward(
        before=compact(score=2, automated=1),
        after=compact(score=2, automated=1),
        raw_reward=0,
        terminal_success=False,
        execution_error=False,
    )
    error = calculate_reward(
        before=compact(score=2, automated=1),
        after=compact(score=2, automated=1),
        raw_reward=0,
        terminal_success=False,
        execution_error=True,
    )

    assert neutral.normalized_reward == 0.0
    assert error.normalized_reward == -0.25


def test_open_play_reward_uses_automated_growth() -> None:
    reward = calculate_reward(
        before=compact(score=100, automated=10, quota=None),
        after=compact(score=150, automated=15, quota=None),
        raw_reward=50,
        terminal_success=False,
        execution_error=False,
    )

    assert reward.normalized_reward == pytest.approx(0.462117, rel=1e-5)


def test_terminal_success_has_maximum_reward() -> None:
    reward = calculate_reward(
        before=compact(score=15, automated=12),
        after=compact(score=16, automated=13),
        raw_reward=1,
        terminal_success=True,
        execution_error=False,
    )

    assert reward.normalized_reward == 1.0
