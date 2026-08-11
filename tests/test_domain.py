from pathlib import Path

import httpx
import respx

from adapt1_fle.adapt.client import AdaptClient
from adapt1_fle.adapt.domain import (
    FactorioDomain,
    load_domain_definition,
    normalize_strategy_response,
)
from adapt1_fle.models import (
    CompactState,
    InteractionIds,
    SelectionSource,
    StrategySelection,
)

BASE_URL = "https://adapt.example"


def state(*, phase: str = "assembly", error: bool = False) -> CompactState:
    return CompactState(
        task_key="electronic_circuit_throughput",
        goal="Produce circuits",
        target_item="electronic-circuit",
        phase=phase,
        step=3,
        trajectory_length=64,
        tick=120,
        elapsed_seconds=2,
        score=1,
        automated_score=0,
        quota=16,
        progress=0.0625,
        last_action_error=error,
    )


def test_loads_versioned_domain_contract() -> None:
    definition = load_domain_definition(Path("configs/domain.factorio.v1.yaml"))

    assert definition.revision == "1"
    assert definition.learning["sequential"]["enabled"] is True
    assert definition.learning["credit_assignment"]["neutral_reward"] == 0.0
    assert len(definition.contract_hash) == 64


def test_unique_policy_score_is_selected() -> None:
    selection = normalize_strategy_response(
        {
            "decision_id": "decision-1",
            "policy_scores": {
                "advances_goal": {
                    "assemble": {"score": 0.9},
                    "inspect": {"score": 0.2},
                }
            },
            "supporting_memories": [{"memory_id": "memory-1"}],
        },
        state=state(),
    )

    assert selection.policy == "assemble"
    assert selection.source is SelectionSource.ADAPT_1
    assert selection.score == 0.9
    assert selection.decision_id == "decision-1"
    assert selection.supporting_evidence == ["memory-1"]


def test_tied_scores_use_logged_phase_fallback() -> None:
    selection = normalize_strategy_response(
        {"policy_scores": {"assemble": 0.5, "inspect": 0.5}},
        state=state(phase="assembly"),
    )

    assert selection.policy == "assemble"
    assert selection.source is SelectionSource.FALLBACK
    assert "tied" in selection.reason


def test_execution_error_falls_back_to_debug() -> None:
    selection = normalize_strategy_response({}, state=state(error=True))

    assert selection.policy == "debug"
    assert selection.abstained is True
    assert selection.source is SelectionSource.FALLBACK


def test_feedback_binds_sealed_decision_and_sequential_fields() -> None:
    definition = load_domain_definition("configs/domain.factorio.v1.yaml")
    client = AdaptClient(base_url=BASE_URL, api_key="secret")
    domain = FactorioDomain(client, definition)
    selection = StrategySelection(
        policy="assemble",
        source=SelectionSource.ADAPT_1,
        decision_id="decision-1",
        reason="selected",
    )
    ids = InteractionIds(
        run_id="run-1",
        episode_id="episode-1",
        interaction_id="interaction-3",
        event_id="event-3",
        trial_id="trial-1",
        step=3,
    )

    payload = domain.build_feedback(
        ids=ids,
        selection=selection,
        next_state=state(),
        reward=0.4,
        terminal=False,
        execution_error=False,
    )

    assert payload["decision_id"] == "decision-1"
    assert payload["relation"] == "advances_goal"
    assert payload["policy"] == "assemble"
    assert payload["values"]["step_reward"] == 0.4
    assert payload["values"]["terminal"] is False
    assert payload["metadata"]["episode_id"] == "episode-1"
    assert payload["metadata"]["step"] == 3


@respx.mock
async def test_ensure_creates_only_after_not_found() -> None:
    definition = load_domain_definition("configs/domain.factorio.v1.yaml")
    respx.get(f"{BASE_URL}/api/v1/domains/factorio-test").mock(
        return_value=httpx.Response(404, json={"detail": "not found"})
    )
    create = respx.post(f"{BASE_URL}/api/v1/domains").mock(
        return_value=httpx.Response(200, json={"status": "success"})
    )

    async with AdaptClient(base_url=BASE_URL, api_key="secret") as client:
        domain = FactorioDomain(client, definition, domain_id="factorio-test")
        status, _ = await domain.ensure()

    assert status == "created"
    assert create.called
    assert create.calls[0].request.headers["Authorization"] == "Bearer secret"
