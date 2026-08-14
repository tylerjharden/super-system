from pathlib import Path

import httpx
import pytest
import respx

from adapt1_fle.adapt.client import AdaptClient
from adapt1_fle.adapt.domain import (
    DomainContractError,
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

    assert definition.revision == "3"
    assert definition.learning["sequential"]["enabled"] is True
    assert definition.learning["credit_assignment"]["neutral_reward"] == 0.0
    assert len(definition.contract_hash) == 64

    followup = load_domain_definition(Path("configs/domain.factorio.v2.yaml"))
    assert followup.revision == "4"
    assert followup.learning["policy"]["exploration_mode"] == "ucb"


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


def test_unknown_returned_policy_is_a_contract_error() -> None:
    with pytest.raises(DomainContractError, match="unknown policy"):
        normalize_strategy_response(
            {"selected_policy": "launch_rocket_immediately"},
            state=state(),
        )
    with pytest.raises(DomainContractError, match="abstained must be a boolean"):
        normalize_strategy_response(
            {"abstained": "yes"},
            state=state(),
        )


def test_explicit_abstention_forces_fallback_and_unseals_feedback() -> None:
    selection = normalize_strategy_response(
        {
            "abstained": True,
            "selected_policy": "assemble",
            "decision_id": "sealed-but-not-executed",
        },
        state=state(phase="logistics"),
    )

    assert selection.policy == "logistics"
    assert selection.source is SelectionSource.FALLBACK
    assert selection.abstained is True

    definition = load_domain_definition("configs/domain.factorio.v1.yaml")
    domain = FactorioDomain(AdaptClient(base_url=BASE_URL, api_key="secret"), definition)
    payload = domain.build_feedback(
        ids=InteractionIds(
            run_id="run",
            episode_id="episode",
            interaction_id="interaction",
            event_id="event",
            trial_id="trial",
            step=1,
        ),
        selection=selection,
        next_state=state(),
        reward=0,
        terminal_success=False,
        episode_end=False,
        execution_error=False,
    )
    assert "decision_id" not in payload
    assert payload["metadata"]["selected_by"] == "fallback"


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
        terminal_success=False,
        episode_end=True,
        execution_error=False,
    )

    assert payload["decision_id"] == "decision-1"
    assert payload["relation"] == "advances_goal"
    assert payload["policy"] == "assemble"
    assert payload["outcome"] == "progress"
    assert payload["values"]["step_reward"] == 0.4
    assert payload["values"]["terminal"] is True
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


@respx.mock
async def test_existing_domain_must_return_required_contract_fields() -> None:
    definition = load_domain_definition("configs/domain.factorio.v1.yaml")
    respx.get(f"{BASE_URL}/api/v1/domains/factorio-test").mock(
        return_value=httpx.Response(
            200,
            json={"domain_id": "factorio-test", "description": definition.description},
        )
    )

    async with AdaptClient(base_url=BASE_URL, api_key="secret") as client:
        domain = FactorioDomain(client, definition, domain_id="factorio-test")
        with pytest.raises(ValueError, match="missing required contract field schema"):
            await domain.ensure()
