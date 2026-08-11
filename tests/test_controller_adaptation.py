import json
from pathlib import Path

import httpx
import respx

from adapt1_fle.adapt.client import AdaptClient
from adapt1_fle.adapt.domain import FactorioDomain, load_domain_definition
from adapt1_fle.adapt.memory import FactorioMemory
from adapt1_fle.agent.controller import AdaptiveController
from adapt1_fle.agent.model import StaticPolicyGenerator
from adapt1_fle.agent.prompt import ConversationWindow
from adapt1_fle.config import RunMode
from adapt1_fle.ledger import RunLedger
from adapt1_fle.models import CompactState, ExecutionResult, InteractionIds

BASE_URL = "https://adapt.example"


def state(*, step: int, score: float) -> CompactState:
    return CompactState(
        task_key="iron_plate_throughput",
        goal="Produce 16 iron plates",
        target_item="iron-plate",
        phase="assembly",
        step=step,
        trajectory_length=4,
        tick=step * 60,
        elapsed_seconds=float(step),
        score=score,
        automated_score=score,
        quota=16,
        progress=min(score / 16, 1),
    )


def ids() -> InteractionIds:
    return InteractionIds(
        run_id="run-1",
        episode_id="episode-1",
        interaction_id="interaction-0",
        event_id="event-0",
        trial_id="episode-1",
        step=0,
    )


def execution() -> ExecutionResult:
    return ExecutionResult(
        reward=16,
        production_score=16,
        automated_production_score=16,
        terminated=True,
        truncated=False,
        error_occurred=False,
        output="success",
        ticks=60,
    )


def make_controller(
    *,
    mode: RunMode,
    ledger: RunLedger,
    client: AdaptClient,
) -> AdaptiveController:
    definition = load_domain_definition("configs/domain.factorio.v1.yaml")
    domain = FactorioDomain(client, definition, domain_id="factorio-test")
    memory = FactorioMemory(client)
    return AdaptiveController(
        mode=mode,
        run_id="run-1",
        domain_revision="1",
        generator=StaticPolicyGenerator(),
        conversation=ConversationWindow("system", max_messages=5),
        ledger=ledger,
        domain=domain,
        memory=memory,
    )


@respx.mock
async def test_training_binds_query_to_feedback_and_memory(tmp_path: Path) -> None:
    respx.post(f"{BASE_URL}/api/v1/domains/factorio-test/query").mock(
        return_value=httpx.Response(
            200,
            json={"selected_policy": "assemble", "decision_id": "decision-1"},
        )
    )
    respx.post(f"{BASE_URL}/api/v1/memory/query").mock(
        return_value=httpx.Response(
            200,
            json={"memory_context": "Use direct insertion for bootstrap."},
        )
    )
    feedback = respx.post(f"{BASE_URL}/api/v1/domains/factorio-test/feedback").mock(
        return_value=httpx.Response(200, json={"status": "success"})
    )
    memory_store = respx.post(f"{BASE_URL}/api/v1/memory/store").mock(
        return_value=httpx.Response(200, json={"memory_id": "memory-1"})
    )
    ledger = RunLedger.create(
        tmp_path,
        "run-1",
        {"run_id": "run-1", "mode": "train"},
    )

    async with AdaptClient(base_url=BASE_URL, api_key="secret") as client:
        controller = make_controller(mode=RunMode.TRAIN, ledger=ledger, client=client)
        pending = await controller.decide(
            ids=ids(),
            state=state(step=0, score=0),
            detailed_observation="empty factory",
        )
        await controller.observe(
            pending=pending,
            execution=execution(),
            after_state=state(step=1, score=16),
        )

    feedback_body = json.loads(feedback.calls[0].request.content)
    assert feedback_body["decision_id"] == "decision-1"
    assert feedback_body["policy"] == "assemble"
    assert feedback_body["values"]["step_reward"] == 1.0
    assert feedback_body["values"]["terminal"] is True
    assert memory_store.called
    event = next(iter(ledger.read_events()))
    assert event["selection"]["source"] == "adapt_1"
    assert event["feedback_exchange"]["status_code"] == 200
    assert event["memory_write_exchange"]["status_code"] == 200


@respx.mock
async def test_frozen_mode_performs_no_writes(tmp_path: Path) -> None:
    domain_query = respx.post(f"{BASE_URL}/api/v1/domains/factorio-test/query").mock(
        return_value=httpx.Response(200, json={"selected_policy": "verify"})
    )
    memory_query = respx.post(f"{BASE_URL}/api/v1/memory/query").mock(
        return_value=httpx.Response(200, json={"memory_context": ""})
    )
    feedback = respx.post(f"{BASE_URL}/api/v1/domains/factorio-test/feedback").mock(
        return_value=httpx.Response(500)
    )
    memory_store = respx.post(f"{BASE_URL}/api/v1/memory/store").mock(
        return_value=httpx.Response(500)
    )
    ledger = RunLedger.create(
        tmp_path,
        "run-1",
        {"run_id": "run-1", "mode": "frozen"},
    )

    async with AdaptClient(base_url=BASE_URL, api_key="secret") as client:
        controller = make_controller(mode=RunMode.FROZEN, ledger=ledger, client=client)
        pending = await controller.decide(
            ids=ids(),
            state=state(step=0, score=0),
            detailed_observation="factory",
        )
        await controller.observe(
            pending=pending,
            execution=execution(),
            after_state=state(step=1, score=16),
        )

    domain_body = json.loads(domain_query.calls[0].request.content)
    memory_body = json.loads(memory_query.calls[0].request.content)
    assert domain_body["update_memory_state"] is False
    assert memory_body["update_memory_state"] is False
    assert feedback.call_count == 0
    assert memory_store.call_count == 0
