import json

import httpx
import pytest
import respx

from adapt1_fle.adapt.client import (
    AdaptClient,
    AmbiguousWriteError,
    PermanentAdaptError,
)

BASE_URL = "https://adapt.example"


@respx.mock
async def test_public_health_does_not_send_authentication() -> None:
    route = respx.get(f"{BASE_URL}/healthz").mock(
        return_value=httpx.Response(200, json={"ok": True})
    )
    async with AdaptClient(base_url=BASE_URL, api_key=None) as client:
        body, exchange = await client.health()

    assert body == {"ok": True}
    assert exchange.status_code == 200
    assert "Authorization" not in route.calls[0].request.headers


@respx.mock
async def test_safe_read_retries_transient_status() -> None:
    route = respx.get(f"{BASE_URL}/api/v1/domains").mock(
        side_effect=[
            httpx.Response(503, json={"detail": "warming"}, headers={"Retry-After": "0"}),
            httpx.Response(200, json={"domains": ["factorio"]}),
        ]
    )
    async with AdaptClient(
        base_url=BASE_URL,
        api_key="secret",
        read_retry_attempts=2,
    ) as client:
        body, exchange = await client.list_domains()

    assert body["domains"] == ["factorio"]
    assert route.call_count == 2
    assert exchange.attempt_count == 2
    assert [attempt["status_code"] for attempt in exchange.attempts] == [503, 200]


@respx.mock
async def test_mutating_transient_response_is_ambiguous_and_not_retried() -> None:
    route = respx.post(f"{BASE_URL}/api/v1/domains/factorio/feedback").mock(
        return_value=httpx.Response(503, json={"detail": "gateway timeout"})
    )
    async with AdaptClient(base_url=BASE_URL, api_key="secret") as client:
        with pytest.raises(AmbiguousWriteError) as raised:
            await client.submit_feedback("factorio", {"outcome": "progress"})

    assert route.call_count == 1
    assert raised.value.exchange.ambiguous is True


@respx.mock
async def test_any_write_side_server_error_is_ambiguous() -> None:
    route = respx.post(f"{BASE_URL}/api/v1/memory/store").mock(
        return_value=httpx.Response(500, json={"detail": "internal failure"})
    )
    async with AdaptClient(base_url=BASE_URL, api_key="secret") as client:
        with pytest.raises(AmbiguousWriteError) as raised:
            await client.store_memory(
                message="evidence",
                response="stored",
                context={},
            )

    assert route.call_count == 1
    assert raised.value.exchange.status_code == 500
    assert raised.value.exchange.ambiguous is True


@respx.mock
async def test_permanent_error_retains_redacted_exchange() -> None:
    respx.get(f"{BASE_URL}/api/v1/domains").mock(
        return_value=httpx.Response(401, json={"detail": "invalid token"})
    )
    async with AdaptClient(base_url=BASE_URL, api_key="secret") as client:
        with pytest.raises(PermanentAdaptError) as raised:
            await client.list_domains()

    assert raised.value.exchange.status_code == 401
    assert raised.value.exchange.request is None
    assert "secret" not in raised.value.exchange.model_dump_json()


@respx.mock
async def test_frozen_domain_query_is_read_only() -> None:
    route = respx.post(f"{BASE_URL}/api/v1/domains/factorio/query").mock(
        return_value=httpx.Response(
            200,
            json={
                "decision_id": "decision-1",
                "new_additive_field": {"kept": True},
            },
        )
    )
    async with AdaptClient(base_url=BASE_URL, api_key="secret") as client:
        body, exchange = await client.query_domain(
            "factorio",
            question="choose",
            top_k=5,
            frozen=True,
        )

    request = json.loads(route.calls[0].request.content)
    assert request["allow_exploration"] is False
    assert request["update_memory_state"] is False
    assert body["new_additive_field"] == {"kept": True}
    assert exchange.response == body
