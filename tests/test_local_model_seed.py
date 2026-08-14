import json

import httpx
import pytest
import respx

from adapt1_fle.agent.model import FLEPolicyGenerator, PolicyGenerationError


@pytest.mark.asyncio
@respx.mock
async def test_ollama_generation_records_seed_in_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama.test/v1")
    route = respx.post("http://ollama.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "```python\nprint(inspect_inventory())\n```"}}],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            },
        )
    )
    generator = FLEPolicyGenerator("ollama-qwen2.5-coder:7b", seed=812345)

    policy = await generator.generate([{"role": "user", "content": "inspect"}])

    request = json.loads(route.calls[0].request.content)
    assert request["seed"] == 812345
    assert policy.code == "print(inspect_inventory())"
    assert policy.total_tokens == 15


@pytest.mark.asyncio
@respx.mock
async def test_ollama_read_timeout_retries_with_same_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama.test/v1")
    route = respx.post("http://ollama.test/v1/chat/completions").mock(
        side_effect=[
            httpx.ReadTimeout("model stalled"),
            httpx.Response(
                200,
                json={
                    "choices": [
                        {"message": {"content": "```python\nprint(inspect_inventory())\n```"}}
                    ],
                    "usage": {},
                },
            ),
        ]
    )
    generator = FLEPolicyGenerator("ollama-qwen2.5-coder:7b", seed=812346)

    policy = await generator.generate([{"role": "user", "content": "inspect"}])

    assert policy.code == "print(inspect_inventory())"
    assert len(route.calls) == 2
    assert [json.loads(call.request.content)["seed"] for call in route.calls] == [
        812346,
        812346,
    ]


@pytest.mark.asyncio
@respx.mock
async def test_ollama_transport_retry_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama.test/v1")
    route = respx.post("http://ollama.test/v1/chat/completions").mock(
        side_effect=httpx.ReadTimeout("model stalled")
    )
    generator = FLEPolicyGenerator(
        "ollama-qwen2.5-coder:7b",
        transport_retries=2,
        seed=812347,
    )

    with pytest.raises(httpx.ReadTimeout, match="model stalled"):
        await generator.generate([{"role": "user", "content": "inspect"}])

    assert len(route.calls) == 3


@pytest.mark.asyncio
@respx.mock
async def test_malformed_generation_exhausts_repairs_with_same_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama.test/v1")
    route = respx.post("http://ollama.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "```python\ninvalid = = syntax\n```"}}],
                "usage": {},
            },
        )
    )
    generator = FLEPolicyGenerator("ollama-qwen2.5-coder:7b", seed=812348)

    with pytest.raises(PolicyGenerationError, match="invalid Python"):
        await generator.generate([{"role": "user", "content": "inspect"}])

    assert len(route.calls) == 4
    assert [json.loads(call.request.content)["seed"] for call in route.calls] == [
        812348,
        812348,
        812348,
        812348,
    ]
