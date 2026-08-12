import json

import httpx
import pytest
import respx

from adapt1_fle.agent.model import FLEPolicyGenerator


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
                "choices": [
                    {
                        "message": {
                            "content": "```python\nprint(inspect_inventory())\n```"
                        }
                    }
                ],
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
