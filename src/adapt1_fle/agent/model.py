"""FLE-compatible code-synthesis model boundary."""

from __future__ import annotations

import ast
import time
from collections.abc import Sequence
from typing import Any, Protocol

from fle.agents.llm.api_factory import APIFactory
from fle.agents.llm.parsing import parse_response

from adapt1_fle.models import GeneratedPolicy


class PolicyGenerationError(RuntimeError):
    """The model failed to return executable Python."""


class PolicyGenerator(Protocol):
    """Injectable policy-generation boundary."""

    async def generate(self, messages: Sequence[dict[str, str]]) -> GeneratedPolicy:
        """Generate one validated FLE Python action."""


class FLEPolicyGenerator:
    """Generate policies using FLE's provider routing and parser."""

    def __init__(
        self,
        model: str,
        *,
        max_tokens: int = 4096,
        temperature: float = 0.2,
        api_key_config_file: str | None = None,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._factory = APIFactory(model, api_key_config_file=api_key_config_file)

    async def generate(self, messages: Sequence[dict[str, str]]) -> GeneratedPolicy:
        started = time.monotonic()
        response = await self._factory.acall(
            messages=list(messages),
            n_samples=1,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            model=self.model,
        )
        latency = time.monotonic() - started
        policy = parse_response(response)
        if policy is None or not policy.code.strip():
            raise PolicyGenerationError(
                "model response did not contain Python in a fenced code block"
            )
        validate_python(policy.code)

        usage = getattr(response, "usage", None)
        prompt_tokens = _integer_attribute(usage, "prompt_tokens", "input_tokens")
        completion_tokens = _integer_attribute(usage, "completion_tokens", "output_tokens")
        total_tokens = _integer_attribute(usage, "total_tokens")
        if total_tokens == 0:
            total_tokens = prompt_tokens + completion_tokens

        return GeneratedPolicy(
            code=policy.code,
            raw_content=_response_content(response),
            model=self.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_seconds=latency,
        )


class StaticPolicyGenerator:
    """Deterministic generator used by smoke tests and local diagnostics."""

    def __init__(self, code: str = "print(inspect_inventory())") -> None:
        validate_python(code)
        self.code = code

    async def generate(self, messages: Sequence[dict[str, str]]) -> GeneratedPolicy:
        del messages
        return GeneratedPolicy(
            code=self.code,
            raw_content=f"```python\n{self.code}\n```",
            model="static-smoke-policy",
        )


def validate_python(code: str) -> None:
    """Reject malformed or oversized programs before FLE execution."""

    if not code.strip():
        raise PolicyGenerationError("generated policy is empty")
    if len(code) > 10_000:
        raise PolicyGenerationError("generated policy exceeds FLE's 10,000 character limit")
    try:
        ast.parse(code)
    except SyntaxError as error:
        raise PolicyGenerationError(f"generated policy is invalid Python: {error}") from error


def _integer_attribute(value: Any, *names: str) -> int:
    for name in names:
        attribute = getattr(value, name, None)
        if isinstance(attribute, int) and not isinstance(attribute, bool):
            return max(attribute, 0)
    return 0


def _response_content(response: Any) -> str:
    choices = getattr(response, "choices", None)
    if isinstance(choices, list) and choices:
        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None)
        if isinstance(content, str):
            return content
    return str(response)
