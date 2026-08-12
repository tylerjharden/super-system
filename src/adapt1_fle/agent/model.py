"""FLE-compatible code-synthesis model boundary."""

from __future__ import annotations

import ast
import os
import re
import sys
import time
from collections.abc import Sequence
from typing import Any, Protocol

import httpx
from fle.agents.llm.api_factory import APIFactory
from fle.agents.llm.parsing import parse_response

from adapt1_fle.models import GeneratedPolicy

DEFAULT_MAX_TOKENS = 1024
DEFAULT_TEMPERATURE = 0.2

_REPAIR_PROMPT = (
    "Your previous reply was not usable. Reply with exactly one fenced "
    "```python``` block containing the next Factorio Learning Environment action. "
    "Call FLE tools directly: never import fle, flet, flapi, or any non-standard "
    "library. Do not redefine helpers, paste JSON state, or include prose outside "
    "the fence."
)

_FENCED_PYTHON = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.IGNORECASE | re.DOTALL)


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
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        api_key_config_file: str | None = None,
        parse_retries: int = 1,
        seed: int | None = None,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.parse_retries = max(parse_retries, 0)
        self.seed = seed
        if seed is not None and not model.startswith("ollama"):
            raise ValueError("model_seed is currently supported only for Ollama models")
        self._factory = APIFactory(model, api_key_config_file=api_key_config_file)

    async def generate(self, messages: Sequence[dict[str, str]]) -> GeneratedPolicy:
        started = time.monotonic()
        working = [dict(message) for message in messages]
        prompt_tokens = 0
        completion_tokens = 0
        raw_content = ""
        last_error = "model response did not contain Python in a fenced code block"

        for attempt in range(self.parse_retries + 1):
            if self.model.startswith("ollama"):
                response = None
                raw_content, usage = await self._call_ollama(working)
            else:
                response = await self._factory.acall(
                    messages=working,
                    n_samples=1,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    model=self.model,
                )
                raw_content = _response_content(response)
                usage = getattr(response, "usage", None)
            prompt_tokens += _integer_attribute(usage, "prompt_tokens", "input_tokens")
            completion_tokens += _integer_attribute(usage, "completion_tokens", "output_tokens")

            code = _extract_python_code(response, raw_content)
            if code is None:
                last_error = "model response did not contain Python in a fenced code block"
            else:
                try:
                    validate_python(code)
                except PolicyGenerationError as error:
                    last_error = str(error)
                else:
                    total_tokens = prompt_tokens + completion_tokens
                    return GeneratedPolicy(
                        code=code,
                        raw_content=raw_content,
                        model=self.model,
                        input_messages=[dict(message) for message in messages],
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=total_tokens,
                        latency_seconds=time.monotonic() - started,
                    )

            if attempt >= self.parse_retries:
                break
            working = [
                *working,
                {"role": "assistant", "content": raw_content or "(empty response)"},
                {"role": "user", "content": _REPAIR_PROMPT},
            ]

        raise PolicyGenerationError(last_error)

    async def _call_ollama(
        self, messages: list[dict[str, str]]
    ) -> tuple[str, dict[str, Any]]:
        base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1").rstrip("/")
        payload: dict[str, Any] = {
            "model": self.model.removeprefix("ollama-"),
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        if self.seed is not None:
            payload["seed"] = self.seed
        async with httpx.AsyncClient(timeout=300) as client:
            response = await client.post(
                f"{base_url}/chat/completions",
                headers={"Authorization": "Bearer ollama"},
                json=payload,
            )
            response.raise_for_status()
            body = response.json()
        choices = body.get("choices", []) if isinstance(body, dict) else []
        if not choices or not isinstance(choices[0], dict):
            return "", {}
        message = choices[0].get("message")
        content = message.get("content") if isinstance(message, dict) else ""
        usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
        return str(content or ""), usage


class StaticPolicyGenerator:
    """Deterministic generator used by smoke tests and local diagnostics."""

    def __init__(self, code: str = "print(inspect_inventory())") -> None:
        validate_python(code)
        self.code = code

    async def generate(self, messages: Sequence[dict[str, str]]) -> GeneratedPolicy:
        return GeneratedPolicy(
            code=self.code,
            raw_content=f"```python\n{self.code}\n```",
            model="static-smoke-policy",
            input_messages=[dict(message) for message in messages],
        )


def validate_python(code: str) -> None:
    """Reject malformed or oversized programs before FLE execution."""

    if not code.strip():
        raise PolicyGenerationError("generated policy is empty")
    if len(code) > 10_000:
        raise PolicyGenerationError("generated policy exceeds FLE's 10,000 character limit")
    try:
        tree = ast.parse(code)
    except SyntaxError as error:
        raise PolicyGenerationError(f"generated policy is invalid Python: {error}") from error
    imported_roots = {
        alias.name.partition(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported_roots.update(
        node.module.partition(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    )
    unsupported = sorted(imported_roots - sys.stdlib_module_names)
    if unsupported:
        raise PolicyGenerationError(
            "generated policy imports unavailable modules: " + ", ".join(unsupported)
        )


def _extract_python_code(response: Any, raw_content: str) -> str | None:
    """Prefer FLE's parser, then fall back to the last fenced Python block."""

    try:
        policy = parse_response(response)
    except Exception:
        policy = None
    if policy is not None and isinstance(policy.code, str) and policy.code.strip():
        return policy.code
    matches = [match.group(1).strip() for match in _FENCED_PYTHON.finditer(raw_content)]
    for candidate in reversed(matches):
        if candidate:
            return candidate
    return None


def _integer_attribute(value: Any, *names: str) -> int:
    for name in names:
        if isinstance(value, dict):
            attribute = value.get(name)
        else:
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
