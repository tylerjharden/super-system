"""Prompt construction with bounded conversational context."""

from __future__ import annotations

import json
from collections.abc import Sequence

from adapt1_fle.models import CompactState, MemoryContext, StrategySelection

POLICY_REQUIREMENTS = """## Action contract
- Return one Python program inside a fenced ```python block.
- Use only the FLE API and Python standard library available in the REPL.
- Keep the action focused, incremental, and at most 50 lines.
- Inspect or assert uncertain state instead of guessing.
- Preserve working automation and reuse saved namespace functions.
- The Adapt-1 strategy is guidance, not permission to ignore observed constraints.
"""


def build_system_prompt(
    *,
    fle_system_prompt: str,
    goal: str,
    trajectory_length: int,
) -> str:
    return (
        f"{fle_system_prompt.rstrip()}\n\n"
        "## REI Adapt-1 research controller\n"
        "Adapt-1 supplies a structured high-level strategy and supporting evidence. "
        "You own only the boundary task of translating the current public Factorio "
        "observation and that strategy into exact executable FLE Python.\n\n"
        f"## Task objective\n{goal}\n"
        f"You have at most {trajectory_length} actions.\n\n"
        f"{POLICY_REQUIREMENTS.rstrip()}"
    )


def build_step_prompt(
    *,
    state: CompactState,
    detailed_observation: str,
    selection: StrategySelection,
    memory: MemoryContext,
) -> str:
    support = ", ".join(selection.supporting_evidence) or "none"
    missing = "; ".join(selection.missing_evidence) or "none reported"
    memory_text = memory.text.strip()[:4_000] or "No relevant cross-episode evidence."
    state_json = json.dumps(
        state.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"""## Step {state.step + 1}/{state.trajectory_length}

### Adapt-1 structured strategy
- relation: {selection.relation}
- selected policy: {selection.policy}
- selected by: {selection.source.value}
- score/support: {selection.score if selection.score is not None else "not reported"}
- abstained: {selection.abstained}
- reason: {selection.reason}
- supporting evidence IDs: {support}
- missing evidence: {missing}

### Relevant persistent evidence
{memory_text}

### Compact public state
```json
{state_json}
```

### Detailed FLE observation
{detailed_observation}

Translate the current state and selected strategy into the next exact FLE Python action.
"""


class ConversationWindow:
    """Bounded user/assistant history that always preserves the system contract."""

    def __init__(self, system_prompt: str, *, max_messages: int = 17) -> None:
        if max_messages < 3:
            raise ValueError("max_messages must preserve system plus one complete turn")
        self.max_messages = max_messages
        self._messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]

    def messages_with(self, user_prompt: str) -> list[dict[str, str]]:
        history = self._trimmed_history(reserve=1)
        return [*history, {"role": "user", "content": user_prompt}]

    def commit(self, user_prompt: str, assistant_content: str) -> None:
        self._messages.extend(
            [
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": assistant_content},
            ]
        )
        self._messages = self._trimmed_history()

    @property
    def messages(self) -> Sequence[dict[str, str]]:
        return tuple(self._messages)

    def _trimmed_history(self, *, reserve: int = 0) -> list[dict[str, str]]:
        limit = self.max_messages - reserve
        if len(self._messages) <= limit:
            return list(self._messages)

        system = self._messages[0]
        available = max(limit - 1, 2)
        if available % 2 != 0:
            available -= 1
        recent = self._messages[-available:]
        if recent and recent[0]["role"] == "assistant":
            recent = recent[1:]
        return [system, *recent]
