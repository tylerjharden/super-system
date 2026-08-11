"""Selective Adapt-1 Memory use for cross-episode Factorio evidence."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Any

from adapt1_fle.adapt.client import AdaptClient
from adapt1_fle.models import (
    ApiExchange,
    CompactState,
    ExecutionResult,
    MemoryContext,
    RewardRecord,
    StrategySelection,
)


class FactorioMemory:
    """Query useful evidence and avoid flooding Memory with every transition."""

    def __init__(self, client: AdaptClient, *, namespace: str, top_k: int = 8) -> None:
        self.client = client
        self.namespace = namespace
        self.top_k = top_k
        self._stored_fingerprints: set[str] = set()
        self._error_counts: Counter[str] = Counter()

    async def query(self, state: CompactState, *, frozen: bool) -> MemoryContext:
        message = (
            "Retrieve reusable Factorio evidence for "
            f"task={state.task_key}, target={state.target_item}, phase={state.phase}, "
            f"error={state.last_error_category or 'none'}."
        )
        response, exchange = await self.client.query_memory(
            message=message,
            top_k=self.top_k,
            metadata_filter={
                "application": "adapt1-fle",
                "domain_id": self.namespace,
                "task_key": state.task_key,
            },
            frozen=frozen,
        )
        return normalize_memory_response(response, exchange)

    async def maybe_store(
        self,
        *,
        before: CompactState,
        after: CompactState,
        selection: StrategySelection,
        execution: ExecutionResult,
        reward: RewardRecord,
        run_id: str,
    ) -> ApiExchange | None:
        reason = self._storage_reason(after, execution, reward)
        if reason is None:
            return None

        lesson = _lesson_text(
            before=before,
            after=after,
            selection=selection,
            execution=execution,
            reward=reward,
            reason=reason,
        )
        fingerprint = hashlib.sha256(lesson.encode("utf-8")).hexdigest()
        if fingerprint in self._stored_fingerprints:
            return None

        _, exchange = await self.client.store_memory(
            message=lesson,
            response="Retain as scoped Factorio strategy evidence.",
            context={
                "application": "adapt1-fle",
                "domain_id": self.namespace,
                "run_id": run_id,
                "task_key": before.task_key,
                "phase": before.phase,
                "policy": selection.policy,
                "reason": reason,
                "fingerprint": fingerprint,
            },
        )
        self._stored_fingerprints.add(fingerprint)
        return exchange

    def _storage_reason(
        self,
        after: CompactState,
        execution: ExecutionResult,
        reward: RewardRecord,
    ) -> str | None:
        if reward.terminal_success:
            return "task_success"
        if reward.normalized_reward >= 0.2:
            return "meaningful_progress"
        if execution.error_occurred:
            category = after.last_error_category or "unknown"
            self._error_counts[category] += 1
            if self._error_counts[category] >= 2:
                return "recurring_failure"
        return None


def normalize_memory_response(
    response: dict[str, Any],
    exchange: ApiExchange | None = None,
) -> MemoryContext:
    context = response.get("memory_context", "")
    if isinstance(context, list):
        text = "\n".join(_memory_item_text(item) for item in context)
    elif isinstance(context, dict):
        text = json.dumps(context, sort_keys=True)
    else:
        text = str(context or "")

    memory_ids: list[str] = []
    for key in ("memories", "supporting_memories", "results", "items"):
        value = response.get(key)
        if not isinstance(value, list):
            continue
        for item in value:
            if isinstance(item, dict):
                identifier = item.get("memory_id") or item.get("id")
                if isinstance(identifier, str):
                    memory_ids.append(identifier)

    confidence = response.get("confidence_score")
    normalized_confidence = (
        float(confidence)
        if isinstance(confidence, int | float) and not isinstance(confidence, bool)
        else None
    )
    return MemoryContext(
        text=text,
        memory_ids=memory_ids,
        confidence=normalized_confidence,
        raw_response=response,
        exchange=exchange,
    )


def _memory_item_text(item: Any) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        for key in ("content", "text", "memory", "user_message"):
            value = item.get(key)
            if isinstance(value, str):
                return value
        return json.dumps(item, sort_keys=True)
    return str(item)


def _lesson_text(
    *,
    before: CompactState,
    after: CompactState,
    selection: StrategySelection,
    execution: ExecutionResult,
    reward: RewardRecord,
    reason: str,
) -> str:
    return (
        f"Factorio evidence ({reason}): on task {before.task_key} in phase {before.phase}, "
        f"policy {selection.policy} changed score {before.score:.3f}->{after.score:.3f} "
        f"and automated score {before.automated_score:.3f}->{after.automated_score:.3f}; "
        f"normalized reward={reward.normalized_reward:.3f}, "
        f"execution_error={execution.error_occurred}, terminal={execution.terminated}."
    )
