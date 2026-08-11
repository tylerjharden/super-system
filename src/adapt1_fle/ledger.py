"""Crash-conscious append-only provenance ledger."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterator, Mapping
from itertools import pairwise
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from adapt1_fle.models import (
    InteractionRecord,
    RunCompletion,
    RunMetrics,
    SelectionSource,
)

SENSITIVE_KEY = re.compile(
    r"(authorization|api[_-]?key|(^|[_-])(token|secret|password)$)",
    re.IGNORECASE,
)
BEARER_VALUE = re.compile(r"(?i)\bBearer\s+\S+")


class LedgerCorruptionError(RuntimeError):
    """The append-only sequence cannot be reconstructed safely."""


class RunLedger:
    """One manifest, checkpoint, and ordered JSONL stream per run."""

    def __init__(self, run_dir: Path, *, sequence: int) -> None:
        self.run_dir = run_dir
        self.events_path = run_dir / "events.jsonl"
        self.manifest_path = run_dir / "manifest.json"
        self.checkpoint_path = run_dir / "checkpoint.json"
        self._sequence = sequence

    @classmethod
    def create(
        cls,
        root: str | Path,
        run_id: str,
        manifest: Mapping[str, Any],
    ) -> RunLedger:
        run_dir = Path(root) / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        ledger = cls(run_dir, sequence=0)
        if ledger.events_path.exists():
            ledger._sequence = _validated_sequence(ledger.events_path)

        safe_manifest = redact(dict(manifest))
        if ledger.manifest_path.exists():
            existing = json.loads(ledger.manifest_path.read_text(encoding="utf-8"))
            if existing != safe_manifest:
                raise ValueError(f"run {run_id!r} already exists with a different manifest")
        else:
            _atomic_json_write(ledger.manifest_path, safe_manifest)
        return ledger

    @classmethod
    def open(cls, run_dir: str | Path) -> RunLedger:
        path = Path(run_dir)
        manifest = path / "manifest.json"
        if not manifest.exists():
            raise FileNotFoundError(f"run manifest does not exist: {manifest}")
        events = path / "events.jsonl"
        sequence = _validated_sequence(events) if events.exists() else 0
        return cls(path, sequence=sequence)

    def append(self, event: BaseModel | Mapping[str, Any]) -> int:
        payload = event.model_dump(mode="json") if isinstance(event, BaseModel) else dict(event)
        next_sequence = self._sequence + 1
        envelope = {"sequence": next_sequence, "event": redact(payload)}
        encoded = json.dumps(envelope, sort_keys=True, separators=(",", ":"))
        with self.events_path.open("a", encoding="utf-8") as stream:
            stream.write(encoded + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self._sequence = next_sequence
        return next_sequence

    def checkpoint(self, data: Mapping[str, Any]) -> None:
        payload = {"last_sequence": self._sequence, **redact(dict(data))}
        _atomic_json_write(self.checkpoint_path, payload)

    def read_events(self) -> Iterator[dict[str, Any]]:
        if not self.events_path.exists():
            return
        expected = 1
        with self.events_path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                envelope = json.loads(line)
                if envelope.get("sequence") != expected:
                    raise LedgerCorruptionError(
                        f"event sequence mismatch at line {line_number}: expected {expected}"
                    )
                event = envelope.get("event")
                if not isinstance(event, dict):
                    raise LedgerCorruptionError(f"invalid event at line {line_number}")
                yield event
                expected += 1

    @property
    def sequence(self) -> int:
        return self._sequence


def summarize_run(ledger: RunLedger) -> RunMetrics:
    manifest = json.loads(ledger.manifest_path.read_text(encoding="utf-8"))
    run_id = str(manifest.get("run_id", ledger.run_dir.name))
    mode = str(manifest.get("mode", "unknown"))
    interactions: list[InteractionRecord] = []
    completion: RunCompletion | None = None
    for raw_event in ledger.read_events():
        if raw_event.get("kind") == "interaction":
            interactions.append(InteractionRecord.model_validate(raw_event))
        elif raw_event.get("kind") == "completion":
            completion = RunCompletion.model_validate(raw_event)

    scores = [record.before_state.score for record in interactions]
    if interactions:
        scores.append(interactions[-1].after_state.score)
    score_auc = sum((left + right) / 2 for left, right in pairwise(scores))
    final_score = completion.final_score if completion else (scores[-1] if scores else 0.0)
    final_automated = (
        completion.final_automated_score
        if completion
        else (interactions[-1].after_state.automated_score if interactions else 0.0)
    )
    ambiguous_write_count = sum(_ambiguous_writes(record) for record in interactions)
    completion_status = completion.status if completion else "incomplete"

    return RunMetrics(
        run_id=run_id,
        mode=mode,
        completion_status=completion_status,
        operational_failure=completion_status in {"failed", "incomplete"}
        or ambiguous_write_count > 0,
        steps=len(interactions),
        success=completion.success if completion else False,
        final_score=final_score,
        final_automated_score=final_automated,
        score_auc=score_auc,
        normalized_reward_sum=sum(record.reward.normalized_reward for record in interactions),
        adapt_selection_count=sum(
            record.selection.source is SelectionSource.ADAPT_1 for record in interactions
        ),
        fallback_count=sum(
            record.selection.source is SelectionSource.FALLBACK for record in interactions
        ),
        abstention_count=sum(record.selection.abstained for record in interactions),
        execution_error_count=sum(record.execution.error_occurred for record in interactions),
        ambiguous_write_count=ambiguous_write_count,
        token_count=sum(record.generated_policy.total_tokens for record in interactions),
        model_latency_seconds=sum(
            record.generated_policy.latency_seconds for record in interactions
        ),
        adapt_latency_seconds=sum(_adapt_latency(record) for record in interactions),
    )


def redact(value: Any, *, key: str | None = None) -> Any:
    """Recursively redact credential-bearing keys and bearer strings."""

    if key is not None and SENSITIVE_KEY.search(key):
        return "***"
    if isinstance(value, Mapping):
        return {str(item_key): redact(item, key=str(item_key)) for item_key, item in value.items()}
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, tuple):
        return [redact(item) for item in value]
    if isinstance(value, str):
        return BEARER_VALUE.sub("Bearer ***", value)
    return value


def _adapt_latency(record: InteractionRecord) -> float:
    exchanges = (
        record.selection.exchange,
        record.memory.exchange,
        record.feedback_exchange,
        record.memory_write_exchange,
    )
    return sum(exchange.elapsed_seconds for exchange in exchanges if exchange is not None)


def _ambiguous_writes(record: InteractionRecord) -> int:
    return sum(
        exchange is not None and exchange.ambiguous
        for exchange in (record.feedback_exchange, record.memory_write_exchange)
    )


def _validated_sequence(path: Path) -> int:
    sequence = 0
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            envelope = json.loads(line)
            expected = sequence + 1
            if envelope.get("sequence") != expected:
                raise LedgerCorruptionError(
                    f"event sequence mismatch at line {line_number}: expected {expected}"
                )
            sequence = expected
    return sequence


def _atomic_json_write(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
