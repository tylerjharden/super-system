from pathlib import Path

import pytest

from adapt1_fle.agent.model import PolicyGenerationError, validate_python
from adapt1_fle.agent.prompt import ConversationWindow
from adapt1_fle.ledger import LedgerCorruptionError, RunLedger, redact


def test_python_validation_blocks_invalid_or_oversized_code() -> None:
    with pytest.raises(PolicyGenerationError, match="invalid Python"):
        validate_python("if:")
    with pytest.raises(PolicyGenerationError, match="10,000"):
        validate_python("x" * 10_001)

    validate_python("print(inspect_inventory())")


def test_conversation_window_preserves_system_and_complete_recent_turns() -> None:
    conversation = ConversationWindow("system", max_messages=5)
    for index in range(5):
        conversation.commit(f"user-{index}", f"assistant-{index}")

    messages = list(conversation.messages)

    assert messages[0] == {"role": "system", "content": "system"}
    assert len(messages) <= 5
    assert messages[-2:] == [
        {"role": "user", "content": "user-4"},
        {"role": "assistant", "content": "assistant-4"},
    ]


def test_redaction_preserves_usage_metrics_and_removes_credentials() -> None:
    result = redact(
        {
            "api_key": "secret",
            "REI_TOKEN": "secret",
            "prompt_tokens": 12,
            "total_tokens": 20,
            "text": "Authorization: Bearer actual-secret",
        }
    )

    assert result["api_key"] == "***"
    assert result["REI_TOKEN"] == "***"
    assert result["prompt_tokens"] == 12
    assert result["total_tokens"] == 20
    assert result["text"] == "Authorization: Bearer ***"


def test_ledger_rejects_sequence_corruption(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text('{"run_id":"run"}\n', encoding="utf-8")
    (run_dir / "events.jsonl").write_text(
        '{"sequence":2,"event":{"kind":"completion"}}\n',
        encoding="utf-8",
    )

    with pytest.raises(LedgerCorruptionError, match="expected 1"):
        RunLedger.open(run_dir)


def test_failed_append_does_not_advance_sequence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger = RunLedger.create(tmp_path, "run", {"run_id": "run"})
    original_open = Path.open

    def fail_event_open(path: Path, *args: object, **kwargs: object) -> object:
        if path == ledger.events_path:
            raise OSError("disk full")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_event_open)
    with pytest.raises(OSError, match="disk full"):
        ledger.append({"kind": "test"})

    assert ledger.sequence == 0


def test_existing_manifest_must_match(tmp_path: Path) -> None:
    RunLedger.create(tmp_path, "run", {"run_id": "run", "mode": "baseline"})

    with pytest.raises(ValueError, match="different manifest"):
        RunLedger.create(tmp_path, "run", {"run_id": "run", "mode": "train"})
