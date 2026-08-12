from adapt1_fle.adapt.memory import (
    FAILURE_DIAGNOSTIC,
    POSITIVE_ONLY,
    profile_evidence_candidates,
)
from adapt1_fle.models import (
    CompactState,
    ExecutionResult,
    GeneratedPolicy,
    InteractionIds,
    InteractionRecord,
    MemoryContext,
    RewardRecord,
    SelectionSource,
    StrategySelection,
)


def test_failure_diagnostics_require_prior_positive_evidence() -> None:
    records = [
        _record(0, reward=-0.25, error=True),
        _record(1, reward=0.25, error=False),
        _record(2, reward=-0.25, error=True),
        _record(3, reward=-0.25, error=True),
    ]

    positive = profile_evidence_candidates(records, profile=POSITIVE_ONLY)
    diagnostic = profile_evidence_candidates(records, profile=FAILURE_DIAGNOSTIC)

    assert [(record.ids.step, reason) for record, reason in positive] == [
        (1, "meaningful_progress")
    ]
    assert [(record.ids.step, reason) for record, reason in diagnostic] == [
        (1, "meaningful_progress"),
        (3, "recurring_failure_after_positive"),
    ]


def _record(step: int, *, reward: float, error: bool) -> InteractionRecord:
    before = _state(step, error=False)
    after = _state(step + 1, error=error)
    return InteractionRecord(
        ids=InteractionIds(
            run_id="run",
            episode_id="episode",
            interaction_id=f"interaction-{step}",
            event_id=f"event-{step}",
            trial_id="trial",
            step=step,
        ),
        domain_id="domain",
        domain_revision="4",
        mode="train",
        before_state=before,
        selection=StrategySelection(
            policy="mine",
            source=SelectionSource.FORCED_EXPLORATION,
            reason="coverage",
        ),
        memory=MemoryContext(),
        generated_policy=GeneratedPolicy(
            code="print(inspect_inventory())",
            raw_content="```python\nprint(inspect_inventory())\n```",
            model="model",
        ),
        execution=ExecutionResult(
            reward=0,
            production_score=after.score,
            automated_production_score=after.automated_score,
            terminated=False,
            truncated=False,
            error_occurred=error,
            output="error" if error else "ok",
            ticks=60,
        ),
        after_state=after,
        reward=RewardRecord(
            raw_reward=0,
            normalized_reward=reward,
            previous_score=before.score,
            current_score=after.score,
            previous_automated_score=before.automated_score,
            current_automated_score=after.automated_score,
            terminal_success=False,
            execution_error=error,
            rationale="test",
        ),
    )


def _state(step: int, *, error: bool) -> CompactState:
    return CompactState(
        task_key="iron_ore_throughput",
        goal="Produce iron ore",
        phase="debugging" if error else "extraction",
        step=step,
        trajectory_length=12,
        tick=step * 60,
        elapsed_seconds=step,
        score=float(step),
        automated_score=float(step),
        last_action_error=error,
        last_error_category="execution_error" if error else None,
    )
