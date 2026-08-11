"""Public, bounded reward contract for Adapt-1 sequential feedback."""

from __future__ import annotations

import math

from adapt1_fle.models import CompactState, RewardRecord

EXECUTION_ERROR_PENALTY = 0.25


def calculate_reward(
    *,
    before: CompactState,
    after: CompactState,
    raw_reward: float,
    terminal_success: bool,
    execution_error: bool,
) -> RewardRecord:
    """Normalize observable FLE progress without adding hidden solver labels."""

    score_delta = after.score - before.score
    automated_delta = after.automated_score - before.automated_score

    if terminal_success and not execution_error:
        normalized = 1.0
        rationale = "public task verifier reported terminal success"
    elif before.quota is not None and before.quota > 0:
        normalized = score_delta / before.quota
        rationale = (
            f"public throughput score delta {score_delta:.6g} normalized by quota "
            f"{before.quota:.6g}"
        )
    else:
        denominator = max(abs(before.automated_score), 1.0)
        normalized = math.tanh(automated_delta / denominator)
        rationale = (
            f"automated production delta {automated_delta:.6g} normalized by "
            f"prior magnitude {denominator:.6g}"
        )

    if execution_error:
        normalized -= EXECUTION_ERROR_PENALTY
        rationale += f"; execution error penalty {EXECUTION_ERROR_PENALTY:.2f}"

    bounded = min(max(normalized, -1.0), 1.0)
    return RewardRecord(
        raw_reward=float(raw_reward),
        normalized_reward=bounded,
        previous_score=before.score,
        current_score=after.score,
        previous_automated_score=before.automated_score,
        current_automated_score=after.automated_score,
        terminal_success=terminal_success,
        execution_error=execution_error,
        rationale=rationale,
    )
