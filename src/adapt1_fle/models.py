"""Shared typed contracts for the research harness."""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


JsonObject = dict[str, Any]


class StrictModel(BaseModel):
    """Base model that rejects accidental contract drift."""

    model_config = ConfigDict(extra="forbid")


class SelectionSource(StrEnum):
    ADAPT_1 = "adapt_1"
    APPLICATION = "application"
    CONTROLLER = "controller"
    FALLBACK = "fallback"


class FailureKind(StrEnum):
    ADAPT_TRANSPORT = "adapt_transport"
    ADAPT_CONTRACT = "adapt_contract"
    MODEL_TRANSPORT = "model_transport"
    MODEL_OUTPUT = "model_output"
    FLE_EXECUTION = "fle_execution"
    TASK = "task"
    INTERRUPTED = "interrupted"


class CompactState(StrictModel):
    """Public, pre-consequence Factorio state supplied to Adapt-1."""

    task_key: str
    goal: str
    target_item: str | None = None
    phase: str
    step: int = Field(ge=0)
    trajectory_length: int = Field(ge=1)
    tick: int = Field(ge=0)
    elapsed_seconds: float = Field(ge=0)
    score: float
    automated_score: float
    quota: float | None = None
    progress: float | None = None
    inventory: dict[str, int] = Field(default_factory=dict)
    entity_counts: dict[str, int] = Field(default_factory=dict)
    entity_status_counts: dict[str, int] = Field(default_factory=dict)
    flow_inputs: dict[str, float] = Field(default_factory=dict)
    flow_outputs: dict[str, float] = Field(default_factory=dict)
    crafted: dict[str, float] = Field(default_factory=dict)
    harvested: dict[str, float] = Field(default_factory=dict)
    researched_count: int = Field(default=0, ge=0)
    last_action_error: bool = False
    last_error_category: str | None = None


class ApiExchange(StrictModel):
    """Redacted wire record for one Adapt-1 request."""

    request_id: str
    method: str
    route: str
    request: JsonObject | None = None
    status_code: int | None = None
    response: JsonObject | None = None
    elapsed_seconds: float = Field(ge=0)
    ambiguous: bool = False
    error: str | None = None


class StrategySelection(StrictModel):
    """Normalized result consumed by the application controller."""

    relation: str = "advances_goal"
    policy: str
    source: SelectionSource
    score: float | None = None
    decision_id: str | None = None
    supporting_evidence: list[str] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)
    abstained: bool = False
    reason: str
    raw_response: JsonObject = Field(default_factory=dict)
    exchange: ApiExchange | None = None


class MemoryContext(StrictModel):
    """Normalized Adapt-1 Memory context for the policy generator."""

    text: str = ""
    memory_ids: list[str] = Field(default_factory=list)
    confidence: float | None = None
    raw_response: JsonObject = Field(default_factory=dict)
    exchange: ApiExchange | None = None


class GeneratedPolicy(StrictModel):
    """Validated Python generated for one FLE action."""

    code: str
    raw_content: str
    model: str
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    latency_seconds: float = Field(default=0, ge=0)


class ExecutionResult(StrictModel):
    """Observable consequence of an exact FLE program."""

    reward: float
    production_score: float
    automated_production_score: float
    terminated: bool
    truncated: bool
    error_occurred: bool
    output: str
    ticks: int = Field(ge=0)
    policy_execution_seconds: float = Field(default=0, ge=0)
    achievements: JsonObject = Field(default_factory=dict)


class RewardRecord(StrictModel):
    """Raw and normalized reward contract."""

    raw_reward: float
    normalized_reward: float = Field(ge=-1, le=1)
    previous_score: float
    current_score: float
    previous_automated_score: float
    current_automated_score: float
    terminal_success: bool
    execution_error: bool
    rationale: str


class InteractionIds(StrictModel):
    """Stable identities for one ordered learner interaction."""

    run_id: str
    episode_id: str
    interaction_id: str
    event_id: str
    trial_id: str
    step: int = Field(ge=0)


class InteractionRecord(StrictModel):
    """Complete application-owned record around one environment transition."""

    kind: str = "interaction"
    recorded_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    ids: InteractionIds
    domain_id: str | None = None
    domain_revision: str | None = None
    mode: str
    before_state: CompactState
    selection: StrategySelection
    memory: MemoryContext
    generated_policy: GeneratedPolicy
    execution: ExecutionResult
    after_state: CompactState
    reward: RewardRecord
    feedback_exchange: ApiExchange | None = None
    memory_write_exchange: ApiExchange | None = None
    failure_kind: FailureKind | None = None


class RunCompletion(StrictModel):
    """Terminal run record."""

    kind: str = "completion"
    recorded_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    run_id: str
    episode_id: str
    status: str
    steps_completed: int = Field(ge=0)
    success: bool
    final_score: float
    final_automated_score: float
    error: str | None = None


class RunMetrics(StrictModel):
    """Metrics reconstructed from an append-only run ledger."""

    run_id: str
    mode: str
    steps: int
    success: bool
    final_score: float
    final_automated_score: float
    score_auc: float
    normalized_reward_sum: float
    adapt_selection_count: int
    fallback_count: int
    abstention_count: int
    execution_error_count: int
    token_count: int
    model_latency_seconds: float
    adapt_latency_seconds: float
