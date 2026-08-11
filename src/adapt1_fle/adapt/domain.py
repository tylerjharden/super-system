"""Versioned Factorio Domain lifecycle and policy normalization."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from adapt1_fle.adapt.client import AdaptClient, PermanentAdaptError
from adapt1_fle.models import (
    ApiExchange,
    CompactState,
    InteractionIds,
    JsonObject,
    SelectionSource,
    StrategySelection,
)

STRATEGIES = (
    "inspect",
    "gather",
    "craft",
    "power",
    "mine",
    "smelt",
    "logistics",
    "assemble",
    "research",
    "debug",
    "optimize",
    "verify",
)

PHASE_FALLBACKS = {
    "bootstrap": "inspect",
    "power": "power",
    "extraction": "mine",
    "smelting": "smelt",
    "logistics": "logistics",
    "assembly": "assemble",
    "research": "research",
    "debugging": "debug",
    "optimization": "optimize",
    "verification": "verify",
}


class DomainDefinition(BaseModel):
    """Local Domain contract plus non-API revision metadata."""

    model_config = ConfigDict(extra="forbid")

    revision: str
    domain_id: str
    description: str
    schema_: JsonObject = Field(alias="schema")
    hypotheses: list[JsonObject]
    query_templates: JsonObject
    learning: JsonObject

    def api_payload(self, *, domain_id: str | None = None) -> JsonObject:
        return {
            "domain_id": domain_id or self.domain_id,
            "description": self.description,
            "schema": self.schema_,
            "hypotheses": self.hypotheses,
            "query_templates": self.query_templates,
            "learning": self.learning,
        }

    @property
    def contract_hash(self) -> str:
        encoded = json.dumps(self.api_payload(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def load_domain_definition(path: str | Path) -> DomainDefinition:
    parsed = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise ValueError(f"Domain definition must be a mapping: {path}")
    return DomainDefinition.model_validate(parsed)


class FactorioDomain:
    """Application boundary for Factorio strategy learning."""

    relation = "advances_goal"

    def __init__(
        self,
        client: AdaptClient,
        definition: DomainDefinition,
        *,
        domain_id: str | None = None,
        top_k: int = 12,
    ) -> None:
        self.client = client
        self.definition = definition
        self.domain_id = domain_id or definition.domain_id
        self.top_k = top_k

    async def ensure(self, *, create_if_missing: bool = True) -> tuple[str, ApiExchange]:
        """Create the Domain if absent and reject incompatible existing state."""

        try:
            remote, exchange = await self.client.get_domain(self.domain_id)
        except PermanentAdaptError as error:
            if error.exchange.status_code != 404:
                raise
            if not create_if_missing:
                raise ValueError(
                    f"frozen Domain {self.domain_id!r} does not exist; train or create it first"
                ) from error
            _, exchange = await self.client.create_domain(
                self.definition.api_payload(domain_id=self.domain_id)
            )
            return "created", exchange

        expected = self.definition.api_payload(domain_id=self.domain_id)
        for key in ("description", "schema", "hypotheses", "query_templates", "learning"):
            if key in remote and not _is_subset(expected[key], remote[key]):
                raise ValueError(
                    f"Domain {self.domain_id!r} differs at {key}; use a new versioned domain_id"
                )
        return "existing", exchange

    async def select(
        self,
        state: CompactState,
        *,
        frozen: bool,
    ) -> StrategySelection:
        question = build_strategy_question(state)
        response, exchange = await self.client.query_domain(
            self.domain_id,
            question=question,
            top_k=self.top_k,
            return_fields=[
                "decision_id",
                "policy_scores",
                "ranked_hypotheses",
                "supporting_memories",
                "missing_evidence",
                "learning_state",
            ],
            metadata_filter=None,
            frozen=frozen,
        )
        return normalize_strategy_response(response, state=state, exchange=exchange)

    def build_feedback(
        self,
        *,
        ids: InteractionIds,
        selection: StrategySelection,
        next_state: CompactState,
        reward: float,
        terminal_success: bool,
        episode_end: bool,
        execution_error: bool,
    ) -> JsonObject:
        if terminal_success:
            outcome = "success"
        elif execution_error:
            outcome = "execution_error"
        elif reward > 0:
            outcome = "progress"
        else:
            outcome = "no_progress"

        payload: JsonObject = {
            "session_id": "ignored",
            "feedback_kind": "execution",
            "outcome": outcome,
            "relation": selection.relation,
            "policy": selection.policy,
            "values": {
                "reward": reward,
                "step_reward": reward,
                "next_state": next_state.model_dump(mode="json"),
                "terminal": episode_end,
            },
            "metadata": {
                "run_id": ids.run_id,
                "trial_id": ids.trial_id,
                "event_id": ids.event_id,
                "episode_id": ids.episode_id,
                "interaction_id": ids.interaction_id,
                "step": ids.step,
                "task_key": next_state.task_key,
                "relation": selection.relation,
                "policy": selection.policy,
                "selected_by": selection.source.value,
            },
        }
        if selection.source is SelectionSource.ADAPT_1 and selection.decision_id is not None:
            payload["decision_id"] = selection.decision_id
        return payload


def build_strategy_question(state: CompactState) -> str:
    state_json = json.dumps(state.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    policies = ", ".join(STRATEGIES)
    return (
        "Select the best-supported high-level strategy for the current Factorio state. "
        f"Available policies: {policies}. Return support for relation advances_goal. "
        f"Current public state: {state_json}"
    )


def normalize_strategy_response(
    response: JsonObject,
    *,
    state: CompactState,
    exchange: ApiExchange | None = None,
) -> StrategySelection:
    scores = _extract_policy_scores(response)
    direct_policy = _extract_direct_policy(response)

    source = SelectionSource.FALLBACK
    abstained = bool(response.get("abstained", False))
    selected_score: float | None = None
    reason: str

    if direct_policy in STRATEGIES:
        policy = direct_policy
        selected_score = scores.get(policy)
        source = SelectionSource.ADAPT_1
        reason = "Adapt-1 returned a supported policy selection"
    elif scores:
        highest = max(scores.values())
        winners = sorted(policy for policy, score in scores.items() if score == highest)
        if len(winners) == 1:
            policy = winners[0]
            selected_score = highest
            source = SelectionSource.ADAPT_1
            reason = "Adapt-1 policy score was uniquely highest"
        else:
            policy = fallback_policy(state)
            reason = f"Adapt-1 policy scores tied across {', '.join(winners)}"
    else:
        policy = fallback_policy(state)
        reason = "Adapt-1 returned no usable policy support"
        abstained = True

    return StrategySelection(
        policy=policy,
        source=source,
        score=selected_score,
        decision_id=_string_or_none(response.get("decision_id")),
        supporting_evidence=_extract_evidence_ids(response.get("supporting_memories")),
        missing_evidence=_extract_text_list(response.get("missing_evidence")),
        abstained=abstained,
        reason=reason,
        raw_response=response,
        exchange=exchange,
    )


def fallback_policy(state: CompactState) -> str:
    if state.last_action_error:
        return "debug"
    return PHASE_FALLBACKS.get(state.phase, "inspect")


def _extract_policy_scores(response: JsonObject) -> dict[str, float]:
    scores: dict[str, float] = {}
    raw_scores = response.get("policy_scores")
    if isinstance(raw_scores, dict):
        relation_scores = raw_scores.get("advances_goal")
        if isinstance(relation_scores, dict):
            _add_dict_scores(scores, relation_scores)
        _add_dict_scores(scores, raw_scores)
    elif isinstance(raw_scores, list):
        _add_list_scores(scores, raw_scores)

    hypotheses = response.get("ranked_hypotheses")
    if isinstance(hypotheses, list):
        _add_list_scores(scores, hypotheses)
    return scores


def _add_dict_scores(scores: dict[str, float], values: dict[Any, Any]) -> None:
    for key, value in values.items():
        if key not in STRATEGIES:
            continue
        score = _numeric_score(value)
        if score is not None:
            scores[str(key)] = score


def _add_list_scores(scores: dict[str, float], values: list[Any]) -> None:
    for item in values:
        if not isinstance(item, dict):
            continue
        policy = item.get("policy") or item.get("id") or item.get("name")
        if policy not in STRATEGIES:
            continue
        score = _numeric_score(item.get("score", item.get("policy_score", item.get("support"))))
        if score is not None:
            scores[str(policy)] = score


def _numeric_score(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, dict):
        for key in ("score", "value", "support", "expected_reward"):
            nested = value.get(key)
            if isinstance(nested, int | float) and not isinstance(nested, bool):
                return float(nested)
    return None


def _extract_direct_policy(response: JsonObject) -> str | None:
    for key in ("selected_policy", "selected_value", "policy", "selection"):
        value = response.get(key)
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            nested = value.get("policy") or value.get("value")
            if isinstance(nested, str):
                return nested
    return None


def _extract_evidence_ids(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        if isinstance(item, str):
            result.append(item)
        elif isinstance(item, dict):
            identifier = item.get("memory_id") or item.get("id")
            if isinstance(identifier, str):
                result.append(identifier)
    return result


def _extract_text_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _string_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _is_subset(expected: Any, actual: Any) -> bool:
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return False
        return all(
            key in actual and _is_subset(value, actual[key]) for key, value in expected.items()
        )
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) < len(expected):
            return False
        return all(_is_subset(item, actual[index]) for index, item in enumerate(expected))
    return bool(expected == actual)
