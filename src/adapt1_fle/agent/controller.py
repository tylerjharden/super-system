"""End-to-end decision and feedback controller."""

from __future__ import annotations

from dataclasses import dataclass

from adapt1_fle.adapt.client import AdaptClientError
from adapt1_fle.adapt.domain import FactorioDomain, fallback_policy
from adapt1_fle.adapt.memory import FactorioMemory
from adapt1_fle.agent.model import PolicyGenerator
from adapt1_fle.agent.prompt import ConversationWindow, build_step_prompt
from adapt1_fle.config import RunMode
from adapt1_fle.factorio.reward import calculate_reward
from adapt1_fle.ledger import RunLedger
from adapt1_fle.models import (
    ApiExchange,
    CompactState,
    ExecutionResult,
    FailureKind,
    GeneratedPolicy,
    InteractionIds,
    InteractionRecord,
    MemoryContext,
    RewardRecord,
    SelectionSource,
    StrategySelection,
)


@dataclass(frozen=True)
class PendingDecision:
    """Decision-time state retained until its external consequence arrives."""

    ids: InteractionIds
    before_state: CompactState
    selection: StrategySelection
    memory: MemoryContext
    generated_policy: GeneratedPolicy
    user_prompt: str


class AdaptiveController:
    """Connect Adapt-1 reasoning to exact FLE action consequences."""

    def __init__(
        self,
        *,
        mode: RunMode,
        run_id: str,
        domain_revision: str,
        generator: PolicyGenerator,
        conversation: ConversationWindow,
        ledger: RunLedger,
        domain: FactorioDomain | None = None,
        memory: FactorioMemory | None = None,
    ) -> None:
        if mode is not RunMode.BASELINE and domain is None:
            raise ValueError("Adapt-enabled controller requires a FactorioDomain")
        self.mode = mode
        self.run_id = run_id
        self.domain_revision = domain_revision
        self.generator = generator
        self.conversation = conversation
        self.ledger = ledger
        self.domain = domain
        self.memory = memory

    async def decide(
        self,
        *,
        ids: InteractionIds,
        state: CompactState,
        detailed_observation: str,
    ) -> PendingDecision:
        if self.mode is RunMode.BASELINE:
            selection = StrategySelection(
                policy=fallback_policy(state),
                source=SelectionSource.CONTROLLER,
                abstained=True,
                reason="baseline arm does not call Adapt-1",
            )
            memory = MemoryContext()
        else:
            if self.domain is None:
                raise RuntimeError("Adapt Domain disappeared after controller validation")
            selection = await self.domain.select(
                state,
                frozen=self.mode is RunMode.FROZEN,
                run_id=self.run_id,
            )
            memory = (
                await self.memory.query(state, frozen=self.mode is RunMode.FROZEN)
                if self.memory is not None
                else MemoryContext()
            )

        user_prompt = build_step_prompt(
            state=state,
            detailed_observation=detailed_observation,
            selection=selection,
            memory=memory,
        )
        messages = self.conversation.messages_with(user_prompt)
        generated = await self.generator.generate(messages)
        self.conversation.commit(user_prompt, generated.raw_content)
        return PendingDecision(
            ids=ids,
            before_state=state,
            selection=selection,
            memory=memory,
            generated_policy=generated,
            user_prompt=user_prompt,
        )

    async def observe(
        self,
        *,
        pending: PendingDecision,
        execution: ExecutionResult,
        after_state: CompactState,
    ) -> InteractionRecord:
        reward = calculate_reward(
            before=pending.before_state,
            after=after_state,
            raw_reward=execution.reward,
            terminal_success=execution.terminated,
            execution_error=execution.error_occurred,
        )
        feedback_exchange = None
        memory_write_exchange = None
        failure_kind = None

        try:
            if self.mode is RunMode.TRAIN:
                if self.domain is None:
                    raise RuntimeError("training controller requires a Domain")
                payload = self.domain.build_feedback(
                    ids=pending.ids,
                    selection=pending.selection,
                    next_state=after_state,
                    reward=reward.normalized_reward,
                    terminal=execution.terminated or execution.truncated,
                    execution_error=execution.error_occurred,
                )
                _, feedback_exchange = await self.domain.client.submit_feedback(
                    self.domain.domain_id, payload
                )
                if self.memory is not None:
                    memory_write_exchange = await self.memory.maybe_store(
                        before=pending.before_state,
                        after=after_state,
                        selection=pending.selection,
                        execution=execution,
                        reward=reward,
                        run_id=self.run_id,
                    )
        except AdaptClientError as error:
            failure_kind = FailureKind.ADAPT_TRANSPORT
            if feedback_exchange is None:
                feedback_exchange = error.exchange
            elif memory_write_exchange is None:
                memory_write_exchange = error.exchange
            record = self._record(
                pending=pending,
                execution=execution,
                after_state=after_state,
                reward=reward,
                feedback_exchange=feedback_exchange,
                memory_write_exchange=memory_write_exchange,
                failure_kind=failure_kind,
            )
            self.ledger.append(record)
            self.ledger.checkpoint(
                {
                    "run_id": self.run_id,
                    "last_confirmed_step": pending.ids.step - 1,
                    "ambiguous_interaction": pending.ids.interaction_id,
                }
            )
            raise

        record = self._record(
            pending=pending,
            execution=execution,
            after_state=after_state,
            reward=reward,
            feedback_exchange=feedback_exchange,
            memory_write_exchange=memory_write_exchange,
            failure_kind=failure_kind,
        )
        self.ledger.append(record)
        self.ledger.checkpoint(
            {
                "run_id": self.run_id,
                "episode_id": pending.ids.episode_id,
                "last_confirmed_step": pending.ids.step,
                "last_interaction_id": pending.ids.interaction_id,
            }
        )
        return record

    def _record(
        self,
        *,
        pending: PendingDecision,
        execution: ExecutionResult,
        after_state: CompactState,
        reward: RewardRecord,
        feedback_exchange: ApiExchange | None,
        memory_write_exchange: ApiExchange | None,
        failure_kind: FailureKind | None,
    ) -> InteractionRecord:
        return InteractionRecord(
            ids=pending.ids,
            domain_id=self.domain.domain_id if self.domain is not None else None,
            domain_revision=self.domain_revision if self.domain is not None else None,
            mode=self.mode.value,
            before_state=pending.before_state,
            selection=pending.selection,
            memory=pending.memory,
            generated_policy=pending.generated_policy,
            execution=execution,
            after_state=after_state,
            reward=reward,
            feedback_exchange=feedback_exchange,
            memory_write_exchange=memory_write_exchange,
            failure_kind=failure_kind,
        )
