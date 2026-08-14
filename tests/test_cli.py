from pathlib import Path

import pytest

from adapt1_fle.adapt.domain import STRATEGIES, FactorioDomain
from adapt1_fle.cli import (
    _arm_settings,
    balanced_strategy_schedule,
    build_parser,
    comparison_fingerprint,
    execute_run,
)
from adapt1_fle.config import RunMode, Settings
from adapt1_fle.curriculum import BenchmarkArm
from adapt1_fle.ledger import RunLedger


def test_cli_parses_static_baseline_run() -> None:
    arguments = build_parser().parse_args(
        [
            "run",
            "--mode",
            "baseline",
            "--env-id",
            "iron_ore_throughput",
            "--steps",
            "1",
            "--static-policy",
        ]
    )

    assert arguments.command == "run"
    assert arguments.mode == "baseline"
    assert arguments.steps == 1
    assert arguments.static_policy is True


def test_cold_online_arm_isolates_domain_and_disables_memory() -> None:
    settings = Settings(adapt_api_key="secret")

    cold, domain_enabled, memory_enabled = _arm_settings(
        settings,
        arm=BenchmarkArm.COLD_ONLINE,
        run_id="run-12345678",
        env_id="iron_ore_throughput",
        steps=8,
    )

    assert cold.mode is RunMode.TRAIN
    assert cold.domain_id.endswith("12345678")
    assert domain_enabled is True
    assert memory_enabled is False


def test_comparison_fingerprint_excludes_run_and_arm_identity() -> None:
    baseline = Settings(mode=RunMode.BASELINE, run_id="baseline-run")
    frozen = Settings(
        mode=RunMode.FROZEN,
        run_id="frozen-run",
        domain_id="different-runtime-domain",
    )

    assert comparison_fingerprint(
        baseline,
        domain_contract_hash="contract",
        generator_id="model",
    ) == comparison_fingerprint(
        frozen,
        domain_contract_hash="contract",
        generator_id="model",
    )


def test_comparison_fingerprint_uses_actual_generator() -> None:
    settings = Settings()

    assert comparison_fingerprint(
        settings,
        domain_contract_hash="contract",
        generator_id="static-smoke-policy",
    ) != comparison_fingerprint(
        settings,
        domain_contract_hash="contract",
        generator_id=settings.model,
    )


def test_balanced_strategy_schedule_is_seeded_and_complete() -> None:
    first = balanced_strategy_schedule(seed=42, episode_ordinal=3, steps=12)
    second = balanced_strategy_schedule(seed=42, episode_ordinal=3, steps=12)

    assert first == second
    assert set(first) == set(STRATEGIES)
    assert len(first) == len(set(first)) == 12


def test_memory_profile_arms_are_isolated() -> None:
    settings = Settings(adapt_api_key="secret", memory_scope="domain")

    positive, domain_enabled, memory_enabled = _arm_settings(
        settings,
        arm=BenchmarkArm.WARM_POSITIVE,
        run_id="run-positive",
        env_id="iron_ore_throughput",
        steps=12,
        model_seed=812000,
    )
    diagnostic, _, _ = _arm_settings(
        settings,
        arm=BenchmarkArm.WARM_DIAGNOSTIC,
        run_id="run-diagnostic",
        env_id="iron_ore_throughput",
        steps=12,
        model_seed=812000,
    )

    assert domain_enabled is memory_enabled is True
    assert positive.memory_profile == "positive_only"
    assert diagnostic.memory_profile == "failure_diagnostic"
    assert positive.model_seed == diagnostic.model_seed == 812000


async def test_adapt_setup_failure_records_terminal_lifecycle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings(
        adapt_api_key="secret",
        ledger_root=tmp_path,
        mode=RunMode.FROZEN,
        run_id="setup-failure",
        trajectory_length=1,
    )

    async def fail_domain_setup(
        self: FactorioDomain,
        *,
        create_if_missing: bool = True,
    ) -> tuple[str, object]:
        raise RuntimeError("authentication setup failed")

    monkeypatch.setattr(FactorioDomain, "ensure", fail_domain_setup)

    with pytest.raises(RuntimeError, match="authentication setup failed"):
        await execute_run(
            settings,
            static_policy=True,
            enable_domain=True,
            enable_memory=False,
            benchmark_arm=BenchmarkArm.DOMAIN_ONLY.value,
        )

    ledger = RunLedger.open(tmp_path / "setup-failure")
    events = list(ledger.read_events())
    assert [event["kind"] for event in events] == ["failure", "completion"]
    assert events[0]["phase"] == "setup"
    assert events[1]["status"] == "failed"
