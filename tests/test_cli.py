from adapt1_fle.cli import _arm_settings, build_parser, comparison_fingerprint
from adapt1_fle.config import RunMode, Settings
from adapt1_fle.curriculum import BenchmarkArm


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
    ) == comparison_fingerprint(
        frozen,
        domain_contract_hash="contract",
    )
