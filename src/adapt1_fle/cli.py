"""Command-line interface for training, evaluation, and operations."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import importlib.resources
import json
import os
import socket
import subprocess
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fle.env.gym_env.registry import get_environment_info, list_available_environments
from fle.env.utils.controller_loader.system_prompt_generator import SystemPromptGenerator

from adapt1_fle.adapt.client import AdaptClient
from adapt1_fle.adapt.domain import FactorioDomain, load_domain_definition
from adapt1_fle.adapt.memory import FactorioMemory
from adapt1_fle.agent.controller import AdaptiveController
from adapt1_fle.agent.model import FLEPolicyGenerator, StaticPolicyGenerator
from adapt1_fle.agent.prompt import ConversationWindow, build_system_prompt
from adapt1_fle.config import RunMode, Settings
from adapt1_fle.curriculum import (
    BenchmarkArm,
    load_curriculum,
    pending_curriculum_jobs,
)
from adapt1_fle.evaluation import build_benchmark_summary
from adapt1_fle.ledger import RunLedger
from adapt1_fle.models import RunCompletion
from adapt1_fle.report import write_report
from adapt1_fle.runner import TrajectoryRunner

DEFAULT_CONFIG = "configs/research-preview.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="adapt1-fle",
        description="REI Adapt-1 research preview for Factorio Learning Environment",
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("doctor", help="validate FLE, Adapt-1, and model setup")

    domain = commands.add_parser("domain", help="manage the versioned Adapt Domain")
    domain_commands = domain.add_subparsers(dest="domain_command", required=True)
    domain_commands.add_parser("create")
    domain_commands.add_parser("status")
    delete = domain_commands.add_parser("delete")
    delete.add_argument("--yes", action="store_true", help="confirm destructive deletion")

    run = commands.add_parser("run", help="run one Factorio trajectory")
    _add_run_arguments(run)
    run.add_argument("--benchmark-arm")
    run.add_argument("--experiment-id")

    train = commands.add_parser("train", help="run or resume curriculum training")
    train.add_argument("--curriculum", default="configs/curriculum.v1.yaml")
    train.add_argument("--limit", type=int, default=None)
    train.add_argument("--steps", type=int, default=None)
    train.add_argument("--static-policy", action="store_true")

    evaluate = commands.add_parser("evaluate", help="run controlled benchmark arms")
    evaluate.add_argument("--curriculum", default="configs/curriculum.v1.yaml")
    evaluate.add_argument(
        "--arm",
        action="append",
        choices=[arm.value for arm in BenchmarkArm],
    )
    evaluate.add_argument("--task", action="append")
    evaluate.add_argument("--episodes", type=int, default=1)
    evaluate.add_argument("--steps", type=int, default=None)
    evaluate.add_argument("--static-policy", action="store_true")

    report = commands.add_parser("report", help="aggregate immutable run ledgers")
    report.add_argument("--ledger-root", default=None)
    report.add_argument("--experiment-id", default=None)
    report.add_argument("--output", default="reports/latest")
    return parser


def _add_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--mode", choices=[mode.value for mode in RunMode], default=None)
    parser.add_argument("--env-id", default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--static-policy", action="store_true")
    parser.add_argument("--no-domain", action="store_true")
    parser.add_argument("--no-memory", action="store_true")


def main(argv: list[str] | None = None) -> None:
    load_dotenv()
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        exit_code = asyncio.run(dispatch(arguments))
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        exit_code = 130
    except Exception as error:
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        exit_code = 1
    raise SystemExit(exit_code)


async def dispatch(arguments: argparse.Namespace) -> int:
    if arguments.command == "doctor":
        return await doctor(arguments.config)
    if arguments.command == "domain":
        return await domain_command(arguments)
    if arguments.command == "run":
        return await run_command(arguments)
    if arguments.command == "train":
        return await train_command(arguments)
    if arguments.command == "evaluate":
        return await evaluate_command(arguments)
    if arguments.command == "report":
        return report_command(arguments)
    raise ValueError(f"unsupported command: {arguments.command}")


async def doctor(config_path: str) -> int:
    settings = Settings.load(config_path)
    checks: dict[str, Any] = {}
    checks["python"] = {
        "ok": sys.version_info[:2] == (3, 12),
        "version": sys.version.split()[0],
    }
    fle_version = importlib.metadata.version("factorio-learning-environment")
    environments = list_available_environments()
    checks["fle"] = {
        "ok": settings.env_id in environments,
        "version": fle_version,
        "environment_count": len(environments),
        "configured_environment": settings.env_id,
    }
    checks["factorio"] = {
        "ok": _tcp_available("127.0.0.1", 27000),
        "rcon_address": "127.0.0.1:27000",
    }
    key = _api_key(settings)
    async with AdaptClient(
        base_url=settings.adapt_base_url,
        api_key=key,
        timeout_seconds=settings.request_timeout_seconds,
        read_retry_attempts=settings.read_retry_attempts,
    ) as client:
        health, _ = await client.health()
        version, _ = await client.version()
        auth_ok = False
        auth_error = None
        if key:
            try:
                await client.list_domains()
                auth_ok = True
            except Exception as error:
                auth_error = f"{type(error).__name__}: {error}"
        checks["adapt_1"] = {
            "ok": bool(health.get("ok")),
            "health": health,
            "version": version,
            "credential_present": bool(key),
            "authentication_ok": auth_ok,
            "authentication_error": auth_error,
        }
    credential_name = _model_credential_name(settings.model)
    model_credential_present = (
        bool(os.getenv(credential_name)) or credential_name == "OLLAMA_API_KEY"
    )
    checks["model"] = {
        "model": settings.model,
        "credential_name": credential_name,
        "credential_present": model_credential_present,
    }
    checks["ledger"] = {
        "ok": _writable_parent(settings.ledger_root),
        "path": str(settings.ledger_root),
    }
    required_ok = (
        checks["python"]["ok"]
        and checks["fle"]["ok"]
        and checks["factorio"]["ok"]
        and checks["adapt_1"]["ok"]
        and (not settings.adapt_enabled or checks["adapt_1"]["authentication_ok"])
        and checks["model"]["credential_present"]
        and checks["ledger"]["ok"]
    )
    checks["ok"] = required_ok
    print(json.dumps(checks, indent=2, sort_keys=True))
    return 0 if required_ok else 1


async def domain_command(arguments: argparse.Namespace) -> int:
    settings = Settings.load(arguments.config)
    key = _api_key(settings)
    if not key:
        raise ValueError("Domain operations require ADAPT1_API_KEY or a REI key alias")
    definition = load_domain_definition(settings.domain_config_path)
    async with _client(settings) as client:
        domain = FactorioDomain(
            client,
            definition,
            domain_id=settings.domain_id,
            top_k=settings.adapt_top_k,
        )
        if arguments.domain_command == "create":
            status, exchange = await domain.ensure()
            print(
                json.dumps(
                    {
                        "status": status,
                        "exchange": exchange.model_dump(mode="json"),
                    },
                    indent=2,
                )
            )
            return 0
        if arguments.domain_command == "status":
            response, _ = await client.get_domain(settings.domain_id)
            print(json.dumps(response, indent=2, sort_keys=True))
            return 0
        if arguments.domain_command == "delete":
            if not arguments.yes:
                raise ValueError("Domain deletion requires --yes")
            response, _ = await client.delete_domain(settings.domain_id)
            print(json.dumps(response, indent=2, sort_keys=True))
            return 0
    raise ValueError(f"unsupported domain command: {arguments.domain_command}")


async def run_command(arguments: argparse.Namespace) -> int:
    overrides = {
        "mode": arguments.mode,
        "env_id": arguments.env_id,
        "trajectory_length": arguments.steps,
        "run_id": arguments.run_id,
    }
    settings = Settings.load(arguments.config, overrides=overrides)
    completion, run_dir = await execute_run(
        settings,
        static_policy=arguments.static_policy,
        enable_domain=not arguments.no_domain,
        enable_memory=not arguments.no_memory,
        benchmark_arm=arguments.benchmark_arm,
        manifest_extra={"experiment_id": arguments.experiment_id},
    )
    print(
        json.dumps(
            {
                "completion": completion.model_dump(mode="json"),
                "run_dir": str(run_dir),
            },
            indent=2,
        )
    )
    return 0


async def train_command(arguments: argparse.Namespace) -> int:
    overrides = {"mode": RunMode.TRAIN, "trajectory_length": arguments.steps}
    settings = Settings.load(arguments.config, overrides=overrides)
    settings.validate_for_execution()
    curriculum = load_curriculum(
        arguments.curriculum,
        available_tasks=list_available_environments(),
    )
    jobs = pending_curriculum_jobs(curriculum, settings.ledger_root)
    if arguments.limit is not None:
        jobs = jobs[: arguments.limit]
    for job in jobs:
        run_id = _new_run_id(f"train-{job.env_id}")
        job_settings = _updated_settings(
            settings,
            {"env_id": job.env_id, "run_id": run_id},
        )
        completion, run_dir = await execute_run(
            job_settings,
            static_policy=arguments.static_policy,
            enable_domain=True,
            enable_memory=settings.memory_enabled,
            benchmark_arm="curriculum_train",
            manifest_extra={
                "curriculum_revision": curriculum.revision,
                "curriculum_job_id": job.job_id,
                "curriculum_stage": job.stage,
                "curriculum_ordinal": job.ordinal,
            },
        )
        print(
            f"{job.job_id}: {completion.status} score={completion.final_score:.3f} ledger={run_dir}"
        )
    if not jobs:
        print("Curriculum already complete.")
    return 0


async def evaluate_command(arguments: argparse.Namespace) -> int:
    settings = Settings.load(arguments.config)
    curriculum = load_curriculum(
        arguments.curriculum,
        available_tasks=list_available_environments(),
    )
    arms = arguments.arm or [BenchmarkArm.BASELINE.value, BenchmarkArm.WARM_FROZEN.value]
    tasks = arguments.task or curriculum.held_out_tasks
    experiment_id = _new_run_id("experiment")
    for arm_name in arms:
        arm = BenchmarkArm(arm_name)
        for env_id in tasks:
            for episode in range(arguments.episodes):
                run_id = _new_run_id(f"eval-{arm.value}-{env_id}")
                arm_settings, domain_enabled, memory_enabled = _arm_settings(
                    settings,
                    arm=arm,
                    run_id=run_id,
                    env_id=env_id,
                    steps=arguments.steps,
                )
                completion, run_dir = await execute_run(
                    arm_settings,
                    static_policy=arguments.static_policy,
                    enable_domain=domain_enabled,
                    enable_memory=memory_enabled,
                    benchmark_arm=arm.value,
                    manifest_extra={
                        "held_out": env_id in curriculum.held_out_tasks,
                        "evaluation_episode": episode,
                        "curriculum_revision": curriculum.revision,
                        "experiment_id": experiment_id,
                    },
                )
                print(
                    f"{arm.value}/{env_id}/{episode}: {completion.status} "
                    f"score={completion.final_score:.3f} ledger={run_dir}"
                )
    return 0


def report_command(arguments: argparse.Namespace) -> int:
    settings = Settings.load(arguments.config)
    ledger_root = arguments.ledger_root or settings.ledger_root
    summary = build_benchmark_summary(
        ledger_root,
        experiment_id=arguments.experiment_id,
    )
    json_path, markdown_path = write_report(summary, arguments.output)
    print(
        json.dumps(
            {
                "runs": summary.total_runs,
                "json": str(json_path),
                "markdown": str(markdown_path),
            },
            indent=2,
        )
    )
    return 0


async def execute_run(
    settings: Settings,
    *,
    static_policy: bool,
    enable_domain: bool,
    enable_memory: bool,
    benchmark_arm: str | None,
    manifest_extra: dict[str, Any] | None = None,
) -> tuple[RunCompletion, Path]:
    settings.validate_for_execution()
    run_id = settings.run_id or _new_run_id(settings.env_id)
    episode_id = f"{run_id}-episode-000"
    definition = load_domain_definition(settings.domain_config_path)
    manifest: dict[str, Any] = {
        "run_id": run_id,
        "episode_id": episode_id,
        "created_at": datetime.now(UTC).isoformat(),
        "mode": settings.mode.value,
        "benchmark_arm": benchmark_arm,
        "env_id": settings.env_id,
        "model": "static-smoke-policy" if static_policy else settings.model,
        "trajectory_length": settings.trajectory_length,
        "domain_id": settings.domain_id if enable_domain else None,
        "domain_revision": definition.revision if enable_domain else None,
        "domain_contract_hash": definition.contract_hash if enable_domain else None,
        "memory_enabled": enable_memory,
        "fle_version": importlib.metadata.version("factorio-learning-environment"),
        "harness_version": importlib.metadata.version("adapt1-fle"),
        "git_sha": _git_sha(),
        "settings": settings.safe_dump(),
        "config_hash": config_hash(settings),
        "comparison_fingerprint": comparison_fingerprint(
            settings,
            domain_contract_hash=definition.contract_hash,
        ),
    }
    if manifest_extra:
        manifest.update(manifest_extra)
    ledger = RunLedger.create(settings.ledger_root, run_id, manifest)

    async with _client(settings) as client:
        domain = (
            FactorioDomain(
                client,
                definition,
                domain_id=settings.domain_id,
                top_k=settings.adapt_top_k,
            )
            if settings.adapt_enabled and enable_domain
            else None
        )
        if domain is not None:
            await domain.ensure(create_if_missing=settings.mode is RunMode.TRAIN)
        memory = (
            FactorioMemory(client, top_k=min(settings.adapt_top_k, 12))
            if settings.adapt_enabled and enable_memory
            else None
        )
        environment_info = get_environment_info(settings.env_id)
        if environment_info is None:
            raise ValueError(f"unknown FLE environment: {settings.env_id}")
        fle_prompt = SystemPromptGenerator(
            str(importlib.resources.files("fle") / "env")
        ).generate_for_agent(agent_idx=0, num_agents=1)
        system_prompt = build_system_prompt(
            fle_system_prompt=fle_prompt,
            goal=str(environment_info["description"]),
            trajectory_length=settings.trajectory_length,
        )
        generator = StaticPolicyGenerator() if static_policy else FLEPolicyGenerator(settings.model)
        controller = AdaptiveController(
            mode=settings.mode,
            run_id=run_id,
            domain_revision=definition.revision,
            generator=generator,
            conversation=ConversationWindow(
                system_prompt,
                max_messages=settings.max_messages,
            ),
            ledger=ledger,
            domain=domain,
            memory=memory,
        )
        runner = TrajectoryRunner(
            env_id=settings.env_id,
            trajectory_length=settings.trajectory_length,
            controller=controller,
            run_id=run_id,
            episode_id=episode_id,
        )
        completion = await runner.run()
    return completion, ledger.run_dir


def _arm_settings(
    settings: Settings,
    *,
    arm: BenchmarkArm,
    run_id: str,
    env_id: str,
    steps: int | None,
) -> tuple[Settings, bool, bool]:
    update: dict[str, Any] = {
        "run_id": run_id,
        "env_id": env_id,
    }
    if steps is not None:
        update["trajectory_length"] = steps

    if arm is BenchmarkArm.BASELINE:
        update["mode"] = RunMode.BASELINE
        return _updated_settings(settings, update), False, False
    if arm is BenchmarkArm.COLD_ONLINE:
        update["mode"] = RunMode.TRAIN
        update["domain_id"] = f"{settings.domain_id}-cold-{run_id[-8:]}"
        return _updated_settings(settings, update), True, False
    if arm is BenchmarkArm.WARM_FROZEN:
        update["mode"] = RunMode.FROZEN
        return _updated_settings(settings, update), True, True
    if arm is BenchmarkArm.DOMAIN_ONLY:
        update["mode"] = RunMode.FROZEN
        return _updated_settings(settings, update), True, False
    if arm is BenchmarkArm.MEMORY_ONLY:
        update["mode"] = RunMode.FROZEN
        return _updated_settings(settings, update), False, True
    raise ValueError(f"unsupported benchmark arm: {arm}")


def _client(settings: Settings) -> AdaptClient:
    return AdaptClient(
        base_url=settings.adapt_base_url,
        api_key=_api_key(settings),
        timeout_seconds=settings.request_timeout_seconds,
        read_retry_attempts=settings.read_retry_attempts,
    )


def _updated_settings(settings: Settings, update: dict[str, Any]) -> Settings:
    payload = settings.model_dump()
    payload.update(update)
    return Settings.model_validate(payload)


def _api_key(settings: Settings) -> str | None:
    return settings.adapt_api_key.get_secret_value() if settings.adapt_api_key is not None else None


def _new_run_id(prefix: str) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    safe_prefix = prefix.replace("_", "-")
    return f"{safe_prefix}-{timestamp}-{uuid.uuid4().hex[:8]}"


def _model_credential_name(model: str) -> str:
    if "/" in model or model.startswith("open-router"):
        return "OPEN_ROUTER_API_KEY"
    if model.startswith("claude"):
        return "ANTHROPIC_API_KEY"
    if model.startswith("openai"):
        return "OPENAI_API_KEY"
    if model.startswith("gemini"):
        return "GEMINI_API_KEY"
    if model.startswith("deepseek"):
        return "DEEPSEEK_API_KEY"
    if model.startswith("together"):
        return "TOGETHER_API_KEY"
    if model.startswith("ollama"):
        return "OLLAMA_API_KEY"
    return "MODEL_API_KEY"


def _tcp_available(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


def _writable_parent(path: Path) -> bool:
    parent = path if path.exists() else path.parent
    while not parent.exists() and parent != parent.parent:
        parent = parent.parent
    return os.access(parent, os.W_OK)


def _git_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def config_hash(settings: Settings) -> str:
    encoded = json.dumps(settings.safe_dump(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def comparison_fingerprint(
    settings: Settings,
    *,
    domain_contract_hash: str,
) -> str:
    contract = {
        "model": settings.model,
        "trajectory_length": settings.trajectory_length,
        "fle_version": importlib.metadata.version("factorio-learning-environment"),
        "domain_contract_hash": domain_contract_hash,
        "reward_contract": "factorio-public-reward-v1",
        "prompt_contract": "adapt1-fle-prompt-v1",
    }
    encoded = json.dumps(contract, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
