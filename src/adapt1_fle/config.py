"""Validated runtime configuration."""

from __future__ import annotations

import os
from collections.abc import Mapping
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator


class RunMode(StrEnum):
    """Supported state mutation modes."""

    BASELINE = "baseline"
    TRAIN = "train"
    FROZEN = "frozen"


ENVIRONMENT_FIELDS: dict[str, str] = {
    "ADAPT1_BASE_URL": "adapt_base_url",
    "ADAPT1_API_KEY": "adapt_api_key",
    "ADAPT1_DOMAIN_ID": "domain_id",
    "ADAPT1_DOMAIN_CONFIG": "domain_config_path",
    "ADAPT1_FLE_MODEL": "model",
    "ADAPT1_FLE_ENV_ID": "env_id",
    "ADAPT1_FLE_MODE": "mode",
    "ADAPT1_FLE_LEDGER_ROOT": "ledger_root",
    "ADAPT1_FLE_TRAJECTORY_LENGTH": "trajectory_length",
    "ADAPT1_FLE_MAX_MESSAGES": "max_messages",
    "ADAPT1_FLE_MAX_WORKERS": "max_workers",
    "ADAPT1_FLE_MEMORY_ENABLED": "memory_enabled",
    "ADAPT1_FLE_TOP_K": "adapt_top_k",
    "ADAPT1_FLE_TIMEOUT_SECONDS": "request_timeout_seconds",
    "ADAPT1_FLE_MODEL_SEED": "model_seed",
    "ADAPT1_FLE_MEMORY_PROFILE": "memory_profile",
    "ADAPT1_FLE_MEMORY_SCOPE": "memory_scope",
}

API_KEY_ALIASES = (
    "ADAPT1_API_KEY",
    "REI_API_KEY",
    "REI_SECRET_TOKEN",
    "REI_UNIT_API_KEY",
)

BOOLEAN_FIELDS = {"memory_enabled"}
INTEGER_FIELDS = {
    "trajectory_length",
    "max_messages",
    "max_workers",
    "adapt_top_k",
    "model_seed",
}
FLOAT_FIELDS = {"request_timeout_seconds"}


class Settings(BaseModel):
    """Application settings with explicit execution validation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    adapt_base_url: str = "https://rei-neuroadapt-api.reilabs.org"
    adapt_api_key: SecretStr | None = None
    domain_id: str = "factorio-strategy-v3"
    domain_config_path: Path = Path("configs/domain.factorio.v1.yaml")
    model: str = "ollama-qwen2.5-coder:7b"
    env_id: str = "iron_ore_throughput"
    mode: RunMode = RunMode.BASELINE
    ledger_root: Path = Path(".fle/adapt1/runs")
    trajectory_length: int = Field(default=64, ge=1, le=10_000)
    max_messages: int = Field(default=17, ge=3, le=101)
    max_workers: int = Field(default=1, ge=1, le=64)
    memory_enabled: bool = True
    adapt_top_k: int = Field(default=12, ge=1, le=100)
    request_timeout_seconds: float = Field(default=120.0, gt=0, le=300)
    read_retry_attempts: int = Field(default=4, ge=1, le=10)
    model_seed: int | None = Field(default=None, ge=0)
    memory_profile: str = "default"
    memory_scope: str = "task"
    run_id: str | None = None

    @field_validator("adapt_base_url")
    @classmethod
    def normalize_base_url(cls, value: str) -> str:
        return value.rstrip("/")

    @field_validator("domain_id")
    @classmethod
    def validate_domain_id(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned or any(character.isspace() for character in cleaned):
            raise ValueError("domain_id must be non-empty and contain no whitespace")
        return cleaned

    @field_validator("memory_profile")
    @classmethod
    def validate_memory_profile(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned or any(character.isspace() for character in cleaned):
            raise ValueError("memory_profile must be non-empty and contain no whitespace")
        return cleaned

    @field_validator("memory_scope")
    @classmethod
    def validate_memory_scope(cls, value: str) -> str:
        if value not in {"task", "domain"}:
            raise ValueError("memory_scope must be 'task' or 'domain'")
        return value

    @model_validator(mode="after")
    def validate_writer_count(self) -> Settings:
        if self.mode is RunMode.TRAIN and self.max_workers != 1:
            raise ValueError("training requires exactly one writer per Adapt-1 Domain")
        return self

    @property
    def adapt_enabled(self) -> bool:
        return self.mode is not RunMode.BASELINE

    @property
    def writes_enabled(self) -> bool:
        return self.mode is RunMode.TRAIN

    def validate_for_execution(self) -> None:
        """Validate credentials and state rules immediately before a run."""

        if self.adapt_enabled and self.adapt_api_key is None:
            raise ValueError(
                "Adapt-enabled runs require ADAPT1_API_KEY or a supported REI key alias"
            )

    def safe_dump(self) -> dict[str, Any]:
        """Return a JSON-safe, redacted representation."""

        dumped = self.model_dump(mode="json")
        dumped["adapt_api_key"] = "***" if self.adapt_api_key is not None else None
        return dumped

    @classmethod
    def load(
        cls,
        config_path: str | Path | None = None,
        *,
        overrides: Mapping[str, Any] | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> Settings:
        """Load YAML, then environment, then explicit overrides."""

        data: dict[str, Any] = {}
        if config_path is not None:
            path = Path(config_path)
            parsed = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            if not isinstance(parsed, dict):
                raise ValueError(f"configuration must be a mapping: {path}")
            data.update(parsed)

        environment = environ if environ is not None else os.environ
        for env_name, field_name in ENVIRONMENT_FIELDS.items():
            if env_name in environment:
                data[field_name] = _coerce_environment_value(field_name, environment[env_name])

        if "adapt_api_key" not in data:
            for alias in API_KEY_ALIASES:
                value = environment.get(alias)
                if value:
                    data["adapt_api_key"] = value
                    break

        if overrides:
            data.update({key: value for key, value in overrides.items() if value is not None})

        return cls.model_validate(data)


def _coerce_environment_value(field_name: str, value: str) -> Any:
    if field_name in BOOLEAN_FIELDS:
        normalized = value.strip().lower()
        if normalized not in {"true", "false", "1", "0", "yes", "no"}:
            raise ValueError(f"{field_name} must be a boolean")
        return normalized in {"true", "1", "yes"}
    if field_name in INTEGER_FIELDS:
        return int(value)
    if field_name in FLOAT_FIELDS:
        return float(value)
    return value
