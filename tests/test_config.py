from pathlib import Path

import pytest

from adapt1_fle.config import RunMode, Settings


def test_load_precedence_and_redaction(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        "model: yaml-model\ntrajectory_length: 10\nmode: baseline\n",
        encoding="utf-8",
    )

    settings = Settings.load(
        config,
        environ={
            "ADAPT1_FLE_MODEL": "env-model",
            "ADAPT1_API_KEY": "super-secret",
        },
        overrides={"model": "override-model"},
    )

    assert settings.model == "override-model"
    assert settings.trajectory_length == 10
    assert settings.adapt_api_key is not None
    assert settings.adapt_api_key.get_secret_value() == "super-secret"
    assert settings.safe_dump()["adapt_api_key"] == "***"
    assert "super-secret" not in repr(settings)


def test_supported_rei_key_alias_is_loaded() -> None:
    settings = Settings.load(environ={"REI_UNIT_API_KEY": "alias-secret"})

    assert settings.adapt_api_key is not None
    assert settings.adapt_api_key.get_secret_value() == "alias-secret"


def test_adapt_execution_requires_key() -> None:
    settings = Settings(mode=RunMode.FROZEN)

    with pytest.raises(ValueError, match="ADAPT1_API_KEY"):
        settings.validate_for_execution()


def test_training_rejects_multiple_writers() -> None:
    with pytest.raises(ValueError, match="one writer"):
        Settings(mode=RunMode.TRAIN, max_workers=2)


def test_domain_id_rejects_whitespace() -> None:
    with pytest.raises(ValueError, match="whitespace"):
        Settings(domain_id="invalid domain")
