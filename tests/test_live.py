import os
import uuid

import pytest
from dotenv import load_dotenv

from adapt1_fle.adapt.client import AdaptClient
from adapt1_fle.adapt.domain import FactorioDomain, load_domain_definition
from adapt1_fle.config import Settings
from adapt1_fle.models import CompactState

load_dotenv()


@pytest.mark.live
async def test_live_adapt_health_and_authenticated_domain() -> None:
    if os.getenv("RUN_LIVE_TESTS") != "1":
        pytest.skip("set RUN_LIVE_TESTS=1 to enable external Adapt-1 smoke")
    settings = Settings.load()
    if settings.adapt_api_key is None:
        pytest.skip("Adapt-1 credential is not available")

    domain_id = f"adapt1-fle-live-{uuid.uuid4().hex}"
    definition = load_domain_definition("configs/domain.factorio.v1.yaml")
    async with AdaptClient(
        base_url=settings.adapt_base_url,
        api_key=settings.adapt_api_key.get_secret_value(),
    ) as client:
        health, _ = await client.health()
        assert health["ok"] is True
        domain = FactorioDomain(client, definition, domain_id=domain_id)
        try:
            status, _ = await domain.ensure()
            assert status == "created"
            selection = await domain.select(_live_state(), frozen=False)
            assert selection.policy
        finally:
            await client.delete_domain(domain_id)


def _live_state() -> CompactState:
    return CompactState(
        task_key="iron_ore_throughput",
        goal="Produce 16 iron ore per minute",
        target_item="iron-ore",
        phase="bootstrap",
        step=0,
        trajectory_length=1,
        tick=0,
        elapsed_seconds=0,
        score=0,
        automated_score=0,
        quota=16,
        progress=0,
    )
