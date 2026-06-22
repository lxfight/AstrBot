import asyncio
import sys
import types

import pytest


@pytest.fixture
def stub_provider_manager_module():
    original_module = sys.modules.get("astrbot.core.provider.manager")
    stub_module = types.ModuleType("astrbot.core.provider.manager")

    class ProviderManager: ...

    setattr(stub_module, "ProviderManager", ProviderManager)
    sys.modules["astrbot.core.provider.manager"] = stub_module

    try:
        yield
    finally:
        if original_module is not None:
            sys.modules["astrbot.core.provider.manager"] = original_module
        else:
            sys.modules.pop("astrbot.core.provider.manager", None)


@pytest.mark.asyncio
async def test_rate_limiter_serializes_call_slots(
    monkeypatch, stub_provider_manager_module
):
    from astrbot.core.knowledge_base.kb_helper import RateLimiter

    now = 10.0
    sleeps = []

    def monotonic():
        return now

    async def sleep(delay):
        nonlocal now
        sleeps.append(delay)
        now += delay

    monkeypatch.setattr(
        "astrbot.core.knowledge_base.kb_helper.time.monotonic", monotonic
    )
    monkeypatch.setattr("astrbot.core.knowledge_base.kb_helper.asyncio.sleep", sleep)
    limiter = RateLimiter(max_rpm=60)

    async def acquire_once():
        async with limiter:
            pass

    await asyncio.gather(acquire_once(), acquire_once())

    assert sleeps == [1.0]
    assert limiter.last_call_time == 11.0
