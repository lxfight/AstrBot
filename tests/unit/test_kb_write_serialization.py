import asyncio
import sys
import types
from types import SimpleNamespace

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
async def test_kb_helper_write_lock_serializes_uploads(stub_provider_manager_module):
    from astrbot.core.knowledge_base.kb_helper import KBHelper

    helper = KBHelper.__new__(KBHelper)
    helper._write_lock = asyncio.Lock()
    active_uploads = 0
    max_active_uploads = 0

    async def upload_locked(**_kwargs):
        nonlocal active_uploads, max_active_uploads
        active_uploads += 1
        max_active_uploads = max(max_active_uploads, active_uploads)
        await asyncio.sleep(0.01)
        active_uploads -= 1
        return SimpleNamespace()

    helper._upload_document_locked = upload_locked

    await asyncio.gather(
        KBHelper.upload_document(helper, "one.txt", None, "txt"),
        KBHelper.upload_document(helper, "two.txt", None, "txt"),
    )

    assert max_active_uploads == 1
