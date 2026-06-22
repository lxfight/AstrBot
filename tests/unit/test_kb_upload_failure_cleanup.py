import sys
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from astrbot.core.exceptions import KnowledgeBaseUploadError
from astrbot.core.knowledge_base.kb_db_sqlite import KBSQLiteDatabase


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
async def test_upload_document_rolls_back_chunks_when_metadata_save_fails(
    stub_provider_manager_module,
):
    from astrbot.core.knowledge_base.kb_helper import KBHelper

    helper = KBHelper.__new__(KBHelper)
    helper.kb = SimpleNamespace(kb_id="kb-1")
    helper.vec_db = AsyncMock()
    helper.vec_db.count_documents.return_value = 0
    helper._ensure_vec_db = AsyncMock()

    db_context = MagicMock()
    db_context.__aenter__ = AsyncMock(side_effect=RuntimeError("metadata failed"))
    db_context.__aexit__ = AsyncMock()
    helper.kb_db = MagicMock()
    helper.kb_db.get_db.return_value = db_context
    helper.refresh_kb = AsyncMock()

    with pytest.raises(KnowledgeBaseUploadError, match="元数据保存失败"):
        await KBHelper.upload_document(
            helper,
            file_name="doc.txt",
            file_content=None,
            file_type="txt",
            chunk_size=512,
            chunk_overlap=50,
            batch_size=32,
            tasks_limit=3,
            max_retries=3,
            pre_chunked_text=["chunk"],
        )

    helper.vec_db.insert_batch.assert_awaited_once()
    doc_id = helper.vec_db.insert_batch.await_args.kwargs["metadatas"][0]["kb_doc_id"]
    helper.vec_db.delete_documents.assert_awaited_once_with(
        metadata_filters={"kb_doc_id": doc_id}
    )


@pytest.mark.asyncio
async def test_upload_document_rolls_back_metadata_when_refresh_fails(
    stub_provider_manager_module,
    tmp_path,
):
    from astrbot.core.knowledge_base.kb_helper import KBHelper

    helper = KBHelper.__new__(KBHelper)
    helper.kb = SimpleNamespace(kb_id="kb-1")
    helper.vec_db = AsyncMock()
    helper.vec_db.count_documents.return_value = 0
    helper._ensure_vec_db = AsyncMock()
    helper.kb_db = KBSQLiteDatabase(str(tmp_path / "kb.db"))
    await helper.kb_db.initialize()
    update_kb_stats = AsyncMock(wraps=helper.kb_db.update_kb_stats)
    helper.kb_db.update_kb_stats = update_kb_stats
    helper.refresh_kb = AsyncMock()
    helper.refresh_document = AsyncMock(side_effect=RuntimeError("refresh failed"))

    try:
        with pytest.raises(KnowledgeBaseUploadError, match="元数据更新失败"):
            await KBHelper.upload_document(
                helper,
                file_name="doc.txt",
                file_content=None,
                file_type="txt",
                chunk_size=512,
                chunk_overlap=50,
                batch_size=32,
                tasks_limit=3,
                max_retries=3,
                pre_chunked_text=["chunk"],
            )

        helper.vec_db.delete_documents.assert_awaited_once()
        doc_id = helper.vec_db.delete_documents.await_args.kwargs["metadata_filters"][
            "kb_doc_id"
        ]
        assert await helper.kb_db.get_document_by_id(doc_id) is None
        helper.kb_db.update_kb_stats.assert_awaited_once()
        helper.refresh_kb.assert_awaited_once()
    finally:
        await helper.kb_db.close()


@pytest.mark.asyncio
async def test_upload_document_updates_stats_after_document_refresh(
    stub_provider_manager_module,
):
    from astrbot.core.knowledge_base.kb_helper import KBHelper

    helper = KBHelper.__new__(KBHelper)
    helper.kb = SimpleNamespace(kb_id="kb-1")
    helper.vec_db = AsyncMock()
    helper._ensure_vec_db = AsyncMock()
    helper.kb_db = MagicMock()
    db_context = MagicMock()
    session = MagicMock()
    session.begin.return_value.__aenter__ = AsyncMock(return_value=None)
    session.begin.return_value.__aexit__ = AsyncMock(return_value=None)
    session.commit = AsyncMock()
    session.refresh = AsyncMock()
    db_context.__aenter__ = AsyncMock(return_value=session)
    db_context.__aexit__ = AsyncMock(return_value=None)
    helper.kb_db.get_db.return_value = db_context
    calls = []

    async def refresh_document(_doc_id):
        calls.append("refresh_document")

    async def update_kb_stats(**_kwargs):
        calls.append("update_kb_stats")

    helper.refresh_document = refresh_document
    helper.kb_db.update_kb_stats = update_kb_stats
    helper.refresh_kb = AsyncMock()

    await KBHelper.upload_document(
        helper,
        file_name="doc.txt",
        file_content=None,
        file_type="txt",
        chunk_size=512,
        chunk_overlap=50,
        batch_size=32,
        tasks_limit=3,
        max_retries=3,
        pre_chunked_text=["chunk"],
    )

    assert calls == ["refresh_document", "update_kb_stats"]
