"""Tests for batch knowledge-base document deletion."""

import pytest
from unittest.mock import AsyncMock, MagicMock, call


def _build_helper():
    from astrbot.core.knowledge_base.kb_helper import KBHelper
    from astrbot.core.knowledge_base.models import KnowledgeBase

    kb = KnowledgeBase(
        kb_name="test-kb",
        kb_id="kb-test-1",
        embedding_provider_id="emb-1",
        chunk_size=512,
        chunk_overlap=50,
    )
    helper = KBHelper.__new__(KBHelper)
    helper.kb = kb
    helper.kb_db = AsyncMock()
    helper.vec_db = AsyncMock()
    helper.refresh_kb = AsyncMock()
    return helper


class TestBatchDeleteKbDb:
    """Verify batch delete at the kb_db_sqlite layer."""

    @pytest.mark.asyncio
    async def test_empty_list(self):
        """Empty list returns empty dict."""
        from astrbot.core.knowledge_base.kb_db_sqlite import KBSQLiteDatabase

        kb_db = KBSQLiteDatabase.__new__(KBSQLiteDatabase)
        vec_db = AsyncMock()

        results = await kb_db.delete_documents_by_ids([], vec_db)

        assert results == {}
        vec_db.delete_documents.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_batch_delete_success(self):
        """All documents deleted successfully."""
        from astrbot.core.knowledge_base.kb_db_sqlite import KBSQLiteDatabase

        kb_db = KBSQLiteDatabase.__new__(KBSQLiteDatabase)

        session = AsyncMock()
        session.__aenter__.return_value = session
        session.begin = MagicMock(return_value=session)
        kb_db.get_db = MagicMock(return_value=session)

        vec_db = AsyncMock()
        vec_db.delete_documents = AsyncMock()

        results = await kb_db.delete_documents_by_ids(
            ["doc-1", "doc-2", "doc-3"],
            vec_db,
        )

        assert results == {"doc-1": True, "doc-2": True, "doc-3": True}
        assert vec_db.delete_documents.await_count == 3

    @pytest.mark.asyncio
    async def test_partial_failure(self):
        """One vec_db failure doesn't block other deletions."""
        from astrbot.core.knowledge_base.kb_db_sqlite import KBSQLiteDatabase

        kb_db = KBSQLiteDatabase.__new__(KBSQLiteDatabase)

        session = AsyncMock()
        session.__aenter__.return_value = session
        session.begin = MagicMock(return_value=session)
        kb_db.get_db = MagicMock(return_value=session)

        vec_db = AsyncMock()

        async def _delete_side_effect(metadata_filters):
            if metadata_filters["kb_doc_id"] == "doc-2":
                raise RuntimeError("vec_db error")

        vec_db.delete_documents = AsyncMock(side_effect=_delete_side_effect)

        results = await kb_db.delete_documents_by_ids(
            ["doc-1", "doc-2", "doc-3"],
            vec_db,
        )

        assert results["doc-1"] is True
        assert results["doc-2"] is False
        assert results["doc-3"] is True


class TestBatchDeleteHelper:
    """Verify batch delete at the KBHelper layer."""

    @pytest.mark.asyncio
    async def test_delete_documents(self):
        """Helper calls kb_db and updates stats."""
        helper = _build_helper()
        helper.kb_db.delete_documents_by_ids = AsyncMock(
            return_value={"doc-1": True, "doc-2": True}
        )
        helper.kb_db.update_kb_stats = AsyncMock()

        results = await helper.delete_documents(["doc-1", "doc-2"])

        assert results == {"doc-1": True, "doc-2": True}
        helper.kb_db.delete_documents_by_ids.assert_awaited_once()
        helper.kb_db.update_kb_stats.assert_awaited_once()
        helper.refresh_kb.assert_awaited_once()
