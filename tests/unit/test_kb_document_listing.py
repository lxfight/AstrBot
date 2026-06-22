from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from astrbot.core.knowledge_base.kb_db_sqlite import KBSQLiteDatabase
from astrbot.core.knowledge_base.models import KBDocument
from astrbot.dashboard.services.knowledge_base_service import KnowledgeBaseService


def make_service(kb_manager=None) -> KnowledgeBaseService:
    service = KnowledgeBaseService.__new__(KnowledgeBaseService)
    service.core_lifecycle = SimpleNamespace(kb_manager=kb_manager)
    service.upload_progress = {}
    service.upload_tasks = {}
    return service


@pytest.mark.asyncio
async def test_list_documents_returns_total_count():
    kb_helper = MagicMock()
    kb_helper.list_documents = AsyncMock(return_value=[])
    kb_helper.count_documents = AsyncMock(return_value=42)
    kb_manager = MagicMock()
    kb_manager.get_kb = AsyncMock(return_value=kb_helper)
    service = make_service(kb_manager)

    result = await service.list_documents(
        kb_id="kb-1",
        page=2,
        page_size=10,
        search="  txt  ",
    )

    assert result == {"items": [], "page": 2, "page_size": 10, "total": 42}
    kb_helper.list_documents.assert_awaited_once_with(
        offset=10,
        limit=10,
        search="txt",
    )
    kb_helper.count_documents.assert_awaited_once_with(search="txt")


@pytest.mark.asyncio
async def test_list_documents_filters_items_and_total_by_search(tmp_path):
    db = KBSQLiteDatabase(str(tmp_path / "kb.db"))
    await db.initialize()
    try:
        async with db.get_db() as session, session.begin():
            session.add_all(
                [
                    KBDocument(
                        doc_id="doc-1",
                        kb_id="kb-1",
                        doc_name="Product FAQ.txt",
                        file_type="txt",
                        file_size=1,
                        file_path="/tmp/product-faq.txt",
                    ),
                    KBDocument(
                        doc_id="doc-2",
                        kb_id="kb-1",
                        doc_name="Release Notes.md",
                        file_type="md",
                        file_size=1,
                        file_path="/tmp/release-notes.md",
                    ),
                    KBDocument(
                        doc_id="doc-3",
                        kb_id="kb-1",
                        doc_name="Manual.pdf",
                        file_type="pdf",
                        file_size=1,
                        file_path="/tmp/manual.pdf",
                    ),
                    KBDocument(
                        doc_id="doc-4",
                        kb_id="kb-2",
                        doc_name="Product FAQ.txt",
                        file_type="txt",
                        file_size=1,
                        file_path="/tmp/other-product-faq.txt",
                    ),
                ]
            )

        docs = await db.list_documents_by_kb("kb-1", offset=0, limit=10, search="FAQ")
        total = await db.count_documents_by_kb("kb-1", search="FAQ")
        type_docs = await db.list_documents_by_kb(
            "kb-1",
            offset=0,
            limit=10,
            search="pdf",
        )

        assert [doc.doc_id for doc in docs] == ["doc-1"]
        assert total == 1
        assert [doc.doc_id for doc in type_docs] == ["doc-3"]
    finally:
        await db.close()
