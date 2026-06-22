from unittest.mock import AsyncMock, MagicMock

import pytest

from astrbot.core.knowledge_base.kb_db_sqlite import KBSQLiteDatabase


@pytest.mark.asyncio
async def test_delete_document_keeps_metadata_when_vector_delete_fails():
    db = KBSQLiteDatabase.__new__(KBSQLiteDatabase)
    db.get_document_by_id = AsyncMock(return_value=object())
    db.get_db = MagicMock()
    vec_db = AsyncMock()
    vec_db.delete_documents.side_effect = RuntimeError("delete failed")

    with pytest.raises(RuntimeError, match="delete failed"):
        await KBSQLiteDatabase.delete_document_by_id(db, "doc-1", vec_db)

    vec_db.delete_documents.assert_awaited_once_with(
        metadata_filters={"kb_doc_id": "doc-1"}
    )
    db.get_db.assert_not_called()
