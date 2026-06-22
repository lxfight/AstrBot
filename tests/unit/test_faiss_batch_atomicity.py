from unittest.mock import AsyncMock

import pytest

from astrbot.core.db.vec_db.faiss_impl.vec_db import FaissVecDB
from astrbot.core.exceptions import KnowledgeBaseUploadError


@pytest.mark.asyncio
async def test_insert_batch_rolls_back_document_rows_when_vector_insert_fails():
    vec_db = FaissVecDB.__new__(FaissVecDB)
    vec_db.embedding_provider = AsyncMock()
    vec_db.embedding_provider.get_embeddings_batch.return_value = [
        [0.1, 0.2],
        [0.3, 0.4],
    ]
    vec_db.document_storage = AsyncMock()
    vec_db.document_storage.insert_documents_batch.return_value = [1, 2]
    vec_db.embedding_storage = AsyncMock()
    vec_db.embedding_storage.dimension = 2
    vec_db.embedding_storage.insert_batch.side_effect = RuntimeError("faiss failed")

    with pytest.raises(RuntimeError, match="faiss failed"):
        await FaissVecDB.insert_batch(
            vec_db,
            contents=["chunk-1", "chunk-2"],
            metadatas=[{}, {}],
            ids=["doc-1", "doc-2"],
        )

    vec_db.embedding_storage.delete.assert_awaited_once_with([1, 2])
    assert [
        call.args[0]
        for call in vec_db.document_storage.delete_document_by_doc_id.await_args_list
    ] == ["doc-1", "doc-2"]


@pytest.mark.asyncio
async def test_insert_batch_validates_vectors_before_document_rows():
    vec_db = FaissVecDB.__new__(FaissVecDB)
    vec_db.embedding_provider = AsyncMock()
    vec_db.embedding_provider.get_embeddings_batch.return_value = [[0.1, 0.2, 0.3]]
    vec_db.document_storage = AsyncMock()
    vec_db.embedding_storage = AsyncMock()
    vec_db.embedding_storage.dimension = 2

    with pytest.raises(KnowledgeBaseUploadError, match="向量化失败"):
        await FaissVecDB.insert_batch(
            vec_db,
            contents=["chunk"],
            metadatas=[{}],
            ids=["doc-1"],
        )

    vec_db.document_storage.insert_documents_batch.assert_not_called()
    vec_db.embedding_storage.insert_batch.assert_not_called()
