from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import Request

from astrbot.dashboard.api.knowledge_bases import retrieve_knowledge_base
from astrbot.dashboard.schemas import KnowledgeBaseRetrieveRequest


def test_retrieve_request_can_use_route_kb_name_as_default():
    payload = KnowledgeBaseRetrieveRequest(query="hello").canonical_payload(
        kb_name="Route KB"
    )

    assert payload == {"query": "hello", "kb_names": ["Route KB"]}


def test_retrieve_request_drops_deprecated_threshold_and_rerank():
    payload = KnowledgeBaseRetrieveRequest(
        query="hello",
        top_k=3,
        debug=True,
    ).canonical_payload(kb_name="Docs")

    assert payload == {
        "query": "hello",
        "top_k": 3,
        "debug": True,
        "kb_names": ["Docs"],
    }
    assert "threshold" not in payload
    assert "rerank" not in payload


@pytest.mark.asyncio
async def test_retrieve_route_resolves_path_kb_id_to_canonical_kb_names():
    service = MagicMock()
    service.get_kb = AsyncMock(return_value={"kb_name": "Route KB"})
    service.retrieve = AsyncMock(return_value={"results": [], "total": 0})
    payload = KnowledgeBaseRetrieveRequest(
        query="hello",
        top_k=2,
        debug=False,
    )

    response = await retrieve_knowledge_base(
        "route-kb-id",
        payload,
        _auth=object(),
        service=service,
    )

    assert response == {
        "status": "ok",
        "message": None,
        "data": {"results": [], "total": 0},
    }
    service.get_kb.assert_awaited_once_with("route-kb-id")
    service.retrieve.assert_awaited_once_with(
        {
            "query": "hello",
            "top_k": 2,
            "debug": False,
            "kb_names": ["Route KB"],
        }
    )


@pytest.mark.asyncio
async def test_retrieve_route_requires_resolved_kb_name():
    service = MagicMock()
    service.get_kb = AsyncMock(return_value={})
    service.retrieve = AsyncMock()

    response = await retrieve_knowledge_base(
        "route-kb-id",
        KnowledgeBaseRetrieveRequest(query="hello"),
        _auth=object(),
        service=service,
    )

    assert response["status"] == "error"
    assert response["message"] == "知识库不存在"
    service.retrieve.assert_not_awaited()
