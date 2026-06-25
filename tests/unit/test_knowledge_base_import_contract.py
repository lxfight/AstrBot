from astrbot.dashboard.schemas import (
    KnowledgeBaseImportRequest,
    KnowledgeBaseUrlImportRequest,
)


def test_import_request_canonical_payload():
    payload = KnowledgeBaseImportRequest(
        documents=[{"file_name": "a.txt", "chunks": ["hi"]}],
        batch_size=8,
    ).canonical_payload()

    assert payload == {
        "documents": [{"file_name": "a.txt", "chunks": ["hi"]}],
        "batch_size": 8,
    }


def test_url_import_request_canonical_payload_drops_deprecated_urls():
    payload = KnowledgeBaseUrlImportRequest(
        url="https://example.com",
        urls=["https://unused.example.com"],
        enable_cleaning=True,
        cleaning_provider_id="llm-1",
    ).canonical_payload()

    assert payload == {
        "url": "https://example.com",
        "enable_cleaning": True,
        "cleaning_provider_id": "llm-1",
    }
    assert "urls" not in payload
