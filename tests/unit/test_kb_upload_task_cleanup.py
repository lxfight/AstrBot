from types import SimpleNamespace

import pytest

from astrbot.dashboard.services.knowledge_base_service import (
    MAX_TASK_RESULTS,
    TASK_RESULT_TTL_SECONDS,
    KnowledgeBaseService,
)


def make_service() -> KnowledgeBaseService:
    service = KnowledgeBaseService.__new__(KnowledgeBaseService)
    service.core_lifecycle = SimpleNamespace(kb_manager=None)
    service.upload_progress = {}
    service.upload_tasks = {}
    return service


def test_finished_task_cleanup_removes_expired_results(monkeypatch):
    service = make_service()
    monkeypatch.setattr(
        "astrbot.dashboard.services.knowledge_base_service.time.monotonic",
        lambda: 10_000,
    )
    service.upload_tasks = {
        "old": {
            "status": "completed",
            "finished_at": 10_000 - TASK_RESULT_TTL_SECONDS - 1,
        },
        "processing": {"status": "processing"},
    }
    service.upload_progress = {
        "old": {"status": "completed"},
        "processing": {"status": "processing"},
    }

    service.cleanup_finished_tasks()

    assert "old" not in service.upload_tasks
    assert "old" not in service.upload_progress
    assert "processing" in service.upload_tasks


def test_finished_task_cleanup_caps_retained_results():
    service = make_service()
    now = 10_000
    service.upload_tasks = {
        f"task-{idx}": {
            "status": "completed",
            "finished_at": now - 100 + idx,
        }
        for idx in range(MAX_TASK_RESULTS + 2)
    }
    service.upload_progress = {
        task_id: {"status": "completed"} for task_id in service.upload_tasks
    }

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            "astrbot.dashboard.services.knowledge_base_service.time.monotonic",
            lambda: now,
        )
        service.cleanup_finished_tasks()

    assert len(service.upload_tasks) == MAX_TASK_RESULTS
    assert "task-0" not in service.upload_tasks
    assert "task-1" not in service.upload_tasks
