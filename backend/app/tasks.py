import logging
from threading import Lock
from uuid import uuid4

from app.database import SessionLocal
from app.ingestion import IngestionError, ingest_document
from app.schemas import DocumentOut, TaskOut
from app.study_guide import create_study_guide

logger = logging.getLogger("rag_tutor.tasks")
_TASKS: dict[str, dict[str, object]] = {}
_TASKS_LOCK = Lock()


def create_task(task_type: str, message: str) -> TaskOut:
    task_id = str(uuid4())
    task = {
        "task_id": task_id,
        "task_type": task_type,
        "status": "queued",
        "progress": 0,
        "message": message,
        "document": None,
        "study_guide": None,
        "error": None,
    }
    with _TASKS_LOCK:
        if len(_TASKS) > 100:
            completed = [key for key, value in _TASKS.items() if value["status"] in {"completed", "failed"}]
            for key in completed[:25]:
                _TASKS.pop(key, None)
        _TASKS[task_id] = task
    return TaskOut.model_validate(task)


def get_task(task_id: str) -> TaskOut | None:
    with _TASKS_LOCK:
        task = _TASKS.get(task_id)
        return TaskOut.model_validate(dict(task)) if task else None


def update_task(task_id: str, **changes: object) -> None:
    with _TASKS_LOCK:
        task = _TASKS.get(task_id)
        if task:
            task.update(changes)


def _progress(task_id: str, progress: int, message: str) -> None:
    update_task(task_id, status="running", progress=progress, message=message)


def run_upload_task(task_id: str, filename: str, content: bytes) -> None:
    update_task(task_id, status="running", progress=11, message="Starting document processing")
    db = SessionLocal()
    try:
        document = ingest_document(
            db,
            filename,
            content,
            progress_callback=lambda value, message: _progress(task_id, value, message),
        )
        update_task(
            task_id,
            status="completed",
            progress=100,
            message="Document ready",
            document=DocumentOut.model_validate(document).model_dump(mode="json"),
        )
    except IngestionError as exc:
        db.rollback()
        update_task(task_id, status="failed", progress=100, message="Upload could not be processed", error=str(exc))
    except Exception:
        db.rollback()
        logger.exception("Background document ingestion failed for %s", filename)
        update_task(task_id, status="failed", progress=100, message="Upload could not be processed", error="Document ingestion failed. Check API logs for details.")
    finally:
        db.close()


def run_study_guide_task(task_id: str, document_id: str) -> None:
    update_task(task_id, status="running", progress=3, message="Preparing all slides")
    db = SessionLocal()
    try:
        guide = create_study_guide(
            db,
            document_id,
            progress_callback=lambda value, message: _progress(task_id, value, message),
        )
        update_task(
            task_id,
            status="completed",
            progress=100,
            message=f"Finished reading all {guide.slides_processed} slides",
            study_guide=guide.model_dump(mode="json"),
        )
    except ValueError as exc:
        update_task(task_id, status="failed", progress=100, message="Could not create study notes", error=str(exc))
    except Exception:
        db.rollback()
        logger.exception("Could not create study notes for document %s", document_id)
        update_task(task_id, status="failed", progress=100, message="Could not create study notes", error="Study-note generation failed. Check API logs for details.")
    finally:
        db.close()
