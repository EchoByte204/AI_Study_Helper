import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import BackgroundTasks, Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from sqlalchemy import func, select, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.config import get_settings
from app.database import get_db
from app.models import ConversationMessage, Document, Figure
from app.schemas import (
    AskRequest,
    AskResponse,
    DocumentOut,
    EvaluateRequest,
    EvaluateResponse,
    QuizRequest,
    QuizResponse,
    SessionMessage,
    SessionOut,
    SessionSummary,
    TaskOut,
)
from app.tasks import create_task, get_task, run_study_guide_task, run_upload_task
from app.tutor import TutorService, citation_list, figure_citation_list, figure_output_list

settings = get_settings()
logging.basicConfig(level=settings.log_level.upper())
logger = logging.getLogger("rag_tutor")


@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info("RAG tutor API starting")
    yield
    logger.info("RAG tutor API stopped")


app = FastAPI(
    title="Advanced Agentic RAG Learning Assistant",
    description="Ask questions about PDFs and PowerPoint files with page and slide citations.",
    version="0.1.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)


@app.exception_handler(SQLAlchemyError)
async def database_error_handler(_: Request, exc: SQLAlchemyError) -> JSONResponse:
    logger.exception("Database request failed", exc_info=exc)
    return JSONResponse(
        status_code=503,
        content={"detail": "Database schema or connection is unavailable. Confirm the database is running and migrations are current."},
    )


@app.exception_handler(Exception)
async def unexpected_error_handler(_: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unexpected API request failure", exc_info=exc)
    return JSONResponse(status_code=500, content={"detail": "Unexpected API error. Check the API logs for details."})


@app.get("/health")
def health(db: Session = Depends(get_db)) -> dict[str, str]:  # noqa: B008
    try:
        db.execute(text("SELECT 1"))
    except SQLAlchemyError as exc:
        raise HTTPException(status_code=503, detail="Database unavailable") from exc
    return {"status": "ok", "database": "ok"}


@app.post("/documents", response_model=TaskOut, status_code=202)
async def upload_document(background_tasks: BackgroundTasks, file: UploadFile = File(...)) -> TaskOut:  # noqa: B008
    if not file.filename or Path(file.filename).suffix.lower() not in {".pdf", ".pptx"}:
        raise HTTPException(status_code=415, detail="Upload a PDF or PowerPoint (.pptx) file")
    content = await file.read(settings.max_upload_mb * 1024 * 1024 + 1)
    if len(content) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File exceeds {settings.max_upload_mb} MB limit")
    task = create_task("upload", "Upload received")
    background_tasks.add_task(run_upload_task, task.task_id, file.filename, content)
    return task


@app.get("/tasks/{task_id}", response_model=TaskOut)
def read_task(task_id: str) -> TaskOut:
    task = get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found or API restarted")
    return task


@app.post("/documents/{document_id}/study-guide", response_model=TaskOut, status_code=202)
def create_document_study_guide(
    document_id: str, background_tasks: BackgroundTasks, db: Session = Depends(get_db)  # noqa: B008
) -> TaskOut:
    document = db.get(Document, document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Presentation not found")
    if not document.filename.lower().endswith((".pptx", ".pptx.pdf")):
        raise HTTPException(status_code=415, detail="Detailed study notes require a PowerPoint presentation")
    task = create_task("study_guide", "Preparing all slides")
    background_tasks.add_task(run_study_guide_task, task.task_id, document_id)
    return task


@app.get("/documents", response_model=list[DocumentOut])
def list_documents(db: Session = Depends(get_db)) -> list[Document]:  # noqa: B008
    return list(db.scalars(select(Document).order_by(Document.created_at.desc())))


@app.delete("/documents/{document_id}", status_code=204)
def delete_document(document_id: str, db: Session = Depends(get_db)) -> None:  # noqa: B008
    document = db.get(Document, document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    media_paths = list(db.scalars(select(Figure.storage_path).where(Figure.document_id == document_id)))
    db.delete(document)
    db.commit()
    for media_path in media_paths:
        try:
            Path(media_path).unlink(missing_ok=True)
        except OSError:
            logger.warning("Could not remove stored figure file: %s", media_path)
    media_folder = Path(settings.media_dir).resolve() / document_id
    if media_folder.exists():
        try:
            media_folder.rmdir()
        except OSError:
            logger.warning("Could not remove figure folder: %s", media_folder)


@app.get("/figures/{figure_id}/image")
def get_figure_image(figure_id: str, db: Session = Depends(get_db)) -> FileResponse:  # noqa: B008
    figure = db.get(Figure, figure_id)
    if not figure:
        raise HTTPException(status_code=404, detail="Figure not found")
    path = Path(figure.storage_path).resolve()
    media_root = Path(settings.media_dir).resolve()
    if not path.is_relative_to(media_root) or not path.is_file():
        raise HTTPException(status_code=404, detail="Figure image not found")
    return FileResponse(path, media_type=figure.media_type, headers={"Cache-Control": "private, max-age=3600"})


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest, db: Session = Depends(get_db)) -> AskResponse:  # noqa: B008
    service = TutorService(db)
    structured_answer, session_id, results, figures, queries, tool_used = service.answer(
        request.question,
        request.session_id,
        request.top_k or settings.top_k,
        request.explanation_level,
    )
    text_citations = citation_list(results)
    document_ids = {item.figure.document_id for item in figures}
    documents = {doc.id: doc for doc in db.scalars(select(Document).where(Document.id.in_(document_ids)))} if document_ids else {}
    figure_citations = figure_citation_list(figures, documents, len(text_citations) + 1)
    citations = text_citations + figure_citations
    figure_outputs = figure_output_list(figures, documents, len(text_citations) + 1)
    return AskResponse(
        answer=structured_answer.direct_answer.text,
        structured_answer=structured_answer,
        citations=citations,
        figures=figure_outputs,
        session_id=session_id,
        retrieval_queries=queries,
        tool_used=tool_used,
    )


@app.post("/tutor/quiz", response_model=QuizResponse)
def create_quiz(request: QuizRequest, db: Session = Depends(get_db)) -> QuizResponse:  # noqa: B008
    service = TutorService(db)
    questions, results = service.quiz(request.topic, request.count, request.top_k or settings.top_k)
    if not results:
        raise HTTPException(status_code=404, detail="No relevant material found. Upload a PDF or refine the topic")
    return QuizResponse(topic=request.topic, questions=questions, citations=citation_list(results))


@app.post("/tutor/evaluate", response_model=EvaluateResponse)
def evaluate(request: EvaluateRequest, db: Session = Depends(get_db)) -> EvaluateResponse:  # noqa: B008
    service = TutorService(db)
    result, results = service.evaluate(request.question, request.student_answer, request.top_k or settings.top_k)
    if not results:
        raise HTTPException(status_code=404, detail="No relevant material found. Upload a PDF or refine the question")
    return result


@app.get("/sessions/{session_id}", response_model=SessionOut)
def get_session(session_id: str, db: Session = Depends(get_db)) -> SessionOut:  # noqa: B008
    messages = list(
        db.scalars(
            select(ConversationMessage)
            .where(ConversationMessage.session_id == session_id)
            .order_by(ConversationMessage.created_at)
        )
    )
    return SessionOut(
        session_id=session_id,
        messages=[SessionMessage(role=message.role, content=message.content, created_at=message.created_at, structured_answer=message.structured_answer) for message in messages],
    )


@app.get("/sessions", response_model=list[SessionSummary])
def list_sessions(db: Session = Depends(get_db)) -> list[SessionSummary]:  # noqa: B008
    rows = db.execute(
        select(
            ConversationMessage.session_id,
            func.min(ConversationMessage.created_at).label("started_at"),
            func.max(ConversationMessage.created_at).label("updated_at"),
        )
        .where(ConversationMessage.role == "user")
        .group_by(ConversationMessage.session_id)
        .order_by(func.max(ConversationMessage.created_at).desc())
        .limit(100)
    ).all()
    output = []
    for session_id, _, updated_at in rows:
        first_user_message = db.scalar(
            select(ConversationMessage.content)
            .where(ConversationMessage.session_id == session_id, ConversationMessage.role == "user")
            .order_by(ConversationMessage.created_at)
            .limit(1)
        ) or "New study session"
        title = " ".join(first_user_message.split())
        output.append(SessionSummary(session_id=session_id, title=title[:72], updated_at=updated_at))
    return output
