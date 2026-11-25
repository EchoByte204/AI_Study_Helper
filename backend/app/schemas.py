from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    filename: str
    page_count: int
    chunk_count: int
    figure_count: int = 0
    created_at: datetime


class Citation(BaseModel):
    number: int
    document_id: str
    filename: str
    page: int
    excerpt: str
    score: float
    kind: str = "text"
    image_url: str | None = None


class FigureOut(BaseModel):
    number: int
    id: str
    document_id: str
    filename: str
    page_number: int
    caption: str
    image_url: str
    score: float


class AskRequest(BaseModel):
    question: str = Field(min_length=2, max_length=4000)
    session_id: str | None = None
    top_k: int | None = Field(default=None, ge=1, le=20)
    explanation_level: str = Field(default="detailed", pattern="^(simple|clear|detailed)$")


class AnswerClaim(BaseModel):
    text: str
    source_numbers: list[int] = Field(default_factory=list)


class AnswerSection(BaseModel):
    title: str
    content: list[AnswerClaim] = Field(default_factory=list)


class StudyGuideOut(BaseModel):
    document_id: str
    filename: str
    slide_count: int
    slides_processed: int
    coverage_complete: bool
    empty_slides: list[int] = Field(default_factory=list)
    sections: list[AnswerSection] = Field(default_factory=list)
    figures: list[FigureOut] = Field(default_factory=list)


class TaskOut(BaseModel):
    task_id: str
    task_type: Literal["upload", "study_guide"]
    status: Literal["queued", "running", "completed", "failed"]
    progress: int = Field(ge=0, le=100)
    message: str
    document: DocumentOut | None = None
    study_guide: StudyGuideOut | None = None
    error: str | None = None


class StructuredAnswer(BaseModel):
    direct_answer: AnswerClaim
    sections: list[AnswerSection] = Field(default_factory=list)
    explanation: list[AnswerClaim] = Field(default_factory=list)
    formula: AnswerClaim | None = None
    formula_terms: list[AnswerClaim] = Field(default_factory=list)
    additional_context: list[str] = Field(default_factory=list)
    key_points: list[AnswerClaim] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    abstained: bool = False


class AskResponse(BaseModel):
    answer: str
    structured_answer: StructuredAnswer
    citations: list[Citation]
    figures: list[FigureOut] = Field(default_factory=list)
    session_id: str
    retrieval_queries: list[str]
    tool_used: str


class QuizRequest(BaseModel):
    topic: str = Field(min_length=2, max_length=1000)
    count: int = Field(default=5, ge=1, le=15)
    top_k: int | None = Field(default=None, ge=1, le=20)


class QuizQuestion(BaseModel):
    question: str
    answer: str
    evidence: str
    citation: Citation | None = None


class QuizResponse(BaseModel):
    topic: str
    questions: list[QuizQuestion]
    citations: list[Citation]


class EvaluateRequest(BaseModel):
    question: str = Field(min_length=2, max_length=2000)
    student_answer: str = Field(min_length=1, max_length=4000)
    top_k: int | None = Field(default=None, ge=1, le=20)


class EvaluateResponse(BaseModel):
    score: float = Field(ge=0, le=1)
    verdict: str
    feedback: str
    missing_points: list[str]
    citations: list[Citation]


class SessionMessage(BaseModel):
    role: str
    content: str
    created_at: datetime
    structured_answer: StructuredAnswer | None = None


class SessionSummary(BaseModel):
    session_id: str
    title: str
    updated_at: datetime


class SessionOut(BaseModel):
    session_id: str
    messages: list[SessionMessage]
