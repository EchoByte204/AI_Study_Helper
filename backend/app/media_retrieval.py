from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.embeddings import EmbeddingService
from app.models import Figure
from app.retrieval import bm25_scores


@dataclass
class RetrievedFigure:
    figure: Figure
    score: float


def search_figures(
    db: Session, question: str, top_k: int = 3, document_ids: set[str] | None = None
) -> list[RetrievedFigure]:
    figure_query = select(Figure)
    if document_ids is not None:
        if not document_ids:
            return []
        figure_query = figure_query.where(Figure.document_id.in_(document_ids))
    figures = list(db.scalars(figure_query))
    if not figures:
        return []
    vector = EmbeddingService().embed([question])[0]
    distance = Figure.embedding.cosine_distance(vector)
    vector_query = select(Figure, (1.0 - distance).label("similarity"))
    if document_ids is not None:
        vector_query = vector_query.where(Figure.document_id.in_(document_ids))
    vector_rows = list(db.execute(vector_query.order_by(distance).limit(max(top_k * 4, 12))))
    vector_scores = {figure.id: float(similarity) for figure, similarity in vector_rows}
    lexical = bm25_scores(question, [figure.description for figure in figures])
    lexical_scores = {figure.id: score for figure, score in zip(figures, lexical, strict=True)}
    best_similarity = max(vector_scores.values(), default=0.0)
    best_lexical = max(lexical_scores.values(), default=0.0)
    if best_similarity < 0.20 and best_lexical <= 0:
        return []
    candidates = {figure.id: figure for figure, _ in vector_rows}
    candidates.update({figure.id: figure for figure in figures if lexical_scores[figure.id] > 0})
    ranked = sorted(
        candidates.values(),
        key=lambda figure: (vector_scores.get(figure.id, 0.0) * 0.7 + min(lexical_scores.get(figure.id, 0.0), 5) * 0.03),
        reverse=True,
    )
    rendered_slides = {
        (figure.document_id, figure.page_number)
        for figure in ranked
        if figure.description.startswith("Rendered slide from ")
    }
    if rendered_slides:
        ranked = [
            figure for figure in ranked
            if (figure.document_id, figure.page_number) not in rendered_slides
            or figure.description.startswith("Rendered slide from ")
        ]
    return [RetrievedFigure(figure, vector_scores.get(figure.id, 0.0)) for figure in ranked[:top_k]]
