import math
import re
from collections import defaultdict
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.embeddings import EmbeddingService
from app.llm import LLMService
from app.models import Chunk, Document

WORD_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "do", "does", "for", "from",
    "how", "i", "in", "is", "it", "of", "on", "or", "that", "the", "this", "to",
    "was", "what", "when", "where", "which", "who", "why", "with",
}


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in WORD_RE.findall(text)]


def bm25_scores(query: str, documents: list[str], k1: float = 1.5, b: float = 0.75) -> list[float]:
    """Small-corpus BM25 implementation; intentionally avoids another runtime service."""
    return bm25_scores_for_tokens(tokenize(query), [tokenize(document) for document in documents], k1=k1, b=b)


def bm25_scores_for_tokens(
    query_tokens: list[str], tokenized_documents: list[list[str]], k1: float = 1.5, b: float = 0.75
) -> list[float]:
    """Score a pre-tokenized corpus so rewritten queries don't repeat corpus work."""
    if not tokenized_documents:
        return []
    lengths = [len(document) for document in tokenized_documents]
    avg_len = sum(lengths) / max(1, len(lengths))
    frequencies: dict[str, int] = defaultdict(int)
    for tokens in tokenized_documents:
        for token in set(tokens):
            frequencies[token] += 1
    scores = []
    for tokens, length in zip(tokenized_documents, lengths):
        tf: dict[str, int] = defaultdict(int)
        for token in tokens:
            tf[token] += 1
        score = 0.0
        for term in set(query_tokens):
            df = frequencies.get(term, 0)
            if not df:
                continue
            idf = math.log(1 + (len(tokenized_documents) - df + 0.5) / (df + 0.5))
            term_frequency = tf.get(term, 0)
            denominator = term_frequency + k1 * (1 - b + b * length / max(avg_len, 1))
            score += idf * term_frequency * (k1 + 1) / denominator if denominator else 0
        scores.append(score)
    return scores


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, item_id in enumerate(ranking, start=1):
            scores[item_id] += 1.0 / (k + rank)
    return dict(scores)


def weighted_reciprocal_rank_fusion(
    rankings: list[list[str]], weights: list[float], k: int = 60
) -> dict[str, float]:
    """Fuse rankings while allowing the original question to outweigh rewrites."""
    scores: dict[str, float] = defaultdict(float)
    for ranking, weight in zip(rankings, weights):
        for rank, item_id in enumerate(ranking, start=1):
            scores[item_id] += weight / (k + rank)
    return dict(scores)


@dataclass
class RetrievedChunk:
    chunk: Chunk
    document: Document
    score: float


class RetrievalService:
    def __init__(self, db: Session, embeddings: EmbeddingService | None = None, llm: LLMService | None = None):
        self.db = db
        self.embeddings = embeddings or EmbeddingService()
        self.llm = llm or LLMService()

    def search(
        self, question: str, top_k: int = 5, document_ids: set[str] | None = None
    ) -> tuple[list[RetrievedChunk], list[str]]:
        queries = self.llm.rewrite_queries(question)
        # Keep the student's wording first and prevent duplicate rewrites from
        # getting extra votes in the fused ranking.
        queries = list(dict.fromkeys(" ".join(q.split()) for q in [question, *queries] if q.strip()))[:3]
        chunk_query = select(Chunk)
        if document_ids is not None:
            if not document_ids:
                return [], queries
            chunk_query = chunk_query.where(Chunk.document_id.in_(document_ids))
        chunks = list(
            self.db.scalars(chunk_query.order_by(Chunk.document_id, Chunk.page_number, Chunk.chunk_index, Chunk.id))
        )
        if not chunks:
            return [], queries
        doc_ids = {chunk.document_id for chunk in chunks}
        documents = {doc.id: doc for doc in self.db.scalars(select(Document).where(Document.id.in_(doc_ids)))}
        vector_rankings: list[list[str]] = []
        ranking_weights: list[float] = []
        similarities: dict[str, float] = defaultdict(float)
        candidate_limit = min(len(chunks), max(top_k * 8, 50))
        for query_index, (query, vector) in enumerate(zip(queries, self.embeddings.embed(queries))):
            distance = Chunk.embedding.cosine_distance(vector)
            ranked_rows = list(
                self.db.execute(
                    select(Chunk)
                    .where(Chunk.document_id.in_(doc_ids))
                    .add_columns((1.0 - distance).label("cosine_similarity"))
                    .order_by(distance)
                    .limit(candidate_limit)
                )
            )
            ranked = [row[0] for row in ranked_rows]
            vector_rankings.append([chunk.id for chunk in ranked])
            for chunk, similarity in ranked_rows:
                similarities[chunk.id] = max(similarities[chunk.id], float(similarity))
            ranking_weights.append(1.5 if query_index == 0 else 1.0)

        lexical_rankings: list[list[str]] = []
        lexical_scores: dict[str, float] = defaultdict(float)
        tokenized_chunks = [tokenize(chunk.content) for chunk in chunks]
        for query_index, query in enumerate(queries):
            scores = bm25_scores_for_tokens(tokenize(query), tokenized_chunks)
            for chunk, score in zip(chunks, scores):
                lexical_scores[chunk.id] = max(lexical_scores[chunk.id], score)
            lexical_rankings.append(
                [chunks[i].id for i in sorted(range(len(chunks)), key=lambda i: (-scores[i], i)) if scores[i] > 0][:candidate_limit]
            )
            ranking_weights.append(1.5 if query_index == 0 else 1.0)

        fused = weighted_reciprocal_rank_fusion(vector_rankings + lexical_rankings, ranking_weights)
        # Avoid answering from an arbitrary nearest neighbor when neither retrieval
        # path found a meaningful match. This is a conservative abstention gate.
        best_vector_similarity = max(similarities.values(), default=0.0)
        best_lexical_score = max(lexical_scores.values(), default=0.0)
        if best_vector_similarity < 0.20 and best_lexical_score <= 0:
            return [], queries

        by_id = {chunk.id: chunk for chunk in chunks}
        # Rerank the fused candidate set with query term coverage and vector rank evidence.
        query_terms = set(tokenize(question)) - STOP_WORDS
        candidates = [by_id[item_id] for item_id in fused]
        def rerank(chunk: Chunk) -> float:
            terms = set(tokenize(chunk.content)) - STOP_WORDS
            coverage = len(query_terms & terms) / max(len(query_terms), 1)
            return fused[chunk.id] + 0.015 * coverage + 0.005 * max(similarities.get(chunk.id, 0), 0)

        candidates.sort(key=lambda chunk: (-rerank(chunk), chunk.document_id, chunk.page_number, chunk.chunk_index, chunk.id))
        result = []
        for chunk in candidates:
            doc = documents.get(chunk.document_id)
            if doc:
                result.append(RetrievedChunk(chunk, doc, rerank(chunk)))
            if len(result) == top_k:
                break
        return result, queries
