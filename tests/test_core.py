from app.embeddings import local_embedding
from app.retrieval import bm25_scores, reciprocal_rank_fusion
from app.text_processing import chunk_pages, normalize_text


def test_normalize_fixes_hyphenated_line_break_and_whitespace():
    assert normalize_text("inter-\n national   study") == "international study"


def test_chunk_pages_preserves_page_numbers_and_overlap():
    page = "alpha beta gamma " * 45
    chunks = chunk_pages([page, "second page text"], chunk_size=150, overlap=30)
    assert len(chunks) > 2
    assert chunks[0].page_number == 1
    assert any(chunk.page_number == 2 for chunk in chunks)
    first, second = chunks[0], chunks[1]
    assert first.content[-20:] in second.content


def test_chunk_pages_rejects_invalid_overlap():
    try:
        chunk_pages(["text"], chunk_size=120, overlap=120)
    except ValueError as exc:
        assert "overlap" in str(exc)
    else:
        raise AssertionError("invalid overlap should fail")


def test_bm25_ranks_matching_document_higher():
    scores = bm25_scores("mitochondria energy", ["A mitochondria produces cellular energy", "A neuron transmits signals"])
    assert scores[0] > scores[1]


def test_rrf_combines_multiple_rankings():
    scores = reciprocal_rank_fusion([["a", "b"], ["b", "c"]])
    assert scores["b"] > scores["a"] > scores["c"]


def test_local_embedding_is_repeatable_and_normalized():
    first = local_embedding("mitochondria make energy", 128)
    assert first == local_embedding("mitochondria make energy", 128)
    assert abs(sum(value * value for value in first) - 1) < 1e-6
