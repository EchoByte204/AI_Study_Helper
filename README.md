# Advanced Agentic RAG Learning Assistant

An API-first learning assistant for studying course PDFs and PowerPoint presentations. It indexes page- and slide-aware text, figures, native PPT tables, and chart values; retrieves evidence with vector and BM25 search; and returns structured, detailed explanations. Students can also generate a navigable study guide from every slide in a selected PPT, make quizzes, and receive feedback on short answers.

## What is included

- **Ingestion:** PDF and PPTX upload, SHA-256 duplicate detection, page/slide-aware text extraction, native PPT table and chart extraction, repeated image de-duplication, overlapping chunks, and document metadata. Upload runs as a background task and reports progress.
- **Retrieval:** PostgreSQL + pgvector cosine search, BM25 lexical search, reciprocal-rank fusion, optional LLM query rewriting, and a lightweight evidence reranker. Each query is searched through both dense and lexical paths; RRF gives the original question 1.5× the weight of each rewrite, then reranks a widened candidate pool.
- **Tutor:** long-form answers with numbered sections, equations, definitions, worked examples, and tables when supported by the material. PowerPoint slides are used as the primary evidence when a deck is available. API responses retain slide references; the interface keeps them out of the prose. Greetings are handled without retrieval.
- **Chat history:** conversations are stored in PostgreSQL, listed in the sidebar by their first question, and can be reopened later. Starting a new study session keeps earlier conversations. Saved structured answers reopen in the detailed answer layout; older answers remain readable as plain text.
- **Study guides:** select a PPT in the library to process every slide in order. Long decks are processed in batches; the guide reports slide coverage, has an outline, supports expand/collapse, and shows relevant extracted figures beside sections.
- **Evaluation:** retrieval metrics, a calculation-checked Amdahl's Law regression set, and a CSV workflow for manual side-by-side model review. Instructor review is pending for the starter answer set; it is not presented as instructor-approved.
- **Client and deployment:** React/Vite UI, API and frontend containers, pgvector database, health checks, and environment examples.

The local embedding fallback is meant for development and demos. It uses hashed-token embeddings; configure a vision-capable chat provider for generated answers and image-aware explanations. Embedded PDF pictures, PDF pages containing vector drawings, rendered pages from PPTX-exported PDFs, and PPTX picture shapes are indexed with nearby page/slide text. Native PPT tables and cached chart series are extracted. SmartArt and scanned-page OCR are not supported yet. Study-guide coverage means every slide was passed through the extraction and generation pipeline; it does not guarantee every fact was interpreted correctly. Additional context may use well-established background knowledge and is labeled separately from slide-based content. Review important answers against the deck.

## Quick start (Docker)

Requirements: Docker Desktop with Compose.

1. Copy `.env.example` to `.env`. The defaults run with local fallback models. Add `OPENAI_API_KEY` to enable hosted embeddings, query rewriting, and model responses.
2. Start the stack:

   ```powershell
   docker compose up --build
   ```

3. Open the app at [http://localhost:8080](http://localhost:8080). The API is at [http://localhost:8000](http://localhost:8000), with Swagger UI at `/docs`.
4. Upload a text-based PDF or PPTX in the UI. A progress bar tracks extraction and indexing. Ask questions in the tutor, or use the sparkle action beside a PPT in the library to create a study guide from every slide.

The database is persisted in the `postgres_data` volume. Stop with `Ctrl+C`; run `docker compose down` to stop containers. Add `-v` only when you intend to delete the database volume.

## Local development

Requirements: Python 3.11+, Node.js 20+, and PostgreSQL 16 with the pgvector extension (or use the database service from Compose).

```powershell
Copy-Item .env.example .env
docker compose up -d db
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m alembic -c backend/alembic.ini upgrade head
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --app-dir backend
```

In a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

The Vite dev server proxies `/api` to the local API. You can use the API directly with `http://localhost:8000` as well.

## Configuration

See `.env.example` for all settings. Important variables:

| Variable | Purpose |
| --- | --- |
| `DATABASE_URL` | Host-side PostgreSQL URL. The Compose database is published on `localhost:5433` for local Alembic/Uvicorn commands; the API container connects internally to `db:5432`. |
| `OPENAI_API_KEY` | Optional API key for OpenAI-compatible embeddings and vision/chat. Empty means local embeddings and no generated tutor answer. |
| `OPENAI_BASE_URL` | Optional compatible API base URL. |
| `EMBEDDING_MODEL` | Embedding model name; defaults to `text-embedding-3-small`. |
| `EMBEDDING_DIMENSIONS` | Must match the selected embedding model and the database vector column (default 1536). |
| `EMBEDDING_BATCH_SIZE` / `EMBEDDING_WORKERS` | Upload embedding throughput. Defaults batch 256 passages and process up to 3 remote batches concurrently; lower workers if the provider rate-limits requests. |
| `CHAT_MODEL` | Chat model used for rewriting, agent tool calls, answers, quizzes, and evaluation. |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | Chunking limits in characters. |
| `TOP_K` | Number of source chunks returned by retrieval. |
| `CORS_ORIGINS` | Comma-separated browser origins. |

If you change `EMBEDDING_DIMENSIONS`, create a matching database migration before indexing documents. Do not switch embedding models/dimensions on an already populated index without re-embedding its documents.

## API overview

- `GET /health` — service and database readiness.
- `POST /documents` — upload a PDF or PPTX (`multipart/form-data`, field `file`). Returns `202 Accepted` with a task ID.
- `GET /tasks/{task_id}` — poll upload or study-guide progress and fetch the completed result.
- `POST /documents/{document_id}/study-guide` — process all slides from a selected PPT and generate detailed study notes.
- `GET /figures/{figure_id}/image` — view a figure extracted from a PDF or PPTX.
- `GET /documents` — list indexed documents.
- `DELETE /documents/{document_id}` — remove a document and its chunks.
- `POST /ask` — ask a detailed question. Body: `{"question":"…","session_id":null,"top_k":12,"explanation_level":"detailed"}`.
- `POST /tutor/quiz` — generate a quiz from retrieved material.
- `POST /tutor/evaluate` — evaluate a student's short answer against retrieved evidence.
- `GET /sessions/{session_id}` — read saved conversation turns.
- `GET /sessions` — list up to 100 saved conversations, newest first, with a title derived from each first question.

Example upload and question:

```bash
curl -F "file=@biology.pdf" http://localhost:8000/documents
curl -H "Content-Type: application/json" \
  -d '{"question":"What is natural selection?","top_k":5}' \
  http://localhost:8000/ask
```

Responses from `/ask` include a compatibility `answer` string, `structured_answer`, `citations`, and a `figures` array. The structured answer includes an opening answer, numbered `sections`, optional separately labeled `additional_context`, limitations, and an `abstained` flag. Section claims retain `source_numbers` that link to passages or figures in `citations`; the current UI hides citation chips to keep the prose uncluttered. The API still returns source filename, page/slide, excerpt, and figure metadata. The tutor removes unsupported claims where it can detect missing source labels; this is not a proof of correctness.

`POST /documents/{document_id}/study-guide` returns a task. Poll `GET /tasks/{task_id}` until complete; its `study_guide` result contains `slide_count`, `slides_processed`, `coverage_complete`, `empty_slides`, `sections`, and relevant figures. `coverage_complete` is true only after the pipeline has processed every slide. Large decks are split into batches to control prompt size.

Example `/ask` response structure:

```json
{
  "direct_answer": {"text": "…", "source_numbers": [1]},
  "sections": [
    {"title": "1. Core idea", "content": [{"text": "…", "source_numbers": [1, 2]}]}
  ],
  "additional_context": [],
  "limitations": [],
  "abstained": false
}
```

## Evaluation

Add questions and gold source identifiers to `eval/dataset.json` (copy the example file). A gold source is `{"filename":"biology.pdf","page":12}`. Upload the corpus first, then run:

```powershell
python eval/run_eval.py --dataset eval/dataset.json --api http://localhost:8000 --k 5
```

The script reports Recall@K, MRR, citation correctness, an explicitly approximate lexical groundedness score, latency, and estimated cost when `COST_PER_1K_INPUT_TOKENS` is set. The lexical metric is a review aid, not a substitute for human labels or an LLM judge. Evaluation quality depends on representative questions and accurate gold labels.

Check arithmetic in the starter Amdahl cases with:

```powershell
python eval/check_study_guide_cases.py
```

The seven arithmetic cases pass automatically. The answer set is marked `instructor_review_pending`; ask a course instructor to review `eval/study_guide_cases.json` before treating it as an instructor-approved benchmark. To collect generated answers for scoring, run the API with one model configuration and export a review sheet, then repeat with a different `--model-name` and output path. The sheet flags whether the expected numeric result appears in each answer; reviewers still need to check the actual formula and working:

```powershell
python eval/run_answer_review.py --model-name gpt-4o-mini --output eval/reviews/gpt-4o-mini.csv
```

Score factual accuracy, slide coverage, formula math, and explanation quality from 0 to 2 in the CSV. Compare averages only after reviewer scores are filled in.

Run the unit suite with `python -m pytest`. The tests cover deterministic chunking, retrieval ranking, local fallback behavior, PPT table/chart extraction, duplicate-image removal, and slide progress reporting. For an API/database smoke check, start Compose, upload a sample document, and poll its task ID; no fixture textbook is included.

## Roadmap

The implementation provides a runnable vertical slice for each project phase. Suggested next improvements are production controls and richer evaluation rather than replacing the core architecture:

1. **RAG foundation:** ingestion, chunking, embeddings, pgvector, semantic retrieval, answer endpoint.
2. **Advanced retrieval:** implemented with BM25 + pgvector dense search, weighted RRF, a wider candidate pool, query rewriting, reranking, and source attribution. Next evaluation step: tune these retrieval settings against a labeled course corpus.
3. **Agentic tutor:** search tool workflow, structured JSON response parsing, quizzes, answer evaluation, session memory.
4. **Quality:** labeled evaluation data and metrics; expand the starter corpus and add human-reviewed judgments.
5. **Deployment:** React client, Compose containers, migrations, environment configuration, logs, tests, and runbook.
6. **Final validation:** benchmark provider/model choices on your own labeled corpus, run end-to-end checks, and record a demo.

## Design notes and limits

- PDF extraction uses text-layer content via `pypdf`; scanned PDFs need OCR before upload.
- The ingestion-version migration allows a duplicate upload of an older document to rebuild its chunks and figures with the updated extractor. The original uploaded file itself is not retained by the app.
- Background task status is held in API process memory. Restarting the API loses active progress/result records; use a durable task queue and shared store before multi-worker deployment.
- The default reranker combines lexical coverage and vector similarity; it is replaceable with a trained cross-encoder for higher-quality ranking.
- BM25 scores the indexed chunk corpus in application memory, which is appropriate for a learning/demo corpus. Chunk tokenization is reused across query rewrites; move lexical ranking into a dedicated search index for very large corpora.
- Document upload size and CORS are configurable. Add authentication, per-user document ownership, rate limits, backups, and secrets management before exposing this service publicly.
- The answer model is instructed to use slide evidence, source labels, and only limited high-confidence additional context. Retrieval and automatic checks reduce errors but cannot guarantee correctness.

## Project layout

```text
backend/app/       FastAPI routes, background tasks, ingestion, retrieval, tutor, and study-guide services
backend/migrations Alembic schema migrations
frontend/          React/Vite learning interface
eval/              Retrieval metrics, arithmetic cases, and answer-review tooling
tests/             Unit tests for retrieval, text processing, and PPT extraction
docker-compose.yml Local database, API, and frontend stack
```

## License

No license has been selected yet. Add one before distributing or reusing this project.
