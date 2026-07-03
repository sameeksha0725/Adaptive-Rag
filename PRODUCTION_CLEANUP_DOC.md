# Production Cleanup & Implementation Summary

This document captures the production-ready cleanup performed on the Adaptive RAG project, its architecture, configuration, implementation details, and how to run and validate the system.

## Goals

- Preserve the existing architecture and API surface.
- Centralize configuration and expose meaningful constants.
- Propagate document metadata end-to-end for source citation.
- Add an optional retrieval analytics panel in Streamlit.
- Implement a modular confidence scoring system and expose it through API/UI.
- Keep code quality improvements modular and non-invasive.

---

## High-level Architecture

User Query → Query Analysis → Hybrid Retriever
  • FAISS
  • BM25
Hybrid results → Reciprocal Rank Fusion (RRF) → CrossEncoder Reranker → Document Grading → Generate (Final Answer)
  • Rewrite Query can loop back to Hybrid Retriever
Fallbacks: Web Search, General LLM
Outputs: Final Answer + Source Citations + Confidence Score

Mermaid diagram (matches README):

```mermaid
flowchart TD
  A[User Query] --> B[Query Analysis]
  B --> C[Hybrid Retriever]
  C --> FA[FAISS]
  C --> BM[BM25]
  FA --> RRF[Reciprocal Rank Fusion]
  BM --> RRF
  RRF --> X[CrossEncoder Reranker]
  X --> G[Document Grading]
  G --> GEN[Generate]
  G --> RW[Rewrite Query]
  RW --> C
  B --> Web[Web Search (fallback)]
  B --> GL[General LLM (fallback)]
  GEN --> OUT[Output: Final Answer + Sources + Confidence]
```

---

## What changed (summary)

- Metadata propagation: `filename`, `page`, `chunk_id`, `retrieval_method`, `retrieval_score`, `rrf_score`, `reranker_score` are carried from loader through chunking, embedding, retrievers, hybrid fusion, reranker, graph, and finally used in generation.
- Cross-Encoder reranking is added (configurable) using `sentence-transformers.CrossEncoder`.
- Confidence scoring implemented in `src/rag/confidence.py` and exposed via LangGraph state and API.
- Streamlit UI receives structured response (`content`, `confidence`, `sources`, `retrieval_analytics`) and shows an optional `Retrieval Analytics` sidebar.
- Configuration values centralized in `src/config/prompts.yaml` and exposed in `src/config/settings.py`.

---

## Modified files

- src/config/prompts.yaml — add `retriever.rrf_k` and cleanup reranker entries
- src/config/settings.py — expose config constants (RERANKER_MODEL, USE_RERANKER, RETRIEVAL_TOP_K, FINAL_TOP_K, RRF_K, MAX_RETRIES, LLM_MODEL, EMBEDDING_MODEL)
- src/rag/document_upload.py — enrich chunk metadata after splitting
- src/rag/faiss_retriever.py — annotate FAISS results with retrieval metadata
- src/rag/bm25_retriever.py — annotate BM25 results with retrieval metadata
- src/rag/hybrid_retriever.py — attach hybrid `rrf_score` metadata
- src/rag/reranker.py — attach `reranker_score` to candidate meta
- src/rag/retriever_setup.py — support helper search functions and wiring
- src/rag/graph_builder.py — propagate `rrf_score`, compute `confidence`, return `retrieval_analytics` and `sources` in node outputs
- src/rag/confidence.py — NEW modular confidence scorer
- src/api/routes.py — API response now returns structured `result` with content, confidence, sources, retrieval_analytics
- streamlit_app/utils/api_client.py — return structured payload for frontend
- streamlit_app/pages/chat.py — add optional sidebar `Retrieval Analytics` rendering and show confidence
- README.md — documentation and mermaid diagram updates

## New files

- src/rag/confidence.py

---

## Dependency list

The project relies on the following Python packages (see `requirements.txt` for exact versions):

- langchain (core and community components)
- langgraph
- ollama
- faiss-cpu (or faiss)
- rank-bm25
- sentence-transformers
- tavily-python
- bs4, beautifulsoup4
- requests
- pydantic
- python-dotenv
- PyYAML
- fastapi, uvicorn
- motor, pymongo
- streamlit

---

## Confidence scoring (implementation detail)

- Implemented in `src/rag/confidence.py` as `ConfidenceScorer`.
- Inputs: list of candidate dicts with `meta` keys (retrieval_score, rrf_score, reranker_score).
- Steps:
  1. Collect component scores across candidates.
  2. Min-max normalize each component.
  3. Average normalized component per component.
  4. Weighted sum (weights normalized internally).
  5. Map to 0–100 percentage and round.
- Default weights: semantic 25%, BM25 15%, RRF 30%, reranker 30%.
- Scorer is modular; replace as needed.

---

## Example API response

POST /rag/query

```json
{
  "result": {
    "content": "We can rotate tokens and validate JWT claims to authenticate users.",
    "confidence": 92,
    "sources": ["handbook.pdf (Page 7)", "policies.pdf (Chunk 14)"],
    "retrieval_analytics": [
      {
        "text": "...document excerpt...",
        "meta": {
          "filename": "handbook.pdf",
          "page": 7,
          "chunk_id": 3,
          "retrieval_method": "Hybrid",
          "retrieval_score": 0.83,
          "rrf_score": 0.42,
          "reranker_score": 2.10
        }
      }
    ]
  }
}
```

---

## Streamlit UI: Retrieval Analytics panel

- Collapsible sidebar panel titled **Retrieval Analytics** (collapsed by default).
- Shows System Info:
  - Hybrid Search: Enabled
  - Embedding Model
  - LLM Model
  - Reranker Model
  - Top-K Retrieved
  - Final Documents Passed to LLM
- Shows Retrieved Chunks (post-rerank) as a table with columns:
  - final_ranking, filename, page, chunk_id, retrieval_method, semantic_similarity, bm25_score, rrf_score, crossencoder_score
- The panel is optional and does not change chat behavior for normal users.

---

## Manual setup steps

1. Create virtualenv and install dependencies:

```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows
pip install -r requirements.txt
```

2. Create `.env` and set necessary env vars:

```
OLLAMA_MODEL=qwen3:latest
EMBEDDING_MODEL=<provider-name>
TAVILY_API_KEY=<key>
QDRANT_URL=...
MONGODB_URL=...
```

3. Run backend and frontend:

```bash
uvicorn src.main:app --reload
streamlit run streamlit_app/home.py
```

4. Upload documents and test queries. Inspect Retrieval Analytics via sidebar.

---

## Suggested future enhancements

- Persist `retrieval_analytics` per chat message in MongoDB for audit and analysis.
- Add UI controls to tune confidence weights and reranker usage.
- Add unit/integration tests covering retrieval and confidence outputs.
- Integrate remote vector DB (Qdrant) for scalable production usage.
- Add document viewer and deep-linking into cited chunks.

---

## Contacts

- Repository maintainer: (update as appropriate)


*Document generated by automated project cleanup assistant.*
