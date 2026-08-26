# RecoveryPath AI

<p align="center">
  <strong>Evidence-grounded, multilingual support for alcohol recovery</strong>
</p>

<p align="center">
  RecoveryPath AI combines conversational intent understanding, hybrid retrieval,
  source-grounded generation, claim validation, and citation verification to provide
  clear and responsible educational support about alcohol use and recovery.
</p>

> [!IMPORTANT]
> RecoveryPath AI provides educational information only. It does not diagnose medical
> conditions, prescribe treatment, calculate dosages, or replace a qualified doctor,
> pharmacist, addiction specialist, or emergency service.

## Overview

RecoveryPath AI is a multilingual Retrieval-Augmented Generation application focused
on alcohol recovery information. Users can ask questions naturally in Arabic,
Egyptian colloquial Arabic, English, and other supported styles.

The system understands the user's intent and conversation context, retrieves relevant
information from an indexed medical knowledge base, generates a concise answer in the
user's language, and validates every medical claim against the cited evidence before
showing the final response.

## Core Features

- Multilingual question understanding and conversational follow-up handling.
- Egyptian colloquial Arabic response style when appropriate.
- Model-backed intent classification without question-specific regular expressions.
- Social-message routing for greetings, thanks, acknowledgements, and farewells.
- Safety routing for urgent situations, dosage requests, personalized treatment, and
  prompt-injection attempts.
- Hybrid semantic and keyword retrieval across indexed documents.
- Cross-language retrieval for Arabic questions over English medical sources.
- Candidate fusion, deduplication, reranking, and model-based retrieval retry.
- Evidence-strength and relevance gates before answer generation.
- Source-aware context construction with document, page, asset, and chunk metadata.
- Grounded generation with strict sentence-level citations.
- Citation repair when citation structure is incomplete.
- Independent claim extraction and evidence-support evaluation.
- Removal of unsupported claims instead of discarding an otherwise supported answer.
- Rebuilding and revalidating supported partial answers.
- Temporary PDF and TXT analysis without permanent database storage in user mode.
- Streamlit chat history and a responsive animated assistant interface.
- Developer diagnostics for retrieval, claims, citations, sources, and latency.

## System Architecture

```text
User Message
    |
    v
Streamlit Frontend
    |
    v
FastAPI Request Validation
    |
    v
Intent + Conversation Understanding
    |
    +--> Social response, no retrieval
    |
    +--> Safety or clarification response
    |
    v
Semantic Query + Keyword Hints
    |
    v
Hybrid Retrieval
    |
    +--> Vector Search
    +--> Keyword Search
    |
    v
RRF Fusion + Deduplication
    |
    v
Reranking
    |
    +--> Model-Based Retrieval Retry when needed
    |
    v
Relevance and Evidence-Strength Gates
    |
    v
Context Builder
    |
    v
Grounded Answer Generation
    |
    v
Citation Compliance and Repair
    |
    v
Claim Extraction and Support Evaluation
    |
    v
Unsupported Claim Removal
    |
    v
Supported Answer Rebuilding and Revalidation
    |
    v
Citation Accuracy Evaluation
    |
    v
Final User Response
```

## Technology Stack

### Backend

- Python 3.12
- FastAPI
- Pydantic and Pydantic Settings
- SQLAlchemy async sessions
- PostgreSQL
- pgvector or a compatible vector store
- PyMuPDF for PDF extraction
- Groq-compatible language-model providers
- Cohere or a compatible reranking provider

### Frontend

- Streamlit
- Custom CSS
- Session-based chat history
- Temporary PDF and TXT upload support

### Retrieval and Validation

- Dense semantic retrieval
- Keyword retrieval
- Reciprocal Rank Fusion
- Candidate deduplication
- Cross-encoder reranking
- Query retry and rewriting
- Claim-support evaluation
- Citation-completeness and citation-accuracy evaluation

## Project Structure

```text
.
├── frontend/
│   ├── app.py
│   ├── api_client.py
│   ├── assets/
│   ├── components/
│   │   ├── animated_assistant.py
│   │   ├── chat.py
│   │   └── ingestion.py
│   └── styles/
│       └── custom.css
├── src/
│   ├── main.py
│   ├── helpers/
│   ├── models/
│   ├── parsers/
│   ├── routes/
│   │   ├── ingestion.py
│   │   └── rag.py
│   ├── services/
│   │   ├── RAGService.py
│   │   ├── RAGPromptBuilder.py
│   │   ├── IntentUnderstandingService.py
│   │   ├── RetrievalPipelineService.py
│   │   ├── HybridRetrievalService.py
│   │   ├── RetrievalRetryService.py
│   │   ├── ContextBuilder.py
│   │   ├── ClaimExtractor.py
│   │   ├── ClaimSupportEvaluator.py
│   │   ├── CitationRepairService.py
│   │   ├── CitationComplianceValidator.py
│   │   ├── UnsupportedClaimPruner.py
│   │   ├── SupportedAnswerRebuilder.py
│   │   └── EphemeralDocumentService.py
│   └── stores/
├── requirements.txt
└── README.md
```

## Getting Started

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd "Medical RAG"
```

### 2. Create the Python environment with `uv`

```bash
uv venv med_rag
source med_rag/bin/activate
uv pip install -r requirements.txt
```

### 3. Configure environment variables

Create or update:

```text
src/.env
```

Use the following structure and never commit real secrets:

```env
DATABASE_URL=postgresql+asyncpg://USER:PASSWORD@HOST:PORT/DATABASE

GROQ_API_KEY=your_groq_key
COHERE_API_KEY=your_cohere_key

GENERATION_PROVIDER=groq
RAG_BLOCK_ON_ANY_UNSUPPORTED_CLAIM=true

SHOW_DEVELOPER_MODE=false
BACKEND_API_URL=http://127.0.0.1:8000
```

> [!WARNING]
> Add `.env` and `src/.env` to `.gitignore`. Rotate any secret that has ever been
> committed or shown publicly.

### 4. Start the backend

```bash
python -m uvicorn src.main:app \
  --host 127.0.0.1 \
  --port 8000 \
  --log-level info
```

API documentation:

```text
http://127.0.0.1:8000/docs
```

### 5. Start the frontend

Open a second terminal:

```bash
source med_rag/bin/activate
BACKEND_API_URL=http://127.0.0.1:8000 streamlit run frontend/app.py
```

Frontend:

```text
http://localhost:8501
```

## API Example

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/rag/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "ما الآثار الصحية طويلة المدى للكحول؟",
    "conversation_history": [],
    "project_id": null,
    "asset_id": null,
    "retrieval_limit": 5,
    "generation_provider": "groq",
    "temperature": 0,
    "max_output_tokens": 1200
  }'
```

## Safety Behavior

RecoveryPath AI distinguishes between general educational information and requests
that require professional or emergency support.

```text
General treatment information  -> Evidence-grounded answer
Personalized treatment choice  -> Safe refusal
Dosage request                  -> Professional-care guidance
Urgent symptoms                 -> Emergency guidance
Prompt injection                -> Safe refusal
Out-of-scope request            -> Scope guidance
Social message                  -> Short direct response without retrieval
```

## Privacy

- Temporary user uploads must not be committed or permanently stored unless the user
  explicitly chooses a persistent ingestion workflow.
- Model-provider secrets remain on the backend.
- Previous assistant messages provide conversational context but are never treated as
  medical evidence.
- Uploaded document text is treated as untrusted evidence, not as executable system
  instructions.

## Known Limitations

- Response latency depends on retrieval, generation, claim evaluation, and provider
  availability.
- Trial API keys can have strict monthly and endpoint limits.
- Temporary documents may be reprocessed between requests unless session caching is
  enabled.
- Retrieval and reranking thresholds must be recalibrated when changing embedding or
  reranking models.
- RecoveryPath AI is not a substitute for professional care or local emergency
  services.

## License

Add the repository license here before public distribution.
