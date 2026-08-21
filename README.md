# RecoveryPath AI

> A bilingual, evidence-grounded RAG assistant that retrieves alcohol-recovery guidance from trusted PDF sources, validates every factual claim and citation, and safely refuses unsupported or personalized medical requests.

## Overview

RecoveryPath AI is an Arabic-English Retrieval-Augmented Generation system designed to provide accessible and traceable information about alcohol recovery.

The system processes uploaded PDF guidance, retrieves the most relevant evidence, generates an answer using only the selected passages, and validates every factual claim before displaying the response.

RecoveryPath AI is designed as an informational support tool. It does not diagnose users, calculate personalized dosages, select individual treatments, or replace qualified healthcare professionals.

## Key Features

- Arabic and English question support
- PDF ingestion using PyMuPDF
- Section-aware semantic chunking
- Multilingual embeddings
- PostgreSQL and pgvector storage
- Semantic and keyword hybrid retrieval
- Cross-language keyword retrieval for Arabic questions
- Reciprocal Rank Fusion
- Exact and near-duplicate removal
- Cohere passage reranking
- Evidence-strength classification
- Evidence-grounded answer generation
- Sentence-level source citations
- Bounded citation repair
- Atomic claim extraction
- Independent claim-support evaluation
- Citation and metadata accuracy validation
- Pre-generation and post-generation safety gates
- Contextual safe refusal
- Evidence inspector
- Project-wide and single-document search
- Conversation history
- FastAPI backend and Streamlit interface

## System Architecture

```text
PDF Upload
    |
    v
PyMuPDF Extraction
    |
    v
Section Detection
    |
    v
Semantic Chunking
    |
    v
Multilingual Embeddings
    |
    v
PostgreSQL + pgvector
    |
    v
Semantic Search + Keyword Search
    |
    v
Cross-Language Keyword Retrieval
    |
    v
Reciprocal Rank Fusion
    |
    v
Candidate Deduplication
    |
    v
Cohere Reranking
    |
    v
Top 5 Relevant Passages
    |
    v
Relevance Gate
    |
    v
Evidence Strength Classification
    |
    v
Citation-Ready Context
    |
    v
Grounded Answer Generation
    |
    v
Citation Compliance and Repair
    |
    v
Evidence Building
    |
    v
Atomic Claim Extraction
    |
    v
Independent Claim Support Evaluation
    |
    v
Faithfulness and Citation Accuracy
    |
    v
Verified Grounded Answer
or
Contextual Safe Refusal
```

## Ingestion Pipeline

### PDF Extraction

RecoveryPath AI uses PyMuPDF to extract structured text from uploaded PDF files while preserving:

- Document name
- Section title
- Page number
- Element ordering
- Source metadata

### Semantic Chunking

The system uses section-aware semantic chunking instead of fixed-size sliding windows.

| Parameter | Value |
|---|---:|
| Minimum chunk size | 120 estimated tokens |
| Target chunk size | 350 estimated tokens |
| Maximum chunk size | 500 estimated tokens |
| Similarity threshold | 0.55 |
| Fixed overlap | None |

Adjacent elements are compared using cosine similarity. A chunk is closed when the content changes semantically, reaches its preferred size, or would exceed the maximum size.

Context continuity is preserved through document sections and semantic adjacency rather than fixed token overlap.

## Embeddings and Storage

| Component | Technology |
|---|---|
| Embedding model | `embed-multilingual-light-v3.0` |
| Relational database | PostgreSQL |
| Vector search | pgvector |

The multilingual embedding model enables cross-language retrieval, including Arabic questions over English source material.

PostgreSQL stores the source text and metadata, while pgvector stores the corresponding embedding vectors.

## Hybrid Retrieval

RecoveryPath AI combines two retrieval paths:

### Semantic Search

The user question is embedded and compared with stored chunk vectors. This path captures semantically related content even when the wording or language differs.

### Keyword Search

PostgreSQL keyword retrieval identifies exact terms such as medicine names, clinical phrases, and section terminology.

Arabic questions also receive a cross-language keyword query to improve matching against English source documents.

### Reciprocal Rank Fusion

Semantic and keyword rankings are combined using Reciprocal Rank Fusion:

```text
RRF contribution = 1 / (60 + rank)
```

The system retrieves:

```text
Semantic candidates: 20
Keyword candidates:  20
Fused candidates:    20
```

## Candidate Deduplication

The candidate set is filtered before reranking.

Duplicates are identified using:

- Repeated asset and chunk identifiers
- Repeated normalized text
- Token Jaccard similarity of at least `0.90`
- A minimum of `20` tokens for near-duplicate comparison

This prevents redundant passages from occupying the final evidence context.

## Reranking

The remaining candidates are reranked using:

```text
Cohere rerank-v4.0-pro
```

The reranker compares the original question with every candidate and returns the five passages that answer the question most directly.

```text
Maximum tokens per candidate: 4096
Final retrieval
