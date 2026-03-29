# Hybrid Information Retrieval for PubMed

A biomedical search engine that combines lexical retrieval (BM25F), semantic retrieval (Word2Vec + FAISS), and transformer-based re-ranking (MiniLM) for context-aware PubMed article search.

## Overview

This project implements a hybrid ranking pipeline over PubMed-style biomedical documents.

It is designed to:
- preserve exact keyword matching (BM25F),
- recover semantic similarity (Word2Vec embeddings + FAISS ANN search),
- improve top-rank quality via transformer re-ranking,
- provide a simple FastAPI backend and browser UI for interactive search.

## Retrieval Pipeline

1. Query preprocessing
- Lowercasing, tokenization, stopword removal, lemmatization/stemming.
- Medical abbreviation expansion (`Query/QueryPreprocessor.py`).

2. Lexical retrieval
- BM25F-like scoring over title and abstract fields.
- Higher weight assigned to titles (`TITLE_WEIGHT=2`, `ABSTRACT_WEIGHT=1`).

3. Semantic retrieval
- Word2Vec document/query embeddings.
- FAISS vector index lookup for nearest neighbors.

4. Score fusion
- Dynamic lexical/semantic weighting based on query length:
  - short queries: higher semantic weight,
  - longer queries: balanced weighting.

5. Transformer re-ranking
- Candidate results are reranked with `all-MiniLM-L6-v2` cosine similarity.

6. Metrics endpoint
- Precision@K and nDCG are computed when relevant IDs are provided.

## Project Structure

```text
.
+-- PreProcessing/
¦   +-- PreProcessingService.py      # Parse PubMed XML + clean/token pipeline + Mongo insert
+-- Indexing/
¦   +-- IndexingService.py           # BM25F index generation and metadata storage in MongoDB
+-- Embedding/
¦   +-- EmbeddingGenerationService.py# Word2Vec training + FAISS index build/checkpointing
+-- Query/
¦   +-- QueryPreprocessor.py         # Query text normalization
¦   +-- QueryProcessing.py           # FastAPI hybrid search service
+-- FrontEndDesign/
¦   +-- index.html                   # Lightweight frontend UI
+-- Model/
¦   +-- word2vec.model               # Local model artifact
+-- pubmed_baseline/                 # Input PubMed `.xml.gz` files
+-- modeldownloader.py               # Optional model/index downloader (Google Drive)
+-- run.sh                           # Linux/macOS startup helper
+-- run.bat                          # Windows startup helper
+-- requirement.txt                  # Python dependencies
```

## Tech Stack

- Python 3.10+
- FastAPI + Uvicorn
- MongoDB Atlas (document storage + BM25 metadata)
- Gensim Word2Vec
- FAISS (vector search)
- sentence-transformers (`all-MiniLM-L6-v2`) for reranking
- NLTK (tokenization, stopwords, lemmatization)

## Setup

### 1. Clone

```bash
git clone https://github.com/ADITYATALEKAR/Hybrid-Information-Retrieval.git
cd Hybrid-Information-Retrieval
```

### 2. Install dependencies

```bash
pip install -r requirement.txt
```

### 3. Download/prepare model artifacts

```bash
python modeldownloader.py
```

If you already have `Model/word2vec.model` and `Model/w2v_index.index`, this step can be skipped.

### 4. Start the API server

```bash
cd Query
uvicorn QueryProcessing:app --host 0.0.0.0 --port 8088
```

### 5. Open frontend

Open `FrontEndDesign/index.html` in a browser.

## One-command local startup

### Windows

```bat
run.bat
```

### Linux/macOS

```bash
chmod +x run.sh
./run.sh
```

These scripts install dependencies, run model download, launch FastAPI, and open the frontend.

## API Endpoints

Base URL: `http://localhost:8088`

### `GET /search/lexical`
Lexical-only BM25F search.

Example:
```text
/search/lexical?query=breast+cancer+therapy&top_k=10
```

### `GET /search/hybrid`
Hybrid lexical + semantic + reranked search.

Example:
```text
/search/hybrid?query=breast+cancer+therapy&top_k=10
```

### `GET /search/hybrid-with-metrics`
Hybrid search plus evaluation metrics.

Parameters:
- `query` (string)
- `relevant_ids` (comma-separated article IDs)
- `top_k` (int, optional)

Example:
```text
/search/hybrid-with-metrics?query=lung+cancer&relevant_ids=12345,67890&top_k=10
```

## Data + Indexing Workflow (Offline)

Run these modules in sequence when rebuilding from raw PubMed XML:

1. Preprocess and store documents
```bash
python PreProcessing/PreProcessingService.py
```

2. Build BM25F index in MongoDB
```bash
python Indexing/IndexingService.py
```

3. Train/load Word2Vec and build FAISS vector index
```bash
python Embedding/EmbeddingGenerationService.py
```

4. Serve queries
```bash
python Query/QueryProcessing.py
```

## Configuration Notes

The current codebase uses embedded constants for MongoDB URI, DB names, and model/index paths.

For production-quality deployments, move these to environment variables:
- `MONGO_URI`
- `DB_NAME`
- model/index paths
- external API keys

## Known Limitations

- Sensitive credentials/API keys are currently hardcoded in source files.
- NLTK asset download is triggered at runtime in multiple modules.
- No test suite is included yet.
- `requirement.txt` is named singular; some tooling expects `requirements.txt`.
- Frontend is a static page without build tooling/auth/rate limiting.

## Future Improvements

- Move all secrets/config to `.env`.
- Add dataset snapshots and reproducible training/index scripts.
- Add unit/integration tests for preprocessing, scoring, and API routes.
- Add Docker + Compose for API + Mongo dependencies.
- Add CI checks for lint/test/type validation.

## Disclaimer

This project is for research/academic exploration of biomedical retrieval techniques. It is not a medical device and should not be used for clinical decision-making.
