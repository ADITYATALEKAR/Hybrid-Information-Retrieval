"""
Hybrid Query Processor

Combines lexical (BM25F) and semantic (Word2Vec + FAISS) search with BERT reranking for biomedical articles.
Process flow:
1. Expands queries using Word2Vec similarities
2. Generates weighted embeddings (medical terms ×2 weight)
3. Retrieves candidates via parallel BM25F (title-weighted) and FAISS search
4. Dynamically fuses scores (70% semantic for short queries, 50-50% for long)
5. Reranks using BERT cross-encoder (70% BERT + 30% original score)
6. Returns optimized results with title/abstract and match provenance

Optimizations:
- Lazy-loaded BERT with thread-safe initialization
- FAISS for sub-millisecond vector search
- Query expansion and medical term boosting
"""

import math
import os
import logging
import sys
import threading
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI, HTTPException
from gensim.models import Word2Vec
from pymongo import MongoClient
import numpy as np
import faiss
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers.util import cos_sim


from QueryPreprocessor import preprocess_query
from Indexing.IndexingService import TITLE_WEIGHT, ABSTRACT_WEIGHT
from sentence_transformers import SentenceTransformer
from typing import List, Dict
from typing import List, Dict
import nltk

nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Initialize FastAPI
app = FastAPI(title="PubMed Query Processor")
WORD2VEC_VECTOR_SIZE = 300
WORD2VEC_MODEL_PATH = os.path.abspath(os.path.join("..", "Model", "word2vec.model"))
BERT_RERANKER_MODEL = "all-MiniLM-L6-v2"

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Configuration
CONFIG = {
    "mongo_uri":"mongodb+srv://ec2user:group9informationqmul@cluster0.fy8gzsz.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0",  # e.g., "mongodb+srv://user:password@cluster0.mongodb.net/"
    "db_name": "search-engine",
    "articles_collection": "articles",
    "index_collection": "bm25f_index",
    "metadata_collection": "bm25f_metadata",
    "faiss_index": "../Model/w2v_index.index",
    "max_results": 100
}

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BM25F_Searcher:
    def __init__(self):
        self.client = MongoClient(CONFIG["mongo_uri"],tlsAllowInvalidCertificates=True)
        self.db = self.client[CONFIG["db_name"]]
        self.index_col = self.db[CONFIG["index_collection"]]
        self.metadata_col = self.db[CONFIG["metadata_collection"]]
        self.articles_col = self.db[CONFIG["articles_collection"]]

        # Load BM25F metadata
        self.metadata = self.metadata_col.find_one()
        if not self.metadata:
            raise RuntimeError("BM25F metadata not found")

        self.title_idf = self.metadata["title_idf"]
        self.abstract_idf = self.metadata["abstract_idf"]
        self.title_avgdl = self.metadata["title_avgdl"]
        self.abstract_avgdl = self.metadata["abstract_avgdl"]

    def get_article_by_id(self, article_id):
        return self.articles_col.find_one({"id": article_id}, {"_id": 0, "title": 1, "abstract": 1})

    def bm25f_score(self, query_terms, doc_id):
        """Calculate BM25F score for a document"""
        doc_data = self.index_col.find_one({"article_id": doc_id})
        if not doc_data:
            return 0.0

        title_vector = doc_data["title_bm25_vector"]
        abstract_vector = doc_data["abstract_bm25_vector"]
        title_len = doc_data["title_length"]
        abstract_len = doc_data["abstract_length"]

        # BM25F parameters
        k1 = 1.5
        b = 0.75

        score = 0.0
        for term in query_terms:
            # Title score component
            tf_title = title_vector.get(term, 0)
            idf_title = self.title_idf.get(term, 0)
            numerator_title = tf_title * (k1 + 1)
            denominator_title = tf_title + k1 * (1 - b + b * (title_len / self.title_avgdl))
            score += TITLE_WEIGHT * idf_title * (numerator_title / denominator_title)

            # Abstract score component
            tf_abstract = abstract_vector.get(term, 0)
            idf_abstract = self.abstract_idf.get(term, 0)
            numerator_abstract = tf_abstract * (k1 + 1)
            denominator_abstract = tf_abstract + k1 * (1 - b + b * (abstract_len / self.abstract_avgdl))
            score += ABSTRACT_WEIGHT * idf_abstract * (numerator_abstract / denominator_abstract)

        return score

    def search(self, query, top_k=10):
        """BM25F search with field weighting"""
        query_terms = preprocess_query(query, return_tokens=True)

        # Get all documents that contain at least one query term
        candidate_docs = set()
        for term in query_terms:
            docs = self.index_col.find({
                "$or": [
                    {f"title_bm25_vector.{term}": {"$exists": True}},
                    {f"abstract_bm25_vector.{term}": {"$exists": True}}
                ]
            }, {"article_id": 1})
            candidate_docs.update(doc["article_id"] for doc in docs)

        # Score each candidate document
        scored_docs = []
        for doc_id in candidate_docs:
            score = self.bm25f_score(query_terms, doc_id)
            if score > 0:
                scored_docs.append((doc_id, score))

        # Sort by score and get top results
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        top_docs = scored_docs[:top_k]

        # Retrieve full documents
        results = []
        for doc_id, score in top_docs:
            doc = self.get_article_by_id(doc_id)
            if doc:
                results.append({
                    "id": doc_id,
                    "score": float(score),
                    "title": doc.get("title", ""),
                    "abstract": doc.get("abstract", ""),
                    "match_type": "lexical"
                })
        return results


class QueryProcessor:
    def __init__(self):
        """Initialize with pre-loaded indexes"""
        self.bm25_searcher = BM25F_Searcher()
        self.faiss_index = faiss.read_index(CONFIG["faiss_index"])
        self.doc_ids = []  # Should match FAISS index order
        self.load_doc_ids()
        self.client = MongoClient(CONFIG["mongo_uri"],tlsAllowInvalidCertificates=True)
        self.db = self.client[CONFIG["db_name"]]
        self.articles_col = self.db[CONFIG["articles_collection"]]
        self.model = Word2Vec.load(WORD2VEC_MODEL_PATH)
        self.bert_reranker = None
        self.bert_reranker_lock = threading.Lock()

    def load_doc_ids(self):
        """Load document IDs in same order as FAISS index"""
        with MongoClient(CONFIG["mongo_uri"],tlsAllowInvalidCertificates=True) as client:
            self.doc_ids = [doc["id"] for doc in
                            client[CONFIG["db_name"]][CONFIG["articles_collection"]]
                            .find({}, {"id": 1}).sort("_id", 1)]

    def semantic_search(self, query_embedding, top_k=10):
        """Word2Vec vector similarity search"""
        query_embedding = np.expand_dims(query_embedding, axis=0).astype('float32')
        distances, indices = self.faiss_index.search(query_embedding, top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            doc_id = self.doc_ids[idx]
            doc = self.articles_col.find_one({"id": doc_id}, {"_id": 0})
            if doc:
                results.append({
                    "id": doc_id,
                    "score": float(1 / (1 + dist)),  # Convert distance to similarity
                    "title": doc.get("title", ""),
                    "abstract": doc.get("abstract", ""),
                    "match_type": "semantic"
                })
        return results

    def get_bert_reranker(self):
        """Lazy load the BERT re-ranker"""
        if self.bert_reranker is None:
            with self.bert_reranker_lock:
                if self.bert_reranker is None:
                    logger.info("Loading BERT re-ranker model...")
                    self.bert_reranker = SentenceTransformer('all-MiniLM-L6-v2')
                    # self.bert_reranker = CrossEncoder(BERT_RERANKER_MODEL)
                    logger.info("loaded")
        return self.bert_reranker

    def bert_rerank(self, query: str, documents: List[Dict], top_k: int = 10) -> List[Dict]:
        """
        BERT re-ranking with proper score normalization and validation
        Args:
            query: The search query
            documents: List of document dicts (must contain 'title' and 'abstract')
            top_k: Number of documents to return
        Returns:
            Re-ranked list of documents with updated scores
        """
        if not documents:
            return []

        # Validate document format
        validated_docs = []

        for doc in documents:
            if not all(key in doc for key in ['title', 'abstract', 'score']):
                continue
            validated_docs.append(doc)

        if not validated_docs:
         return []

        if not documents:
            return []

        # 1. Encode the query once
        query_embedding = self.get_bert_reranker().encode(query, convert_to_tensor=True)

        # 2. Prepare document texts (title + abstract)
        doc_texts = [f"{doc['title']} [SEP] {doc['abstract']}" for doc in documents]

        # 3. Encode all documents in a batch
        doc_embeddings = self.get_bert_reranker().encode(doc_texts, convert_to_tensor=True)

        # 4. Compute cosine similarity between query and documents
        print("computing cosize")
        try:
            similarities = cos_sim(query_embedding, doc_embeddings)[0]
        except Exception as e:
            logging.error(e)

        # 5. Update documents with combined scores
        for doc, sim_score in zip(documents, similarities):
            doc["original_score"] = doc["score"]
            doc["bert_score"] = float(sim_score)
            doc["score"] = 0.7 * sim_score + 0.3 * doc["original_score"]

        # 6. Sort by new combined score
        documents.sort(key=lambda x: x["score"], reverse=True)
        return documents[:top_k]

    def get_query_embedding(self, query: str) -> np.ndarray:
        """query embedding with expansion and weighting"""
        tokens = preprocess_query(query, return_tokens=True)
        expanded_tokens = self.expand_query_terms(tokens)
        return self.embed_tokens(expanded_tokens)

    def expand_query_terms(self, tokens):
        """ query with similar terms"""
        expanded = set(tokens)
        for token in tokens:
            try:
                if token in self.model.wv:
                    similar = self.model.wv.most_similar(token, topn=2)
                    expanded.update([word for word, _ in similar])
            except KeyError:
                continue
        return list(expanded)

    def embed_tokens(self, tokens):
        """Embed tokens with medical term weighting"""
        vectors = []
        weights = []
        MEDICAL_TERMS = {
            "disease", "treatment", "therapy", "diagnosis", "patient",
            "clinical", "symptom", "drug", "protein", "gene"
        }

        for word in tokens[:500]:  # Same limit as documents
            try:
                if word in self.model.wv:
                    weight = 2.0 if word.lower() in MEDICAL_TERMS else 1.0
                    vectors.append(self.model.wv[word])
                    weights.append(weight)
            except Exception:
                continue

        if vectors:
            weights = np.array(weights) / sum(weights)
            return np.average(vectors, axis=0, weights=weights)
        return np.zeros(WORD2VEC_VECTOR_SIZE)

    def hybrid_search(self, query, query_embedding, top_k=10):
        """hybrid search with dynamic weighting"""
        # Get more candidates initially
        lexical_results = self.bm25_searcher.search(query, top_k * 5)
        semantic_results = self.semantic_search(query_embedding, top_k * 5)

        # Dynamic weighting based on query characteristics
        query_length = len(query.split())
        if query_length <= 2:
            # Short query - rely more on semantic search
            bm25_weight = 0.3
            w2v_weight = 0.7
        else:
            # Longer query - use balanced weights
            bm25_weight = 0.5
            w2v_weight = 0.5

        # Combine results with score normalization
        max_bm25 = max([r["score"] for r in lexical_results]) if lexical_results else 1
        max_w2v = max([r["score"] for r in semantic_results]) if semantic_results else 1

        combined = {}
        for res in lexical_results:
            combined[res["id"]] = {
                "score": (res["score"] / max_bm25) * bm25_weight,
                "doc": res
            }

        for res in semantic_results:
            if res["id"] in combined:
                combined[res["id"]]["score"] += (res["score"] / max_w2v) * w2v_weight
            else:
                combined[res["id"]] = {
                    "score": (res["score"] / max_w2v) * w2v_weight,
                    "doc": res
                }

        # Sort and return top results
        sorted_results = sorted(combined.values(), key=lambda x: x["score"], reverse=True)
        candidates = [res["doc"] for res in sorted_results[:top_k * 3]]  # More candidates for re-ranking

        # Apply BERT re-ranking
        reranked_results = self.bert_rerank(query, candidates, top_k)

        return reranked_results
        # return [res["doc"] for res in sorted_results[:top_k]]

    def calculate_ndcg(self, ranked_results: List[Dict], relevant_ids: List[str], k: int = 10) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain (nDCG) for search results.
        
        Args:
            ranked_results: List of search results (each must contain 'id')
            relevant_ids: List of document IDs considered relevant
            k: Cutoff point for evaluation
            
        Returns:
            nDCG score (float between 0 and 1)
        """
        if not ranked_results or not relevant_ids:
            return 0.0
            
        # Calculate DCG
        dcg = 0.0
        for i, result in enumerate(ranked_results[:k], 1):
            rel = 1 if result['id'] in relevant_ids else 0
            dcg += rel / math.log2(i + 1)
            
        # Calculate IDCG (ideal ordering)
        ideal_relevance = [1] * min(len(relevant_ids), k)
        idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal_relevance))
        
        return dcg / idcg if idcg > 0 else 0.0

    def calculate_precision_at_k(self, ranked_results: List[Dict], relevant_ids: List[str], k: int = 10) -> float:
        """
        Calculate Precision@K for search results.
        
        Args:
            ranked_results: List of search results (each must contain 'id')
            relevant_ids: List of document IDs considered relevant
            k: Cutoff point for evaluation
            
        Returns:
            Precision@K score (float between 0 and 1)
        """
        if not ranked_results or not relevant_ids:
            return 0.0
            
        top_k = ranked_results[:k]
        relevant_count = sum(1 for result in top_k if result['id'] in relevant_ids)
        return relevant_count / k

# Initialize processor at startup
processor = QueryProcessor()


# API Endpoints
@app.get("/search/lexical")
async def search_lexical(query: str, top_k: int = 10):
    """Pure BM25F search"""
    try:
        return {"results": processor.bm25_searcher.search(query, top_k)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/search/hybrid")
async def search_hybrid(query: str, top_k: int = 10):
    """Hybrid search"""
    try:
        query_embedding = processor.get_query_embedding(query)
        return {"results": processor.hybrid_search(query, query_embedding, top_k)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# New endpoint with metrics
@app.get("/search/hybrid-with-metrics")
async def hybrid_search_with_metrics(
        query: str,
        relevant_ids: str,
        top_k: int = 10
):
    """
    - hybrid search that returns:
    - Search results
    - Precision@K
    - nDCG metrics
    """
    try:
        # Get hybrid search results
        query_embedding = processor.get_query_embedding(query)
        results = processor.hybrid_search(query, query_embedding, top_k)

        # Process relevant IDs
        relevant_ids_list = [id.strip() for id in relevant_ids.split(",") if id.strip()]
        # Debug logging
        print(f"Relevant IDs: {relevant_ids_list}")
        print(f"Result IDs: {[r['id'] for r in results]}")
        print(f"Common IDs: {set(relevant_ids_list) & set(str(r['id']) for r in results)}")
        # Calculate metrics
        precision = processor.calculate_precision_at_k(results, relevant_ids_list, top_k)
        ndcg = processor.calculate_ndcg(results, relevant_ids_list, top_k)

        return {
            "results": results,
            "metrics": {
                "precision@k": precision,
                "nDCG": ndcg
            },
            "query": query,
            "k": top_k
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




    
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8088)
