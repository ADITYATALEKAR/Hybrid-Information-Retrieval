import logging
from rank_bm25 import BM25Okapi
from pymongo import MongoClient
from nltk.tokenize import word_tokenize
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
MONGO_URI = "mongodb+srv://ec2user:group9informationqmul@cluster0.fy8gzsz.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0",  # e.g., "mongodb+srv://user:password@cluster0.mongodb.net/"
DB_NAME = "search-engine"
COLLECTION_NAME = "articles"
INDEX_COLLECTION_NAME = "bm25f_index"

BATCH_SIZE = 1000  # Process articles in chunks
TITLE_WEIGHT = 2    # Assign higher importance to titles
ABSTRACT_WEIGHT = 1  # Assign lower weight to abstracts

def fetch_articles_from_mongodb():
    """Fetch only articles with valid abstracts from MongoDB"""
    try:
        client = MongoClient(MONGO_URI)
        db = client[DB_NAME]
        collection = db[COLLECTION_NAME]

        # Enhanced query to filter out empty/null abstracts and ensure abstract exists
        query = {
            "abstract": {
                "$exists": True,
                "$ne": None,
                "$nin": ["", "N/A", "NA", "None", "none"],
                "$not": {"$regex": r"^\s*$"}  # Exclude whitespace-only abstracts
            }
        }

        projection = {"_id": 0, "id": 1, "title": 1, "abstract": 1}

        articles = []
        cursor = collection.find(query, projection).batch_size(BATCH_SIZE)

        for doc in cursor:
            # Additional client-side validation
            if doc.get("abstract") and str(doc["abstract"]).strip():
                articles.append(doc)
                if len(articles) >= BATCH_SIZE:
                    yield articles
                    articles = []

        if articles:  # Yield remaining valid articles
            yield articles

        logging.info("Finished fetching all articles with valid abstracts.")
    except Exception as e:
        logging.error(f"Error fetching articles from MongoDB: {e}")
        raise
    finally:
        client.close()

def create_bm25f_index(articles):
    """Create BM25F index with medical-specific tuning"""
    try:
        title_corpus = []
        abstract_corpus = []
        article_ids = []

        for article in articles:
            # Ensure we have both title and abstract
            if not article.get("title") or not article.get("abstract"):
                continue

            title_tokens = word_tokenize(article["title"])
            abstract_tokens = word_tokenize(article["abstract"])

            # Skip if either field is empty after tokenization
            if not title_tokens or not abstract_tokens:
                continue

            title_corpus.append(title_tokens)
            abstract_corpus.append(abstract_tokens)
            article_ids.append(article["id"])

        if not article_ids:
            logging.warning("No valid articles found for indexing")
            return None, None, None, None, None

        # Medical-specific BM25 parameters
        bm25_title = BM25Okapi(title_corpus, k1=1.2, b=0.5)
        bm25_abstract = BM25Okapi(abstract_corpus, k1=1.8, b=0.85)

        logging.info(f"Created index for {len(article_ids)} articles")
        return bm25_title, bm25_abstract, article_ids, title_corpus, abstract_corpus
    except Exception as e:
        logging.error(f"Error creating BM25F index: {e}")
        return None, None, None, None, None

def store_index_in_mongodb(bm25_title, bm25_abstract, article_ids, title_corpus, abstract_corpus):
    """Store BM25F index in MongoDB."""
    if not article_ids:
        logging.warning("No articles to index")
        return

    try:
        client = MongoClient(MONGO_URI)
        db = client[DB_NAME]

        # Clear existing index
        db[INDEX_COLLECTION_NAME].delete_many({})

        # Store new index in batches
        index_data = []
        for i, doc_id in enumerate(article_ids):
            index_data.append({
                "article_id": doc_id,
                "title_bm25_vector": bm25_title.doc_freqs[i],
                "abstract_bm25_vector": bm25_abstract.doc_freqs[i],
                "title_length": len(title_corpus[i]),
                "abstract_length": len(abstract_corpus[i])
            })

            if len(index_data) >= BATCH_SIZE:
                db[INDEX_COLLECTION_NAME].insert_many(index_data)
                index_data = []

        if index_data:
            db[INDEX_COLLECTION_NAME].insert_many(index_data)

        # Store/update global metadata
        db["bm25f_metadata1"].replace_one({}, {
            "title_idf": bm25_title.idf,
            "abstract_idf": bm25_abstract.idf,
            "title_avgdl": sum(len(doc) for doc in title_corpus) / len(title_corpus),
            "abstract_avgdl": sum(len(doc) for doc in abstract_corpus) / len(abstract_corpus),
            "total_articles": len(article_ids)
        }, upsert=True)

        logging.info(f"Successfully indexed {len(article_ids)} articles")
    except Exception as e:
        logging.error(f"Error storing BM25F index: {e}")
        raise
    finally:
        client.close()

def index_articles():
    """Main function to index valid articles using BM25F."""
    try:
        total_processed = 0
        for articles_batch in fetch_articles_from_mongodb():
            bm25_title, bm25_abstract, article_ids, title_corpus, abstract_corpus = create_bm25f_index(articles_batch)
            if bm25_title and bm25_abstract:
                store_index_in_mongodb(bm25_title, bm25_abstract, article_ids, title_corpus, abstract_corpus)
                total_processed += len(article_ids)

        logging.info(f"BM25F indexing completed. Total articles indexed: {total_processed}")
    except Exception as e:
        logging.error(f"Indexing failed: {e}")
        raise

if __name__ == "__main__":
    index_articles()