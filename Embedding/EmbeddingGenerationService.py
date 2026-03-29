import os
import logging
from pathlib import Path

from pymongo import MongoClient
from gensim.models import Word2Vec
import numpy as np
import faiss
import gc
import psutil

# Configuration
MONGO_URI = "mongodb+srv://ec2user:group9informationqmul@cluster0.fy8gzsz.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0",  # e.g., "mongodb+srv://user:password@cluster0.mongodb.net/"
DB_NAME = "search-engine"
COLLECTION_NAME = "articles"
WORD2VEC_VECTOR_SIZE = 300
BATCH_SIZE = 20000
MEMORY_THRESHOLD = 80
CHECKPOINT_INTERVAL = 100000
WORD2VEC_MODEL_PATH = os.path.abspath(os.path.join("..", "Model", "word2vec.model"))


# Absolute path with directory creation
VECTOR_DB_PATH = os.path.abspath(os.path.join("..", "Model", "w2v_index.index"))
os.makedirs(os.path.dirname(VECTOR_DB_PATH), exist_ok=True)



logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class MongoDocumentStream:
    def __init__(self, mongo_uri, db_name, collection_name):
        self.mongo_uri = mongo_uri
        self.db_name = db_name
        self.collection_name = collection_name

    def __iter__(self):
        client = MongoClient(self.mongo_uri)
        query = {"abstract": {"$exists": True, "$ne": ""}}

        cursor = client[self.db_name][self.collection_name].find(
            query, {"_id": 1, "title": 1, "abstract": 1},
            batch_size=2000
        ).sort("_id", 1)

        for doc in cursor:
            yield f"{doc['title']} {doc['abstract']}".split()
        client.close()

def train_word2vec():
    """Train Word2Vec model only if it doesn't exist"""
    model_file = Path(WORD2VEC_MODEL_PATH)

    # Check if model already exists
    if model_file.exists():
        logging.info(f"Loading existing model from {WORD2VEC_MODEL_PATH}")
        return Word2Vec.load(str(model_file))
    logging.info("Training Word2Vec model...")
    stream = MongoDocumentStream(MONGO_URI, DB_NAME, COLLECTION_NAME)

    # Count total documents for progress reporting
    with MongoClient(MONGO_URI) as client:
        total_docs = client[DB_NAME][COLLECTION_NAME].count_documents(
            {"abstract": {"$exists": True, "$ne": ""}}
        )

    logging.info(f"Building vocabulary from {total_docs} documents...")

    # Initialize the model with optimized parameters
    model = Word2Vec(
        vector_size=WORD2VEC_VECTOR_SIZE,
        window=5,
        min_count=2,
        workers=min(8, os.cpu_count()),
        sg=1,  # Use skip-gram (better for infrequent words)
        hs=0,  # Use negative sampling
        negative=10,  # Number of negative samples
        ns_exponent=0.75,  # Better value for medical texts
        sample=1e-5,  # Downsampling threshold
        alpha=0.025,  # Initial learning rate
        min_alpha=0.0001,  # Final learning rate
        epochs=20  # Increased from default 5 to 20
    )

    # Build vocabulary
    model.build_vocab(stream)
    logging.info(f"Vocabulary size: {len(model.wv.index_to_key)}")

    # Train model with progress reporting
    logging.info(f"Training Word2Vec model with {model.epochs} epochs...")
    model.train(
        stream,
        total_examples=model.corpus_count,
        epochs=model.epochs,
        compute_loss=True  # Track training progress
    )

    logging.info(f"Final training loss: {model.get_latest_training_loss()}")

    # Save model
    logging.info(f"Saving Word2Vec model to {WORD2VEC_MODEL_PATH}")
    model.save(WORD2VEC_MODEL_PATH)

    # Optimize memory usage
    model.wv.init_sims(replace=True)

    return model
def process_documents(model):
    index = faiss.IndexFlatL2(WORD2VEC_VECTOR_SIZE)
    processed = 0

    if os.path.exists(VECTOR_DB_PATH):
        index = faiss.read_index(VECTOR_DB_PATH)
        processed = index.ntotal
        logging.info(f"Resuming from {processed} documents")

    # Always open the client here for each query
    with MongoClient(MONGO_URI) as client:
        total_docs = client[DB_NAME][COLLECTION_NAME].count_documents(
            {"abstract": {"$exists": True, "$ne": ""}}
        )

    last_id = None
    if processed > 0:
        with MongoClient(MONGO_URI) as client:
            last_id = client[DB_NAME][COLLECTION_NAME].find_one(
                {}, {"_id": 1}, skip=processed-1
            )["_id"]

    batch_size = BATCH_SIZE
    while processed < total_docs:
        mem = psutil.virtual_memory()
        if mem.percent > MEMORY_THRESHOLD:
            batch_size = max(1000, batch_size // 2)
            logging.warning(f"Memory {mem.percent}%, reducing batch to {batch_size}")

        query = {"abstract": {"$exists": True, "$ne": ""}}
        if last_id:
            query["_id"] = {"$gt": last_id}

        with MongoClient(MONGO_URI) as client:  # Open a new MongoClient for each query
            cursor = client[DB_NAME][COLLECTION_NAME].find(
                query,
                {"_id": 1, "id": 1, "title": 1, "abstract": 1},
                limit=batch_size
            ).sort("_id", 1)

            embeddings = np.zeros((min(1000, batch_size), WORD2VEC_VECTOR_SIZE), dtype=np.float32)
            batch_count = 0

            for doc in cursor:
                text = f"{doc['title']} {doc['abstract']}"
                embeddings[batch_count] = get_doc_embedding(text, model)
                batch_count += 1
                last_id = doc["_id"]

                if batch_count >= 1000:
                    index.add(embeddings[:batch_count])
                    processed += batch_count
                    batch_count = 0
                    gc.collect()

                    if processed % CHECKPOINT_INTERVAL == 0:
                        save_checkpoint(index, processed)

            if batch_count > 0:
                index.add(embeddings[:batch_count])
                processed += batch_count

    save_checkpoint(index, processed)
    logging.info(f"Completed processing {processed} documents")

def save_checkpoint(index, processed):
    try:
        faiss.write_index(index, VECTOR_DB_PATH)
        if not os.path.exists(VECTOR_DB_PATH):
            raise RuntimeError("Index file not created")
        size_mb = os.path.getsize(VECTOR_DB_PATH) / (1024 * 1024)
        logging.info(f"Saved checkpoint at {processed} ({size_mb:.2f} MB)")
    except Exception as e:
        logging.error(f"Checkpoint failed: {e}")
        raise

def get_doc_embedding(text, model):
    words = text.split()[:500]
    vectors = []
    weights = []

    MEDICAL_TERMS = {
        # General Medical Terms
        "disease", "treatment", "therapy", "diagnosis", "patient", "clinical",
        "symptom", "drug", "medicine", "prognosis", "etiology", "pathology",
        "prevention", "rehabilitation", "screening", "epidemiology", "mortality",

        # Anatomy and Physiology
        "anatomy", "physiology", "organ", "tissue", "cell", "molecule",
        "organelle", "system", "function", "structure",

        # Biomedical Sciences
        "protein", "gene", "dna", "rna", "genome", "transcriptome", "proteome",
        "metabolome", "enzyme", "receptor", "antibody", "antigen", "pathogen",
        "microbiome", "biomarker", "mutation", "polymorphism", "expression",

        # Diseases and Conditions
        "cancer", "tumor", "neoplasm", "malignancy", "infection", "inflammation",
        "autoimmune", "degenerative", "congenital", "chronic", "acute", "syndrome",
        "disorder", "deficiency", "allergy", "trauma", "injury", "fracture",

        # Medical Specialties
        "cardiology", "neurology", "oncology", "pediatrics", "geriatrics",
        "immunology", "endocrinology", "gastroenterology", "hematology",
        "nephrology", "pulmonology", "rheumatology", "dermatology", "psychiatry",

        # Treatments and Interventions
        "surgery", "transplant", "chemotherapy", "radiotherapy", "immunotherapy",
        "vaccine", "antibiotic", "antiviral", "antifungal", "analgesic",
        "anesthetic", "biologic", "stent", "graft", "prosthesis", "dialysis",

        # Diagnostic Terms
        "biopsy", "imaging", "radiology", "ultrasound", "mri", "ct", "pet",
        "xray", "endoscopy", "colonoscopy", "assay", "test", "marker", "score",

        # Pharmacology
        "pharmacokinetics", "pharmacodynamics", "dose", "dosage", "toxicity",
        "interaction", "metabolism", "clearance", "half-life", "bioavailability",

        # Research Methods
        "randomized", "controlled", "trial", "cohort", "case-control",
        "meta-analysis", "systematic", "review", "longitudinal", "cross-sectional",
        "in-vitro", "in-vivo", "ex-vivo", "animal-model", "clinical-trial",

        # Healthcare Concepts
        "primary-care", "public-health", "epidemic", "pandemic", "outbreak",
        "surveillance", "prevalence", "incidence", "morbidity", "comorbidity",
        "quality-of-life", "patient-reported", "outcomes", "adherence", "compliance",

        # Emerging Areas
        "precision-medicine", "personalized-medicine", "genomic-medicine",
        "telemedicine", "digital-health", "wearable", "ai", "machine-learning",
        "deep-learning", "biobank", "big-data", "omics", "single-cell",

        # Additional Important Terms
        "signaling", "pathway", "mechanism", "regulation", "homeostasis",
        "apoptosis", "necrosis", "hypoxia", "ischemia", "oxidative", "stress",
        "inflammation", "fibrosis", "remodeling", "regeneration", "plasticity"
    }

    for word in words:
        try:
            if word in model.wv:
                # Higher weight for medical terms
                weight = 2.0 if word.lower() in MEDICAL_TERMS else 1.0
                vectors.append(model.wv[word])
                weights.append(weight)
        except Exception:
            continue

    if vectors:
        weights = np.array(weights) / sum(weights)  # Normalize weights
        return np.average(vectors, axis=0, weights=weights)
    return np.zeros(WORD2VEC_VECTOR_SIZE)
if __name__ == "__main__":
    try:
        model = train_word2vec()
        process_documents(model)
    except Exception as e:
        logging.error(f"Fatal error: {e}", exc_info=True)
    finally:
        gc.collect()