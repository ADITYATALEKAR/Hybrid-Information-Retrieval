import re
import gzip
import os
import time
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
import nltk
from multiprocessing import Pool, cpu_count
import logging
import xml.etree.ElementTree as ET
from pymongo import MongoClient
import ssl
from tqdm import tqdm
import signal
import sys
import os

# Create unverified HTTPS context for NLTK downloads
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
    handlers=[
        logging.FileHandler("preprocessing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Download NLTK resources with error handling
def download_nltk_resources():
    resources = ["punkt", "stopwords", "wordnet"]
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
            logger.info(f"Successfully downloaded NLTK resource: {resource}")
        except Exception as e:
            logger.error(f"Failed to download NLTK resource {resource}: {e}")
            raise

# Custom stopwords (domain-specific for biomedical articles)
CUSTOM_STOPWORDS = {
    "patient", "study", "method", "result", "analysis", "research",
    "paper", "article", "conclusion", "background", "objective",
    "aim", "purpose", "data", "finding", "findings"
}

# MongoDB configuration
MONGO_URI = "mongodb+srv://ec2user:group9informationqmul@cluster0.fy8gzsz.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0",  # e.g., "mongodb+srv://user:password@cluster0.mongodb.net/"
DB_NAME = "search-engine"
COLLECTION_NAME = "articles"

# Add graceful shutdown handler
def signal_handler(sig, frame):
    logger.info("Received interrupt signal. Shutting down gracefully...")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def parse_pubmed_xml(file_path):
    """
    Parse a PubMed XML file and extract article details.

    Args:
        file_path: Path to the gzipped XML file

    Returns:
        list: Extracted article data in dictionary format
    """
    articles = []
    skipped_count = 0

    try:
        with gzip.open(file_path, "rb") as f:
            tree = ET.parse(f)
            root = tree.getroot()

            for article in root.findall(".//PubmedArticle"):
                try:
                    # Extract PMID (required field)
                    pmid_elem = article.find(".//PMID")
                    if pmid_elem is None:
                        skipped_count += 1
                        continue
                    article_id = pmid_elem.text

                    # Extract title (might be None)
                    title_elem = article.find(".//ArticleTitle")
                    title = title_elem.text if title_elem is not None and title_elem.text else ""

                    # Extract abstract (critical check)
                    abstract_elem = article.find(".//AbstractText")
                    if abstract_elem is None or not abstract_elem.text or not abstract_elem.text.strip():
                        skipped_count += 1
                        continue
                    abstract = abstract_elem.text

                    # Extract authors - more robust handling
                    authors = []
                    author_list = article.findall(".//Author")
                    for author in author_list:
                        last_name = author.find(".//LastName")
                        first_name = author.find(".//ForeName")
                        last_name_text = last_name.text if last_name is not None else ""
                        first_name_text = first_name.text if first_name is not None else ""
                        if last_name_text or first_name_text:
                            authors.append(f"{first_name_text} {last_name_text}".strip())

                    # Extract publication date
                    year_elem = article.find(".//PubDate/Year")
                    if year_elem is not None:
                        pub_year = year_elem.text
                    else:
                        # Try alternative date elements
                        medline_date = article.find(".//PubDate/MedlineDate")
                        if medline_date is not None and medline_date.text:
                            # Extract year from formats like "2015 Jan" or "2015-2016"
                            year_match = re.search(r'\d{4}', medline_date.text)
                            pub_year = year_match.group(0) if year_match else ""
                        else:
                            pub_year = ""

                    # Extract journal name
                    journal_elem = article.find(".//Journal/Title")
                    journal = journal_elem.text if journal_elem is not None else ""

                    # Extract keywords
                    keywords = []
                    keyword_elems = article.findall(".//Keyword")
                    for keyword in keyword_elems:
                        if keyword.text:
                            keywords.append(keyword.text)

                    # Extract MeSH terms for better searchability
                    mesh_terms = []
                    mesh_elems = article.findall(".//MeshHeading/DescriptorName")
                    for term in mesh_elems:
                        if term.text:
                            mesh_terms.append(term.text)

                    articles.append({
                        "id": article_id,
                        "title": title,
                        "abstract": abstract,
                        "authors": authors,
                        "publication_date": pub_year,
                        "journal": journal,
                        "keywords": keywords,
                        "mesh_terms": mesh_terms,
                        "processed": False  # Flag to track processing status
                    })
                except Exception as e:
                    logger.warning(f"Error parsing individual article: {e}")
                    skipped_count += 1
                    continue

        logger.info(f"Parsed {len(articles)} articles from {file_path}, skipped {skipped_count} articles")
    except Exception as e:
        logger.error(f"Error parsing file {file_path}: {str(e)}")

    return articles

# Update clean_text function with improved handling of biomedical text
def clean_text(text):
    """
    More careful cleaning preserving medical terms, numeric values, and specific punctuation

    Args:
        text: The text to clean

    Returns:
        str: Cleaned text
    """
    if not text:
        return ""

    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)

    # Keep medical-relevant characters (dashes, slashes, decimals, percentage)
    # but remove other special characters
    text = re.sub(r"[^a-zA-Z0-9\s\-\./%]", "", text)

    # Preserve decimal numbers as they're important in medical context
    text = re.sub(r'(\d+)\.(\d+)', r'\1DECIMAL\2', text)

    # Lowercase the text
    text = text.lower()

    # Restore decimal points
    text = text.replace('decimal', '.')

    return text.strip()

# Add medical abbreviation expansion dictionary
MEDICAL_ABBREVIATIONS = {
    "ca": "cancer",
    "mi": "myocardial infarction",
    "copd": "chronic obstructive pulmonary disease",
    "cvd": "cardiovascular disease",
    "hf": "heart failure",
    "htn": "hypertension",
    "dm": "diabetes mellitus",
    "ckd": "chronic kidney disease",
    "af": "atrial fibrillation",
    "cad": "coronary artery disease",
    "pe": "pulmonary embolism",
    "dvt": "deep vein thrombosis",
    "ra": "rheumatoid arthritis",
    "sle": "systemic lupus erythematosus",
    "ms": "multiple sclerosis",
    "ibd": "inflammatory bowel disease",
    "gerd": "gastroesophageal reflux disease"
    # Add more as needed
}

def expand_abbreviations(tokens):
    """
    Expand medical abbreviations in tokens

    Args:
        tokens: List of word tokens

    Returns:
        list: Tokens with expanded abbreviations
    """
    return [MEDICAL_ABBREVIATIONS.get(token.lower(), token) for token in tokens]

def remove_stopwords(tokens):
    """
    Remove stopwords from a list of tokens.

    Args:
        tokens: List of word tokens

    Returns:
        list: Tokens with stopwords removed
    """
    stop_words = set(stopwords.words("english")).union(CUSTOM_STOPWORDS)
    return [word for word in tokens if word.lower() not in stop_words and len(word) > 1]

def lemmatize_tokens(tokens):
    """
    Lemmatize a list of tokens.

    Args:
        tokens: List of word tokens

    Returns:
        list: Lemmatized tokens
    """
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word) for word in tokens]

def stem_tokens(tokens):
    """
    Stem a list of tokens.

    Args:
        tokens: List of word tokens

    Returns:
        list: Stemmed tokens
    """
    stemmer = PorterStemmer()
    return [stemmer.stem(word) for word in tokens]

def preprocess_article(article, use_stemming=False):
    """
    Preprocess an article's text fields

    Args:
        article: Dictionary containing article data
        use_stemming: Whether to use stemming instead of lemmatization

    Returns:
        dict: Processed article or None if processing failed
    """
    try:
        # Create a copy to preserve original data
        processed_article = article.copy()

        # Skip if title and abstract are both empty
        if not processed_article.get("title", "").strip() and not processed_article.get("abstract", "").strip():
            logger.warning(f"Skipping article {processed_article.get('id', 'unknown')} with empty title and abstract")
            return None

        # Clean and tokenize title
        title_text = clean_text(processed_article.get("title", ""))
        title_tokens = word_tokenize(title_text) if title_text else []

        # Clean and tokenize abstract (required field)
        abstract_text = clean_text(processed_article.get("abstract", ""))
        if not abstract_text:
            logger.warning(f"Skipping article {processed_article.get('id', 'unknown')} with empty abstract after cleaning")
            return None

        abstract_tokens = word_tokenize(abstract_text)

        # Process title tokens if they exist
        if title_tokens:
            # Expand abbreviations
            title_tokens = expand_abbreviations(title_tokens)
            # Remove stopwords
            title_tokens = remove_stopwords(title_tokens)
            # Apply stemming or lemmatization
            if use_stemming:
                title_tokens = stem_tokens(title_tokens)
            else:
                title_tokens = lemmatize_tokens(title_tokens)

        # Process abstract tokens
        abstract_tokens = expand_abbreviations(abstract_tokens)
        abstract_tokens = remove_stopwords(abstract_tokens)
        if use_stemming:
            abstract_tokens = stem_tokens(abstract_tokens)
        else:
            abstract_tokens = lemmatize_tokens(abstract_tokens)

        # Update processed article
        processed_article["processed_title"] = " ".join(title_tokens)
        processed_article["processed_abstract"] = " ".join(abstract_tokens)

        # Preserve original text
        processed_article["original_title"] = article.get("title", "")
        processed_article["original_abstract"] = article.get("abstract", "")

        # Mark as processed
        processed_article["processed"] = True
        processed_article["processed_date"] = time.strftime("%Y-%m-%d %H:%M:%S")

        return processed_article
    except Exception as e:
        logger.error(f"Error processing article {article.get('id', 'unknown')}: {e}")
        return None

def process_chunk(chunk, use_stemming):
    """
    Process a chunk of articles.

    Args:
        chunk: List of article dictionaries
        use_stemming: Whether to use stemming instead of lemmatization

    Returns:
        list: Processed articles
    """
    preprocessed_chunk = []
    for article in chunk:
        processed_article = preprocess_article(article, use_stemming)
        if processed_article:
            preprocessed_chunk.append(processed_article)
    return preprocessed_chunk

def create_mongodb_indexes(collection):
    """
    Create useful indexes for the MongoDB collection

    Args:
        collection: MongoDB collection object
    """
    try:
        # Create text indexes for search
        collection.create_index([
            ("processed_title", "text"),
            ("processed_abstract", "text"),
            ("mesh_terms", "text"),
            ("keywords", "text")
        ])

        # Create regular indexes for common queries
        collection.create_index("id", unique=True)
        collection.create_index("publication_date")
        collection.create_index("authors")
        collection.create_index("journal")
        collection.create_index("processed")

        logger.info("Created MongoDB indexes")
    except Exception as e:
        logger.error(f"Error creating MongoDB indexes: {e}")

def insert_into_mongodb(articles, batch_size=500):
    """
    Insert processed articles into MongoDB with batching

    Args:
        articles: List of article dictionaries
        batch_size: Size of batches for insertion
    """
    if not articles:
        logger.warning("No articles to insert into MongoDB. Skipping insertion.")
        return

    client = None
    try:
        # Connect to MongoDB with optimized connection pooling
        client = MongoClient(MONGO_URI,
                             maxPoolSize=50,
                             connectTimeoutMS=30000,
                             socketTimeoutMS=60000,
                             waitQueueTimeoutMS=30000)
        db = client[DB_NAME]
        collection = db[COLLECTION_NAME]

        # Process in batches
        for i in range(0, len(articles), batch_size):
            batch = articles[i:i+batch_size]
            # Use ordered=False for better performance
            result = collection.insert_many(batch, ordered=False)
            logger.info(f"Inserted batch of {len(result.inserted_ids)} articles into MongoDB.")

        logger.info(f"Completed inserting total of {len(articles)} articles")

    except Exception as e:
        logger.error(f"Error inserting into MongoDB: {e}")
    finally:
        if client:
            client.close()

def preprocess_and_store(folder_path, use_stemming=False, num_processes=None, chunk_size=1000, limit_files=None):
    """
    Preprocess PubMed XML files and store data in MongoDB.

    Args:
        folder_path: Path to folder containing .xml.gz files
        use_stemming: Whether to use stemming instead of lemmatization
        num_processes: Number of processes for parallel processing
        chunk_size: Number of articles to process in a chunk
        limit_files: Optional limit on number of files to process
    """
    start_time = time.time()
    try:
        # Download necessary NLTK resources
        download_nltk_resources()

        # Set number of processes
        num_processes = num_processes or max(1, cpu_count() - 1)  # Leave one CPU free
        logger.info(f"Using {num_processes} processes for preprocessing")

        # Initialize MongoDB client for indexes
        client = MongoClient(MONGO_URI)
        db = client[DB_NAME]
        collection = db[COLLECTION_NAME]
        create_mongodb_indexes(collection)

        # Get list of XML files
        xml_files = [f for f in os.listdir(folder_path) if f.endswith(".xml.gz")]
        if limit_files:
            xml_files = xml_files[:limit_files]

        logger.info(f"Found {len(xml_files)} XML files to process")

        # Process each file
        total_articles = 0
        total_processed = 0

        for file_idx, file_name in enumerate(tqdm(xml_files, desc="Processing files")):
            file_path = os.path.join(folder_path, file_name)
            logger.info(f"Processing file {file_idx+1}/{len(xml_files)}: {file_name}")

            # Parse the file and extract articles
            articles = parse_pubmed_xml(file_path)
            total_articles += len(articles)

            # Process articles in chunks
            for i in range(0, len(articles), chunk_size):
                chunk = articles[i:min(i+chunk_size, len(articles))]

                # Process chunk using multiprocessing
                with Pool(num_processes) as pool:
                    processed_chunks = pool.starmap(
                        process_chunk,
                        [(chunk, use_stemming)]
                    )

                # Flatten the processed chunks
                processed_articles = [article for sublist in processed_chunks for article in sublist if article]
                total_processed += len(processed_articles)

                # Insert processed articles into MongoDB
                if processed_articles:
                    insert_into_mongodb(processed_articles)

                # Log progress
                logger.info(f"Progress: {total_processed}/{total_articles} articles processed")

        elapsed_time = time.time() - start_time
        logger.info(f"Preprocessing complete. Total: {total_processed}/{total_articles} articles processed in {elapsed_time:.2f} seconds")

    except Exception as e:
        logger.error(f"Error in preprocessing pipeline: {e}")
    finally:
        # Close MongoDB connection
        if client:
            client.close()

# Example usage
if __name__ == "__main__":
    # Command line arguments can be added here using argparse
    folder_path = "/pubmed_baseline"
    use_stemming = False  # Set to True to use stemming instead of lemmatization
    num_processes = 4     # Number of processes for parallel processing
    chunk_size = 1000     # Number of articles to process at a time
    limit_files = None    # Set to an integer to limit the number of files processed (useful for testing)

    logger.info("Starting PubMed preprocessing pipeline")
    preprocess_and_store(folder_path, use_stemming, num_processes, chunk_size, limit_files)
    logger.info("Preprocessing pipeline completed")
