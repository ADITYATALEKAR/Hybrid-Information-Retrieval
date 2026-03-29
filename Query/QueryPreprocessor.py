import os
import re
import sys

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
# Apply SSL workaround (on top of your script)
import ssl
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from PreProcessing.PreProcessingService import expand_abbreviations

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


# Download required NLTK assets (safe-guarded for first-time execution)
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

# Custom stopwords (domain-specific)
MEDICAL_STOPWORDS = {
    "patient", "patients", "study", "studies", "method", "methods",
    "result", "results", "analysis", "research", "paper", "papers",
    "article", "articles", "data", "clinical", "trial"
}

# Shared preprocessing utils
stop_words = set(stopwords.words("english")).union(MEDICAL_STOPWORDS)
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()


def clean_text(text: str) -> str:
    """Match document cleaning exactly"""
    text = re.sub(r"[^a-zA-Z0-9\s\-\./]", "", text)
    return text.lower()

def tokenize(text: str) -> list:
    """Tokenize cleaned text."""
    return word_tokenize(text)


def remove_stopwords(tokens: list) -> list:
    """Remove standard and custom stopwords."""
    return [word for word in tokens if word not in stop_words]


def lemmatize_tokens(tokens: list) -> list:
    """Lemmatize tokens."""
    return [lemmatizer.lemmatize(word) for word in tokens]


def stem_tokens(tokens: list) -> list:
    """Stem tokens."""
    return [stemmer.stem(word) for word in tokens]


def preprocess_query(query: str, use_stemming: bool = False, return_tokens: bool = False) -> str | list:
    cleaned = clean_text(query)
    tokens = tokenize(cleaned)
    tokens = expand_abbreviations(tokens)  # Use same abbreviation expansion
    tokens = remove_stopwords(tokens)
    tokens = stem_tokens(tokens) if use_stemming else lemmatize_tokens(tokens)
    return tokens if return_tokens else " ".join(tokens)
