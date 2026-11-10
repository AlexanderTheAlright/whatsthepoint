"""
Text Analysis Module
====================

Text analysis tools:
- Text preprocessing and cleaning
- Complexity and readability metrics
- Similarity measures
- Topic modeling (LDA, NMF)
- TF-IDF vectorization
- BERT embeddings and classification
- Ollama LLM integration for text classification
- Sentiment analysis (transformer, lexicon, and LLM-based)
- N-gram analysis

Author: whatsthepoint Package
"""

import re
import string
from collections import Counter
from typing import List, Union, Dict, Tuple, Optional
import warnings

import numpy as np
import pandas as pd

# Optional dependencies
try:
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.decomposition import LatentDirichletAllocation, NMF
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    warnings.warn("scikit-learn not available. Some text analysis features will be limited.")

try:
    import nltk
    HAS_NLTK = True
except ImportError:
    HAS_NLTK = False

try:
    from transformers import (
        AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
        pipeline
    )
    import torch
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


# ============================================================================
# Text Preprocessing
# ============================================================================

def clean_text(
    text: str,
    lowercase: bool = True,
    remove_punctuation: bool = True,
    remove_numbers: bool = False,
    remove_extra_spaces: bool = True
) -> str:
    """
    Clean and preprocess text.

    Parameters
    ----------
    text : str
        Input text to clean
    lowercase : bool, default=True
        Convert text to lowercase
    remove_punctuation : bool, default=True
        Remove punctuation marks
    remove_numbers : bool, default=False
        Remove numeric characters
    remove_extra_spaces : bool, default=True
        Remove extra whitespace

    Returns
    -------
    str
        Cleaned text
    """
    if not isinstance(text, str):
        return ""

    # Convert to lowercase
    if lowercase:
        text = text.lower()

    # Remove punctuation
    if remove_punctuation:
        text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove numbers
    if remove_numbers:
        text = re.sub(r'\d+', '', text)

    # Remove extra spaces
    if remove_extra_spaces:
        text = ' '.join(text.split())

    return text


def tokenize(text: str, method: str = 'simple') -> List[str]:
    """
    Tokenize text into words.

    Parameters
    ----------
    text : str
        Input text to tokenize
    method : str, default='simple'
        Tokenization method: 'simple' or 'nltk'

    Returns
    -------
    list of str
        List of tokens
    """
    if method == 'simple':
        return text.split()
    elif method == 'nltk' and HAS_NLTK:
        return nltk.word_tokenize(text)
    else:
        return text.split()


def remove_stopwords(
    tokens: List[str],
    stopwords: Optional[List[str]] = None,
    language: str = 'english'
) -> List[str]:
    """
    Remove stopwords from token list.

    Parameters
    ----------
    tokens : list of str
        List of tokens
    stopwords : list of str, optional
        Custom stopword list. If None, uses nltk stopwords
    language : str, default='english'
        Language for stopwords (if using nltk)

    Returns
    -------
    list of str
        Tokens with stopwords removed
    """
    if stopwords is None:
        if HAS_NLTK:
            try:
                from nltk.corpus import stopwords as nltk_stopwords
                stopwords = set(nltk_stopwords.words(language))
            except LookupError:
                # If stopwords not downloaded, use basic English stopwords
                stopwords = {
                    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're",
                    "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he',
                    'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's",
                    'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
                    'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are',
                    'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do',
                    'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because',
                    'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against',
                    'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
                    'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
                    'further', 'then', 'once'
                }
        else:
            # Basic stopwords if nltk not available
            stopwords = {
                'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
                'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
                'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these',
                'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'
            }

    return [token for token in tokens if token.lower() not in stopwords]


# ============================================================================
# Complexity Metrics
# ============================================================================

def flesch_reading_ease(text: str) -> float:
    """
    Calculate Flesch Reading Ease score.

    Score interpretation:
    90-100: Very Easy
    80-89: Easy
    70-79: Fairly Easy
    60-69: Standard
    50-59: Fairly Difficult
    30-49: Difficult
    0-29: Very Confusing

    Parameters
    ----------
    text : str
        Input text

    Returns
    -------
    float
        Flesch Reading Ease score
    """
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    words = text.split()
    syllables = sum(_count_syllables(word) for word in words)

    if len(sentences) == 0 or len(words) == 0:
        return 0.0

    # Flesch Reading Ease = 206.835 - 1.015 * (total words / total sentences) - 84.6 * (total syllables / total words)
    score = 206.835 - 1.015 * (len(words) / len(sentences)) - 84.6 * (syllables / len(words))
    return score


def flesch_kincaid_grade(text: str) -> float:
    """
    Calculate Flesch-Kincaid Grade Level.

    Returns the U.S. grade level required to understand the text.

    Parameters
    ----------
    text : str
        Input text

    Returns
    -------
    float
        Grade level
    """
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    words = text.split()
    syllables = sum(_count_syllables(word) for word in words)

    if len(sentences) == 0 or len(words) == 0:
        return 0.0

    # Grade = 0.39 * (total words / total sentences) + 11.8 * (total syllables / total words) - 15.59
    grade = 0.39 * (len(words) / len(sentences)) + 11.8 * (syllables / len(words)) - 15.59
    return max(0, grade)


def _count_syllables(word: str) -> int:
    """Count syllables in a word (approximate)."""
    word = word.lower()
    vowels = 'aeiouy'
    syllable_count = 0
    previous_was_vowel = False

    for char in word:
        is_vowel = char in vowels
        if is_vowel and not previous_was_vowel:
            syllable_count += 1
        previous_was_vowel = is_vowel

    # Adjust for silent 'e'
    if word.endswith('e'):
        syllable_count -= 1

    # Every word has at least one syllable
    if syllable_count == 0:
        syllable_count = 1

    return syllable_count


def lexical_diversity(text: str) -> float:
    """
    Calculate lexical diversity (Type-Token Ratio).

    Parameters
    ----------
    text : str
        Input text

    Returns
    -------
    float
        Lexical diversity score (0-1)
    """
    tokens = tokenize(clean_text(text, remove_punctuation=True))

    if len(tokens) == 0:
        return 0.0

    return len(set(tokens)) / len(tokens)


def average_word_length(text: str) -> float:
    """
    Calculate average word length in characters.

    Parameters
    ----------
    text : str
        Input text

    Returns
    -------
    float
        Average word length
    """
    words = text.split()

    if len(words) == 0:
        return 0.0

    return sum(len(word) for word in words) / len(words)


def complexity_scores(text: str) -> Dict[str, float]:
    """
    Calculate multiple complexity and readability scores.

    Parameters
    ----------
    text : str
        Input text

    Returns
    -------
    dict
        Dictionary of complexity metrics
    """
    return {
        'flesch_reading_ease': flesch_reading_ease(text),
        'flesch_kincaid_grade': flesch_kincaid_grade(text),
        'lexical_diversity': lexical_diversity(text),
        'avg_word_length': average_word_length(text),
        'word_count': len(text.split()),
        'sentence_count': len(re.split(r'[.!?]+', text))
    }


# ============================================================================
# Similarity Measures
# ============================================================================

def jaccard_similarity(text1: str, text2: str) -> float:
    """
    Calculate Jaccard similarity between two texts.

    Parameters
    ----------
    text1 : str
        First text
    text2 : str
        Second text

    Returns
    -------
    float
        Jaccard similarity (0-1)
    """
    tokens1 = set(tokenize(clean_text(text1)))
    tokens2 = set(tokenize(clean_text(text2)))

    if len(tokens1) == 0 and len(tokens2) == 0:
        return 1.0

    intersection = tokens1.intersection(tokens2)
    union = tokens1.union(tokens2)

    return len(intersection) / len(union) if len(union) > 0 else 0.0


def cosine_similarity_texts(texts: List[str], method: str = 'tfidf') -> np.ndarray:
    """
    Calculate pairwise cosine similarity between texts.

    Parameters
    ----------
    texts : list of str
        List of text documents
    method : str, default='tfidf'
        Vectorization method: 'tfidf' or 'count'

    Returns
    -------
    np.ndarray
        Similarity matrix (n_texts x n_texts)
    """
    if not HAS_SKLEARN:
        raise ImportError("scikit-learn is required for cosine similarity")

    if method == 'tfidf':
        vectorizer = TfidfVectorizer()
    else:
        vectorizer = CountVectorizer()

    vectors = vectorizer.fit_transform(texts)
    return cosine_similarity(vectors)


def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Calculate Levenshtein (edit) distance between two strings.

    Parameters
    ----------
    s1 : str
        First string
    s2 : str
        Second string

    Returns
    -------
    int
        Edit distance
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # Cost of insertions, deletions, or substitutions
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


# ============================================================================
# TF-IDF and Vectorization
# ============================================================================

def create_tfidf_matrix(
    texts: List[str],
    max_features: Optional[int] = None,
    ngram_range: Tuple[int, int] = (1, 1),
    min_df: int = 1,
    max_df: float = 1.0
) -> Tuple[pd.DataFrame, TfidfVectorizer]:
    """
    Create TF-IDF matrix from texts.

    Parameters
    ----------
    texts : list of str
        List of text documents
    max_features : int, optional
        Maximum number of features to extract
    ngram_range : tuple, default=(1, 1)
        Range of n-grams to extract
    min_df : int, default=1
        Minimum document frequency
    max_df : float, default=1.0
        Maximum document frequency (proportion)

    Returns
    -------
    pd.DataFrame
        TF-IDF matrix (documents x terms)
    TfidfVectorizer
        Fitted vectorizer
    """
    if not HAS_SKLEARN:
        raise ImportError("scikit-learn is required for TF-IDF")

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df
    )

    tfidf_matrix = vectorizer.fit_transform(texts)

    # Convert to DataFrame
    df = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=vectorizer.get_feature_names_out()
    )

    return df, vectorizer


def get_top_tfidf_terms(
    tfidf_df: pd.DataFrame,
    n_terms: int = 10,
    by_document: bool = False
) -> Union[pd.DataFrame, pd.Series]:
    """
    Get top TF-IDF terms.

    Parameters
    ----------
    tfidf_df : pd.DataFrame
        TF-IDF matrix from create_tfidf_matrix
    n_terms : int, default=10
        Number of top terms to return
    by_document : bool, default=False
        If True, return top terms per document; if False, return overall top terms

    Returns
    -------
    pd.DataFrame or pd.Series
        Top terms (and scores)
    """
    if by_document:
        top_terms = {}
        for idx in tfidf_df.index:
            row = tfidf_df.loc[idx]
            top = row.nlargest(n_terms)
            top_terms[idx] = top
        return pd.DataFrame(top_terms).T
    else:
        # Overall top terms (average TF-IDF across documents)
        avg_tfidf = tfidf_df.mean(axis=0)
        return avg_tfidf.nlargest(n_terms)


# ============================================================================
# Topic Modeling
# ============================================================================

class TopicModel:
    """
    Topic modeling using LDA or NMF.

    Parameters
    ----------
    n_topics : int, default=5
        Number of topics to extract
    method : str, default='lda'
        Method: 'lda' (Latent Dirichlet Allocation) or 'nmf' (Non-negative Matrix Factorization)
    max_features : int, optional
        Maximum number of features for vectorization
    ngram_range : tuple, default=(1, 1)
        Range of n-grams
    random_state : int, optional
        Random state for reproducibility

    Attributes
    ----------
    model : object
        Fitted topic model
    vectorizer : object
        Fitted vectorizer
    doc_topic_matrix : np.ndarray
        Document-topic distribution matrix
    """

    def __init__(
        self,
        n_topics: int = 5,
        method: str = 'lda',
        max_features: Optional[int] = None,
        ngram_range: Tuple[int, int] = (1, 1),
        random_state: Optional[int] = None
    ):
        if not HAS_SKLEARN:
            raise ImportError("scikit-learn is required for topic modeling")

        self.n_topics = n_topics
        self.method = method.lower()
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.random_state = random_state

        self.model = None
        self.vectorizer = None
        self.doc_topic_matrix = None

    def fit(self, texts: List[str]):
        """
        Fit topic model to texts.

        Parameters
        ----------
        texts : list of str
            List of text documents

        Returns
        -------
        self
        """
        # Vectorize texts
        if self.method == 'lda':
            # LDA works with count vectors
            self.vectorizer = CountVectorizer(
                max_features=self.max_features,
                ngram_range=self.ngram_range
            )
            doc_term_matrix = self.vectorizer.fit_transform(texts)

            # Fit LDA
            self.model = LatentDirichletAllocation(
                n_components=self.n_topics,
                random_state=self.random_state
            )
        elif self.method == 'nmf':
            # NMF works with TF-IDF
            self.vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                ngram_range=self.ngram_range
            )
            doc_term_matrix = self.vectorizer.fit_transform(texts)

            # Fit NMF
            self.model = NMF(
                n_components=self.n_topics,
                random_state=self.random_state
            )
        else:
            raise ValueError(f"Unknown method: {self.method}. Use 'lda' or 'nmf'")

        # Fit model and get document-topic matrix
        self.doc_topic_matrix = self.model.fit_transform(doc_term_matrix)

        return self

    def get_topics(self, n_words: int = 10) -> pd.DataFrame:
        """
        Get top words for each topic.

        Parameters
        ----------
        n_words : int, default=10
            Number of top words per topic

        Returns
        -------
        pd.DataFrame
            Topic-word matrix
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        feature_names = self.vectorizer.get_feature_names_out()

        topics = {}
        for topic_idx, topic in enumerate(self.model.components_):
            top_indices = topic.argsort()[-n_words:][::-1]
            top_words = [feature_names[i] for i in top_indices]
            top_weights = [topic[i] for i in top_indices]
            topics[f'Topic {topic_idx + 1}'] = top_words

        return pd.DataFrame(topics)

    def get_document_topics(self) -> pd.DataFrame:
        """
        Get topic distribution for each document.

        Returns
        -------
        pd.DataFrame
            Document-topic distribution
        """
        if self.doc_topic_matrix is None:
            raise ValueError("Model not fitted. Call fit() first.")

        return pd.DataFrame(
            self.doc_topic_matrix,
            columns=[f'Topic {i+1}' for i in range(self.n_topics)]
        )

    def get_dominant_topic(self) -> pd.Series:
        """
        Get dominant topic for each document.

        Returns
        -------
        pd.Series
            Dominant topic index for each document
        """
        doc_topics = self.get_document_topics()
        return doc_topics.idxmax(axis=1)


# ============================================================================
# N-gram Analysis
# ============================================================================

def extract_ngrams(
    text: str,
    n: int = 2,
    top_k: Optional[int] = None
) -> List[Tuple[Tuple[str, ...], int]]:
    """
    Extract n-grams from text.

    Parameters
    ----------
    text : str
        Input text
    n : int, default=2
        N-gram size (2 for bigrams, 3 for trigrams, etc.)
    top_k : int, optional
        Return only top K most common n-grams

    Returns
    -------
    list of tuples
        List of (ngram, count) tuples
    """
    tokens = tokenize(clean_text(text))

    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i:i+n])
        ngrams.append(ngram)

    ngram_counts = Counter(ngrams)

    if top_k is not None:
        return ngram_counts.most_common(top_k)
    else:
        return list(ngram_counts.items())


def ngram_frequency(
    texts: List[str],
    n: int = 2,
    top_k: int = 20
) -> pd.DataFrame:
    """
    Calculate n-gram frequencies across multiple texts.

    Parameters
    ----------
    texts : list of str
        List of text documents
    n : int, default=2
        N-gram size
    top_k : int, default=20
        Number of top n-grams to return

    Returns
    -------
    pd.DataFrame
        N-gram frequency table
    """
    all_ngrams = []

    for text in texts:
        ngrams = extract_ngrams(text, n=n)
        all_ngrams.extend(ngrams)

    # Aggregate counts
    ngram_counter = Counter()
    for ngram, count in all_ngrams:
        ngram_counter[ngram] += count

    # Get top K
    top_ngrams = ngram_counter.most_common(top_k)

    return pd.DataFrame(
        top_ngrams,
        columns=['ngram', 'frequency']
    )


# ============================================================================
# BERT Embeddings and Classification
# ============================================================================

class BERTModel:
    """
    BERT-based text embeddings and classification.

    Parameters
    ----------
    model_name : str, default='bert-base-uncased'
        Pretrained model name from HuggingFace
    task : str, default='embedding'
        Task type: 'embedding', 'classification', or 'sentiment'
    device : str, optional
        Device to use ('cuda' or 'cpu'). If None, automatically detects.

    Attributes
    ----------
    tokenizer : AutoTokenizer
        BERT tokenizer
    model : AutoModel or AutoModelForSequenceClassification
        BERT model
    """

    def __init__(
        self,
        model_name: str = 'bert-base-uncased',
        task: str = 'embedding',
        device: Optional[str] = None
    ):
        if not HAS_TRANSFORMERS:
            raise ImportError("transformers and torch are required for BERT functionality. Install them with: pip install transformers torch")

        self.model_name = model_name
        self.task = task.lower()

        # Set device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if self.task == 'embedding':
            self.model = AutoModel.from_pretrained(model_name)
        elif self.task in ['classification', 'sentiment']:
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        else:
            raise ValueError(f"Unknown task: {task}. Use 'embedding', 'classification', or 'sentiment'")

        self.model.to(self.device)
        self.model.eval()

    def get_embeddings(
        self,
        texts: Union[str, List[str]],
        pooling: str = 'mean'
    ) -> np.ndarray:
        """
        Get BERT embeddings for text(s).

        Parameters
        ----------
        texts : str or list of str
            Input text(s)
        pooling : str, default='mean'
            Pooling strategy: 'mean', 'max', or 'cls'

        Returns
        -------
        np.ndarray
            Embeddings array (n_texts x embedding_dim)
        """
        if isinstance(texts, str):
            texts = [texts]

        embeddings = []

        with torch.no_grad():
            for text in texts:
                # Tokenize
                inputs = self.tokenizer(
                    text,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=512
                ).to(self.device)

                # Get model outputs
                outputs = self.model(**inputs)

                # Apply pooling
                if pooling == 'cls':
                    # Use [CLS] token embedding
                    embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                elif pooling == 'mean':
                    # Mean pooling
                    embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                elif pooling == 'max':
                    # Max pooling
                    embedding = outputs.last_hidden_state.max(dim=1)[0].cpu().numpy()
                else:
                    raise ValueError(f"Unknown pooling: {pooling}")

                embeddings.append(embedding[0])

        return np.array(embeddings)

    def predict(
        self,
        texts: Union[str, List[str]],
        return_probs: bool = False
    ) -> Union[List[int], Tuple[List[int], np.ndarray]]:
        """
        Predict class labels for text(s).

        Parameters
        ----------
        texts : str or list of str
            Input text(s)
        return_probs : bool, default=False
            If True, also return class probabilities

        Returns
        -------
        list of int
            Predicted class labels
        np.ndarray, optional
            Class probabilities (if return_probs=True)
        """
        if self.task == 'embedding':
            raise ValueError("Model is configured for embeddings, not classification")

        if isinstance(texts, str):
            texts = [texts]

        predictions = []
        probabilities = []

        with torch.no_grad():
            for text in texts:
                inputs = self.tokenizer(
                    text,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=512
                ).to(self.device)

                outputs = self.model(**inputs)
                logits = outputs.logits

                # Get predicted class
                pred = torch.argmax(logits, dim=1).cpu().numpy()[0]
                predictions.append(pred)

                # Get probabilities
                if return_probs:
                    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                    probabilities.append(probs)

        if return_probs:
            return predictions, np.array(probabilities)
        return predictions

    def similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity between two texts using BERT embeddings.

        Parameters
        ----------
        text1 : str
            First text
        text2 : str
            Second text

        Returns
        -------
        float
            Cosine similarity score
        """
        embeddings = self.get_embeddings([text1, text2])

        # Cosine similarity
        similarity = np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
        )

        return float(similarity)


def bert_embeddings(
    texts: Union[str, List[str]],
    model_name: str = 'bert-base-uncased',
    pooling: str = 'mean'
) -> np.ndarray:
    """
    Quick function to get BERT embeddings.

    Parameters
    ----------
    texts : str or list of str
        Input text(s)
    model_name : str, default='bert-base-uncased'
        Pretrained model name
    pooling : str, default='mean'
        Pooling strategy

    Returns
    -------
    np.ndarray
        Embeddings array
    """
    model = BERTModel(model_name=model_name, task='embedding')
    return model.get_embeddings(texts, pooling=pooling)


# ============================================================================
# Ollama Integration for Text Classification
# ============================================================================

class OllamaClassifier:
    """
    Text classification using Ollama LLMs.

    Parameters
    ----------
    model : str, default='llama2'
        Ollama model name (e.g., 'llama2', 'mistral', 'codellama')
    base_url : str, default='http://localhost:11434'
        Ollama API base URL
    temperature : float, default=0.1
        Sampling temperature (lower = more deterministic)

    Examples
    --------
    >>> classifier = OllamaClassifier(model='llama2')
    >>> result = classifier.classify(
    ...     text="This product is amazing!",
    ...     categories=["positive", "negative", "neutral"]
    ... )
    """

    def __init__(
        self,
        model: str = 'llama2',
        base_url: str = 'http://localhost:11434',
        temperature: float = 0.1
    ):
        if not HAS_REQUESTS:
            raise ImportError("requests is required for Ollama integration. Install it with: pip install requests")

        self.model = model
        self.base_url = base_url.rstrip('/')
        self.temperature = temperature

    def _call_ollama(self, prompt: str) -> str:
        """Call Ollama API with a prompt."""
        url = f"{self.base_url}/api/generate"

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "temperature": self.temperature
        }

        try:
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            return response.json()['response']
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Ollama API call failed: {e}")

    def classify(
        self,
        text: str,
        categories: List[str],
        description: Optional[str] = None
    ) -> Dict[str, Union[str, float]]:
        """
        Classify text into one of the provided categories.

        Parameters
        ----------
        text : str
            Text to classify
        categories : list of str
            List of possible categories
        description : str, optional
            Additional context or instructions for classification

        Returns
        -------
        dict
            Classification result with 'category' and 'explanation'
        """
        # Build prompt
        categories_str = ", ".join([f"'{cat}'" for cat in categories])

        prompt = f"""Classify the following text into one of these categories: {categories_str}

Text: "{text}"
"""

        if description:
            prompt += f"\nContext: {description}\n"

        prompt += f"""
Respond with ONLY the category name from the list: {categories_str}
Do not include any explanation or additional text, just the category name.

Category:"""

        # Get response
        response = self._call_ollama(prompt).strip()

        # Clean response (remove quotes, extra whitespace)
        response = response.strip('"\'').strip()

        # Try to match to a valid category (case-insensitive)
        matched_category = None
        response_lower = response.lower()
        for cat in categories:
            if cat.lower() in response_lower or response_lower in cat.lower():
                matched_category = cat
                break

        if matched_category is None:
            # If no match, take first word as best guess
            matched_category = response.split()[0] if response else categories[0]
            warnings.warn(f"Could not match response '{response}' to categories. Using: {matched_category}")

        return {
            'category': matched_category,
            'raw_response': response
        }

    def classify_batch(
        self,
        texts: List[str],
        categories: List[str],
        description: Optional[str] = None
    ) -> List[Dict[str, Union[str, float]]]:
        """
        Classify multiple texts.

        Parameters
        ----------
        texts : list of str
            Texts to classify
        categories : list of str
            List of possible categories
        description : str, optional
            Additional context or instructions

        Returns
        -------
        list of dict
            Classification results
        """
        results = []
        for text in texts:
            result = self.classify(text, categories, description)
            results.append(result)
        return results

    def extract_entities(
        self,
        text: str,
        entity_types: Optional[List[str]] = None
    ) -> List[Dict[str, str]]:
        """
        Extract named entities from text.

        Parameters
        ----------
        text : str
            Input text
        entity_types : list of str, optional
            Types of entities to extract (e.g., ['person', 'organization', 'location'])

        Returns
        -------
        list of dict
            Extracted entities with 'text' and 'type' keys
        """
        if entity_types:
            types_str = ", ".join(entity_types)
            prompt = f"""Extract entities of the following types from the text: {types_str}

Text: "{text}"

List each entity on a new line in the format: entity_text | entity_type

Entities:
"""
        else:
            prompt = f"""Extract all named entities (people, organizations, locations, etc.) from the text.

Text: "{text}"

List each entity on a new line in the format: entity_text | entity_type

Entities:
"""

        response = self._call_ollama(prompt)

        # Parse response
        entities = []
        for line in response.strip().split('\n'):
            line = line.strip()
            if '|' in line:
                parts = line.split('|')
                if len(parts) >= 2:
                    entities.append({
                        'text': parts[0].strip(),
                        'type': parts[1].strip()
                    })

        return entities

    def summarize(
        self,
        text: str,
        max_length: Optional[int] = None
    ) -> str:
        """
        Generate a summary of the text.

        Parameters
        ----------
        text : str
            Text to summarize
        max_length : int, optional
            Maximum length of summary in words

        Returns
        -------
        str
            Summary text
        """
        prompt = f"""Provide a concise summary of the following text:

Text: "{text}"
"""

        if max_length:
            prompt += f"\nLimit the summary to approximately {max_length} words.\n"

        prompt += "\nSummary:"

        return self._call_ollama(prompt).strip()


# ============================================================================
# Sentiment Analysis
# ============================================================================

class SentimentAnalyzer:
    """
    Sentiment analysis using various methods.

    Parameters
    ----------
    method : str, default='transformer'
        Method to use: 'transformer', 'lexicon', or 'ollama'
    model_name : str, optional
        Model name for transformer or ollama methods
    device : str, optional
        Device for transformer models ('cuda' or 'cpu')

    Examples
    --------
    >>> analyzer = SentimentAnalyzer(method='transformer')
    >>> result = analyzer.analyze("This is a great product!")
    >>> print(result['label'], result['score'])
    """

    def __init__(
        self,
        method: str = 'transformer',
        model_name: Optional[str] = None,
        device: Optional[str] = None
    ):
        self.method = method.lower()

        if self.method == 'transformer':
            if not HAS_TRANSFORMERS:
                raise ImportError("transformers and torch are required for the transformer method. Install them with: pip install transformers torch")

            if model_name is None:
                model_name = 'distilbert-base-uncased-finetuned-sst-2-english'

            self.pipeline = pipeline(
                "sentiment-analysis",
                model=model_name,
                device=0 if (device == 'cuda' or (device is None and torch.cuda.is_available())) else -1
            )

        elif self.method == 'lexicon':
            # Use simple lexicon-based approach
            self.positive_words = {
                'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
                'awesome', 'love', 'best', 'perfect', 'happy', 'beautiful',
                'brilliant', 'outstanding', 'superb', 'delightful', 'enjoy'
            }
            self.negative_words = {
                'bad', 'terrible', 'awful', 'horrible', 'worst', 'hate',
                'poor', 'disappointing', 'useless', 'waste', 'annoying',
                'frustrating', 'broken', 'garbage', 'fail', 'disaster'
            }

        elif self.method == 'ollama':
            if model_name is None:
                model_name = 'llama2'
            self.ollama = OllamaClassifier(model=model_name)

        else:
            raise ValueError(f"Unknown method: {method}. Use 'transformer', 'lexicon', or 'ollama'")

    def analyze(
        self,
        text: Union[str, List[str]]
    ) -> Union[Dict[str, Union[str, float]], List[Dict[str, Union[str, float]]]]:
        """
        Analyze sentiment of text(s).

        Parameters
        ----------
        text : str or list of str
            Input text(s)

        Returns
        -------
        dict or list of dict
            Sentiment analysis results with 'label' and 'score' keys
        """
        single_input = isinstance(text, str)
        if single_input:
            texts = [text]
        else:
            texts = text

        if self.method == 'transformer':
            results = self.pipeline(texts)
            # Standardize output format
            for result in results:
                result['label'] = result['label'].upper()

        elif self.method == 'lexicon':
            results = []
            for txt in texts:
                score = self._lexicon_score(txt)
                if score > 0.1:
                    label = 'POSITIVE'
                elif score < -0.1:
                    label = 'NEGATIVE'
                else:
                    label = 'NEUTRAL'

                results.append({
                    'label': label,
                    'score': abs(score)
                })

        elif self.method == 'ollama':
            results = []
            for txt in texts:
                result = self.ollama.classify(
                    txt,
                    categories=['positive', 'negative', 'neutral'],
                    description="Sentiment analysis"
                )
                results.append({
                    'label': result['category'].upper(),
                    'score': 1.0  # Ollama doesn't provide confidence scores by default
                })

        return results[0] if single_input else results

    def _lexicon_score(self, text: str) -> float:
        """Calculate sentiment score using lexicon method."""
        tokens = tokenize(clean_text(text.lower()))

        positive_count = sum(1 for token in tokens if token in self.positive_words)
        negative_count = sum(1 for token in tokens if token in self.negative_words)

        total_words = len(tokens)
        if total_words == 0:
            return 0.0

        # Normalize by text length
        score = (positive_count - negative_count) / total_words
        return score

    def analyze_aspects(
        self,
        text: str,
        aspects: List[str]
    ) -> Dict[str, Dict[str, Union[str, float]]]:
        """
        Analyze sentiment for specific aspects mentioned in text.

        Parameters
        ----------
        text : str
            Input text
        aspects : list of str
            Aspects to analyze (e.g., ['quality', 'price', 'service'])

        Returns
        -------
        dict
            Sentiment for each aspect
        """
        if self.method != 'ollama':
            raise NotImplementedError("Aspect-based sentiment analysis is only available with ollama method")

        results = {}
        for aspect in aspects:
            # Create aspect-specific prompt
            prompt = f"""Analyze the sentiment about "{aspect}" in the following text.

Text: "{text}"

Classify the sentiment about {aspect} as: positive, negative, or neutral.
If {aspect} is not mentioned, respond with "not_mentioned".

Sentiment:"""

            response = self.ollama._call_ollama(prompt).strip().lower()

            if 'not' in response and 'mention' in response:
                label = 'NOT_MENTIONED'
                score = 0.0
            elif 'positive' in response:
                label = 'POSITIVE'
                score = 1.0
            elif 'negative' in response:
                label = 'NEGATIVE'
                score = 1.0
            else:
                label = 'NEUTRAL'
                score = 0.5

            results[aspect] = {'label': label, 'score': score}

        return results


def analyze_sentiment(
    texts: Union[str, List[str]],
    method: str = 'transformer'
) -> Union[Dict[str, Union[str, float]], List[Dict[str, Union[str, float]]]]:
    """
    Quick function to analyze sentiment.

    Parameters
    ----------
    texts : str or list of str
        Input text(s)
    method : str, default='transformer'
        Analysis method ('transformer', 'lexicon', or 'ollama')

    Returns
    -------
    dict or list of dict
        Sentiment analysis results
    """
    analyzer = SentimentAnalyzer(method=method)
    return analyzer.analyze(texts)


# ============================================================================
# Convenience Functions
# ============================================================================

def analyze_corpus(
    texts: List[str],
    text_ids: Optional[List] = None,
    include_complexity: bool = True,
    include_topics: bool = True,
    n_topics: int = 5
) -> Dict[str, pd.DataFrame]:
    """
    Corpus-level summary analysis.

    Parameters
    ----------
    texts : list of str
        List of text documents
    text_ids : list, optional
        Document identifiers
    include_complexity : bool, default=True
        Include complexity metrics
    include_topics : bool, default=True
        Include topic modeling
    n_topics : int, default=5
        Number of topics for topic modeling

    Returns
    -------
    dict
        Dictionary containing:
        - 'complexity': Complexity metrics (if requested)
        - 'topics': Topic model results (if requested)
        - 'tfidf': Top TF-IDF terms
    """
    if text_ids is None:
        text_ids = list(range(len(texts)))

    results = {}

    # Complexity metrics
    if include_complexity:
        complexity_data = []
        for text_id, text in zip(text_ids, texts):
            scores = complexity_scores(text)
            scores['text_id'] = text_id
            complexity_data.append(scores)
        results['complexity'] = pd.DataFrame(complexity_data)

    # Topic modeling
    if include_topics and HAS_SKLEARN:
        topic_model = TopicModel(n_topics=n_topics, method='lda')
        topic_model.fit(texts)

        results['topics'] = {
            'topic_words': topic_model.get_topics(),
            'document_topics': topic_model.get_document_topics(),
            'dominant_topic': topic_model.get_dominant_topic()
        }

    # TF-IDF
    if HAS_SKLEARN:
        tfidf_df, _ = create_tfidf_matrix(texts)
        results['tfidf'] = get_top_tfidf_terms(tfidf_df, n_terms=20)

    return results
