"""
BERT Cross-Encoder Wrapper for RankingSHAP

This module provides a BERT-based ranking model that can be used with RankingSHAP
for text-based feature attribution explanations. It uses a cross-encoder architecture
where query and document are concatenated and scored together.

Compatible with Snellius GPU cluster.
"""

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re

# Try to import nltk for stemming (used in vocabulary building)
try:
    from nltk.stem import PorterStemmer

    stemmer = PorterStemmer()
    HAS_NLTK = True
except ImportError:
    HAS_NLTK = False
    print("Warning: NLTK not found. Stemming disabled. Install with `pip install nltk`")


def tokenize_and_stem(text):
    """Simple tokenization and optional stemming for vocabulary building."""
    tokens = re.findall(r"\b\w+\b", text.lower())
    if HAS_NLTK:
        return [stemmer.stem(t) for t in tokens]
    return tokens


class BERTRankingWrapper:
    """
    BERT Cross-Encoder wrapper for ranking tasks.

    This class wraps a pre-trained BERT cross-encoder model (e.g., ms-marco-MiniLM)
    to provide relevance scores for query-document pairs. It's designed to work
    with RankingSHAP's masking mechanism where features (words) can be removed
    from documents.

    Attributes:
        model_name (str): Name of the pre-trained model from HuggingFace
        device (str): Device to run inference on ('cuda' or 'cpu')
        batch_size (int): Batch size for inference
    """

    def __init__(
        self,
        corpus_passages,
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
        device=None,
        batch_size=32,
        max_length=512,
    ):
        """
        Initialize the BERT ranking wrapper.

        Args:
            corpus_passages: List of document strings (stored for reference)
            model_name: HuggingFace model name for cross-encoder
            device: 'cuda', 'cpu', or None (auto-detect)
            batch_size: Batch size for inference
            max_length: Maximum sequence length for tokenization
        """
        self.corpus_passages = corpus_passages
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length

        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"[BERTRankingWrapper] Using device: {self.device}")
        print(f"[BERTRankingWrapper] Loading model: {model_name}")

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        # Query and vocabulary for masking
        self.query_text = ""
        self.vocabulary = []
        self.debug = False

        # Cache original documents for reconstruction
        self.original_docs = corpus_passages

    def set_query(self, query_string, vocabulary):
        """
        Set the current query and vocabulary mapping for the binary feature matrix.

        Args:
            query_string: The query text
            vocabulary: List of vocabulary words (stemmed tokens)
        """
        self.query_text = query_string
        self.vocabulary = vocabulary

    def _reconstruct_document(self, doc_idx, binary_vector):
        """
        Reconstruct a document from a binary feature vector.

        This method takes the original document and removes words that are
        masked (have value 0 in the binary vector).

        Args:
            doc_idx: Index of the document in corpus
            binary_vector: Binary vector indicating which vocabulary words to keep

        Returns:
            Reconstructed document string with masked words removed
        """
        original_doc = self.original_docs[doc_idx]

        # Get set of allowed vocabulary words (those with 1 in binary vector)
        allowed_words = set()
        for idx, val in enumerate(binary_vector):
            if val > 0.5 and idx < len(self.vocabulary):
                allowed_words.add(self.vocabulary[idx])

        # Reconstruct document by filtering words
        original_tokens = re.findall(r"\b\w+\b", original_doc.lower())

        # Keep only tokens whose stemmed form is in allowed_words
        reconstructed_tokens = []
        for token in original_tokens:
            stemmed = stemmer.stem(token) if HAS_NLTK else token
            if stemmed in allowed_words:
                reconstructed_tokens.append(token)

        return " ".join(reconstructed_tokens)

    def predict(self, binary_feature_matrix):
        """
        Score documents based on the binary feature matrix.

        Each row in the matrix corresponds to a document. The binary values
        indicate which vocabulary words are present (1) or masked (0).

        Args:
            binary_feature_matrix: (n_docs, n_vocab) binary matrix

        Returns:
            numpy array of relevance scores for each document
        """
        scores = []

        # Reconstruct documents based on binary masks
        reconstructed_docs = []
        for doc_idx, doc_vector in enumerate(binary_feature_matrix):
            reconstructed = self._reconstruct_document(doc_idx, doc_vector)
            reconstructed_docs.append(reconstructed)

            if self.debug and doc_idx == 0:
                print(
                    f"[DEBUG] Doc {doc_idx} Original: {self.original_docs[doc_idx][:100]}..."
                )
                print(f"[DEBUG] Doc {doc_idx} Reconstructed: {reconstructed[:100]}...")

        # Score in batches
        with torch.no_grad():
            for i in range(0, len(reconstructed_docs), self.batch_size):
                batch_docs = reconstructed_docs[i : i + self.batch_size]

                # Prepare query-document pairs for cross-encoder
                pairs = [(self.query_text, doc) for doc in batch_docs]

                # Tokenize
                inputs = self.tokenizer(
                    [p[0] for p in pairs],  # queries
                    [p[1] for p in pairs],  # documents
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )

                # Move to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # Get scores
                outputs = self.model(**inputs)
                batch_scores = outputs.logits.squeeze(-1).cpu().numpy()

                # Handle single score case
                if batch_scores.ndim == 0:
                    batch_scores = np.array([batch_scores.item()])

                scores.extend(batch_scores.tolist())

        return np.array(scores)

    def score_pairs(self, query, documents):
        """
        Directly score query-document pairs without masking.

        Args:
            query: Query string
            documents: List of document strings

        Returns:
            numpy array of relevance scores
        """
        scores = []

        with torch.no_grad():
            for i in range(0, len(documents), self.batch_size):
                batch_docs = documents[i : i + self.batch_size]

                inputs = self.tokenizer(
                    [query] * len(batch_docs),
                    batch_docs,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )

                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.model(**inputs)
                batch_scores = outputs.logits.squeeze(-1).cpu().numpy()

                if batch_scores.ndim == 0:
                    batch_scores = np.array([batch_scores.item()])

                scores.extend(batch_scores.tolist())

        return np.array(scores)


class BERTBiEncoderWrapper:
    """
    BERT Bi-Encoder wrapper for more efficient ranking.

    This uses separate encoders for query and document, which is faster
    but slightly less accurate than cross-encoders. Useful for larger
    document sets.
    """

    def __init__(
        self,
        corpus_passages,
        model_name="sentence-transformers/msmarco-distilbert-base-v4",
        device=None,
        batch_size=32,
    ):
        """
        Initialize the BERT bi-encoder wrapper.

        Args:
            corpus_passages: List of document strings
            model_name: HuggingFace model name for sentence transformer
            device: 'cuda', 'cpu', or None (auto-detect)
            batch_size: Batch size for inference
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers required for bi-encoder. "
                "Install with: pip install sentence-transformers"
            )

        self.corpus_passages = corpus_passages
        self.model_name = model_name
        self.batch_size = batch_size

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"[BERTBiEncoderWrapper] Using device: {self.device}")
        print(f"[BERTBiEncoderWrapper] Loading model: {model_name}")

        self.model = SentenceTransformer(model_name, device=self.device)

        self.query_text = ""
        self.vocabulary = []
        self.original_docs = corpus_passages
        self.debug = False

    def set_query(self, query_string, vocabulary):
        """Set the current query and vocabulary."""
        self.query_text = query_string
        self.vocabulary = vocabulary
        # Pre-encode query for efficiency
        self.query_embedding = self.model.encode(
            query_string, convert_to_tensor=True, show_progress_bar=False
        )

    def _reconstruct_document(self, doc_idx, binary_vector):
        """Reconstruct document from binary vector."""
        original_doc = self.original_docs[doc_idx]

        allowed_words = set()
        for idx, val in enumerate(binary_vector):
            if val > 0.5 and idx < len(self.vocabulary):
                allowed_words.add(self.vocabulary[idx])

        original_tokens = re.findall(r"\b\w+\b", original_doc.lower())
        reconstructed_tokens = []
        for token in original_tokens:
            stemmed = stemmer.stem(token) if HAS_NLTK else token
            if stemmed in allowed_words:
                reconstructed_tokens.append(token)

        return " ".join(reconstructed_tokens)

    def predict(self, binary_feature_matrix):
        """Score documents based on binary feature matrix using cosine similarity."""
        reconstructed_docs = []
        for doc_idx, doc_vector in enumerate(binary_feature_matrix):
            reconstructed = self._reconstruct_document(doc_idx, doc_vector)
            reconstructed_docs.append(reconstructed if reconstructed else " ")

        # Encode documents
        doc_embeddings = self.model.encode(
            reconstructed_docs,
            convert_to_tensor=True,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )

        # Compute cosine similarity
        from sentence_transformers import util

        scores = util.cos_sim(self.query_embedding, doc_embeddings)[0]

        return scores.cpu().numpy()
