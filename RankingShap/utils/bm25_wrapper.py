import numpy as np
from rank_bm25 import BM25Okapi
import re

# Try to import nltk, fallback to simple processing if not available
try:
    from nltk.stem import PorterStemmer

    stemmer = PorterStemmer()
    HAS_NLTK = True
except ImportError:
    HAS_NLTK = False
    print("Warning: NLTK not found. Stemming disabled. Install with `pip install nltk`")


def tokenize_and_stem(text):
    # Simple regex tokenization to remove punctuation
    tokens = re.findall(r"\b\w+\b", text.lower())
    if HAS_NLTK:
        return [stemmer.stem(t) for t in tokens]
    return tokens


class BM25Wrapper:
    def __init__(self, corpus_passages):
        """
        corpus_passages: List of strings (documents) to index.
        """
        # Tokenize and stem corpus
        self.tokenized_corpus = [tokenize_and_stem(doc) for doc in corpus_passages]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        self.query_tokens = []
        self.vocabulary = []
        self.debug = False

    def set_query(self, query_string, vocabulary):
        """
        Sets the current query and the vocabulary mapping for the binary matrix.
        """
        self.query_tokens = tokenize_and_stem(query_string)
        self.vocabulary = vocabulary

    def predict(self, binary_feature_matrix):
        """
        Input: (n_samples, n_features) binary matrix.
        1 (or >0) = Keep word, 0 = Drop word.
        """
        scores = []

        for i, doc_vector in enumerate(binary_feature_matrix):
            # Reconstruct the document from the binary vector
            # Use > 0.5 to handle float inputs safely (1.0 vs 1)
            reconstructed_tokens = [
                self.vocabulary[idx]
                for idx, val in enumerate(doc_vector)
                if val > 0.5 and idx < len(self.vocabulary)
            ]

            if self.debug and i == 0:
                print(f"[DEBUG] Doc {i} Reconstructed: {reconstructed_tokens}")

            score = self._score_modified_doc(reconstructed_tokens)
            scores.append(score)

        return np.array(scores)

    def _score_modified_doc(self, doc_tokens):
        # Manual BM25 scoring for a single dynamic document
        doc_len = len(doc_tokens)
        if doc_len == 0:
            return 0.0

        score = 0.0
        doc_freqs = {}
        for t in doc_tokens:
            doc_freqs[t] = doc_freqs.get(t, 0) + 1

        for q in self.query_tokens:
            if q in doc_freqs:
                # Check if q is in the BM25 IDF dictionary
                if q not in self.bm25.idf:
                    if self.debug:
                        print(
                            f"[DEBUG] Warning: Query term '{q}' not in BM25 Corpus IDF."
                        )
                    continue

                freq = doc_freqs[q]
                idf = self.bm25.idf[q]

                # Standard BM25 Formula
                numerator = idf * freq * (self.bm25.k1 + 1)
                denominator = freq + self.bm25.k1 * (
                    1 - self.bm25.b + self.bm25.b * doc_len / self.bm25.avgdl
                )
                term_score = numerator / denominator
                score += term_score

                if self.debug:
                    print(
                        f"[DEBUG] Term '{q}': freq={freq}, idf={idf:.3f}, score={term_score:.3f}"
                    )

        return score
