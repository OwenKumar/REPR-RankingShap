import numpy as np
import shap
import pandas as pd
from scipy.stats import kendalltau
from functools import partial
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize

# Download punkt for tokenization if not present
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")


def rank_list(vector):
    """
    Returns ndarray containing rank(i) for documents at position i
    (Higher score = Rank 1)
    """
    temp = vector.argsort()[::-1]
    ranks = np.empty_like(temp)
    ranks[temp] = np.arange(1, len(vector) + 1)
    return ranks


def placeholder_predict(array):
    return np.zeros(array.shape[0])


def text_model_predict_val(
    binary_mask_array,
    tokenized_documents,
    tokenized_query,
    vocabulary_list,
    original_ranks,
    similarity_coefficient,
):
    """
    This is the core wrapper.
    1. Receives a matrix of binary masks (1 = keep word, 0 = hide word).
    2. Reconstructs the documents based on the mask.
    3. Runs BM25 on the masked documents.
    4. Compares the new ranking to the original ranking.
    """
    scores = []

    # Iterate over each SHAP sample (each row is a different mask configuration)
    for mask in binary_mask_array:

        # Identify which tokens are "active" in this perturbation
        # mask is size [num_unique_tokens]
        # We need a set for O(1) lookup
        active_indices = np.where(mask == 1)[0]
        active_tokens = set([vocabulary_list[i] for i in active_indices])

        # Reconstruct the document corpus for this specific perturbation
        # If a token in the doc is NOT in active_tokens, it is "masked" (removed)
        masked_corpus = []
        for doc_tokens in tokenized_documents:
            filtered_doc = [t for t in doc_tokens if t in active_tokens]
            masked_corpus.append(filtered_doc)

        # Re-initialize BM25 with the masked corpus
        # Note: In a pure "feature attribution" sense, we usually freeze the model (IDF)
        # and only change input (TF). However, creating a new BM25 object is the
        # standard way to simulate "the model seeing different text".
        # If the doc becomes empty, BM25 handles it (score 0).
        if not any(masked_corpus):
            # Edge case: all words masked
            new_scores = np.zeros(len(tokenized_documents))
        else:
            bm25 = BM25Okapi(masked_corpus)
            new_scores = bm25.get_scores(tokenized_query)

        new_rank = rank_list(np.array(new_scores))

        # Calculate similarity (e.g., Kendall's Tau)
        sim_score = similarity_coefficient(original_ranks, new_rank)
        scores.append(sim_score)

    return np.array(scores)


class TextRankingShap:
    def __init__(
        self,
        nsample_permutations=5000,
        rank_similarity_coefficient=lambda x, y: kendalltau(x, y)[0],
    ):
        self.nsamples = nsample_permutations
        self.rank_similarity_coefficient = rank_similarity_coefficient
        self.explainer = None

    def get_query_explanation(self, query_str, documents_list):
        """
        query_str: "what is the capital of france"
        documents_list: ["paris is the capital", "france is a country", ...]
        """

        # 1. Tokenize inputs
        tokenized_query = word_tokenize(query_str.lower())
        tokenized_docs = [word_tokenize(doc.lower()) for doc in documents_list]

        # 2. Build Vocabulary (The Features)
        # We only care about words that actually appear in these specific documents.
        # This reduces feature space from 30k+ (English dict) to ~50-200 (Local context).
        # This is CRITICAL for efficiency.
        vocabulary = set()
        for doc in tokenized_docs:
            vocabulary.update(doc)

        vocabulary_list = sorted(list(vocabulary))
        num_features = len(vocabulary_list)

        print(f"  - Local vocabulary size (features): {num_features}")

        # 3. Get Original Ranking
        original_bm25 = BM25Okapi(tokenized_docs)
        original_scores = original_bm25.get_scores(tokenized_query)
        original_ranks = rank_list(np.array(original_scores))

        # 4. Define the prediction wrapper
        # We define a wrapper function instead of passing partial directly
        # to avoid issues with SHAP's internal function inspection.
        def predict_wrapper(mask):
            return text_model_predict_val(
                mask,
                tokenized_documents=tokenized_docs,
                tokenized_query=tokenized_query,
                vocabulary_list=vocabulary_list,
                original_ranks=original_ranks,
                similarity_coefficient=self.rank_similarity_coefficient,
            )

        # 5. Initialize KernelExplainer
        # We use a dummy background (all zeros) because we are doing token masking.
        # In KernelSHAP for text, "background" is usually the empty string (all zeros mask).
        background_data = np.zeros((1, num_features))

        # We pass the wrapper directly to the explainer initialization
        # instead of setting self.explainer.model later.
        self.explainer = shap.KernelExplainer(predict_wrapper, background_data)

        # 6. Run SHAP
        # The 'input' to shap is a vector of 1s (meaning "all words present")
        input_instance = np.ones((1, num_features))

        # Run
        print(f"  - Running KernelSHAP with {self.nsamples} samples...")
        shap_values = self.explainer.shap_values(
            input_instance, nsamples=self.nsamples, silent=True
        )[
            0
        ]  # Shape [1, num_features]

        # 7. Map back to tokens
        # shap_values is an array of shape (num_features,).
        # We map index i -> vocabulary_list[i]

        attribution_dict = {
            vocabulary_list[i]: shap_values[i] for i in range(num_features)
        }

        # Sort by absolute importance
        sorted_attributions = sorted(
            attribution_dict.items(), key=lambda x: abs(x[1]), reverse=True
        )

        return sorted_attributions
