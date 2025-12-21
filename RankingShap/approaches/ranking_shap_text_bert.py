"""
RankingSHAP for Text with BERT Cross-Encoder.

This module extends RankingShapText to work with BERT-based ranking models
instead of BM25. The key difference is that BERT uses the full context of
words rather than bag-of-words, so masking effects may differ.
"""

import numpy as np
import shap
from scipy.stats import kendalltau
from functools import partial
from shap.utils._legacy import convert_to_model

from utils.explanation import AttributionExplanation, SelectionExplanation
from utils.helper_functions import rank_list


def placeholder_predict(array):
    """Placeholder for KernelExplainer initialization."""
    return np.array([0] * len(array))


def new_model_predict_bert_text(
    masks,
    original_model_predict,
    query_features,
    similarity_coefficient=lambda x, y: kendalltau(x, y)[0],
):
    """
    Custom prediction function for BERT Text RankingSHAP.

    This function is called by SHAP to evaluate the effect of masking features.

    Args:
        masks: (n_samples, n_vocab) binary masks from SHAP
        original_model_predict: Function that takes binary matrix and returns scores
        query_features: (n_docs, n_vocab) original binary document matrix
        similarity_coefficient: Metric to compare rankings (default: Kendall's Tau)

    Returns:
        Array of similarity scores for each mask configuration
    """
    # Get original ranking (all features present)
    original_scores = original_model_predict(query_features)
    og_rank = rank_list(original_scores)

    scores = []

    for mask in masks:
        # Apply mask: element-wise AND with the mask
        # If mask[i] = 0, feature i is removed from ALL documents
        masked_query_features = query_features * mask

        # Get new scores from BERT
        new_pred = original_model_predict(masked_query_features)
        new_rank = rank_list(new_pred)

        # Compare rankings
        score = similarity_coefficient(og_rank, new_rank)
        scores.append(score)

    return np.array(scores)


class RankingShapBertText:
    """
    RankingSHAP for BERT-based text ranking models.

    This class provides listwise feature attribution explanations for BERT
    cross-encoder ranking models on text data.

    The key insight is that we use a binary vocabulary representation where
    each feature indicates whether a word is present in a document. By
    masking words across all documents simultaneously, we can determine
    which words are most important for the overall ranking decision.

    Attributes:
        original_model: BERT wrapper with predict() method
        explanation_size: Number of top features to include in selection
        nsamples: Number of SHAP samples (higher = more accurate but slower)
        rank_similarity_coefficient: Function to compare rankings
    """

    def __init__(
        self,
        original_model,
        background_data=None,
        explanation_size=10,
        name="rankingshap_bert_text",
        nsample_permutations="auto",
        rank_similarity_coefficient=lambda x, y: kendalltau(x, y)[0],
    ):
        """
        Initialize the BERT text explainer.

        Args:
            original_model: BERT wrapper with predict() method accepting binary matrix
            background_data: Not strictly needed (0 = word absent)
            explanation_size: Number of top features for selection explanation
            name: Name for logging/saving
            nsample_permutations: SHAP sampling parameter ('auto' or int)
            rank_similarity_coefficient: Ranking comparison function
        """
        self.original_model = original_model
        self.explanation_size = explanation_size
        self.nsamples = nsample_permutations
        self.name = name
        self.rank_similarity_coefficient = rank_similarity_coefficient
        self.explainer = None

    def get_query_explanation(self, query_features, query_id=""):
        """
        Generate feature attributions for a query.

        Args:
            query_features: Binary matrix (n_docs, n_vocab)
            query_id: Identifier for logging

        Returns:
            tuple: (SelectionExplanation, AttributionExplanation)
        """
        num_features = query_features.shape[1]

        # Background: zeros = words absent
        background_zeros = np.zeros((1, num_features))

        # Initialize KernelExplainer
        self.explainer = shap.KernelExplainer(
            placeholder_predict, background_zeros, nsamples=self.nsamples
        )

        # Create partial predict function with fixed query_features
        custom_predict = partial(
            new_model_predict_bert_text,
            original_model_predict=self.original_model,
            query_features=query_features,
            similarity_coefficient=self.rank_similarity_coefficient,
        )

        self.explainer.model = convert_to_model(custom_predict)

        # Explain: start from all words present (ones)
        instance_all_present = np.ones((1, num_features))

        # Run SHAP
        shap_values = self.explainer.shap_values(
            instance_all_present, nsamples=self.nsamples
        )[0]

        # Format results
        exp_dict = {i: shap_values[i] for i in range(len(shap_values))}
        sorted_exp = sorted(exp_dict.items(), key=lambda item: item[1], reverse=True)

        # Create explanation objects
        feature_attributes = AttributionExplanation(
            explanation=sorted_exp, num_features=num_features, query_id=query_id
        )

        feature_selection = SelectionExplanation(
            [sorted_exp[i][0] for i in range(min(self.explanation_size, num_features))],
            num_features=num_features,
            query_id=query_id,
        )

        return feature_selection, feature_attributes


class RankingShapBertTextWeighted:
    """
    RankingSHAP for BERT with weighted rank difference objective.

    This version uses a weighted objective that emphasizes the top of the ranking,
    similar to NDCG weighting.
    """

    def __init__(
        self,
        original_model,
        background_data=None,
        explanation_size=10,
        name="rankingshap_bert_text_weighted",
        nsample_permutations="auto",
    ):
        self.original_model = original_model
        self.explanation_size = explanation_size
        self.nsamples = nsample_permutations
        self.name = name
        self.explainer = None

        # Weighted rank difference (emphasizes top positions)
        def weighted_rank_diff(x, y):
            """
            Weighted rank difference that penalizes changes at top positions more.
            """
            eps = np.finfo(float).eps
            return -np.sum((np.array(y) - np.array(x)) / np.log2(np.array(x) + eps + 1))

        self.rank_similarity_coefficient = weighted_rank_diff

    def get_query_explanation(self, query_features, query_id=""):
        """Generate feature attributions with weighted objective."""
        num_features = query_features.shape[1]

        background_zeros = np.zeros((1, num_features))

        self.explainer = shap.KernelExplainer(
            placeholder_predict, background_zeros, nsamples=self.nsamples
        )

        custom_predict = partial(
            new_model_predict_bert_text,
            original_model_predict=self.original_model,
            query_features=query_features,
            similarity_coefficient=self.rank_similarity_coefficient,
        )

        self.explainer.model = convert_to_model(custom_predict)

        instance_all_present = np.ones((1, num_features))

        shap_values = self.explainer.shap_values(
            instance_all_present, nsamples=self.nsamples
        )[0]

        exp_dict = {i: shap_values[i] for i in range(len(shap_values))}
        sorted_exp = sorted(exp_dict.items(), key=lambda item: item[1], reverse=True)

        feature_attributes = AttributionExplanation(
            explanation=sorted_exp, num_features=num_features, query_id=query_id
        )

        feature_selection = SelectionExplanation(
            [sorted_exp[i][0] for i in range(min(self.explanation_size, num_features))],
            num_features=num_features,
            query_id=query_id,
        )

        return feature_selection, feature_attributes
