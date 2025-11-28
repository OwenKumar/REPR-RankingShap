import numpy as np
import shap
from scipy.stats import kendalltau
from functools import partial
from shap.utils._legacy import convert_to_model
from utils.explanation import AttributionExplanation, SelectionExplanation
from utils.helper_functions import rank_list


def placeholder_predict(array):
    """
    Placeholder initialization function for KernelExplainer.
    """
    return np.array([0] * len(array))


def new_model_predict_text(
    masks,
    original_model_predict,
    query_features,
    similarity_coefficient=lambda x, y: kendalltau(x, y)[0],
):
    """
    Custom prediction function for Text RankingSHAP.

    Args:
        masks (np.array): Shape (n_samples, n_vocab). Binary masks from SHAP (1=keep, 0=drop).
        original_model_predict (func): Function that takes binary matrix and returns scores.
        query_features (np.array): Shape (n_docs, n_vocab). The original binary document matrix.
        similarity_coefficient (func): Metric to compare rankings (e.g., Kendall's Tau).
    """
    # 1. Get the original ranking (with all features active)
    # For the original ranking, we assume the mask is all 1s
    original_scores = original_model_predict(query_features)
    og_rank = rank_list(original_scores)

    scores = []

    # 2. Iterate through each SHAP perturbation (mask)
    for mask in masks:
        # Apply the mask to the entire document set.
        # If mask[i] is 0, feature i is removed from ALL documents.
        # query_features: (n_docs, n_features)
        # mask: (n_features,)
        masked_query_features = query_features * mask

        # Get new scores from BM25 wrapper
        new_pred = original_model_predict(masked_query_features)
        new_rank = rank_list(new_pred)

        # Compare new ranking to original ranking
        score = similarity_coefficient(og_rank, new_rank)
        scores.append(score)

    return np.array(scores)


class RankingShapText:
    def __init__(
        self,
        original_model,
        background_data=None,  # Not strictly needed for text if we assume 0=absent
        explanation_size=10,
        name="rankingshap_text",
        nsample_permutations="auto",
        rank_similarity_coefficient=lambda x, y: kendalltau(x, y)[0],
    ):
        self.original_model = original_model
        self.explanation_size = explanation_size
        self.nsamples = nsample_permutations
        self.name = name
        self.rank_similarity_coefficient = rank_similarity_coefficient

        # We initialize the explainer later per query because num_features changes dynamically
        self.explainer = None

    def get_query_explanation(self, query_features, query_id=""):
        """
        Args:
            query_features (np.array): Binary Matrix (n_docs, n_vocab)
            query_id (str): ID for logging
        """
        num_features = query_features.shape[1]

        # 1. Define Background (Zeros = Words Absent)
        # SHAP needs a background to "mask" towards. For text, absence is 0.
        background_zeros = np.zeros((1, num_features))

        # 2. Initialize KernelExplainer dynamically for this vocabulary size
        self.explainer = shap.KernelExplainer(
            placeholder_predict, background_zeros, nsamples=self.nsamples
        )

        # 3. Define the partial predict function for this specific query
        # This freezes the 'query_features' (documents) into the function
        custom_predict = partial(
            new_model_predict_text,
            original_model_predict=self.original_model,
            query_features=query_features,
            similarity_coefficient=self.rank_similarity_coefficient,
        )

        self.explainer.model = convert_to_model(custom_predict)

        # 4. Run SHAP
        # We explain an instance of "All Ones" (meaning all words are initially present)
        instance_all_present = np.ones((1, num_features))

        # The explainer will toggle 1s to 0s (background) to measure impact
        shap_values = self.explainer.shap_values(
            instance_all_present, nsamples=self.nsamples
        )[
            0
        ]  # shap_values returns list, we want the first (and only) output dimension

        # 5. Format Results
        # Map index back to attribution score
        exp_dict = {i: shap_values[i] for i in range(len(shap_values))}

        # Sort by importance
        sorted_exp = sorted(exp_dict.items(), key=lambda item: item[1], reverse=True)

        # 6. Create Return Objects
        feature_attributes = AttributionExplanation(
            explanation=sorted_exp, num_features=num_features, query_id=query_id
        )

        feature_selection = SelectionExplanation(
            [sorted_exp[i][0] for i in range(min(self.explanation_size, num_features))],
            num_features=num_features,
            query_id=query_id,
        )

        return feature_selection, feature_attributes
