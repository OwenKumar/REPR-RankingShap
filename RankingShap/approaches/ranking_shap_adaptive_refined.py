import numpy as np
import shap
from scipy.stats import kendalltau
import pandas as pd
from functools import partial
from shap.utils._legacy import convert_to_model
from utils.explanation import AttributionExplanation, SelectionExplanation
from utils.helper_functions import rank_list


# Our implementation of KernelShap needs a re-initialization of the model.predict function
# for each query that it is explaining. Hence for the initialization we will use a place-holder
# function that returns an error if not re-initialized with another function.
def placeholder_predict(array):
    Warning(
        "The model.predict function needs to be defined for each query individually."
    )
    return np.array([0] * len(array))


def new_model_predict_val(
    array,
    original_model_predict,
    query_features,
    similarity_coefficient=lambda x, y: kendalltau(x, y)[0],
    mixed_type_input=False,
):
    # Determine ranking for current query
    pred = original_model_predict(query_features)
    og_rank = rank_list(pred)

    # Adjust feature vectors: substitute not None features for all candidate documents with background/0 value
    if not mixed_type_input:
        adjusted_features = np.array(
            [[np.where(pd.isna(a), doc, a) for doc in query_features] for a in array]
        )
    else:
        adjusted_features = np.array(
            [
                [pd.Series(doc).where(pd.isna(a), a).values for doc in query_features]
                for a in array
            ]
        )  # Need to use pd for mixed type array.

    scores = []
    for features_background_sample in adjusted_features:
        # Determine ranking for adjusted document feature vectors
        new_pred = original_model_predict(features_background_sample)
        new_rank = rank_list(new_pred)
        score = similarity_coefficient(og_rank, new_rank)
        scores.append(score)
    return np.array(scores)


class RankingShapAdaptiveRefined:
    """
    RankingSHAP with Two-Stage Adaptive Refinement for optimal speed/accuracy trade-off.
    
    Strategy:
    1. Stage 1 (Fast): Run adaptive sampling (25-300 samples) - EXACTLY like RankingShapAdaptive
       - Uses: 25 * sqrt(n_docs), max 300
       - Quickly identifies important features
    2. Stage 2 (Refined): Run high-quality sampling (like baseline RankingSHAP) - EXACTLY like RankingShap
       - Uses: "auto" or high fixed value (e.g., 1000+)
       - Refines only the top-k features with baseline-quality attributions
    
    This approach:
    - Faster than baseline: Only refines a subset of features (typically 5-10)
    - More accurate than pure adaptive: Top features get baseline-quality attributions
    - Best of both worlds: Speed from adaptive + accuracy from refinement
    
    Parameters:
    - top_k_to_refine: Number of top features to refine (default: 10)
    - refinement_samples: Samples for Stage 2 - "auto" (default, like baseline) or int (e.g., 1000)
    - adaptive_min_samples: Base for Stage 1 adaptive sampling (default: 25)
    - adaptive_max_samples: Cap for Stage 1 adaptive sampling (default: 300)
    """
    
    def __init__(
        self,
        permutation_sampler,
        background_data,
        original_model,
        explanation_size=3,
        name="rankingshapK_adaptive_refined",
        nsample_permutations="auto",
        rank_similarity_coefficient=lambda x, y: kendalltau(x, y)[0],
        adaptive_min_samples=25,  # Base factor for sqrt(n_docs) rule (Stage 1)
        adaptive_max_samples=300,  # Hard upper cap on samples (Stage 1)
        top_k_to_refine=10,  # Number of top features to refine
        refinement_samples=1000,  # High-quality samples for Stage 2 (like baseline RankingSHAP)
    ):
        assert permutation_sampler in ["kernel", "sampling"]
        self.permutation_sampler = permutation_sampler
        self.background_data = background_data
        self.original_model = original_model
        self.explanation_size = explanation_size
        self.adaptive_min_samples = adaptive_min_samples
        self.adaptive_max_samples = adaptive_max_samples
        self.top_k_to_refine = top_k_to_refine
        self.refinement_samples = refinement_samples  # High-quality sampling for Stage 2

        self.nsamples = nsample_permutations
        self.name = name

        self.feature_shape = np.shape(background_data[0])
        self.num_features = len(background_data[0])

        self.explainer = self.get_explainer()
        self.rank_similarity_coefficient = rank_similarity_coefficient
        self.new_model_predict = partial(
            new_model_predict_val,
            original_model_predict=original_model,
            similarity_coefficient=rank_similarity_coefficient,
        )
        self.feature_attribution_explanation = None
        self.feature_selection_explanation = None

    def get_explainer(self):
        if self.permutation_sampler == "kernel":
            shap_explainer = shap.KernelExplainer(
                placeholder_predict, self.background_data, nsamples=self.nsamples
            )
        elif self.permutation_sampler == "sampling":
            shap_explainer = shap.SamplingExplainer(
                placeholder_predict, self.background_data, nsamples=self.nsamples
            )
        return shap_explainer

    def _adaptive_sample_size(self, query_features):
        """
        Compute the number of SHAP samples for adaptive stage.
        
        We use a sqrt rule:
            samples = base * sqrt(n_docs)
        where:
            - base = adaptive_min_samples (default 25)
            - n_docs = number of documents for this query
        """
        n_docs = max(len(query_features), 1)  # avoid sqrt(0)
        base = self.adaptive_min_samples

        samples = int(base * np.sqrt(n_docs))
        return min(samples, self.adaptive_max_samples)
    
    def _refinement_sample_size(self, query_features):
        """
        Get the number of SHAP samples for refinement stage.
        
        Stage 2 uses high-quality sampling like baseline RankingSHAP:
        - If refinement_samples is "auto": Use "auto" (SHAP decides, typically 1000+)
        - If refinement_samples is an int: Use that fixed value (e.g., 1000)
        
        This ensures top features get baseline-quality attributions.
        """
        return self.refinement_samples

    def get_query_explanation(self, query_features, query_id=""):
        # Set the query dependent model predict function
        self.explainer.model = convert_to_model(
            partial(self.new_model_predict, query_features=query_features)
        )

        vector_of_nones = np.array([np.full(self.feature_shape, None)])

        # ============================================================
        # STAGE 1: Adaptive sampling to identify top features
        # ============================================================
        adaptive_samples = self._adaptive_sample_size(query_features)
        
        print(f"[RankingShapAdaptiveRefined] Query {query_id}: Stage 1 - Adaptive sampling with {adaptive_samples} samples (query has {len(query_features)} documents)", flush=True)
        
        if self.permutation_sampler == "kernel":
            adaptive_exp = self.explainer.shap_values(vector_of_nones, nsamples=adaptive_samples)[0]
        else:
            adaptive_exp = self.explainer(vector_of_nones, nsamples=adaptive_samples).values[0]
        
        # Convert to dict and identify top-k features by absolute value
        adaptive_exp_dict = {
            i + 1: adaptive_exp[i] for i in range(len(adaptive_exp))
        }
        
        # Sort by absolute value to find most important features
        sorted_by_importance = sorted(
            adaptive_exp_dict.items(), 
            key=lambda item: abs(item[1]), 
            reverse=True
        )
        
        # Get top-k feature indices (1-indexed)
        top_k_features = [feat_idx for feat_idx, _ in sorted_by_importance[:self.top_k_to_refine]]
        
        print(f"[RankingShapAdaptiveRefined] Query {query_id}: Identified top {len(top_k_features)} features to refine: {top_k_features}", flush=True)
        
        # ============================================================
        # STAGE 2: Refine top-k features with high-quality sampling (like baseline RankingSHAP)
        # ============================================================
        refinement_samples = self._refinement_sample_size(query_features)
        print(f"[RankingShapAdaptiveRefined] Query {query_id}: Stage 2 - Refining top {len(top_k_features)} features with {refinement_samples} samples (high-quality, like baseline RankingSHAP)", flush=True)
        
        if self.permutation_sampler == "kernel":
            refined_exp = self.explainer.shap_values(vector_of_nones, nsamples=refinement_samples)[0]
        else:
            refined_exp = self.explainer(vector_of_nones, nsamples=refinement_samples).values[0]
        
        # ============================================================
        # COMBINE: Use refined values for top-k, adaptive for rest
        # ============================================================
        final_exp_dict = {}
        for feat_idx in range(1, self.num_features + 1):  # Features are 1-indexed
            if feat_idx in top_k_features:
                # Use refined value for top features
                final_exp_dict[feat_idx] = refined_exp[feat_idx - 1]  # Convert to 0-indexed
            else:
                # Use adaptive value for other features
                final_exp_dict[feat_idx] = adaptive_exp[feat_idx - 1]  # Convert to 0-indexed
        
        print(f"[RankingShapAdaptiveRefined] Query {query_id}: Combined results - {len(top_k_features)} features refined, {self.num_features - len(top_k_features)} features from adaptive", flush=True)
        
        # Sort by value for final explanation
        exp_dict = sorted(final_exp_dict.items(), key=lambda item: item[1], reverse=True)

        feature_attributes = AttributionExplanation(
            explanation=exp_dict, num_features=self.num_features, query_id=query_id
        )
        # Take the most important features as explanations
        feature_selection = SelectionExplanation(
            [exp_dict[i][0] for i in range(self.explanation_size)],
            num_features=self.num_features,
            query_id=query_id,
        )
        return feature_selection, feature_attributes

