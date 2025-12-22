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


class RankingShapAdaptive:
    """
    RankingSHAP with Query-Adaptive Sampling for faster computation.

    Idea (in simple words):
    - Easy queries (few documents) need fewer SHAP samples.
    - Hard queries (many documents) need more SHAP samples.
    - We set the number of samples using a sqrt rule: samples ≈ base * sqrt(#docs).

    Concretely:
    - Let base = adaptive_min_samples (default 25).
    - For a query with n documents we use:
        samples = min(base * sqrt(n_docs), adaptive_max_samples)

      Examples:
        n_docs = 4   -> ~  50 samples
        n_docs = 9   -> ~  75 samples
        n_docs = 16  -> ~ 100 samples
        n_docs = 25  -> ~ 125 samples
        n_docs = 36  -> ~ 150 samples

    This is more scientific than fixed thresholds:
    - It scales smoothly with query size.
    - It avoids giving a 20-doc query and a 40-doc query the same sample size.
    - It is grounded in Monte-Carlo theory (error ~ 1/sqrt(n)).
    """
    
    def __init__(
        self,
        permutation_sampler,
        background_data,
        original_model,
        explanation_size=3,
        name="rankingshapK_adaptive",
        nsample_permutations="auto",
        rank_similarity_coefficient=lambda x, y: kendalltau(x, y)[0],
        adaptive_min_samples=25,  # Base factor for sqrt(n_docs) rule
        adaptive_max_samples=300,  # Hard upper cap on samples
    ):
        assert permutation_sampler in ["kernel", "sampling"]
        self.permutation_sampler = permutation_sampler
        self.background_data = background_data
        self.original_model = original_model
        self.explanation_size = explanation_size
        self.adaptive_min_samples = adaptive_min_samples
        self.adaptive_max_samples = adaptive_max_samples

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
        Compute the number of SHAP samples for this query.

        We use a sqrt rule:
            samples = base * sqrt(n_docs)
        where:
            - base = adaptive_min_samples (default 25)
            - n_docs = number of documents for this query

        Rationale:
            - More documents -> more complex ranking -> more samples.
            - But complexity does not grow linearly, so we use sqrt(n_docs).
        """
        n_docs = max(len(query_features), 1)  # avoid sqrt(0)
        base = self.adaptive_min_samples

        samples = int(base * np.sqrt(n_docs))
        return min(samples, self.adaptive_max_samples)

    def get_query_explanation(self, query_features, query_id=""):
        # Set the query dependent model predict function
        self.explainer.model = convert_to_model(
            partial(self.new_model_predict, query_features=query_features)
        )

        vector_of_nones = np.array([np.full(self.feature_shape, None)])

        # Adaptive sampling: determine sample size based on query
        adaptive_samples = self._adaptive_sample_size(query_features)
        
        print(f"[RankingShapAdaptive] Query {query_id}: Using adaptive sampling with {adaptive_samples} samples (query has {len(query_features)} documents)", flush=True)
        
        # Use adaptive sample size directly
        # The adaptive approach is query-adaptive (fewer samples for simpler queries)
        # This is faster than always using "auto" or large fixed samples
        if self.permutation_sampler == "kernel":
            exp = self.explainer.shap_values(vector_of_nones, nsamples=adaptive_samples)[0]
        else:
            exp = self.explainer(vector_of_nones, nsamples=adaptive_samples).values[0]
        
        print(f"[RankingShapAdaptive] Query {query_id}: Completed with {adaptive_samples} samples", flush=True)

        exp_dict = {
            i + 1: exp[i] for i in range(len(exp))
        }  # We start numbering the features with 1.

        exp_dict = sorted(exp_dict.items(), key=lambda item: item[1], reverse=True)

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

