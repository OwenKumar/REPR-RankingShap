import pandas as pd
from utils.helper_functions import rank_based_on_column_per_query
from evaluation.evaluate_explanations import calculate_validity_completeness
from scipy.stats import kendalltau
from utils.helper_functions import get_queryids_as_list, get_documents_per_query
import numpy as np


def get_estimated_ground_truth_feature_importance(file):
    feature_attribute_df = pd.read_csv(file)
    return feature_attribute_df[
        ["query_number", "feature_number", "attribution_value", "std"]
    ]  # TODO why do we use the std here?


def eval_feature_attribution(
    attributes_to_evaluate,
    model,
    background,
    eval_data,
    ground_truth_file_path="../results/feature_attributes/"
    + "attribution_values_for_evaluation.csv",
):
    attribution_df = pd.read_csv(attributes_to_evaluate, index_col=0)
    ground_truth_attributes = get_estimated_ground_truth_feature_importance(
        ground_truth_file_path
    )
    results_df = pd.DataFrame()

    # prepare dataframe by adding some helper columns.
    attribution_df = attribution_df.merge(
        ground_truth_attributes,
        on=["query_number", "feature_number"],
        suffixes=("_exp", "_gt"),
    )
    attribution_df["abs_difference"] = (
        (
            attribution_df["attribution_value_exp"]
            - attribution_df["attribution_value_gt"]
        )
        ** 2
    ).abs()
    attribution_df["squared_difference"] = attribution_df["abs_difference"] ** 2
    attribution_df = attribution_df.reset_index()
    attribution_df = rank_based_on_column_per_query(
        attribution_df,
        name_column_to_rank="attribution_value_exp",
        new_column_name="exp_ranked",
        biggest_first=True,
    )
    attribution_df = rank_based_on_column_per_query(
        attribution_df,
        name_column_to_rank="attribution_value_gt",
        new_column_name="gt_ranked",
        biggest_first=True,
    )
    attribution_df["abs_rank_difference"] = (
        attribution_df["exp_ranked"] - attribution_df["gt_ranked"]
    ).abs()

    def metrics_on_subset(sub_df, r_df, suffix_metric=""):
        r_df["L1_norm" + suffix_metric] = sub_df.groupby(["query_number"])[
            "abs_difference"
        ].mean()
        r_df["L2_norm" + suffix_metric] = sub_df.groupby(["query_number"])[
            "squared_difference"
        ].mean()
        r_df["spearmans_footrule_metric" + suffix_metric] = sub_df.groupby(
            ["query_number"]
        )["abs_rank_difference"].mean()
        return r_df

    # results_df = metrics_on_subset(attribution_df, results_df, suffix_metric="")


    EX, _, Eqids = eval_data
    queries = get_queryids_as_list(Eqids)
    

    print("Evaluating all ", len(queries), " queries")

    # keep track of query start position in test set
    list_trckr = 0
    i = 1
    qid_count_list = get_documents_per_query(Eqids)

    results = [[], [], [], [], []]
    k_s = [1, 3, 5, 7, 10]

    # Defining the metrics by which to evaluate
    exposure_diff = lambda x, y: np.sum(np.abs(1/(np.log2(np.array(x) + 2)) - 1/(np.log2(np.array(y) + 2))))
    kendall_tau = lambda x, y: kendalltau(x, y)[0]

    # select all query document pairs for a certain query and calculate the validity and completeness
    for query in queries:
        query_len = qid_count_list[query]
        # Get the query to pass 
        current_query = EX[list_trckr : (list_trckr + query_len)]

        # top-k_metrics
        for i in range(len(k_s)):
            top_k = attribution_df[attribution_df.exp_ranked <= k_s[i]]['feature_number'].values # Take the top k explanation features


            val_kendall, comp_kendall = calculate_validity_completeness(
                        current_query, 
                        model,
                        top_k, 
                        background_data=background,
                        mixed_type_input=False, 
                        rank_similarity_coefficient=kendall_tau,
                        )
            
            print(val_kendall)
            val_expo, comp_expo = calculate_validity_completeness(
                        current_query, 
                        model, 
                        top_k, 
                        background_data=background, 
                        mixed_type_input=False, 
                        rank_similarity_coefficient=exposure_diff,
                        )

            results[i].append([val_kendall, comp_kendall, val_expo, comp_expo])

        # Update list_tracker
        list_trckr += query_len

    
    results = np.mean(np.asarray(results).transpose((0, 2, 1)), axis=2)
    print(results)
    return results