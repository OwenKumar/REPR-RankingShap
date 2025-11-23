import pandas as pd
from evaluation.evaluate2_feature_attribution import eval_feature_attribution
from pathlib import Path

import argparse
import lightgbm
import numpy as np
from utils.background_data import BackgroundData
from utils.helper_functions import get_data

parser = argparse.ArgumentParser(description="Your script description")

parser.add_argument(
    "--dataset",
    type=str,
    required=True,
    choices=["MSLR-WEB10K", "MQ2008"],
    help="The dataset to use MQ2008 or MSLR-WEB10K",
)
parser.add_argument(
    "--file_name_ground_truth",
    required=True,
    type=str,
    help="File name of truth attribution values",
)
parser.add_argument(
    "--test", action="store_true", help="If true uses test files for evaluation"
)

parser.add_argument(
    "--model_file",
    required=True,
    type=str,
    help="Path to the model file of the model that we want to approximate the feature importance for",
)


args = parser.parse_args()
print(args, flush=True)

dataset = args.dataset

# ---- arguments for metrics

# We assume that the model has been trained and saved in a model file
model_file = args.model_file

model = lightgbm.Booster(
    model_file=(str((Path("results/model_files/") / model_file).absolute()))
)

background_data = BackgroundData(
    np.load(
        Path("results/background_data_files/train_background_data_" + dataset + ".npy")
    ),
    summarization_type=None,
)

# Get train, eval_data
data_directory = Path("data/" + dataset + "/Fold1/")
# train_data = get_data(data_file=data_directory / "train.txt")
test_data = get_data(data_file=data_directory / "test.txt")
# eval_data = get_data(data_file=data_directory / "vali.txt")



# -------- end arguments for metrics


file_name_ground_truth = args.file_name_ground_truth
path_to_attribution_folder = Path("results/results_" + dataset + "/feature_attributes/")

path_to_ground_truth_attributes = path_to_attribution_folder / file_name_ground_truth


approaches = [
    "rankingshapK",
    # "rankingshapW",
    # "greedy_iter",
    # "greedy_iter_full",
    # "pointwise_lime",
    # "pointwise_shap",
    # "random",
    # "rankinglime",
]
if args.test:
    approaches = [a + "_test" for a in approaches]

eval_df = []


for approach in approaches:
    path_to_attribute_values = path_to_attribution_folder / (
        approach + "_eval" + ".csv"
    )

    mean_attribute_evaluation = eval_feature_attribution(
        attributes_to_evaluate=path_to_attribute_values,
        model = model.predict,
        eval_data=test_data,
        background = background_data.background_summary,
        ground_truth_file_path=path_to_ground_truth_attributes,
    )

    # mean_attribute_evaluation = attribution_evaluation_per_query.mean()
    # mean_attribute_evaluation["approach"] = approach
    

    df = pd.DataFrame({'Pre_ken': mean_attribute_evaluation[:, 0], 'Del_ken': mean_attribute_evaluation[:, 1]})
    print(df)
    df.insert(0, "approach", [(approach + "@" + i) for i in [1,3,5,7,10] ])

    print(df)

    eval_df.append(df)

# TODO update these fields vv

mean_attribute_evaluation = pd.concat(eval_df)

# mean_attribute_evaluation = mean_attribute_evaluation.set_index(["approach"])

# evaluation_for_table = mean_attribute_evaluation[
#     [
#         "spearmans_footrule_metric",
#         "spearmans_footrule_metric@3",
#         "spearmans_footrule_metric@10",
#         "L1_norm",
#         "L1_norm@3",
#         "L1_norm@10",
#     ]
# ]
# evaluation_for_table = evaluation_for_table.rename(
#     {
#         "spearmans_footrule_metric": "order",
#         "spearmans_footrule_metric@3": "order@3",
#         "spearmans_footrule_metric@10": "order@10",
#         "L1_norm": "valdis",
#         "L1_norm@3": "valdis@3",
#         "L1_norm@10": "valdis@10",
#     },
#     axis=1,
# )

# evaluation_for_table = evaluation_for_table.round(
#     {
#         "order": 1,
#         "order@3": 1,
#         "order@10": 1,
#         "valdis": 4,
#         "valdis@3": 4,
#         "valdis@10": 4,
#     }
# )

print(mean_attribute_evaluation)
