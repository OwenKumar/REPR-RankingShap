import pandas as pd
from evaluation.evaluate_feature_attribution import eval_feature_attribution
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

parser.add_argument(
    "--approach",
    type=str,
    default="all",
    help="Choose to run a specific approach.",
)

parser.add_argument(
    "--fold",
    required=False,
    type=int,
    default=1,
    help="Which fold of the data to use.",
)


args = parser.parse_args()
print(args, flush=True)

dataset = args.dataset
fold = args.fold
# ---- arguments for metrics

# We assume that the model has been trained and saved in a model file
model_file = args.model_file
# Include fold in model file name if not already present
if f"_fold{fold}" not in model_file:
    model_file_with_fold = f"{model_file}_fold{fold}"
else:
    model_file_with_fold = model_file

model = lightgbm.Booster(
    model_file=(str((Path("results/model_files/") / model_file_with_fold).absolute()))
)

# Load fold-specific background data
background_data = BackgroundData(
    np.load(
        Path(f"results/background_data_files/train_background_data_{dataset}_fold{fold}.npy")
    ),
    summarization_type=None,
)

# Get train, eval_data
data_directory = Path("data/" + dataset + f"/Fold{fold}/")
# train_data = get_data(data_file=data_directory / "train.txt")
test_data = get_data(data_file=data_directory / "test.txt")
# eval_data = get_data(data_file=data_directory / "vali.txt")



# -------- end arguments for metrics


file_name_ground_truth = args.file_name_ground_truth
# Include fold in output folder path
path_to_attribution_folder = Path(f"results/results_{dataset}_fold{fold}/feature_attributes/")

path_to_ground_truth_attributes = path_to_attribution_folder / file_name_ground_truth


approaches = [
    "rankingshapK",
    "rankingshapW",
    "rankingshapK_adaptive",
    "greedy_iter",
    "greedy_iter_full",
    "pointwise_lime",
    "pointwise_shap",
    "random",
    "rankinglime",
    #"rankingsharp",
]

if args.approach in approaches:
    approaches = [args.approach]

if args.test:
    approaches = [a + "_test" for a in approaches]

eval_df = []
processed_approaches = []
skipped_approaches = []


for approach in approaches:
    path_to_attribute_values = path_to_attribution_folder / (
        approach + "_eval" + ".csv"
    )

    # Check if file exists before processing
    if not path_to_attribute_values.exists():
        print(f"Warning: File not found: {path_to_attribute_values}, skipping approach '{approach}'", flush=True)
        skipped_approaches.append(approach)
        continue

    try:
        mean_attribute_evaluation = eval_feature_attribution(
            attributes_to_evaluate=path_to_attribute_values,
            model = model.predict,
            eval_data=test_data,
            background = background_data.background_summary,
            ground_truth_file_path=path_to_ground_truth_attributes,
        )

        # mean_attribute_evaluation = attribution_evaluation_per_query.mean()
        # mean_attribute_evaluation["approach"] = approach
        

        df = pd.DataFrame({'Pre_ken': mean_attribute_evaluation[:, 0], 'Del_ken': mean_attribute_evaluation[:, 1], \
                           'Pre_exp': mean_attribute_evaluation[:, 2], 'Del_exp': mean_attribute_evaluation[:, 3]})

        df.insert(0, "approach", ["{}@{}".format(approach, i) for i in [1,3,5,7,10] ])

        eval_df.append(df)
        processed_approaches.append(approach)
        print(f"Successfully processed approach: '{approach}'", flush=True)
    
    except Exception as e:
        print(f"Error processing approach '{approach}': {e}", flush=True)
        print(f"Skipping approach '{approach}' due to error", flush=True)
        skipped_approaches.append(approach)
        continue

# Print summary statistics
print("\n" + "="*60, flush=True)
print("Evaluation Summary:", flush=True)
print(f"  Total approaches in list: {len(approaches)}", flush=True)
print(f"  Successfully processed: {len(processed_approaches)}", flush=True)
print(f"  Skipped (missing files or errors): {len(skipped_approaches)}", flush=True)
if processed_approaches:
    print(f"  Processed approaches: {', '.join(processed_approaches)}", flush=True)
if skipped_approaches:
    print(f"  Skipped approaches: {', '.join(skipped_approaches)}", flush=True)
print("="*60 + "\n", flush=True)

# Handle empty results gracefully
if len(eval_df) == 0:
    print("ERROR: No attribution files were found or processed successfully.", flush=True)
    print("Please ensure that at least one attribution file exists in:", flush=True)
    print(f"  {path_to_attribution_folder}", flush=True)
    print("\nExpected file format: <approach_name>_eval.csv", flush=True)
    print("Example: rankingshapK_adaptive_eval.csv", flush=True)
    exit(1)

mean_attribute_evaluation = pd.concat(eval_df)

mean_attribute_evaluation = mean_attribute_evaluation.set_index(["approach"])

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
