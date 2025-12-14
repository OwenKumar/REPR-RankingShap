from utils.get_explanations import calculate_all_query_explanations
from utils.helper_functions import get_data, get_queryids_as_list
import lightgbm
import numpy as np
from scipy.stats import kendalltau
from utils.background_data import BackgroundData
from approaches.ranking_shap import RankingShap
from approaches.ranking_shap_adaptive import RankingShapAdaptive
from approaches.ranking_shap_adaptive_refined import RankingShapAdaptiveRefined
from approaches.ranking_lime import RankingLIME
from approaches.ranking_sharp import RankingSharp
from approaches.greedy_listwise import GreedyListwise
from approaches.pointwise_lime import AggregatedLime
from approaches.pointwise_shap import AggregatedShap
from approaches.random_explainer import RandomExplainer
from pathlib import Path
import time
import json
import numpy as np

import argparse

parser = argparse.ArgumentParser(description="Runs different explanation approaches")

parser.add_argument(
    "--model_file",
    required=True,
    type=str,
    help="Path to the model file of the model that we want to approximate the feature importance for",
)
parser.add_argument(
    "--dataset",
    type=str,
    required=True,
    choices=["MSLR-WEB10K", "MQ2008"],
    help="The dataset to use MQ2008 or MSLR-WEB10K",
)
parser.add_argument(
    "--experiment_iteration",
    required=True,
    type=int,
    help="Enables us to run the same experiment several times",
)
parser.add_argument("--test", action="store_true", help="If true runs only one query")

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
experiment_iteration = args.experiment_iteration
test = args.test

# We assume that the model has been trained and saved in a model file
model_file = args.model_file

model = lightgbm.Booster(
    model_file=(str((Path("results/model_files/") / model_file).absolute()))
)


explanation_size = 5


progress = False
if test:
    num_queries_eval = 1
else:
    num_queries_eval = None

# Get train, eval_data
data_directory = Path("data/" + dataset + f"/Fold{fold}/")
train_data = get_data(data_file=data_directory / "train.txt")
test_data = get_data(data_file=data_directory / "test.txt")
eval_data = get_data(data_file=data_directory / "vali.txt")

path_to_attribution_folder = Path("results/results_" + dataset + "/feature_attributes/")

num_features = len(test_data[0][0])

background_data = BackgroundData(
    np.load(
        Path("results/background_data_files/train_background_data_" + dataset + ".npy")
    ),
    summarization_type=None,
)

rank_similarity_coefficient = lambda x, y: kendalltau(x, y)[0]
weighted_rank_similarity_coefficient = lambda x, y: -np.sum((y - x)/(np.log2(x + np.finfo(float).eps)))

# Define all the explainers
ranking_shapK_explainer = RankingShap(
    permutation_sampler="kernel",
    background_data=background_data.background_summary,
    original_model=model.predict,
    explanation_size=explanation_size,
    name="rankingshapK",
    rank_similarity_coefficient=rank_similarity_coefficient,
)

ranking_shapW_explainer = RankingShap(
    permutation_sampler="kernel",
    background_data=background_data.background_summary,
    original_model=model.predict,
    explanation_size=explanation_size,
    name="rankingshapW",
    rank_similarity_coefficient=weighted_rank_similarity_coefficient,
)


greedy_explainer_0_iter = GreedyListwise(
    background_data=background_data.background_summary,
    model=model.predict,
    explanation_size=explanation_size,
    name="greedy_iter",
    feature_attribution_method="iter",
)


greedy_explainer_0_full = GreedyListwise(
    background_data=background_data.background_summary,
    model=model.predict,
    explanation_size=num_features,
    name="greedy_iter_full",
    feature_attribution_method="iter",
)


aggregated_lime_explainer = AggregatedLime(
    background_data=background_data.background_summary,
    model=model.predict,
    explanation_size=explanation_size,
    name="pointwise_lime",
    aggregate_over_top=5,
)

aggregated_shap_explainer = AggregatedShap(
    background_data=background_data.background_summary,
    model=model.predict,
    explanation_size=explanation_size,
    name="pointwise_shap",
    aggregate_over_top=5,
    nsamples=2**10,
)

random_explainer = RandomExplainer(
    explanation_size=explanation_size, name="random", num_features=num_features
)

ranking_lime_explainer = RankingLIME(
    background_data=background_data.background_summary,
    original_model=model.predict,
    explanation_size=explanation_size,
    name="rankinglime",
    rank_similarity_coefficient=rank_similarity_coefficient,
    use_entry=0,
    individual_masking=True,
)

# ranking_sharp_explainer = RankingSharp(
#     background_data=background_data.background_summary,
#     original_model=model.predict,
#     explanation_size=explanation_size,
#     name="rankingsharp",
#     rank_similarity_coefficient=rank_similarity_coefficient,
# )

explainers = [
    random_explainer,
    aggregated_shap_explainer,
    aggregated_lime_explainer,
    ranking_shapK_explainer,
    ranking_shapW_explainer,
    greedy_explainer_0_iter,
    ranking_lime_explainer,
    # ranking_sharp_explainer,
]

explainers = []

# Add adaptive RankingSHAP for MQ2008 testing
# Uses sqrt-based sampling: samples = base * sqrt(n_docs)
ranking_shapK_adaptive_explainer = RankingShapAdaptive(
    permutation_sampler="kernel",
    background_data=background_data.background_summary,
    original_model=model.predict,
    explanation_size=explanation_size,
    name="rankingshapK_adaptive",
    rank_similarity_coefficient=rank_similarity_coefficient,
    adaptive_min_samples=25,  # Base factor for sqrt(n_docs) rule
    adaptive_max_samples=300,  # Hard upper cap on samples
)
explainers.append(ranking_shapK_adaptive_explainer)

# Add two-stage adaptive refined RankingSHAP
# Stage 1: Adaptive sampling to identify top features
# Stage 2: Refine top-k features with high-quality sampling
ranking_shapK_adaptive_refined_explainer = RankingShapAdaptiveRefined(
    permutation_sampler="kernel",
    background_data=background_data.background_summary,
    original_model=model.predict,
    explanation_size=explanation_size,
    name="rankingshapK_adaptive_refined",
    rank_similarity_coefficient=rank_similarity_coefficient,
    adaptive_min_samples=25,  # Base factor for sqrt(n_docs) rule (Stage 1 - like adaptive)
    adaptive_max_samples=300,  # Hard upper cap on samples (Stage 1 - like adaptive)
    top_k_to_refine=10,  # Number of top features to refine
    refinement_samples="auto",  # Stage 2: High-quality sampling like baseline RankingSHAP
)
explainers.append(ranking_shapK_adaptive_refined_explainer)

# Add two-stage adaptive refined RankingSHAP
# Stage 1: Adaptive sampling to identify top features
# Stage 2: Refine top-k features with high-quality sampling
ranking_shapK_adaptive_refined_explainer = RankingShapAdaptiveRefined(
    permutation_sampler="kernel",
    background_data=background_data.background_summary,
    original_model=model.predict,
    explanation_size=explanation_size,
    name="rankingshapK_adaptive_refined",
    rank_similarity_coefficient=rank_similarity_coefficient,
    adaptive_min_samples=25,  # Base factor for sqrt(n_docs) rule (Stage 1 - like adaptive)
    adaptive_max_samples=300,  # Hard upper cap on samples (Stage 1 - like adaptive)
    top_k_to_refine=10,  # Number of top features to refine
    refinement_samples="auto",  # Stage 2: High-quality sampling like baseline RankingSHAP
)
explainers.append(ranking_shapK_adaptive_refined_explainer)

names = {explainer.name: explainer for explainer in explainers}
if args.approach in names:
    explainers = [names[args.approach]]

# Track timing results for comparison
timing_results = []
per_query_timing_data = {}  # Store per-query timing for each explainer

# Check for GPU availability
try:
    import torch
    gpu_available = torch.cuda.is_available()
    if gpu_available:
        gpu_device = torch.cuda.get_device_name(0)
        print(f"[Timing] GPU available: {gpu_device}", flush=True)
    else:
        print("[Timing] GPU not available (using CPU)", flush=True)
except ImportError:
    gpu_available = False
    print("[Timing] PyTorch not available, GPU timing disabled", flush=True)

for exp in explainers:
    if test:
        path_to_attribute_values = path_to_attribution_folder / (exp.name + "_test.csv")
    else:
        path_to_attribute_values = path_to_attribution_folder / (exp.name + ".csv")

    print("\n" + "="*80, flush=True)
    print(f"Starting {exp.name}", flush=True)
    print(f"Target csv will be {path_to_attribute_values}", flush=True)
    print("="*80, flush=True)
    
    # Start timing
    start_time = time.time()
    start_cpu_time = time.process_time()
    
    # GPU timing (if available)
    if gpu_available:
        try:
            import torch
            torch.cuda.synchronize()  # Ensure GPU operations are synchronized
            gpu_start_event = torch.cuda.Event(enable_timing=True)
            gpu_end_event = torch.cuda.Event(enable_timing=True)
            gpu_start_event.record()
        except:
            gpu_start_event = None
            gpu_end_event = None
    else:
        gpu_start_event = None
        gpu_end_event = None
    
    # Run the explainer with per-query timing enabled
    per_query_timing = calculate_all_query_explanations(
        explainer=exp,
        eval_data=test_data,
        num_queries_to_eval=num_queries_eval,
        progress=True,
        safe_attributions_to=path_to_attribute_values,
        track_per_query_timing=True,
    )
    
    # End timing
    end_time = time.time()
    end_cpu_time = time.process_time()
    
    # Calculate elapsed times
    wall_clock_time = end_time - start_time
    cpu_time = end_cpu_time - start_cpu_time
    
    # GPU timing (if available)
    gpu_time = None
    if gpu_available and gpu_start_event is not None:
        try:
            import torch
            gpu_end_event.record()
            torch.cuda.synchronize()
            gpu_time = gpu_start_event.elapsed_time(gpu_end_event) / 1000.0  # Convert ms to seconds
        except:
            pass
    
    # Calculate number of queries processed
    if num_queries_eval is not None:
        num_queries_processed = int(num_queries_eval)  # Convert to native Python int
    else:
        # Get all unique query IDs from test_data
        _, _, Eqids = test_data
        num_queries_processed = int(len(get_queryids_as_list(Eqids)))  # Convert to native Python int
    
    # Store timing results (ensure all values are native Python types for JSON serialization)
    timing_result = {
        "explainer": exp.name,
        "wall_clock_time_seconds": float(wall_clock_time),  # Ensure float
        "cpu_time_seconds": float(cpu_time),  # Ensure float
        "num_queries": num_queries_processed,
        "dataset": dataset,
        "experiment_iteration": int(experiment_iteration),  # Ensure int
    }
    
    if gpu_time is not None:
        timing_result["gpu_time_seconds"] = float(gpu_time)  # Ensure float
    
    # Store per-query timing data
    if per_query_timing:
        per_query_timing_data[exp.name] = per_query_timing
        timing_result["per_query_timing"] = per_query_timing
        
        # Calculate per-query statistics
        wall_times = [q["wall_clock_time_seconds"] for q in per_query_timing]
        cpu_times = [q["cpu_time_seconds"] for q in per_query_timing]
        num_docs = [q["num_documents"] for q in per_query_timing]
        
        timing_result["per_query_stats"] = {
            "wall_clock": {
                "mean": float(np.mean(wall_times)),
                "std": float(np.std(wall_times)),
                "min": float(np.min(wall_times)),
                "max": float(np.max(wall_times)),
                "median": float(np.median(wall_times)),
            },
            "cpu": {
                "mean": float(np.mean(cpu_times)),
                "std": float(np.std(cpu_times)),
                "min": float(np.min(cpu_times)),
                "max": float(np.max(cpu_times)),
                "median": float(np.median(cpu_times)),
            },
            "num_documents": {
                "mean": float(np.mean(num_docs)),
                "std": float(np.std(num_docs)),
                "min": int(np.min(num_docs)),
                "max": int(np.max(num_docs)),
                "median": float(np.median(num_docs)),
            },
        }
    
    timing_results.append(timing_result)
    
    # Print timing summary
    print("\n" + "="*80, flush=True)
    print(f"Timing Summary for {exp.name}:", flush=True)
    print(f"  Total wall-clock time: {wall_clock_time:.2f} seconds ({wall_clock_time/60:.2f} minutes)", flush=True)
    print(f"  Total CPU time: {cpu_time:.2f} seconds ({cpu_time/60:.2f} minutes)", flush=True)
    if gpu_time is not None:
        print(f"  Total GPU time: {gpu_time:.2f} seconds ({gpu_time/60:.2f} minutes)", flush=True)
    
    # Print per-query statistics if available
    if per_query_timing and len(per_query_timing) > 0:
        stats = timing_result.get("per_query_stats", {})
        if stats:
            print(f"\n  Per-query statistics ({len(per_query_timing)} queries):", flush=True)
            print(f"    Wall-clock: mean={stats['wall_clock']['mean']:.3f}s, std={stats['wall_clock']['std']:.3f}s, "
                  f"min={stats['wall_clock']['min']:.3f}s, max={stats['wall_clock']['max']:.3f}s", flush=True)
            print(f"    CPU: mean={stats['cpu']['mean']:.3f}s, std={stats['cpu']['std']:.3f}s, "
                  f"min={stats['cpu']['min']:.3f}s, max={stats['cpu']['max']:.3f}s", flush=True)
            print(f"    Query complexity: mean={stats['num_documents']['mean']:.1f} docs, "
                  f"range=[{stats['num_documents']['min']}-{stats['num_documents']['max']}]", flush=True)
    
    print("="*80 + "\n", flush=True)

# Save timing results to JSON file
timing_output_file = path_to_attribution_folder / f"timing_results_iter{experiment_iteration}.json"
with open(timing_output_file, 'w') as f:
    json.dump(timing_results, f, indent=2)

print("\n" + "="*80, flush=True)
print("TIMING COMPARISON SUMMARY", flush=True)
print("="*80, flush=True)

if len(timing_results) == 2:
    baseline = timing_results[0]
    adaptive = timing_results[1]
    
    print(f"\nBaseline ({baseline['explainer']}):", flush=True)
    print(f"  Wall-clock: {baseline['wall_clock_time_seconds']:.2f}s", flush=True)
    print(f"  CPU: {baseline['cpu_time_seconds']:.2f}s", flush=True)
    
    print(f"\nAdaptive ({adaptive['explainer']}):", flush=True)
    print(f"  Wall-clock: {adaptive['wall_clock_time_seconds']:.2f}s", flush=True)
    print(f"  CPU: {adaptive['cpu_time_seconds']:.2f}s", flush=True)
    
    # Calculate overall speedup
    wall_speedup = baseline['wall_clock_time_seconds'] / adaptive['wall_clock_time_seconds']
    cpu_speedup = baseline['cpu_time_seconds'] / adaptive['cpu_time_seconds']
    
    print(f"\nSpeedup:", flush=True)
    print(f"  Wall-clock: {wall_speedup:.2f}x {'faster' if wall_speedup > 1 else 'slower'}", flush=True)
    print(f"  CPU: {cpu_speedup:.2f}x {'faster' if cpu_speedup > 1 else 'slower'}", flush=True)
    
    if 'gpu_time_seconds' in baseline and 'gpu_time_seconds' in adaptive:
        gpu_speedup = baseline['gpu_time_seconds'] / adaptive['gpu_time_seconds']
        print(f"  GPU: {gpu_speedup:.2f}x {'faster' if gpu_speedup > 1 else 'slower'}", flush=True)
    
    # Additional per-query analysis (extension)
    if 'per_query_timing' in baseline and 'per_query_timing' in adaptive:
        baseline_times = {q['query_id']: q['wall_clock_time_seconds'] for q in baseline['per_query_timing']}
        adaptive_times = {q['query_id']: q['wall_clock_time_seconds'] for q in adaptive['per_query_timing']}
        
        # Calculate per-query speedups
        per_query_speedups = []
        per_query_num_docs = []
        for qid in baseline_times:
            if qid in adaptive_times:
                speedup = baseline_times[qid] / adaptive_times[qid]
                per_query_speedups.append(speedup)
                # Get num_docs from baseline
                baseline_q = next(q for q in baseline['per_query_timing'] if q['query_id'] == qid)
                per_query_num_docs.append(baseline_q['num_documents'])
        
        if per_query_speedups:
            print(f"\n" + "-"*80, flush=True)
            print("ADDITIONAL PER-QUERY ANALYSIS", flush=True)
            print("-"*80, flush=True)
            print(f"\nPer-Query Speedup Statistics:", flush=True)
            print(f"  Mean: {np.mean(per_query_speedups):.2f}x", flush=True)
            print(f"  Std: {np.std(per_query_speedups):.2f}x", flush=True)
            print(f"  Min: {np.min(per_query_speedups):.2f}x", flush=True)
            print(f"  Max: {np.max(per_query_speedups):.2f}x", flush=True)
            print(f"  Median: {np.median(per_query_speedups):.2f}x", flush=True)
            
            # Correlation with query complexity
            if len(per_query_num_docs) > 1:
                correlation = np.corrcoef(per_query_num_docs, per_query_speedups)[0, 1]
                print(f"\nSpeedup vs Query Complexity (num_documents):", flush=True)
                print(f"  Correlation: {correlation:.3f}", flush=True)
                if abs(correlation) > 0.3:
                    direction = "positive" if correlation > 0 else "negative"
                    print(f"  → Speedup has {direction} correlation with query size", flush=True)
                else:
                    print(f"  → Speedup is relatively independent of query size", flush=True)

print(f"\nDetailed timing results saved to: {timing_output_file}", flush=True)
print("="*80 + "\n", flush=True)
