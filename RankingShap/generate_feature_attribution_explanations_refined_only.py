"""
Generate feature attribution explanations for ONLY the adaptive_refined approach.
This script is designed to run only the refined approach and compare with existing
rankingshapK results (which are loaded from previous runs).
"""

from utils.get_explanations import calculate_all_query_explanations
from utils.helper_functions import get_data, get_queryids_as_list
import lightgbm
import numpy as np
from scipy.stats import kendalltau
from utils.background_data import BackgroundData
from approaches.ranking_shap_adaptive_refined import RankingShapAdaptiveRefined
from pathlib import Path
import time
import json
import numpy as np

import argparse

parser = argparse.ArgumentParser(description="Runs ONLY the adaptive_refined explanation approach")

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
    type=int,
    default=1,
    help="The iteration number of the experiment",
)
parser.add_argument(
    "--test", action="store_true", help="If true uses test files for evaluation"
)
parser.add_argument(
    "--fold",
    required=False,
    type=int,
    default=1,
    help="Fold number to use (default: 1)",
)

args = parser.parse_args()
print(args, flush=True)

dataset = args.dataset
fold = args.fold
experiment_iteration = args.experiment_iteration
test = args.test

# Load fold-specific data
data_directory = Path(f"data/{dataset}/Fold{fold}/")
if test:
    eval_data = get_data(data_file=data_directory / "test.txt")
else:
    eval_data = get_data(data_file=data_directory / "vali.txt")

# Load model
model_file = args.model_file
if f"_fold{fold}" not in model_file:
    model_file_with_fold = f"{model_file}_fold{fold}"
else:
    model_file_with_fold = model_file

model = lightgbm.Booster(
    model_file=(str((Path("results/model_files/") / model_file_with_fold).absolute()))
)

# Load background data
background_data = BackgroundData(
    np.load(
        Path(f"results/background_data_files/train_background_data_{dataset}_fold{fold}.npy")
    ),
    summarization_type=None,
)

# Setup paths
path_to_attribution_folder = Path(f"results/results_{dataset}_fold{fold}/feature_attributes/")
path_to_attribution_folder.mkdir(parents=True, exist_ok=True)

# Configuration
explanation_size = 3
rank_similarity_coefficient = lambda x, y: kendalltau(x, y)[0]

# Create ONLY the adaptive_refined explainer
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

explainers = [ranking_shapK_adaptive_refined_explainer]

# Track timing results
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
        eval_data=eval_data,
        num_queries_to_eval=None,
        progress=True,
        safe_attributions_to=path_to_attribute_values,
        track_per_query_timing=True,  # Enable per-query timing
    )
    
    # End timing
    end_time = time.time()
    end_cpu_time = time.process_time()
    
    # GPU timing (if available)
    if gpu_available and gpu_start_event is not None:
        try:
            import torch
            gpu_end_event.record()
            torch.cuda.synchronize()
            gpu_time_ms = gpu_start_event.elapsed_time(gpu_end_event) / 1000.0  # Convert to seconds
        except:
            gpu_time_ms = None
    else:
        gpu_time_ms = None
    
    # Calculate timing
    wall_clock_time = end_time - start_time
    cpu_time = end_cpu_time - start_cpu_time
    
    # Store timing results
    timing_result = {
        'explainer': exp.name,
        'wall_clock_time': float(wall_clock_time),
        'cpu_time': float(cpu_time),
        'gpu_time': float(gpu_time_ms) if gpu_time_ms is not None else None,
    }
    timing_results.append(timing_result)
    
    # Store per-query timing
    if per_query_timing:
        per_query_timing_data[exp.name] = per_query_timing
    
    print(f"\n[Timing] {exp.name}:", flush=True)
    print(f"  Wall-clock time: {wall_clock_time:.2f} seconds", flush=True)
    print(f"  CPU time: {cpu_time:.2f} seconds", flush=True)
    if gpu_time_ms is not None:
        print(f"  GPU time: {gpu_time_ms:.2f} seconds", flush=True)

# Load existing rankingshapK timing results for comparison
baseline_timing_file = path_to_attribution_folder / f"timing_results_fold{fold}_iter{experiment_iteration}.json"
baseline_timing_data = None
baseline_per_query_timing = None

if baseline_timing_file.exists():
    print(f"\n[Comparison] Loading baseline timing from: {baseline_timing_file}", flush=True)
    with open(baseline_timing_file, 'r') as f:
        baseline_data = json.load(f)
        # baseline_data is a list of timing results
        # Find rankingshapK timing
        if isinstance(baseline_data, list):
            for result in baseline_data:
                if result.get('explainer') == 'rankingshapK':
                    baseline_timing_data = result
                    break
        elif isinstance(baseline_data, dict):
            # Handle dict format if needed
            baseline_timing_data = baseline_data.get('rankingshapK', None)
        
        # Per-query timing is stored in the timing_result dict itself
        if baseline_timing_data and 'per_query_timing' in baseline_timing_data:
            baseline_per_query_timing = baseline_timing_data['per_query_timing']
    
    if baseline_timing_data:
        print(f"[Comparison] Found baseline rankingshapK timing", flush=True)
        
        # Calculate speedup
        baseline_wall = baseline_timing_data.get('wall_clock_time', 0)
        baseline_cpu = baseline_timing_data.get('cpu_time', 0)
        
        refined_wall = timing_results[0]['wall_clock_time']
        refined_cpu = timing_results[0]['cpu_time']
        
        if baseline_wall > 0:
            speedup_wall = baseline_wall / refined_wall
            print(f"\n[Speedup] Wall-clock time:", flush=True)
            print(f"  Baseline: {baseline_wall:.2f}s", flush=True)
            print(f"  Refined:  {refined_wall:.2f}s", flush=True)
            print(f"  Speedup:  {speedup_wall:.2f}x", flush=True)
        
        if baseline_cpu > 0:
            speedup_cpu = baseline_cpu / refined_cpu
            print(f"\n[Speedup] CPU time:", flush=True)
            print(f"  Baseline: {baseline_cpu:.2f}s", flush=True)
            print(f"  Refined:  {refined_cpu:.2f}s", flush=True)
            print(f"  Speedup:  {speedup_cpu:.2f}x", flush=True)
    else:
        print(f"[Comparison] Warning: Could not find rankingshapK timing in baseline file", flush=True)
else:
    print(f"\n[Comparison] Warning: Baseline timing file not found: {baseline_timing_file}", flush=True)
    print(f"  Will save timing results but cannot compare with baseline", flush=True)

# Save timing results (including comparison if available)
timing_output = {
    'refined': timing_results[0],
    'baseline': baseline_timing_data if baseline_timing_data else None,
    'speedup': {}
}

# Add per-query timing to refined if available
if per_query_timing_data:
    timing_output['refined']['per_query_timing'] = per_query_timing_data.get('rankingshapK_adaptive_refined', [])

# Add per-query timing to baseline if available
if baseline_timing_data and baseline_per_query_timing:
    timing_output['baseline'] = baseline_timing_data.copy()
    timing_output['baseline']['per_query_timing'] = baseline_per_query_timing

# Calculate speedup if baseline available
if baseline_timing_data:
    baseline_wall = baseline_timing_data.get('wall_clock_time', 0)
    baseline_cpu = baseline_timing_data.get('cpu_time', 0)
    refined_wall = timing_results[0]['wall_clock_time']
    refined_cpu = timing_results[0]['cpu_time']
    
    if baseline_wall > 0:
        timing_output['speedup']['wall_clock'] = baseline_wall / refined_wall
    if baseline_cpu > 0:
        timing_output['speedup']['cpu'] = baseline_cpu / refined_cpu

# Save refined timing results
refined_timing_file = path_to_attribution_folder / f"timing_results_refined_fold{fold}_iter{experiment_iteration}.json"
with open(refined_timing_file, 'w') as f:
    json.dump(timing_output, f, indent=2)
print(f"\n[Timing] Saved refined timing results to: {refined_timing_file}", flush=True)

print("\n" + "="*80, flush=True)
print("COMPLETED: Adaptive Refined approach", flush=True)
print("="*80, flush=True)

