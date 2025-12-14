"""
Fast experiment to test different base values for adaptive sampling.
Tests multiple base values on a small subset of queries to find optimal base.
Compares both speed AND quality metrics (Pre_ken, Del_ken, Pre_exp, Del_exp) vs baseline.
"""

from utils.get_explanations import calculate_all_query_explanations
from utils.helper_functions import get_data, get_queryids_as_list
from evaluation.evaluate_feature_attribution import eval_feature_attribution
import lightgbm
import numpy as np
from scipy.stats import kendalltau
from utils.background_data import BackgroundData
from approaches.ranking_shap import RankingShap
from approaches.ranking_shap_adaptive import RankingShapAdaptive
from pathlib import Path
import time
import json
import pandas as pd
import shutil

import argparse

parser = argparse.ArgumentParser(description="Test different base values for adaptive sampling")

parser.add_argument(
    "--model_file",
    required=True,
    type=str,
    help="Path to the model file",
)
parser.add_argument(
    "--dataset",
    type=str,
    required=True,
    choices=["MSLR-WEB10K", "MQ2008"],
    help="The dataset to use",
)
parser.add_argument(
    "--fold",
    required=False,
    type=int,
    default=1,
    help="Fold number to use (default: 1)",
)
parser.add_argument(
    "--num_queries",
    required=False,
    type=int,
    default=20,
    help="Number of queries to test (default: 20, for speed)",
)
parser.add_argument(
    "--base_values",
    required=False,
    type=str,
    default="25,30,40,50,60,80,100,120,150",
    help="Comma-separated list of base values to test",
)
parser.add_argument(
    "--ground_truth_file",
    required=False,
    type=str,
    default=None,
    help="Ground truth attribution file (if None, will skip quality evaluation)",
)
parser.add_argument(
    "--experiment_iteration",
    required=False,
    type=int,
    default=1,
    help="Experiment iteration number (for finding existing baseline timing file, default: 1)",
)
parser.add_argument(
    "--use_existing_baseline",
    required=False,
    action="store_true",
    help="Use existing baseline from main results folder (scales timing to match num_queries)",
)

args = parser.parse_args()
print(args, flush=True)

dataset = args.dataset
fold = args.fold
num_queries = args.num_queries

# Parse base values
base_values = [int(x.strip()) for x in args.base_values.split(',')]
print(f"\nTesting base values: {base_values}", flush=True)

# Load fold-specific data
data_directory = Path(f"data/{dataset}/Fold{fold}/")
eval_data = get_data(data_file=data_directory / "vali.txt")

print(f"Will test on first {num_queries} queries for speed", flush=True)

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

# Setup output directory
output_dir = Path(f"results/base_value_test_fold{fold}/")
output_dir.mkdir(parents=True, exist_ok=True)
attribution_dir = output_dir / "feature_attributes"
attribution_dir.mkdir(parents=True, exist_ok=True)

# Configuration
explanation_size = 3
rank_similarity_coefficient = lambda x, y: kendalltau(x, y)[0]
max_samples = 2000  # Cap like "auto"

# Results storage
results = []
baseline_metrics = None

print(f"\n{'='*80}")
print("STEP 1: Run Baseline RankingSHAP for comparison")
print(f"{'='*80}\n")

# Run baseline first (or load if exists)
baseline_attribution_file = attribution_dir / "rankingshapK_baseline.csv"
baseline_timing_file = attribution_dir / "baseline_timing.json"

# Initialize baseline timing variables
baseline_wall_time = None
baseline_cpu_time = None

# Check for existing baseline in main results folder
existing_baseline_attribution = None
existing_baseline_timing = None
if args.use_existing_baseline:
    # Try to find existing baseline files
    main_results_dir = Path(f"results/results_{dataset}_fold{fold}/feature_attributes/")
    
    # Try both possible attribution file names
    for attr_file_name in ["rankingshapK.csv", "rankingshapK_test.csv"]:
        potential_attr_file = main_results_dir / attr_file_name
        if potential_attr_file.exists():
            existing_baseline_attribution = potential_attr_file
            print(f"Found existing baseline attribution file: {potential_attr_file}", flush=True)
            break
    
    # Try to find timing file
    timing_file_name = f"timing_results_fold{fold}_iter{args.experiment_iteration}.json"
    potential_timing_file = main_results_dir / timing_file_name
    if potential_timing_file.exists():
        existing_baseline_timing = potential_timing_file
        print(f"Found existing baseline timing file: {potential_timing_file}", flush=True)
    
    if existing_baseline_attribution and existing_baseline_timing:
        print("Using existing baseline with scaled timing...", flush=True)
        
        # Copy attribution file
        shutil.copy2(existing_baseline_attribution, baseline_attribution_file)
        print(f"Copied baseline attribution file to: {baseline_attribution_file}", flush=True)
        
        # Load timing and scale it
        with open(existing_baseline_timing, 'r') as f:
            existing_timing = json.load(f)
        
        # Count queries in existing baseline attribution file
        baseline_df = pd.read_csv(baseline_attribution_file, index_col=0)
        num_queries_in_baseline = baseline_df['query_number'].nunique()
        
        # Get timing from existing file (look for rankingshapK entry)
        baseline_timing_data = None
        if isinstance(existing_timing, list):
            # If it's a list, find rankingshapK entry
            for entry in existing_timing:
                if entry.get('explainer') == 'rankingshapK':
                    baseline_timing_data = entry
                    break
        elif isinstance(existing_timing, dict):
            # If it's a dict, might be direct timing data
            if 'wall_clock_time_seconds' in existing_timing or 'total_wall_time' in existing_timing:
                baseline_timing_data = existing_timing
            else:
                # Try to find rankingshapK in nested structure
                for key, value in existing_timing.items():
                    if isinstance(value, dict) and value.get('explainer') == 'rankingshapK':
                        baseline_timing_data = value
                        break
        
        if baseline_timing_data:
            # Try both field name formats (wall_clock_time_seconds or total_wall_time)
            existing_wall_time = baseline_timing_data.get('wall_clock_time_seconds') or baseline_timing_data.get('total_wall_time', 0)
            existing_cpu_time = baseline_timing_data.get('cpu_time_seconds') or baseline_timing_data.get('total_cpu_time', 0)
            
            # Scale timing to match num_queries
            if num_queries_in_baseline > 0:
                scale_factor = num_queries / num_queries_in_baseline
                baseline_wall_time = existing_wall_time * scale_factor
                baseline_cpu_time = existing_cpu_time * scale_factor
                print(f"Scaled baseline timing: {num_queries_in_baseline} queries -> {num_queries} queries", flush=True)
                print(f"  Original: {existing_wall_time:.2f}s wall, {existing_cpu_time:.2f}s CPU", flush=True)
                print(f"  Scaled:   {baseline_wall_time:.2f}s wall, {baseline_cpu_time:.2f}s CPU", flush=True)
            else:
                baseline_wall_time = existing_wall_time
                baseline_cpu_time = existing_cpu_time
                print(f"Warning: Could not determine query count, using original timing", flush=True)
        else:
            print(f"Warning: Could not find rankingshapK timing in file, will run baseline", flush=True)
            existing_baseline_attribution = None  # Force re-run
        
        # Save scaled timing file
        with open(baseline_timing_file, 'w') as f:
            json.dump({
                'total_wall_time': baseline_wall_time,
                'total_cpu_time': baseline_cpu_time,
                'source': 'scaled_from_existing',
                'original_queries': num_queries_in_baseline,
                'scaled_to_queries': num_queries,
            }, f, indent=2)
    elif args.use_existing_baseline:
        print("Warning: --use_existing_baseline was set but baseline files not found.", flush=True)
        print("Will run baseline on limited queries instead.", flush=True)

if baseline_wall_time is None and baseline_attribution_file.exists() and baseline_timing_file.exists():
    print("Loading baseline results from file...", flush=True)
    with open(baseline_timing_file, 'r') as f:
        baseline_timing = json.load(f)
    baseline_wall_time = baseline_timing.get('total_wall_time', 0)
    baseline_cpu_time = baseline_timing.get('total_cpu_time', 0)

if baseline_wall_time is None:
    print("Running baseline RankingSHAP...", flush=True)
    baseline_explainer = RankingShap(
        permutation_sampler="kernel",
        background_data=background_data.background_summary,
        original_model=model.predict,
        explanation_size=explanation_size,
        name="rankingshapK",
        rank_similarity_coefficient=rank_similarity_coefficient,
    )
    
    start_time = time.time()
    start_cpu_time = time.process_time()
    
    calculate_all_query_explanations(
        explainer=baseline_explainer,
        eval_data=eval_data,
        num_queries_to_eval=num_queries,
        progress=True,
        safe_attributions_to=baseline_attribution_file,
        track_per_query_timing=False,
    )
    
    end_time = time.time()
    end_cpu_time = time.process_time()
    
    baseline_wall_time = end_time - start_time
    baseline_cpu_time = end_cpu_time - start_cpu_time
    
    with open(baseline_timing_file, 'w') as f:
        json.dump({
            'total_wall_time': baseline_wall_time,
            'total_cpu_time': baseline_cpu_time,
        }, f, indent=2)

print(f"Baseline time: {baseline_wall_time:.2f}s", flush=True)

# Evaluate baseline if ground truth available
ground_truth_path = None
if args.ground_truth_file:
    ground_truth_path = Path(f"results/results_{dataset}_fold{fold}/feature_attributes/{args.ground_truth_file}")
    if ground_truth_path.exists():
        print("\nEvaluating baseline against ground truth...", flush=True)
        try:
            baseline_metrics = eval_feature_attribution(
                attributes_to_evaluate=baseline_attribution_file,
                model=model.predict,
                eval_data=eval_data,  # Use same eval_data (limited queries)
                background=background_data.background_summary,
                ground_truth_file_path=ground_truth_path,
            )
            print("Baseline metrics (k=1,3,5,7,10):", flush=True)
            print(f"  Pre_ken: {baseline_metrics[:, 0]}")
            print(f"  Del_ken: {baseline_metrics[:, 1]}")
            print(f"  Pre_exp: {baseline_metrics[:, 2]}")
            print(f"  Del_exp: {baseline_metrics[:, 3]}")
        except Exception as e:
            print(f"Warning: Could not evaluate baseline: {e}", flush=True)
            baseline_metrics = None
    else:
        print(f"Warning: Ground truth file not found: {ground_truth_path}", flush=True)
        print("Skipping quality evaluation. Run with --ground_truth_file to enable.", flush=True)
        baseline_metrics = None
else:
    baseline_metrics = None

print(f"\n{'='*80}")
print(f"STEP 2: Testing {len(base_values)} base values on {num_queries} queries")
print(f"{'='*80}\n")

for base in base_values:
    print(f"\n{'='*80}")
    print(f"Testing base = {base}")
    print(f"{'='*80}\n")
    
    # Create explainer with this base value
    explainer = RankingShapAdaptive(
        permutation_sampler="kernel",
        background_data=background_data.background_summary,
        original_model=model.predict,
        explanation_size=explanation_size,
        name=f"rankingshapK_adaptive_base{base}",
        rank_similarity_coefficient=rank_similarity_coefficient,
        adaptive_min_samples=base,  # Test different base
        adaptive_max_samples=max_samples,  # High cap like "auto"
    )
    
    # Save attribution file
    attribution_file = attribution_dir / f"rankingshapK_adaptive_base{base}.csv"
    
    # Time the execution
    start_time = time.time()
    start_cpu_time = time.process_time()
    
    # Run on limited queries
    per_query_timing = calculate_all_query_explanations(
        explainer=explainer,
        eval_data=eval_data,
        num_queries_to_eval=num_queries,
        progress=True,
        safe_attributions_to=attribution_file,
        track_per_query_timing=True,
    )
    
    end_time = time.time()
    end_cpu_time = time.process_time()
    
    wall_clock_time = end_time - start_time
    cpu_time = end_cpu_time - start_cpu_time
    
    # Calculate statistics from per-query timing
    if per_query_timing:
        wall_times = [q['wall_clock_time_seconds'] for q in per_query_timing]
        cpu_times = [q['cpu_time_seconds'] for q in per_query_timing]
        num_docs = [q['num_documents'] for q in per_query_timing]
        samples_used = [int(base * np.sqrt(n)) for n in num_docs]
        samples_used = [min(s, max_samples) for s in samples_used]
        
        avg_samples = np.mean(samples_used)
        avg_wall_time = np.mean(wall_times)
        avg_cpu_time = np.mean(cpu_times)
    else:
        avg_samples = 0
        avg_wall_time = 0
        avg_cpu_time = 0
    
    # Evaluate quality if ground truth available
    quality_metrics = None
    if ground_truth_path and ground_truth_path.exists():
        try:
            quality_metrics = eval_feature_attribution(
                attributes_to_evaluate=attribution_file,
                model=model.predict,
                eval_data=eval_data,  # Use same eval_data (limited queries)
                background=background_data.background_summary,
                ground_truth_file_path=ground_truth_path,
            )
        except Exception as e:
            print(f"Warning: Could not evaluate quality: {e}", flush=True)
            quality_metrics = None
    
    # Calculate speedup vs baseline
    speedup_wall = baseline_wall_time / wall_clock_time if wall_clock_time > 0 else 0
    speedup_cpu = baseline_cpu_time / cpu_time if cpu_time > 0 else 0
    
    # Calculate quality difference vs baseline (if available)
    quality_diff = {}
    if baseline_metrics is not None and quality_metrics is not None:
        # Compare at k=10 (last index)
        k10_idx = 4  # k=10 is index 4 in [1,3,5,7,10]
        quality_diff['Pre_ken_diff'] = float(quality_metrics[k10_idx, 0] - baseline_metrics[k10_idx, 0])
        quality_diff['Del_ken_diff'] = float(quality_metrics[k10_idx, 1] - baseline_metrics[k10_idx, 1])
        quality_diff['Pre_exp_diff'] = float(quality_metrics[k10_idx, 2] - baseline_metrics[k10_idx, 2])
        quality_diff['Del_exp_diff'] = float(quality_metrics[k10_idx, 3] - baseline_metrics[k10_idx, 3])
        
        # Calculate overall quality score (closer to baseline = better)
        # For Pre_ken and Pre_exp: higher is better, so positive diff is good
        # For Del_ken and Del_exp: more negative is better, so negative diff is good
        quality_score = (
            quality_diff['Pre_ken_diff'] -  # Positive is good
            abs(quality_diff['Del_ken_diff']) -  # Closer to 0 is good (Del_ken is negative)
            abs(quality_diff['Pre_exp_diff']) -  # Closer to 0 is good
            abs(quality_diff['Del_exp_diff'])  # Closer to 0 is good
        )
        quality_diff['overall_score'] = quality_score
    else:
        quality_diff = {
            'Pre_ken_diff': None,
            'Del_ken_diff': None,
            'Pre_exp_diff': None,
            'Del_exp_diff': None,
            'overall_score': None,
        }
    
    result = {
        'base': base,
        'total_wall_time': wall_clock_time,
        'total_cpu_time': cpu_time,
        'avg_wall_time_per_query': avg_wall_time,
        'avg_cpu_time_per_query': avg_cpu_time,
        'avg_samples_per_query': avg_samples,
        'speedup_wall': speedup_wall,
        'speedup_cpu': speedup_cpu,
        'num_queries': num_queries,
        **quality_diff,
    }
    
    if quality_metrics is not None:
        result['Pre_ken'] = float(quality_metrics[k10_idx, 0])
        result['Del_ken'] = float(quality_metrics[k10_idx, 1])
        result['Pre_exp'] = float(quality_metrics[k10_idx, 2])
        result['Del_exp'] = float(quality_metrics[k10_idx, 3])
    
    results.append(result)
    
    print(f"\nBase {base} Results:")
    print(f"  Total time: {wall_clock_time:.2f}s")
    print(f"  Speedup vs baseline: {speedup_wall:.2f}x")
    print(f"  Avg samples per query: {avg_samples:.1f}")
    if quality_metrics is not None:
        print(f"  Quality metrics (k=10):")
        print(f"    Pre_ken: {quality_metrics[k10_idx, 0]:.4f} (baseline: {baseline_metrics[k10_idx, 0]:.4f}, diff: {quality_diff['Pre_ken_diff']:+.4f})")
        print(f"    Del_ken: {quality_metrics[k10_idx, 1]:.4f} (baseline: {baseline_metrics[k10_idx, 1]:.4f}, diff: {quality_diff['Del_ken_diff']:+.4f})")
        print(f"    Pre_exp: {quality_metrics[k10_idx, 2]:.4f} (baseline: {baseline_metrics[k10_idx, 2]:.4f}, diff: {quality_diff['Pre_exp_diff']:+.4f})")
        print(f"    Del_exp: {quality_metrics[k10_idx, 3]:.4f} (baseline: {baseline_metrics[k10_idx, 3]:.4f}, diff: {quality_diff['Del_exp_diff']:+.4f})")

# Create comparison table
print(f"\n{'='*80}")
print("COMPREHENSIVE COMPARISON RESULTS")
print(f"{'='*80}\n")

df = pd.DataFrame(results)
df = df.sort_values('base')

# Print speed comparison
print("SPEED COMPARISON:")
print(f"{'Base':<8} {'Time (s)':<12} {'Speedup':<12} {'Avg Samples':<15}")
print("-" * 60)
for _, row in df.iterrows():
    print(f"{int(row['base']):<8} {row['total_wall_time']:<12.2f} {row['speedup_wall']:<12.2f} {row['avg_samples_per_query']:<15.1f}")

# Print quality comparison if available
if baseline_metrics is not None and any(df['Pre_ken'].notna()):
    print(f"\n{'='*80}")
    print("QUALITY COMPARISON (vs Baseline RankingSHAP, k=10):")
    print(f"{'='*80}\n")
    print(f"{'Base':<8} {'Pre_ken':<12} {'Del_ken':<12} {'Pre_exp':<12} {'Del_exp':<12} {'Quality Score':<15}")
    print("-" * 80)
    for _, row in df.iterrows():
        if pd.notna(row.get('Pre_ken')):
            pre_ken = f"{row['Pre_ken']:.4f}"
            del_ken = f"{row['Del_ken']:.4f}"
            pre_exp = f"{row['Pre_exp']:.4f}"
            del_exp = f"{row['Del_exp']:.4f}"
            q_score = f"{row.get('overall_score', 0):.4f}" if pd.notna(row.get('overall_score')) else "N/A"
            print(f"{int(row['base']):<8} {pre_ken:<12} {del_ken:<12} {pre_exp:<12} {del_exp:<12} {q_score:<15}")
    
    # Show differences
    print(f"\n{'='*80}")
    print("QUALITY DIFFERENCES (vs Baseline, k=10):")
    print(f"{'='*80}\n")
    print(f"{'Base':<8} {'Pre_ken Δ':<12} {'Del_ken Δ':<12} {'Pre_exp Δ':<12} {'Del_exp Δ':<12}")
    print("-" * 80)
    for _, row in df.iterrows():
        if pd.notna(row.get('Pre_ken_diff')):
            pre_ken_diff = f"{row['Pre_ken_diff']:+.4f}"
            del_ken_diff = f"{row['Del_ken_diff']:+.4f}"
            pre_exp_diff = f"{row['Pre_exp_diff']:+.4f}"
            del_exp_diff = f"{row['Del_exp_diff']:+.4f}"
            print(f"{int(row['base']):<8} {pre_ken_diff:<12} {del_ken_diff:<12} {pre_exp_diff:<12} {del_exp_diff:<12}")

# Find best base (balance of speed and quality)
print(f"\n{'='*80}")
print("BEST BASE RECOMMENDATION")
print(f"{'='*80}\n")

if baseline_metrics is not None and any(df['overall_score'].notna()):
    # Normalize metrics (0-1 scale)
    df['speedup_norm'] = (df['speedup_wall'] - df['speedup_wall'].min()) / (df['speedup_wall'].max() - df['speedup_wall'].min())
    
    # For quality score, normalize (higher is better)
    quality_scores = df['overall_score'].dropna()
    if len(quality_scores) > 0:
        df['quality_norm'] = 0.0
        df.loc[quality_scores.index, 'quality_norm'] = (
            (quality_scores - quality_scores.min()) / (quality_scores.max() - quality_scores.min())
        )
        
        # Combined score: balance speed and quality
        weight_speed = 0.4  # 40% weight on speed
        weight_quality = 0.6  # 60% weight on quality
        
        df['combined_score'] = (
            weight_speed * df['speedup_norm'] + 
            weight_quality * df['quality_norm']
        )
        
        best_base = df.loc[df['combined_score'].idxmax(), 'base']
        
        print(f"Best base (weighted: {int(weight_speed*100)}% speed, {int(weight_quality*100)}% quality): {int(best_base)}")
        print(f"\nDetailed scores:")
        for _, row in df.iterrows():
            if pd.notna(row.get('combined_score')):
                print(f"  Base {int(row['base']):<4}: combined={row['combined_score']:.3f} (speed: {row['speedup_norm']:.3f}, quality: {row['quality_norm']:.3f})")
    else:
        # Fallback to speed only
        best_base = df.loc[df['speedup_wall'].idxmax(), 'base']
        print(f"Best base (speed only, quality not available): {int(best_base)}")
else:
    # No quality data, use speed only
    best_base = df.loc[df['speedup_wall'].idxmax(), 'base']
    print(f"Best base (speed only, quality not evaluated): {int(best_base)}")
    print("Run with --ground_truth_file to enable quality comparison")

# Save results
output_file = output_dir / "base_comparison_results.json"
with open(output_file, 'w') as f:
    json.dump({
        'num_queries_tested': num_queries,
        'baseline_time': baseline_wall_time,
        'baseline_metrics': baseline_metrics.tolist() if baseline_metrics is not None else None,
        'results': results,
        'best_base': int(best_base),
    }, f, indent=2)

print(f"\nResults saved to: {output_file}")

# Save CSV for easy viewing
csv_file = output_dir / "base_comparison_results.csv"
df.to_csv(csv_file, index=False)
print(f"CSV saved to: {csv_file}")

print(f"\n{'='*80}")
print("EXPERIMENT COMPLETE")
print(f"{'='*80}\n")
print(f"Recommended base value: {int(best_base)}")
print("Test this base value on full dataset to confirm results")
