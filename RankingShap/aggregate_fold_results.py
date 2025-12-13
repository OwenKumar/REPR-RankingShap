#!/usr/bin/env python3
"""
Aggregate results across all folds for timing and evaluation metrics.

This script:
1. Aggregates timing results from all folds (mean ± std)
2. Aggregates evaluation results from all folds (mean ± std)
3. Outputs summary reports for publication
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from collections import defaultdict


def aggregate_timing_results(dataset, experiment_iteration=1, num_folds=5):
    """
    Aggregate timing results across all folds.
    
    Returns:
        dict: Aggregated timing statistics
    """
    print("\n" + "="*80)
    print("AGGREGATING TIMING RESULTS ACROSS FOLDS")
    print("="*80)
    
    timing_data = defaultdict(list)
    per_fold_results = {}
    
    # Load timing results from each fold
    for fold in range(1, num_folds + 1):
        timing_file = Path(f"results/results_{dataset}_fold{fold}/feature_attributes/timing_results_fold{fold}_iter{experiment_iteration}.json")
        
        if not timing_file.exists():
            print(f"Warning: Timing file not found for fold {fold}: {timing_file}")
            continue
        
        print(f"Loading timing results from fold {fold}...")
        with open(timing_file, 'r') as f:
            fold_timing = json.load(f)
        
        per_fold_results[fold] = fold_timing
        
        # Extract data for each explainer
        for result in fold_timing:
            explainer_name = result['explainer']
            timing_data[explainer_name].append({
                'fold': fold,
                'wall_clock_time_seconds': result['wall_clock_time_seconds'],
                'cpu_time_seconds': result['cpu_time_seconds'],
                'num_queries': result['num_queries'],
                'gpu_time_seconds': result.get('gpu_time_seconds'),
                'per_query_stats': result.get('per_query_stats', {}),
            })
    
    if not timing_data:
        print("ERROR: No timing data found across any folds!")
        return None
    
    # Aggregate statistics
    aggregated = {}
    
    for explainer_name, fold_results in timing_data.items():
        wall_times = [r['wall_clock_time_seconds'] for r in fold_results]
        cpu_times = [r['cpu_time_seconds'] for r in fold_results]
        num_queries = [r['num_queries'] for r in fold_results]
        gpu_times = [r['gpu_time_seconds'] for r in fold_results if r['gpu_time_seconds'] is not None]
        
        aggregated[explainer_name] = {
            'wall_clock_time': {
                'mean': float(np.mean(wall_times)),
                'std': float(np.std(wall_times)),
                'min': float(np.min(wall_times)),
                'max': float(np.max(wall_times)),
                'values': wall_times,  # Keep individual values
            },
            'cpu_time': {
                'mean': float(np.mean(cpu_times)),
                'std': float(np.std(cpu_times)),
                'min': float(np.min(cpu_times)),
                'max': float(np.max(cpu_times)),
                'values': cpu_times,
            },
            'num_queries': {
                'mean': float(np.mean(num_queries)),
                'values': num_queries,
            },
        }
        
        if gpu_times:
            aggregated[explainer_name]['gpu_time'] = {
                'mean': float(np.mean(gpu_times)),
                'std': float(np.std(gpu_times)),
                'values': gpu_times,
            }
        
        # Aggregate per-query statistics if available
        per_query_wall_means = []
        per_query_cpu_means = []
        for r in fold_results:
            if 'per_query_stats' in r and r['per_query_stats']:
                per_query_wall_means.append(r['per_query_stats'].get('wall_clock', {}).get('mean'))
                per_query_cpu_means.append(r['per_query_stats'].get('cpu', {}).get('mean'))
        
        if per_query_wall_means:
            aggregated[explainer_name]['per_query_wall_clock'] = {
                'mean': float(np.mean(per_query_wall_means)),
                'std': float(np.std(per_query_wall_means)),
            }
        if per_query_cpu_means:
            aggregated[explainer_name]['per_query_cpu'] = {
                'mean': float(np.mean(per_query_cpu_means)),
                'std': float(np.std(per_query_cpu_means)),
            }
    
    # Calculate speedup if both baseline and adaptive are present
    if 'rankingshapK' in aggregated and 'rankingshapK_adaptive' in aggregated:
        baseline = aggregated['rankingshapK']
        adaptive = aggregated['rankingshapK_adaptive']
        
        # Calculate speedup for each fold
        speedups_wall = []
        speedups_cpu = []
        
        for fold in range(1, num_folds + 1):
            if fold in per_fold_results:
                baseline_fold = next((r for r in per_fold_results[fold] if r['explainer'] == 'rankingshapK'), None)
                adaptive_fold = next((r for r in per_fold_results[fold] if r['explainer'] == 'rankingshapK_adaptive'), None)
                
                if baseline_fold and adaptive_fold:
                    speedup_wall = baseline_fold['wall_clock_time_seconds'] / adaptive_fold['wall_clock_time_seconds']
                    speedup_cpu = baseline_fold['cpu_time_seconds'] / adaptive_fold['cpu_time_seconds']
                    speedups_wall.append(speedup_wall)
                    speedups_cpu.append(speedup_cpu)
        
        if speedups_wall:
            aggregated['speedup'] = {
                'wall_clock': {
                    'mean': float(np.mean(speedups_wall)),
                    'std': float(np.std(speedups_wall)),
                    'min': float(np.min(speedups_wall)),
                    'max': float(np.max(speedups_wall)),
                    'values': speedups_wall,
                },
                'cpu': {
                    'mean': float(np.mean(speedups_cpu)),
                    'std': float(np.std(speedups_cpu)),
                    'min': float(np.min(speedups_cpu)),
                    'max': float(np.max(speedups_cpu)),
                    'values': speedups_cpu,
                },
            }
    
    return aggregated


def aggregate_evaluation_results(dataset, num_folds=5):
    """
    Aggregate evaluation results across all folds.
    
    Returns:
        dict: Aggregated evaluation statistics
    """
    print("\n" + "="*80)
    print("AGGREGATING EVALUATION RESULTS ACROSS FOLDS")
    print("="*80)
    
    all_eval_results = {}
    
    # Load evaluation results from each fold
    for fold in range(1, num_folds + 1):
        eval_folder = Path(f"results/results_{dataset}_fold{fold}/feature_attributes/")
        
        if not eval_folder.exists():
            print(f"Warning: Evaluation folder not found for fold {fold}: {eval_folder}")
            continue
        
        # The evaluation script prints results but doesn't save them to a standard file
        # We need to check if there's a saved evaluation file or parse the output
        # For now, we'll look for any CSV files that might contain evaluation results
        print(f"Checking fold {fold} evaluation results...")
        
        # Note: The evaluation script currently just prints results
        # We would need to modify it to save results, or parse from stdout
        # For now, we'll create a placeholder structure
    
    # TODO: This requires modifying evaluate_feature_attribution_with_ground_truth.py
    # to save results to a file that we can aggregate here
    
    return all_eval_results


def print_timing_summary(aggregated_timing):
    """Print a formatted summary of aggregated timing results."""
    print("\n" + "="*80)
    print("AGGREGATED TIMING SUMMARY (Across All Folds)")
    print("="*80)
    
    for explainer_name, stats in aggregated_timing.items():
        if explainer_name == 'speedup':
            continue
        
        print(f"\n{explainer_name.upper()}:")
        print(f"  Wall-clock time: {stats['wall_clock_time']['mean']:.2f} ± {stats['wall_clock_time']['std']:.2f} seconds")
        print(f"    Range: [{stats['wall_clock_time']['min']:.2f}, {stats['wall_clock_time']['max']:.2f}] seconds")
        print(f"  CPU time: {stats['cpu_time']['mean']:.2f} ± {stats['cpu_time']['std']:.2f} seconds")
        print(f"    Range: [{stats['cpu_time']['min']:.2f}, {stats['cpu_time']['max']:.2f}] seconds")
        
        if 'gpu_time' in stats:
            print(f"  GPU time: {stats['gpu_time']['mean']:.2f} ± {stats['gpu_time']['std']:.2f} seconds")
        
        if 'per_query_wall_clock' in stats:
            print(f"  Per-query wall-clock: {stats['per_query_wall_clock']['mean']:.3f} ± {stats['per_query_wall_clock']['std']:.3f} seconds")
    
    if 'speedup' in aggregated_timing:
        speedup = aggregated_timing['speedup']
        print(f"\n{'='*80}")
        print("SPEEDUP (Baseline vs Adaptive):")
        print(f"  Wall-clock speedup: {speedup['wall_clock']['mean']:.2f}x ± {speedup['wall_clock']['std']:.2f}x")
        print(f"    Range: [{speedup['wall_clock']['min']:.2f}x, {speedup['wall_clock']['max']:.2f}x]")
        print(f"  CPU speedup: {speedup['cpu']['mean']:.2f}x ± {speedup['cpu']['std']:.2f}x")
        print(f"    Range: [{speedup['cpu']['min']:.2f}x, {speedup['cpu']['max']:.2f}x]")
        print("="*80)


def main():
    parser = argparse.ArgumentParser(description="Aggregate results across all folds")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["MSLR-WEB10K", "MQ2008"],
        help="The dataset to aggregate results for",
    )
    parser.add_argument(
        "--experiment_iteration",
        type=int,
        default=1,
        help="Experiment iteration number (default: 1)",
    )
    parser.add_argument(
        "--num_folds",
        type=int,
        default=5,
        help="Number of folds to aggregate (default: 5)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for aggregated results (default: results/results_{dataset}_aggregated/)",
    )
    
    args = parser.parse_args()
    print(args, flush=True)
    
    dataset = args.dataset
    experiment_iteration = args.experiment_iteration
    num_folds = args.num_folds
    
    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(f"results/results_{dataset}_aggregated/")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Aggregate timing results
    aggregated_timing = aggregate_timing_results(dataset, experiment_iteration, num_folds)
    
    if aggregated_timing:
        # Print summary
        print_timing_summary(aggregated_timing)
        
        # Save aggregated timing results
        timing_output_file = output_dir / f"aggregated_timing_iter{experiment_iteration}.json"
        with open(timing_output_file, 'w') as f:
            json.dump(aggregated_timing, f, indent=2)
        print(f"\nAggregated timing results saved to: {timing_output_file}")
        
        # Save summary as CSV for easy viewing
        summary_rows = []
        for explainer_name, stats in aggregated_timing.items():
            if explainer_name == 'speedup':
                continue
            summary_rows.append({
                'explainer': explainer_name,
                'wall_clock_mean': stats['wall_clock_time']['mean'],
                'wall_clock_std': stats['wall_clock_time']['std'],
                'cpu_mean': stats['cpu_time']['mean'],
                'cpu_std': stats['cpu_time']['std'],
            })
        
        if 'speedup' in aggregated_timing:
            summary_rows.append({
                'explainer': 'speedup',
                'wall_clock_mean': aggregated_timing['speedup']['wall_clock']['mean'],
                'wall_clock_std': aggregated_timing['speedup']['wall_clock']['std'],
                'cpu_mean': aggregated_timing['speedup']['cpu']['mean'],
                'cpu_std': aggregated_timing['speedup']['cpu']['std'],
            })
        
        summary_df = pd.DataFrame(summary_rows)
        summary_csv = output_dir / f"timing_summary_iter{experiment_iteration}.csv"
        summary_df.to_csv(summary_csv, index=False)
        print(f"Timing summary CSV saved to: {summary_csv}")
    else:
        print("ERROR: Could not aggregate timing results!")
        return 1
    
    # Aggregate evaluation results (if available)
    # Note: This requires evaluation script to save results to files
    aggregated_eval = aggregate_evaluation_results(dataset, num_folds)
    
    print("\n" + "="*80)
    print("AGGREGATION COMPLETE")
    print("="*80)
    print(f"Results saved to: {output_dir}")
    
    return 0


if __name__ == "__main__":
    exit(main())

