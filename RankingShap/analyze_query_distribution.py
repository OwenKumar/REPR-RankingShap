"""
Analyze the distribution of documents per query in MQ2008 dataset.
This helps determine if the 300 sample cap is limiting performance.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from utils.helper_functions import get_data, get_queryids_as_list, get_documents_per_query

def analyze_dataset(dataset="MQ2008", fold=1, split="test"):
    """Analyze document distribution for a dataset."""
    
    print(f"\n{'='*80}")
    print(f"Analyzing {dataset} - Fold {fold} - {split} split")
    print(f"{'='*80}\n")
    
    # Load data
    data_directory = Path(f"data/{dataset}/Fold{fold}/")
    if split == "test":
        data_file = data_directory / "test.txt"
    elif split == "vali":
        data_file = data_directory / "vali.txt"
    else:
        data_file = data_directory / "train.txt"
    
    if not data_file.exists():
        print(f"Error: File not found: {data_file}")
        return None
    
    eval_data = get_data(data_file=data_file)
    EX, _, Eqids = eval_data
    
    # Get document counts per query
    qid_count_list = get_documents_per_query(Eqids)
    doc_counts = qid_count_list.values
    
    # Calculate statistics
    print("Document Distribution Statistics:")
    print(f"  Total queries: {len(doc_counts)}")
    print(f"  Mean documents per query: {np.mean(doc_counts):.2f}")
    print(f"  Median documents per query: {np.median(doc_counts):.2f}")
    print(f"  Min documents per query: {np.min(doc_counts)}")
    print(f"  Max documents per query: {np.max(doc_counts)}")
    print(f"  Std documents per query: {np.std(doc_counts):.2f}")
    
    # Calculate percentiles
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    print(f"\nPercentiles:")
    for p in percentiles:
        val = np.percentile(doc_counts, p)
        print(f"  {p}th percentile: {val:.1f} documents")
    
    # Analyze adaptive sampling impact
    print(f"\n{'='*80}")
    print("Adaptive Sampling Analysis (25 * sqrt(n_docs), cap at 300):")
    print(f"{'='*80}\n")
    
    base = 25
    cap = 300
    
    # Calculate samples for each query
    samples_with_cap = np.minimum(base * np.sqrt(doc_counts), cap)
    samples_without_cap = base * np.sqrt(doc_counts)
    
    # Find queries hitting the cap
    queries_hitting_cap = np.sum(samples_with_cap >= cap)
    queries_hitting_cap_pct = (queries_hitting_cap / len(doc_counts)) * 100
    
    print(f"Queries hitting the cap (300 samples):")
    print(f"  Count: {queries_hitting_cap} / {len(doc_counts)}")
    print(f"  Percentage: {queries_hitting_cap_pct:.2f}%")
    
    # Find threshold (where cap kicks in)
    # 25 * sqrt(n_docs) = 300
    # sqrt(n_docs) = 12
    # n_docs = 144
    threshold_docs = (cap / base) ** 2
    print(f"\nCap threshold: Queries with >{threshold_docs:.0f} documents hit the cap")
    
    queries_above_threshold = np.sum(doc_counts > threshold_docs)
    print(f"  Queries above threshold: {queries_above_threshold} ({queries_above_threshold/len(doc_counts)*100:.2f}%)")
    
    # Calculate average samples
    avg_samples_with_cap = np.mean(samples_with_cap)
    avg_samples_without_cap = np.mean(samples_without_cap)
    
    print(f"\nAverage samples per query:")
    print(f"  With cap (300): {avg_samples_with_cap:.2f} samples")
    print(f"  Without cap: {avg_samples_without_cap:.2f} samples")
    print(f"  Difference: {avg_samples_without_cap - avg_samples_with_cap:.2f} samples ({((avg_samples_without_cap - avg_samples_with_cap) / avg_samples_with_cap * 100):.2f}% increase)")
    
    # Show examples
    print(f"\n{'='*80}")
    print("Sample Queries (showing impact of cap):")
    print(f"{'='*80}\n")
    print(f"{'Query Size':<15} {'Samples (capped)':<20} {'Samples (no cap)':<20} {'Difference':<15}")
    print("-" * 70)
    
    # Show some examples
    example_sizes = sorted(set([int(np.min(doc_counts)), int(np.median(doc_counts)), 
                                int(np.max(doc_counts)), int(threshold_docs)]))
    for size in example_sizes:
        if size <= np.max(doc_counts):
            capped = min(base * np.sqrt(size), cap)
            uncapped = base * np.sqrt(size)
            diff = uncapped - capped
            print(f"{size:<15} {capped:<20.1f} {uncapped:<20.1f} {diff:<15.1f}")
    
    # Distribution summary
    print(f"\n{'='*80}")
    print("Distribution Summary:")
    print(f"{'='*80}\n")
    
    bins = [0, 10, 25, 50, 100, 144, 200, 300, np.inf]
    bin_labels = ['0-10', '10-25', '25-50', '50-100', '100-144', '144-200', '200-300', '300+']
    
    for i in range(len(bins)-1):
        count = np.sum((doc_counts >= bins[i]) & (doc_counts < bins[i+1]))
        pct = (count / len(doc_counts)) * 100
        if count > 0:
            print(f"  {bin_labels[i]:<12} docs: {count:>4} queries ({pct:>5.2f}%)")
    
    return {
        'doc_counts': doc_counts,
        'samples_with_cap': samples_with_cap,
        'samples_without_cap': samples_without_cap,
        'queries_hitting_cap': queries_hitting_cap,
        'avg_samples_with_cap': avg_samples_with_cap,
        'avg_samples_without_cap': avg_samples_without_cap,
    }

def analyze_from_timing_file(timing_file_path):
    """Analyze document distribution from timing file if it has per-query data."""
    import json
    
    print(f"\n{'='*80}")
    print(f"Analyzing from timing file: {timing_file_path}")
    print(f"{'='*80}\n")
    
    with open(timing_file_path, 'r') as f:
        timing_data = json.load(f)
    
    # Look for per-query timing data
    doc_counts = []
    for result in timing_data:
        if 'per_query_timing' in result:
            for query_data in result['per_query_timing']:
                if 'num_documents' in query_data:
                    doc_counts.append(query_data['num_documents'])
    
    if not doc_counts:
        print("No per-query document counts found in timing file.")
        print("Trying to analyze from dataset directly...")
        return None
    
    doc_counts = np.array(doc_counts)
    
    # Calculate statistics (same as analyze_dataset)
    print("Document Distribution Statistics (from timing file):")
    print(f"  Total queries: {len(doc_counts)}")
    print(f"  Mean documents per query: {np.mean(doc_counts):.2f}")
    print(f"  Median documents per query: {np.median(doc_counts):.2f}")
    print(f"  Min documents per query: {np.min(doc_counts)}")
    print(f"  Max documents per query: {np.max(doc_counts)}")
    print(f"  Std documents per query: {np.std(doc_counts):.2f}")
    
    # Rest of analysis (same as analyze_dataset)
    base = 25
    cap = 300
    samples_with_cap = np.minimum(base * np.sqrt(doc_counts), cap)
    samples_without_cap = base * np.sqrt(doc_counts)
    queries_hitting_cap = np.sum(samples_with_cap >= cap)
    threshold_docs = (cap / base) ** 2
    
    print(f"\nQueries hitting the cap (300 samples): {queries_hitting_cap} / {len(doc_counts)} ({queries_hitting_cap/len(doc_counts)*100:.2f}%)")
    print(f"Cap threshold: Queries with >{threshold_docs:.0f} documents hit the cap")
    
    avg_samples_with_cap = np.mean(samples_with_cap)
    avg_samples_without_cap = np.mean(samples_without_cap)
    
    print(f"\nAverage samples per query:")
    print(f"  With cap (300): {avg_samples_with_cap:.2f} samples")
    print(f"  Without cap: {avg_samples_without_cap:.2f} samples")
    print(f"  Difference: {avg_samples_without_cap - avg_samples_with_cap:.2f} samples ({((avg_samples_without_cap - avg_samples_with_cap) / avg_samples_with_cap * 100):.2f}% increase)")
    
    return {
        'doc_counts': doc_counts,
        'queries_hitting_cap': queries_hitting_cap,
        'avg_samples_with_cap': avg_samples_with_cap,
        'avg_samples_without_cap': avg_samples_without_cap,
    }


if __name__ == "__main__":
    import sys
    
    # Try to analyze from timing file if provided
    if len(sys.argv) > 1:
        timing_file = sys.argv[1]
        results = analyze_from_timing_file(timing_file)
    else:
        # Analyze test split (what's typically used for evaluation)
        results = analyze_dataset(dataset="MQ2008", fold=1, split="test")
    
    if results:
        print(f"\n{'='*80}")
        print("RECOMMENDATION:")
        print(f"{'='*80}\n")
        
        if results['queries_hitting_cap'] / len(results['doc_counts']) > 0.1:
            print(f"⚠️  {results['queries_hitting_cap']/len(results['doc_counts'])*100:.1f}% of queries hit the cap")
            print(f"   Consider removing or increasing the cap for better accuracy")
        else:
            print(f"✓ Only {results['queries_hitting_cap']/len(results['doc_counts'])*100:.1f}% of queries hit the cap")
            print(f"   The cap is probably not limiting performance significantly")
        
        print(f"\nAverage samples would increase by {((results['avg_samples_without_cap'] - results['avg_samples_with_cap']) / results['avg_samples_with_cap'] * 100):.1f}% without cap")
        print(f"This would improve accuracy but slow down computation")
    else:
        print("\nTo run this analysis:")
        print("1. On Snellius: python analyze_query_distribution.py")
        print("2. From timing file: python analyze_query_distribution.py <path_to_timing_file.json>")

