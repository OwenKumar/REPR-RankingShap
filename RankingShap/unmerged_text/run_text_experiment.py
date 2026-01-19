"""
Run RankingSHAP text experiment using locally cached MS MARCO data.

Usage:
    # Step 1: Download data once (requires internet)
    python download_msmarco_data.py --num_queries 250 --num_docs 100

    # Step 2: Run experiments (offline, fast)
    python run_text_experiment.py --num_queries 250 --num_docs 100

    # Quick test
    python run_text_experiment.py --num_queries 10 --num_docs 100 --nsamples 1000
"""

import subprocess
import sys
import argparse
from pathlib import Path
import time


def run_command(command):
    """Run a command and handle output."""
    print(f"Running: {' '.join(command)}")

    start_time = time.time()
    result = subprocess.run(command, capture_output=True, text=True)
    elapsed = time.time() - start_time

    if result.returncode != 0:
        print("❌ Command failed:")
        print(result.stderr)
        print(result.stdout)
        sys.exit(1)
    else:
        print(f"✅ Completed in {elapsed:.1f}s")
        lines = result.stdout.strip().split("\n")
        if len(lines) > 30:
            print("... (output truncated) ...")
            print("\n".join(lines[-30:]))
        else:
            print(result.stdout)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Run RankingSHAP text experiment (local data)"
    )
    parser.add_argument("--num_queries", type=int, default=250)
    parser.add_argument("--num_docs", type=int, default=100)
    parser.add_argument("--top_k", type=int, nargs="+", default=[10, 20, 100])
    parser.add_argument("--nsamples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_dir", type=str, default="data")
    args = parser.parse_args()

    experiment_tag = f"q{args.num_queries}_docs{args.num_docs}"

    # Check if data exists
    data_file = (
        Path(args.data_dir) / f"msmarco_q{args.num_queries}_docs{args.num_docs}.jsonl"
    )
    if not data_file.exists():
        # Try to find any suitable data file
        data_path = Path(args.data_dir)
        if data_path.exists():
            candidates = list(data_path.glob("msmarco_q*_docs*.jsonl"))
            if candidates:
                print(
                    f"Note: Exact data file not found, but found: {[c.name for c in candidates]}"
                )
            else:
                print(f"❌ No data files found in {args.data_dir}/")
                print("\nPlease download data first:")
                print(
                    f"  python download_msmarco_data.py --num_queries {args.num_queries} --num_docs {args.num_docs}"
                )
                sys.exit(1)
        else:
            print(f"❌ Data directory not found: {args.data_dir}/")
            print("\nPlease download data first:")
            print(
                f"  python download_msmarco_data.py --num_queries {args.num_queries} --num_docs {args.num_docs}"
            )
            sys.exit(1)

    print("=" * 70)
    print("🚀 RankingSHAP Text Experiment (Local Data)")
    print("=" * 70)
    print(f"   Queries:        {args.num_queries}")
    print(f"   Docs/query:     {args.num_docs}")
    print(f"   Top-K (eval):   {args.top_k}")
    print(f"   SHAP samples:   {args.nsamples}")
    print(f"   Data dir:       {args.data_dir}")
    print("=" * 70)

    # Ensure output directories exist
    Path("results/results_MSMARCO/feature_attributes").mkdir(
        parents=True, exist_ok=True
    )
    Path("results/results_MSMARCO_fidelity").mkdir(parents=True, exist_ok=True)

    # Step 1: Generate explanations
    print("\n" + "=" * 70)
    print("STEP 1: Generating RankingSHAP Explanations")
    print("=" * 70)

    gen_cmd = [
        sys.executable,
        "generate_feature_attribution_explanations_text2.py",
        "--num_queries",
        str(args.num_queries),
        "--num_docs",
        str(args.num_docs),
        "--nsamples",
        str(args.nsamples),
        "--seed",
        str(args.seed),
        "--data_dir",
        args.data_dir,
    ]
    run_command(gen_cmd)

    # Step 2: Evaluate
    print("\n" + "=" * 70)
    print("STEP 2: Evaluating Fidelity")
    print("=" * 70)

    eval_cmd = [
        sys.executable,
        "evaluate_rankingshap_text_fidelity2.py",
        "--num_docs",
        str(args.num_docs),
        "--num_queries",
        str(args.num_queries),
        "--top_k",
    ] + [str(k) for k in args.top_k]

    run_command(eval_cmd)

    # Summary
    results_dir = Path("results/results_MSMARCO/feature_attributes")
    fidelity_dir = Path("results/results_MSMARCO_fidelity")

    print("\n" + "=" * 70)
    print("🎉 Experiment Complete!")
    print("=" * 70)
    print("\nOutput files:")
    print(
        f"  - Attributions: {results_dir / f'rankingshap_text_bm25_{experiment_tag}.csv'}"
    )
    print(f"  - Query data:   {results_dir / f'query_data_{experiment_tag}.jsonl'}")
    print(f"  - Fidelity:     {fidelity_dir / f'fidelity_{experiment_tag}.csv'}")


if __name__ == "__main__":
    main()
