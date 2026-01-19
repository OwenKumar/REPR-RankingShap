#!/usr/bin/env python3
"""
Runner script for BERT Text RankingSHAP Experiments.

This script orchestrates the generation and evaluation of BERT-based
RankingSHAP explanations. It can run locally or be used as a template
for SLURM jobs on Snellius.

Usage:
    # Basic run
    python run_bert_text_experiment.py --num_queries 50 --top_k 10

    # Multiple top_k values
    python run_bert_text_experiment.py --num_queries 100 --top_k 10 20

    # Different model
    python run_bert_text_experiment.py --model cross-encoder/ms-marco-TinyBERT-L-2-v2
"""

import subprocess
import sys
import argparse
from pathlib import Path
import time
import os


def run_command(command, description=""):
    """Run a command and handle output."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(command)}")
    print("=" * 60)

    start_time = time.time()

    # Run with real-time output
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )

    # Stream output
    for line in process.stdout:
        print(line, end="")

    process.wait()
    elapsed = time.time() - start_time

    if process.returncode != 0:
        print(f"\n❌ FAILED after {elapsed:.1f}s (exit code {process.returncode})")
        return False
    else:
        print(f"\n✅ Completed in {elapsed:.1f}s")
        return True


def check_gpu():
    """Check if GPU is available."""
    try:
        import torch

        if torch.cuda.is_available():
            print(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
            print(
                f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
            )
            return True
        else:
            print("⚠️  No GPU detected, will run on CPU (slow)")
            return False
    except ImportError:
        print("⚠️  PyTorch not installed, cannot check GPU")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run BERT Text RankingSHAP Experiments"
    )
    parser.add_argument(
        "--num_queries", type=int, default=50, help="Number of queries to process"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        nargs="+",
        default=[10],
        help="Top K values to test (e.g., --top_k 10 20 100)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        help="MS MARCO split (train/validation)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        help="HuggingFace model name",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for BERT inference"
    )
    parser.add_argument(
        "--nsamples", type=str, default="auto", help="SHAP samples ('auto' or integer)"
    )
    parser.add_argument(
        "--skip_generation",
        action="store_true",
        help="Skip generation, only run evaluation",
    )
    parser.add_argument(
        "--skip_evaluation",
        action="store_true",
        help="Skip evaluation, only run generation",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("🚀 RankingSHAP BERT Text Experiment Runner")
    print("=" * 60)
    print(f"Model:       {args.model}")
    print(f"Queries:     {args.num_queries}")
    print(f"Top K:       {args.top_k}")
    print(f"Split:       {args.split}")
    print(f"Seed:        {args.seed}")
    print("=" * 60)

    # Check GPU
    check_gpu()

    # Ensure output directories exist
    results_dir = Path("results/results_MSMARCO_BERT/feature_attributes")
    results_dir.mkdir(parents=True, exist_ok=True)

    model_short = args.model.split("/")[-1]

    # Process each top_k value
    for k in args.top_k:
        print(f"\n\n>>> Processing Top K = {k} <<<")

        experiment_tag = f"q{args.num_queries}_top{k}"

        # Clean up previous results
        output_file = (
            results_dir / f"rankingshap_bert_{model_short}_{experiment_tag}.csv"
        )
        eval_file = (
            results_dir / f"rankingshap_bert_{model_short}_{experiment_tag}_eval.csv"
        )
        query_data_file = (
            results_dir / f"query_data_bert_{model_short}_{experiment_tag}.jsonl"
        )

        if not args.skip_generation:
            # Remove previous files
            for f in [output_file, eval_file, query_data_file]:
                if f.exists():
                    print(f"Removing: {f}")
                    f.unlink()

            # Step 1: Generate explanations
            gen_cmd = [
                sys.executable,
                "generate_feature_attribution_explanations_bert_text.py",
                "--split",
                args.split,
                "--top_k",
                str(k),
                "--num_queries",
                str(args.num_queries),
                "--model_name",
                args.model,
                "--seed",
                str(args.seed),
                "--batch_size",
                str(args.batch_size),
                "--nsamples",
                args.nsamples,
                "--objective",
                "kendall",
            ]

            success = run_command(gen_cmd, f"Generating BERT Explanations (Top {k})")
            if not success:
                print(f"❌ Generation failed for Top K = {k}")
                continue

        if not args.skip_evaluation:
            # Step 2: Evaluate fidelity
            eval_cmd = [
                sys.executable,
                "evaluate_rankingshap_bert_text_fidelity.py",
                "--split",
                args.split,
                "--top_k",
                str(k),
                "--num_queries",
                str(args.num_queries),
                "--model_name",
                args.model,
            ]

            success = run_command(eval_cmd, f"Evaluating Fidelity (Top {k})")
            if not success:
                print(f"❌ Evaluation failed for Top K = {k}")

    print("\n" + "=" * 60)
    print("🎉 All experiments completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
