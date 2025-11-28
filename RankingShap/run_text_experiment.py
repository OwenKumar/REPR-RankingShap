import subprocess
import sys
import argparse
from pathlib import Path
import time


def run_command(command):
    print(f"Running: {' '.join(command)}")
    start_time = time.time()
    result = subprocess.run(command, capture_output=True, text=True)
    end_time = time.time()

    if result.returncode != 0:
        print("❌ Command failed:")
        print(result.stderr)
        sys.exit(1)
    else:
        print(f"✅ Command successful ({end_time - start_time:.2f}s)")
        # Print only the last few lines of stdout to keep it clean, or all if it's short
        lines = result.stdout.splitlines()
        if len(lines) > 20:
            print("... (output truncated) ...")
            print("\n".join(lines[-20:]))
        else:
            print(result.stdout)


def main():
    parser = argparse.ArgumentParser(description="Run RankingSHAP Text Experiment")
    parser.add_argument(
        "--num_queries", type=int, default=50, help="Number of queries to process"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        nargs="+",
        default=[10],
        help="List of Top K values to run (e.g. 10 20 100)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        help="MS MARCO split (train/validation)",
    )
    args = parser.parse_args()

    print("========================================")
    print(f"🚀 Starting RankingSHAP Text Experiment")
    print(f"   Queries: {args.num_queries}")
    print(f"   Top K:   {args.top_k}")
    print(f"   Split:   {args.split}")
    print("========================================")

    # Ensure directories exist
    results_dir = Path("results/results_MSMARCO/feature_attributes")
    results_dir.mkdir(parents=True, exist_ok=True)

    for k in args.top_k:
        print(f"\n\n>>> Processing Top K = {k} <<<")

        experiment_tag = f"q{args.num_queries}_top{k}"

        # Clean up previous results for this specific K
        output_file = results_dir / f"rankingshap_text_bm25_{experiment_tag}.csv"
        eval_file = results_dir / f"rankingshap_text_bm25_{experiment_tag}_eval.csv"
        query_data_file = results_dir / f"query_data_{experiment_tag}.jsonl"

        if output_file.exists():
            print(f"Removing previous result file: {output_file}")
            output_file.unlink()
        if eval_file.exists():
            print(f"Removing previous eval file: {eval_file}")
            eval_file.unlink()
        if query_data_file.exists():
            print(f"Removing previous query data file: {query_data_file}")
            query_data_file.unlink()

        # 1. Run Generation
        print(f"\n--- Step 1: Generating Explanations (Top {k}) ---")
        gen_cmd = [
            sys.executable,
            "generate_feature_attribution_explanations_text.py",
            "--split",
            args.split,
            "--top_k",
            str(k),
            "--num_queries",
            str(args.num_queries),
        ]
        run_command(gen_cmd)

        # 2. Run Evaluation
        print(f"\n--- Step 2: Evaluating Fidelity (Top {k}) ---")
        eval_cmd = [
            sys.executable,
            "evaluate_rankshap_text_fidelity.py",
            "--split",
            args.split,
            "--top_k",
            str(k),
            "--num_queries",
            str(args.num_queries),
        ]
        run_command(eval_cmd)

    print("\n🎉 All Experiments Completed!")


if __name__ == "__main__":
    main()
