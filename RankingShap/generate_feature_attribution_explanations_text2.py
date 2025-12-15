"""
Generate RankingSHAP explanations for MS MARCO text data.

Uses locally cached data (from download_msmarco_data.py).
Uses the correct RankingShapText interface with get_query_explanation().

Usage:
    # First download data once:
    python download_msmarco_data.py --num_queries 250 --num_docs 100

    # Then run experiments (fast, no network):
    python generate_feature_attribution_explanations_text2.py --num_queries 250 --num_docs 100
"""

import argparse
import json
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

warnings.filterwarnings("ignore")

from approaches.ranking_shap_text import RankingShapText
from utils.bm25_wrapper import BM25Wrapper, tokenize_and_stem


def load_local_data(data_file: Path, num_queries: int = None) -> list:
    """Load data from local JSONL file."""
    data = []
    with open(data_file, "r") as f:
        for line in f:
            data.append(json.loads(line))
            if num_queries and len(data) >= num_queries:
                break
    return data


def generate_explanations(
    num_queries: int = 250,
    num_docs: int = 100,
    nsamples: int = 5000,
    seed: int = 42,
    data_dir: str = "data",
):
    """
    Generate RankingSHAP feature attributions for text ranking.
    """
    random.seed(seed)
    np.random.seed(seed)

    # Find data file
    data_path = Path(data_dir)
    data_file = None

    # Try exact match first
    exact_file = data_path / f"msmarco_q{num_queries}_docs{num_docs}.jsonl"
    if exact_file.exists():
        data_file = exact_file
    else:
        # Look for any file with enough queries
        candidates = list(data_path.glob("msmarco_q*_docs*.jsonl"))
        for c in candidates:
            with open(c) as f:
                count = sum(1 for _ in f)
            if count >= num_queries:
                data_file = c
                print(f"Using data file: {data_file} ({count} queries available)")
                break

    if data_file is None or not data_file.exists():
        print(f"Error: No suitable data file found in {data_dir}/")
        print("\nPlease run first:")
        print(
            f"  python download_msmarco_data.py --num_queries {num_queries} --num_docs {num_docs}"
        )
        return

    # Load data
    print("=" * 50)
    print("Loading local MS MARCO data")
    print("=" * 50)
    print(f"  Data file: {data_file}")

    data = load_local_data(data_file, num_queries)
    print(f"  Loaded {len(data)} queries")

    if not data:
        print("Error: No data loaded!")
        return

    # Verify document count
    actual_docs = len(data[0]["documents"])
    if actual_docs < num_docs:
        print(f"  Warning: Data has {actual_docs} docs per query, requested {num_docs}")
        num_docs = actual_docs

    # Setup output paths
    experiment_tag = f"q{len(data)}_docs{num_docs}"
    results_dir = Path("results/results_MSMARCO/feature_attributes")
    results_dir.mkdir(parents=True, exist_ok=True)

    output_file = results_dir / f"rankingshap_text_bm25_{experiment_tag}.csv"
    query_data_file = results_dir / f"query_data_{experiment_tag}.jsonl"

    # Clear previous results
    if output_file.exists():
        output_file.unlink()
    if query_data_file.exists():
        query_data_file.unlink()

    print(f"\nProcessing {len(data)} queries with {num_docs} docs each")
    print(f"SHAP samples: {nsamples}")
    print(f"Output: {output_file}")
    print("=" * 50)

    # Define rank similarity metric
    rank_similarity_coefficient = lambda x, y: kendalltau(x, y)[0]

    # Initialize explainer (will update model per query)
    explainer = RankingShapText(
        original_model=None,
        explanation_size=10,
        name=f"rankingshap_text_bm25_{experiment_tag}",
        nsample_permutations=nsamples,
        rank_similarity_coefficient=rank_similarity_coefficient,
    )

    for i, item in enumerate(data):
        query_id = item["query_id"]
        query_text = item["query_text"]
        docs = item["documents"][:num_docs]

        print(f"\nQuery {i+1}/{len(data)} (ID: {query_id})")
        print(f"  Query: {query_text[:60]}...")

        # Build vocabulary from ALL documents
        vocabulary = sorted(
            list(set([word for doc in docs for word in tokenize_and_stem(doc)]))
        )

        print(f"  Documents: {len(docs)}")
        print(f"  Vocabulary size: {len(vocabulary)}")

        if len(vocabulary) == 0:
            print("  Skipping: empty vocabulary")
            continue

        # Build feature matrix for all documents
        feature_matrix = []
        for doc in docs:
            doc_words = set(tokenize_and_stem(doc))
            row = [1 if w in doc_words else 0 for w in vocabulary]
            feature_matrix.append(row)
        feature_matrix = np.array(feature_matrix)

        # Initialize BM25 wrapper
        model = BM25Wrapper(docs)
        model.set_query(query_text, vocabulary)

        # Get original ranking
        orig_scores = model.predict(feature_matrix)
        orig_ranking = np.argsort(orig_scores)[::-1].tolist()

        print(f"  Original ranking computed")

        # Update explainer with current model
        explainer.original_model = model.predict

        # Run RankingSHAP using the correct interface
        try:
            selection, attribution = explainer.get_query_explanation(
                query_features=feature_matrix,
                query_id=query_id,
            )
        except Exception as e:
            print(f"  Error in SHAP: {e}")
            import traceback

            traceback.print_exc()
            continue

        # Save query data for evaluation
        query_record = {
            "query_id": query_id,
            "original_query_id": item.get("original_query_id", query_id),
            "query_text": query_text,
            "documents": docs,
            "doc_ids": item.get("doc_ids", []),
            "vocabulary": vocabulary,
            "original_ranking": orig_ranking,
        }

        with open(query_data_file, "a") as f:
            f.write(json.dumps(query_record) + "\n")

        # Save attributions using the attribution object's method
        print(f"  Writing to {output_file}")
        attribution.safe_to_file(output_file)

        print(f"  Attribution saved")

    # Convert to eval format
    print("\n" + "=" * 50)
    print("Converting to eval format...")

    if output_file.exists():
        _convert_to_eval_format(output_file, results_dir, experiment_tag)
    else:
        print("Warning: No output file created - no successful explanations generated")

    print("Done!")


def _convert_to_eval_format(output_file: Path, results_dir: Path, experiment_tag: str):
    """Convert wide format to long format for evaluation."""
    df = pd.read_csv(output_file)
    df = df.set_index("feature_number")

    df_long = df.stack().reset_index()
    df_long.columns = ["feature_number", "query_number", "attribution_value"]
    df_long["query_number"] = df_long["query_number"].astype(int)
    df_long = df_long[["query_number", "feature_number", "attribution_value"]]
    df_long = df_long.set_index(["query_number", "feature_number"])

    eval_file = results_dir / f"rankingshap_text_bm25_{experiment_tag}_eval.csv"
    df_long.to_csv(eval_file)
    print(f"Saved eval format to: {eval_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate RankingSHAP explanations (local data)"
    )
    parser.add_argument("--num_queries", type=int, default=250)
    parser.add_argument("--num_docs", type=int, default=100)
    parser.add_argument("--nsamples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_dir", type=str, default="data")
    args = parser.parse_args()

    generate_explanations(
        num_queries=args.num_queries,
        num_docs=args.num_docs,
        nsamples=args.nsamples,
        seed=args.seed,
        data_dir=args.data_dir,
    )


if __name__ == "__main__":
    main()
