#!/usr/bin/env python3
"""
Generate Feature Attribution Explanations for BERT Text Ranking on MS MARCO.

This script generates RankingSHAP explanations for a BERT cross-encoder model
on the MS MARCO dataset. It's designed to run on the Snellius GPU cluster.

Usage:
    python generate_feature_attribution_explanations_bert_text.py \
        --split validation \
        --top_k 10 \
        --num_queries 250 \
        --model_name cross-encoder/ms-marco-MiniLM-L-6-v2

For Snellius, ensure you have:
    - A GPU node allocated (e.g., sbatch with --gpus=1)
    - The correct conda/virtual environment activated
    - Required packages: transformers, torch, shap, datasets
"""

import argparse
import json
import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import kendalltau

# Add parent directory to path if needed
sys.path.insert(0, str(Path(__file__).parent))

# Import our modules
from approaches.ranking_shap_bert_text import (
    RankingShapBertText,
    RankingShapBertTextWeighted,
)
from utils.bert_wrapper import BERTRankingWrapper, tokenize_and_stem
from utils.msmarco_loader import (
    sample_msmarco_queries,
)  # Use BM25 for initial retrieval


def setup_environment():
    """Setup environment variables for Snellius compatibility."""
    # Disable tokenizers parallelism warning
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Print GPU info
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(
            f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        )
    else:
        print("WARNING: CUDA not available, running on CPU (will be slow)")


def prepare_for_eval(path_to_attribute_values):
    """Convert attribution CSV to evaluation format."""
    experiment_results = pd.read_csv(path_to_attribute_values)
    experiment_results = experiment_results.set_index("feature_number")
    experiment_results = experiment_results.stack().swaplevel().sort_index()
    experiment_results = experiment_results.reset_index().rename(
        columns={"level_0": "query_number", 0: "attribution_value"}
    )
    experiment_results = experiment_results.set_index(
        ["query_number", "feature_number"]
    )
    eval_path = Path(str(path_to_attribute_values).split(".")[0] + "_eval.csv")
    experiment_results.to_csv(eval_path)
    print(f"Saved eval format to: {eval_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run RankingSHAP for BERT Text Ranking"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        help="MS MARCO split: train or validation",
    )
    parser.add_argument(
        "--top_k", type=int, default=10, help="Number of docs to explain per query"
    )
    parser.add_argument(
        "--num_queries", type=int, default=250, help="Number of queries to explain"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        help="HuggingFace model name for cross-encoder",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for BERT inference"
    )
    parser.add_argument(
        "--retrieval_depth",
        type=int,
        default=100,
        help="Number of docs to retrieve with BM25 before BERT re-ranking",
    )
    parser.add_argument(
        "--nsamples",
        type=str,
        default="auto",
        help="Number of SHAP samples ('auto' or integer)",
    )
    parser.add_argument(
        "--explanation_size",
        type=int,
        default=10,
        help="Number of top features for selection explanation",
    )
    parser.add_argument(
        "--objective",
        type=str,
        choices=["kendall", "weighted"],
        default="kendall",
        help="Ranking similarity objective",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use ('cuda', 'cpu', or None for auto)",
    )

    args = parser.parse_args()

    # Setup
    setup_environment()

    # Parse nsamples
    nsamples = args.nsamples if args.nsamples == "auto" else int(args.nsamples)

    # Setup paths
    output_path = Path("results/results_MSMARCO_BERT/feature_attributes/")
    output_path.mkdir(parents=True, exist_ok=True)

    experiment_tag = f"q{args.num_queries}_top{args.top_k}"
    model_short = args.model_name.split("/")[-1]
    query_data_file = (
        output_path / f"query_data_bert_{model_short}_{experiment_tag}.jsonl"
    )

    # Clear previous query data
    if query_data_file.exists():
        query_data_file.unlink()

    print("=" * 60)
    print("RankingSHAP BERT Text Experiment")
    print("=" * 60)
    print(f"Model: {args.model_name}")
    print(f"Split: {args.split}")
    print(f"Num queries: {args.num_queries}")
    print(f"Top K (explanation depth): {args.top_k}")
    print(f"Retrieval depth (BM25): {args.retrieval_depth}")
    print(f"Objective: {args.objective}")
    print(f"SHAP samples: {nsamples}")
    print(f"Output: {output_path}")
    print("=" * 60)

    # Determine device
    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Define similarity coefficient
    if args.objective == "kendall":
        rank_similarity_coefficient = lambda x, y: kendalltau(x, y)[0]
        explainer_name = f"rankingshap_bert_{model_short}_{experiment_tag}"
    else:
        # Weighted objective defined in the explainer class
        explainer_name = f"rankingshap_bert_weighted_{model_short}_{experiment_tag}"

    print(f"\n>>> Sampling {args.num_queries} queries from MS MARCO ({args.split})...")

    # Sample queries using BM25 for initial retrieval
    try:
        sampled_queries = sample_msmarco_queries(
            num_queries=args.num_queries,
            split=args.split,
            top_k=args.retrieval_depth,  # Retrieve more for IDF context
            seed=args.seed,
        )
    except Exception as exc:
        print(f"Failed to sample queries: {exc}")
        sys.exit(1)

    print(f"Successfully sampled {len(sampled_queries)} queries")

    # Initialize BERT model (once, reused for all queries)
    print(f"\n>>> Loading BERT model: {args.model_name}")

    # We'll initialize the wrapper with a dummy corpus first
    # and update it per query
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name)
    model.to(device)
    model.eval()

    print("BERT model loaded successfully!")

    # Process each query
    print(f"\n>>> Processing {len(sampled_queries)} queries...")

    for query_idx, sampled_query in enumerate(sampled_queries):
        source_query_id = sampled_query["query_id"]
        query_text = sampled_query["query_text"]
        docs_full = sampled_query["documents"]

        print(f"\nQuery {query_idx + 1}/{len(sampled_queries)} (ID: {source_query_id})")
        print(f"  Query: {query_text[:80]}...")
        print(f"  Retrieved docs: {len(docs_full)}")

        if len(docs_full) < 2:
            print("  Skipping: insufficient documents")
            continue

        # Step 1: Use BERT to re-rank all retrieved docs and select top_k
        print(f"  Re-ranking with BERT...")

        with torch.no_grad():
            all_scores = []
            for i in range(0, len(docs_full), args.batch_size):
                batch_docs = docs_full[i : i + args.batch_size]
                inputs = tokenizer(
                    [query_text] * len(batch_docs),
                    batch_docs,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = model(**inputs)
                batch_scores = outputs.logits.squeeze(-1).cpu().numpy()
                if batch_scores.ndim == 0:
                    batch_scores = np.array([batch_scores.item()])
                all_scores.extend(batch_scores.tolist())

        # Select top_k docs
        sorted_indices = np.argsort(all_scores)[::-1]
        top_k_indices = sorted_indices[: args.top_k]
        top_k_docs = [docs_full[i] for i in top_k_indices]

        print(f"  Selected top {len(top_k_docs)} docs for explanation")

        # Save query data for evaluation
        query_record = {
            "query_id": query_idx,
            "dataset_query_id": source_query_id,
            "query_text": query_text,
            "documents": top_k_docs,
        }
        with open(query_data_file, "a") as f:
            f.write(json.dumps(query_record) + "\n")

        # Step 2: Build vocabulary for explanation
        vocabulary = sorted(
            list(set([word for doc in top_k_docs for word in tokenize_and_stem(doc)]))
        )
        num_features = len(vocabulary)
        print(f"  Vocabulary size: {num_features}")

        if num_features == 0:
            print("  Skipping: empty vocabulary")
            continue

        # Step 3: Build binary feature matrix
        feature_matrix = []
        for doc in top_k_docs:
            doc_words = set(tokenize_and_stem(doc))
            row = [1 if w in doc_words else 0 for w in vocabulary]
            feature_matrix.append(row)
        feature_matrix = np.array(feature_matrix)

        # Step 4: Create BERT wrapper for this query
        bert_wrapper = BERTRankingWrapper(
            corpus_passages=top_k_docs,
            model_name=args.model_name,
            device=device,
            batch_size=args.batch_size,
        )
        bert_wrapper.set_query(query_text, vocabulary)

        # Step 5: Initialize explainer
        if args.objective == "kendall":
            explainer = RankingShapBertText(
                original_model=bert_wrapper.predict,
                explanation_size=args.explanation_size,
                name=explainer_name,
                nsample_permutations=nsamples,
                rank_similarity_coefficient=rank_similarity_coefficient,
            )
        else:
            explainer = RankingShapBertTextWeighted(
                original_model=bert_wrapper.predict,
                explanation_size=args.explanation_size,
                name=explainer_name,
                nsample_permutations=nsamples,
            )

        # Step 6: Generate explanations
        print(f"  Running RankingSHAP...")

        try:
            selection, attribution = explainer.get_query_explanation(
                query_features=feature_matrix, query_id=query_idx
            )

            # Save results
            save_file = output_path / f"{explainer_name}.csv"
            attribution.safe_to_file(save_file)

            print(
                f"  Done! Top 3 features: {[vocabulary[i] for i, _ in attribution.explanation[:3] if i < len(vocabulary)]}"
            )

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback

            traceback.print_exc()
            continue

    # Convert to eval format
    print("\n>>> Converting to eval format...")
    save_file = output_path / f"{explainer_name}.csv"
    if save_file.exists():
        prepare_for_eval(save_file)

    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
