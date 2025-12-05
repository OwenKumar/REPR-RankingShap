import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

# Import the new Text Explainer
from approaches.ranking_shap_text import RankingShapText

# Import your custom utils (You need to create these!)
from utils.bm25_wrapper import BM25Wrapper, tokenize_and_stem
from utils.msmarco_loader import sample_msmarco_queries

parser = argparse.ArgumentParser(description="Run RankingSHAP on Text (Hugging Face)")
parser.add_argument(
    "--split",
    type=str,
    default="validation",
    help="MS MARCO split: train or validation",
)
parser.add_argument(
    "--top_k", type=int, default=10, help="Number of docs to retrieve per query"
)
parser.add_argument(
    "--num_queries", type=int, default=250, help="Number of queries to explain"
)
parser.add_argument(
    "--seed", type=int, default=42, help="Random seed for sampling queries"
)
args = parser.parse_args()

# Setup Paths
output_path = Path("results/results_MSMARCO/feature_attributes/")
output_path.mkdir(parents=True, exist_ok=True)
experiment_tag = f"q{args.num_queries}_top{args.top_k}"
query_data_file = output_path / f"query_data_{experiment_tag}.jsonl"

# Define Metrics
rank_similarity_coefficient = lambda x, y: kendalltau(x, y)[0]

# Initialize Explainer Wrapper
explainer = RankingShapText(
    original_model=None,  # Will be set per query
    explanation_size=10,
    name=f"rankingshap_text_bm25_{experiment_tag}",
    rank_similarity_coefficient=rank_similarity_coefficient,
)

print(f"Starting Text Explanation for first {args.num_queries} queries...")
print(f"Retrieval Depth: 100 (Fixed)")
print(f"Explanation Depth: {args.top_k}")

# Sample random queries upfront (always retrieve 100 docs for IDF context)
try:
    sampled_queries = sample_msmarco_queries(
        num_queries=args.num_queries, split=args.split, top_k=100, seed=args.seed
    )
except Exception as exc:
    print(f"Failed to sample queries: {exc}")
    sys.exit(1)

for query_idx, sampled_query in enumerate(sampled_queries):
    source_query_id = sampled_query["query_id"]
    query_text = sampled_query["query_text"]
    docs_100 = sampled_query["documents"]

    print(
        f"Processing Query Index {query_idx} (dataset id: {source_query_id})...",
        flush=True,
    )

    # 2. Initialize BM25 Model with all 100 docs (Fixes IDF)
    model_wrapper = BM25Wrapper(docs_100)

    # Build vocab for all 100 docs to score them
    vocab_100 = sorted(
        list(set([word for doc in docs_100 for word in tokenize_and_stem(doc)]))
    )
    model_wrapper.set_query(query_text, vocab_100)

    # Create feature matrix for all 100 docs
    feature_matrix_100 = []
    for doc in docs_100:
        doc_words = set(tokenize_and_stem(doc))
        row = [1 if w in doc_words else 0 for w in vocab_100]
        feature_matrix_100.append(row)
    feature_matrix_100 = np.array(feature_matrix_100)

    # Score all 100 docs
    scores_100 = model_wrapper.predict(feature_matrix_100)

    # 3. Select Top K docs for Explanation
    # Sort descending
    sorted_indices = np.argsort(scores_100)[::-1]

    # Take top K indices
    top_k_indices = sorted_indices[: args.top_k]

    # Get the actual documents
    top_k_docs = [docs_100[i] for i in top_k_indices]

    # Save ONLY the explained documents to query_data for evaluation
    query_record = {
        "query_id": query_idx,
        "dataset_query_id": source_query_id,
        "query_text": query_text,
        "documents": top_k_docs,
    }
    with open(query_data_file, "a") as f:
        f.write(json.dumps(query_record) + "\n")

    # 4. Prepare for SHAP (Explain only top K)
    # Rebuild vocabulary for just the top K docs to minimize feature space
    vocabulary_small = sorted(
        list(set([word for doc in top_k_docs for word in tokenize_and_stem(doc)]))
    )
    num_features = len(vocabulary_small)
    print(f"  - Vocab size (Top {args.top_k}): {num_features}")

    # Build Binary Feature Matrix for Top K
    feature_matrix_small = []
    for doc in top_k_docs:
        doc_words = set(tokenize_and_stem(doc))
        row = [1 if w in doc_words else 0 for w in vocabulary_small]
        feature_matrix_small.append(row)
    feature_matrix_small = np.array(feature_matrix_small)

    # 5. Update Explainer Model
    # We reuse the SAME model_wrapper (so IDF is still based on 100 docs),
    # but we update its vocabulary mapping to match our new small feature matrix.
    model_wrapper.set_query(query_text, vocabulary_small)
    explainer.original_model = model_wrapper.predict

    # 6. Run RankingSHAP
    selection, attribution = explainer.get_query_explanation(
        query_features=feature_matrix_small, query_id=query_idx
    )

    # 7. Save Results
    save_file = output_path / f"{explainer.name}.csv"
    attribution.safe_to_file(save_file)


def prepare_for_eval(path_to_attribute_values):
    experiment_results = pd.read_csv(path_to_attribute_values)
    experiment_results = experiment_results.set_index("feature_number")
    experiment_results = experiment_results.stack().swaplevel().sort_index()
    experiment_results = experiment_results.reset_index().rename(
        columns={"level_0": "query_number", 0: "attribution_value"}
    )
    experiment_results = experiment_results.set_index(
        ["query_number", "feature_number"]
    )
    experiment_results.to_csv(
        Path(str(path_to_attribute_values).split(".")[0] + "_eval.csv")
    )


# Convert to eval format (Long format: query_number, feature_number, attribution_value)
print("Converting to eval format...")
save_file = output_path / f"{explainer.name}.csv"
if save_file.exists():
    prepare_for_eval(save_file)

print("Done!")
