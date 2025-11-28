import pandas as pd
import numpy as np
import json
from scipy.stats import kendalltau, weightedtau
import itertools
from pathlib import Path

# Import the utils we created earlier
from utils.bm25_wrapper import BM25Wrapper, tokenize_and_stem

# from utils.msmarco_loader import load_msmarco_query # No longer needed


class TextRankSHAPEvaluator:
    def __init__(self, query_data_path, split="validation", top_k=10):
        self.split = split
        self.top_k = top_k
        self.query_data = self._load_query_data(query_data_path)

    def _load_query_data(self, path):
        """Loads query data from JSONL file into a dictionary."""
        data = {}
        if not Path(path).exists():
            print(f"Warning: Query data file {path} not found.")
            return data

        with open(path, "r") as f:
            for line in f:
                try:
                    record = json.loads(line)
                    data[record["query_id"]] = record
                except json.JSONDecodeError:
                    continue
        return data

    def calculate_w_kendall_tau(self, rank_a, rank_b):
        """
        Weighted Fidelity (wFidelity) as described in RankSHAP paper.
        w_ij = |rank_a[i] - rank_a[j]|
        """
        # Map doc_index -> rank
        r_a = {doc_idx: r for r, doc_idx in enumerate(rank_a)}
        r_b = {doc_idx: r for r, doc_idx in enumerate(rank_b)}

        items = list(r_a.keys())
        if len(items) < 2:
            return 0.0

        numerator = 0.0
        denominator = 0.0

        for i, j in itertools.combinations(items, 2):
            # Weight depends on the position difference in the ORIGINAL ranking (model output)
            w_ij = abs(r_a[i] - r_a[j])

            # Concordance: 1 if same order, -1 if swapped
            sgn_a = np.sign(r_a[i] - r_a[j])
            sgn_b = np.sign(r_b[i] - r_b[j])

            numerator += w_ij * (sgn_a * sgn_b)
            denominator += w_ij

        return numerator / denominator if denominator != 0 else 0.0

    def evaluate(self, attribution_file):
        print(f"Evaluating {attribution_file}...")

        # 1. Load Attributions
        # Format: query_number, feature_number, attribution_value
        df = pd.read_csv(attribution_file)

        fidelities = []
        w_fidelities = []

        # Group by Query ID
        if "query_number" not in df.columns:
            print(
                "Error: 'query_number' column not found. Ensure you are using the _eval.csv file."
            )
            return np.nan, np.nan

        grouped = df.groupby("query_number")

        for qid, group in grouped:
            # 2. Load Data for this SPECIFIC Query
            # Use cached data instead of re-streaming
            if qid not in self.query_data:
                print(f"Skipping QID {qid} (Data not found in cache)")
                continue

            record = self.query_data[qid]
            docs = record["documents"]
            query_text = record["query_text"]

            if len(docs) < 2:
                continue

            # 3. Rebuild Vocabulary (Features)
            # MUST match exactly what was done during generation
            vocabulary = sorted(
                list(set([w for doc in docs for w in tokenize_and_stem(doc)]))
            )

            # 4. Get Original Ranking (Ground Truth)
            # Re-initialize BM25 for this query
            model = BM25Wrapper(docs)
            model.set_query(query_text, vocabulary)

            # Create "All Ones" matrix (all words present)
            # Shape: (n_docs, n_vocab)
            input_matrix = []
            for doc in docs:
                row = [1 if w in tokenize_and_stem(doc) else 0 for w in vocabulary]
                input_matrix.append(row)
            input_matrix = np.array(input_matrix)

            # Predict Original Scores
            orig_scores = model.predict(input_matrix)
            local_indices = np.arange(len(docs))
            orig_ranking = local_indices[np.argsort(orig_scores)[::-1]]

            # 5. Get Reconstructed Ranking (Explanation)
            # Map feature_idx -> attribution value
            attributions = dict(
                zip(group["feature_number"], group["attribution_value"])
            )

            recon_scores = []
            for i in range(len(docs)):
                score = 0.0
                # Sum of (Attribution * Presence)
                # If word j is in Doc i, we add Attribution[j]
                doc_vector = input_matrix[i]
                for feat_idx, is_present in enumerate(doc_vector):
                    if is_present == 1:
                        score += attributions.get(feat_idx, 0.0)
                recon_scores.append(score)

            recon_ranking = local_indices[np.argsort(recon_scores)[::-1]]

            # 6. Calculate Metrics
            # Convert rankings to rank vectors (rank of each item 0..N-1)
            # orig_ranking is [doc_id_at_rank_0, doc_id_at_rank_1, ...]
            # We want [rank_of_doc_0, rank_of_doc_1, ...]
            rank_vector_orig = [
                np.where(orig_ranking == i)[0][0] for i in local_indices
            ]
            rank_vector_recon = [
                np.where(recon_ranking == i)[0][0] for i in local_indices
            ]

            tau, _ = kendalltau(rank_vector_orig, rank_vector_recon)

            # Use custom weighted fidelity matching the paper
            w_tau = self.calculate_w_kendall_tau(orig_ranking, recon_ranking)

            if not np.isnan(tau):
                fidelities.append(tau)
                w_fidelities.append(w_tau)

        return np.mean(fidelities), np.mean(w_fidelities)


# ==========================================
# EXECUTION BLOCK
# ==========================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--num_queries", type=int, default=250)
    args = parser.parse_args()

    experiment_tag = f"q{args.num_queries}_top{args.top_k}"

    # Path to the query data file generated by the generation script
    query_data_path = (
        f"results/results_MSMARCO/feature_attributes/query_data_{experiment_tag}.jsonl"
    )

    evaluator = TextRankSHAPEvaluator(
        query_data_path=query_data_path, split=args.split, top_k=args.top_k
    )

    files = [
        f"results/results_MSMARCO/feature_attributes/rankingshap_text_bm25_{experiment_tag}_eval.csv"
    ]

    print(f"{'Method':<40} | {'Fidelity':<10} | {'wFidelity':<10}")
    print("-" * 66)

    for f in files:
        if Path(f).exists():
            fid, w_fid = evaluator.evaluate(f)
            print(f"{f:<40} | {fid:<10.4f} | {w_fid:<10.4f}")
        else:
            print(f"{f:<40} | NOT FOUND")
