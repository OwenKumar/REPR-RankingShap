"""
Optimized evaluation script for RankingSHAP text experiments.

Matches original evaluation logic exactly but much faster.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, weightedtau

from utils.bm25_wrapper import BM25Wrapper, tokenize_and_stem


def kendalls_tau(a, b):
    """Standard Kendall's Tau correlation."""
    res = kendalltau(a, b)
    return res.correlation if not np.isnan(res.correlation) else 0.0


def weighted_kendalls_tau(a, b):
    """Weighted Kendall's Tau as used in RankSHAP paper."""
    res = weightedtau(a, b)
    return res.correlation if not np.isnan(res.correlation) else 0.0


class TextRankingSHAPEvaluator:
    def __init__(self, query_data_path):
        self.query_data = self._load_query_data(query_data_path)
        print(f"Loaded {len(self.query_data)} queries from {query_data_path}")

        # Pre-compute everything needed per query
        print("Pre-computing feature matrices and BM25 scores...")
        self.feature_matrices = {}
        self.bm25_scores = {}  # Store actual BM25 scores per document

        for qid, record in self.query_data.items():
            docs = record["documents"]
            query_text = record["query_text"]
            vocabulary = record.get("vocabulary", None)

            if vocabulary is None:
                vocabulary = sorted(
                    list(set([w for doc in docs for w in tokenize_and_stem(doc)]))
                )
                record["vocabulary"] = vocabulary

            # Build feature matrix
            feature_matrix = self._build_feature_matrix(docs, vocabulary)
            self.feature_matrices[qid] = feature_matrix

            # Compute BM25 scores for all documents
            model = BM25Wrapper(docs)
            model.set_query(query_text, vocabulary)
            self.bm25_scores[qid] = model.predict(feature_matrix)

        print("  Done.")

    def _load_query_data(self, path):
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

    def evaluate_all(self, attribution_file, top_k_values):
        """
        Evaluate all top_k values efficiently.
        """
        print(f"\nEvaluating {attribution_file}")

        if not Path(attribution_file).exists():
            print(f"  File not found!")
            return {k: (np.nan, np.nan) for k in top_k_values}

        df = pd.read_csv(attribution_file)

        if "query_number" not in df.columns:
            print("  Error: 'query_number' column not found.")
            return {k: (np.nan, np.nan) for k in top_k_values}

        # Initialize results storage
        results = {k: {"fidelities": [], "w_fidelities": []} for k in top_k_values}

        grouped = df.groupby("query_number")

        for qid, group in grouped:
            if qid not in self.query_data:
                continue

            record = self.query_data[qid]
            docs = record["documents"]
            vocabulary = record["vocabulary"]
            n_docs = len(docs)

            if n_docs < 2:
                continue

            # Get pre-computed data
            feature_matrix = self.feature_matrices[qid]
            bm25_scores = self.bm25_scores[qid]

            # Get original ranking from BM25 scores
            orig_ranking = np.argsort(bm25_scores)[::-1]

            # Get attributions as dict
            attributions = dict(
                zip(group["feature_number"], group["attribution_value"])
            )

            # Compute reconstruction scores for ALL documents
            recon_scores = np.zeros(n_docs)
            for doc_idx in range(n_docs):
                doc_vector = feature_matrix[doc_idx]
                score = sum(
                    attributions.get(feat_idx, 0.0)
                    for feat_idx, is_present in enumerate(doc_vector)
                    if is_present == 1
                )
                recon_scores[doc_idx] = score

            # Evaluate for each top_k
            for top_k in top_k_values:
                if top_k >= n_docs:
                    eval_doc_indices = list(range(n_docs))
                else:
                    # Take top-k from original ranking
                    eval_doc_indices = list(orig_ranking[:top_k])

                n_eval = len(eval_doc_indices)
                if n_eval < 2:
                    continue

                # Get BM25 scores for eval documents
                orig_scores_subset = bm25_scores[eval_doc_indices]
                recon_scores_subset = recon_scores[eval_doc_indices]

                # Compute local rankings (within the eval set)
                local_indices = np.arange(n_eval)
                orig_ranking_local = local_indices[np.argsort(orig_scores_subset)[::-1]]
                recon_ranking_local = local_indices[
                    np.argsort(recon_scores_subset)[::-1]
                ]

                # Convert to rank vectors for correlation
                rank_vector_orig = np.zeros(n_eval, dtype=int)
                rank_vector_recon = np.zeros(n_eval, dtype=int)

                for r, idx in enumerate(orig_ranking_local):
                    rank_vector_orig[idx] = r
                for r, idx in enumerate(recon_ranking_local):
                    rank_vector_recon[idx] = r

                # Compute metrics
                tau = kendalls_tau(rank_vector_orig, rank_vector_recon)
                w_tau = weighted_kendalls_tau(rank_vector_orig, rank_vector_recon)

                if not np.isnan(tau):
                    results[top_k]["fidelities"].append(tau)
                    results[top_k]["w_fidelities"].append(w_tau)

        # Compute means
        final_results = {}
        for k in top_k_values:
            fids = results[k]["fidelities"]
            w_fids = results[k]["w_fidelities"]
            if fids:
                final_results[k] = (np.mean(fids), np.mean(w_fids))
            else:
                final_results[k] = (np.nan, np.nan)

        return final_results

    def _build_feature_matrix(self, docs, vocabulary):
        """Build binary feature matrix for documents."""
        vocab_idx = {w: i for i, w in enumerate(vocabulary)}

        matrix = np.zeros((len(docs), len(vocabulary)), dtype=np.int8)
        for doc_idx, doc in enumerate(docs):
            doc_words = set(tokenize_and_stem(doc))
            for word in doc_words:
                if word in vocab_idx:
                    matrix[doc_idx, vocab_idx[word]] = 1
        return matrix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_docs", type=int, default=100)
    parser.add_argument("--num_queries", type=int, default=250)
    parser.add_argument(
        "--top_k",
        type=int,
        nargs="+",
        default=[10, 20, 100],
        help="Top-K values to evaluate",
    )
    args = parser.parse_args()

    experiment_tag = f"q{args.num_queries}_docs{args.num_docs}"
    results_dir = Path("results/results_MSMARCO/feature_attributes")

    query_data_path = results_dir / f"query_data_{experiment_tag}.jsonl"

    if not query_data_path.exists():
        print(f"Query data not found at {query_data_path}")
        print("Please run the generation script first.")
        return

    evaluator = TextRankingSHAPEvaluator(query_data_path)

    attr_file = results_dir / f"rankingshap_text_bm25_{experiment_tag}_eval.csv"

    # Evaluate all top_k values in single pass
    all_results = evaluator.evaluate_all(str(attr_file), args.top_k)

    print("\n" + "=" * 70)
    print(f"{'Method':<40} | {'Top-K':<8} | {'Fidelity':<10} | {'wFidelity':<10}")
    print("=" * 70)

    results = []
    for k in args.top_k:
        fid, w_fid = all_results[k]
        print(f"{'RankingSHAP':<40} | {k:<8} | {fid:<10.4f} | {w_fid:<10.4f}")
        results.append(
            {
                "method": "RankingSHAP",
                "top_k": k,
                "fidelity": fid,
                "wFidelity": w_fid,
            }
        )

    # Save results
    out_dir = Path("results/results_MSMARCO_fidelity")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"fidelity_{experiment_tag}.csv"

    df_out = pd.DataFrame(results)
    df_out.to_csv(out_path, index=False)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
