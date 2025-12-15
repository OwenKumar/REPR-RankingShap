"""
Corrected evaluation script for RankingSHAP text experiments.

Evaluates fidelity and weighted fidelity as per the RankSHAP paper.
Supports evaluating at different top-K levels (10, 20, 100).
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
        """
        Initialize evaluator with cached query data.

        Args:
            query_data_path: Path to JSONL file with query data from generation
        """
        self.query_data = self._load_query_data(query_data_path)
        print(f"Loaded {len(self.query_data)} queries from {query_data_path}")

    def _load_query_data(self, path):
        """Load query data from JSONL file."""
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

    def evaluate(self, attribution_file, top_k=None):
        """
        Evaluate attributions using Fidelity and wFidelity.

        Args:
            attribution_file: Path to CSV with attributions
            top_k: Optional, only consider top-K documents for evaluation
                   (None means use all documents)

        Returns:
            Tuple of (mean_fidelity, mean_weighted_fidelity)
        """
        print(f"\nEvaluating {attribution_file}")
        if top_k:
            print(f"  Focusing on top-{top_k} documents")

        # Load attributions
        if not Path(attribution_file).exists():
            print(f"  File not found!")
            return np.nan, np.nan

        df = pd.read_csv(attribution_file)

        if "query_number" not in df.columns:
            print("  Error: 'query_number' column not found.")
            return np.nan, np.nan

        fidelities = []
        w_fidelities = []

        grouped = df.groupby("query_number")

        for qid, group in grouped:
            # Get query data
            if qid not in self.query_data:
                continue

            record = self.query_data[qid]
            docs = record["documents"]
            query_text = record["query_text"]
            vocabulary = record.get("vocabulary", None)

            # If vocabulary wasn't saved, rebuild it (backwards compatibility)
            if vocabulary is None:
                vocabulary = sorted(
                    list(set([w for doc in docs for w in tokenize_and_stem(doc)]))
                )

            if len(docs) < 2:
                continue

            # Determine which documents to evaluate
            if top_k and top_k < len(docs):
                # Get original ranking and take top-K
                orig_ranking = record.get("original_ranking", None)
                if orig_ranking is None:
                    # Compute original ranking if not saved
                    model = BM25Wrapper(docs)
                    model.set_query(query_text, vocabulary)
                    input_matrix = self._build_feature_matrix(docs, vocabulary)
                    orig_scores = model.predict(input_matrix)
                    orig_ranking = np.argsort(orig_scores)[::-1]

                # Take only top-K document indices
                eval_doc_indices = list(orig_ranking[:top_k])
            else:
                eval_doc_indices = list(range(len(docs)))

            n_eval_docs = len(eval_doc_indices)
            if n_eval_docs < 2:
                continue

            # Get documents for evaluation
            eval_docs = [docs[i] for i in eval_doc_indices]

            # Rebuild BM25 for evaluation docs
            # Note: For fair comparison, we should use the same IDF as original
            # Here we use the full corpus for IDF consistency
            model = BM25Wrapper(docs)  # Use all docs for IDF
            model.set_query(query_text, vocabulary)

            # Build feature matrix for evaluation documents
            input_matrix = self._build_feature_matrix(eval_docs, vocabulary)

            # Get original ranking for evaluation documents
            orig_scores = []
            for idx in eval_doc_indices:
                doc = docs[idx]
                doc_vector = [
                    1 if w in set(tokenize_and_stem(doc)) else 0 for w in vocabulary
                ]
                score = model.predict(np.array([doc_vector]))[0]
                orig_scores.append(score)
            orig_scores = np.array(orig_scores)

            local_indices = np.arange(n_eval_docs)
            orig_ranking_local = local_indices[np.argsort(orig_scores)[::-1]]

            # Get attributions and compute reconstruction scores
            attributions = dict(
                zip(group["feature_number"], group["attribution_value"])
            )

            recon_scores = []
            for i, doc_idx in enumerate(eval_doc_indices):
                score = 0.0
                doc_vector = input_matrix[i]
                for feat_idx, is_present in enumerate(doc_vector):
                    if is_present == 1:
                        score += attributions.get(feat_idx, 0.0)
                recon_scores.append(score)

            recon_ranking_local = local_indices[np.argsort(recon_scores)[::-1]]

            # Convert to rank vectors for correlation
            rank_vector_orig = np.zeros(n_eval_docs, dtype=int)
            rank_vector_recon = np.zeros(n_eval_docs, dtype=int)

            for r, doc_idx in enumerate(orig_ranking_local):
                rank_vector_orig[doc_idx] = r
            for r, doc_idx in enumerate(recon_ranking_local):
                rank_vector_recon[doc_idx] = r

            # Compute metrics
            tau = kendalls_tau(rank_vector_orig, rank_vector_recon)
            w_tau = weighted_kendalls_tau(rank_vector_orig, rank_vector_recon)

            if not np.isnan(tau):
                fidelities.append(tau)
                w_fidelities.append(w_tau)

        if not fidelities:
            return np.nan, np.nan

        return np.mean(fidelities), np.mean(w_fidelities)

    def _build_feature_matrix(self, docs, vocabulary):
        """Build binary feature matrix for documents."""
        matrix = []
        for doc in docs:
            doc_words = set(tokenize_and_stem(doc))
            row = [1 if w in doc_words else 0 for w in vocabulary]
            matrix.append(row)
        return np.array(matrix)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_docs", type=int, default=100)
    parser.add_argument("--num_queries", type=int, default=250)
    parser.add_argument(
        "--top_k",
        type=int,
        nargs="+",
        default=[10, 20, 100],
        help="Top-K values to evaluate (e.g., 10 20 100)",
    )
    args = parser.parse_args()

    experiment_tag = f"q{args.num_queries}_docs{args.num_docs}"
    results_dir = Path("results/results_MSMARCO/feature_attributes")

    # Path to query data
    query_data_path = results_dir / f"query_data_{experiment_tag}.jsonl"

    if not query_data_path.exists():
        print(f"Query data not found at {query_data_path}")
        print("Please run the generation script first.")
        return

    evaluator = TextRankingSHAPEvaluator(query_data_path)

    # Attribution file to evaluate
    attr_file = results_dir / f"rankingshap_text_bm25_{experiment_tag}_eval.csv"

    print("\n" + "=" * 70)
    print(f"{'Method':<40} | {'Top-K':<8} | {'Fidelity':<10} | {'wFidelity':<10}")
    print("=" * 70)

    results = []

    for k in args.top_k:
        fid, w_fid = evaluator.evaluate(str(attr_file), top_k=k)
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
