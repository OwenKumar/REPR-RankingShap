#!/usr/bin/env python3
"""
Evaluate RankingSHAP Fidelity for BERT Text Ranking.

This script calculates Fidelity and wFidelity metrics for BERT-based
RankingSHAP explanations on MS MARCO, following the methodology from
the RankSHAP paper.

Fidelity: Kendall's tau between original ranking and reconstructed ranking
wFidelity: Weighted Kendall's tau (emphasizes top positions)

Usage:
    python evaluate_rankingshap_bert_text_fidelity.py \
        --split validation \
        --top_k 10 \
        --num_queries 250 \
        --model_name cross-encoder/ms-marco-MiniLM-L-6-v2
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import kendalltau, weightedtau

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.bm25_wrapper import tokenize_and_stem


def kendalls_tau(a, b):
    """Standard Kendall's tau correlation."""
    res = kendalltau(a, b)
    return res.correlation if not np.isnan(res.correlation) else 0.0


def weighted_kendalls_tau(a, b):
    """Weighted Kendall's tau (emphasizes agreement on top-ranked items)."""
    res = weightedtau(a, b)
    return res.correlation if not np.isnan(res.correlation) else 0.0


class BERTTextEvaluator:
    """
    Evaluator for BERT-based RankingSHAP explanations.

    Calculates Fidelity and wFidelity by comparing:
    1. Original ranking (from BERT model)
    2. Reconstructed ranking (from attribution-weighted feature sums)
    """

    def __init__(
        self,
        query_data_path,
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
        device=None,
        batch_size=32,
    ):
        """
        Initialize evaluator.

        Args:
            query_data_path: Path to JSONL file with query data
            model_name: BERT model to use for scoring
            device: 'cuda' or 'cpu'
            batch_size: Batch size for inference
        """
        self.query_data = self._load_query_data(query_data_path)
        self.model_name = model_name
        self.batch_size = batch_size

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"[BERTTextEvaluator] Device: {self.device}")
        print(f"[BERTTextEvaluator] Model: {model_name}")

        # Load BERT model
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

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

    def _score_documents(self, query_text, documents):
        """Score query-document pairs with BERT."""
        scores = []

        with torch.no_grad():
            for i in range(0, len(documents), self.batch_size):
                batch_docs = documents[i : i + self.batch_size]

                inputs = self.tokenizer(
                    [query_text] * len(batch_docs),
                    batch_docs,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                outputs = self.model(**inputs)
                batch_scores = outputs.logits.squeeze(-1).cpu().numpy()

                if batch_scores.ndim == 0:
                    batch_scores = np.array([batch_scores.item()])

                scores.extend(batch_scores.tolist())

        return np.array(scores)

    def evaluate(self, attribution_file):
        """
        Evaluate attributions and calculate Fidelity/wFidelity.

        Args:
            attribution_file: Path to CSV with attributions

        Returns:
            tuple: (mean_fidelity, mean_wfidelity)
        """
        print(f"Evaluating: {attribution_file}")

        # Load attributions
        df = pd.read_csv(attribution_file)

        if "query_number" not in df.columns:
            print("Error: 'query_number' column not found.")
            return np.nan, np.nan

        fidelities = []
        w_fidelities = []

        grouped = df.groupby("query_number")

        for qid, group in grouped:
            # Get query data
            if qid not in self.query_data:
                print(f"  Skipping QID {qid} (not in query data)")
                continue

            record = self.query_data[qid]
            docs = record["documents"]
            query_text = record["query_text"]

            if len(docs) < 2:
                continue

            # Rebuild vocabulary (must match generation)
            vocabulary = sorted(
                list(set([w for doc in docs for w in tokenize_and_stem(doc)]))
            )

            # Get original ranking from BERT
            orig_scores = self._score_documents(query_text, docs)
            local_indices = np.arange(len(docs))
            orig_ranking = local_indices[np.argsort(orig_scores)[::-1]]

            # Build reconstructed ranking from attributions
            attributions = dict(
                zip(group["feature_number"], group["attribution_value"])
            )

            # Build feature matrix
            input_matrix = []
            for doc in docs:
                doc_words = set(tokenize_and_stem(doc))
                row = [1 if w in doc_words else 0 for w in vocabulary]
                input_matrix.append(row)
            input_matrix = np.array(input_matrix)

            # Calculate reconstruction scores
            recon_scores = []
            for i in range(len(docs)):
                score = 0.0
                doc_vector = input_matrix[i]
                for feat_idx, is_present in enumerate(doc_vector):
                    if is_present == 1:
                        score += attributions.get(feat_idx, 0.0)
                recon_scores.append(score)

            recon_ranking = local_indices[np.argsort(recon_scores)[::-1]]

            # Calculate metrics
            n_docs = len(docs)
            rank_vector_orig = np.zeros(n_docs, dtype=int)
            rank_vector_recon = np.zeros(n_docs, dtype=int)

            for r, doc_idx in enumerate(orig_ranking):
                rank_vector_orig[doc_idx] = r

            for r, doc_idx in enumerate(recon_ranking):
                rank_vector_recon[doc_idx] = r

            tau = kendalls_tau(rank_vector_orig, rank_vector_recon)
            w_tau = weighted_kendalls_tau(rank_vector_orig, rank_vector_recon)

            if not np.isnan(tau):
                fidelities.append(tau)
                w_fidelities.append(w_tau)

        mean_fid = np.mean(fidelities) if fidelities else np.nan
        mean_wfid = np.mean(w_fidelities) if w_fidelities else np.nan

        print(f"  Evaluated {len(fidelities)} queries")

        return mean_fid, mean_wfid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--num_queries", type=int, default=250)
    parser.add_argument(
        "--model_name", type=str, default="cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=32)

    args = parser.parse_args()

    experiment_tag = f"q{args.num_queries}_top{args.top_k}"
    model_short = args.model_name.split("/")[-1]

    # Path to query data
    query_data_path = (
        f"results/results_MSMARCO_BERT/feature_attributes/"
        f"query_data_bert_{model_short}_{experiment_tag}.jsonl"
    )

    if not Path(query_data_path).exists():
        print(f"ERROR: Query data not found: {query_data_path}")
        print("Run generation script first.")
        sys.exit(1)

    # Initialize evaluator
    evaluator = BERTTextEvaluator(
        query_data_path=query_data_path,
        model_name=args.model_name,
        device=args.device,
        batch_size=args.batch_size,
    )

    # Files to evaluate
    base_path = Path("results/results_MSMARCO_BERT/feature_attributes/")

    files_to_eval = [
        f"rankingshap_bert_{model_short}_{experiment_tag}_eval.csv",
        f"rankingshap_bert_weighted_{model_short}_{experiment_tag}_eval.csv",
    ]

    print("\n" + "=" * 70)
    print(f"{'Method':<50} | {'Fidelity':<10} | {'wFidelity':<10}")
    print("-" * 70)

    results = []

    for fname in files_to_eval:
        fpath = base_path / fname
        if fpath.exists():
            fid, w_fid = evaluator.evaluate(str(fpath))
            print(f"{fname:<50} | {fid:<10.4f} | {w_fid:<10.4f}")
            results.append(
                {
                    "method": fname,
                    "fidelity": fid,
                    "wFidelity": w_fid,
                }
            )
        else:
            print(f"{fname:<50} | NOT FOUND")

    # Save results
    if results:
        results_df = pd.DataFrame(results)
        out_path = base_path / f"fidelity_bert_{model_short}_{experiment_tag}.csv"
        results_df.to_csv(out_path, index=False)
        print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
