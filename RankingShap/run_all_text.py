"""
Corrected RankingSHAP Text Experiment Runner

KEY FIX: Generates SEPARATE explanations for each top-k setting (10, 20, 100)
instead of generating one explanation for 100 docs and evaluating on subsets.

Usage:
    python run_corrected_experiment.py --data_file RankingShap/data/msmarco_q250_docs100.jsonl
"""

import argparse
import json
import random
import time
import sys
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, weightedtau
from rank_bm25 import BM25Okapi

try:
    from nltk.stem import PorterStemmer

    stemmer = PorterStemmer()
    HAS_NLTK = True
except ImportError:
    HAS_NLTK = False
    print("Warning: NLTK not found. Install with: pip install nltk")

import shap
from functools import partial
from shap.utils._legacy import convert_to_model


# ============================================================================
# TEXT PROCESSING
# ============================================================================


def tokenize_and_stem(text):
    """Tokenize and stem text using Porter stemmer."""
    tokens = re.findall(r"\b\w+\b", text.lower())
    if HAS_NLTK:
        return [stemmer.stem(t) for t in tokens]
    return tokens


def build_vocabulary_with_query(query_text, docs):
    """
    Build vocabulary from BOTH query and documents.

    Paper: "Stemmed tokens from the vocabulary of the query-document sets"
    """
    all_tokens = set()

    # Add query tokens
    for token in tokenize_and_stem(query_text):
        all_tokens.add(token)

    # Add document tokens
    for doc in docs:
        for token in tokenize_and_stem(doc):
            all_tokens.add(token)

    return sorted(list(all_tokens))


def build_feature_matrix(docs, vocabulary):
    """Build binary feature matrix for documents."""
    vocab_idx = {w: i for i, w in enumerate(vocabulary)}
    matrix = np.zeros((len(docs), len(vocabulary)), dtype=np.float32)

    for doc_idx, doc in enumerate(docs):
        doc_words = set(tokenize_and_stem(doc))
        for word in doc_words:
            if word in vocab_idx:
                matrix[doc_idx, vocab_idx[word]] = 1

    return matrix


# ============================================================================
# BM25 WRAPPER
# ============================================================================


class BM25Wrapper:
    """BM25 wrapper that supports feature masking for SHAP explanations."""

    def __init__(self, corpus_passages):
        self.corpus_passages = corpus_passages
        self.tokenized_corpus = [tokenize_and_stem(doc) for doc in corpus_passages]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        self.query_tokens = []
        self.vocabulary = []

    def set_query(self, query_string, vocabulary):
        self.query_tokens = tokenize_and_stem(query_string)
        self.vocabulary = vocabulary

    def predict(self, binary_feature_matrix):
        """Score documents based on binary feature matrix."""
        scores = []
        for doc_vector in binary_feature_matrix:
            # Keep only vocabulary words with value 1
            reconstructed_tokens = [
                self.vocabulary[idx]
                for idx, val in enumerate(doc_vector)
                if val > 0.5 and idx < len(self.vocabulary)
            ]
            score = self._score_modified_doc(reconstructed_tokens)
            scores.append(score)
        return np.array(scores)

    def _score_modified_doc(self, doc_tokens):
        """Compute BM25 score for document with modified tokens."""
        doc_len = len(doc_tokens)
        if doc_len == 0:
            return 0.0

        doc_freqs = {}
        for t in doc_tokens:
            doc_freqs[t] = doc_freqs.get(t, 0) + 1

        score = 0.0
        for q in self.query_tokens:
            if q in doc_freqs and q in self.bm25.idf:
                freq = doc_freqs[q]
                idf = self.bm25.idf[q]
                numerator = idf * freq * (self.bm25.k1 + 1)
                denominator = freq + self.bm25.k1 * (
                    1 - self.bm25.b + self.bm25.b * doc_len / self.bm25.avgdl
                )
                score += numerator / denominator
        return score


# ============================================================================
# RANKINGSHAP IMPLEMENTATION
# ============================================================================


def placeholder_predict(array):
    return np.array([0] * len(array))


def ranking_value_function(
    masks, original_model_predict, query_features, similarity_coefficient
):
    """Compute ranking similarity for each SHAP perturbation."""
    original_scores = original_model_predict(query_features)
    n_docs = len(original_scores)

    # Convert to ranks
    orig_order = np.argsort(original_scores)[::-1]
    orig_ranks = np.zeros(n_docs, dtype=int)
    for r, idx in enumerate(orig_order):
        orig_ranks[idx] = r + 1

    scores = []
    for mask in masks:
        masked_query_features = query_features * mask
        new_pred = original_model_predict(masked_query_features)

        new_order = np.argsort(new_pred)[::-1]
        new_ranks = np.zeros(n_docs, dtype=int)
        for r, idx in enumerate(new_order):
            new_ranks[idx] = r + 1

        similarity = similarity_coefficient(orig_ranks, new_ranks)
        scores.append(similarity)

    return np.array(scores)


def get_shap_attributions(model_predict, feature_matrix, nsamples=5000):
    """Generate SHAP attributions using KernelSHAP."""
    num_features = feature_matrix.shape[1]
    rank_similarity = lambda x, y: kendalltau(x, y)[0]

    background_zeros = np.zeros((1, num_features))
    explainer = shap.KernelExplainer(
        placeholder_predict, background_zeros, nsamples=nsamples
    )

    custom_predict = partial(
        ranking_value_function,
        original_model_predict=model_predict,
        query_features=feature_matrix,
        similarity_coefficient=rank_similarity,
    )
    explainer.model = convert_to_model(custom_predict)

    instance_all_present = np.ones((1, num_features))
    shap_values = explainer.shap_values(instance_all_present, nsamples=nsamples)[0]

    return {i: shap_values[i] for i in range(len(shap_values))}


# ============================================================================
# FIDELITY COMPUTATION
# ============================================================================


def compute_fidelity(feature_matrix, attributions, original_scores):
    """
    Compute Fidelity = Kendall's tau between original and reconstructed ranking.

    Reconstruction: recon_score(doc) = sum of attributions for present features
    """
    n_docs = feature_matrix.shape[0]
    if n_docs < 2:
        return np.nan, np.nan

    # Compute reconstruction scores
    recon_scores = np.zeros(n_docs)
    for doc_idx in range(n_docs):
        doc_vector = feature_matrix[doc_idx]
        score = sum(
            attributions.get(feat_idx, 0.0)
            for feat_idx, val in enumerate(doc_vector)
            if val > 0.5
        )
        recon_scores[doc_idx] = score

    # Create rank vectors
    orig_order = np.argsort(original_scores)[::-1]
    orig_ranks = np.zeros(n_docs, dtype=int)
    for r, idx in enumerate(orig_order):
        orig_ranks[idx] = r + 1

    recon_order = np.argsort(recon_scores)[::-1]
    recon_ranks = np.zeros(n_docs, dtype=int)
    for r, idx in enumerate(recon_order):
        recon_ranks[idx] = r + 1

    tau, _ = kendalltau(orig_ranks, recon_ranks)
    w_tau, _ = weightedtau(orig_ranks, recon_ranks)

    return (tau if not np.isnan(tau) else 0.0, w_tau if not np.isnan(w_tau) else 0.0)


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================


def run_experiment_for_topk(data, top_k, nsamples=5000, seed=42):
    """
    Run experiment for a specific top-k setting.

    CRITICAL: Generates SEPARATE explanations for each top-k,
    using only top-k documents.
    """
    random.seed(seed)
    np.random.seed(seed)

    fidelities = []
    w_fidelities = []

    print(f"\nProcessing {len(data)} queries for top-{top_k}...")
    start_time = time.time()

    for i, item in enumerate(data):
        query_text = item["query_text"]
        all_docs = item["documents"]

        # KEY FIX #1: Take only top-k documents
        docs = all_docs[:top_k]

        if len(docs) < 2:
            continue

        # KEY FIX #2: Build vocabulary from query + top-k docs only
        vocabulary = build_vocabulary_with_query(query_text, docs)

        if len(vocabulary) == 0:
            continue

        # Build feature matrix for top-k docs
        feature_matrix = build_feature_matrix(docs, vocabulary)

        # KEY FIX #3: Initialize BM25 with only top-k docs
        model = BM25Wrapper(docs)
        model.set_query(query_text, vocabulary)

        # Get original BM25 scores
        original_scores = model.predict(feature_matrix)

        try:
            # Generate SHAP attributions for THIS specific top-k setup
            attributions = get_shap_attributions(
                model.predict, feature_matrix, nsamples
            )
        except Exception as e:
            print(f"  Query {i} failed: {e}")
            continue

        # Compute fidelity
        fid, w_fid = compute_fidelity(feature_matrix, attributions, original_scores)

        if not np.isnan(fid):
            fidelities.append(fid)
            w_fidelities.append(w_fid)

        if (i + 1) % 25 == 0:
            elapsed = time.time() - start_time
            eta = elapsed / (i + 1) * (len(data) - i - 1)
            print(
                f"  Processed {i + 1}/{len(data)} queries "
                f"(elapsed: {elapsed:.0f}s, ETA: {eta:.0f}s)"
            )

    elapsed = time.time() - start_time
    print(f"  Completed in {elapsed:.1f}s")

    return (
        np.mean(fidelities) if fidelities else np.nan,
        np.mean(w_fidelities) if w_fidelities else np.nan,
        len(fidelities),
    )


def load_data(data_file, num_queries):
    """Load query data from JSONL file."""
    data = []
    with open(data_file, "r") as f:
        for line in f:
            data.append(json.loads(line))
            if len(data) >= num_queries:
                break
    return data


def main():
    parser = argparse.ArgumentParser(description="Run corrected RankingSHAP experiment")
    parser.add_argument(
        "--data_file",
        type=str,
        default="RankingShap/data/msmarco_q250_docs100.jsonl",
        help="Path to JSONL data file",
    )
    parser.add_argument(
        "--num_queries", type=int, default=250, help="Number of queries to process"
    )
    parser.add_argument(
        "--nsamples", type=int, default=5000, help="Number of SHAP samples"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        nargs="+",
        default=[10, 20, 100],
        help="Top-K values to evaluate",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    if not Path(args.data_file).exists():
        print(f"Error: Data file not found: {args.data_file}")
        sys.exit(1)

    print("=" * 70)
    print("CORRECTED RankingSHAP Text Experiment")
    print("=" * 70)
    print(f"Data file: {args.data_file}")

    data = load_data(args.data_file, args.num_queries)
    print(f"Loaded {len(data)} queries")
    print(f"SHAP samples: {args.nsamples}")
    print(f"Top-K values: {args.top_k}")
    print("=" * 70)

    results = []

    for top_k in args.top_k:
        print(f"\n{'='*70}")
        print(f"RUNNING TOP-{top_k} EXPERIMENT")
        print(f"{'='*70}")

        fid, w_fid, n_queries = run_experiment_for_topk(
            data, top_k, nsamples=args.nsamples, seed=args.seed
        )

        results.append(
            {
                "top_k": top_k,
                "fidelity": fid,
                "wFidelity": w_fid,
                "n_queries": n_queries,
            }
        )

        print(f"\n  Top-{top_k} Results:")
        print(f"    Fidelity:  {fid:.4f}")
        print(f"    wFidelity: {w_fid:.4f}")

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"{'Top-K':<10} | {'Fidelity':<12} | {'wFidelity':<12}")
    print("-" * 40)
    for r in results:
        print(f"{r['top_k']:<10} | {r['fidelity']:<12.4f} | {r['wFidelity']:<12.4f}")

    print("\n" + "=" * 70)
    print("COMPARISON WITH PAPER (Table 2, MM/BM25)")
    print("=" * 70)
    print(
        f"{'Top-K':<10} | {'Your Result':<12} | {'RankingSHAP':<12} | {'RankSHAP':<12}"
    )
    print("-" * 55)
    paper_rankingshap = {10: 0.52, 20: 0.45, 100: 0.31}
    paper_rankshap = {10: 0.63, 20: 0.54, 100: 0.47}
    for r in results:
        k = r["top_k"]
        print(
            f"{k:<10} | {r['fidelity']:<12.4f} | {paper_rankingshap.get(k, 'N/A'):<12} | {paper_rankshap.get(k, 'N/A'):<12}"
        )


if __name__ == "__main__":
    main()
