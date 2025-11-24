import numpy as np
import pandas as pd
from scipy.stats import kendalltau, weightedtau
from text_ranking_shap import TextRankingShap
from datasets import load_dataset
import time
from tqdm import tqdm


def calculate_faithfulness_metrics(original_scores, attribution_dict, docs_tokens):
    """
    Calculates Fidelity (Kendall's Tau) and Weighted Fidelity.

    Args:
        original_scores: Array of original BM25 scores for the documents.
        attribution_dict: Dictionary {token: shapley_value}.
        docs_tokens: List of lists of tokens for each document.

    Returns:
        tau: Kendall's Tau between original ranking and explanation-based ranking.
        w_tau: Weighted Kendall's Tau.
    """
    num_docs = len(docs_tokens)
    if num_docs < 2:
        return np.nan, np.nan

    # 1. Reconstruct scores based on attribution
    # Score(doc) = Sum(ShapleyValue(token) for token in doc)
    reconstructed_scores = []
    for doc in docs_tokens:
        # Sum attributions of tokens present in the document
        # If a token isn't in the attribution dict (wasn't in top features or pruned), value is 0
        score = sum(attribution_dict.get(token, 0) for token in doc)
        reconstructed_scores.append(score)

    reconstructed_scores = np.array(reconstructed_scores)

    # 2. Get Rankings (Indices of sorted arrays)
    # We use negative because argsort sorts ascending, we want descending (high score = rank 1)
    original_ranking = np.argsort(-original_scores)
    reconstructed_ranking = np.argsort(-reconstructed_scores)

    # 3. Calculate Metrics
    # Kendall's Tau: Correlation between the two orderings
    tau, _ = kendalltau(original_ranking, reconstructed_ranking)

    # Weighted Tau: Penalizes swaps at the top of the list more heavily
    w_tau, _ = weightedtau(original_ranking, reconstructed_ranking)

    return tau, w_tau


def run_faithfulness_experiment(num_queries=10, print_every=50):
    """
    Runs the experiment on `num_queries` from MS MARCO.
    """
    print(f"--- Starting Faithfulness Experiment (N={num_queries}) ---")

    # 1. Initialize Explainer
    explainer = TextRankingShap(
        nsample_permutations=2500
    )  # Lower samples slightly for speed

    # 2. Load Data
    try:
        dataset = load_dataset(
            "microsoft/ms_marco", "v1.1", split="validation", streaming=True
        )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    results = []

    # Iterate through dataset
    count = 0
    start_global_time = time.time()

    # We iterate manually to handle the streaming dataset logic cleanly with tqdm
    dataset_iter = iter(dataset)

    with tqdm(total=num_queries, desc="Processing Queries") as pbar:
        while count < num_queries:
            try:
                sample = next(dataset_iter)
            except StopIteration:
                print("Reached end of dataset.")
                break

            query = sample["query"]
            passages = sample["passages"]["passage_text"]
            # Keep only non-empty passages
            passages = [p for p in passages if len(p.strip()) > 0]

            # Skip queries with too few docs to rank
            if len(passages) < 2:
                continue

            # --- A. Get Explanations ---
            # 1. Get Attributions (The "Explanation")
            try:
                # Suppress internal prints by redirecting stdout or just rely on library silence
                # TextRankingShap prints info, so we might want to silence it for large runs.
                # For now, we assume the user accepts some noise or modifies TextRankingShap to be quiet.
                sorted_attributions = explainer.get_query_explanation(query, passages)
                attribution_dict = dict(sorted_attributions)
            except Exception as e:
                # Only print errors if absolutely necessary
                # print(f"Skipping query {sample['query_id']} due to error: {e}")
                continue

            # --- B. Get Ground Truth (Original Model Scores) ---
            from rank_bm25 import BM25Okapi
            from nltk.tokenize import word_tokenize

            tokenized_query = word_tokenize(query.lower())
            tokenized_docs = [word_tokenize(doc.lower()) for doc in passages]

            bm25 = BM25Okapi(tokenized_docs)
            original_scores = np.array(bm25.get_scores(tokenized_query))

            # --- C. Evaluate Faithfulness ---
            fidelity, w_fidelity = calculate_faithfulness_metrics(
                original_scores, attribution_dict, tokenized_docs
            )

            results.append(
                {
                    "query_id": sample["query_id"],
                    "num_docs": len(passages),
                    "fidelity_tau": fidelity,
                    "weighted_fidelity": w_fidelity,
                }
            )

            count += 1
            pbar.update(1)

            # Periodic Update
            if count % print_every == 0:
                avg_fid = np.mean([r["fidelity_tau"] for r in results])
                tqdm.write(
                    f"Progress: {count}/{num_queries} | Avg Fidelity: {avg_fid:.4f}"
                )

    # 3. Aggregate Results
    df = pd.DataFrame(results)
    print("\n--- Experiment Results ---")
    print(df.describe())

    avg_fidelity = df["fidelity_tau"].mean()
    avg_w_fidelity = df["weighted_fidelity"].mean()

    print(f"\nAverage Fidelity (Kendall's Tau): {avg_fidelity:.4f}")
    print(f"Average Weighted Fidelity: {avg_w_fidelity:.4f}")
    print(f"Total Time: {time.time() - start_global_time:.2f}s")

    return df


if __name__ == "__main__":
    # Set to 2500 as requested
    df_results = run_faithfulness_experiment(num_queries=2500, print_every=50)
    df_results.to_csv("faithfulness_results_2500.csv", index=False)
    print("Results saved to faithfulness_results_2500.csv")
