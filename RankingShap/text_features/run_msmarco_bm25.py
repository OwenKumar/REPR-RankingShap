import pandas as pd
import numpy as np
from RankingShap.text_features.text_ranking_shap import TextRankingShap
import time
from datasets import load_dataset


# --- 1. Hugging Face MS MARCO Loader ---
def get_hf_msmarco_data(num_queries=2):
    """
    Loads a small subset of the MS MARCO dataset from Hugging Face.
    We use 'microsoft/ms_marco' (v1.1) which contains the structure:
    {
        'query_id': int,
        'query': str,
        'passages': {
            'passage_text': list[str],
            'is_selected': list[int],
            'url': list[str]
        }
        ...
    }
    """
    print("Loading MS MARCO dataset from Hugging Face...")
    # 'test' split often doesn't have labels in v1.1 public release,
    # so we use 'validation' or 'train' to ensure we get passages.
    # streaming=True allows us to load just the first few examples without downloading 8GB.
    dataset = load_dataset(
        "microsoft/ms_marco", "v1.1", split="validation", streaming=True
    )

    formatted_data = []

    # Take the first N examples
    for i, sample in enumerate(dataset):
        if i >= num_queries:
            break

        # Extract fields
        qid = sample["query_id"]
        query_text = sample["query"]

        # The 'passages' field is a dictionary with a list of texts
        # sample['passages'] = {'passage_text': ["...", "..."], 'is_selected': [0, 1], ...}
        passage_texts = sample["passages"]["passage_text"]

        # Filter out empty passages if any
        clean_passages = [p for p in passage_texts if len(p.strip()) > 0]

        formatted_data.append(
            {"query_id": qid, "query": query_text, "passages": clean_passages}
        )

    return formatted_data


# --- 2. Main Execution ---


def main():
    print("--- Starting RankSHAP for Text (BM25 / MS MARCO) ---")

    # Initialize the Text Explainer
    # We limit to 5000 samples as requested for efficiency
    explainer = TextRankingShap(nsample_permutations=5000)

    # Load data from Hugging Face
    try:
        dataset = get_hf_msmarco_data(num_queries=2)
    except Exception as e:
        print(f"Error loading from Hugging Face: {e}")
        print(
            "Please ensure you have internet access and the 'datasets' library installed."
        )
        return

    for entry in dataset:
        qid = entry["query_id"]
        query = entry["query"]
        docs = entry["passages"]

        print(f"\nProcessing Query {qid}: '{query}'")
        print(f"Number of documents: {len(docs)}")

        if len(docs) == 0:
            print("Skipping query with no passages.")
            continue

        start_time = time.time()

        # Run Explanation
        attributions = explainer.get_query_explanation(query, docs)

        end_time = time.time()
        print(f"Time taken: {end_time - start_time:.2f} seconds")

        # Display Results
        print("\nTop 10 Feature Attributions (Tokens):")
        print(f"{'Token':<20} | {'Shapley Value':<15}")
        print("-" * 40)
        for token, val in attributions[:10]:
            print(f"{token:<20} | {val:.5f}")

    print("\n--- Done ---")


if __name__ == "__main__":
    main()
