import random
from datasets import load_dataset
from rank_bm25 import BM25Okapi

from utils.bm25_wrapper import tokenize_and_stem


def _rank_passages_bm25(query_text, passages, top_k):
    """Return the top_k passages ranked by BM25 with respect to query_text."""
    if not passages:
        return []

    tokenized_corpus = [tokenize_and_stem(p) for p in passages]
    bm25 = BM25Okapi(tokenized_corpus)
    query_tokens = tokenize_and_stem(query_text)
    scores = bm25.get_scores(query_tokens)

    sorted_indices = sorted(
        range(len(passages)), key=lambda idx: scores[idx], reverse=True
    )
    top_indices = sorted_indices[: min(top_k, len(passages))]
    return [passages[i] for i in top_indices]


def sample_msmarco_queries(num_queries, split="validation", top_k=100, seed=None):
    """Reservoir-sample random queries (and their passages) from MS MARCO."""
    if num_queries <= 0:
        raise ValueError("num_queries must be positive.")

    dataset = load_dataset("microsoft/ms_marco", "v1.1", split=split, streaming=True)
    rng = random.Random(seed)
    reservoir = []

    for idx, sample in enumerate(dataset):
        query_text = sample["query"]
        passages = sample["passages"]["passage_text"]
        top_docs = _rank_passages_bm25(query_text, passages, top_k)

        entry = {
            "query_id": sample.get("query_id", idx),
            "query_text": query_text,
            "documents": top_docs,
        }

        if len(reservoir) < num_queries:
            reservoir.append(entry)
        else:
            swap_index = rng.randint(0, idx)
            if swap_index < num_queries:
                reservoir[swap_index] = entry

    if len(reservoir) < num_queries:
        raise ValueError(
            f"Requested {num_queries} queries but only found {len(reservoir)} in split '{split}'."
        )

    return reservoir
