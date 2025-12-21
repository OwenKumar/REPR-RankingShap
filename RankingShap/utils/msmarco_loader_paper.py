"""
MS MARCO data loader using ir_datasets.

This provides access to the official top-1000 BM25 retrieval results,
which matches the RankSHAP paper's experimental setup.

Available datasets:
- msmarco-passage/dev/small: 6,980 queries with top-1000 BM25 docs
- msmarco-passage/trec-dl-2019/judged: 43 queries with NIST judgments
- msmarco-passage/trec-dl-2020/judged: 54 queries with NIST judgments
"""

import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import ir_datasets


def load_msmarco_with_bm25_docs(
    dataset_name: str = "msmarco-passage/dev/small",
    num_queries: int = 250,
    num_docs: int = 100,
    seed: int = 42,
) -> List[Dict]:
    """
    Load MS MARCO queries with their top-K BM25 retrieved documents.

    Args:
        dataset_name: ir_datasets dataset identifier
        num_queries: Number of queries to sample
        num_docs: Number of top documents per query (max 1000)
        seed: Random seed for query sampling

    Returns:
        List of dicts with keys: query_id, query_text, documents, doc_ids
    """
    print(f"Loading dataset: {dataset_name}")
    dataset = ir_datasets.load(dataset_name)

    # Load document store for looking up passage text
    docs_store = dataset.docs_store()

    # Load queries
    queries = {q.query_id: q.text for q in dataset.queries_iter()}
    print(f"  Total queries: {len(queries)}")

    # Load scored docs (top-1000 BM25 results per query)
    # Group by query_id
    print("  Loading BM25 retrieval results (this may take a moment)...")
    query_docs = defaultdict(list)

    for scoreddoc in dataset.scoreddocs_iter():
        query_docs[scoreddoc.query_id].append(scoreddoc.doc_id)

    print(f"  Queries with scored docs: {len(query_docs)}")

    # Filter to queries that have enough documents
    valid_query_ids = [
        qid for qid, doc_ids in query_docs.items() if len(doc_ids) >= num_docs
    ]
    print(f"  Queries with >= {num_docs} docs: {len(valid_query_ids)}")

    # Sample queries
    random.seed(seed)
    if len(valid_query_ids) < num_queries:
        print(f"  Warning: Only {len(valid_query_ids)} queries available, using all")
        sampled_ids = valid_query_ids
    else:
        sampled_ids = random.sample(valid_query_ids, num_queries)

    print(f"  Sampled {len(sampled_ids)} queries")

    # Build result list
    results = []
    for i, qid in enumerate(sampled_ids):
        query_text = queries.get(qid, "")
        doc_ids = query_docs[qid][:num_docs]  # Take top-K

        # Look up passage text for each doc_id
        documents = []
        for doc_id in doc_ids:
            try:
                doc = docs_store.get(doc_id)
                documents.append(doc.text)
            except KeyError:
                # Skip if document not found
                continue

        if len(documents) >= num_docs:
            results.append(
                {
                    "query_id": i,  # Use sequential ID for simplicity
                    "original_query_id": qid,
                    "query_text": query_text,
                    "documents": documents[:num_docs],
                    "doc_ids": doc_ids[:num_docs],
                }
            )

        if (i + 1) % 50 == 0:
            print(f"    Processed {i + 1}/{len(sampled_ids)} queries")

    print(f"  Final dataset: {len(results)} queries with {num_docs} docs each")
    return results


def load_trec_dl_2019(num_docs: int = 100) -> List[Dict]:
    """
    Load TREC DL 2019 judged queries (43 queries with NIST relevance judgments).

    This is a smaller but high-quality evaluation set.
    """
    return load_msmarco_with_bm25_docs(
        dataset_name="msmarco-passage/trec-dl-2019/judged",
        num_queries=43,  # All available
        num_docs=num_docs,
        seed=42,
    )


def load_trec_dl_2020(num_docs: int = 100) -> List[Dict]:
    """
    Load TREC DL 2020 judged queries (54 queries with NIST relevance judgments).
    """
    return load_msmarco_with_bm25_docs(
        dataset_name="msmarco-passage/trec-dl-2020/judged",
        num_queries=54,  # All available
        num_docs=num_docs,
        seed=42,
    )


def load_msmarco_dev(
    num_queries: int = 250, num_docs: int = 100, seed: int = 42
) -> List[Dict]:
    """
    Load MS MARCO dev set queries with BM25 top-1000.

    This matches the RankSHAP paper setup:
    - 250 randomly sampled queries
    - Top 100 BM25 retrieved documents per query
    """
    return load_msmarco_with_bm25_docs(
        dataset_name="msmarco-passage/dev/small",
        num_queries=num_queries,
        num_docs=num_docs,
        seed=seed,
    )


if __name__ == "__main__":
    # Test the loader
    print("=" * 60)
    print("Testing MS MARCO loader with ir_datasets")
    print("=" * 60)

    # Load a small sample
    data = load_msmarco_dev(num_queries=5, num_docs=100, seed=42)

    print(f"\nLoaded {len(data)} queries")
    for item in data[:2]:
        print(f"\nQuery {item['query_id']}: {item['query_text'][:80]}...")
        print(f"  Documents: {len(item['documents'])}")
        print(f"  First doc preview: {item['documents'][0][:100]}...")
