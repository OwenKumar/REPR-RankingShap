"""
Download and cache MS MARCO data locally for RankingSHAP experiments.

Run this once to create a local dataset, then use it for all experiments.

Usage:
    python download_msmarco_data.py --num_queries 250 --num_docs 100

This creates:
    data/msmarco_q250_docs100.jsonl
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import ir_datasets


def download_msmarco_data(
    output_dir: str = "data",
    dataset_name: str = "msmarco-passage/dev/small",
    num_queries: int = 250,
    num_docs: int = 100,
    seed: int = 42,
):
    """
    Download MS MARCO queries with their top-K BM25 retrieved documents.

    Args:
        output_dir: Directory to save the data
        dataset_name: ir_datasets dataset identifier
        num_queries: Number of queries to sample
        num_docs: Number of top documents per query (max 1000)
        seed: Random seed for query sampling
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Determine output filename
    if "trec-dl-2019" in dataset_name:
        output_file = output_path / f"msmarco_trecdl2019_docs{num_docs}.jsonl"
    elif "trec-dl-2020" in dataset_name:
        output_file = output_path / f"msmarco_trecdl2020_docs{num_docs}.jsonl"
    else:
        output_file = output_path / f"msmarco_q{num_queries}_docs{num_docs}.jsonl"

    if output_file.exists():
        print(f"Data file already exists: {output_file}")
        print("Delete it first if you want to re-download.")
        return output_file

    print("=" * 60)
    print("Downloading MS MARCO data")
    print("=" * 60)
    print(f"  Dataset: {dataset_name}")
    print(f"  Queries: {num_queries}")
    print(f"  Docs per query: {num_docs}")
    print(f"  Output: {output_file}")
    print("=" * 60)

    print("\nLoading dataset (this may download data on first run)...")
    dataset = ir_datasets.load(dataset_name)

    # Load document store
    print("Loading document store...")
    docs_store = dataset.docs_store()

    # Load queries
    print("Loading queries...")
    queries = {q.query_id: q.text for q in dataset.queries_iter()}
    print(f"  Total queries: {len(queries)}")

    # Load scored docs (top-1000 BM25 results per query)
    print("Loading BM25 retrieval results...")
    query_docs = defaultdict(list)

    total_scored = 0
    for scoreddoc in dataset.scoreddocs_iter():
        query_docs[scoreddoc.query_id].append(scoreddoc.doc_id)
        total_scored += 1
        if total_scored % 500000 == 0:
            print(f"  Processed {total_scored:,} scored docs...")

    print(f"  Total scored docs: {total_scored:,}")
    print(f"  Queries with results: {len(query_docs)}")

    # Filter to queries that have enough documents
    valid_query_ids = [
        qid for qid, doc_ids in query_docs.items() if len(doc_ids) >= num_docs
    ]
    print(f"  Queries with >= {num_docs} docs: {len(valid_query_ids)}")

    # Sample queries
    random.seed(seed)
    if len(valid_query_ids) < num_queries:
        print(f"  Warning: Only {len(valid_query_ids)} queries available")
        sampled_ids = valid_query_ids
    else:
        sampled_ids = random.sample(valid_query_ids, num_queries)

    print(f"\nSampled {len(sampled_ids)} queries")
    print("Fetching document texts...")

    # Fetch and save data
    saved_count = 0
    with open(output_file, "w") as f:
        for i, qid in enumerate(sampled_ids):
            query_text = queries.get(qid, "")
            doc_ids = query_docs[qid][:num_docs]

            # Fetch document texts
            documents = []
            for doc_id in doc_ids:
                try:
                    doc = docs_store.get(doc_id)
                    documents.append(doc.text)
                except KeyError:
                    documents.append("")  # Placeholder for missing docs

            # Only save if we have enough valid documents
            valid_docs = [d for d in documents if d.strip()]
            if len(valid_docs) >= num_docs * 0.9:  # Allow 10% missing
                record = {
                    "query_id": saved_count,
                    "original_query_id": qid,
                    "query_text": query_text,
                    "documents": documents[:num_docs],
                    "doc_ids": doc_ids[:num_docs],
                }
                f.write(json.dumps(record) + "\n")
                saved_count += 1

            if (i + 1) % 50 == 0:
                print(
                    f"  Processed {i + 1}/{len(sampled_ids)} queries, saved {saved_count}"
                )

    print(f"\n✅ Saved {saved_count} queries to {output_file}")

    # Print file size
    size_mb = output_file.stat().st_size / (1024 * 1024)
    print(f"   File size: {size_mb:.1f} MB")

    return output_file


def download_all_standard_datasets():
    """Download all standard datasets used in experiments."""

    configs = [
        # Main experiment: 250 queries, 100 docs (matches RankSHAP paper)
        {"num_queries": 250, "num_docs": 100},
        # Quick test: 50 queries, 100 docs
        {"num_queries": 50, "num_docs": 100},
        # Full retrieval depth: 250 queries, 1000 docs
        # {"num_queries": 250, "num_docs": 1000},
    ]

    for config in configs:
        print(f"\n{'='*60}")
        download_msmarco_data(**config)


def main():
    parser = argparse.ArgumentParser(
        description="Download MS MARCO data for RankingSHAP experiments"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="msmarco-passage/dev/small",
        help="ir_datasets dataset name",
    )
    parser.add_argument(
        "--num_queries",
        type=int,
        default=250,
        help="Number of queries to download (default: 250)",
    )
    parser.add_argument(
        "--num_docs",
        type=int,
        default=100,
        help="Number of documents per query (default: 100)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data",
        help="Output directory (default: data)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all standard datasets",
    )
    args = parser.parse_args()

    if args.all:
        download_all_standard_datasets()
    else:
        download_msmarco_data(
            output_dir=args.output_dir,
            dataset_name=args.dataset,
            num_queries=args.num_queries,
            num_docs=args.num_docs,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
