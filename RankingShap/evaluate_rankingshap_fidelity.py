import pandas as pd
import numpy as np
import lightgbm as lgb
from scipy.stats import kendalltau, weightedtau
from sklearn.datasets import load_svmlight_file
import itertools
import os


class RankingSHAPEvaluator:
    def __init__(self, model_path, dataset_path, n_features=46):
        """
        Initialize with model and dataset.
        n_features: Total number of features in the dataset (MQ2008 usually has 46).
        """
        self.n_features = n_features
        self.model = self._load_model(model_path)
        self.X, self.y, self.qids = self._load_dataset(dataset_path)

    def _load_model(self, model_path):
        """Loads the LightGBM model."""
        print(f"Loading model from {model_path}...")
        try:
            return lgb.Booster(model_file=model_path)
        except Exception as e:
            print(f"Error loading model: {e}")
            return None

    def _load_dataset(self, dataset_path):
        """
        Loads the dataset (SVMLight format).
        Adjusts 1-based indexing (common in .txt) to 0-based (common in models).
        """
        print(f"Loading dataset from {dataset_path}...")
        # Load with n_features + 1 to handle potential 1-based indexing gracefully
        X, y, qids = load_svmlight_file(
            dataset_path, query_id=True, n_features=self.n_features + 1
        )

        # Convert to dense numpy array
        X = X.toarray()

        # Check if column 0 is empty (indicates 1-based indexing in file)
        if np.all(X[:, 0] == 0):
            print(
                "Detected 1-based indexing in dataset. Dropping column 0 to align with 0-based model."
            )
            X = X[:, 1:]  # Drop column 0
        else:
            # If column 0 is not empty, we might need to trim the end if it's too wide
            X = X[:, : self.n_features]

        return X, y, qids

    def load_attributions(self, filepath):
        """
        Parses attribution CSVs.
        Expected format: query_number, feature_number, attribution_value
        """
        # print(f"Loading attributions from {filepath}...")
        # filepath = "results/results_MQ2008/feature_attributes/" + filepath
        df = pd.read_csv(filepath)

        attributions = {}
        # Group by query_number for faster access
        grouped = df.groupby("query_number")

        for qid, group in grouped:
            # Map feature_number to value.
            # Assuming CSV uses 1-based indexing (1..46), we shift to 0-based (0..45)
            # to match our adjusted X matrix.
            feats = dict(zip(group["feature_number"] - 1, group["attribution_value"]))
            attributions[qid] = feats

        return attributions

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
        """
        Calculates Fidelity and wFidelity for a specific attribution file.
        """
        attributions = self.load_attributions(attribution_file)
        fidelities = []
        w_fidelities = []

        unique_qids = np.unique(self.qids)

        for qid in unique_qids:
            # 1. Get documents associated with this query
            indices = np.where(self.qids == qid)[0]
            if len(indices) < 2:
                continue  # Need at least 2 docs to rank

            docs_X = self.X[indices]

            # 2. Get Original Ranking (Ground Truth)
            # We ask the LightGBM model to score these documents
            orig_scores = self.model.predict(docs_X)
            # Sort indices descending by score
            # We store the 'local' index (0 to len(docs)-1) to simplify tau calculation
            local_indices = np.arange(len(docs_X))
            orig_ranking = local_indices[np.argsort(orig_scores)[::-1]]

            # 3. Get Reconstructed Ranking (Explanation)
            if qid not in attributions:
                continue

            phi = attributions[qid]

            # Calculate Reconstruction Score: Sum(Attribution_i * Feature_Value_i)
            recon_scores = []
            for i in range(len(docs_X)):
                score = 0.0
                for feat_idx, attr_val in phi.items():
                    # Ensure we don't go out of bounds
                    if feat_idx < docs_X.shape[1]:
                        score += attr_val * docs_X[i, feat_idx]
                recon_scores.append(score)

            recon_ranking = local_indices[np.argsort(recon_scores)[::-1]]

            # 4. Calculate Metrics

            # Standard Fidelity (Kendall's Tau)
            # We align the rankings based on the items (0, 1, 2...)
            # orig_ranking is [doc_id_at_rank_0, doc_id_at_rank_1, ...]
            # We want [rank_of_doc_0, rank_of_doc_1, ...]
            rank_vector_orig = [
                np.where(orig_ranking == i)[0][0] for i in local_indices
            ]
            rank_vector_recon = [
                np.where(recon_ranking == i)[0][0] for i in local_indices
            ]

            tau, _ = kendalltau(rank_vector_orig, rank_vector_recon)

            # Weighted Fidelity (Custom implementation matching paper)
            # Note: calculate_w_kendall_tau expects the RANKINGS (lists of items), not rank vectors.
            w_tau = self.calculate_w_kendall_tau(orig_ranking, recon_ranking)

            if not np.isnan(tau):
                fidelities.append(tau)
                w_fidelities.append(w_tau)

        # Return averages
        return np.mean(fidelities), np.mean(w_fidelities)


# ==========================================
# EXECUTION BLOCK
# ==========================================

if __name__ == "__main__":
    # Update these paths if files are in a specific subdirectory
    MODEL_FILE = "results/model_files/model_MQ2008"
    DATASET_FILE = "data/MQ2008/Fold1/test.txt"  # Use 'test.txt' from your upload

    # List of attribution files you uploaded
    ATTRIBUTION_FILES = [
        "rankingshap_eval.csv",
        "rankinglime_eval.csv",
        "pointwise_lime_eval.csv",
        "pointwise_shap_eval.csv",
        "greedy_iter_full_eval.csv",
        "random_eval.csv",
    ]

    # Initialize Evaluator
    # We explicitly set n_features=46 for MQ2008
    evaluator = RankingSHAPEvaluator(MODEL_FILE, DATASET_FILE, n_features=46)

    print(f"\n{'Method':<30} | {'Fidelity':<10} | {'wFidelity':<10}")
    print("-" * 56)
    # Collect results to write to CSV
    results_rows = []

    for csv_file in ATTRIBUTION_FILES:
        file_path = "results/results_MQ2008/feature_attributes/" + csv_file
        if os.path.exists(file_path):
            fid, w_fid = evaluator.evaluate(file_path)
            print(f"{csv_file:<30} | {fid:<10.4f} | {w_fid:<10.4f}")
            results_rows.append(
                {
                    "method": csv_file,
                    "fidelity": float(fid) if fid is not None else np.nan,
                    "wFidelity": float(w_fid) if w_fid is not None else np.nan,
                }
            )
        else:
            print(f"{csv_file:<30} | File not found")
            results_rows.append(
                {
                    "method": csv_file,
                    "fidelity": np.nan,
                    "wFidelity": np.nan,
                }
            )

    # Ensure output directory exists
    out_dir = "results/results_MQ2008_fidelity"
    try:
        os.makedirs(out_dir, exist_ok=True)
    except Exception as e:
        print(f"Warning: could not create output directory {out_dir}: {e}")

    out_path = os.path.join(out_dir, "fidelity.csv")
    try:
        df_out = pd.DataFrame(results_rows)
        df_out.to_csv(out_path, index=False)
        print(f"\nSaved fidelity results to: {out_path}")
    except Exception as e:
        print(f"Error writing results CSV to {out_path}: {e}")
