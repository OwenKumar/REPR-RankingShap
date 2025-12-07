# Adaptive Sampling for RankingSHAP

## Summary

This document explains the adaptive sampling approach implemented for RankingSHAP to make it faster while maintaining scientific validity. The adaptive approach uses a query-dependent sample size based on the number of documents in each query.

## 1. Where Sampling is Done

**Location:** `approaches/ranking_shap.py` (original) and `approaches/ranking_shap_adaptive.py` (adaptive)

- **Line 92/96** (original): `shap.KernelExplainer` or `shap.SamplingExplainer` initialized with `nsamples`
- **Line 112/114** (original): `shap_values(vector_of_nones, nsamples=self.nsamples)` - **This is where actual sampling happens**
- **Line 166/168** (adaptive): `shap_values(vector_of_nones, nsamples=adaptive_samples)` - Uses adaptive sample size
- **Inside SHAP**: Samples feature coalitions (subsets) to estimate Shapley values

## 2. How It's Done Now (Original RankingSHAP)

- **Uniform random sampling** over all feature coalitions
- `nsamples="auto"` → SHAP decides automatically (often 100-1000+ samples)
- Each sample evaluates the model with a different feature combination
- **No prioritization** - all coalitions sampled equally
- **Fixed sample size** - same number of samples for all queries regardless of complexity

## 3. New Adaptive Sampling Approach

### Key Idea: **Query-Adaptive Sample Size Using Square Root Rule**

The core insight is simple: **Easy queries (few documents) need fewer SHAP samples, while hard queries (many documents) need more SHAP samples.**

We implement this using a **square root rule** to determine the number of samples:
```
samples = base_factor * sqrt(number_of_documents_in_query)
```

### Formula Details

- **Base Factor**: `adaptive_min_samples` (default: 25)
- **Hard Cap**: `adaptive_max_samples` (default: 300)
- **Final Formula**: `samples = min(base_factor * sqrt(n_docs), max_samples)`

### Examples (with base=25, max=300):

| Number of Documents | Calculation | Samples Used |
|---------------------|-------------|--------------|
| 1                   | 25 * √1     | 25          |
| 4                   | 25 * √4     | 50          |
| 9                   | 25 * √9     | 75          |
| 16                  | 25 * √16    | 100         |
| 25                  | 25 * √25    | 125         |
| 36                  | 25 * √36    | 150         |
| 50                  | 25 * √50    | ~177        |
| 100                 | 25 * √100   | 250         |
| 144                 | 25 * √144   | 300 (capped)|

### Scientific Validity

✅ **Query-adaptive**: Matches sample size to query complexity (number of documents)
✅ **Smooth scaling**: Uses square root to avoid over-sampling for large queries
✅ **Monte-Carlo theory**: Error in Monte-Carlo approximations typically scales as ~1/√n, so using √n samples is well-grounded
✅ **Maintains Shapley properties**: Still uses proper Shapley value computation, just with fewer samples for simpler queries
✅ **No convergence checking needed**: The sqrt rule provides a principled way to determine sample size upfront

## 4. Implementation

**File:** `approaches/ranking_shap_adaptive.py`

**Key Parameters:**
- `adaptive_min_samples=25`: Base factor for sqrt(n_docs) rule (default)
- `adaptive_max_samples=300`: Hard upper cap on samples (default)
- `permutation_sampler="kernel"`: Uses KernelExplainer (same as rankingshapK)

**How it works:**
1. For each query, compute adaptive sample size: `samples = min(adaptive_min_samples * sqrt(n_docs), adaptive_max_samples)`
2. Run SHAP with this adaptive sample size (single pass, no convergence checking)
3. Return Shapley values computed with the query-appropriate number of samples

**Key Method:** `_adaptive_sample_size(query_features)` (lines 129-147)
- Takes query features as input
- Computes number of documents: `n_docs = len(query_features)`
- Returns: `int(adaptive_min_samples * sqrt(n_docs))` capped at `adaptive_max_samples`

## 5. Usage

The adaptive version is added to the explainers list in `generate_feature_attribution_explanations.py`:

```python
explainers = []

# Add adaptive RankingSHAP
# Uses sqrt-based sampling: samples = base * sqrt(n_docs)
ranking_shapK_adaptive_explainer = RankingShapAdaptive(
    permutation_sampler="kernel",
    background_data=background_data.background_summary,
    original_model=model.predict,
    explanation_size=explanation_size,
    name="rankingshapK_adaptive",
    rank_similarity_coefficient=rank_similarity_coefficient,
    adaptive_min_samples=25,  # Base factor for sqrt(n_docs) rule
    adaptive_max_samples=300,  # Hard upper cap on samples
)
explainers.append(ranking_shapK_adaptive_explainer)
```

**Note:** The adaptive explainer works on any dataset (MQ2008, MSLR-WEB10K, etc.) - there is no dataset restriction.

**Output file:** `rankingshapK_adaptive.csv` (or `rankingshapK_adaptive_test.csv` for test mode)

## 6. Evaluation Plan

### Speed Comparison:
- Compare runtime: `rankingshapK` vs `rankingshapK_adaptive`
- Expected: Adaptive should be faster, especially for queries with few documents
- Speedup depends on query distribution: queries with fewer documents will see larger speedups

### Accuracy Comparison:
- Compare attributions: `rankingshapK.csv` vs `rankingshapK_adaptive.csv`
- Metrics to check:
  1. **Top-k feature agreement**: Do they select the same top features?
  2. **Attribution correlation**: Spearman correlation of attribution values
  3. **L1/L2 distance**: How different are the attribution values?

### Success Criteria:
✅ Adaptive is **faster** (especially for simpler queries)
✅ Attributions are **similar** (correlation > 0.9, top-k agreement > 80%)
✅ Results are **scientifically valid** (sqrt-based rule is grounded in Monte-Carlo theory)

## 7. Expected Results

Based on the sqrt rule with `base=25` and `max=300`:

- **Simple queries** (1-4 docs): ~25-50 samples, very fast
- **Medium queries** (5-16 docs): ~56-100 samples, moderate speed
- **Complex queries** (17-36 docs): ~103-150 samples, still faster than original
- **Very complex queries** (37+ docs): ~151-300 samples (capped), faster than original

**Speedup**: Varies by query complexity. Simple queries see the largest speedup (often 3-5x), while complex queries may see 1.5-2x speedup compared to using `nsamples="auto"` or large fixed samples.

## 8. Running the Experiment

```bash
# Test on MQ2008 (single query for quick test)
python generate_feature_attribution_explanations.py \
    --dataset MQ2008 \
    --model_file <your_model> \
    --experiment_iteration 1 \
    --test

# Full run on MQ2008
python generate_feature_attribution_explanations.py \
    --dataset MQ2008 \
    --model_file <your_model> \
    --experiment_iteration 1

# Can also run on other datasets (e.g., MSLR-WEB10K)
python generate_feature_attribution_explanations.py \
    --dataset MSLR-WEB10K \
    --model_file <your_model> \
    --experiment_iteration 1
```

## 9. Comparing Results

After running, compare:
- `rankingshapK.csv` (original with fixed/auto samples)
- `rankingshapK_adaptive.csv` (new with adaptive samples)

Use evaluation scripts to check:
- Feature selection agreement
- Attribution value correlation
- Runtime comparison

**Evaluation scripts:**
- `evaluate_feature_attribution_with_ground_truth.py` - Compares against ground truth
- `evaluate_rankingshap_fidelity.py` - Measures fidelity metrics
- `plot_metrics.py` - Generates comparison plots

All evaluation scripts have been updated to include `rankingshapK_adaptive` and will gracefully skip missing files for other methods.

