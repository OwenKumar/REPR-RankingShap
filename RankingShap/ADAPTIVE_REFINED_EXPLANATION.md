# RankingSHAP Adaptive Refined: Two-Stage Approach

## Overview

`RankingShapAdaptiveRefined` is a two-stage explainer that combines the speed of adaptive sampling with the accuracy of high-quality refinement. It provides a smart trade-off between computational efficiency and explanation quality.

## How It Works

### Stage 1: Fast Adaptive Sampling
- Runs adaptive sampling (25-300 samples based on query size)
- Uses the sqrt rule: `samples = base * sqrt(n_docs)`
- Quickly identifies which features are most important
- **Goal**: Get a rough estimate of feature importance fast

### Stage 2: Refined Sampling
- Takes the top-k features identified in Stage 1 (default: top 10)
- Runs high-quality sampling (default: 1000 samples) to refine only these top features
- **Goal**: Get baseline-quality attributions for the most important features

### Combination
- **Top-k features**: Use refined values (high accuracy)
- **Other features**: Use adaptive values (fast, good enough for less important features)

## Why This Works

1. **Speed**: Most features use fast adaptive sampling (25-300 samples)
2. **Accuracy**: Important features get refined with 1000+ samples
3. **Smart**: Only refines features that matter, saving computation
4. **Best of Both**: Faster than baseline, more accurate than pure adaptive

## Expected Performance

### Speed
- **Faster than baseline**: Only refines 10 features instead of all
- **Slower than pure adaptive**: Adds refinement stage for top features
- **Expected speedup**: ~10-15x faster than baseline (vs 20x for pure adaptive)

### Accuracy
- **Better than pure adaptive**: Top features have baseline-quality attributions
- **Similar to baseline**: Top features should match baseline accuracy
- **Better exposure metrics**: Should perform better on exposure than pure adaptive

## Parameters

```python
RankingShapAdaptiveRefined(
    permutation_sampler="kernel",
    background_data=background_data.background_summary,
    original_model=model.predict,
    explanation_size=3,
    name="rankingshapK_adaptive_refined",
    rank_similarity_coefficient=rank_similarity_coefficient,
    adaptive_min_samples=25,      # Base for adaptive sampling
    adaptive_max_samples=300,     # Cap for adaptive sampling
    top_k_to_refine=10,           # Number of top features to refine
    refinement_samples=1000,      # High-quality samples for refinement
)
```

### Key Parameters

- **`top_k_to_refine`**: Number of top features to refine (default: 10)
  - Higher = more accurate but slower
  - Lower = faster but less accurate
  - Recommended: 5-15 depending on dataset

- **`refinement_samples`**: Number of samples for refinement (default: 1000)
  - Higher = more accurate but slower
  - Lower = faster but less accurate
  - Recommended: 500-2000 depending on desired accuracy

## Example Usage

The explainer is automatically included in `generate_feature_attribution_explanations.py`:

```python
ranking_shapK_adaptive_refined_explainer = RankingShapAdaptiveRefined(
    permutation_sampler="kernel",
    background_data=background_data.background_summary,
    original_model=model.predict,
    explanation_size=explanation_size,
    name="rankingshapK_adaptive_refined",
    rank_similarity_coefficient=rank_similarity_coefficient,
    adaptive_min_samples=25,
    adaptive_max_samples=300,
    top_k_to_refine=10,
    refinement_samples=1000,
)
```

## Comparison with Other Methods

| Method | Speed | Accuracy | Best For |
|--------|-------|----------|----------|
| **Baseline RankingSHAP** | Slow (1x) | High | Maximum accuracy |
| **Pure Adaptive** | Fast (20x) | Medium | Maximum speed |
| **Adaptive Refined** | Fast (10-15x) | High | Balanced speed/accuracy |

## Expected Results

### Timing
- Overall speedup: ~10-15x faster than baseline
- Per-query: Faster for all queries, especially large ones
- Still faster than baseline even with refinement stage

### Quality Metrics
- **Kendall-tau**: Should match baseline (top features refined)
- **Exposure**: Should be better than pure adaptive (top features refined)
- **Overall**: Should be close to baseline quality

## When to Use

- ✅ You want better accuracy than pure adaptive
- ✅ You can tolerate slightly slower than pure adaptive
- ✅ You want a good balance between speed and accuracy
- ✅ You care about exposure metrics (top-position accuracy)

## When NOT to Use

- ❌ You need maximum speed (use pure adaptive)
- ❌ You need maximum accuracy (use baseline)
- ❌ You have very limited computational resources

## Technical Details

### Feature Selection
Features are selected for refinement based on **absolute attribution value** from Stage 1:
- Top-k features by `|attribution_value|` are refined
- This ensures we refine the most impactful features

### Combination Strategy
- Top-k features: Use Stage 2 (refined) values
- Other features: Use Stage 1 (adaptive) values
- Final explanation: Sorted by combined attribution values

## Future Improvements

Potential enhancements:
1. **Adaptive top-k**: Adjust `top_k_to_refine` based on query size
2. **Progressive refinement**: Refine more features for complex queries
3. **Confidence-based**: Only refine features with high uncertainty
4. **Iterative refinement**: Multiple refinement passes for top features

