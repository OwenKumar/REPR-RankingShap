# ShaRP Memory Fix: What's Needed vs Optional

## 🚨 **CRITICAL FIX (Required - No Accuracy Loss)**

### ✅ **Create ShaRP Instance Once in `__init__`**

**Problem**: Original code created a new ShaRP instance and fitted it for **every single query**. This caused:
- Massive memory usage: O(queries × background_size × features)
- OOM errors on Snellius

**Fix**: Create ShaRP once in `__init__` and reuse it for all queries.

**Impact on Accuracy**: ✅ **NONE** - This is purely an efficiency fix, no accuracy loss.

**Code Change**:
```python
# BEFORE (WRONG - causes OOM):
def get_query_explanation(...):
    sharp = ShaRP(...)  # Created for EVERY query
    sharp.fit(background_data)  # Fitted for EVERY query
    ...

# AFTER (CORRECT):
def __init__(...):
    self.sharp_explainer = self._initialize_sharp()  # Created ONCE

def get_query_explanation(...):
    sharp = self.sharp_explainer  # Reuse existing instance
    ...
```

---

## ⚙️ **OPTIONAL OPTIMIZATIONS (Configurable - Trade Accuracy for Memory)**

### 1. **Limit Background Data Samples** (`max_background_samples`)

**What it does**: Samples background data before fitting ShaRP.

**Default**: `None` (use all background data)

**When needed**: 
- Only if `background_data` is very large (>1000 samples)
- Your code typically uses ~100 samples (from `train_model.py`), so **usually NOT needed**

**Accuracy Impact**:
- `None` (use all): ✅ Best accuracy
- `100`: ⚠️ Slight accuracy loss if original background_data > 100
- `50`: ⚠️ More accuracy loss

**Recommendation**: 
- **Keep `None`** if background_data is already ~100 samples (typical case)
- Only set if you have huge background_data and get OOM errors

---

### 2. **ShaRP's `sample_size` Parameter** (`sharp_sample_size`)

**What it does**: Controls number of perturbations for Shapley value computation in ShaRP.

**Default**: `None` (ShaRP uses all samples - most accurate)

**When needed**: 
- If you still get OOM errors after the critical fix
- ShaRP internally computes many feature coalitions - this limits them

**Accuracy Impact**:
- `None`: ✅ Best accuracy (uses all perturbations)
- `100`: ⚠️ Slight accuracy loss (fewer perturbations)
- `50`: ⚠️ More accuracy loss

**Recommendation**:
- **Start with `None`** (best accuracy)
- Only reduce if you still get OOM errors
- Try `100` first, then `50` if needed

---

### 3. **Limit Documents Per Query** (`max_docs_per_query`)

**What it does**: Limits how many documents are processed per query.

**Default**: `None` (process all documents - best accuracy)

**When needed**:
- Only if queries have many documents (>20) and you get OOM
- Typical queries have 5-15 documents, so usually NOT needed

**Accuracy Impact**:
- `None`: ✅ Best accuracy (uses all documents)
- `10`: ⚠️ Slight accuracy loss if query has >10 docs
- `5`: ⚠️ More accuracy loss

**Recommendation**:
- **Keep `None`** (process all documents)
- Only set if queries are very large (>20 docs) and you get OOM

---

### 4. **Parallel Jobs** (`n_jobs`)

**What it does**: Controls parallelism in ShaRP.

**Default**: `1` (sequential - saves memory)

**Accuracy Impact**: ✅ **NONE** - This only affects speed, not accuracy

**Recommendation**: Keep at `1` to save memory

---

## 📊 **Summary: What You Actually Need**

### ✅ **REQUIRED (No Accuracy Loss)**
1. **Create ShaRP once** - This fixes the OOM error

### ⚠️ **OPTIONAL (Only if Still Getting OOM)**
2. **Limit `sharp_sample_size`** - Only if ShaRP itself is memory-intensive
3. **Limit `max_background_samples`** - Only if background_data > 1000 samples
4. **Limit `max_docs_per_query`** - Only if queries have >20 documents

---

## 🎯 **Recommended Configuration for Maximum Accuracy**

```python
ranking_sharp_explainer = RankingSharp(
    background_data=background_data.background_summary,
    original_model=model.predict,
    explanation_size=explanation_size,
    name="rankingsharp",
    rank_similarity_coefficient=rank_similarity_coefficient,
    # All optional parameters set to None = maximum accuracy
    sharp_sample_size=None,        # Use all ShaRP samples (best accuracy)
    max_background_samples=None,   # Use all background data (best accuracy)
    max_docs_per_query=None,      # Process all documents (best accuracy)
    n_jobs=1,                      # Sequential (saves memory, no accuracy impact)
)
```

**This configuration**:
- ✅ Fixes the OOM error (critical fix)
- ✅ Maximizes accuracy (no unnecessary limits)
- ✅ Should work if background_data is ~100 samples (typical)

---

## 🔧 **If You Still Get OOM Errors**

Try these in order (each reduces accuracy slightly):

### Step 1: Limit ShaRP's internal sampling
```python
sharp_sample_size=100,  # Reduce from None to 100
```

### Step 2: If still OOM, limit background data
```python
max_background_samples=100,  # If background_data is huge
```

### Step 3: If still OOM, limit documents
```python
max_docs_per_query=10,  # If queries have many documents
```

---

## 📈 **Expected Accuracy vs Memory Trade-offs**

| Configuration | Accuracy | Memory Usage | When to Use |
|--------------|----------|--------------|-------------|
| All `None` | ✅ Best | Medium | **Recommended** - Use this first |
| `sharp_sample_size=100` | ⚠️ Good | Lower | If Step 1 needed |
| `max_background_samples=100` | ⚠️ Good | Lower | If Step 2 needed |
| `max_docs_per_query=10` | ⚠️ Good | Lower | If Step 3 needed |
| All limits | ⚠️ Reduced | Lowest | Last resort |

---

## ✅ **Current Implementation**

The code now:
1. ✅ **Fixes the critical OOM bug** (creates ShaRP once)
2. ✅ **Defaults to maximum accuracy** (all limits are `None`)
3. ✅ **Allows configuration** if you need to reduce memory
4. ✅ **Provides clear warnings** when limits are applied

**You should be able to run with maximum accuracy now!** The critical fix (creating ShaRP once) should be enough to prevent OOM errors in most cases.

